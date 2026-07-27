"""
PrivToken-ReID — Post-Hoc Attacker Training and Evaluation.

Trains multiple attacker architectures against a FROZEN tokenizer to test
whether privacy claims are robust across heterogeneous adversaries.

The key insight: if privacy only holds against your training attacker (UNet),
it is fragile. This script trains stronger attackers and reports if they
can break the privacy guarantee.

Usage:
    python train_eval_attackers.py --checkpoint checkpoints_v3/best_model.pth
    python train_eval_attackers.py --checkpoint checkpoints_v3/best_model.pth --attackers residual,transformer
"""

import os
import sys
import argparse
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm
import yaml
import gc

from src.datasets import get_dataloader
from src.tokenizer_v4 import PrivacyTokenizerV4
from src.reid_head import BNNeck
from src.attacker import UNetDecoder
from src.attackers import (
    get_attacker,
    get_all_attacker_names,
    IdentityAttacker,
)
from src.losses import LPIPSPerceptualLoss, compute_reconstruction_loss
from src.evaluate import compute_privacy_metrics


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate multiple attackers on a frozen tokenizer."
    )
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to tokenizer checkpoint (.pth).')
    parser.add_argument('--config', type=str, default='configs/v4_cuhk03classic.yaml',
                        help='Path to YAML config file.')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--attackers', type=str, default=None,
                        help='Comma-separated list of attacker names. '
                             'Default: all available.')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Training epochs per attacker.')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate for attacker training.')
    return parser.parse_args()


def load_config(path):
    """Load YAML config."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def load_checkpoint_compat(model, state_dict, model_name):
    """Load a checkpoint while tolerating small architecture drift.

    Some CUHK03 checkpoints were produced by the V4 tokenizer, which adds
    `log_noise_scale`. The attacker-eval script only needs the shared encoder
    and VQ weights, so we drop unknown keys and report them instead of failing.
    """
    model_state = model.state_dict()
    filtered_state = {
        key: value for key, value in state_dict.items()
        if key in model_state and model_state[key].shape == value.shape
    }

    skipped_keys = sorted(set(state_dict) - set(filtered_state))
    incompatible = model.load_state_dict(filtered_state, strict=False)

    if skipped_keys:
        print(f"  [Compat] Ignored {model_name} keys: {skipped_keys}")
    if incompatible.missing_keys or incompatible.unexpected_keys:
        print(f"  [Compat] {model_name} missing keys: {incompatible.missing_keys}")
        print(f"  [Compat] {model_name} unexpected keys: {incompatible.unexpected_keys}")

    return incompatible


def denormalize(imgs, device):
    """Reverse ImageNet normalization."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    return (imgs * std + mean).clamp(0, 1)


def train_reconstruction_attacker(attacker, tokenizer, train_loader, lpips_fn,
                                   device, epochs=20, lr=1e-4):
    """
    Train a reconstruction attacker on a frozen tokenizer.

    Args:
        attacker: Attacker model (unfrozen).
        tokenizer: PrivacyTokenizer (frozen).
        train_loader: Training data loader.
        lpips_fn: LPIPS loss function.
        device: Compute device.
        epochs: Training epochs.
        lr: Learning rate.

    Returns:
        dict with training metrics.
    """
    optimizer = Adam(attacker.parameters(), lr=lr, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    tokenizer.eval()
    for p in tokenizer.parameters():
        p.requires_grad_(False)

    best_loss = float('inf')

    for epoch in range(1, epochs + 1):
        attacker.train()
        epoch_losses = []

        pbar = tqdm(train_loader, desc=f"  Epoch {epoch}/{epochs}")
        for imgs, labels, _, _ in pbar:
            imgs = imgs.to(device)
            imgs_unnorm = denormalize(imgs, device)

            with torch.no_grad():
                z_q, _, _, _, _, _ = tokenizer(imgs)  # V4 returns 6 values

            recon = attacker(z_q.detach())
            loss = compute_reconstruction_loss(recon, imgs_unnorm, lpips_fn)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(attacker.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_losses.append(loss.item())
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        scheduler.step()
        avg_loss = np.mean(epoch_losses)
        print(f"    Avg loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss

    # Re-enable tokenizer gradients
    for p in tokenizer.parameters():
        p.requires_grad_(True)

    return {'best_recon_loss': best_loss}


def train_identity_attacker(attacker, tokenizer, train_loader, device,
                             epochs=20, lr=1e-3):
    """
    Train an identity attacker on a frozen tokenizer.

    The identity attacker predicts person ID directly from tokens.
    If it succeeds, the tokens are leaking identity information.

    Args:
        attacker: IdentityAttacker (unfrozen).
        tokenizer: PrivacyTokenizer (frozen).
        train_loader: Training data loader.
        device: Compute device.
        epochs: Training epochs.
        lr: Learning rate.

    Returns:
        dict with identity classification metrics.
    """
    optimizer = Adam(attacker.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    ce_fn = nn.CrossEntropyLoss()

    tokenizer.eval()
    for p in tokenizer.parameters():
        p.requires_grad_(False)

    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        attacker.train()
        correct, total = 0, 0
        epoch_losses = []

        pbar = tqdm(train_loader, desc=f"  Epoch {epoch}/{epochs}")
        for imgs, labels, _, _ in pbar:
            imgs = imgs.to(device)
            labels = labels.to(device).long()

            with torch.no_grad():
                z_q, _, _, _, _, _ = tokenizer(imgs)  # V4 returns 6 values

            logits = attacker(z_q.detach())
            if labels.numel() > 0:
                min_label = int(labels.min().item())
                max_label = int(labels.max().item())
                num_classes = int(logits.size(1))
                if min_label < 0 or max_label >= num_classes:
                    raise ValueError(
                        f"Identity labels out of range: min={min_label}, max={max_label}, "
                        f"num_classes={num_classes}"
                    )
            loss = ce_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = logits.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            epoch_losses.append(loss.item())

            pbar.set_postfix({'loss': f"{loss.item():.3f}",
                              'acc': f"{100*correct/total:.1f}%"})

        scheduler.step()
        acc = 100 * correct / total
        print(f"    Accuracy: {acc:.1f}%")

        if acc > best_acc:
            best_acc = acc

    for p in tokenizer.parameters():
        p.requires_grad_(True)

    return {'identity_top1': best_acc}


def evaluate_attacker_privacy(attacker, tokenizer, cfg, device, attacker_name):
    """
    Evaluate a trained attacker's reconstruction quality on the test set.

    Args:
        attacker: Trained attacker model.
        tokenizer: PrivacyTokenizer.
        cfg: Config dict.
        device: Compute device.
        attacker_name: Name of the attacker.

    Returns:
        dict with PSNR, SSIM, LPIPS metrics.
    """
    if isinstance(attacker, IdentityAttacker):
        # Identity attacker doesn't produce reconstructions
        return {'psnr': 0.0, 'ssim': 0.0, 'lpips': 0.0, 'type': 'identity'}

    metrics = compute_privacy_metrics(tokenizer, attacker,
                                       get_dataloader(cfg, 'gallery')[0], device)
    return {
        'psnr': metrics['psnr'],
        'ssim': metrics['ssim'],
        'lpips': metrics['lpips'],
        'type': 'reconstruction',
    }


def main():
    """Train and evaluate multiple attackers."""
    args = parse_args()
    cfg = load_config(args.config)

    device = torch.device(args.device) if args.device else \
        torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Device] {device}")

    # Load frozen tokenizer
    print(f"\n[Tokenizer] Loading from {args.checkpoint}...")
    tokenizer = PrivacyTokenizerV4(
        codebook_size=cfg['codebook_size'],
        token_dim=cfg['token_dim'],
        vq_beta=cfg['vq_beta'],
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    load_checkpoint_compat(tokenizer, ckpt['tokenizer'], 'tokenizer')
    tokenizer.eval()
    print(f"  Loaded (epoch {ckpt.get('epoch', '?')})")

    # Also load the original UNet attacker for comparison
    original_attacker = UNetDecoder().to(device)
    load_checkpoint_compat(original_attacker, ckpt['attacker'], 'attacker')
    original_attacker.eval()

    # Select attackers
    if args.attackers:
        attacker_names = [a.strip() for a in args.attackers.split(',')]
    else:
        attacker_names = get_all_attacker_names()

    print(f"\n[Attackers] Will train: {attacker_names}")

    # Data
    train_loader, _ = get_dataloader(cfg, 'train')
    cfg['num_classes'] = getattr(train_loader.dataset, 'num_classes', cfg['num_classes'])

    # LPIPS for reconstruction attackers
    lpips_fn = LPIPSPerceptualLoss().to(device)
    lpips_fn.fn.eval()
    for p in lpips_fn.fn.parameters():
        p.requires_grad_(False)

    # Results
    all_results = {}
    output_dir = cfg.get('output_dir', 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, 'attacker_robustness.json')

    # --- Resume Capability ---
    if os.path.exists(json_path):
        print(f"\n[Resume] Found existing results at {json_path}")
        with open(json_path, 'r') as f:
            try:
                all_results = json.load(f)
                print(f"  Loaded previously completed attackers: {list(all_results.keys())}")
            except json.JSONDecodeError:
                print("  Failed to parse existing JSON, starting fresh.")

    # First: evaluate the original trained UNet
    if 'unet_original' not in all_results:
        print(f"\n{'=' * 60}")
        print("A0: Original UNet Attacker (from training)")
        print(f"{'=' * 60}")
        orig_metrics = evaluate_attacker_privacy(original_attacker, tokenizer, cfg,
                                                  device, 'unet_original')
        orig_metrics['trained_epochs'] = 'co-trained'
        all_results['unet_original'] = orig_metrics
        print(f"  PSNR={orig_metrics['psnr']:.2f}  SSIM={orig_metrics['ssim']:.4f}  "
              f"LPIPS={orig_metrics['lpips']:.4f}")

    # Train and evaluate each new attacker
    for i, name in enumerate(attacker_names, 1):
        if name in all_results:
            print(f"\n{'=' * 60}")
            print(f"A{i}: Skipping {name} attacker (already completed!)")
            print(f"{'=' * 60}")
            continue

        print(f"\n{'=' * 60}")
        print(f"A{i}: Training {name} attacker ({args.epochs} epochs)")
        print(f"{'=' * 60}")

        attacker = get_attacker(name, token_dim=cfg['token_dim'],
                                 num_classes=train_loader.dataset.num_classes).to(device)
        num_params = sum(p.numel() for p in attacker.parameters())
        print(f"  Parameters: {num_params:,}")

        # Train
        if name == 'identity':
            train_metrics = train_identity_attacker(
                attacker, tokenizer, train_loader, device,
                epochs=args.epochs, lr=args.lr * 10,  # higher LR for classifier
            )
        else:
            train_metrics = train_reconstruction_attacker(
                attacker, tokenizer, train_loader, lpips_fn, device,
                epochs=args.epochs, lr=args.lr,
            )

        # Evaluate
        print(f"\n  Evaluating {name}...")
        eval_metrics = evaluate_attacker_privacy(attacker, tokenizer, cfg,
                                                  device, name)
        eval_metrics.update(train_metrics)
        eval_metrics['num_params'] = num_params
        eval_metrics['trained_epochs'] = args.epochs
        all_results[name] = eval_metrics

        if name == 'identity':
            print(f"  Identity Top-1: {train_metrics['identity_top1']:.1f}%")
        else:
            print(f"  PSNR={eval_metrics['psnr']:.2f}  "
                  f"SSIM={eval_metrics['ssim']:.4f}  "
                  f"LPIPS={eval_metrics['lpips']:.4f}")

        # --- NEW: Save trained attacker model checkpoint ---
        ckpt_dir = cfg.get('checkpoint_dir', 'checkpoints')
        os.makedirs(ckpt_dir, exist_ok=True)
        attacker_ckpt_path = os.path.join(ckpt_dir, f'attacker_{name}_ep{args.epochs}.pth')
        torch.save(attacker.state_dict(), attacker_ckpt_path)
        print(f"  → Saved attacker weights to {attacker_ckpt_path}")

        # --- NEW: Save results progressively so progress is never lost ---
        output_dir = cfg.get('output_dir', 'outputs')
        os.makedirs(output_dir, exist_ok=True)
        json_path = os.path.join(output_dir, 'attacker_robustness.json')
        with open(json_path, 'w') as f:
            json.dump(all_results, f, indent=2)
            
        # --- NEW: Aggressively purge GPU Memory to prevent Windows Shared Memory paging ---
        del attacker
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Summary table
    print(f"\n{'=' * 80}")
    print("ATTACKER ROBUSTNESS SUMMARY")
    print(f"{'=' * 80}")
    print(f"{'Attacker':<20} {'Params':>10} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8} {'ID-Acc':>8}")
    print("-" * 62)

    for name, metrics in all_results.items():
        params = f"{metrics.get('num_params', 0):,}" if 'num_params' in metrics else 'co-trained'
        psnr = f"{metrics['psnr']:.2f}" if metrics.get('type') != 'identity' else '-'
        ssim = f"{metrics['ssim']:.4f}" if metrics.get('type') != 'identity' else '-'
        lpips_val = f"{metrics['lpips']:.4f}" if metrics.get('type') != 'identity' else '-'
        id_acc = f"{metrics.get('identity_top1', 0):.1f}%" if 'identity_top1' in metrics else '-'
        print(f"{name:<20} {params:>10} {psnr:>8} {ssim:>8} {lpips_val:>8} {id_acc:>8}")

    print(f"\n  → Full Results successfully written to {json_path}")

    # Paper-ready conclusion
    print(f"\n{'=' * 60}")
    recon_attackers = {k: v for k, v in all_results.items()
                       if v.get('type') != 'identity' and 'ssim' in v}
    if recon_attackers:
        max_ssim = max(v['ssim'] for v in recon_attackers.values())
        worst_attacker = max(recon_attackers, key=lambda k: recon_attackers[k]['ssim'])
        print(f"  Worst-case reconstruction: {worst_attacker} (SSIM={max_ssim:.4f})")

    if 'identity' in all_results:
        id_acc = all_results['identity'].get('identity_top1', 0)
        print(f"  Identity prediction from tokens: {id_acc:.1f}%")

    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()

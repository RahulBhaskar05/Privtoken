"""
PrivToken-ReID — Baseline Training Script.

Trains two baselines for fair comparison:
    Baseline A (plain):   ResNet50 + BNNeck + CE + Triplet (no VQ, no privacy)
    Baseline B (vq_only): ResNet50 + VQ + BNNeck + CE + Triplet (no privacy adversarial)

Usage:
    python train_baseline.py --config configs/baseline_plain.yaml
    python train_baseline.py --config configs/baseline_vq_only.yaml
"""

import os
import sys
import argparse
import random
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import yaml

from src.datasets import get_dataloader
from src.tokenizer import PrivacyTokenizer
from src.reid_head import BNNeck
from src.losses import (
    TripletLossWithHardMining,
    CrossEntropyWithLabelSmoothing,
    compute_reid_loss,
)
from src.evaluate import (
    extract_features,
    compute_cmc_map,
)


class PlainEncoder(nn.Module):
    """
    Plain ResNet-50 encoder without VQ — direct feature extraction.

    Architecture:
        Input image (B, 3, 256, 128)
        → ResNet-50 (last-stride=1) → (B, 2048, 16, 8)
        → 1×1 Conv + BN → (B, token_dim, 16, 8)

    No quantization, no information bottleneck.
    Serves as the upper-bound utility reference.
    """

    def __init__(self, token_dim=512):
        super().__init__()
        from torchvision.models import resnet50, ResNet50_Weights

        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        backbone.layer4[0].conv2.stride = (1, 1)
        backbone.layer4[0].downsample[0].stride = (1, 1)
        self.encoder = nn.Sequential(*list(backbone.children())[:-2])

        self.proj = nn.Conv2d(2048, token_dim, kernel_size=1, bias=False)
        self.proj_bn = nn.BatchNorm2d(token_dim)

    def forward(self, x):
        """
        Args:
            x (Tensor): Input images, shape (B, 3, 256, 128).

        Returns:
            feat (Tensor): Projected features, shape (B, token_dim, 16, 8).
        """
        feat = self.encoder(x)
        feat = self.proj_bn(self.proj(feat))
        return feat


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="PrivToken-ReID Baseline Training")
    parser.add_argument('--config', type=str, required=True,
                        help='Path to YAML config file.')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda or cpu).')
    return parser.parse_args()


def load_config(path):
    """Load YAML configuration file."""
    if not os.path.isfile(path):
        print(f"[ERROR] Config file not found: {path}")
        sys.exit(1)
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(path, epoch, model, reid_head, optimizer, best_rank1, mode):
    """Save training checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        'epoch': epoch,
        'reid_head': reid_head.state_dict(),
        'optimizer': optimizer.state_dict(),
        'best_rank1': best_rank1,
        'mode': mode,
    }
    if mode == 'plain':
        state['encoder'] = model.state_dict()
    else:
        state['tokenizer'] = model.state_dict()
    torch.save(state, path)
    print(f"  → Checkpoint saved: {path}")


def evaluate_baseline(model, reid_head, cfg, device, mode):
    """
    Run CMC/mAP evaluation for baseline models.

    Args:
        model: PlainEncoder or PrivacyTokenizer.
        reid_head: BNNeck head.
        cfg: Config dict.
        device: Compute device.
        mode: 'plain' or 'vq_only'.

    Returns:
        dict with rank1, rank5, rank10, mAP.
    """
    model.eval()
    reid_head.eval()

    gallery_loader, _ = get_dataloader(cfg, 'gallery')
    query_loader, _ = get_dataloader(cfg, 'query')

    def _extract(loader):
        all_features, all_pids, all_camids = [], [], []
        with torch.no_grad():
            for imgs, pids, camids, _ in tqdm(loader, desc="Extracting"):
                imgs = imgs.to(device)
                if mode == 'plain':
                    feat = model(imgs)
                else:
                    feat, _, _, _ = model(imgs)
                fn, _, _ = reid_head(feat)
                all_features.append(fn.cpu().numpy())
                all_pids.append(pids.numpy() if isinstance(pids, torch.Tensor) else np.array(pids))
                all_camids.append(camids.numpy() if isinstance(camids, torch.Tensor) else np.array(camids))
        return (np.concatenate(all_features),
                np.concatenate(all_pids),
                np.concatenate(all_camids))

    print("\n[Eval] Extracting gallery features...")
    g_feat, g_pids, g_camids = _extract(gallery_loader)
    print("[Eval] Extracting query features...")
    q_feat, q_pids, q_camids = _extract(query_loader)
    print("[Eval] Computing CMC/mAP...")
    cmc, mAP = compute_cmc_map(q_feat, q_pids, q_camids, g_feat, g_pids, g_camids)

    results = {
        'rank1': cmc[0] * 100,
        'rank5': cmc[4] * 100,
        'rank10': cmc[9] * 100,
        'mAP': mAP * 100,
    }
    print(f"\n  Rank-1: {results['rank1']:.2f}%  Rank-5: {results['rank5']:.2f}%  "
          f"Rank-10: {results['rank10']:.2f}%  mAP: {results['mAP']:.2f}%")
    return results


def main():
    """Main training entry point for baselines."""
    args = parse_args()
    cfg = load_config(args.config)

    mode = cfg.get('baseline_mode', 'plain')
    assert mode in ('plain', 'vq_only'), f"baseline_mode must be 'plain' or 'vq_only', got '{mode}'"

    device = torch.device(args.device) if args.device else \
        torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Device] Using: {device}")
    print(f"[Mode] Baseline: {mode}")

    set_seed(cfg['seed'])

    os.makedirs(cfg['checkpoint_dir'], exist_ok=True)
    os.makedirs(cfg['log_dir'], exist_ok=True)
    os.makedirs(cfg['output_dir'], exist_ok=True)

    writer = SummaryWriter(log_dir=cfg['log_dir'])

    # Data
    print("\n[Data] Loading Market-1501...")
    train_loader, train_dataset = get_dataloader(cfg, 'train')

    # Models
    print("\n[Models] Initializing...")
    if mode == 'plain':
        model = PlainEncoder(token_dim=cfg['token_dim']).to(device)
        print(f"  PlainEncoder params: {sum(p.numel() for p in model.parameters()):,}")
    else:
        model = PrivacyTokenizer(
            codebook_size=cfg['codebook_size'],
            token_dim=cfg['token_dim'],
            vq_beta=cfg['vq_beta'],
        ).to(device)
        print(f"  Tokenizer params:   {sum(p.numel() for p in model.parameters()):,}")

    reid_head = BNNeck(
        token_dim=cfg['token_dim'],
        num_classes=cfg['num_classes'],
    ).to(device)
    print(f"  ReID head params:   {sum(p.numel() for p in reid_head.parameters()):,}")

    # Optimizer & Scheduler
    optimizer = SGD(
        list(model.parameters()) + list(reid_head.parameters()),
        lr=cfg['lr_main'],
        momentum=cfg['momentum'],
        weight_decay=cfg['weight_decay'],
    )
    scheduler = MultiStepLR(optimizer, milestones=cfg['lr_milestones'], gamma=cfg['lr_gamma'])

    # Loss
    ce_loss_fn = CrossEntropyWithLabelSmoothing(num_classes=cfg['num_classes'], smoothing=0.1)
    triplet_loss_fn = TripletLossWithHardMining(margin=cfg['triplet_margin'])

    # CSV log
    csv_path = os.path.join(cfg['log_dir'], 'training_log.csv')
    with open(csv_path, 'w') as f:
        f.write('epoch,ce_loss,triplet_loss,vq_loss,total_loss,rank1,rank5,rank10,mAP\n')

    # Training loop
    best_rank1 = 0.0
    print(f"\n{'=' * 60}")
    print(f"Training {mode} baseline: {cfg['total_epochs']} epochs")
    print(f"{'=' * 60}\n")

    for epoch in range(1, cfg['total_epochs'] + 1):
        model.train()
        reid_head.train()

        epoch_losses = {'ce': [], 'triplet': [], 'vq': [], 'total': []}

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg['total_epochs']} [{mode}]")
        for imgs, labels, camids, paths in pbar:
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            if mode == 'plain':
                feat = model(imgs)
                vq_loss = torch.tensor(0.0, device=device)
            else:
                feat, vq_loss, indices, utilisation = model(imgs)

            fn, ft, logits = reid_head(feat)
            loss_reid, ce, tri = compute_reid_loss(
                logits, ft, labels, ce_loss_fn, triplet_loss_fn,
                alpha=cfg['alpha_triplet'],
            )

            total_loss = loss_reid + vq_loss
            total_loss.backward()
            nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(reid_head.parameters()),
                max_norm=10.0,
            )
            optimizer.step()

            epoch_losses['ce'].append(ce.item())
            epoch_losses['triplet'].append(tri.item())
            epoch_losses['vq'].append(vq_loss.item())
            epoch_losses['total'].append(total_loss.item())

            pbar.set_postfix({
                'CE': f"{ce.item():.3f}",
                'Tri': f"{tri.item():.3f}",
                'Tot': f"{total_loss.item():.3f}",
            })

        scheduler.step()

        # Epoch summary
        avg = {k: np.mean(v) for k, v in epoch_losses.items()}
        print(f"\n[Epoch {epoch}] CE={avg['ce']:.4f} Tri={avg['triplet']:.4f} "
              f"VQ={avg['vq']:.4f} Total={avg['total']:.4f}")

        # TensorBoard
        writer.add_scalar('Loss/ce', avg['ce'], epoch)
        writer.add_scalar('Loss/triplet', avg['triplet'], epoch)
        writer.add_scalar('Loss/total', avg['total'], epoch)
        if mode == 'vq_only':
            writer.add_scalar('Loss/vq', avg['vq'], epoch)

        # Codebook reseeding for VQ-only baseline
        if mode == 'vq_only' and epoch % 5 == 0:
            with torch.no_grad():
                dead_mask = model.vq.ema_cluster_size < 1.0
                n_dead = dead_mask.sum().item()
                if n_dead > 0:
                    sample_imgs = imgs[:min(32, imgs.size(0))]
                    feat = model.encoder(sample_imgs)
                    feat = model.proj_bn(model.proj(feat))
                    live_vecs = feat.permute(0, 2, 3, 1).reshape(-1, cfg['token_dim'])
                    perm = torch.randperm(live_vecs.size(0), device=device)[:n_dead]
                    model.vq.embedding[dead_mask] = live_vecs[perm].detach()
                    model.vq.ema_cluster_size[dead_mask] = 1.0
                    model.vq.ema_embed_sum[dead_mask] = live_vecs[perm].detach()
                    print(f"  [Epoch {epoch}] Reseeded {n_dead} dead codebook entries")

        # Evaluation
        eval_results = None
        if epoch % cfg['eval_every'] == 0 or epoch == cfg['total_epochs']:
            eval_results = evaluate_baseline(model, reid_head, cfg, device, mode)
            writer.add_scalar('Eval/rank1', eval_results['rank1'], epoch)
            writer.add_scalar('Eval/rank5', eval_results['rank5'], epoch)
            writer.add_scalar('Eval/mAP', eval_results['mAP'], epoch)

            if eval_results['rank1'] > best_rank1:
                best_rank1 = eval_results['rank1']
                save_checkpoint(
                    os.path.join(cfg['checkpoint_dir'], 'best_model.pth'),
                    epoch, model, reid_head, optimizer, best_rank1, mode,
                )
                print(f"  ★ New best Rank-1: {best_rank1:.2f}%")

        # Periodic checkpoint
        if epoch % cfg['save_every'] == 0:
            save_checkpoint(
                os.path.join(cfg['checkpoint_dir'], f'checkpoint_ep{epoch}.pth'),
                epoch, model, reid_head, optimizer, best_rank1, mode,
            )

        # CSV log
        with open(csv_path, 'a') as f:
            r1 = eval_results['rank1'] if eval_results else ''
            r5 = eval_results['rank5'] if eval_results else ''
            r10 = eval_results['rank10'] if eval_results else ''
            mAP = eval_results['mAP'] if eval_results else ''
            f.write(f"{epoch},{avg['ce']:.6f},{avg['triplet']:.6f},{avg['vq']:.6f},"
                    f"{avg['total']:.6f},{r1},{r5},{r10},{mAP}\n")

    # Training complete
    print(f"\n{'=' * 60}")
    print(f"Baseline [{mode}] training complete!")
    print(f"  Best Rank-1: {best_rank1:.2f}%")
    print(f"{'=' * 60}")

    save_checkpoint(
        os.path.join(cfg['checkpoint_dir'], 'final_model.pth'),
        cfg['total_epochs'], model, reid_head, optimizer, best_rank1, mode,
    )

    # Save final results
    final_results = evaluate_baseline(model, reid_head, cfg, device, mode)
    final_results['mode'] = mode
    final_results['best_rank1'] = best_rank1
    with open(os.path.join(cfg['output_dir'], 'eval_results.json'), 'w') as f:
        json.dump(final_results, f, indent=2)

    writer.close()


if __name__ == '__main__':
    main()

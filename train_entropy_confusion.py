"""
PrivToken-ReID — Entropy Confusion Training Script.

Identical pipeline to train_v4.py with ONE architectural difference:
the Gradient Reversal Layer (GRL) identity adversary is REMOVED.

Instead, the privacy game for identity confusion is played as a
plain minimax optimisation:
  - Adversary step  : maximise token classification accuracy (standard CE).
  - Tokenizer step  : minimise classification confidence (entropy maximisation
                      on the token representations, i.e. the tokenizer pushes
                      its tokens toward a uniform class distribution).

This is the main minimax entropy confusion training pipeline.
Every other component is kept identical:
  - VQ-VAE tokenizer (PrivacyTokenizerV4)
  - Multi-granularity ReID head (PCB-style)
  - UNet reconstruction attacker (inner-loop adversarial training)
  - Token noise regularisation
  - Entropy-guided privacy weighting

Three-stage schedule (unchanged):
    Stage 1 (Warmup, 1-warmup_epochs):     ReID + VQ + Center only.
    Stage 2 (Adversarial, ramp phase):      Full minimax + privacy losses.
    Stage 3 (Fine-tune, remainder):         All losses, reduced LR.

Usage:
    python train_entropy_confusion.py
    python train_entropy_confusion.py --config configs/v4_market1501_entropy_confusion.yaml
    python train_entropy_confusion.py --resume checkpoints_entropy_confusion/checkpoint_ep40.pth
"""

import os
import sys
import argparse
import random
import json
import gc

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import yaml

from src.datasets import get_dataloader
from src.tokenizer_v4 import PrivacyTokenizerV4
from src.reid_head_v4 import MultiGranularityHead
from src.attacker import UNetDecoder
from src.losses import (
    TripletLossWithHardMining,
    CrossEntropyWithLabelSmoothing,
    LPIPSPerceptualLoss,
    compute_reconstruction_loss,
    compute_region_weighted_reconstruction_loss,
)
from src.losses_v4 import (
    CenterLoss,
    TokenNoiseRegularization,
    EntropyGuidedPrivacyLoss,
    compute_multipart_reid_loss,
)
from src.evaluate_v4 import evaluate_full_v4, run_posthoc_attacker_suite


# =========================================================================
# No-GRL Identity Adversary
# =========================================================================

class PlainIdentityAdversary(nn.Module):
    """
    Plain MLP identity classifier — NO Gradient Reversal Layer.

    Used in a minimax game:
      - Adversary step : train with standard CE to MAXIMISE classification
                         accuracy on quantised token features (detached).
      - Tokeniser step : entropy maximisation so the tokens become identity-
                         confusing. The adversary MLP weights are frozen
                         during this call so gradients only flow into z_q
                         and back through the tokeniser.

    Replaces DeepIdentityAdversary (from losses_v4.py) without any hook magic.
    """

    def __init__(self, input_dim: int, num_classes: int,
                 hidden_dim: int = 1024, num_layers: int = 3,
                 dropout: float = 0.5):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        layers = []
        in_dim = input_dim
        for _ in range(num_layers - 1):
            layers += [
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ]
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, num_classes))
        self.mlp = nn.Sequential(*layers)

    def _pool(self, z_q: torch.Tensor) -> torch.Tensor:
        """GAP over spatial dims: (B, D, H, W) -> (B, D)."""
        if z_q.dim() == 4:
            return self.gap(z_q).flatten(1)
        return z_q  # already flat

    def forward(self, z_q: torch.Tensor) -> torch.Tensor:
        """Return logits (B, num_classes)."""
        return self.mlp(self._pool(z_q))

    # ------------------------------------------------------------------
    # Adversary step: normal CE, maximise classification accuracy
    # ------------------------------------------------------------------
    def adversary_loss(self, z_q: torch.Tensor,
                       labels: torch.Tensor):
        """
        Standard cross-entropy for the adversary.
        Always called with detached z_q so gradients flow only into the MLP.

        Returns:
            loss  (Tensor): CE loss scalar.
            acc   (float) : top-1 accuracy (%).
        """
        logits = self.forward(z_q.detach())
        loss = F.cross_entropy(logits, labels)
        acc  = (logits.argmax(1) == labels).float().mean().item() * 100.0
        return loss, acc

    # ------------------------------------------------------------------
    # Tokeniser step: entropy maximisation to confuse the adversary
    # ------------------------------------------------------------------
    def tokenizer_confusion_loss(self, z_q: torch.Tensor) -> torch.Tensor:
        """
        Entropy maximisation loss for the tokeniser outer step.
        Adversary MLP weights are frozen during this call; gradients only
        flow into z_q and therefore into the tokeniser backbone.

        Minimising this loss maximises H(softmax(logits)), driving the
        tokeniser to produce representations that the adversary cannot
        classify.

        Returns:
            loss_confusion (Tensor): scalar loss (lower -> harder to classify).
        """
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

        logits = self.forward(z_q)        # grads flow into z_q, not mlp
        probs  = F.softmax(logits, dim=-1)
        # Entropy H = -sum(p * log(p)) -- maximise by minimising -H
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
        loss_confusion = -entropy

        self.train()
        for p in self.parameters():
            p.requires_grad_(True)

        return loss_confusion

    def freeze(self):
        for p in self.parameters():
            p.requires_grad_(False)

    def unfreeze(self):
        for p in self.parameters():
            p.requires_grad_(True)


# =========================================================================
# Helpers
# =========================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="PrivToken-ReID -- No-GRL Training Script")
    parser.add_argument('--config', type=str,
                        default='configs/v4_cuhk03classic.yaml',
                        help='Path to YAML config file.')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from.')
    parser.add_argument('--device', type=str, default=None,
                        help='Device override (cuda / cpu).')
    return parser.parse_args()


def load_config(path: str) -> dict:
    if not os.path.isfile(path):
        print(f"[ERROR] Config not found: {path}")
        sys.exit(1)
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def denormalize(imgs: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Reverse ImageNet normalisation to [0, 1]."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    return (imgs * std + mean).clamp(0, 1)


def compute_batch_psnr(recon: torch.Tensor, target: torch.Tensor) -> float:
    mse  = torch.mean((recon - target) ** 2, dim=[1, 2, 3])
    psnr = 10.0 * torch.log10(1.0 / (mse + 1e-10))
    return psnr.mean().item()


def get_lambda_schedule(epoch: int, warmup_epochs: int,
                        start: float, max_val: float,
                        ramp_epochs: int) -> float:
    """Linearly ramp lambda from start to max_val over ramp_epochs."""
    adv_epoch = epoch - warmup_epochs
    if adv_epoch <= 0:
        return 0.0
    t = min(adv_epoch / max(ramp_epochs, 1), 1.0)
    return start + (max_val - start) * t


# =========================================================================
# Checkpoint helpers
# =========================================================================

def save_checkpoint(path, epoch, tokenizer, reid_head, attacker,
                    id_adversary, center_loss_fn,
                    opt_main, opt_attacker, opt_id_adv, opt_center,
                    best_rank1, collapse_count=0,
                    lambda_priv_start=None, lambda_id_start=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch':             epoch,
        'tokenizer':         tokenizer.state_dict(),
        'reid_head':         reid_head.state_dict(),
        'attacker':          attacker.state_dict(),
        'id_adversary':      id_adversary.state_dict(),
        'center_loss':       center_loss_fn.state_dict(),
        'opt_main':          opt_main.state_dict(),
        'opt_attacker':      opt_attacker.state_dict(),
        'opt_id_adv':        opt_id_adv.state_dict(),
        'opt_center':        opt_center.state_dict(),
        'best_rank1':        best_rank1,
        'collapse_count':    collapse_count,
        'lambda_priv_start': lambda_priv_start,
        'lambda_id_start':   lambda_id_start,
        'script':            'train_entropy_confusion',
    }, path)
    print(f"  -> Checkpoint saved: {path}")


def _load_ckpt_into_models(path, device, tokenizer, reid_head, attacker,
                            id_adversary, center_loss_fn,
                            opt_main, opt_attacker, opt_id_adv, opt_center):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    tokenizer.load_state_dict(ckpt['tokenizer'])
    reid_head.load_state_dict(ckpt['reid_head'])
    attacker.load_state_dict(ckpt['attacker'])
    if 'id_adversary' in ckpt:
        id_adversary.load_state_dict(ckpt['id_adversary'])
    if 'center_loss' in ckpt:
        center_loss_fn.load_state_dict(ckpt['center_loss'])
    opt_main.load_state_dict(ckpt['opt_main'])
    opt_attacker.load_state_dict(ckpt['opt_attacker'])
    if 'opt_id_adv' in ckpt:
        opt_id_adv.load_state_dict(ckpt['opt_id_adv'])
    if 'opt_center' in ckpt:
        opt_center.load_state_dict(ckpt['opt_center'])
    return (ckpt['epoch'] + 1,
            ckpt.get('best_rank1', 0.0),
            ckpt.get('collapse_count', 0),
            ckpt.get('lambda_priv_start'),
            ckpt.get('lambda_id_start'))


# =========================================================================
# Main
# =========================================================================

def main():
    args = parse_args()
    cfg  = load_config(args.config)

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Device] Using: {device}")

    set_seed(cfg['seed'])

    # Default dirs (override in config if needed)
    cfg.setdefault('checkpoint_dir', './checkpoints_entropy_confusion')
    cfg.setdefault('log_dir',        './logs_entropy_confusion')
    cfg.setdefault('output_dir',     './outputs_entropy_confusion')

    os.makedirs(cfg['checkpoint_dir'], exist_ok=True)
    os.makedirs(cfg['log_dir'],        exist_ok=True)
    os.makedirs(cfg['output_dir'],     exist_ok=True)

    writer = SummaryWriter(log_dir=cfg['log_dir'])

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    dataset_name = cfg.get('dataset_name', cfg.get('dataset', 'market1501'))
    print(f"\n[Data] Loading dataset: {dataset_name}")
    train_loader, train_dataset = get_dataloader(cfg, 'train')

    # ------------------------------------------------------------------
    # Models
    # ------------------------------------------------------------------
    print("\n[Models] Initialising Entropy Confusion architecture...")

    enable_noise = cfg.get('lambda_noise', 0.0) > 0
    tokenizer = PrivacyTokenizerV4(
        codebook_size=cfg['codebook_size'],
        token_dim=cfg['token_dim'],
        vq_beta=cfg['vq_beta'],
        backbone_type=cfg.get('backbone_type', 'resnet50'),
        enable_noise=enable_noise,
    ).to(device)

    reid_head = MultiGranularityHead(
        token_dim=cfg['token_dim'],
        num_classes=cfg['num_classes'],
        num_parts=cfg.get('num_parts', 4),
    ).to(device)

    attacker = UNetDecoder().to(device)

    # KEY CHANGE: PlainIdentityAdversary with NO GRL
    id_adversary = PlainIdentityAdversary(
        input_dim=cfg['token_dim'],
        num_classes=cfg['num_classes'],
        hidden_dim=cfg.get('id_adversary_hidden', 1024),
        num_layers=cfg.get('id_adversary_layers', 3),
        dropout=cfg.get('id_adversary_dropout', 0.5),
    ).to(device)

    center_loss_fn     = CenterLoss(
        num_classes=cfg['num_classes'],
        feat_dim=cfg['token_dim'],
    ).to(device)
    noise_reg_fn       = TokenNoiseRegularization()
    entropy_privacy_fn = EntropyGuidedPrivacyLoss()

    print(f"  Tokenizer params:     {sum(p.numel() for p in tokenizer.parameters()):,}")
    print(f"  ReID head params:     {sum(p.numel() for p in reid_head.parameters()):,}")
    print(f"  Attacker params:      {sum(p.numel() for p in attacker.parameters()):,}")
    print(f"  ID Adversary params:  {sum(p.numel() for p in id_adversary.parameters()):,}  [plain MLP, no GRL]")
    print(f"  Center params:        {sum(p.numel() for p in center_loss_fn.parameters()):,}")

    # ------------------------------------------------------------------
    # Optimisers
    # ------------------------------------------------------------------
    opt_main = SGD(
        list(tokenizer.parameters()) + list(reid_head.parameters()),
        lr=cfg['lr_main'],
        momentum=cfg['momentum'],
        weight_decay=cfg['weight_decay'],
    )
    opt_attacker = Adam(
        attacker.parameters(),
        lr=cfg['lr_attacker'],
        betas=(0.5, 0.999),
    )
    opt_id_adv = Adam(
        id_adversary.parameters(),
        lr=cfg.get('lr_id_adversary', 1e-3),
        weight_decay=1e-4,
    )
    opt_center = SGD(
        center_loss_fn.parameters(),
        lr=cfg.get('lr_center', 0.5),
    )

    scheduler_main = MultiStepLR(
        opt_main,
        milestones=cfg['lr_milestones'],
        gamma=cfg['lr_gamma'],
    )

    # ------------------------------------------------------------------
    # Loss functions
    # ------------------------------------------------------------------
    ce_loss_fn      = CrossEntropyWithLabelSmoothing(
        num_classes=cfg['num_classes'], smoothing=0.1)
    triplet_loss_fn = TripletLossWithHardMining(margin=cfg['triplet_margin'])

    lpips_fn = LPIPSPerceptualLoss().to(device)
    lpips_fn.fn.eval()
    for p in lpips_fn.fn.parameters():
        p.requires_grad_(False)

    # ------------------------------------------------------------------
    # Hyperparameters & collapse-recovery state
    # ------------------------------------------------------------------
    lambda_priv_start   = cfg.get('lambda_priv_start', 0.005)
    lambda_priv_max     = cfg.get('lambda_priv_max', 0.05)
    lambda_id_start     = cfg.get('lambda_id_start', 0.01)
    lambda_id_max       = cfg.get('lambda_id_max', 0.5)
    lambda_id_ramp      = cfg.get('lambda_id_ramp_epochs', 40)
    lambda_center       = cfg.get('lambda_center', 0.0005)
    lambda_noise        = cfg.get('lambda_noise', 0.01)
    use_entropy_privacy = cfg.get('use_entropy_privacy', True)
    use_region_privacy  = cfg.get('use_region_privacy', True)

    collapse_threshold  = cfg.get('adversarial_collapse_threshold', 70.0)
    max_recoveries      = cfg.get('max_collapse_recoveries', 3)
    collapse_count      = 0
    stage1_ckpt_path    = os.path.join(cfg['checkpoint_dir'], 'stage1_best.pth')

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------
    start_epoch = 1
    best_rank1  = 0.0

    if args.resume and os.path.isfile(args.resume):
        print(f"\n[Resume] Loading: {args.resume}")
        (start_epoch, best_rank1, collapse_count, lp, li) = \
            _load_ckpt_into_models(
                args.resume, device, tokenizer, reid_head, attacker,
                id_adversary, center_loss_fn,
                opt_main, opt_attacker, opt_id_adv, opt_center)
        if lp is not None:
            lambda_priv_start = lp
        if li is not None:
            lambda_id_start = li
        for _ in range(start_epoch - 1):
            scheduler_main.step()
        print(f"  Resumed from epoch {start_epoch - 1}, "
              f"best_rank1={best_rank1:.2f}%")

    # ------------------------------------------------------------------
    # CSV log header
    # ------------------------------------------------------------------
    csv_path = os.path.join(cfg['log_dir'], 'training_log.csv')
    if start_epoch == 1:
        with open(csv_path, 'w') as f:
            f.write(
                'epoch,stage,ce_loss,triplet_loss,vq_loss,center_loss,'
                'attacker_loss,privacy_loss,id_confusion_loss,id_adv_acc,'
                'noise_loss,total_loss,batch_psnr,noise_scale,codebook_util,'
                'lambda_priv,lambda_id,'
                'rank1,rank5,rank10,mAP,mINP,'
                'eval_psnr,eval_ssim,eval_lpips,pu_score\n'
            )

    # ------------------------------------------------------------------
    # Schedule params
    # ------------------------------------------------------------------
    warmup_epochs   = cfg['warmup_epochs']
    total_epochs    = cfg['total_epochs']
    adv_ramp_epochs = cfg.get('adversarial_ramp_epochs', 50)
    inner_steps     = cfg.get('attacker_inner_steps', 3)
    id_inner_steps  = cfg.get('id_adversary_inner_steps', 2)

    print(f"\n{'=' * 70}")
    print(f"Entropy Confusion Training: epochs {start_epoch}-{total_epochs}")
    print(f"  Stage 1 (Warmup):      epochs 1-{warmup_epochs}")
    print(f"  Stage 2 (Adversarial): epochs {warmup_epochs + 1}-"
          f"{warmup_epochs + adv_ramp_epochs}")
    print(f"  Stage 3 (Fine-tune):   epochs "
          f"{warmup_epochs + adv_ramp_epochs + 1}-{total_epochs}")
    print(f"  Identity adversary:    PLAIN MLP minimax  [no GRL]")
    print(f"  Collapse threshold:    Rank-1 < {collapse_threshold}% after epoch 40")
    print(f"{'=' * 70}\n")

    global_step = (start_epoch - 1) * len(train_loader)
    epoch_attacker_history = []

    # ==================================================================
    # Training Loop
    # ==================================================================
    for epoch in range(start_epoch, total_epochs + 1):
        is_warmup   = (epoch <= warmup_epochs)
        is_finetune = (epoch >  warmup_epochs + adv_ramp_epochs)

        stage_str = (
            'Stage1-Warmup'      if is_warmup   else
            'Stage3-Finetune'    if is_finetune  else
            'Stage2-Adversarial'
        )

        tokenizer.train()
        reid_head.train()
        attacker.train()

        epoch_losses = {
            'ce': [], 'triplet': [], 'vq': [], 'center': [],
            'attacker': [], 'privacy': [], 'id_confusion': [],
            'id_adv_acc': [], 'noise': [], 'total': [],
            'psnr': [], 'cb_util': [],
        }

        lambda_priv = get_lambda_schedule(
            epoch, warmup_epochs, lambda_priv_start, lambda_priv_max,
            ramp_epochs=adv_ramp_epochs)
        lambda_id   = get_lambda_schedule(
            epoch, warmup_epochs, lambda_id_start, lambda_id_max,
            ramp_epochs=lambda_id_ramp)

        pbar = tqdm(train_loader,
                    desc=f"Epoch {epoch}/{total_epochs} [{stage_str}]")
        expected_batches = len(train_loader)
        batches_seen = 0

        for batch_idx, (imgs, labels, camids, paths) in enumerate(pbar):
            imgs   = imgs.to(device)
            labels = labels.to(device)
            global_step  += 1
            batches_seen += 1

            imgs_unnorm = denormalize(imgs, device)

            # ===========================================================
            # Step A — Reconstruction attacker inner loop (ALWAYS TRAINED FROM EPOCH 1)
            # ===========================================================
            attacker.unfreeze()
            tokenizer.eval()
            reid_head.eval()

            step_att_losses = []
            for _ in range(inner_steps):
                with torch.no_grad():
                    z_q_det, _, _, _, _, z_q_noisy_det = tokenizer(imgs)
                recon_att = attacker(z_q_noisy_det.detach())
                loss_att  = compute_reconstruction_loss(
                    recon_att, imgs_unnorm, lpips_fn)
                opt_attacker.zero_grad()
                loss_att.backward()
                opt_attacker.step()
                step_att_losses.append(loss_att.item())

            att_loss_val = float(np.mean(step_att_losses))

            if is_warmup:
                # -------------------------------------------------------
                # STAGE 1: ReID warmup only
                # -------------------------------------------------------
                attacker.freeze()
                tokenizer.train()
                reid_head.train()

                opt_main.zero_grad()
                opt_center.zero_grad()

                z_q, vq_loss, indices, utilisation, entropy, z_q_noisy = \
                    tokenizer(imgs)
                outputs = reid_head(z_q)

                loss_reid, ce, tri = compute_multipart_reid_loss(
                    outputs, labels, ce_loss_fn, triplet_loss_fn,
                    alpha_triplet=cfg['alpha_triplet'])
                loss_center = center_loss_fn(outputs['global_ft'], labels)

                total_loss = loss_reid + vq_loss + lambda_center * loss_center
                total_loss.backward()

                nn.utils.clip_grad_norm_(
                    list(tokenizer.parameters()) + list(reid_head.parameters()),
                    max_norm=10.0)
                opt_main.step()
                opt_center.step()

                priv_loss_val    = 0.0
                id_confusion_val = 0.0
                id_adv_acc_val   = 0.0
                noise_loss_val   = 0.0
                batch_psnr       = 0.0
                cb_util          = utilisation

            else:
                # -------------------------------------------------------
                # STAGE 2 / 3: Full adversarial training (no GRL)
                # -------------------------------------------------------

                # Step B — Identity adversary inner loop (maximise accuracy)
                # Adversary trained on DETACHED tokens -- no gradient to tokeniser.
                id_adversary.unfreeze()
                for _ in range(id_inner_steps):
                    with torch.no_grad():
                        z_q_det, _, _, _, _, _ = tokenizer(imgs)
                    loss_adv, id_adv_acc_val = id_adversary.adversary_loss(
                        z_q_det, labels)
                    opt_id_adv.zero_grad()
                    loss_adv.backward()
                    opt_id_adv.step()

                # Step C — Tokeniser outer loop (entropy maximisation)
                # Reconstruction attacker frozen.
                # Adversary MLP is frozen INSIDE tokenizer_confusion_loss().
                attacker.freeze()
                tokenizer.train()
                reid_head.train()

                opt_main.zero_grad()
                opt_center.zero_grad()

                z_q, vq_loss, indices, utilisation, entropy, z_q_noisy = \
                    tokenizer(imgs)

                # ReID utility
                outputs = reid_head(z_q)
                loss_reid, ce, tri = compute_multipart_reid_loss(
                    outputs, labels, ce_loss_fn, triplet_loss_fn,
                    alpha_triplet=cfg['alpha_triplet'])
                loss_center = center_loss_fn(outputs['global_ft'], labels)

                # Reconstruction privacy (tokeniser wants bad reconstruction)
                recon = attacker(z_q_noisy)
                if use_entropy_privacy:
                    loss_recon = entropy_privacy_fn(
                        recon, imgs_unnorm, entropy, lpips_fn)
                elif use_region_privacy:
                    loss_recon = compute_region_weighted_reconstruction_loss(
                        recon, imgs_unnorm, lpips_fn)
                else:
                    loss_recon = compute_reconstruction_loss(
                        recon, imgs_unnorm, lpips_fn)
                loss_priv = -lambda_priv * loss_recon  # negate

                # Identity confusion via entropy maximisation (no GRL)
                loss_id_confusion = lambda_id * \
                    id_adversary.tokenizer_confusion_loss(z_q)

                # Token noise regularisation
                loss_noise = noise_reg_fn(z_q)

                total_loss = (
                    loss_reid
                    + vq_loss
                    + lambda_center    * loss_center
                    + loss_priv
                    + loss_id_confusion
                    + lambda_noise     * loss_noise
                )
                total_loss.backward()

                nn.utils.clip_grad_norm_(
                    list(tokenizer.parameters()) + list(reid_head.parameters()),
                    max_norm=10.0)
                opt_main.step()
                opt_center.step()

                priv_loss_val    = loss_priv.item()
                id_confusion_val = loss_id_confusion.item()
                noise_loss_val   = loss_noise.item()
                batch_psnr       = compute_batch_psnr(recon.detach(), imgs_unnorm)
                cb_util          = utilisation

            # ===========================================================
            # TensorBoard
            # ===========================================================
            writer.add_scalar('Loss/ce',      ce.item(),     global_step)
            writer.add_scalar('Loss/triplet', tri.item(),    global_step)
            vq_val  = vq_loss.item() if torch.is_tensor(vq_loss) else float(vq_loss)
            ctr_val = loss_center.item() if torch.is_tensor(loss_center) else float(loss_center)
            tot_val = total_loss.item() if torch.is_tensor(total_loss) else float(total_loss)
            writer.add_scalar('Loss/vq',     vq_val,         global_step)
            writer.add_scalar('Loss/center', ctr_val,        global_step)
            writer.add_scalar('Loss/total',  tot_val,        global_step)
            writer.add_scalar('Metrics/batch_util', cb_util, global_step)

            global_util = (
                (tokenizer.vq.ema_cluster_size > 1.0).sum().item()
                / tokenizer.vq.K
            )
            writer.add_scalar('Metrics/global_util', global_util, global_step)
            writer.add_scalar('Train/noise_scale',
                              tokenizer.noise_scale.item(), global_step)

            if not is_warmup:
                writer.add_scalar('Loss/attacker_recon',  att_loss_val,      global_step)
                writer.add_scalar('Loss/privacy',         priv_loss_val,     global_step)
                writer.add_scalar('Loss/id_confusion',    id_confusion_val,  global_step)
                writer.add_scalar('Loss/noise_reg',       noise_loss_val,    global_step)
                writer.add_scalar('Metrics/id_adv_acc',   id_adv_acc_val,    global_step)
                writer.add_scalar('Metrics/batch_psnr',   batch_psnr,        global_step)
                writer.add_scalar('Train/lambda_priv',    lambda_priv,       global_step)
                writer.add_scalar('Train/lambda_id',      lambda_id,         global_step)

            epoch_losses['ce'].append(ce.item())
            epoch_losses['triplet'].append(tri.item())
            epoch_losses['vq'].append(vq_val)
            epoch_losses['center'].append(ctr_val)
            epoch_losses['attacker'].append(att_loss_val)
            epoch_losses['privacy'].append(priv_loss_val)
            epoch_losses['id_confusion'].append(id_confusion_val)
            epoch_losses['id_adv_acc'].append(id_adv_acc_val)
            epoch_losses['noise'].append(noise_loss_val)
            epoch_losses['total'].append(tot_val)
            epoch_losses['psnr'].append(batch_psnr)
            epoch_losses['cb_util'].append(cb_util)

            pbar.set_postfix({
                'CE':  f"{ce.item():.3f}",
                'Tri': f"{tri.item():.3f}",
                'VQ':  f"{vq_val:.3f}",
                'Tot': f"{tot_val:.3f}",
            })

        scheduler_main.step()

        if batches_seen != expected_batches:
            raise RuntimeError(
                f"Batch count mismatch: seen={batches_seen} "
                f"expected={expected_batches}")

        # ==============================================================
        # Codebook reseeding (every 5 epochs)
        # ==============================================================
        if epoch % 5 == 0:
            with torch.no_grad():
                dead_mask = tokenizer.vq.ema_cluster_size < 1.0
                n_dead    = dead_mask.sum().item()
                if n_dead > 0:
                    sample_imgs = imgs[:min(32, imgs.size(0))]
                    feat        = tokenizer.get_projected_features(sample_imgs)
                    live_vecs   = feat.permute(0, 2, 3, 1).reshape(-1, cfg['token_dim'])
                    perm        = torch.randperm(live_vecs.size(0), device=device)[:n_dead]
                    tokenizer.vq.embedding[dead_mask]        = live_vecs[perm].detach()
                    tokenizer.vq.ema_cluster_size[dead_mask] = 1.0
                    tokenizer.vq.ema_embed_sum[dead_mask]    = live_vecs[perm].detach()
                    print(f"  [Epoch {epoch}] Reseeded {n_dead} dead codebook entries")

        # ==============================================================
        # Epoch summary
        # ==============================================================
        avg = {k: np.mean(v) if v else 0.0 for k, v in epoch_losses.items()}
        print(
            f"\n[Epoch {epoch}] {stage_str} | "
            f"CE={avg['ce']:.4f} Tri={avg['triplet']:.4f} VQ={avg['vq']:.4f} "
            f"Ctr={avg['center']:.4f} Att={avg['attacker']:.4f} "
            f"Priv={avg['privacy']:.4f} IDConf={avg['id_confusion']:.4f} "
            f"IDAcc={avg['id_adv_acc']:.1f}% Noise={avg['noise']:.4f} "
            f"Tot={avg['total']:.4f} PSNR={avg['psnr']:.2f} "
            f"noise_sigma={tokenizer.noise_scale.item():.4f}"
        )
        used = (tokenizer.vq.ema_cluster_size > 1.0).sum().item()
        print(
            f"  Codebook: {used}/{tokenizer.vq.K} active "
            f"({100 * used / tokenizer.vq.K:.1f}%) | "
            f"lam_priv={lambda_priv:.4f} lam_id={lambda_id:.4f}"
        )
        epoch_attacker_history.append(avg['attacker'])

        # ==============================================================
        # Stage 1 checkpoint (for collapse recovery)
        # ==============================================================
        if epoch == warmup_epochs:
            save_checkpoint(
                stage1_ckpt_path, epoch, tokenizer, reid_head, attacker,
                id_adversary, center_loss_fn,
                opt_main, opt_attacker, opt_id_adv, opt_center,
                best_rank1, collapse_count,
                lambda_priv_start, lambda_id_start)
            print("  -> Stage 1 checkpoint saved for collapse recovery")

        # ==============================================================
        # Evaluation
        # ==============================================================
        eval_results = None
        if epoch % cfg['eval_every'] == 0 or epoch == total_epochs:
            eval_results = evaluate_full_v4(
                tokenizer, reid_head, attacker, cfg, device)

            for key in ('rank1', 'rank5', 'rank10', 'mAP', 'mINP',
                        'psnr', 'ssim', 'lpips', 'pu_score'):
                writer.add_scalar(f'Eval/{key}', eval_results[key], epoch)

            # ----------------------------------------------------------
            # Adversarial collapse detection
            # ----------------------------------------------------------
            if (epoch > 40 and not is_warmup
                    and eval_results['rank1'] < collapse_threshold):
                collapse_count += 1
                print(f"\n  [!] ADVERSARIAL COLLAPSE DETECTED [!]")
                print(f"    Rank-1={eval_results['rank1']:.2f}% "
                      f"< threshold={collapse_threshold}%")
                print(f"    Collapse #{collapse_count}/{max_recoveries}")

                if collapse_count >= max_recoveries:
                    print("  [X] Maximum recoveries exceeded. Stopping training.")
                    break

                if os.path.isfile(stage1_ckpt_path):
                    print(f"  -> Restoring from {stage1_ckpt_path}")
                    ckpt = torch.load(stage1_ckpt_path, map_location=device,
                                      weights_only=False)
                    tokenizer.load_state_dict(ckpt['tokenizer'])
                    reid_head.load_state_dict(ckpt['reid_head'])
                    lambda_priv_start *= 0.5
                    lambda_priv_max   *= 0.5
                    lambda_id_start   *= 0.5
                    lambda_id_max     *= 0.5
                    print(f"  -> Halved lambdas: priv_max={lambda_priv_max:.4f}, "
                          f"id_max={lambda_id_max:.4f}")
                else:
                    print("  [WARNING] No stage1 checkpoint found for recovery.")

            # ----------------------------------------------------------
            # Best model
            # ----------------------------------------------------------
            if eval_results['rank1'] > best_rank1:
                best_rank1 = eval_results['rank1']
                save_checkpoint(
                    os.path.join(cfg['checkpoint_dir'], 'best_model.pth'),
                    epoch, tokenizer, reid_head, attacker,
                    id_adversary, center_loss_fn,
                    opt_main, opt_attacker, opt_id_adv, opt_center,
                    best_rank1, collapse_count,
                    lambda_priv_start, lambda_id_start)
                print(f"  ** New best Rank-1: {best_rank1:.2f}%")

        # ==============================================================
        # Periodic checkpoint
        # ==============================================================
        if epoch % cfg['save_every'] == 0:
            save_checkpoint(
                os.path.join(cfg['checkpoint_dir'], f'checkpoint_ep{epoch}.pth'),
                epoch, tokenizer, reid_head, attacker,
                id_adversary, center_loss_fn,
                opt_main, opt_attacker, opt_id_adv, opt_center,
                best_rank1, collapse_count,
                lambda_priv_start, lambda_id_start)

        # ==============================================================
        # CSV log
        # ==============================================================
        with open(csv_path, 'a') as f:
            r1   = eval_results['rank1']    if eval_results else ''
            r5   = eval_results['rank5']    if eval_results else ''
            r10  = eval_results['rank10']   if eval_results else ''
            mAP  = eval_results['mAP']      if eval_results else ''
            mINP = eval_results['mINP']     if eval_results else ''
            ep   = eval_results['psnr']     if eval_results else ''
            es   = eval_results['ssim']     if eval_results else ''
            el   = eval_results['lpips']    if eval_results else ''
            pu   = eval_results['pu_score'] if eval_results else ''
            f.write(
                f"{epoch},{stage_str},{avg['ce']:.6f},{avg['triplet']:.6f},"
                f"{avg['vq']:.6f},{avg['center']:.6f},"
                f"{avg['attacker']:.6f},{avg['privacy']:.6f},"
                f"{avg['id_confusion']:.6f},{avg['id_adv_acc']:.2f},"
                f"{avg['noise']:.6f},{avg['total']:.6f},"
                f"{avg['psnr']:.4f},{tokenizer.noise_scale.item():.4f},"
                f"{avg['cb_util']:.4f},{global_util:.4f},"
                f"{lambda_priv:.6f},{lambda_id:.6f},"
                f"{r1},{r5},{r10},{mAP},{mINP},{ep},{es},{el},{pu}\n"
            )

    # ==================================================================
    # Training Complete
    # ==================================================================
    print(f"\n{'=' * 70}")
    print("Entropy Confusion Training Complete!")
    print(f"  Best Rank-1:         {best_rank1:.2f}%")
    print(f"  Collapse recoveries: {collapse_count}/{max_recoveries}")
    print(f"  Checkpoints:         {cfg['checkpoint_dir']}")
    print(f"{'=' * 70}")

    save_checkpoint(
        os.path.join(cfg['checkpoint_dir'], 'final_model.pth'),
        total_epochs, tokenizer, reid_head, attacker,
        id_adversary, center_loss_fn,
        opt_main, opt_attacker, opt_id_adv, opt_center,
        best_rank1, collapse_count,
        lambda_priv_start, lambda_id_start)

    # ==================================================================
    # Post-Hoc Strong Attacker Validation
    # ==================================================================
    if cfg.get('posthoc_eval', True):
        print("\n" + "=" * 70)
        print("PHASE 2: Post-Hoc Strong Attacker Validation")
        print("=" * 70)

        final_ckpt_path = os.path.join(cfg['checkpoint_dir'], 'final_model.pth')
        if os.path.isfile(final_ckpt_path):
            ckpt = torch.load(final_ckpt_path, map_location=device,
                              weights_only=False)
            tokenizer.load_state_dict(ckpt['tokenizer'])
            reid_head.load_state_dict(ckpt['reid_head'])
            print(f"  Loaded final model (epoch {ckpt.get('epoch', '?')}, "
                  f"Rank-1={ckpt.get('best_rank1', 0):.2f}%)")

        posthoc_results = run_posthoc_attacker_suite(tokenizer, cfg, device)

        # Attacker convergence canary check (first 10 vs final 10 epochs)
        first_10_att = epoch_attacker_history[:10]
        final_10_att = epoch_attacker_history[-10:]
        mean_first = float(np.mean(first_10_att)) if first_10_att else 0.0
        mean_final = float(np.mean(final_10_att)) if final_10_att else 0.0

        if mean_first < 1e-4:
            print("\n" + "!" * 70)
            print("[WARNING] ATTACKER CONVERGENCE CANARY TRIGGERED!")
            print(f"  Suspiciously low initial attacker loss: mean_first = {mean_first:.6f} < 1e-4")
            print("!" * 70 + "\n")
            attacker_converged = False
        elif mean_final >= mean_first:
            print("\n" + "!" * 70)
            print("[WARNING] ATTACKER CONVERGENCE CANARY TRIGGERED!")
            print(f"  Mean attacker loss over first 10 epochs: {mean_first:.6f}")
            print(f"  Mean attacker loss over final 10 epochs: {mean_final:.6f}")
            print("  Attacker loss did not decrease meaningfully.")
            print("!" * 70 + "\n")
            attacker_converged = False
        else:
            print(f"\n[Attacker Canary Check] PASS: initial={mean_first:.4f} -> final={mean_final:.4f}")
            attacker_converged = True

        final_results['attacker_converged'] = attacker_converged
        final_results['best_rank1']     = best_rank1
        final_results['collapse_count'] = collapse_count
        final_results['noise_scale']    = tokenizer.noise_scale.item()

        json_path = os.path.join(cfg['output_dir'], 'eval_results_v4.json')
        with open(json_path, 'w') as f:
            json.dump(final_results, f, indent=2)
        print(f"\n  -> Final results saved to {json_path}")

        print(f"\n{'=' * 70}")
        print("PUBLICATION-READY SUMMARY  [Entropy Confusion]")
        print(f"{'=' * 70}")
        print(f"  [ReID Utility]")
        print(f"    Rank-1:  {final_results['rank1']:.2f}%")
        print(f"    Rank-5:  {final_results['rank5']:.2f}%")
        print(f"    Rank-10: {final_results['rank10']:.2f}%")
        print(f"    mAP:     {final_results['mAP']:.2f}%")
        print(f"    mINP:    {final_results['mINP']:.2f}%")
        print(f"  [Visual Privacy (co-trained attacker)]")
        print(f"    PSNR:    {final_results['psnr']:.2f} dB")
        print(f"    SSIM:    {final_results['ssim']:.4f}")
        print(f"    LPIPS:   {final_results['lpips']:.4f}")
        print(f"  [Composite]")
        print(f"    PU-Score: {final_results['pu_score']:.1f}")
        print(f"  [Post-Hoc Validation (independent strong attackers)]")
        print(f"    Strong Recon PSNR: "
              f"{posthoc_results.get('posthoc_recon_psnr', 0):.2f} dB")
        print(f"    Strong Recon SSIM: "
              f"{posthoc_results.get('posthoc_recon_ssim', 0):.4f}")
        print(f"    Strong ID Top-1:   "
              f"{posthoc_results.get('posthoc_id_top1', 0):.1f}%")
        chance = 100.0 / cfg['num_classes']
        print(f"    Chance level:      {chance:.1f}%")
        print(f"  [Training Meta]")
        print(f"    Noise scale sigma:   {final_results['noise_scale']:.4f}")
        print(f"    Collapse recoveries: {collapse_count}")
        print(f"{'=' * 70}")

    writer.close()
    print("\nDone.")


if __name__ == '__main__':
    main()

"""
Loss functions for PrivToken-ReID.

Provides:
- TripletLossWithHardMining: Hard positive/negative mining triplet loss
- CrossEntropyWithLabelSmoothing: Label-smoothed cross-entropy
- LPIPSPerceptualLoss: Perceptual similarity wrapper
- compute_reid_loss(): Combined CE + Triplet
- compute_reconstruction_loss(): L1 + LPIPS for privacy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips


class TripletLossWithHardMining(nn.Module):
    """
    Batch-hard triplet loss with online hard positive/negative mining.

    For each anchor in the batch, selects the hardest positive (farthest same-ID)
    and hardest negative (closest different-ID), then applies margin ranking loss.

    Args:
        margin (float): Triplet margin (default: 0.3).

    Inputs:
        feat (Tensor): Feature embeddings, shape (B, D).
        labels (Tensor): Identity labels, shape (B,).

    Returns:
        loss (Tensor): Scalar triplet loss.
    """

    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin

    def forward(self, feat, labels):
        """
        Args:
            feat (Tensor): shape (B, D) — feature vectors (ft from BNNeck).
            labels (Tensor): shape (B,) — identity labels.

        Returns:
            loss (Tensor): scalar triplet loss.
        """
        # Pairwise L2 distance matrix: (B, B)
        dist = torch.cdist(feat, feat, p=2)  # (B, B)

        # Masks for same-identity and different-identity pairs
        same_mask = labels.unsqueeze(0) == labels.unsqueeze(1)  # (B, B)
        diff_mask = ~same_mask  # (B, B)

        # Hardest positive: maximum distance among same-identity pairs
        # Set non-positive distances to 0, take max
        ap = (dist * same_mask.float()).max(dim=1)[0]  # (B,)

        # Hardest negative: minimum distance among different-identity pairs
        # Set same-identity distances to a large value, then take min
        an = (dist + same_mask.float() * 1e6).min(dim=1)[0]  # (B,)

        # Triplet loss with margin
        loss = F.relu(ap - an + self.margin).mean()

        return loss


class CrossEntropyWithLabelSmoothing(nn.Module):
    """
    Cross-entropy loss with label smoothing.

    Distributes a fraction of the probability mass uniformly across all classes,
    which acts as a regularizer and prevents over-confident predictions.

    Formula: L = (1 - smoothing) * CE(y, p) + smoothing * mean(-log(p))

    Args:
        num_classes (int): Total number of classes.
        smoothing (float): Label smoothing factor (default: 0.1).

    Inputs:
        logits (Tensor): Raw predictions, shape (B, C).
        labels (Tensor): Ground-truth class indices, shape (B,).

    Returns:
        loss (Tensor): Scalar loss.
    """

    def __init__(self, num_classes, smoothing=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits, labels):
        """
        Args:
            logits (Tensor): shape (B, C) — raw class scores.
            labels (Tensor): shape (B,) — ground-truth class indices.

        Returns:
            loss (Tensor): scalar label-smoothed cross-entropy.
        """
        log_probs = F.log_softmax(logits, dim=1)  # (B, C)

        # Smooth target distribution
        # One-hot with smoothing: (1 - s) on correct class, s/C on all classes
        with torch.no_grad():
            smooth_labels = torch.full_like(log_probs, self.smoothing / self.num_classes)
            smooth_labels.scatter_(1, labels.unsqueeze(1), self.confidence + self.smoothing / self.num_classes)

        loss = (-smooth_labels * log_probs).sum(dim=1).mean()

        return loss


class LPIPSPerceptualLoss(nn.Module):
    """
    LPIPS perceptual similarity loss wrapper.

    Uses a pretrained AlexNet to compute perceptual distance between images.
    The LPIPS network is kept frozen — it only serves as a metric.

    Inputs:
        pred (Tensor): Predicted image in [0, 1] range, shape (B, 3, H, W).
        target (Tensor): Target image in [0, 1] range, shape (B, 3, H, W).

    Returns:
        loss (Tensor): Scalar mean LPIPS distance.
    """

    def __init__(self):
        super().__init__()
        self.fn = lpips.LPIPS(net='alex')
        # Keep LPIPS network frozen — never train it
        self.fn.eval()
        for p in self.fn.parameters():
            p.requires_grad_(False)

    def forward(self, pred, target):
        """
        Args:
            pred (Tensor): Predicted images in [0, 1], shape (B, 3, H, W).
            target (Tensor): Target images in [0, 1], shape (B, 3, H, W).

        Returns:
            loss (Tensor): Scalar mean LPIPS distance.
        """
        # Move LPIPS net to same device (lazy device transfer)
        if next(self.fn.parameters()).device != pred.device:
            self.fn = self.fn.to(pred.device)

        # LPIPS expects images in [-1, 1]
        pred_scaled = pred * 2.0 - 1.0   # [0,1] → [-1,1]
        target_scaled = target * 2.0 - 1.0  # [0,1] → [-1,1]

        return self.fn(pred_scaled, target_scaled).mean()


def compute_reid_loss(logits, ft, labels, ce_loss_fn, triplet_loss_fn, alpha=1.0):
    """
    Combined Re-ID loss: cross-entropy + weighted triplet loss.

    Args:
        logits (Tensor): Classification logits, shape (B, C).
        ft (Tensor): Pre-BN features for triplet loss, shape (B, D).
        labels (Tensor): Identity labels, shape (B,).
        ce_loss_fn (nn.Module): Label-smoothed cross-entropy loss.
        triplet_loss_fn (nn.Module): Hard-mining triplet loss.
        alpha (float): Weight for triplet loss term.

    Returns:
        total_loss (Tensor): Scalar combined loss.
        ce_loss (Tensor): Scalar CE loss (for logging).
        triplet_loss (Tensor): Scalar triplet loss (for logging).
    """
    ce_loss = ce_loss_fn(logits, labels)
    triplet_loss = triplet_loss_fn(ft, labels)
    total_loss = ce_loss + alpha * triplet_loss
    return total_loss, ce_loss, triplet_loss


def compute_reconstruction_loss(recon, original, lpips_fn):
    """
    Reconstruction loss: L1 + LPIPS perceptual distance.

    This is MINIMIZED by the attacker and (negated) MAXIMIZED by the tokenizer.

    Args:
        recon (Tensor): Reconstructed image in [0, 1], shape (B, 3, H, W).
        original (Tensor): Original image in [0, 1], shape (B, 3, H, W).
        lpips_fn (LPIPSPerceptualLoss): LPIPS loss module.

    Returns:
        loss (Tensor): Scalar reconstruction loss (L1 + LPIPS).
    """
    l1 = F.l1_loss(recon, original)
    perceptual = lpips_fn(recon, original)
    return l1 + perceptual


def compute_region_weighted_reconstruction_loss(
    recon, original, lpips_fn,
    head_weight=3.0, torso_weight=1.5, legs_weight=0.5,
):
    """
    Region-aware privacy loss: stronger suppression for head/face areas.

    Not all image regions are equally privacy-sensitive. The head/face region
    carries the most biometric identity information, so we apply stronger
    privacy pressure there. The lower body (legs/shoes) carries less
    identity-sensitive information, so we apply lighter pressure.

    Spatial prior on the pixel grid (256×128 image):
        rows 0–63   (top 25%)   → head_weight  (face, hair — most sensitive)
        rows 64–168 (middle 40%) → torso_weight (clothing — moderate)
        rows 169–255 (bottom 35%) → legs_weight  (shoes, legs — least sensitive)

    LPIPS is computed globally (not region-aware) and added uniformly.

    Args:
        recon (Tensor): Reconstructed image in [0, 1], shape (B, 3, H, W).
        original (Tensor): Original image in [0, 1], shape (B, 3, H, W).
        lpips_fn (LPIPSPerceptualLoss): LPIPS loss module.
        head_weight (float): Weight for head/face region.
        torso_weight (float): Weight for torso region.
        legs_weight (float): Weight for lower body region.

    Returns:
        loss (Tensor): Scalar region-weighted reconstruction loss.
    """
    B, C, H, W = recon.shape

    # Define region boundaries (proportional to image height)
    head_end = int(H * 0.25)     # top 25%
    torso_end = int(H * 0.65)    # middle 40%
    # legs: remaining 35%

    # Create spatial weight map
    weight_map = torch.ones(1, 1, H, W, device=recon.device)
    weight_map[:, :, :head_end, :] = head_weight
    weight_map[:, :, head_end:torso_end, :] = torso_weight
    weight_map[:, :, torso_end:, :] = legs_weight

    # Normalize weights so they average to 1.0
    weight_map = weight_map / weight_map.mean()

    # Weighted L1 loss
    l1_map = torch.abs(recon - original)  # (B, 3, H, W)
    weighted_l1 = (l1_map * weight_map).mean()

    # LPIPS is global (perceptual network doesn't support region weighting)
    perceptual = lpips_fn(recon, original)

    return weighted_l1 + perceptual

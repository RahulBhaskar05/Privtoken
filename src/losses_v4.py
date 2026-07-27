"""
V4 Loss Functions for PrivToken-ReID.

Core contributions (paper-novel):
    - DeepIdentityAdversary: 3-layer MLP identity classifier (GRL removed).
      In this ablation the adversary is trained in the inner loop only;
      the tokenizer receives no gradient from this path.
    - TokenNoiseRegularization: Pushes token distribution toward N(0,I),
      breaking pixel-level correlation exploited by Feature Inversion
    - EntropyGuidedPrivacyLoss: Per-token adaptive privacy pressure

Adopted components (standard, cited):
    - CenterLoss: Wen et al., ECCV 2016
    - compute_multipart_reid_loss: PCB-style per-part CE + Triplet

Note (GRL ablation):
    The Gradient Reversal Layer has been fully removed from this version.
    adversarial_forward() now detaches z_q before pooling, so the tokenizer
    receives zero gradient from the identity adversary in the outer step.
    All other losses and training stages are unchanged.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.losses import (
    TripletLossWithHardMining,
    CrossEntropyWithLabelSmoothing,
    LPIPSPerceptualLoss,
    compute_reid_loss,
    compute_reconstruction_loss,
    compute_region_weighted_reconstruction_loss,
)


# =========================================================================
# [GRL REMOVED] — GradientReversalFunction and GradientReversalLayer
# have been fully removed in this ablation. DeepIdentityAdversary no longer
# reverses gradients to the tokenizer. See adversarial_forward() below.
# =========================================================================


# =========================================================================
# Deep Identity Adversary (GRL removed — ablation version)
# =========================================================================

class DeepIdentityAdversary(nn.Module):
    """Deep identity classifier for adversarial privacy training (no GRL).

    A 3-layer MLP with spectral normalization that predicts person identity
    from pooled token features. Spectral norm stabilizes training by bounding
    the Lipschitz constant of the classifier.

    GRL REMOVED: In this ablation the GradientReversalLayer has been fully
    removed. adversarial_forward() detaches z_q before pooling, so the
    tokenizer receives zero gradient from this path. The classifier is still
    trained in the inner loop (classify()), but provides no adversarial
    pressure to the encoder.

    Two forward modes:
        1. classify(): Normal forward for inner-loop adversary pre-training.
           Gradients flow normally — the classifier improves at predicting ID.
        2. adversarial_forward(): Detached forward (no-op for tokenizer).
           Keeps the same call signature as the GRL version so train_v4.py
           requires zero modifications.

    Architecture:
        GAP(z_q.detach()) → SN-Linear(D,1024) → BN → LeakyReLU → Drop
        → SN-Linear(1024,1024) → BN → LeakyReLU → Drop
        → Linear(1024, num_classes) → CE loss

    Args:
        input_dim: Token embedding dimension (default: 512).
        num_classes: Number of identity classes (default: 751).
        hidden_dim: Hidden layer width (default: 1024).
        dropout: Dropout rate (default: 0.5).
    """

    def __init__(self, input_dim=512, num_classes=751, hidden_dim=1024, dropout=0.5):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        # self.grl removed — no GradientReversalLayer

        self.classifier = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(input_dim, hidden_dim)),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            nn.utils.spectral_norm(nn.Linear(hidden_dim, hidden_dim)),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            nn.utils.spectral_norm(nn.Linear(hidden_dim, hidden_dim // 2)),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout // 2),
            nn.Linear(hidden_dim // 2, num_classes),
        )
        self.ce = nn.CrossEntropyLoss()

    def _pool(self, z_q):
        """Global average pool token grid to a single vector."""
        return self.gap(z_q).flatten(1)  # (B, D)

    def classify(self, z_q, labels):
        """Normal forward — for inner-loop adversary pre-training.

        No gradient reversal. The classifier learns to predict identity
        from detached token representations.

        Args:
            z_q (Tensor): Quantized tokens (B, D, H, W) — should be detached.
            labels (Tensor): Person identity labels (B,).

        Returns:
            loss (Tensor): Scalar CE loss.
            accuracy (float): Classification accuracy (%).
        """
        pooled = self._pool(z_q)
        logits = self.classifier(pooled)
        loss = self.ce(logits, labels)
        acc = (logits.argmax(1) == labels).float().mean().item() * 100
        return loss, acc

    def adversarial_forward(self, z_q, labels, lambda_id=1.0):
        """Detached forward — outer-step call with GRL removed.

        z_q is detached before pooling so the tokenizer receives zero
        gradient from this path. The loss still trains the classifier
        (via opt_id_adv.step()), matching the inner-loop classify() behaviour.
        The lambda_id argument is accepted but ignored (kept for API
        compatibility with train_v4.py).

        Args:
            z_q (Tensor): Quantized tokens (B, D, H, W).
            labels (Tensor): Person identity labels (B,).
            lambda_id (float): Ignored (kept for API compatibility).

        Returns:
            loss (Tensor): Scalar CE loss (does NOT flow to tokenizer).
            accuracy (float): Classification accuracy (%).
        """
        pooled = self._pool(z_q.detach())  # detach: no gradient to tokenizer
        logits = self.classifier(pooled)
        loss = self.ce(logits, labels)
        acc = (logits.argmax(1) == labels).float().mean().item() * 100
        return loss, acc

    def freeze(self):
        """Freeze all parameters."""
        for p in self.parameters():
            p.requires_grad_(False)

    def unfreeze(self):
        """Unfreeze all parameters."""
        for p in self.parameters():
            p.requires_grad_(True)


# =========================================================================
# Center Loss (Wen et al., ECCV 2016) — Adopted, not novel
# =========================================================================

class CenterLoss(nn.Module):
    """Center loss for intra-class compactness.

    Learns a center vector for each class and pulls features toward their
    respective centers. Reduces intra-class variation, directly improving mAP.

    Uses its own optimizer (typically SGD with lr=0.5) because the center
    updates have different dynamics than the rest of the network.

    Reference: Wen et al., "A Discriminative Feature Learning Approach
    for Deep Face Recognition", ECCV 2016.

    Args:
        num_classes: Number of identity classes.
        feat_dim: Feature dimension.
    """

    def __init__(self, num_classes, feat_dim):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
        nn.init.xavier_uniform_(self.centers)

    def forward(self, feat, labels):
        """
        Args:
            feat (Tensor): Feature vectors (B, D). Use pre-BN features (ft).
            labels (Tensor): Identity labels (B,).

        Returns:
            loss (Tensor): Scalar center loss.
        """
        # L2-normalize features and centers to prevent exploding distances
        # when feature magnitudes grow during training. Without this,
        # center_loss explodes (1.1 → 315+ by epoch 29) because raw
        # ResNet features have unbounded norms.
        feat_norm = F.normalize(feat, p=2, dim=1)
        centers_norm = F.normalize(self.centers[labels], p=2, dim=1)
        loss = ((feat_norm - centers_norm) ** 2).sum(dim=1).mean()
        return loss


# =========================================================================
# Token Noise Regularization (Paper Contribution)
# =========================================================================

class TokenNoiseRegularization(nn.Module):
    """Regularizes token distribution toward unit Gaussian N(0, I).

    Minimizes the discrepancy between the first two moments of the token
    distribution and those of a standard Gaussian. This has two effects:

    1. Makes the token space look noise-like →  harder for Feature Inversion
       to extract structured information (breaks pixel-level correlation).
    2. Increases unconditional entropy H(Z_q) → combined with GRL identity
       adversary, bounds mutual information I(ID; Z_q) ≈ 0.

    The loss is:
        L_noise = E[||μ(z_q)||²] + E[(σ²(z_q) - 1)²]

    This is equivalent to matching the first two moments of the empirical
    token distribution to N(0, I), which is a relaxation of full KL
    divergence that avoids the Gaussian assumption on individual tokens
    (which would conflict with the discrete VQ structure).
    """

    def forward(self, z_q):
        """
        Args:
            z_q (Tensor): Quantized token features (B, D, H, W).

        Returns:
            loss (Tensor): Scalar moment-matching regularization loss.
        """
        # Flatten to (N, D) where N = B * H * W
        z_flat = z_q.permute(0, 2, 3, 1).reshape(-1, z_q.size(1))

        # First moment: push mean toward 0
        mean = z_flat.mean(dim=0)  # (D,)
        loss_mean = mean.pow(2).mean()

        # Second moment: push variance toward 1
        var = z_flat.var(dim=0)  # (D,)
        loss_var = (var - 1.0).pow(2).mean()

        return loss_mean + loss_var


# =========================================================================
# Entropy-Guided Privacy Loss (Paper Contribution)
# =========================================================================

class EntropyGuidedPrivacyLoss(nn.Module):
    """Per-token adaptive privacy weighting based on information entropy.

    Tokens with high information entropy (typically face, gait) carry more
    identity-sensitive information and receive stronger privacy pressure.
    Tokens with low entropy (background, shoes) receive lighter pressure,
    allowing the model to retain discriminative features there.

    This is analogous to attention-based occlusion handling (OAT, Li et al.)
    but applied to the privacy loss rather than the feature extractor.

    The weight map is computed from token-level entropy and upsampled to
    pixel resolution for the L1 reconstruction loss.
    """

    def __init__(self, base_weight=1.0, entropy_scale=2.0):
        """
        Args:
            base_weight: Minimum weight for any region (default 1.0).
            entropy_scale: How much extra weight high-entropy regions get.
        """
        super().__init__()
        self.base_weight = base_weight
        self.entropy_scale = entropy_scale

    def forward(self, recon, original, token_entropy, lpips_fn):
        """
        Args:
            recon (Tensor): Reconstructed image (B, 3, H_img, W_img) in [0,1].
            original (Tensor): Original image (B, 3, H_img, W_img) in [0,1].
            token_entropy (Tensor): Per-token entropy (B, H_tok, W_tok).
            lpips_fn (LPIPSPerceptualLoss): LPIPS loss module.

        Returns:
            loss (Tensor): Scalar entropy-weighted reconstruction loss.
        """
        B, C, H_img, W_img = recon.shape

        # Normalize entropy to [0, 1] per image
        e_flat = token_entropy.reshape(B, -1)
        e_min = e_flat.min(dim=1, keepdim=True)[0].unsqueeze(2)  # (B, 1, 1)
        e_max = e_flat.max(dim=1, keepdim=True)[0].unsqueeze(2)  # (B, 1, 1)
        e_norm = (token_entropy - e_min) / (e_max - e_min + 1e-8)  # (B, H, W)

        # Upscale to image resolution
        weight_map = F.interpolate(
            e_norm.unsqueeze(1),  # (B, 1, H_tok, W_tok)
            size=(H_img, W_img),
            mode='bilinear',
            align_corners=False,
        )  # (B, 1, H_img, W_img)

        # Higher entropy → stronger privacy weight
        weight_map = self.base_weight + self.entropy_scale * weight_map
        weight_map = weight_map / weight_map.mean()  # normalize to avg=1

        # Weighted L1 loss
        l1_map = torch.abs(recon - original)  # (B, 3, H, W)
        weighted_l1 = (l1_map * weight_map).mean()

        # LPIPS is global (perceptual network doesn't support region weighting)
        perceptual = lpips_fn(recon, original)

        return weighted_l1 + perceptual


# =========================================================================
# Multi-Part ReID Loss
# =========================================================================

def compute_multipart_reid_loss(outputs, labels, ce_loss_fn, triplet_loss_fn,
                                 alpha_triplet=1.0):
    """Combined CE + Triplet loss across global and part-level branches.

    The multi-part structure follows PCB (Sun et al., ECCV 2018) and is
    standard practice in person ReID. It is explicitly NOT claimed as a
    contribution — it is adopted engineering for closing the utility gap.

    Each branch (1 global + N parts) independently computes CE and Triplet
    losses. The total is averaged across branches to prevent any single
    branch from dominating.

    Args:
        outputs (dict): From MultiGranularityHead.forward():
            'global_ft': (B, D) pre-BN features for triplet
            'global_logits': (B, C) logits for CE
            'part_fts': list of (B, D) per-part pre-BN features
            'part_logits': list of (B, C) per-part logits
        labels (Tensor): Identity labels (B,).
        ce_loss_fn: Label-smoothed cross-entropy.
        triplet_loss_fn: Hard-mining triplet loss.
        alpha_triplet: Triplet loss weight.

    Returns:
        total_loss (Tensor): Scalar combined loss.
        ce_loss (Tensor): Average CE component (for logging).
        triplet_loss (Tensor): Average triplet component (for logging).
    """
    # Global branch
    ce_global = ce_loss_fn(outputs['global_logits'], labels)
    tri_global = triplet_loss_fn(outputs['global_ft'], labels)

    total_ce = ce_global
    total_tri = tri_global

    # Part branches
    num_parts = len(outputs['part_logits'])
    for i in range(num_parts):
        total_ce = total_ce + ce_loss_fn(outputs['part_logits'][i], labels)
        total_tri = total_tri + triplet_loss_fn(outputs['part_fts'][i], labels)

    # Average across all branches (1 global + num_parts)
    num_branches = 1 + num_parts
    avg_ce = total_ce / num_branches
    avg_tri = total_tri / num_branches

    total_loss = avg_ce + alpha_triplet * avg_tri
    return total_loss, avg_ce, avg_tri

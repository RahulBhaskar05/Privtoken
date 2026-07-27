"""
Privacy Tokenizer: ResNet-50 encoder with EMA Vector Quantization (VQ-VAE).

Provides:
- VectorQuantizer: EMA-updated discrete codebook with straight-through estimator
- PrivacyTokenizer: Full encoder pipeline (ResNet-50 → projection → VQ)

The VQ codebook creates an information bottleneck that preserves structural
information for Re-ID while destroying fine-grained biometric details.
EMA updates prevent codebook collapse by continuously updating every entry.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
import timm # Added import
import math # Added import


def _adapt_vit_pos_embed(model, new_img_size_h, new_img_size_w):
    """
    Adapts a pre-trained DINOv2 model's positional embeddings for a custom
    rectangular input image size.

    Assumes `model.pos_embed` contains only the spatial patch positional embeddings,
    and CLS/register tokens are handled separately by the model's architecture.
    """
    patch_size = model.patch_embed.patch_size[0]
    orig_pos_embed_spatial = model.pos_embed.clone() # This is assumed to be only spatial patches
    embed_dim = orig_pos_embed_spatial.shape[-1]

    # Original grid size from where the current model.pos_embed_spatial was originally for
    # Based on DEBUG output, this is (518, 518)
    orig_h = model.patch_embed.img_size[0] // patch_size # 518 // 14 = 37
    orig_w = model.patch_embed.img_size[1] // patch_size # 518 // 14 = 37

    # Verify consistency: the number of tokens in pos_embed should match the inferred grid
    if orig_h * orig_w != orig_pos_embed_spatial.shape[1]: # 37*37 = 1369 vs 1369 (should match)
        raise ValueError(
            f"Consistency check failed: Inferred grid ({orig_h}x{orig_w}={orig_h*orig_w}) vs. "
            f"actual spatial pos embeds ({orig_pos_embed_spatial.shape[1]}). "
            f"Model config: img_size={model.patch_embed.img_size}, "
            f"patch_size={patch_size}, pos_embed.shape={model.pos_embed.shape}"
        )

    # Calculate new grid size for our target input image (256x128)
    new_h = new_img_size_h // patch_size # 256 // 14 = 18
    new_w = new_img_size_w // patch_size # 128 // 14 = 9

    # Ensure minimum 1 patch if image size is smaller than patch size
    new_h = max(1, new_h)
    new_w = max(1, new_w)

    # Reshape spatial positional embeddings for interpolation (1, dim, H, W)
    pos_embed_spatial_reshaped = orig_pos_embed_spatial.reshape(1, orig_h, orig_w, embed_dim).permute(0, 3, 1, 2)

    # Interpolate spatial positional embeddings
    pos_embed_resized = F.interpolate(
        pos_embed_spatial_reshaped,
        size=(new_h, new_w),
        mode='bicubic',
        align_corners=False
    )

    # Reshape back to (1, num_patches_new, embed_dim)
    pos_embed_resized = pos_embed_resized.permute(0, 2, 3, 1).flatten(1, 2)

    # Update model's internal pos_embed (only the spatial part)
    model.pos_embed = nn.Parameter(pos_embed_resized)

    # Update model's internal image size tracking in patch_embed
    model.patch_embed.img_size = (new_img_size_h, new_img_size_w)
    if hasattr(model.patch_embed, 'num_patches'):
        model.patch_embed.num_patches = new_h * new_w
    if hasattr(model, 'num_patches'):
        model.num_patches = new_h * new_w

    return model


class VectorQuantizer(nn.Module):
    """VQ-VAE with EMA codebook updates.

    EMA prevents codebook collapse by updating every entry as a running
    average of the encoder outputs assigned to it. Commitment loss only
    — no codebook gradient needed because EMA handles it.

    Args:
        codebook_size: number of codebook entries K
        dim: dimension D of each entry
        beta: commitment loss weight (default 0.25)
        decay: EMA decay rate (default 0.99, higher = slower update)
        eps: Laplace smoothing to prevent division by zero on dead entries
    """
    def __init__(self, codebook_size=512, dim=512, beta=0.10, decay=0.95, eps=1e-5):
        super().__init__()
        self.K = codebook_size
        self.D = dim
        self.beta = beta
        self.decay = decay
        self.eps = eps

        # Codebook is a buffer not a parameter — EMA updates it, not gradient
        embed = torch.randn(codebook_size, dim)
        self.register_buffer('embedding', embed)
        self.register_buffer('ema_cluster_size', torch.zeros(codebook_size))
        self.register_buffer('ema_embed_sum', embed.clone())

    def forward(self, z):
        """
        Args:
            z: (B, D, H, W) — encoder output before quantisation
        Returns:
            z_q_ste: (B, D, H, W) — quantised, straight-through gradient
            vq_loss: scalar — commitment loss only
            indices: (B, H*W) — codebook index per spatial position
            utilisation: float — fraction of K entries used this batch
        """
        B, D, H, W = z.shape
        # Flatten spatial dims: (B, D, H, W) → (B*H*W, D)
        z_flat = z.permute(0, 2, 3, 1).reshape(-1, D)

        # L2 distances to all K codebook entries: (B*H*W, K)
        dist = (z_flat.pow(2).sum(1, keepdim=True)
                - 2 * z_flat @ self.embedding.T
                + self.embedding.pow(2).sum(1))

        indices = dist.argmin(1)                  # (B*H*W,)
        z_q = self.embedding[indices]             # (B*H*W, D)

        # EMA codebook update — training only, no_grad
        if self.training:
            with torch.no_grad():
                one_hot = F.one_hot(indices, self.K).float()   # (B*H*W, K)
                n = one_hot.sum(0)                              # (K,) — assignments per entry
                embed_sum = one_hot.T @ z_flat                  # (K, D) — sum of assigned vecs

                self.ema_cluster_size.mul_(self.decay).add_(n, alpha=1 - self.decay)
                self.ema_embed_sum.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)

                # Laplace smoothing: prevent dead-entry division-by-zero
                n_smooth = ((self.ema_cluster_size + self.eps)
                            / (self.ema_cluster_size.sum() + self.K * self.eps)
                            * self.ema_cluster_size.sum())
                self.embedding.copy_(self.ema_embed_sum / n_smooth.unsqueeze(1))

        # Commitment loss: encoder output should stay close to chosen entry
        vq_loss = self.beta * F.mse_loss(z_q.detach(), z_flat)

        # Straight-through estimator: gradient bypasses argmin
        z_q_ste = z_flat + (z_q - z_flat).detach()
        z_q_ste = z_q_ste.reshape(B, H, W, D).permute(0, 3, 1, 2)  # (B, D, H, W)

        # Codebook utilisation: diagnostic metric, log every step
        utilisation = indices.unique().numel() / self.K

        return z_q_ste, vq_loss, indices.view(B, H * W), utilisation


class PrivacyTokenizer(nn.Module):
    """
    Full privacy tokenization pipeline.

    Architecture:
        Input image (B, 3, 256, 128)
        → ResNet-50 encoder (last-stride=1) → (B, 2048, 16, 8)
        → 1×1 Conv projection + BN → (B, token_dim, 16, 8)
        → VectorQuantizer → discrete tokens (B, token_dim, 16, 8)

    Args:
        codebook_size (int): Number of VQ codebook entries (e.g. 512).
        token_dim (int): Dimension of token embeddings (e.g. 512).
        vq_beta (float): VQ commitment loss weight (default: 0.25).
        backbone_type (str): Type of backbone to use, 'resnet50' or 'vit' (default: 'resnet50').

    Inputs:
        x (Tensor): Input images, shape (B, 3, 256, 128).

    Returns:
        z_q (Tensor): Quantized features, shape (B, token_dim, 16, 8).
        vq_loss (Tensor): Scalar VQ loss.
        indices (Tensor): Codebook indices, shape (B, 128).
        utilisation (float): Fraction of codebook entries used in this batch.
    """

    def __init__(self, codebook_size=512, token_dim=512, vq_beta=0.25, backbone_type='resnet50',
                 img_height=256, img_width=128, token_grid_h=16, token_grid_w=8):
        super().__init__()
        self.backbone_type = backbone_type
        self.output_spatial_size = (token_grid_h, token_grid_w) 

        if backbone_type == 'resnet50':
            # Load pretrained ResNet-50 and remove avgpool + fc
            backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            # Last-stride trick: set layer4 stride=1 to preserve spatial resolution
            # Output changes from (B, 2048, 8, 4) → (B, 2048, 16, 8) = 128 tokens at 16×16px each
            backbone.layer4[0].conv2.stride = (1, 1)
            backbone.layer4[0].downsample[0].stride = (1, 1)
            self.encoder = nn.Sequential(*list(backbone.children())[:-2])
            in_channels_proj = 2048
            self.adaptive_pool = None # Not needed for ResNet
        elif backbone_type == 'vit':
            # Load DINOv2 ViT-B/14
            # We use 'vit_base_patch14_reg4_dinov2_vitb14' which corresponds to ViT-B/14
            self.encoder = timm.create_model('vit_base_patch14_reg4_dinov2', pretrained=True)
            # Adapt positional embeddings for custom input size
            self.encoder = _adapt_vit_pos_embed(self.encoder, img_height, img_width)
            # ViT-B/14 embedding dim is 768
            in_channels_proj = self.encoder.embed_dim # 768

            # Adaptive pooling to match target spatial size (16, 8)
            self.adaptive_pool = nn.AdaptiveAvgPool2d(self.output_spatial_size)
        else:
            raise ValueError(f"Unknown backbone type: {backbone_type}")

        # Project backbone output channels → token_dim with 1x1 conv
        self.proj = nn.Conv2d(in_channels_proj, token_dim, kernel_size=1, bias=False)
        self.proj_bn = nn.BatchNorm2d(token_dim)

        # Vector quantizer (EMA)
        self.vq = VectorQuantizer(codebook_size, token_dim, beta=vq_beta)

    def encode_spatial(self, x):
        """Backbone output as a spatial feature map (B, C, H, W) before projection."""
        if self.backbone_type == 'resnet50':
            return self.encoder(x)
        if self.backbone_type == 'vit':
            features = self.encoder.forward_features(x)
            num_prefix_tokens = getattr(self.encoder, 'num_prefix_tokens', 1)
            patch_tokens = features[:, num_prefix_tokens:]
            B, num_patches, embedding_dim = patch_tokens.shape
            patch_size = self.encoder.patch_embed.patch_size[0]
            H_actual = x.shape[2] // patch_size
            W_actual = x.shape[3] // patch_size
            if num_patches != H_actual * W_actual:
                raise ValueError(
                    f"Number of patches ({num_patches}) does not match "
                    f"calculated grid size ({H_actual}x{W_actual}={H_actual * W_actual})"
                )
            feat = patch_tokens.permute(0, 2, 1).reshape(B, embedding_dim, H_actual, W_actual)
            return self.adaptive_pool(feat)
        raise ValueError(f"Unknown backbone type: {self.backbone_type}")

    def get_projected_features(self, x):
        """Projected features (B, token_dim, H, W) used by VQ and codebook reseeding."""
        return self.proj_bn(self.proj(self.encode_spatial(x)))

    def forward(self, x):
        """
        Args:
            x (Tensor): Input images, shape (B, 3, 256, 128).

        Returns:
            z_q (Tensor): Quantized features, shape (B, token_dim, 16, 8).
            vq_loss (Tensor): Scalar VQ loss.
            indices (Tensor): Codebook indices, shape (B, 128).
            utilisation (float): Fraction of codebook entries used.
        """
        feat = self.get_projected_features(x)              # (B, token_dim, 16, 8)
        z_q, vq_loss, indices, utilisation = self.vq(feat)
        return z_q, vq_loss, indices, utilisation        # removed pre-VQ feat, added utilisation

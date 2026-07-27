import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
import timm # Added import
import math # Added import

from src.tokenizer import VectorQuantizer


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


class PrivacyTokenizerV4(nn.Module):
    """Enhanced privacy tokenization pipeline for v4.

    Architecture (same backbone as v3):
        Input image (B, 3, 256, 128)
        → ResNet-50 encoder (last-stride=1) → (B, 2048, 16, 8)
        → 1×1 Conv projection + BN → (B, token_dim, 16, 8)
        → VectorQuantizer → discrete tokens (B, token_dim, 16, 8)
        → [Training only] Noise injection → (B, token_dim, 16, 8)

    Args:
        codebook_size (int): Number of VQ codebook entries (default: 2048).
        token_dim (int): Token embedding dimension (default: 512).
        vq_beta (float): VQ commitment loss weight (default: 0.08).
        backbone_type (str): Type of backbone to use, 'resnet50' or 'vit' (default: 'resnet50').

    Returns (forward):
        z_q (Tensor): Clean quantized features (B, D, 16, 8).
        vq_loss (Tensor): Scalar VQ commitment loss.
        indices (Tensor): Codebook indices (B, 128).
        utilisation (float): Fraction of codebook entries used.
        token_entropy (Tensor): Per-token entropy (B, 16, 8).
        z_q_noisy (Tensor): Noise-augmented tokens (B, D, 16, 8).
            Same as z_q during eval. During training, z_q + σ*ε.
    """

    def __init__(self, codebook_size=2048, token_dim=512, vq_beta=0.08, backbone_type='resnet50',
                 img_height=256, img_width=128, token_grid_h=16, token_grid_w=8,
                 enable_noise: bool = True):
        super().__init__()
        self.backbone_type = backbone_type
        self.output_spatial_size = (token_grid_h, token_grid_w) # Target spatial size for VQ input
        self.enable_noise = enable_noise

        if backbone_type == 'resnet50':
            # ResNet-50 encoder (same as v3)
            backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            backbone.layer4[0].conv2.stride = (1, 1)
            backbone.layer4[0].downsample[0].stride = (1, 1)
            self.encoder = nn.Sequential(*list(backbone.children())[:-2])
            in_channels_proj = 2048
            self.adaptive_pool = None # Not needed for ResNet
        elif backbone_type == 'vit':
            # Load DINOv2 ViT-B/14
            self.encoder = timm.create_model('vit_base_patch14_reg4_dinov2', pretrained=True)
            # Adapt positional embeddings for custom input size
            self.encoder = _adapt_vit_pos_embed(self.encoder, img_height, img_width)
            # ViT-B/14 embedding dim is 768
            in_channels_proj = self.encoder.embed_dim # 768

            # Adaptive pooling to match target spatial size (16, 8)
            self.adaptive_pool = nn.AdaptiveAvgPool2d(self.output_spatial_size)
        else:
            raise ValueError(f"Unknown backbone type: {backbone_type}")

        # Project backbone output channels → token_dim
        self.proj = nn.Conv2d(in_channels_proj, token_dim, kernel_size=1, bias=False)
        self.proj_bn = nn.BatchNorm2d(token_dim)

        # Vector quantizer (EMA), reusing the proven v3 implementation
        self.vq = VectorQuantizer(codebook_size, token_dim, beta=vq_beta)

        # Learnable noise scale (log-parameterized for unconstrained optimization)
        # exp(-3) ≈ 0.05 initial noise — small enough to not disrupt early training
        self.log_noise_scale = nn.Parameter(torch.tensor(-3.0))

    @property
    def noise_scale(self):
        """Current noise standard deviation, clamped to [0.01, 0.5]."""
        return self.log_noise_scale.exp().clamp(0.01, 0.5)

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
            x (Tensor): Input images (B, 3, 256, 128), ImageNet-normalized.

        Returns:
            z_q (Tensor): Clean quantized features (B, D, 16, 8).
            vq_loss (Tensor): Scalar VQ commitment loss.
            indices (Tensor): Codebook indices (B, 128).
            utilisation (float): Codebook utilization fraction.
            token_entropy (Tensor): Per-token entropy (B, H, W).
            z_q_noisy (Tensor): Noise-injected tokens (training) or z_q (eval).
        """
        feat = self.get_projected_features(x)       # (B, token_dim, 16, 8)

        # Vector quantization
        z_q, vq_loss, indices, utilisation = self.vq(feat)

        B, D, H, W = z_q.shape

        # Compute per-token information entropy
        # Measures assignment uncertainty: how spread is this token across codebook?
        with torch.no_grad():
            z_flat = z_q.permute(0, 2, 3, 1).reshape(-1, D)  # (B*H*W, D)
            # L2 distances to codebook entries
            dist = (z_flat.pow(2).sum(1, keepdim=True)
                    - 2 * z_flat @ self.vq.embedding.T
                    + self.vq.embedding.pow(2).sum(1))  # (B*H*W, K)
            # Softmax probabilities (negative distance → higher prob for closer entries)
            probs = F.softmax(-dist, dim=1)  # (B*H*W, K)
            # Shannon entropy
            token_entropy = -(probs * (probs + 1e-10).log()).sum(dim=1)  # (B*H*W,)
            token_entropy = token_entropy.view(B, H, W)  # (B, H, W)

        # Noise injection: training-time regularizer
        if self.training and self.enable_noise:
            noise = torch.randn_like(z_q) * self.noise_scale
            z_q_noisy = z_q + noise
        else:
            z_q_noisy = z_q  # No noise at inference or when enable_noise=False

        return z_q, vq_loss, indices, utilisation, token_entropy, z_q_noisy

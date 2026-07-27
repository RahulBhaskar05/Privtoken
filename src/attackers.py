"""
Heterogeneous Attacker Suite for PrivToken-ReID.

Multiple attacker architectures for evaluating privacy robustness.
The key principle: if privacy claims only hold against one attacker,
they are fragile. Robustness against diverse attackers is the real claim.

Provides:
    A1: UNetDecoder          — Original baseline attacker (in attacker.py)
    A2: ResidualDecoder      — Stronger decoder with residual connections
    A3: TransformerDecoder   — Cross-attention transformer decoder
    A4: FeatureInversionNet  — MLP/conv feature inversion
    A5: IdentityAttacker     — Directly predicts person ID from tokens

Factory:
    get_attacker(name, token_dim=512) → nn.Module
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttackerBase(nn.Module):
    """Base class for all attackers with freeze/unfreeze support."""

    def freeze(self):
        """Freeze all parameters."""
        for p in self.parameters():
            p.requires_grad_(False)

    def unfreeze(self):
        """Unfreeze all parameters."""
        for p in self.parameters():
            p.requires_grad_(True)


# =========================================================================
# A2: Residual Decoder — Stronger than UNet baseline
# =========================================================================

class ResidualBlock(nn.Module):
    """Residual block with two Conv3x3 + BN + LeakyReLU layers."""

    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.act(x + self.block(x))


class ResidualDecoder(AttackerBase):
    """
    Stronger reconstruction attacker with residual connections.

    ~2x parameters vs UNetDecoder for a more thorough privacy test.

    Architecture:
        (B, 512, 16, 8) → 6 ResBlocks → 4× Upsample → (B, 3, 256, 128)
    """

    def __init__(self, in_channels=512):
        super().__init__()

        # Deep residual bottleneck
        self.bottleneck = nn.Sequential(
            ResidualBlock(in_channels),
            ResidualBlock(in_channels),
            ResidualBlock(in_channels),
            ResidualBlock(in_channels),
            ResidualBlock(in_channels),
            ResidualBlock(in_channels),
        )

        # Upsampling path (same structure as UNet but with residual blocks)
        self.up1 = self._up_block(512, 256)
        self.res1 = ResidualBlock(256)
        self.up2 = self._up_block(256, 128)
        self.res2 = ResidualBlock(128)
        self.up3 = self._up_block(128, 64)
        self.res3 = ResidualBlock(64)
        self.up4 = self._up_block(64, 32)
        self.res4 = ResidualBlock(32)

        self.out_conv = nn.Sequential(
            nn.Conv2d(32, 16, 3, 1, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 3, 1),
            nn.Sigmoid(),
        )

    @staticmethod
    def _up_block(in_c, out_c):
        return nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, z_q):
        """
        Args:
            z_q: (B, 512, 16, 8) — quantised token grid.
        Returns:
            recon: (B, 3, 256, 128) — reconstructed image in [0, 1].
        """
        x = self.bottleneck(z_q)
        x = self.res1(self.up1(x))
        x = self.res2(self.up2(x))
        x = self.res3(self.up3(x))
        x = self.res4(self.up4(x))
        return self.out_conv(x)


# =========================================================================
# A3: Transformer Decoder — Cross-attention based reconstruction
# =========================================================================

class TransformerDecoderBlock(nn.Module):
    """Single transformer block with self-attention + FFN."""

    def __init__(self, dim, num_heads=8, ffn_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout,
                                           batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, int(dim * ffn_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * ffn_ratio), dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # Self-attention with residual
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out
        # FFN with residual
        x = x + self.ffn(self.norm2(x))
        return x


class TransformerDecoder(AttackerBase):
    """
    Transformer-based reconstruction attacker.

    Uses self-attention over the token grid to capture global context
    for reconstruction. This tests whether global reasoning can break
    the privacy guarantee.

    Architecture:
        (B, 512, 16, 8) → flatten → 4× TransformerBlock → reshape
        → ConvTranspose upsampling → (B, 3, 256, 128)
    """

    def __init__(self, token_dim=512, num_blocks=4, num_heads=8):
        super().__init__()
        self.token_dim = token_dim

        # Positional embedding for 16×8 = 128 spatial positions
        self.pos_embed = nn.Parameter(torch.randn(1, 128, token_dim) * 0.02)

        # Transformer blocks
        self.blocks = nn.Sequential(*[
            TransformerDecoderBlock(token_dim, num_heads)
            for _ in range(num_blocks)
        ])
        self.norm = nn.LayerNorm(token_dim)

        # Project back to spatial
        self.proj = nn.Linear(token_dim, token_dim)

        # Conv upsampling (same as UNet)
        self.up1 = self._up_block(512, 256)
        self.up2 = self._up_block(256, 128)
        self.up3 = self._up_block(128, 64)
        self.up4 = self._up_block(64, 32)
        self.out_conv = nn.Sequential(
            nn.Conv2d(32, 3, 1),
            nn.Sigmoid(),
        )

    @staticmethod
    def _up_block(in_c, out_c):
        return nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, z_q):
        """
        Args:
            z_q: (B, 512, 16, 8) — quantised token grid.
        Returns:
            recon: (B, 3, 256, 128) — reconstructed image in [0, 1].
        """
        B, D, H, W = z_q.shape

        # Flatten spatial: (B, D, H, W) → (B, H*W, D)
        x = z_q.flatten(2).permute(0, 2, 1)  # (B, 128, 512)

        # Add positional embedding
        x = x + self.pos_embed

        # Transformer blocks
        x = self.blocks(x)
        x = self.norm(x)
        x = self.proj(x)

        # Reshape back to spatial: (B, 128, 512) → (B, 512, 16, 8)
        x = x.permute(0, 2, 1).reshape(B, D, H, W)

        # Upsample
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        return self.out_conv(x)


# =========================================================================
# A4: Feature Inversion Network — Perceptual inversion
# =========================================================================

class FeatureInversionNet(AttackerBase):
    """
    Lightweight feature inversion attacker.

    Uses a simpler architecture (fewer params) but optimized with
    perceptual loss, testing a different attack strategy.

    Architecture:
        (B, 512, 16, 8) → Conv blocks + bilinear upsampling → (B, 3, 256, 128)
    """

    def __init__(self, in_channels=512):
        super().__init__()

        self.decoder = nn.Sequential(
            # 16×8 → 16×8
            nn.Conv2d(in_channels, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # Upsample: 16×8 → 32×16
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # 32×16 → 64×32
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # 64×32 → 128×64
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # 128×64 → 256×128
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(32, 16, 3, 1, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 3, 1),
            nn.Sigmoid(),
        )

    def forward(self, z_q):
        """
        Args:
            z_q: (B, 512, 16, 8) — quantised token grid.
        Returns:
            recon: (B, 3, 256, 128) — reconstructed image in [0, 1].
        """
        return self.decoder(z_q)


# =========================================================================
# A5: Identity Attacker — Predicts person ID without reconstruction
# =========================================================================

class IdentityAttacker(AttackerBase):
    """
    Directly predicts person identity from token representations.

    This is the most direct privacy test: can an attacker determine WHO
    a person is just from the discrete token grid, without ever
    reconstructing the image?

    Architecture:
        Token grid (B, D, H, W) → GAP → MLP → person ID logits
    """

    def __init__(self, token_dim=512, num_classes=751, hidden_dim=512):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(token_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, z_q):
        """
        Args:
            z_q: (B, D, H, W) — quantised token grid.
        Returns:
            logits: (B, num_classes) — person ID prediction logits.
        """
        pooled = self.gap(z_q).flatten(1)  # (B, D)
        return self.classifier(pooled)


# =========================================================================
# Factory
# =========================================================================

def get_attacker(name, token_dim=512, num_classes=751):
    """
    Create an attacker by name.

    Args:
        name (str): One of 'unet', 'residual', 'transformer',
                     'feature_inversion', 'identity'.
        token_dim (int): Token embedding dimension.
        num_classes (int): Number of identity classes (for IdentityAttacker).

    Returns:
        nn.Module: Attacker model.
    """
    name = name.lower().replace('-', '_')

    if name == 'unet':
        from src.attacker import UNetDecoder
        return UNetDecoder()
    elif name == 'residual':
        return ResidualDecoder(in_channels=token_dim)
    elif name == 'transformer':
        return TransformerDecoder(token_dim=token_dim)
    elif name == 'feature_inversion':
        return FeatureInversionNet(in_channels=token_dim)
    elif name == 'identity':
        return IdentityAttacker(token_dim=token_dim, num_classes=num_classes)
    else:
        raise ValueError(
            f"Unknown attacker: '{name}'. Choose from: "
            f"unet, residual, transformer, feature_inversion, identity"
        )


def get_all_attacker_names():
    """Return list of all available attacker names."""
    return ['unet', 'residual', 'transformer', 'feature_inversion', 'identity']

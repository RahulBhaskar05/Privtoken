"""
UNet-style adversarial decoder (attacker).

Attempts to reconstruct the original image from the quantized token grid.
Used in the adversarial privacy training loop: the attacker tries to MINIMIZE
reconstruction loss, while the tokenizer tries to MAXIMIZE it.

Architecture:
    Input: (B, 512, 16, 8) quantized token grid (after last-stride fix)
    → Bottleneck (2× Conv3x3)
    → 4× upsample blocks (ConvTranspose2d)
    → Output conv → (B, 3, 256, 128) reconstructed image in [0,1]
"""

import torch
import torch.nn as nn


class UNetDecoder(nn.Module):
    """Reconstruction attacker: (B, 512, 16, 8) → (B, 3, 256, 128).

    Upsampling path:
        (B, 512, 16, 8)  →  up1  →  (B, 256, 32, 16)
        (B, 256, 32, 16) →  up2  →  (B, 128, 64, 32)
        (B, 128, 64, 32) →  up3  →  (B, 64, 128, 64)
        (B, 64, 128, 64) →  up4  →  (B, 32, 256, 128)
        (B, 32, 256, 128) → out  →  (B, 3, 256, 128)

    One fewer upsampling step than the 8×4 version because the input
    is already 16×8 after the last-stride fix.
    """
    def __init__(self):
        super().__init__()
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.up1 = self._up_block(512, 256)   # → (B, 256, 32, 16)
        self.up2 = self._up_block(256, 128)   # → (B, 128, 64, 32)
        self.up3 = self._up_block(128, 64)    # → (B, 64, 128, 64)
        self.up4 = self._up_block(64, 32)     # → (B, 32, 256, 128)
        self.out_conv = nn.Sequential(
            nn.Conv2d(32, 3, kernel_size=1),
            nn.Sigmoid()
        )

    @staticmethod
    def _up_block(in_c, out_c):
        return nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, z_q):
        """
        Args:
            z_q: (B, 512, 16, 8) — quantised token grid
        Returns:
            recon: (B, 3, 256, 128) — reconstructed image in [0, 1]
        """
        x = self.bottleneck(z_q)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        return self.out_conv(x)

    def freeze(self):
        """Freeze all parameters — call before tokenizer outer step."""
        for p in self.parameters():
            p.requires_grad_(False)

    def unfreeze(self):
        """Unfreeze all parameters — call before attacker inner steps."""
        for p in self.parameters():
            p.requires_grad_(True)

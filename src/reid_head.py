"""
BNNeck Re-ID head.

Implements the Bag of Tricks (BoT) BNNeck architecture for person
re-identification. Features before BN (ft) are used for triplet loss,
while features after BN (fn) are used for classification and retrieval.

Reference: Luo et al., "Bag of Tricks and a Strong Baseline for Deep
Person Re-identification", CVPR Workshop 2019.
"""

import torch
import torch.nn as nn


class BNNeck(nn.Module):
    """
    BNNeck Re-ID head with separate feature outputs for triplet and CE losses.

    Architecture:
        Input z_q (B, token_dim, H, W)
        → Global Average Pooling → (B, token_dim)       = ft (for triplet)
        → BatchNorm1d → (B, token_dim)                   = fn (for CE + retrieval)
        → Linear classifier → (B, num_classes)            = logits (training only)

    Args:
        token_dim (int): Input feature dimension (e.g. 512).
        num_classes (int): Number of training identities (e.g. 751).

    Inputs:
        z_q (Tensor): Quantized features from tokenizer, shape (B, token_dim, H, W).

    Returns:
        fn (Tensor): Normalized features (after BN), shape (B, token_dim). For CE loss and retrieval.
        ft (Tensor): Raw features (before BN), shape (B, token_dim). For triplet loss.
        logits (Tensor): Classification logits, shape (B, num_classes). Training only.
    """

    def __init__(self, token_dim, num_classes):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # BNNeck: BatchNorm before classifier
        self.bottleneck = nn.BatchNorm1d(token_dim)
        self.bottleneck.bias.requires_grad_(False)  # no bias, standard practice

        # Classifier
        self.classifier = nn.Linear(token_dim, num_classes, bias=False)
        nn.init.normal_(self.classifier.weight, std=0.001)

    def forward(self, z_q):
        """
        Args:
            z_q (Tensor): Quantized token features, shape (B, token_dim, H, W).

        Returns:
            fn (Tensor): BN-normalized features for CE loss & retrieval, shape (B, token_dim).
            ft (Tensor): Raw pooled features for triplet loss, shape (B, token_dim).
            logits (Tensor): Classification logits, shape (B, num_classes).
        """
        # Global average pooling
        ft = self.gap(z_q).flatten(1)  # (B, token_dim)

        # BatchNorm neck
        fn = self.bottleneck(ft)  # (B, token_dim)

        # Classification
        logits = self.classifier(fn)  # (B, num_classes)

        return fn, ft, logits

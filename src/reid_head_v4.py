"""
V4 Multi-Granularity ReID Head.

Adopted from PCB (Sun et al., "Beyond Part Models: Person Retrieval with
Refined Part Pooling", ECCV 2018). This is standard practice in person ReID
and is NOT claimed as a novel contribution. It is engineering infrastructure
that closes the utility gap, allowing the novel privacy mechanisms (GRL, noise
regularization) to operate at competitive Rank-1/mAP levels.

Architecture:
    Input: z_q (B, token_dim, 16, 8) — quantized token grid

    Global branch:
        GAP(z_q) → (B, D) → BN → classifier → logits

    Part branches (4 horizontal stripes):
        z_q[:, :, 0:4, :]  → GAP → BN → classifier  (head region)
        z_q[:, :, 4:8, :]  → GAP → BN → classifier  (upper body)
        z_q[:, :, 8:12, :] → GAP → BN → classifier  (lower body)
        z_q[:, :, 12:16,:] → GAP → BN → classifier  (legs/feet)

    Training: 5 separate CE + Triplet losses (averaged)
    Inference: concatenate 5 BN-normalized features → (B, 5*D)
"""

import torch
import torch.nn as nn


class MultiGranularityHead(nn.Module):
    """Multi-granularity ReID head with global + part-level features.

    Produces separate classification heads per horizontal part, plus one
    global head. At test time, all features are concatenated into a
    single high-dimensional descriptor for maximum discriminability.

    Args:
        token_dim (int): Input feature dimension (default: 512).
        num_classes (int): Number of training identities (default: 751).
        num_parts (int): Number of horizontal stripes (default: 4).
    """

    def __init__(self, token_dim=512, num_classes=751, num_parts=4):
        super().__init__()
        self.token_dim = token_dim
        self.num_parts = num_parts

        # ---- Global branch ----
        self.global_gap = nn.AdaptiveAvgPool2d((1, 1))
        self.global_bn = nn.BatchNorm1d(token_dim)
        self.global_bn.bias.requires_grad_(False)
        self.global_classifier = nn.Linear(token_dim, num_classes, bias=False)
        nn.init.normal_(self.global_classifier.weight, std=0.001)

        # ---- Part branches ----
        self.part_pools = nn.ModuleList([
            nn.AdaptiveAvgPool2d((1, 1)) for _ in range(num_parts)
        ])
        self.part_bns = nn.ModuleList([
            nn.BatchNorm1d(token_dim) for _ in range(num_parts)
        ])
        self.part_classifiers = nn.ModuleList([
            nn.Linear(token_dim, num_classes, bias=False) for _ in range(num_parts)
        ])

        # Initialize part branches
        for bn in self.part_bns:
            bn.bias.requires_grad_(False)
        for cls in self.part_classifiers:
            nn.init.normal_(cls.weight, std=0.001)

    def forward(self, z_q):
        """Extract global + part-level features with classifiers.

        Args:
            z_q (Tensor): Quantized token grid (B, D, H, W).
                Typically (B, 512, 16, 8) after VQ.

        Returns:
            dict with keys:
                'global_ft':  (B, D)  — pre-BN global features (for triplet loss)
                'global_fn':  (B, D)  — post-BN global features (for retrieval)
                'global_logits': (B, C) — global classification logits
                'part_fts':   list of (B, D) — pre-BN part features
                'part_fns':   list of (B, D) — post-BN part features
                'part_logits': list of (B, C) — per-part logits
        """
        B, D, H, W = z_q.shape
        part_h = H // self.num_parts  # 16 // 4 = 4

        # Global
        global_ft = self.global_gap(z_q).flatten(1)          # (B, D)
        global_fn = self.global_bn(global_ft)                 # (B, D)
        global_logits = self.global_classifier(global_fn)     # (B, C)

        # Parts
        part_fts = []
        part_fns = []
        part_logits_list = []

        for i in range(self.num_parts):
            start_row = i * part_h
            end_row = start_row + part_h
            part_slice = z_q[:, :, start_row:end_row, :]     # (B, D, part_h, W)

            ft = self.part_pools[i](part_slice).flatten(1)    # (B, D)
            fn = self.part_bns[i](ft)                         # (B, D)
            logits = self.part_classifiers[i](fn)             # (B, C)

            part_fts.append(ft)
            part_fns.append(fn)
            part_logits_list.append(logits)

        return {
            'global_ft': global_ft,
            'global_fn': global_fn,
            'global_logits': global_logits,
            'part_fts': part_fts,
            'part_fns': part_fns,
            'part_logits': part_logits_list,
        }

    def get_eval_features(self, z_q):
        """Concatenated feature vector for evaluation/retrieval.

        Combines global + all part BN-normalized features into a single
        high-dimensional descriptor. Used at test time for CMC/mAP computation.

        Args:
            z_q (Tensor): Quantized token grid (B, D, H, W).

        Returns:
            features (Tensor): (B, (1 + num_parts) * D) concatenated features.
        """
        out = self.forward(z_q)
        all_features = [out['global_fn']] + out['part_fns']
        return torch.cat(all_features, dim=1)  # (B, 5*512 = 2560)

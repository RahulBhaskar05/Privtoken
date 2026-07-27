# Progressive Ablation Study — Market-1501 (Entropy Confusion Pipeline)

> Each stage is strictly additive: Stage N = Stage N-1 + one new component.
> All stages use `codebook_size: 4096`. Stage 0 posthoc fields are N/A.
> † = stage was re-run due to a bug fix; previous numbers must not be cited.

| Stage | Label | Status | Rank-1↑ | mAP↑ | mINP↑ | PSNR↓ | SSIM↓ | LPIPS↑ | PU↑ | PH-PSNR↓ | PH-SSIM↓ | PH-ID-Top1↓ | Collapse | noise_σ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | Stage 0 — Bare Backbone | ✓ | 90.23% | 72.75% | N/A% | N/A dB | N/A | N/A | N/A | N/A dB | N/A | N/A% | 0 | N/A |
| 1 | Stage 1 — VQ-VAE (4096) | ✓ | 82.57% | 63.38% | 25.62% | 15.49 dB | 0.2436 | 0.3731 | 79.0 | 15.79 dB | 0.2733 | 96.62% | 0 | 0.0584 |
| 2 | Stage 2 — PCB Multi-Granularity Head | ✓ | 81.00% | 60.79% | 24.40% | 14.39 dB | 0.1749 | 0.4040 | 81.7 | 14.72 dB | 0.2237 | 94.65% | 0 | 0.0584 |
| 3 | Stage 3 — Learnable Noise Injection | ✓ | 80.82% | 61.10% | 24.46% | 14.39 dB | 0.1752 | 0.4002 | 81.6 | 14.70 dB | 0.2186 | 94.97% | 0 | 0.0584 |
| 4a | Stage 4a — Co-Trained Attacker + Region-Weighted Loss | ✓ | 80.64% | 60.87% | 24.51% | 14.38 dB | 0.1905 | 0.4037 | 80.8 | 14.63 dB | 0.2175 | 94.71% | 0 | 0.0584 |
| 4b | Stage 4b — Entropy-Guided Privacy Weighting (active attacker) | ✓ | 80.34% | 60.35% | 23.74% | 14.31 dB | 0.1830 | 0.4060 | 81.0 | 14.57 dB | 0.2202 | 94.88% | 0 | 0.0584 |
| 5 | Stage 5 — Full No-GRL Model (Identity Confusion) | ✓† re-run | 80.43% | 60.10% | 23.60% | 14.51 dB | 0.1998 | 0.3997 | 80.2 | 14.82 dB | 0.2222 | 93.96% | 0 | 0.0584 |

## Per-Stage Components Added

- **Stage 0**: ResNet-50 encoder
- **Stage 0**: BNNeck global head
- **Stage 0**: CE+Triplet losses
- **Stage 1**: VQ-VAE bottleneck (codebook_size=4096)
- **Stage 2**: PCB 4-stripe multi-granularity head (num_parts=4)
- **Stage 3**: Learnable noise injection (lambda_noise=0.01)
- **Stage 4a**: Co-trained U-Net attacker with region-weighted reconstruction loss (lambda_priv_start=0.005, attacker_inner_steps=3, use_region_privacy=true)
- **Stage 4b**: Entropy-guided per-token privacy weighting replaces region weighting (use_entropy_privacy=true, use_region_privacy=false; attacker unchanged)
- **Stage 5**: Minimax identity confusion via entropy maximization (lambda_id_start=0.01, lambda_id_max=0.5, id_adversary_inner_steps=2)


## Re-Run Provenance (Supplementary Traceability)

- **Stage 5 (Stage 5 — Full No-GRL Model (Identity Confusion))**: Re-run: Stage 5 sits atop corrected 4b. Previous numbers (rank1=80.43%, mAP=60.10%) were produced on the invalid 4a chain.

## Discarded Stages

- **Old Stage 4a (`entropy_privacy_passive`)**: Discarded — `use_entropy_privacy: true` with `lambda_priv = 0.0` caused `loss_priv = -0.0 * loss_recon = 0`, so the entropy weighting had zero gradient effect. Stage 4a and Stage 3 results were identical to 6 decimal places across all metrics. Replaced by `cotrained_attacker_region` (new 4a) and `entropy_privacy_active` (new 4b).
- **Old Stage 4b (`cotrained_attacker`)**: Discarded — built on the invalid Stage 4a. Replaced by `entropy_privacy_active` (new 4b).
- **Old Stage 5 (`full_no_grl`, rank1=80.43%, mAP=60.10%)**: Superseded — re-run on the corrected 4a→4b chain. Do not cite old numbers.


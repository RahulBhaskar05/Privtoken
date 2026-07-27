Here is the complete guide to running the progressive ablation study and obtaining clean, publication-ready results.

---

## 1. What Was Fixed & Why The Old Results Were Invalid

Before running, here is why the old table was flawed and what has been fixed:

1. **Bug #1 — Attacker Decoupling:** In the old code, the reconstruction attacker was only trained after epoch 30 (`warmup_epochs`). In the new code, **the attacker trains continuously from Epoch 1 in every stage**.
2. **Noise Gating:** Noise injection is now explicitly gated by `enable_noise = (lambda_noise > 0)`. In Stages 1 and 2, noise is **completely OFF** in both training and evaluation, isolating noise injection as a Stage 3 component.
3. **6 Strictly Additive Stages:** Merged old Stage 4a and 4b into a single clean **Stage 4** (Entropy-Guided Reconstruction Privacy).
4. **Automated Validation Suite:** Built-in checks verify attacker convergence, SSIM monotonicity ($\le +0.01$ tolerance), and dual-attacker alignment (co-trained vs. independent post-hoc) after training completes.

---

## 2. The 6-Stage Additive Pipeline Architecture

| Stage | Name | Key Additive Component | Privacy Pressure | Expected Behavior |
|:---:|:--- |:--- |:---:|:--- |
| **0** | `bare_backbone` | ResNet-50 + BNNeck global head | None | **Utility Baseline** (Rank-1 ~88%, mAP ~73%) |
| **1** | `vq_4096` | + VQ-VAE Bottleneck (`codebook_size=4096`, `num_parts=1`) | None | **Undefended VQ Baseline** (SSIM ~0.24–0.27, leaky privacy) |
| **2** | `pcb` | + PCB 4-stripe Multi-Granularity Head (`num_parts=4`) | None | **ReID Utility Head** (Codebook drift from 5 ReID loss heads) |
| **3** | `noise` | + Learnable Token Noise Injection (`lambda_noise=0.01`) | Noise ON | **Partial Privacy Drop** (First privacy defense layer) |
| **4** | `entropy_privacy` | + Entropy-Guided Privacy Loss (`lambda_priv_start=0.005`) | Noise + Recon Loss | **Major Reconstruction Privacy Drop** |
| **5** | `full_no_grl` | + Minimax Identity Confusion (`lambda_id_start=0.01`) | Full Defense | **Full Model Target:** Rank-1 ~80.43%, mAP ~60.10% |

---

## 3. How To Run (Step-by-Step)

### Step 1: Pre-Run Validation Check (Zero Cost)
Confirm all 6 configs match the Single Source of Truth:
```bash
python run_progressive_ablation_market.py --dry-run
```
*(Should output `[OK]` for all 6 stages).*

---

### Step 2: Spot-Check Stage 1 First (~25 mins)
Run Stage 1 alone to verify that the attacker trains continuously from epoch 1:
```bash
python run_progressive_ablation_market.py --stages 1
```

**Verification checks after Stage 1 finishes:**
1. Open `logs_ablation_market/stage1_vq_4096/training_log.csv`:
   - Confirm `attacker_loss` is non-zero and updating starting at **Epoch 1** (not epoch 31).
2. Open `outputs_ablation_market/stage1_vq_4096_results.json`:
   - Confirm `"attacker_converged": true`.

---

### Step 3: Run Remaining Stages (~3 GPU Hours)
Once Stage 1 passes spot-check, launch the remaining stages:
```bash
python run_progressive_ablation_market.py --stages 0 2 3 4 5
```
*(Or run `python run_progressive_ablation_market.py` to run all 6 in sequence).*

---

## 4. Where To Find Your Final Results

Once training finishes, all outputs are consolidated in `outputs_ablation_market/`:

1. **`outputs_ablation_market/ablation_table.md`**: The publication-ready markdown table containing:
   - ReID Utility metrics: Rank-1, Rank-5, Rank-10, mAP, mINP.
   - Visual Privacy metrics (Co-Trained Attacker): PSNR↓, SSIM↓, LPIPS↑, PU-Score.
   - Visual Privacy metrics (Independent Post-Hoc Suite): PH-PSNR↓, PH-SSIM↓, PH-ID-Top1↓.
   - Collapse count & noise scale.
2. **`outputs_ablation_market/consolidated_results.json`**: Complete raw JSON dictionary across all 6 stages.

---

## 5. Summary of Helpful Commands

| Command | Purpose |
|:--- |:--- |
| `python run_progressive_ablation_market.py --dry-run` | Validate configs without running training. |
| `python run_progressive_ablation_market.py --stages 1` | Run Stage 1 spot-check (~25 min). |
| `python run_progressive_ablation_market.py` | Run the complete 6-stage study. |
| `python run_progressive_ablation_market.py --skip-training` | Re-aggregate existing result JSONs into a new table. |
| `python run_progressive_ablation_market.py --device cuda:0` | Specify a target GPU device. |



You can run all 6 stages in a single command:

bash
`python run_progressive_ablation_market.py`


"""
Progressive Ablation Study Orchestrator — Market-1501 (Entropy Confusion Pipeline).

Runs 6 ablation stages in order, each strictly additive (each stage = previous
stage + one new architectural component). Uses train_baseline.py for Stage 0
and train_entropy_confusion.py for Stages 1–5.

For each stage:
  1. Train on Market-1501, saving checkpoints to checkpoints_ablation_market/stage{N}_{name}/
  2. Run the full post-hoc attacker suite (run_posthoc_attacker_suite) against the
     frozen final checkpoint — not just co-trained attacker numbers.
  3. Save results to outputs_ablation_market/stage{N}_{name}_results.json using the
     schema: rank1, rank5, rank10, mAP, mINP, psnr, ssim, lpips, pu_score,
     posthoc_recon_psnr, posthoc_recon_ssim, posthoc_id_top1, collapse_count,
     noise_scale, attacker_converged.
  4. Check collapse_count == 0 and attacker_converged == True before treating the run as valid.

After all stages, outputs a consolidated JSON + markdown table to
outputs_ablation_market/.

Usage:
    python run_progressive_ablation_market.py                     # full run
    python run_progressive_ablation_market.py --dry-run           # config check only
    python run_progressive_ablation_market.py --stages 0 1 2      # subset of stages
    python run_progressive_ablation_market.py --device cuda
    python run_progressive_ablation_market.py --skip-training     # only collect existing results
"""

import argparse
import json
import os
import subprocess
import sys
import time

import yaml


# ===========================================================================
# Stage Definitions (Single Source of Truth)
# ===========================================================================

# NOTE on "attacker_always_trained":
#   This is a documentation-only annotation. It is NOT loaded from any config key
#   and is NOT cross-validated by the dry-run SSOT assertion logic — that logic
#   only checks noise_active, lambda_priv_active, lambda_id_active, use_entropy_privacy,
#   all of which correspond to actual YAML values. attacker_always_trained=False
#   on Stage 0 simply documents that train_baseline.py has no attacker; for Stages 1-5
#   it documents the Fix #1 intent that the attacker trains from epoch 1 onward.

STAGES = [
    {
        "id":     "0",
        "name":   "bare_backbone",
        "label":  "Stage 0 — Bare Backbone",
        "script": "train_baseline.py",
        "config": "configs/ablation_market/stage0_bare_backbone.yaml",
        "checkpoint_dir": "./checkpoints_ablation_market/stage0_bare_backbone",
        "output_dir":     "./outputs_ablation_market/stage0_bare_backbone",
        "result_file":    "./outputs_ablation_market/stage0_bare_backbone_results.json",
        "eval_json_name": "eval_results.json",
        "posthoc_na": True,
        "components_added": ["ResNet-50 encoder", "BNNeck global head", "CE+Triplet losses"],
        "attacker_always_trained": False,
        "noise_active": False,
        "lambda_priv_active": False,
        "lambda_id_active": False,
        "use_entropy_privacy": False,
    },
    {
        "id":     "1",
        "name":   "vq_4096",
        "label":  "Stage 1 — VQ-VAE (4096)",
        "script": "train_entropy_confusion.py",
        "config": "configs/ablation_market/stage1_vq_4096.yaml",
        "checkpoint_dir": "./checkpoints_ablation_market/stage1_vq_4096",
        "output_dir":     "./outputs_ablation_market/stage1_vq_4096",
        "result_file":    "./outputs_ablation_market/stage1_vq_4096_results.json",
        "eval_json_name": "eval_results_v4.json",
        "posthoc_na": False,
        "components_added": ["VQ-VAE bottleneck (codebook_size=4096, num_parts=1)"],
        "attacker_always_trained": True,
        "noise_active": False,
        "lambda_priv_active": False,
        "lambda_id_active": False,
        "use_entropy_privacy": False,
    },
    {
        "id":     "2",
        "name":   "pcb",
        "label":  "Stage 2 — PCB Multi-Granularity Head",
        "script": "train_entropy_confusion.py",
        "config": "configs/ablation_market/stage2_pcb.yaml",
        "checkpoint_dir": "./checkpoints_ablation_market/stage2_pcb",
        "output_dir":     "./outputs_ablation_market/stage2_pcb",
        "result_file":    "./outputs_ablation_market/stage2_pcb_results.json",
        "eval_json_name": "eval_results_v4.json",
        "posthoc_na": False,
        "components_added": ["PCB 4-stripe multi-granularity head (num_parts=4)"],
        "attacker_always_trained": True,
        "noise_active": False,
        "lambda_priv_active": False,
        "lambda_id_active": False,
        "use_entropy_privacy": False,
    },
    {
        "id":     "3",
        "name":   "noise",
        "label":  "Stage 3 — Learnable Noise Injection",
        "script": "train_entropy_confusion.py",
        "config": "configs/ablation_market/stage3_noise.yaml",
        "checkpoint_dir": "./checkpoints_ablation_market/stage3_noise",
        "output_dir":     "./outputs_ablation_market/stage3_noise",
        "result_file":    "./outputs_ablation_market/stage3_noise_results.json",
        "eval_json_name": "eval_results_v4.json",
        "posthoc_na": False,
        "components_added": ["Learnable noise injection (lambda_noise=0.01)"],
        "attacker_always_trained": True,
        "noise_active": True,
        "lambda_priv_active": False,
        "lambda_id_active": False,
        "use_entropy_privacy": False,
    },
    {
        "id":     "4",
        "name":   "entropy_privacy",
        "label":  "Stage 4 — Entropy-Guided Reconstruction Privacy",
        "script": "train_entropy_confusion.py",
        "config": "configs/ablation_market/stage4_entropy_privacy.yaml",
        "checkpoint_dir": "./checkpoints_ablation_market/stage4_entropy_privacy",
        "output_dir":     "./outputs_ablation_market/stage4_entropy_privacy",
        "result_file":    "./outputs_ablation_market/stage4_entropy_privacy_results.json",
        "eval_json_name": "eval_results_v4.json",
        "posthoc_na": False,
        "components_added": [
            "Entropy-guided reconstruction privacy loss "
            "(lambda_priv_start=0.005, lambda_priv_max=0.05, use_entropy_privacy=true)"
        ],
        "attacker_always_trained": True,
        "noise_active": True,
        "lambda_priv_active": True,
        "lambda_id_active": False,
        "use_entropy_privacy": True,
    },
    {
        "id":     "5",
        "name":   "full_no_grl",
        "label":  "Stage 5 — Full No-GRL Model (Identity Confusion)",
        "script": "train_entropy_confusion.py",
        "config": "configs/ablation_market/stage5_full_no_grl.yaml",
        "checkpoint_dir": "./checkpoints_ablation_market/stage5_full_no_grl",
        "output_dir":     "./outputs_ablation_market/stage5_full_no_grl",
        "result_file":    "./outputs_ablation_market/stage5_full_no_grl_results.json",
        "eval_json_name": "eval_results_v4.json",
        "posthoc_na": False,
        "components_added": [
            "Minimax identity confusion via entropy maximization "
            "(lambda_id_start=0.01, lambda_id_max=0.5, id_adversary_inner_steps=2)"
        ],
        "attacker_always_trained": True,
        "noise_active": True,
        "lambda_priv_active": True,
        "lambda_id_active": True,
        "use_entropy_privacy": True,
    },
]


# ===========================================================================
# Helpers
# ===========================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Progressive Ablation Study Orchestrator — Market-1501")
    parser.add_argument(
        "--stages", nargs="+", default=None,
        metavar="STAGE_ID",
        help="Run only these stage IDs (e.g. --stages 0 1 2 4a). "
             "Default: run all stages in order.",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device override passed to training scripts (cuda / cpu).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate config files and print the run plan without training.",
    )
    parser.add_argument(
        "--skip-training", action="store_true",
        help="Skip training; only collect existing result JSONs and build the table.",
    )
    return parser.parse_args()


def banner(text: str, width: int = 72):
    print("\n" + "=" * width)
    print(f"  {text}")
    print("=" * width)


def run_training(stage: dict, device: str | None) -> bool:
    """Run the training script for a single stage. Returns True on success."""
    cmd = [sys.executable, stage["script"], "--config", stage["config"]]
    if device:
        cmd.extend(["--device", device])

    banner(f"TRAINING  {stage['label']}")
    print(f"  Script : {stage['script']}")
    print(f"  Config : {stage['config']}")
    print(f"  Cmd    : {' '.join(cmd)}\n")

    t0 = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - t0

    ok = result.returncode == 0
    status = "SUCCESS" if ok else f"FAILED (exit code {result.returncode})"
    print(f"\n  [{status}] elapsed: {elapsed / 60:.1f} min")
    return ok


def collect_results(stage: dict) -> dict | None:
    """Read the eval JSON written by the training script for this stage."""
    json_path = os.path.join(stage["output_dir"], stage["eval_json_name"])
    if not os.path.isfile(json_path):
        print(f"  [WARNING] Results JSON not found: {json_path}")
        return None
    with open(json_path) as f:
        return json.load(f)


def build_stage_result(stage: dict, raw: dict | None) -> dict:
    """Normalise raw eval JSON into the required result schema for this stage."""

    # Stage 0 (train_baseline.py) has a different output schema —
    # it only writes rank1/rank5/rank10/mAP. Fill everything else as null.
    NULL = None

    if raw is None:
        return {
            "stage_id":              stage["id"],
            "stage_name":            stage["name"],
            "stage_label":           stage["label"],
            "components_added":      stage["components_added"],
            "rank1":                 NULL,
            "rank5":                 NULL,
            "rank10":                NULL,
            "mAP":                   NULL,
            "mINP":                  NULL,
            "psnr":                  NULL,
            "ssim":                  NULL,
            "lpips":                 NULL,
            "pu_score":              NULL,
            "posthoc_recon_psnr":    NULL,
            "posthoc_recon_ssim":    NULL,
            "posthoc_id_top1":       NULL,
            "collapse_count":        NULL,
            "noise_scale":           NULL,
            "valid":                 False,
            "collapse_flagged":      False,
            "error":                 "results JSON missing",
        }

    collapse = raw.get("collapse_count", 0) or 0
    collapse_flagged = (collapse > 0)

    # For Stage 0 there is no VQ or privacy machinery.
    posthoc_na = stage.get("posthoc_na", False)

    result = {
        "stage_id":           stage["id"],
        "stage_name":         stage["name"],
        "stage_label":        stage["label"],
        "components_added":   stage["components_added"],
        # ReID utility
        "rank1":              raw.get("rank1"),
        "rank5":              raw.get("rank5"),
        "rank10":             raw.get("rank10"),
        "mAP":                raw.get("mAP"),
        "mINP":               raw.get("mINP", NULL),
        # Visual privacy (co-trained attacker)
        "psnr":               raw.get("psnr", NULL),
        "ssim":               raw.get("ssim", NULL),
        "lpips":              raw.get("lpips", NULL),
        "pu_score":           raw.get("pu_score", NULL),
        # Post-hoc independent attacker suite
        "posthoc_recon_psnr": NULL if posthoc_na else raw.get("posthoc_recon_psnr"),
        "posthoc_recon_ssim": NULL if posthoc_na else raw.get("posthoc_recon_ssim"),
        "posthoc_id_top1":    NULL if posthoc_na else raw.get("posthoc_id_top1"),
        # Training meta
        "collapse_count":     collapse,
        "noise_scale":        raw.get("noise_scale", NULL),
        # Validity
        "valid":              not collapse_flagged,
        "collapse_flagged":   collapse_flagged,
        # Traceability — present only when stage was re-run due to a bug fix
        "rerun_reason":       stage.get("rerun_reason"),
    }

    return result


def check_and_flag_collapse(result: dict) -> None:
    """Print a prominent warning if collapse_count > 0."""
    if result.get("collapse_flagged"):
        print(f"\n  {'!' * 60}")
        print(f"  [COLLAPSE FLAGGED]  Stage {result['stage_id']} — {result['stage_label']}")
        print(f"  collapse_count = {result['collapse_count']}  →  valid = False")
        print(f"  Results are recorded but marked invalid in the consolidated table.")
        print(f"  {'!' * 60}\n")


def fmt(val, fmt_str=".2f", suffix="") -> str:
    """Format a float or return 'N/A' for None."""
    if val is None:
        return "N/A"
    try:
        return f"{val:{fmt_str}}{suffix}"
    except (ValueError, TypeError):
        return str(val)


def build_markdown_table(results: list[dict]) -> str:
    header = (
        "| Stage | Label | Status | "
        "Rank-1↑ | mAP↑ | mINP↑ | "
        "PSNR↓ | SSIM↓ | LPIPS↑ | PU↑ | "
        "PH-PSNR↓ | PH-SSIM↓ | PH-ID-Top1↓ | "
        "Collapse | noise_σ |"
    )
    sep = (
        "| :--- | :--- | :---: | "
        ":---: | :---: | :---: | "
        ":---: | :---: | :---: | :---: | "
        ":---: | :---: | :---: | "
        ":---: | :---: |"
    )
    rows = [header, sep]

    for r in results:
        if r.get("rerun_reason"):
            valid_str = "✓† re-run" if r.get("valid") else "⚠† COLLAPSED"
        else:
            valid_str = "✓" if r.get("valid") else "⚠ COLLAPSED"
        row = (
            f"| {r['stage_id']} "
            f"| {r['stage_label']} "
            f"| {valid_str} "
            f"| {fmt(r.get('rank1'))}% "
            f"| {fmt(r.get('mAP'))}% "
            f"| {fmt(r.get('mINP'))}% "
            f"| {fmt(r.get('psnr'))} dB "
            f"| {fmt(r.get('ssim'), '.4f')} "
            f"| {fmt(r.get('lpips'), '.4f')} "
            f"| {fmt(r.get('pu_score'), '.1f')} "
            f"| {fmt(r.get('posthoc_recon_psnr'))} dB "
            f"| {fmt(r.get('posthoc_recon_ssim'), '.4f')} "
            f"| {fmt(r.get('posthoc_id_top1'))}% "
            f"| {r.get('collapse_count', 'N/A')} "
            f"| {fmt(r.get('noise_scale'), '.4f')} |"
        )
        rows.append(row)

    return "\n".join(rows)


def print_summary_table(results: list[dict]) -> None:
    banner("PROGRESSIVE ABLATION STUDY — CONSOLIDATED RESULTS (Market-1501)")

    print(f"\n{'Stage':<6} {'Name':<35} {'Valid':<8} "
          f"{'Rank-1':>8} {'mAP':>8} {'mINP':>8} "
          f"{'PSNR':>8} {'SSIM':>8} {'PH-PSNR':>9} {'Collapse':>9}")
    print("-" * 110)

    for r in results:
        valid_str = "OK" if r.get("valid") else "COLLAPSE"
        print(
            f"{r['stage_id']:<6} {r['stage_label'][:34]:<35} {valid_str:<8} "
            f"{fmt(r.get('rank1')):>7}% {fmt(r.get('mAP')):>7}% "
            f"{fmt(r.get('mINP')):>7}% "
            f"{fmt(r.get('psnr')):>6} dB {fmt(r.get('ssim'), '.4f'):>8} "
            f"{fmt(r.get('posthoc_recon_psnr')):>7} dB "
            f"{str(r.get('collapse_count', 'N/A')):>9}"
        )

    print("\n  Note: PSNR ↓ is better for privacy, Rank-1/mAP ↑ is better for utility.")
    print("  PH-* = post-hoc independent attacker (stronger / no co-training advantage).")
    print("  Stage 0 posthoc fields are N/A — no VQ bottleneck, no meaningful privacy metric.")


def validate_ablation_sequence(results: list[dict]) -> list[str]:
    """Execute post-training validation suite checks (a-g) across the stage sequence."""
    warnings = []
    res_map = {str(r["stage_id"]): r for r in results}

    banner("POST-TRAINING ABLATION VALIDATION SUITE")

    # Check 1: Attacker convergence check on all VQ stages
    for sid in ["1", "2", "3", "4", "5"]:
        if sid in res_map:
            r = res_map[sid]
            if not r.get("attacker_converged", True):
                msg = f"Stage {sid} attacker failed convergence check (attacker_converged = False)."
                warnings.append(f"[FAIL] {msg}")
                print(f"  ❌ {msg}")
            else:
                print(f"  ✓ Stage {sid} Attacker Convergence: PASS")

    # Check 2: Monotonicity with +0.01 tolerance and Dual-Attacker Directional Alignment
    vq_stage_ids = [s for s in ["1", "2", "3", "4", "5"] if s in res_map]
    for idx in range(len(vq_stage_ids) - 1):
        s_curr = res_map[vq_stage_ids[idx]]
        s_next = res_map[vq_stage_ids[idx + 1]]

        curr_co = s_curr.get("ssim")
        next_co = s_next.get("ssim")
        curr_ph = s_curr.get("posthoc_recon_ssim")
        next_ph = s_next.get("posthoc_recon_ssim")

        if curr_co is not None and next_co is not None:
            delta_co = next_co - curr_co
            # Monotonicity with +0.01 tolerance
            if delta_co > 0.01:
                msg = f"Monotonicity warning Stage {s_curr['stage_id']}->{s_next['stage_id']}: SSIM increased by {delta_co:+.4f} (exceeds +0.01 tolerance)."
                warnings.append(f"[WARN] {msg}")
                print(f"  ⚠️ {msg}")
            else:
                print(f"  ✓ Monotonicity Stage {s_curr['stage_id']}->{s_next['stage_id']}: PASS (delta={delta_co:+.4f})")

            # Directional Alignment check with post-hoc attacker
            if curr_ph is not None and next_ph is not None:
                delta_ph = next_ph - curr_ph
                # If deltas have opposing signs and absolute magnitudes > 0.005
                if (delta_co * delta_ph < 0) and (abs(delta_co) > 0.005 or abs(delta_ph) > 0.005):
                    msg = f"Directional divergence Stage {s_curr['stage_id']}->{s_next['stage_id']}: Co-trained delta={delta_co:+.4f} vs Post-hoc delta={delta_ph:+.4f}."
                    warnings.append(f"[WARN] {msg}")
                    print(f"  ⚠️ {msg}")
                else:
                    print(f"  ✓ Attacker Directional Alignment Stage {s_curr['stage_id']}->{s_next['stage_id']}: PASS")

    # Check 3: Stage 5 vs Canonical Reference Run (~80.43% Rank-1, ~60.10% mAP)
    if "5" in res_map:
        s5 = res_map["5"]
        r1 = s5.get("rank1")
        map_val = s5.get("mAP")
        if r1 is not None and map_val is not None:
            r1_diff = abs(r1 - 80.43)
            map_diff = abs(map_val - 60.10)
            if r1_diff > 2.5 or map_diff > 2.5:
                msg = f"Stage 5 vs Canonical Run delta exceeds tolerance: Rank-1={r1:.2f}% (ref 80.43%), mAP={map_val:.2f}% (ref 60.10%)."
                warnings.append(f"[WARN] {msg}")
                print(f"  ⚠️ {msg}")
            else:
                print(f"  ✓ Stage 5 Canonical Match: PASS (Rank-1={r1:.2f}%, mAP={map_val:.2f}%)")

    if not warnings:
        print("\n  >>> ALL POST-TRAINING VALIDATION CHECKS PASSED SUCCESSFULLY. <<<")
    else:
        print(f"\n  >>> VALIDATION COMPLETED WITH {len(warnings)} WARNING(S)/FAIL(S). <<<")

    return warnings


# ===========================================================================
# Main
# ===========================================================================

def main():
    args = parse_args()

    # Filter stages if --stages was given
    selected_ids = set(args.stages) if args.stages else None
    stages_to_run = [
        s for s in STAGES
        if selected_ids is None or s["id"] in selected_ids
    ]

    if not stages_to_run:
        print(f"[ERROR] No stages matched --stages {args.stages}. "
              f"Valid IDs: {[s['id'] for s in STAGES]}")
        sys.exit(1)

    # Ensure output root exists
    os.makedirs("outputs_ablation_market", exist_ok=True)

    # ------------------------------------------------------------------
    # Dry-run: validate configs and print plan, then exit
    # ------------------------------------------------------------------
    if args.dry_run:
        banner("DRY-RUN — Config Validation")
        all_ok = True
        for stage in stages_to_run:
            exists = os.path.isfile(stage["config"])
            if not exists:
                print(f"  [MISSING] Stage {stage['id']:>3}  {stage['config']}")
                all_ok = False
                continue

            # Load YAML and assert SSOT flag alignment
            with open(stage["config"]) as f:
                cfg = yaml.safe_load(f)

            cfg_noise = cfg.get("lambda_noise", 0.0) > 0
            cfg_priv = (cfg.get("lambda_priv_start", 0.0) > 0 or cfg.get("lambda_priv_max", 0.0) > 0)
            cfg_id = (cfg.get("lambda_id_start", 0.0) > 0 or cfg.get("lambda_id_max", 0.0) > 0)
            cfg_entropy = cfg.get("use_entropy_privacy", False)

            mismatches = []
            if stage["noise_active"] != cfg_noise:
                mismatches.append(f"noise_active SSOT={stage['noise_active']} vs cfg={cfg_noise}")
            if stage["lambda_priv_active"] != cfg_priv:
                mismatches.append(f"lambda_priv_active SSOT={stage['lambda_priv_active']} vs cfg={cfg_priv}")
            if stage["lambda_id_active"] != cfg_id:
                mismatches.append(f"lambda_id_active SSOT={stage['lambda_id_active']} vs cfg={cfg_id}")
            if stage["use_entropy_privacy"] != cfg_entropy:
                mismatches.append(f"use_entropy_privacy SSOT={stage['use_entropy_privacy']} vs cfg={cfg_entropy}")

            if mismatches:
                print(f"  [MISMATCH] Stage {stage['id']:>3}  {stage['config']}: {', '.join(mismatches)}")
                all_ok = False
            else:
                print(f"  [OK]       Stage {stage['id']:>3}  {stage['config']} (SSOT flags match)")

        if all_ok:
            print("\n  All config files present. Run plan:")
            for i, stage in enumerate(stages_to_run):
                print(f"\n  Step {i + 1}/{len(stages_to_run)}: {stage['label']}")
                print(f"    script : {stage['script']}")
                print(f"    config : {stage['config']}")
                print(f"    output : {stage['result_file']}")
                added = ", ".join(stage["components_added"])
                print(f"    adds   : {added}")
        else:
            print("\n  [ERROR] Missing config files — run aborted.")
            sys.exit(1)
        return

    # ------------------------------------------------------------------
    # Run stages
    # ------------------------------------------------------------------
    all_results = []
    failed_stages = []

    for stage_idx, stage in enumerate(stages_to_run):
        banner(f"Stage {stage['id']} of {len(STAGES)}: {stage['label']}  "
               f"[{stage_idx + 1}/{len(stages_to_run)}]")

        # Create output directories
        os.makedirs(stage["output_dir"], exist_ok=True)
        os.makedirs(stage["checkpoint_dir"], exist_ok=True)

        # ---- Training ------------------------------------------------
        training_ok = True
        if not args.skip_training:
            training_ok = run_training(stage, args.device)
            if not training_ok:
                print(f"\n  [ERROR] Training failed for Stage {stage['id']}. "
                      f"Skipping result collection for this stage.")
                failed_stages.append(stage["id"])
                all_results.append(build_stage_result(stage, None))
                continue

        # ---- Collect results -----------------------------------------
        print(f"\n  Collecting results from: {stage['output_dir']}")
        raw = collect_results(stage)

        if raw is None and args.skip_training:
            print(f"  [WARNING] No results JSON found for Stage {stage['id']} "
                  f"(--skip-training mode). Stage will appear as failed in table.")

        result = build_stage_result(stage, raw)

        # ---- Collapse check ------------------------------------------
        check_and_flag_collapse(result)

        # ---- Save per-stage results JSON -----------------------------
        with open(stage["result_file"], "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Saved: {stage['result_file']}")

        all_results.append(result)

    # ------------------------------------------------------------------
    # Consolidated output
    # ------------------------------------------------------------------
    banner("SAVING CONSOLIDATED RESULTS")

    # Consolidated JSON
    consolidated_json_path = "outputs_ablation_market/consolidated_results.json"
    
    # If we only ran a subset, load existing and merge, else start fresh
    if args.stages:
        try:
            with open(consolidated_json_path, "r") as f:
                existing_results = json.load(f)
            # Map by stage_id for easy lookup
            all_results_map = {r["stage_id"]: r for r in all_results}
            # Update existing with new results
            updated_results = []
            seen_ids = set()
            for r in existing_results:
                if r["stage_id"] in all_results_map:
                    updated_results.append(all_results_map[r["stage_id"]])
                    seen_ids.add(r["stage_id"])
                else:
                    updated_results.append(r)
            # Add any new ones not in the original list
            for r in all_results:
                if r["stage_id"] not in seen_ids:
                    updated_results.append(r)
            all_results = sorted(updated_results, key=lambda x: x["stage_id"])
        except (FileNotFoundError, json.JSONDecodeError):
            pass

    with open(consolidated_json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  JSON  : {consolidated_json_path}")

    # Markdown table
    md_table = build_markdown_table(all_results)
    md_path = "outputs_ablation_market/ablation_table.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Progressive Ablation Study — Market-1501 (Entropy Confusion Pipeline)\n\n")
        f.write("> Each stage is strictly additive: Stage N = Stage N-1 + one new component.\n")
        f.write("> All stages use `codebook_size: 4096`. Stage 0 posthoc fields are N/A.\n")
        f.write("> † = stage was re-run due to a bug fix; previous numbers must not be cited.\n\n")
        f.write(md_table)
        f.write("\n\n## Per-Stage Components Added\n\n")
        for r in all_results:
            for c in r.get("components_added", []):
                f.write(f"- **Stage {r['stage_id']}**: {c}\n")
        f.write("\n")
        # Re-run provenance
        rerun_stages = [r for r in all_results if r.get("rerun_reason")]
        if rerun_stages:
            f.write("\n## Re-Run Provenance (Supplementary Traceability)\n\n")
            for r in rerun_stages:
                f.write(f"- **Stage {r['stage_id']} ({r['stage_label']})**: {r['rerun_reason']}\n")
            f.write("\n")
        # Discarded stages note (for paper supplementary)
        f.write("## Discarded Stages\n\n")
        f.write(
            "- **Old Stage 4a (`entropy_privacy_passive`)**: Discarded — "
            "`use_entropy_privacy: true` with `lambda_priv = 0.0` caused "
            "`loss_priv = -0.0 * loss_recon = 0`, so the entropy weighting had zero "
            "gradient effect. Stage 4a and Stage 3 results were identical to 6 decimal "
            "places across all metrics. Replaced by `cotrained_attacker_region` (new 4a) "
            "and `entropy_privacy_active` (new 4b).\n"
        )
        f.write(
            "- **Old Stage 4b (`cotrained_attacker`)**: Discarded — built on the "
            "invalid Stage 4a. Replaced by `entropy_privacy_active` (new 4b).\n"
        )
        f.write(
            "- **Old Stage 5 (`full_no_grl`, rank1=80.43%, mAP=60.10%)**: "
            "Superseded — re-run on the corrected 4a→4b chain. Do not cite old numbers.\n\n"
        )
        if failed_stages:
            f.write(f"\n> [!WARNING]\n> Training failed for stages: {failed_stages}\n")
        collapsed = [r["stage_id"] for r in all_results if r.get("collapse_flagged")]
        if collapsed:
            f.write(f"\n> [!CAUTION]\n> Adversarial collapse detected in stages: {collapsed} "
                    f"(collapse_count > 0 → valid = False)\n")
    print(f"  Table : {md_path}")

    # Print summary to terminal
    print_summary_table(all_results)

    # Post-training validation suite execution
    validate_ablation_sequence(all_results)

    # Final status
    banner("RUN COMPLETE")
    n_valid = sum(1 for r in all_results if r.get("valid"))
    n_total = len(all_results)
    print(f"  Stages completed: {n_total}")
    print(f"  Valid (no collapse): {n_valid}/{n_total}")
    if failed_stages:
        print(f"  Training failures : {failed_stages}")
    collapsed = [r["stage_id"] for r in all_results if r.get("collapse_flagged")]
    if collapsed:
        print(f"  Collapse-flagged  : {collapsed}")
    print(f"\n  Consolidated JSON : {consolidated_json_path}")
    print(f"  Ablation table    : {md_path}")

    if failed_stages or collapsed:
        sys.exit(2)   # Non-zero exit so CI/shell scripts can detect partial failure


if __name__ == "__main__":
    main()

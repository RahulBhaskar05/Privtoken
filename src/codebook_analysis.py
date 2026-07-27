"""
Codebook Analysis Tools for PrivToken-ReID.

Provides analytical tools to understand what the VQ codebook is learning
and whether it preserves privacy. Reviewers will ask these questions:
    - Are tokens identity-specific or reusable across people?
    - Which codebook entries are used most?
    - Do certain codes only appear for specific identities?

Key analyses:
    - Token usage histogram per identity
    - Token overlap matrix across identities (higher = better privacy)
    - Code perplexity and utilization
    - Identity-specificity score per code
    - Cross-camera token consistency

Usage:
    from src.codebook_analysis import analyze_codebook
    results = analyze_codebook(tokenizer, cfg, device)
"""

import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
from tqdm import tqdm

from src.datasets import get_dataloader


def collect_token_statistics(tokenizer, cfg, device, split='gallery'):
    """
    Collect token usage statistics across the dataset.

    For each image, records which codebook indices were assigned to each
    spatial position, along with the person ID and camera ID.

    Args:
        tokenizer: PrivacyTokenizer (eval mode).
        cfg: Config dict.
        device: Compute device.
        split: Dataset split to analyze.

    Returns:
        dict with:
            'all_indices': list of (B, H*W) arrays — codebook indices per image.
            'all_pids': np.ndarray — person IDs.
            'all_camids': np.ndarray — camera IDs.
            'codebook_size': int — K.
    """
    tokenizer.eval()
    loader, _ = get_dataloader(cfg, split)

    all_indices = []
    all_pids = []
    all_camids = []

    with torch.no_grad():
        for imgs, pids, camids, _ in tqdm(loader, desc=f"Collecting tokens ({split})"):
            imgs = imgs.to(device)
            _, _, indices, _ = tokenizer(imgs)  # indices: (B, H*W)
            all_indices.append(indices.cpu().numpy())
            all_pids.append(
                pids.numpy() if isinstance(pids, torch.Tensor) else np.array(pids)
            )
            all_camids.append(
                camids.numpy() if isinstance(camids, torch.Tensor) else np.array(camids)
            )

    return {
        'all_indices': np.concatenate(all_indices, axis=0),   # (N, H*W)
        'all_pids': np.concatenate(all_pids),                  # (N,)
        'all_camids': np.concatenate(all_camids),              # (N,)
        'codebook_size': cfg['codebook_size'],
    }


def compute_codebook_utilization(stats):
    """
    Compute codebook utilization metrics.

    Args:
        stats: Output from collect_token_statistics().

    Returns:
        dict with utilization metrics.
    """
    K = stats['codebook_size']
    all_indices = stats['all_indices'].flatten()

    # Usage counts
    usage_counts = np.bincount(all_indices, minlength=K)
    utilized = (usage_counts > 0).sum()
    utilization = utilized / K

    # Perplexity (higher = more uniform usage = better)
    probs = usage_counts / usage_counts.sum()
    probs = probs[probs > 0]  # filter zeros
    entropy = -np.sum(probs * np.log2(probs + 1e-10))
    perplexity = 2 ** entropy

    # Gini coefficient (lower = more uniform)
    sorted_counts = np.sort(usage_counts)
    n = len(sorted_counts)
    index = np.arange(1, n + 1)
    gini = (2 * np.sum(index * sorted_counts)) / (n * np.sum(sorted_counts)) - (n + 1) / n

    return {
        'total_entries': K,
        'utilized_entries': int(utilized),
        'utilization_rate': float(utilization),
        'perplexity': float(perplexity),
        'max_perplexity': float(K),  # theoretical max
        'normalized_perplexity': float(perplexity / K),
        'gini_coefficient': float(gini),
        'usage_counts': usage_counts,
        'entropy_bits': float(entropy),
    }


def compute_identity_specificity(stats, top_k_ids=20):
    """
    Compute how identity-specific each codebook entry is.

    A "good" privacy codebook should have entries that are SHARED across
    many identities (low specificity). If certain codes only appear for
    specific people, that's an identity leakage risk.

    Metrics:
        - Per-code identity entropy: higher = shared across more people
        - Identity specificity score: fraction of codes that are highly
          specific to individual identities

    Args:
        stats: Output from collect_token_statistics().
        top_k_ids: Number of top identities to analyze per code.

    Returns:
        dict with identity specificity metrics.
    """
    K = stats['codebook_size']
    all_indices = stats['all_indices']  # (N, H*W)
    all_pids = stats['all_pids']        # (N,)

    # Build code → identity mapping
    code_to_pids = defaultdict(list)
    for i in range(len(all_pids)):
        pid = all_pids[i]
        for code_idx in all_indices[i]:
            code_to_pids[code_idx].append(pid)

    # Per-code identity entropy
    code_entropies = []
    code_num_ids = []
    identity_specific_codes = 0

    for k in range(K):
        pids = code_to_pids.get(k, [])
        if len(pids) == 0:
            continue

        # Count unique IDs using this code
        pid_counts = defaultdict(int)
        for pid in pids:
            pid_counts[pid] += 1

        counts = np.array(list(pid_counts.values()), dtype=np.float64)
        probs = counts / counts.sum()
        entropy = -np.sum(probs * np.log2(probs + 1e-10))

        code_entropies.append(entropy)
        code_num_ids.append(len(pid_counts))

        # A code is "identity-specific" if used by <= 3 identities
        if len(pid_counts) <= 3:
            identity_specific_codes += 1

    if not code_entropies:
        return {
            'mean_code_identity_entropy': 0.0,
            'identity_specific_ratio': 0.0,
            'mean_ids_per_code': 0.0,
        }

    return {
        'mean_code_identity_entropy': float(np.mean(code_entropies)),
        'median_code_identity_entropy': float(np.median(code_entropies)),
        'min_code_identity_entropy': float(np.min(code_entropies)),
        'max_code_identity_entropy': float(np.max(code_entropies)),
        'identity_specific_codes': identity_specific_codes,
        'identity_specific_ratio': identity_specific_codes / max(len(code_entropies), 1),
        'mean_ids_per_code': float(np.mean(code_num_ids)),
        'median_ids_per_code': float(np.median(code_num_ids)),
    }


def compute_token_overlap_across_identities(stats, num_sample_ids=20):
    """
    Compute how much token vocabulary is shared between identity pairs.

    High overlap = codes are generic/structural, not identity-specific.
    This is GOOD for privacy.

    Low overlap = each person uses a unique set of codes.
    This is BAD for privacy (codes leak identity).

    Args:
        stats: Output from collect_token_statistics().
        num_sample_ids: Sample this many IDs for pairwise analysis.

    Returns:
        dict with overlap metrics.
    """
    all_indices = stats['all_indices']  # (N, H*W)
    all_pids = stats['all_pids']        # (N,)

    unique_pids = np.unique(all_pids)
    if len(unique_pids) > num_sample_ids:
        np.random.seed(42)
        sample_pids = np.random.choice(unique_pids, num_sample_ids, replace=False)
    else:
        sample_pids = unique_pids

    # Build per-identity vocabulary (set of used codes)
    pid_vocabs = {}
    for pid in sample_pids:
        mask = all_pids == pid
        indices = all_indices[mask].flatten()
        pid_vocabs[pid] = set(indices.tolist())

    # Pairwise Jaccard similarity
    pids_list = list(pid_vocabs.keys())
    n = len(pids_list)
    jaccard_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            v1 = pid_vocabs[pids_list[i]]
            v2 = pid_vocabs[pids_list[j]]
            intersection = len(v1 & v2)
            union = len(v1 | v2)
            jaccard = intersection / max(union, 1)
            jaccard_matrix[i, j] = jaccard
            jaccard_matrix[j, i] = jaccard

    # Extract off-diagonal (inter-identity) overlaps
    mask = ~np.eye(n, dtype=bool)
    inter_overlaps = jaccard_matrix[mask]

    return {
        'mean_jaccard_overlap': float(np.mean(inter_overlaps)),
        'median_jaccard_overlap': float(np.median(inter_overlaps)),
        'min_jaccard_overlap': float(np.min(inter_overlaps)),
        'max_jaccard_overlap': float(np.max(inter_overlaps)),
        'mean_vocab_size': float(np.mean([len(v) for v in pid_vocabs.values()])),
        'jaccard_matrix': jaccard_matrix,
        'sampled_pids': pids_list,
    }


def compute_cross_camera_consistency(stats):
    """
    Analyze token consistency across different cameras for the same identity.

    For privacy, we WANT codes to be camera-invariant (structural), not
    camera-specific (appearance-memorizing).

    Args:
        stats: Output from collect_token_statistics().

    Returns:
        dict with cross-camera consistency metrics.
    """
    all_indices = stats['all_indices']
    all_pids = stats['all_pids']
    all_camids = stats['all_camids']

    unique_pids = np.unique(all_pids)

    # For each identity, compare token sets across cameras
    cross_cam_overlaps = []

    for pid in unique_pids:
        pid_mask = all_pids == pid
        pid_cams = all_camids[pid_mask]
        pid_indices = all_indices[pid_mask]

        unique_cams = np.unique(pid_cams)
        if len(unique_cams) < 2:
            continue

        # Build per-camera vocabulary
        cam_vocabs = {}
        for cam in unique_cams:
            cam_mask = pid_cams == cam
            cam_vocabs[cam] = set(pid_indices[cam_mask].flatten().tolist())

        # Pairwise overlap between cameras for this identity
        cams = list(cam_vocabs.keys())
        for i in range(len(cams)):
            for j in range(i + 1, len(cams)):
                v1 = cam_vocabs[cams[i]]
                v2 = cam_vocabs[cams[j]]
                jaccard = len(v1 & v2) / max(len(v1 | v2), 1)
                cross_cam_overlaps.append(jaccard)

    if not cross_cam_overlaps:
        return {'mean_cross_camera_overlap': 0.0}

    return {
        'mean_cross_camera_overlap': float(np.mean(cross_cam_overlaps)),
        'median_cross_camera_overlap': float(np.median(cross_cam_overlaps)),
        'std_cross_camera_overlap': float(np.std(cross_cam_overlaps)),
    }


def analyze_codebook(tokenizer, cfg, device, save_dir=None):
    """
    Run full codebook analysis suite.

    Args:
        tokenizer: PrivacyTokenizer.
        cfg: Config dict.
        device: Compute device.
        save_dir: Directory to save results (optional).

    Returns:
        dict: Combined analysis results.
    """
    print("\n" + "=" * 60)
    print("CODEBOOK ANALYSIS")
    print("=" * 60)

    # Collect statistics
    stats = collect_token_statistics(tokenizer, cfg, device, split='gallery')

    # Run analyses
    print("\n[1/4] Codebook utilization...")
    utilization = compute_codebook_utilization(stats)
    print(f"  Utilized: {utilization['utilized_entries']}/{utilization['total_entries']} "
          f"({utilization['utilization_rate']*100:.1f}%)")
    print(f"  Perplexity: {utilization['perplexity']:.1f} / {utilization['max_perplexity']}"
          f" (normalized: {utilization['normalized_perplexity']:.3f})")
    print(f"  Gini coefficient: {utilization['gini_coefficient']:.3f}")

    print("\n[2/4] Identity specificity...")
    specificity = compute_identity_specificity(stats)
    print(f"  Mean identity entropy per code: {specificity['mean_code_identity_entropy']:.2f} bits")
    print(f"  Mean IDs per code: {specificity['mean_ids_per_code']:.1f}")
    print(f"  Identity-specific codes (≤3 IDs): "
          f"{specificity.get('identity_specific_codes', 0)} "
          f"({specificity['identity_specific_ratio']*100:.1f}%)")

    print("\n[3/4] Token overlap across identities...")
    overlap = compute_token_overlap_across_identities(stats)
    print(f"  Mean Jaccard overlap: {overlap['mean_jaccard_overlap']:.3f}")
    print(f"  Mean vocabulary size per ID: {overlap['mean_vocab_size']:.1f}")
    assessment = ('✓ High overlap — codes are generic/structural'
                  if overlap['mean_jaccard_overlap'] > 0.5
                  else '⚠ Low overlap — codes may be identity-specific')
    print(f"  Assessment: {assessment}")

    print("\n[4/4] Cross-camera consistency...")
    cross_cam = compute_cross_camera_consistency(stats)
    print(f"  Mean cross-camera overlap: {cross_cam['mean_cross_camera_overlap']:.3f}")

    # Combine results (exclude non-serializable items)
    results = {}
    for d in [utilization, specificity, overlap, cross_cam]:
        for k, v in d.items():
            if isinstance(v, (int, float, str, bool)):
                results[k] = v

    # Privacy assessment
    privacy_score = 0
    if overlap['mean_jaccard_overlap'] > 0.5:
        privacy_score += 1
    if specificity['identity_specific_ratio'] < 0.1:
        privacy_score += 1
    if utilization['normalized_perplexity'] > 0.3:
        privacy_score += 1

    results['privacy_score'] = privacy_score
    results['privacy_assessment'] = ['Weak', 'Moderate', 'Good', 'Strong'][privacy_score]

    print(f"\n{'=' * 60}")
    print(f"  Codebook Privacy Score: {privacy_score}/3 ({results['privacy_assessment']})")
    print(f"{'=' * 60}\n")

    # Save results
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        json_path = os.path.join(save_dir, 'codebook_analysis.json')
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"  → Results saved to {json_path}")

        # Save usage counts
        np.save(os.path.join(save_dir, 'codebook_usage_counts.npy'),
                utilization['usage_counts'])

    return results

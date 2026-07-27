"""
Evaluation utilities for PrivToken-ReID.

Provides:
- extract_features(): Run inference and collect embeddings
- compute_cmc_map(): Standard Market-1501 CMC/mAP evaluation with junk removal
- compute_privacy_metrics(): PSNR, SSIM, LPIPS between originals and reconstructions
- evaluate_full(): Orchestrates full evaluation pipeline
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.datasets import get_dataloader


def extract_features(tokenizer, reid_head, loader, device):
    """
    Run inference on a data loader and extract Re-ID features.

    Args:
        tokenizer (PrivacyTokenizer): Tokenizer model (eval mode).
        reid_head (BNNeck): Re-ID head (eval mode).
        loader (DataLoader): Data loader (gallery or query split).
        device (torch.device): Compute device.

    Returns:
        features (np.ndarray): Feature embeddings, shape (N, D).
        pids (np.ndarray): Person IDs, shape (N,).
        camids (np.ndarray): Camera IDs, shape (N,).
        img_paths (list): Image file paths, length N.
    """
    tokenizer.eval()
    reid_head.eval()

    all_features = []
    all_pids = []
    all_camids = []
    all_paths = []

    with torch.no_grad():
        for imgs, pids, camids, paths in tqdm(loader, desc="Extracting features"):
            imgs = imgs.to(device)  # (B, 3, H, W)
            z_q, _, _, _ = tokenizer(imgs)  # 4-return: z_q, vq_loss, indices, utilisation
            fn, _, _ = reid_head(z_q)  # (B, token_dim) — use post-BN features for retrieval

            all_features.append(fn.cpu().numpy())
            all_pids.append(pids.numpy() if isinstance(pids, torch.Tensor) else np.array(pids))
            all_camids.append(camids.numpy() if isinstance(camids, torch.Tensor) else np.array(camids))
            all_paths.extend(paths)

    features = np.concatenate(all_features, axis=0)  # (N, D)
    pids = np.concatenate(all_pids, axis=0)           # (N,)
    camids = np.concatenate(all_camids, axis=0)       # (N,)

    return features, pids, camids, all_paths


def compute_cmc_map(query_feat, query_pids, query_camids,
                    gallery_feat, gallery_pids, gallery_camids, max_rank=50):
    """
    Compute CMC curve and mean Average Precision (mAP) using the standard
    Market-1501 evaluation protocol with junk removal.

    Protocol: For each query, gallery images with the SAME person ID AND SAME
    camera ID are treated as junk and excluded from ranking.

    Args:
        query_feat (np.ndarray): Query features, shape (Nq, D).
        query_pids (np.ndarray): Query person IDs, shape (Nq,).
        query_camids (np.ndarray): Query camera IDs, shape (Nq,).
        gallery_feat (np.ndarray): Gallery features, shape (Ng, D).
        gallery_pids (np.ndarray): Gallery person IDs, shape (Ng,).
        gallery_camids (np.ndarray): Gallery camera IDs, shape (Ng,).
        max_rank (int): Maximum rank to compute CMC for (default: 50).

    Returns:
        cmc (np.ndarray): Cumulative match characteristic, shape (max_rank,).
        mAP (float): Mean average precision.
    """
    num_query = query_feat.shape[0]

    # Compute pairwise L2 distance matrix: (Nq, Ng)
    # Using batch computation for memory efficiency
    dist_mat = _compute_distance_matrix(query_feat, gallery_feat)

    all_cmc = []
    all_ap = []

    for i in range(num_query):
        q_pid = query_pids[i]
        q_camid = query_camids[i]

        # Sort gallery by ascending distance
        order = np.argsort(dist_mat[i])
        sorted_pids = gallery_pids[order]
        sorted_camids = gallery_camids[order]

        # Junk mask: same person AND same camera
        junk_mask = (sorted_pids == q_pid) & (sorted_camids == q_camid)

        # Also skip gallery images with pid == -1 (distractors)
        junk_mask = junk_mask | (sorted_pids == -1)

        # Valid (non-junk) mask
        valid_mask = ~junk_mask

        # True matches: same PID, different camera (within valid entries)
        match_mask = (sorted_pids == q_pid) & (sorted_camids != q_camid)

        # If no valid matches exist, skip this query
        if match_mask.sum() == 0:
            continue

        # Remove junk entries from both match_mask and valid_mask
        # Work with valid entries only
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) == 0:
            continue

        # Re-index match positions within the valid-only ranking
        match_in_valid = np.isin(valid_indices,
                                 np.where(match_mask)[0])

        # CMC: cumulative match
        cmc = match_in_valid.cumsum()
        cmc[cmc > 1] = 1  # cap at 1 (found or not)

        # Pad or truncate to max_rank
        if len(cmc) >= max_rank:
            cmc = cmc[:max_rank]
        else:
            pad_value = cmc[-1] if len(cmc) > 0 else 0
            cmc = np.pad(cmc, (0, max_rank - len(cmc)),
                         constant_values=pad_value)

        all_cmc.append(cmc)

        # Average Precision (AP) for this query
        num_matches = match_in_valid.sum()
        if num_matches == 0:
            continue

        cum_tp = match_in_valid.cumsum().astype(float)
        precision_at_k = cum_tp / (np.arange(len(match_in_valid)) + 1.0)
        ap = (precision_at_k * match_in_valid).sum() / num_matches
        all_ap.append(ap)

    if len(all_cmc) == 0:
        print("[WARNING] No valid queries found for evaluation.")
        return np.zeros(max_rank), 0.0

    cmc = np.mean(all_cmc, axis=0)
    mAP = float(np.mean(all_ap))

    return cmc, mAP


def _compute_distance_matrix(query_feat, gallery_feat):
    """
    Compute pairwise L2 distance matrix between query and gallery features.

    Args:
        query_feat (np.ndarray): shape (Nq, D).
        gallery_feat (np.ndarray): shape (Ng, D).

    Returns:
        dist (np.ndarray): shape (Nq, Ng) L2 distances.
    """
    # ||q - g||^2 = ||q||^2 - 2*q·g^T + ||g||^2
    q_sq = np.sum(query_feat ** 2, axis=1, keepdims=True)  # (Nq, 1)
    g_sq = np.sum(gallery_feat ** 2, axis=1, keepdims=True).T  # (1, Ng)
    dist = q_sq - 2 * query_feat @ gallery_feat.T + g_sq  # (Nq, Ng)
    dist = np.clip(dist, 0.0, None)  # numerical stability
    return np.sqrt(dist)


def compute_privacy_metrics(tokenizer, attacker, loader, device, num_batches=20):
    """
    Compute privacy metrics by running the attacker on test data.

    Measures how well the attacker can reconstruct original images from
    quantized tokens. Lower reconstruction quality = better privacy.

    Args:
        tokenizer (PrivacyTokenizer): Tokenizer model (eval mode).
        attacker (UNetDecoder): Attacker decoder (eval mode).
        loader (DataLoader): Test data loader.
        device (torch.device): Compute device.
        num_batches (int): Number of batches to evaluate on.

    Returns:
        dict with keys:
            'psnr' (float): Peak Signal-to-Noise Ratio (lower = better privacy).
            'ssim' (float): Structural Similarity Index (lower = better privacy).
            'lpips' (float): Learned Perceptual Image Patch Similarity (higher = better privacy).
            'recon_samples' (list): List of (original, reconstructed) tensor pairs (8 samples).
    """
    tokenizer.eval()
    attacker.eval()

    MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    all_psnr = []
    all_ssim = []
    all_lpips = []
    recon_samples = []

    # LPIPS network for evaluation
    import lpips as lpips_lib
    lpips_net = lpips_lib.LPIPS(net='alex').to(device)
    lpips_net.eval()

    batch_count = 0
    with torch.no_grad():
        for imgs, _, _, _ in loader:
            if batch_count >= num_batches:
                break
            batch_count += 1

            imgs = imgs.to(device)  # (B, 3, H, W) — normalized

            # Denormalize to [0, 1]
            imgs_unnorm = (imgs * STD + MEAN).clamp(0, 1)  # (B, 3, H, W)

            # Tokenize and reconstruct
            z_q, _, _, _, _, _ = tokenizer(imgs)  # V4 returns 6 values: z_q, vq_loss, indices, util, entropy, z_q_noisy
            recon = attacker(z_q)  # (B, 3, H, W) in [0, 1]

            # PSNR
            mse = F.mse_loss(recon, imgs_unnorm, reduction='none').mean(dim=[1, 2, 3])  # (B,)
            psnr = 10.0 * torch.log10(1.0 / (mse + 1e-10))  # (B,)
            all_psnr.extend(psnr.cpu().numpy().tolist())

            # SSIM (simplified implementation)
            ssim_vals = _compute_ssim_batch(recon, imgs_unnorm)  # (B,)
            all_ssim.extend(ssim_vals.cpu().numpy().tolist())

            # LPIPS
            recon_scaled = recon * 2.0 - 1.0
            target_scaled = imgs_unnorm * 2.0 - 1.0
            lpips_vals = lpips_net(recon_scaled, target_scaled).squeeze()  # (B,)
            if lpips_vals.dim() == 0:
                lpips_vals = lpips_vals.unsqueeze(0)
            all_lpips.extend(lpips_vals.cpu().numpy().tolist())

            # Collect samples for visualization
            if len(recon_samples) < 8:
                num_to_take = min(8 - len(recon_samples), imgs_unnorm.size(0))
                for j in range(num_to_take):
                    recon_samples.append((
                        imgs_unnorm[j].cpu(),
                        recon[j].cpu(),
                    ))

    metrics = {
        'psnr': float(np.mean(all_psnr)),
        'ssim': float(np.mean(all_ssim)),
        'lpips': float(np.mean(all_lpips)),
        'recon_samples': recon_samples,
    }

    return metrics


def _compute_ssim_batch(img1, img2, window_size=11, C1=0.01**2, C2=0.03**2):
    """
    Compute SSIM between two batches of images.

    Args:
        img1 (Tensor): shape (B, 3, H, W) in [0, 1].
        img2 (Tensor): shape (B, 3, H, W) in [0, 1].
        window_size (int): Gaussian window size.
        C1 (float): Stability constant for luminance.
        C2 (float): Stability constant for contrast.

    Returns:
        ssim (Tensor): shape (B,) SSIM values per image.
    """
    # Create Gaussian window
    channel = img1.size(1)
    window = _gaussian_window(window_size, 1.5).to(img1.device)
    window = window.expand(channel, 1, window_size, window_size)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    # Average over spatial dims and channels
    return ssim_map.mean(dim=[1, 2, 3])  # (B,)


def _gaussian_window(size, sigma):
    """
    Create a 2D Gaussian window for SSIM computation.

    Args:
        size (int): Window size.
        sigma (float): Gaussian standard deviation.

    Returns:
        window (Tensor): shape (1, 1, size, size), normalized.
    """
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window = g.unsqueeze(1) @ g.unsqueeze(0)  # outer product: (size, size)
    return window.unsqueeze(0).unsqueeze(0)  # (1, 1, size, size)


def evaluate_full(tokenizer, reid_head, attacker, cfg, device):
    """
    Full evaluation: Re-ID metrics (CMC/mAP) + privacy metrics.

    Args:
        tokenizer (PrivacyTokenizer): Tokenizer model.
        reid_head (BNNeck): Re-ID head.
        attacker (UNetDecoder): Attacker decoder.
        cfg (dict): Configuration dictionary.
        device (torch.device): Compute device.

    Returns:
        dict: Evaluation results with keys:
            rank1, rank5, rank10, mAP (Re-ID performance)
            psnr, ssim, lpips (visual privacy metrics)
            + identity privacy metrics if eval_privacy=true in config
    """
    print("\n" + "=" * 60)
    print("Running full evaluation...")
    print("=" * 60)

    # Create gallery and query dataloaders
    gallery_loader, _ = get_dataloader(cfg, 'gallery')
    query_loader, _ = get_dataloader(cfg, 'query')

    # Extract features
    print("\n[1/3] Extracting gallery features...")
    gallery_feat, gallery_pids, gallery_camids, _ = extract_features(
        tokenizer, reid_head, gallery_loader, device
    )

    print("[2/3] Extracting query features...")
    query_feat, query_pids, query_camids, _ = extract_features(
        tokenizer, reid_head, query_loader, device
    )

    # Compute CMC/mAP
    print("[3/3] Computing CMC and mAP...")
    cmc, mAP = compute_cmc_map(
        query_feat, query_pids, query_camids,
        gallery_feat, gallery_pids, gallery_camids,
    )

    # Compute visual privacy metrics (PSNR/SSIM/LPIPS)
    print("\nComputing visual privacy metrics...")
    privacy = compute_privacy_metrics(tokenizer, attacker, gallery_loader, device)

    results = {
        'rank1': cmc[0] * 100,
        'rank5': cmc[4] * 100,
        'rank10': cmc[9] * 100,
        'mAP': mAP * 100,
        'psnr': privacy['psnr'],
        'ssim': privacy['ssim'],
        'lpips': privacy['lpips'],
        'recon_samples': privacy['recon_samples'],
    }

    # Print summary
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"  [Utility]")
    print(f"    Rank-1:  {results['rank1']:.2f}%")
    print(f"    Rank-5:  {results['rank5']:.2f}%")
    print(f"    Rank-10: {results['rank10']:.2f}%")
    print(f"    mAP:     {results['mAP']:.2f}%")
    print(f"  [Visual Privacy]")
    print(f"    PSNR:    {results['psnr']:.2f} dB")
    print(f"    SSIM:    {results['ssim']:.4f}")
    print(f"    LPIPS:   {results['lpips']:.4f}")

    # Identity privacy evaluation (Phase 2 upgrade)
    if cfg.get('eval_privacy', False):
        try:
            from src.privacy_eval import evaluate_privacy_full
            print("\n  Running identity privacy evaluation...")
            privacy_results = evaluate_privacy_full(
                tokenizer, attacker, cfg, device,
                skip_face=False,
            )
            # Merge privacy results (skip non-serializable items)
            for k, v in privacy_results.items():
                if isinstance(v, (int, float, str, bool)):
                    results[k] = v

            print(f"  [Identity Privacy]")
            print(f"    Recon ReID Rank-1:  {results.get('recon_rank1', 0):.2f}%")
            print(f"    Recon ReID mAP:     {results.get('recon_mAP', 0):.2f}%")
            print(f"    Leakage Ratio:      {results.get('identity_leakage_ratio', 0):.3f}")
            print(f"    Token ID Top-1:     {results.get('token_id_top1', 0):.1f}%")
        except Exception as e:
            print(f"  [WARNING] Privacy evaluation failed: {e}")
            import traceback
            traceback.print_exc()

    print("=" * 60 + "\n")

    return results

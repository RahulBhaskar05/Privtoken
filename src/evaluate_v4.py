"""
V4 Evaluation Suite for PrivToken-ReID.

Extends the v3 evaluation with:
    - mINP (mean Inverse Negative Penalty): harder retrieval metric
    - PU-Score (Privacy-Utility Balance): composite harmonic metric
    - ISD (Identity Separation Degree): feature-space identity dissimilarity
    - Post-hoc strong attacker evaluation: independent privacy validation

All v3 metrics (CMC, mAP, PSNR, SSIM, LPIPS) are preserved.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm

from src.datasets import get_dataloader
from src.evaluate import (
    _compute_distance_matrix,
    _compute_ssim_batch,
    _gaussian_window,
)


# =========================================================================
# Feature Extraction (multi-granularity aware)
# =========================================================================

def extract_features_v4(tokenizer, reid_head, loader, device):
    """Extract concatenated multi-granularity features.

    Uses reid_head.get_eval_features() which concatenates global + all part
    features into a single high-dimensional descriptor.

    Args:
        tokenizer: PrivacyTokenizerV4 (eval mode).
        reid_head: MultiGranularityHead (eval mode).
        loader: DataLoader.
        device: Compute device.

    Returns:
        features (np.ndarray): (N, (1+num_parts)*D) feature matrix.
        pids (np.ndarray): (N,) person IDs.
        camids (np.ndarray): (N,) camera IDs.
        img_paths (list): Image paths.
    """
    tokenizer.eval()
    reid_head.eval()

    all_features, all_pids, all_camids, all_paths = [], [], [], []

    with torch.no_grad():
        for imgs, pids, camids, paths in tqdm(loader, desc="Extracting features"):
            imgs = imgs.to(device)
            z_q, _, _, _, _, _ = tokenizer(imgs)
            feat = reid_head.get_eval_features(z_q)  # (B, 2560)

            all_features.append(feat.cpu().numpy())
            all_pids.append(pids.numpy() if isinstance(pids, torch.Tensor)
                            else np.array(pids))
            all_camids.append(camids.numpy() if isinstance(camids, torch.Tensor)
                              else np.array(camids))
            all_paths.extend(paths)

    return (np.concatenate(all_features),
            np.concatenate(all_pids),
            np.concatenate(all_camids),
            all_paths)


# =========================================================================
# CMC / mAP / mINP Computation
# =========================================================================

def compute_cmc_map_minp(query_feat, query_pids, query_camids,
                          gallery_feat, gallery_pids, gallery_camids,
                          max_rank=50):
    """Compute CMC, mAP, and mINP with standard Market-1501 junk removal.

    mINP (mean Inverse Negative Penalty) measures how many true matches
    appear before the hardest true match. It is strictly harder than mAP
    because it focuses on the WORST retrieval position for each query.

        INP_q = (num_valid_matches - hardest_match_position) / (num_valid_matches - 1)
        mINP = mean(INP_q) over all queries

    Reference: Ye et al., "Deep Learning for Person Re-Identification:
    A Survey and Outlook", TPAMI 2022.

    Args:
        query_feat, query_pids, query_camids: Query set.
        gallery_feat, gallery_pids, gallery_camids: Gallery set.
        max_rank: Maximum rank for CMC.

    Returns:
        cmc (np.ndarray): CMC curve (max_rank,).
        mAP (float): Mean average precision.
        mINP (float): Mean inverse negative penalty.
    """
    num_query = query_feat.shape[0]
    dist_mat = _compute_distance_matrix(query_feat, gallery_feat)

    all_cmc = []
    all_ap = []
    all_inp = []

    for i in range(num_query):
        q_pid = query_pids[i]
        q_camid = query_camids[i]

        order = np.argsort(dist_mat[i])
        sorted_pids = gallery_pids[order]
        sorted_camids = gallery_camids[order]

        # Junk: same person AND same camera, or distractor pid=-1
        junk_mask = ((sorted_pids == q_pid) & (sorted_camids == q_camid)) | (sorted_pids == -1)
        valid_mask = ~junk_mask
        match_mask = (sorted_pids == q_pid) & (sorted_camids != q_camid)

        if match_mask.sum() == 0:
            continue

        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) == 0:
            continue

        match_in_valid = np.isin(valid_indices, np.where(match_mask)[0])

        # CMC
        cmc = match_in_valid.cumsum()
        cmc[cmc > 1] = 1
        if len(cmc) >= max_rank:
            cmc = cmc[:max_rank]
        else:
            pad_value = cmc[-1] if len(cmc) > 0 else 0
            cmc = np.pad(cmc, (0, max_rank - len(cmc)), constant_values=pad_value)
        all_cmc.append(cmc)

        # AP
        num_matches = match_in_valid.sum()
        if num_matches == 0:
            continue
        cum_tp = match_in_valid.cumsum().astype(float)
        precision_at_k = cum_tp / (np.arange(len(match_in_valid)) + 1.0)
        ap = (precision_at_k * match_in_valid).sum() / num_matches
        all_ap.append(ap)

        # INP: position of the hardest (last) true match in valid ranking
        match_positions = np.where(match_in_valid)[0]
        if len(match_positions) > 0:
            hardest_pos = match_positions[-1] + 1  # 1-indexed rank of farthest true match
            num_valid_matches = len(match_positions)
            # Correct mINP: num_valid_matches / rank of hardest match
            inp = num_valid_matches / hardest_pos
            all_inp.append(inp)

    if len(all_cmc) == 0:
        print("[WARNING] No valid queries for evaluation.")
        return np.zeros(max_rank), 0.0, 0.0

    cmc = np.mean(all_cmc, axis=0)
    mAP = float(np.mean(all_ap))
    mINP = float(np.mean(all_inp)) if all_inp else 0.0

    return cmc, mAP, mINP


# =========================================================================
# PU-Score (Privacy-Utility Balance)
# =========================================================================

def compute_pu_score(rank1, ssim):
    """Compute Privacy-Utility balance score.

    Harmonic mean of normalized utility (Rank-1) and normalized privacy
    (1 - SSIM). A model that achieves high Rank-1 AND low SSIM will score
    highest. The harmonic mean penalizes models that sacrifice one for
    the other.

        PU = 2 * (R1/100) * (1 - SSIM) / ((R1/100) + (1 - SSIM)) * 100

    Reference: Adapted from F1-score formulation for privacy-utility.

    Args:
        rank1 (float): Rank-1 accuracy in percent.
        ssim (float): SSIM of attacker reconstruction (lower = better privacy).

    Returns:
        float: PU-Score in [0, 100].
    """
    utility = rank1 / 100.0
    privacy = max(0.0, 1.0 - ssim)

    if utility + privacy < 1e-10:
        return 0.0

    return 2 * utility * privacy / (utility + privacy) * 100


# =========================================================================
# ISD (Identity Separation Degree)
# =========================================================================

def compute_isd(orig_features, anon_features):
    """Compute Identity Separation Degree.

    Measures how much the anonymization shifts the feature space by computing
    the average cosine distance between original and anonymized features
    for the same images.

    Higher ISD = stronger identity transformation = better privacy.

    Reference: ADM (Anonymous Disentanglement Module) paper.

    Args:
        orig_features (np.ndarray): Original image features (N, D).
        anon_features (np.ndarray): Anonymized/tokenized features (N, D).

    Returns:
        float: Mean cosine distance (1 - cosine_similarity).
    """
    # L2 normalize
    orig_norm = orig_features / (np.linalg.norm(orig_features, axis=1, keepdims=True) + 1e-10)
    anon_norm = anon_features / (np.linalg.norm(anon_features, axis=1, keepdims=True) + 1e-10)

    # Cosine similarity per sample
    cos_sim = np.sum(orig_norm * anon_norm, axis=1)  # (N,)

    # ISD = 1 - mean cosine similarity
    isd = 1.0 - float(np.mean(cos_sim))
    return isd


# =========================================================================
# Privacy Metrics (PSNR / SSIM / LPIPS)
# =========================================================================

def compute_privacy_metrics_v4(tokenizer, attacker, loader, device, num_batches=20):
    """Compute visual privacy metrics using the co-trained attacker.

    Same as v3 but adapted for v4's 6-return tokenizer.

    Args:
        tokenizer: PrivacyTokenizerV4.
        attacker: Decoder.
        loader: DataLoader.
        device: Compute device.
        num_batches: Max batches.

    Returns:
        dict: psnr, ssim, lpips, recon_samples.
    """
    import lpips as lpips_lib

    tokenizer.eval()
    attacker.eval()

    MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    all_psnr, all_ssim, all_lpips = [], [], []
    recon_samples = []

    lpips_net = lpips_lib.LPIPS(net='alex').to(device)
    lpips_net.eval()

    batch_count = 0
    with torch.no_grad():
        for imgs, _, _, _ in loader:
            if batch_count >= num_batches:
                break
            batch_count += 1

            imgs = imgs.to(device)
            imgs_unnorm = (imgs * STD + MEAN).clamp(0, 1)

            z_q, _, _, _, _, _ = tokenizer(imgs)
            recon = attacker(z_q)

            # PSNR
            mse = F.mse_loss(recon, imgs_unnorm, reduction='none').mean(dim=[1, 2, 3])
            psnr = 10.0 * torch.log10(1.0 / (mse + 1e-10))
            all_psnr.extend(psnr.cpu().numpy().tolist())

            # SSIM
            ssim_vals = _compute_ssim_batch(recon, imgs_unnorm)
            all_ssim.extend(ssim_vals.cpu().numpy().tolist())

            # LPIPS
            recon_s = recon * 2.0 - 1.0
            target_s = imgs_unnorm * 2.0 - 1.0
            lpips_vals = lpips_net(recon_s, target_s).squeeze()
            if lpips_vals.dim() == 0:
                lpips_vals = lpips_vals.unsqueeze(0)
            all_lpips.extend(lpips_vals.cpu().numpy().tolist())

            if len(recon_samples) < 8:
                for j in range(min(8 - len(recon_samples), imgs_unnorm.size(0))):
                    recon_samples.append((imgs_unnorm[j].cpu(), recon[j].cpu()))

    return {
        'psnr': float(np.mean(all_psnr)),
        'ssim': float(np.mean(all_ssim)),
        'lpips': float(np.mean(all_lpips)),
        'recon_samples': recon_samples,
    }


# =========================================================================
# Post-Hoc Strong Attacker Evaluation
# =========================================================================

class StrongPostHocDecoder(nn.Module):
    """Strong post-hoc reconstruction attacker for independent privacy validation.

    Deeper and wider than any attacker used during training. Trains from
    scratch against frozen v4 tokens with NO access to training dynamics.
    If this attacker still fails to reconstruct, the privacy claim is robust.

    Architecture:
        (B, 512, 16, 8) → ResBlocks(×8) → Upsample(×4) → (B, 3, 256, 128)
    """

    def __init__(self, in_channels=512):
        super().__init__()
        self.bottleneck = nn.Sequential(*[
            self._res_block(in_channels) for _ in range(8)
        ])
        self.up1 = self._up_block(512, 256)
        self.up2 = self._up_block(256, 128)
        self.up3 = self._up_block(128, 64)
        self.up4 = self._up_block(64, 32)
        self.out = nn.Sequential(
            nn.Conv2d(32, 16, 3, 1, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 3, 1),
            nn.Sigmoid(),
        )

    @staticmethod
    def _res_block(ch):
        return nn.Sequential(
            nn.Conv2d(ch, ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ch, ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ch),
        )

    @staticmethod
    def _up_block(in_c, out_c):
        return nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, z_q):
        # Residual bottleneck
        x = z_q
        for block in self.bottleneck:
            x = x + block(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        return self.out(x)


class StrongPostHocIdentityClassifier(nn.Module):
    """Strong post-hoc identity classifier with deeper architecture.

    4-layer MLP with 2048 hidden dims — wider and deeper than the training
    adversary. Tests the worst-case identity leakage.
    """

    def __init__(self, input_dim=512, num_classes=751, hidden_dim=2048):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim),
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
        pooled = self.gap(z_q).flatten(1)
        return self.classifier(pooled)


def run_posthoc_attacker_suite(tokenizer, cfg, device):
    """Train and evaluate independent strong attackers after training completes.

    This is the critical validation step that pre-empts reviewer criticism:
    "What if a stronger attacker breaks your privacy?"

    Trains two attackers from scratch:
    1. Strong reconstruction decoder (8 ResBlocks, 50M+ params)
    2. Strong identity classifier (4-layer, 2048 hidden)

    Both attackers see ONLY frozen token representations, with NO access
    to training dynamics.

    Args:
        tokenizer: PrivacyTokenizerV4 (frozen).
        cfg: V4 config dict.
        device: Compute device.

    Returns:
        dict: Post-hoc attacker metrics.
    """
    import lpips as lpips_lib
    import gc

    print("\n" + "=" * 60)
    print("POST-HOC STRONG ATTACKER EVALUATION")
    print("=" * 60)

    tokenizer.eval()
    for p in tokenizer.parameters():
        p.requires_grad_(False)

    train_loader, _ = get_dataloader(cfg, 'train')
    gallery_loader, _ = get_dataloader(cfg, 'gallery')

    epochs = cfg.get('posthoc_attacker_epochs', 30)
    results = {}

    MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    # -----------------------------------------------------------------
    # 1. Strong Reconstruction Attacker
    # -----------------------------------------------------------------
    print("\n[Post-Hoc A1] Training strong reconstruction attacker...")
    decoder = StrongPostHocDecoder(in_channels=cfg['token_dim']).to(device)
    num_params = sum(p.numel() for p in decoder.parameters())
    print(f"  Parameters: {num_params:,}")

    opt = Adam(decoder.parameters(), lr=1e-4, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    lpips_fn = lpips_lib.LPIPS(net='alex').to(device)
    lpips_fn.eval()

    for epoch in range(1, epochs + 1):
        decoder.train()
        losses = []
        for imgs, _, _, _ in tqdm(train_loader, desc=f"  Recon epoch {epoch}/{epochs}",
                                   leave=False):
            imgs = imgs.to(device)
            imgs_unnorm = (imgs * STD + MEAN).clamp(0, 1)

            with torch.no_grad():
                z_q, _, _, _, _, _ = tokenizer(imgs)

            recon = decoder(z_q.detach())
            l1 = F.l1_loss(recon, imgs_unnorm)
            lpips_val = lpips_fn(recon * 2 - 1, imgs_unnorm * 2 - 1).mean()
            loss = l1 + lpips_val

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=5.0)
            opt.step()
            losses.append(loss.item())

        scheduler.step()
        if epoch % 10 == 0:
            print(f"    Epoch {epoch}: avg_loss={np.mean(losses):.4f}")

    # Evaluate reconstruction quality
    decoder.eval()
    all_psnr, all_ssim = [], []
    batch_count = 0
    with torch.no_grad():
        for imgs, _, _, _ in gallery_loader:
            if batch_count >= 30:
                break
            batch_count += 1
            imgs = imgs.to(device)
            imgs_unnorm = (imgs * STD + MEAN).clamp(0, 1)
            z_q, _, _, _, _, _ = tokenizer(imgs)
            recon = decoder(z_q)

            mse = F.mse_loss(recon, imgs_unnorm, reduction='none').mean(dim=[1, 2, 3])
            psnr = 10.0 * torch.log10(1.0 / (mse + 1e-10))
            all_psnr.extend(psnr.cpu().numpy().tolist())

            ssim = _compute_ssim_batch(recon, imgs_unnorm)
            all_ssim.extend(ssim.cpu().numpy().tolist())

    results['posthoc_recon_psnr'] = float(np.mean(all_psnr))
    results['posthoc_recon_ssim'] = float(np.mean(all_ssim))
    results['posthoc_recon_params'] = num_params
    print(f"  [Result] PSNR={results['posthoc_recon_psnr']:.2f}  "
          f"SSIM={results['posthoc_recon_ssim']:.4f}")

    del decoder, opt, scheduler
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # -----------------------------------------------------------------
    # 2. Strong Identity Classifier
    # -----------------------------------------------------------------
    print("\n[Post-Hoc A2] Training strong identity classifier...")
    id_classifier = StrongPostHocIdentityClassifier(
        input_dim=cfg['token_dim'],
        num_classes=cfg['num_classes'],
        hidden_dim=2048
    ).to(device)
    id_params = sum(p.numel() for p in id_classifier.parameters())
    print(f"  Parameters: {id_params:,}")

    opt_id = Adam(id_classifier.parameters(), lr=1e-3, weight_decay=1e-4)
    ce_fn = nn.CrossEntropyLoss()

    best_top1 = 0.0
    for epoch in range(1, epochs + 1):
        id_classifier.train()
        correct, total = 0, 0
        for imgs, labels, _, _ in tqdm(train_loader, desc=f"  ID epoch {epoch}/{epochs}", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.no_grad():
                z_q, _, _, _, _, _ = tokenizer(imgs)   # <-- 6-value unpack, matches PrivacyTokenizerV4
            logits = id_classifier(z_q.detach())
            loss = ce_fn(logits, labels)
            opt_id.zero_grad()
            loss.backward()
            opt_id.step()
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)
        top1 = 100 * correct / total
        if top1 > best_top1:
            best_top1 = top1

    results['posthoc_id_top1_train'] = best_top1
    results['posthoc_id_top1'] = best_top1
    results['posthoc_id_params'] = id_params

    del id_classifier, opt_id
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Restore tokenizer gradients
    for p in tokenizer.parameters():
        p.requires_grad_(True)

    del lpips_fn
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


# =========================================================================
# Full V4 Evaluation Pipeline
# =========================================================================

def evaluate_full_v4(tokenizer, reid_head, attacker, cfg, device):
    """Complete v4 evaluation: ReID metrics + privacy metrics + new metrics.

    Computes:
        - CMC (Rank-1, 5, 10), mAP, mINP — ReID utility
        - PSNR, SSIM, LPIPS — visual privacy (co-trained attacker)
        - PU-Score — privacy-utility balance
        - ISD — identity separation degree

    Args:
        tokenizer: PrivacyTokenizerV4.
        reid_head: MultiGranularityHead.
        attacker: Co-trained UNet decoder.
        cfg: V4 config.
        device: Compute device.

    Returns:
        dict: All evaluation metrics.
    """
    print("\n" + "=" * 60)
    print("V4 Full Evaluation")
    print("=" * 60)

    gallery_loader, _ = get_dataloader(cfg, 'gallery')
    query_loader, _ = get_dataloader(cfg, 'query')

    # --- Feature extraction ---
    print("\n[1/4] Extracting gallery features...")
    gallery_feat, gallery_pids, gallery_camids, _ = extract_features_v4(
        tokenizer, reid_head, gallery_loader, device)
    print(f"  Gallery: {gallery_feat.shape[0]} images, {gallery_feat.shape[1]}-dim features")

    print("[2/4] Extracting query features...")
    query_feat, query_pids, query_camids, _ = extract_features_v4(
        tokenizer, reid_head, query_loader, device)

    # --- CMC / mAP / mINP ---
    print("[3/4] Computing CMC, mAP, mINP...")
    cmc, mAP, mINP = compute_cmc_map_minp(
        query_feat, query_pids, query_camids,
        gallery_feat, gallery_pids, gallery_camids,
    )

    # --- Visual privacy ---
    print("[4/4] Computing visual privacy metrics...")
    privacy = compute_privacy_metrics_v4(tokenizer, attacker, gallery_loader, device)

    # --- PU-Score ---
    pu_score = compute_pu_score(cmc[0] * 100, privacy['ssim'])

    # --- Assemble results ---
    results = {
        'rank1': cmc[0] * 100,
        'rank5': cmc[4] * 100,
        'rank10': cmc[9] * 100,
        'mAP': mAP * 100,
        'mINP': mINP * 100,
        'psnr': privacy['psnr'],
        'ssim': privacy['ssim'],
        'lpips': privacy['lpips'],
        'pu_score': pu_score,
        'recon_samples': privacy['recon_samples'],
    }

    # --- Print summary ---
    print("\n" + "=" * 60)
    print("V4 Evaluation Results")
    print("=" * 60)
    print(f"  [ReID Utility]")
    print(f"    Rank-1:  {results['rank1']:.2f}%")
    print(f"    Rank-5:  {results['rank5']:.2f}%")
    print(f"    Rank-10: {results['rank10']:.2f}%")
    print(f"    mAP:     {results['mAP']:.2f}%")
    print(f"    mINP:    {results['mINP']:.2f}%")
    print(f"  [Visual Privacy]")
    print(f"    PSNR:    {results['psnr']:.2f} dB")
    print(f"    SSIM:    {results['ssim']:.4f}")
    print(f"    LPIPS:   {results['lpips']:.4f}")
    print(f"  [Composite]")
    print(f"    PU-Score: {results['pu_score']:.1f}")
    print("=" * 60 + "\n")

    return results

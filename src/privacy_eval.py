"""
Privacy Evaluation Suite for PrivToken-ReID.

Comprehensive identity leakage evaluation — the single most important
upgrade for publication readiness. Goes beyond reconstruction quality
(SSIM/LPIPS/PSNR) to measure actual identity leakage.

Provides:
    P1. Attacker ReID Accuracy   — Can identity be recovered from reconstructions?
    P2. Token Identity Classifier — Can identity be inferred directly from tokens?
    P3. Attribute Leakage         — Can attributes be inferred from reconstructions?
    P4. Face Similarity           — Are face embeddings preserved in reconstructions?

Usage:
    from src.privacy_eval import evaluate_privacy_full
    results = evaluate_privacy_full(tokenizer, attacker, cfg, device)
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm
from collections import defaultdict

from torchvision.models import resnet50, ResNet50_Weights
from src.datasets import get_dataloader


# =========================================================================
# P1. Attacker ReID Accuracy
# =========================================================================

class ReIDEvaluatorModel(nn.Module):
    """
    Independent frozen ReID model for evaluating identity leakage.

    This is a SEPARATE model trained on original Market-1501 images.
    We feed RECONSTRUCTED images through it and measure if identity can
    still be recovered. If Rank-1 drops drastically, privacy is working.

    Architecture:
        ResNet-50 (pretrained) → GAP → BN → feature (D=512)

    Usage:
        1. Train on original images: train_reid_evaluator()
        2. Extract features from reconstructions
        3. Compare Rank-1/mAP vs. features from originals
    """

    def __init__(self, num_classes=751, feat_dim=512):
        super().__init__()
        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        backbone.layer4[0].conv2.stride = (1, 1)
        backbone.layer4[0].downsample[0].stride = (1, 1)
        self.encoder = nn.Sequential(*list(backbone.children())[:-2])

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(2048, feat_dim, bias=False)
        self.bn = nn.BatchNorm1d(feat_dim)
        self.bn.bias.requires_grad_(False)
        self.classifier = nn.Linear(feat_dim, num_classes, bias=False)
        nn.init.normal_(self.classifier.weight, std=0.001)

    def forward(self, x):
        """
        Args:
            x (Tensor): Images (B, 3, H, W) — can be normalized or [0,1].

        Returns:
            feat (Tensor): BN-normalized features (B, feat_dim).
            logits (Tensor): Class logits (B, num_classes).
        """
        enc = self.encoder(x)                # (B, 2048, 16, 8)
        pooled = self.gap(enc).flatten(1)     # (B, 2048)
        projected = self.proj(pooled)          # (B, feat_dim)
        feat = self.bn(projected)              # (B, feat_dim)
        logits = self.classifier(feat)         # (B, num_classes)
        return feat, logits

    def extract_features(self, x):
        """Extract features only (no logits)."""
        feat, _ = self.forward(x)
        return feat


def train_reid_evaluator(cfg, device, epochs=30, lr=0.01):
    """
    Train an independent ReID model on original Market-1501 images.

    This model is used as a frozen evaluator to measure identity leakage.
    Trained on the TRAINING split of Market-1501 with standard augmentation.

    Args:
        cfg (dict): Config with data_root, num_classes, etc.
        device (torch.device): Compute device.
        epochs (int): Training epochs.
        lr (float): Learning rate.

    Returns:
        ReIDEvaluatorModel: Trained model (eval mode, frozen).
    """
    save_path = os.path.join(cfg.get('output_dir', 'outputs'), 'reid_evaluator.pth')

    # Check if already trained
    if os.path.isfile(save_path):
        print(f"[ReID Evaluator] Loading from {save_path}")
        model = ReIDEvaluatorModel(num_classes=cfg['num_classes']).to(device)
        state = torch.load(save_path, map_location=device, weights_only=False)
        model.load_state_dict(state['model'])
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        print(f"  Loaded (trained for {state.get('epochs', '?')} epochs, "
              f"train_acc={state.get('final_acc', '?'):.1f}%)")
        return model

    print("[ReID Evaluator] Training independent ReID model on original images...")

    model = ReIDEvaluatorModel(num_classes=cfg['num_classes']).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                                weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                      milestones=[15, 25],
                                                      gamma=0.1)
    ce_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    train_loader, _ = get_dataloader(cfg, 'train')

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        correct, total = 0, 0
        epoch_loss = []

        pbar = tqdm(train_loader, desc=f"[ReID Eval Model] Epoch {epoch}/{epochs}")
        for imgs, labels, _, _ in pbar:
            imgs, labels = imgs.to(device), labels.to(device)

            feat, logits = model(imgs)
            loss = ce_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            epoch_loss.append(loss.item())

            pbar.set_postfix({'loss': f"{loss.item():.3f}",
                              'acc': f"{100*correct/total:.1f}%"})

        scheduler.step()
        acc = 100 * correct / total
        print(f"  Epoch {epoch}: loss={np.mean(epoch_loss):.4f}, acc={acc:.1f}%")

        if acc > best_acc:
            best_acc = acc

    # Save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        'model': model.state_dict(),
        'epochs': epochs,
        'final_acc': best_acc,
    }, save_path)
    print(f"  -> Saved to {save_path} (best acc: {best_acc:.1f}%)")

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def evaluate_reconstruction_reid(tokenizer, attacker, reid_evaluator, cfg, device,
                                  num_batches=50):
    """
    P1: Evaluate identity leakage by running a frozen ReID model on reconstructions.

    This is the single most important privacy metric. If the ReID model can still
    identify people from the reconstructed images, the privacy claim is weak.

    Args:
        tokenizer: PrivacyTokenizer (eval mode).
        attacker: Decoder that reconstructs images from tokens.
        reid_evaluator: Frozen independent ReID model.
        cfg: Config dict.
        device: Compute device.
        num_batches: Max batches to evaluate.

    Returns:
        dict with:
            'recon_rank1': Rank-1 on reconstructed images.
            'recon_mAP': mAP on reconstructed images.
            'orig_rank1': Rank-1 on original images (reference).
            'orig_mAP': mAP on original images (reference).
            'identity_leakage_ratio': recon_rank1 / orig_rank1 (lower = better privacy).
    """
    from src.evaluate import compute_cmc_map

    tokenizer.eval()
    attacker.eval()
    reid_evaluator.eval()

    MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    gallery_loader, _ = get_dataloader(cfg, 'gallery')
    query_loader, _ = get_dataloader(cfg, 'query')

    def _extract_both(loader, max_batches):
        """Extract features from both originals and reconstructions."""
        orig_feats, recon_feats = [], []
        all_pids, all_camids = [], []
        batch_count = 0

        with torch.no_grad():
            for imgs, pids, camids, _ in tqdm(loader, desc="Extracting orig+recon"):
                if batch_count >= max_batches:
                    break
                batch_count += 1

                imgs = imgs.to(device)
                imgs_unnorm = (imgs * STD + MEAN).clamp(0, 1)

                # Features from original images
                # Normalize for evaluator (it expects ImageNet-normalized input)
                orig_feat = reid_evaluator.extract_features(imgs)
                orig_feats.append(orig_feat.cpu().numpy())

                # Reconstruct from tokens
                z_q, _, _, _ = tokenizer(imgs)
                recon = attacker(z_q)  # (B, 3, H, W) in [0, 1]

                # Normalize reconstruction for evaluator
                recon_norm = (recon - MEAN) / STD
                recon_feat = reid_evaluator.extract_features(recon_norm)
                recon_feats.append(recon_feat.cpu().numpy())

                all_pids.append(pids.numpy() if isinstance(pids, torch.Tensor)
                                else np.array(pids))
                all_camids.append(camids.numpy() if isinstance(camids, torch.Tensor)
                                  else np.array(camids))

        return (np.concatenate(orig_feats),
                np.concatenate(recon_feats),
                np.concatenate(all_pids),
                np.concatenate(all_camids))

    print("\n[P1] Evaluating reconstruction identity leakage...")
    print("  Extracting gallery features (original + reconstructed)...")
    g_orig, g_recon, g_pids, g_camids = _extract_both(gallery_loader, num_batches)

    print("  Extracting query features (original + reconstructed)...")
    q_orig, q_recon, q_pids, q_camids = _extract_both(query_loader, num_batches)

    # CMC/mAP on originals (reference)
    print("  Computing CMC/mAP on original images...")
    cmc_orig, mAP_orig = compute_cmc_map(q_orig, q_pids, q_camids,
                                          g_orig, g_pids, g_camids)

    # CMC/mAP on reconstructions
    print("  Computing CMC/mAP on reconstructed images...")
    cmc_recon, mAP_recon = compute_cmc_map(q_recon, q_pids, q_camids,
                                            g_recon, g_pids, g_camids)

    orig_rank1 = cmc_orig[0] * 100
    recon_rank1 = cmc_recon[0] * 100
    orig_mAP = mAP_orig * 100
    recon_mAP = mAP_recon * 100
    leakage_ratio = recon_rank1 / max(orig_rank1, 1e-6)

    print(f"\n  [P1] Reconstruction ReID Results:")
    print(f"    Original:       Rank-1={orig_rank1:.2f}%  mAP={orig_mAP:.2f}%")
    print(f"    Reconstructed:  Rank-1={recon_rank1:.2f}%  mAP={recon_mAP:.2f}%")
    print(f"    Identity Leakage Ratio: {leakage_ratio:.3f} "
          f"({'[OK] Good' if leakage_ratio < 0.5 else '[WARNING] High leakage'})")

    return {
        'orig_rank1': orig_rank1,
        'orig_mAP': orig_mAP,
        'recon_rank1': recon_rank1,
        'recon_mAP': recon_mAP,
        'identity_leakage_ratio': leakage_ratio,
    }


# =========================================================================
# P2. Token Identity Classifier
# =========================================================================

class TokenIdentityClassifier(nn.Module):
    """
    Simple classifier that predicts person ID directly from token representations.

    If this classifier achieves high accuracy, the tokens are leaking identity.
    Tests: "Can identity be inferred from the discrete representation without
    any visual reconstruction?"

    Architecture:
        Token grid (B, D, H, W) → GAP → MLP → person ID
    """

    def __init__(self, token_dim=512, num_classes=751, hidden_dim=256):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(token_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, z_q):
        """
        Args:
            z_q (Tensor): Quantized token grid (B, D, H, W).
        Returns:
            logits (Tensor): Class logits (B, num_classes).
        """
        pooled = self.gap(z_q).flatten(1)  # (B, D)
        return self.classifier(pooled)


def evaluate_token_identity_leakage(tokenizer, cfg, device, epochs=15):
    """
    P2: Train a classifier to predict person ID from tokens.

    Trains a small MLP on the quantized token representations. If it achieves
    high accuracy, the tokens preserve too much identity information.

    Args:
        tokenizer: PrivacyTokenizer (frozen during this evaluation).
        cfg: Config dict.
        device: Compute device.
        epochs: Training epochs for the classifier.

    Returns:
        dict with:
            'token_id_top1': Top-1 accuracy of predicting person ID from tokens.
            'token_id_top5': Top-5 accuracy.
    """
    print("\n[P2] Evaluating token identity leakage...")
    print("  Training identity classifier on quantized tokens...")

    tokenizer.eval()
    for p in tokenizer.parameters():
        p.requires_grad_(False)

    classifier = TokenIdentityClassifier(
        token_dim=cfg['token_dim'],
        num_classes=cfg['num_classes'],
    ).to(device)

    optimizer = Adam(classifier.parameters(), lr=1e-3, weight_decay=1e-4)
    ce_fn = nn.CrossEntropyLoss()

    train_loader, _ = get_dataloader(cfg, 'train')

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        classifier.train()
        correct, correct5, total = 0, 0, 0
        epoch_loss = []

        for imgs, labels, _, _ in tqdm(train_loader,
                                        desc=f"[Token ID] Epoch {epoch}/{epochs}"):
            imgs, labels = imgs.to(device), labels.to(device)

            with torch.no_grad():
                z_q, _, _, _ = tokenizer(imgs)

            logits = classifier(z_q.detach())
            loss = ce_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Top-1 accuracy
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()

            # Top-5 accuracy
            _, top5_preds = logits.topk(5, dim=1)
            correct5 += (top5_preds == labels.unsqueeze(1)).any(dim=1).sum().item()

            total += labels.size(0)
            epoch_loss.append(loss.item())

        top1 = 100 * correct / total
        top5 = 100 * correct5 / total
        print(f"  Epoch {epoch}: loss={np.mean(epoch_loss):.4f}, "
              f"top1={top1:.1f}%, top5={top5:.1f}%")

        if top1 > best_acc:
            best_acc = top1

    # Re-enable tokenizer gradients
    for p in tokenizer.parameters():
        p.requires_grad_(True)

    print(f"\n  [P2] Token Identity Leakage Results:")
    print(f"    Token ID Top-1: {best_acc:.1f}%")
    print(f"    Token ID Top-5: {top5:.1f}%")
    print(f"    Assessment: {'[WARNING] High leakage - tokens preserve identity' if best_acc > 60 else '[OK] Low leakage - tokens scrub identity' if best_acc < 30 else '~ Moderate leakage'}")

    return {
        'token_id_top1': best_acc,
        'token_id_top5': top5,
    }


# =========================================================================
# P3. Attribute Leakage
# =========================================================================

class AttributeClassifier(nn.Module):
    """
    Multi-attribute classifier for privacy leakage analysis.

    Predicts binary/categorical attributes from images:
    - gender (binary)
    - upper clothing color (8 classes)
    - lower clothing color (8 classes)
    - has_backpack (binary)
    - has_hat (binary)

    Uses a lightweight ResNet-18 backbone for efficiency.
    """

    def __init__(self):
        super().__init__()
        from torchvision.models import resnet18, ResNet18_Weights
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(backbone.children())[:-1])  # → (B, 512, 1, 1)

        # Attribute heads
        self.head_gender = nn.Linear(512, 2)
        self.head_upper_color = nn.Linear(512, 8)
        self.head_lower_color = nn.Linear(512, 8)
        self.head_backpack = nn.Linear(512, 2)
        self.head_hat = nn.Linear(512, 2)

    def forward(self, x):
        """
        Args:
            x (Tensor): Images (B, 3, H, W).
        Returns:
            dict of attribute logits.
        """
        feat = self.features(x).flatten(1)  # (B, 512)
        return {
            'gender': self.head_gender(feat),
            'upper_color': self.head_upper_color(feat),
            'lower_color': self.head_lower_color(feat),
            'backpack': self.head_backpack(feat),
            'hat': self.head_hat(feat),
        }


def generate_pseudo_attributes(images, device):
    """
    Generate pseudo attribute labels using visual heuristics.

    Since Market-1501 doesn't have ground-truth attributes readily available
    as tensors, we use simple color-based heuristics for a first pass.
    For a full paper, use the Market-1501 attribute annotations dataset.

    Args:
        images (Tensor): Images in [0,1], shape (B, 3, H, W).

    Returns:
        dict of pseudo attribute tensors.
    """
    B, C, H, W = images.shape

    # Upper body region: top 40% of image
    upper = images[:, :, :int(H * 0.4), :]
    # Lower body region: bottom 40% of image
    lower = images[:, :, int(H * 0.6):, :]

    # Dominant color channel as pseudo color label (0-7 by quantizing hue)
    upper_mean = upper.mean(dim=[2, 3])  # (B, 3)
    lower_mean = lower.mean(dim=[2, 3])  # (B, 3)

    # Simple color quantization: bin by dominant channel ratios
    upper_color = (upper_mean[:, 0] * 3 + upper_mean[:, 1] * 2 + upper_mean[:, 2]).long() % 8
    lower_color = (lower_mean[:, 0] * 2 + lower_mean[:, 1] * 3 + lower_mean[:, 2]).long() % 8

    # Brightness as pseudo gender proxy (very rough — just for metric structure)
    brightness = images.mean(dim=[1, 2, 3])
    gender = (brightness > brightness.median()).long()

    # Top region brightness variance as hat proxy
    head = images[:, :, :int(H * 0.15), :]
    head_var = head.var(dim=[1, 2, 3])
    has_hat = (head_var > head_var.median()).long()

    # Back region as backpack proxy
    back_region = images[:, :, int(H * 0.2):int(H * 0.5), int(W * 0.3):int(W * 0.7)]
    back_intensity = back_region.mean(dim=[1, 2, 3])
    has_backpack = (back_intensity > back_intensity.median()).long()

    return {
        'gender': gender.to(device),
        'upper_color': upper_color.to(device),
        'lower_color': lower_color.to(device),
        'backpack': has_backpack.to(device),
        'hat': has_hat.to(device),
    }


def evaluate_attribute_leakage(tokenizer, attacker, cfg, device, num_batches=30):
    """
    P3: Compare attribute classification accuracy on originals vs reconstructions.

    If attributes are perfectly preserved in reconstructions, privacy is weaker.
    A large accuracy drop on reconstructions indicates attribute information is scrubbed.

    Args:
        tokenizer: PrivacyTokenizer.
        attacker: Decoder.
        cfg: Config dict.
        device: Compute device.
        num_batches: Batches to evaluate.

    Returns:
        dict with per-attribute accuracy on originals and reconstructions.
    """
    print("\n[P3] Evaluating attribute leakage...")

    tokenizer.eval()
    attacker.eval()

    MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    # Train a quick attribute classifier on originals
    attr_model = AttributeClassifier().to(device)
    optimizer = Adam(attr_model.parameters(), lr=1e-3)

    train_loader, _ = get_dataloader(cfg, 'train')

    print("  Training attribute classifier on original images (5 epochs)...")
    for epoch in range(1, 6):
        attr_model.train()
        for batch_idx, (imgs, _, _, _) in enumerate(train_loader):
            if batch_idx >= 50:  # quick training
                break
            imgs = imgs.to(device)
            imgs_unnorm = (imgs * STD + MEAN).clamp(0, 1)
            attrs = generate_pseudo_attributes(imgs_unnorm, device)
            preds = attr_model(imgs)

            loss = sum(F.cross_entropy(preds[k], attrs[k]) for k in attrs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Evaluate on test set
    attr_model.eval()
    gallery_loader, _ = get_dataloader(cfg, 'gallery')

    attr_names = ['gender', 'upper_color', 'lower_color', 'backpack', 'hat']
    orig_correct = {k: 0 for k in attr_names}
    recon_correct = {k: 0 for k in attr_names}
    total_samples = 0

    batch_count = 0
    with torch.no_grad():
        for imgs, _, _, _ in tqdm(gallery_loader, desc="Attribute eval"):
            if batch_count >= num_batches:
                break
            batch_count += 1

            imgs = imgs.to(device)
            imgs_unnorm = (imgs * STD + MEAN).clamp(0, 1)
            attrs = generate_pseudo_attributes(imgs_unnorm, device)

            # Predict on originals
            orig_preds = attr_model(imgs)

            # Predict on reconstructions
            z_q, _, _, _ = tokenizer(imgs)
            recon = attacker(z_q)
            recon_norm = (recon - MEAN) / STD
            recon_preds = attr_model(recon_norm)

            for k in attr_names:
                orig_correct[k] += (orig_preds[k].argmax(1) == attrs[k]).sum().item()
                recon_correct[k] += (recon_preds[k].argmax(1) == attrs[k]).sum().item()

            total_samples += imgs.size(0)

    results = {}
    print(f"\n  [P3] Attribute Leakage Results (N={total_samples}):")
    print(f"  {'Attribute':<15} {'Original':>10} {'Recon':>10} {'Drop':>10}")
    print(f"  {'-'*45}")

    for k in attr_names:
        orig_acc = 100 * orig_correct[k] / total_samples
        recon_acc = 100 * recon_correct[k] / total_samples
        drop = orig_acc - recon_acc
        results[f'attr_{k}_orig'] = orig_acc
        results[f'attr_{k}_recon'] = recon_acc
        results[f'attr_{k}_drop'] = drop
        print(f"  {k:<15} {orig_acc:>9.1f}% {recon_acc:>9.1f}% {drop:>+9.1f}%")

    avg_drop = np.mean([results[f'attr_{k}_drop'] for k in attr_names])
    results['avg_attr_drop'] = avg_drop
    print(f"  {'Average':<15} {'':>10} {'':>10} {avg_drop:>+9.1f}%")
    print(f"  Assessment: {'[OK] Good - attributes scrubbed' if avg_drop > 10 else '[WARNING] Attributes still recoverable'}")

    return results


# =========================================================================
# P4. Face Similarity
# =========================================================================

def evaluate_face_leakage(tokenizer, attacker, cfg, device, num_batches=20):
    """
    P4: Measure face detection rate and embedding similarity on reconstructions.

    Uses MTCNN for face detection and InceptionResnetV1 for face embeddings
    (from facenet-pytorch). If face detection fails on reconstructions but
    succeeds on originals, that's strong evidence of visual privacy.

    Args:
        tokenizer: PrivacyTokenizer.
        attacker: Decoder.
        cfg: Config dict.
        device: Compute device.
        num_batches: Batches to evaluate.

    Returns:
        dict with face detection rates and embedding similarity metrics.
    """
    try:
        from facenet_pytorch import MTCNN, InceptionResnetV1
    except ImportError:
        print("\n[P4] facenet-pytorch not installed — skipping face leakage eval.")
        print("     Install with: pip install facenet-pytorch")
        return {
            'face_detect_orig': -1,
            'face_detect_recon': -1,
            'face_similarity': -1,
            'face_detect_drop': -1,
            'note': 'facenet-pytorch not installed',
        }

    print("\n[P4] Evaluating face leakage...")

    tokenizer.eval()
    attacker.eval()

    MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    # Face detection and embedding
    mtcnn = MTCNN(keep_all=False, device=device, min_face_size=20)
    facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    gallery_loader, _ = get_dataloader(cfg, 'gallery')

    orig_detected = 0
    recon_detected = 0
    total = 0
    cos_similarities = []

    batch_count = 0
    with torch.no_grad():
        for imgs, _, _, _ in tqdm(gallery_loader, desc="Face eval"):
            if batch_count >= num_batches:
                break
            batch_count += 1

            imgs = imgs.to(device)
            imgs_unnorm = (imgs * STD + MEAN).clamp(0, 1)

            z_q, _, _, _ = tokenizer(imgs)
            recon = attacker(z_q)

            for i in range(imgs.size(0)):
                total += 1

                # Convert to PIL for MTCNN
                orig_pil = _tensor_to_pil(imgs_unnorm[i])
                recon_pil = _tensor_to_pil(recon[i])

                # Detect faces
                orig_face = mtcnn(orig_pil)
                recon_face = mtcnn(recon_pil)

                orig_has_face = orig_face is not None
                recon_has_face = recon_face is not None

                if orig_has_face:
                    orig_detected += 1
                if recon_has_face:
                    recon_detected += 1

                # If both have faces, compare embeddings
                if orig_has_face and recon_has_face:
                    orig_emb = facenet(orig_face.unsqueeze(0).to(device))
                    recon_emb = facenet(recon_face.unsqueeze(0).to(device))
                    cos_sim = F.cosine_similarity(orig_emb, recon_emb).item()
                    cos_similarities.append(cos_sim)

    orig_rate = 100 * orig_detected / max(total, 1)
    recon_rate = 100 * recon_detected / max(total, 1)
    detect_drop = orig_rate - recon_rate
    avg_sim = float(np.mean(cos_similarities)) if cos_similarities else 0.0

    print(f"\n  [P4] Face Leakage Results (N={total}):")
    print(f"    Face detection - Original: {orig_rate:.1f}% | Recon: {recon_rate:.1f}% "
          f"| Drop: {detect_drop:+.1f}%")
    print(f"    Face embedding similarity (when both detected): {avg_sim:.3f}")
    print(f"    Assessment: {'[OK] Faces destroyed' if detect_drop > 30 else '[WARNING] Faces partially preserved'}")

    return {
        'face_detect_orig': orig_rate,
        'face_detect_recon': recon_rate,
        'face_detect_drop': detect_drop,
        'face_similarity': avg_sim,
        'num_face_pairs': len(cos_similarities),
    }


def _tensor_to_pil(tensor):
    """Convert a (3, H, W) tensor in [0,1] to PIL Image."""
    from PIL import Image
    arr = (tensor.cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


# =========================================================================
# Full Privacy Evaluation
# =========================================================================

def evaluate_privacy_full(tokenizer, attacker, cfg, device, skip_face=False):
    """
    Run the complete privacy evaluation suite.

    Args:
        tokenizer: PrivacyTokenizer.
        attacker: Decoder.
        cfg: Config dict.
        device: Compute device.
        skip_face: Skip face detection (if facenet-pytorch not available).

    Returns:
        dict: Combined results from all privacy evaluations.
    """
    print("\n" + "=" * 60)
    print("PRIVACY EVALUATION SUITE")
    print("=" * 60)

    results = {}

    # P1: Reconstruction ReID
    reid_evaluator = train_reid_evaluator(cfg, device)
    p1 = evaluate_reconstruction_reid(tokenizer, attacker, reid_evaluator, cfg, device)
    results.update(p1)

    # P2: Token Identity Classifier
    p2 = evaluate_token_identity_leakage(tokenizer, cfg, device)
    results.update(p2)

    # P3: Attribute Leakage
    p3 = evaluate_attribute_leakage(tokenizer, attacker, cfg, device)
    results.update(p3)

    # P4: Face Similarity
    if not skip_face:
        p4 = evaluate_face_leakage(tokenizer, attacker, cfg, device)
        results.update(p4)

    # Summary
    print("\n" + "=" * 60)
    print("PRIVACY EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  P1 - Reconstruction ReID:")
    print(f"        Original:  Rank-1={results.get('orig_rank1', 0):.1f}%  "
          f"mAP={results.get('orig_mAP', 0):.1f}%")
    print(f"        Recon:     Rank-1={results.get('recon_rank1', 0):.1f}%  "
          f"mAP={results.get('recon_mAP', 0):.1f}%")
    print(f"        Leakage:   {results.get('identity_leakage_ratio', 0):.3f}")
    print(f"  P2 - Token ID Classifier:")
    print(f"        Top-1: {results.get('token_id_top1', 0):.1f}%  "
          f"Top-5: {results.get('token_id_top5', 0):.1f}%")
    print(f"  P3 - Attribute Leakage:")
    print(f"        Avg accuracy drop: {results.get('avg_attr_drop', 0):+.1f}%")

    if not skip_face and results.get('face_detect_orig', -1) >= 0:
        print(f"  P4 - Face Detection:")
        print(f"        Detection drop: {results.get('face_detect_drop', 0):+.1f}%")
        print(f"        Embedding similarity: {results.get('face_similarity', 0):.3f}")

    print("=" * 60 + "\n")

    return results

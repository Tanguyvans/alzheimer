#!/usr/bin/env python3
"""
ViT attention rollout for 3D MRI interpretability.

Loads each paper-grade fold checkpoint, monkey-patches the Attention blocks
to capture self-attention, runs forward on selected well-classified cases
(CN and AD), computes attention rollout (Abnar & Zuidema 2020), reshapes
the CLS->patch attention to an 8x8x8 grid, upsamples to 128x128x128, and
saves 3D heatmaps + 2D overlays.

Usage:
    python vit_attention.py --fold 0 [--n-per-class 4]
"""

import argparse
import pickle
import sys
import types
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))
from dataset import MultiModalDataset  # noqa: E402
from interpretability.ft_attention import build_paper_model, TABULAR_FEATURES  # noqa: E402

REPO = Path("/home/tanguy/medical/alzheimer")
EXP = REPO / "experiments" / "multimodal_fusion"


def patched_attention_forward(self, x):
    """Attention.forward that also stores the softmax-attention matrix."""
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
    qkv = qkv.permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    attn = (q @ k.transpose(-2, -1)) * self.scale
    attn = attn.softmax(dim=-1)
    self.last_attn = attn.detach()  # (B, heads, N, N)
    attn_d = self.attn_drop(attn)
    x = (attn_d @ v).transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


def install_attention_hooks(model):
    """Monkey-patch ViT attention blocks to save attention matrices."""
    for blk in model.mri_backbone.blocks:
        blk.attn.forward = types.MethodType(patched_attention_forward, blk.attn)


def attention_rollout(attns_per_layer):
    """
    attns_per_layer: list of (B, heads, N, N) tensors
    Returns: (B, N) attention rollout from CLS token to each patch.
    """
    B, H, N, _ = attns_per_layer[0].shape
    # Average over heads
    avg = [a.mean(dim=1) for a in attns_per_layer]  # list of (B, N, N)
    I = torch.eye(N, device=avg[0].device).unsqueeze(0)  # (1, N, N)
    rolled = I.expand(B, N, N).clone()
    for A in avg:
        A_ = (A + I) / 2.0
        A_ = A_ / A_.sum(dim=-1, keepdim=True)
        rolled = A_ @ rolled
    # CLS token is index 0; patches are 1..N-1
    cls_to_patches = rolled[:, 0, 1:]  # (B, N-1)
    return cls_to_patches


def regenerate_fold_splits(seed, fold, n_folds=5):
    train_df = pd.read_csv(EXP / "data/combined_trajectory/train.csv")
    val_df = pd.read_csv(EXP / "data/combined_trajectory/val.csv")
    test_df = pd.read_csv(EXP / "data/combined_trajectory/test.csv")
    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    labels = all_df["label"].values
    indices = np.arange(len(all_df))
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fi, (trn_idx, tst_idx) in enumerate(skf.split(indices, labels)):
        if fi == fold:
            np.random.seed(seed + fold)
            perm = np.random.permutation(len(trn_idx))
            vs = int(0.1 * len(trn_idx))
            train_local = trn_idx[perm[vs:]]
            return (
                all_df.iloc[train_local].reset_index(drop=True),
                all_df.iloc[tst_idx].reset_index(drop=True),
            )
    raise ValueError(f"fold {fold} not found")


def select_well_classified(model, loader, device, n_per_class=4):
    """Run forward, pick top-confidence CN and AD with correct predictions."""
    model.eval()
    all_probs, all_preds, all_labels, all_idx = [], [], [], []
    with torch.no_grad():
        for b_idx, (mri, tab, y) in enumerate(loader):
            mri = mri.to(device, non_blocking=True)
            tab = tab.to(device, non_blocking=True)
            out = model(mri, tab)
            logits = out[0] if isinstance(out, tuple) else out
            probs = torch.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1)
            all_probs.append(probs.cpu())
            all_preds.append(preds.cpu())
            all_labels.append(y)
    probs = torch.cat(all_probs)
    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    correct = preds == labels
    # CN: label 0, correctly predicted, sorted by prob of class 0 (descending)
    cn_mask = correct & (labels == 0)
    ad_mask = correct & (labels == 1)
    cn_scores = probs[cn_mask, 0]
    ad_scores = probs[ad_mask, 1]
    cn_indices = torch.where(cn_mask)[0]
    ad_indices = torch.where(ad_mask)[0]
    cn_top = cn_indices[torch.argsort(cn_scores, descending=True)[:n_per_class]]
    ad_top = ad_indices[torch.argsort(ad_scores, descending=True)[:n_per_class]]
    return cn_top.tolist(), ad_top.tolist(), probs


def run_fold(fold, seed, n_per_class, device, out_dir):
    ckpt_path = EXP / "cv_results" / f"seed_{seed}" / f"fold_{fold}" / "model.pth"
    scaler_path = EXP / "cv_results" / f"seed_{seed}" / f"fold_{fold}" / "scaler.pkl"
    print(f"\n===== FOLD {fold} =====")
    print(f"  ckpt: {ckpt_path}")

    _, test_df = regenerate_fold_splits(seed, fold)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    # Build test dataset with pre-fit scaler
    test_csv = out_dir / f"_tmp_test_fold{fold}.csv"
    test_df.to_csv(test_csv, index=False)
    test_ds = MultiModalDataset(
        str(test_csv),
        tabular_features=TABULAR_FEATURES,
        target_shape=(128, 128, 128),
        augment=False,
        normalize_tabular=True,
        scaler=scaler,
        use_paper_preprocessing=True,
        target_spacing=1.75,
    )
    loader = DataLoader(test_ds, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)

    model = build_paper_model()
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device).eval()
    install_attention_hooks(model)

    # Pick well-classified cases
    cn_idx, ad_idx, probs = select_well_classified(model, loader, device, n_per_class)
    print(f"  CN cases (top conf): {cn_idx}  probs={[f'{probs[i,0]:.3f}' for i in cn_idx]}")
    print(f"  AD cases (top conf): {ad_idx}  probs={[f'{probs[i,1]:.3f}' for i in ad_idx]}")

    # Forward each selected case and extract rollout
    heatmaps = {}  # key: (class, sample_idx) -> (8,8,8) numpy
    for cls_name, indices in [("CN", cn_idx), ("AD", ad_idx)]:
        for s_idx in indices:
            mri, tab, y = test_ds[s_idx]
            mri = mri.unsqueeze(0).to(device)
            tab = tab.unsqueeze(0).to(device)
            with torch.no_grad():
                _ = model(mri, tab)
            # Collect attention from all 12 ViT blocks
            attns = [blk.attn.last_attn for blk in model.mri_backbone.blocks]
            roll = attention_rollout(attns)  # (1, 512)
            roll_np = roll[0].cpu().numpy()  # (512,)
            # 512 patches = 8x8x8 grid (for 128^3 volume with patch_size=16)
            grid = roll_np.reshape(8, 8, 8)
            heatmaps[(cls_name, s_idx)] = {
                "grid": grid,
                "subject_id": test_df.iloc[s_idx]["subject_id"],
                "scan_path": test_df.iloc[s_idx]["scan_path"],
                "prob_cls": float(probs[s_idx, 0 if cls_name == "CN" else 1]),
            }

    # Save per-class aggregate heatmaps + per-case data
    for cls_name in ("CN", "AD"):
        grids = np.stack([
            heatmaps[k]["grid"] for k in heatmaps if k[0] == cls_name
        ])  # (n_per_class, 8, 8, 8)
        mean_grid = grids.mean(axis=0)
        np.save(out_dir / f"vit_roll_fold{fold}_{cls_name}_cases.npy", grids)
        np.save(out_dir / f"vit_roll_fold{fold}_{cls_name}_mean.npy", mean_grid)
        print(f"  saved fold{fold} {cls_name}: {grids.shape} cases, mean={mean_grid.mean():.4f}")

    # Cleanup temp csv
    test_csv.unlink()
    return heatmaps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--n-per-class", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Device: {device}")

    out_dir = SCRIPT_DIR / "results_new"
    out_dir.mkdir(exist_ok=True, parents=True)

    for fold in args.folds:
        run_fold(fold, args.seed, args.n_per_class, device, out_dir)

    print("\n[*] Done. Heatmaps saved as vit_roll_foldX_{CN,AD}_{cases,mean}.npy")


if __name__ == "__main__":
    main()

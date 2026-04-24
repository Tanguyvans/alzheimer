#!/usr/bin/env python3
"""
Visualize 3D ViT attention heatmaps aggregated across 5 folds.

Loads vit_roll_foldX_{CN,AD}_mean.npy (8x8x8 grids), aggregates across folds,
upsamples to 128x128x128, loads a sample brain volume for background, and
produces multi-slice overlay figures (axial, sagittal, coronal).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from matplotlib.colors import LinearSegmentedColormap

SCRIPT_DIR = Path(__file__).resolve().parent
EXP = SCRIPT_DIR.parent
RESULTS = SCRIPT_DIR / "results_new"


def load_fold_means():
    """Return (cn_all, ad_all): each (n_folds, 8, 8, 8)."""
    cn_all, ad_all = [], []
    for fold in range(5):
        cn = np.load(RESULTS / f"vit_roll_fold{fold}_CN_mean.npy")
        ad = np.load(RESULTS / f"vit_roll_fold{fold}_AD_mean.npy")
        cn_all.append(cn)
        ad_all.append(ad)
    return np.stack(cn_all), np.stack(ad_all)


def upsample_to_128(grid):
    """(n, 8, 8, 8) -> (n, 128, 128, 128) via trilinear interp."""
    t = torch.from_numpy(grid).float().unsqueeze(1)  # (n, 1, 8, 8, 8)
    t = F.interpolate(t, size=(128, 128, 128), mode="trilinear", align_corners=False)
    return t.squeeze(1).numpy()


def normalize(vol, percentile=(1, 99)):
    lo, hi = np.percentile(vol, percentile)
    v = np.clip((vol - lo) / (hi - lo + 1e-6), 0, 1)
    return v


def load_sample_brain():
    """Load a representative CN brain (first CN case from fold 0 test)."""
    # Use the actual subject we used for fold 0 CN (index 182 in fold 0 test)
    # Simpler: just use any subject from combined_trajectory test.csv
    test_df = pd.read_csv(EXP / "data/combined_trajectory/test.csv")
    cn_row = test_df[test_df["label"] == 0].iloc[10]  # pick a CN subject
    path = cn_row["scan_path"]
    print(f"  background brain: {path}")
    img = nib.load(path)
    vol = img.get_fdata().astype(np.float32)
    # Resample/crop to 128^3 (same preproc as dataset uses)
    # For simplicity, use the raw loaded volume — approximate
    from scipy.ndimage import zoom
    zf = [128 / s for s in vol.shape]
    vol128 = zoom(vol, zf, order=1)
    return normalize(vol128)


def plot_slices(background, heatmap, title, out_path):
    """Multi-slice overlay. heatmap is (128,128,128), already normalized."""
    fig, axes = plt.subplots(2, 5, figsize=(18, 7))

    # Axial slices (z axis)
    z_slices = [40, 55, 65, 75, 85]
    for i, z in enumerate(z_slices):
        ax = axes[0, i]
        ax.imshow(background[:, :, z].T, cmap="gray", origin="lower")
        ax.imshow(heatmap[:, :, z].T, cmap="hot", alpha=0.5, origin="lower",
                  vmin=0, vmax=1)
        ax.set_title(f"Axial z={z}")
        ax.axis("off")

    # Sagittal slices (x axis)
    x_slices = [40, 55, 64, 75, 90]
    for i, x in enumerate(x_slices):
        ax = axes[1, i]
        ax.imshow(background[x, :, :].T, cmap="gray", origin="lower")
        ax.imshow(heatmap[x, :, :].T, cmap="hot", alpha=0.5, origin="lower",
                  vmin=0, vmax=1)
        ax.set_title(f"Sagittal x={x}")
        ax.axis("off")

    plt.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {out_path.name}")


def plot_comparison(background, cn_hmap, ad_hmap, diff_map, out_path):
    """3-row plot: CN attention, AD attention, diff (AD-CN)."""
    fig, axes = plt.subplots(3, 6, figsize=(20, 10))

    z_slices = [35, 48, 60, 72, 85, 100]
    for i, z in enumerate(z_slices):
        # CN
        axes[0, i].imshow(background[:, :, z].T, cmap="gray", origin="lower")
        axes[0, i].imshow(cn_hmap[:, :, z].T, cmap="hot", alpha=0.55, origin="lower", vmin=0, vmax=1)
        axes[0, i].set_title(f"CN attention (z={z})")
        axes[0, i].axis("off")
        # AD
        axes[1, i].imshow(background[:, :, z].T, cmap="gray", origin="lower")
        axes[1, i].imshow(ad_hmap[:, :, z].T, cmap="hot", alpha=0.55, origin="lower", vmin=0, vmax=1)
        axes[1, i].set_title(f"AD attention (z={z})")
        axes[1, i].axis("off")
        # Diff
        axes[2, i].imshow(background[:, :, z].T, cmap="gray", origin="lower")
        vmax = np.abs(diff_map).max()
        axes[2, i].imshow(diff_map[:, :, z].T, cmap="coolwarm", alpha=0.6,
                          origin="lower", vmin=-vmax, vmax=vmax)
        axes[2, i].set_title(f"AD-CN diff (z={z})")
        axes[2, i].axis("off")

    plt.suptitle("ViT attention rollout: CN vs AD vs diff (5 folds, 4 cases/class/fold, seed 42)",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {out_path.name}")


def main():
    print("[*] Loading fold means...")
    cn_all, ad_all = load_fold_means()  # (5, 8, 8, 8)
    print(f"    shapes: CN={cn_all.shape}  AD={ad_all.shape}")

    # Aggregate across folds
    cn_mean = cn_all.mean(axis=0)  # (8, 8, 8)
    ad_mean = ad_all.mean(axis=0)

    print("[*] Upsampling to 128^3 via trilinear...")
    cn_up = upsample_to_128(cn_mean[None])[0]  # (128, 128, 128)
    ad_up = upsample_to_128(ad_mean[None])[0]

    # Normalize each to [0, 1] for visualization
    cn_norm = normalize(cn_up, percentile=(5, 99))
    ad_norm = normalize(ad_up, percentile=(5, 99))
    # Diff map (signed)
    diff = ad_up - cn_up

    # Save raw aggregated data
    np.save(RESULTS / "vit_roll_AGG_CN.npy", cn_up)
    np.save(RESULTS / "vit_roll_AGG_AD.npy", ad_up)
    np.save(RESULTS / "vit_roll_AGG_diff.npy", diff)
    print(f"  saved aggregated heatmaps: CN, AD, diff (each 128^3)")

    print("[*] Loading background brain...")
    bg = load_sample_brain()
    print(f"    bg shape: {bg.shape}, range: [{bg.min():.2f}, {bg.max():.2f}]")

    print("[*] Plotting...")
    plot_slices(bg, cn_norm, "ViT attention rollout - CN (avg 5 folds, 4 cases/fold)",
                RESULTS / "vit_roll_AGG_CN_slices.png")
    plot_slices(bg, ad_norm, "ViT attention rollout - AD (avg 5 folds, 4 cases/fold)",
                RESULTS / "vit_roll_AGG_AD_slices.png")
    plot_comparison(bg, cn_norm, ad_norm, diff, RESULTS / "vit_roll_AGG_comparison.png")

    # Top-level stats
    diff_abs = np.abs(diff)
    print(f"\n[*] Stats:")
    print(f"    CN attention mean: {cn_up.mean():.4f}  max: {cn_up.max():.4f}")
    print(f"    AD attention mean: {ad_up.mean():.4f}  max: {ad_up.max():.4f}")
    print(f"    |diff| mean: {diff_abs.mean():.4f}  max: {diff_abs.max():.4f}")

    # Find peak diff location
    max_idx = np.unravel_index(diff_abs.argmax(), diff_abs.shape)
    print(f"    Peak diff location: voxel {max_idx}  sign={'AD>CN' if diff[max_idx]>0 else 'CN>AD'}")


if __name__ == "__main__":
    main()

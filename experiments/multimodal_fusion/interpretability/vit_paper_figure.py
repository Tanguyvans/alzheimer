#!/usr/bin/env python3
"""
Paper-ready ViT attention rollout figure.

Produces a compact 2x3 panel for the paper:
  row 1 = CN attention overlay
  row 2 = AD attention overlay
  cols   = axial z=40, axial z=50, coronal y=72

Plus optional diff-only figure (AD - CN) highlighting discriminative regions.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import zoom

SCRIPT_DIR = Path(__file__).resolve().parent
EXP = SCRIPT_DIR.parent
RESULTS = SCRIPT_DIR / "results_new"


def normalize(vol, percentile=(5, 99)):
    lo, hi = np.percentile(vol, percentile)
    return np.clip((vol - lo) / (hi - lo + 1e-6), 0, 1)


def load_background():
    test_df = pd.read_csv(EXP / "data/combined_trajectory/test.csv")
    cn_row = test_df[test_df["label"] == 0].iloc[10]
    img = nib.load(cn_row["scan_path"])
    vol = img.get_fdata().astype(np.float32)
    zf = [128 / s for s in vol.shape]
    return normalize(zoom(vol, zf, order=1))


def plot_paper_figure(bg, cn_up, ad_up, out_path, slices=None):
    """
    Compact 2x3 figure: 2 rows (CN, AD) x 3 cols (axial z=40, axial z=50, coronal y=72).
    """
    if slices is None:
        slices = [("axial", 40), ("axial", 50), ("coronal", 72)]

    # Normalize each heatmap separately for comparable colormap scaling
    cn_norm = normalize(cn_up, percentile=(5, 99))
    ad_norm = normalize(ad_up, percentile=(5, 99))

    fig, axes = plt.subplots(2, 3, figsize=(11, 7.5))

    labels_row = ["Cognitively Normal (CN)", "AD trajectory"]
    heatmaps = [cn_norm, ad_norm]

    for r in range(2):
        for c, (plane, idx) in enumerate(slices):
            ax = axes[r, c]
            if plane == "axial":
                bg_slice = bg[:, :, idx]
                hm_slice = heatmaps[r][:, :, idx]
                title = f"Axial z={idx}"
            elif plane == "coronal":
                bg_slice = bg[:, idx, :]
                hm_slice = heatmaps[r][:, idx, :]
                title = f"Coronal y={idx}"
            else:  # sagittal
                bg_slice = bg[idx, :, :]
                hm_slice = heatmaps[r][idx, :, :]
                title = f"Sagittal x={idx}"

            ax.imshow(bg_slice.T, cmap="gray", origin="lower")
            ax.imshow(hm_slice.T, cmap="hot", alpha=0.55, origin="lower", vmin=0, vmax=1)
            if r == 0:
                ax.set_title(title, fontsize=11)
            if c == 0:
                ax.set_ylabel(labels_row[r], fontsize=11, fontweight="bold")
            ax.set_xticks([])
            ax.set_yticks([])

    # Add a single colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    sm = plt.cm.ScalarMappable(cmap="hot",
                                norm=plt.Normalize(vmin=0, vmax=1))
    fig.colorbar(sm, cax=cbar_ax, label="Attention (normalized)")

    plt.suptitle(
        "ViT attention rollout: where the model looks for CN vs AD prediction\n"
        "(5 folds × 4 cases/class/fold, seed 42, paper-grade checkpoints)",
        fontsize=11, y=1.0,
    )
    plt.tight_layout(rect=[0, 0, 0.92, 0.96])
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[saved] {out_path.name}")


def plot_diff_figure(bg, diff, out_path, slices=None):
    """Single row of discrimination (AD - CN) slices."""
    if slices is None:
        slices = [("axial", 40), ("axial", 50), ("axial", 60), ("coronal", 72)]

    fig, axes = plt.subplots(1, len(slices), figsize=(4 * len(slices), 4.5))
    vmax = np.abs(diff).max()

    for c, (plane, idx) in enumerate(slices):
        ax = axes[c]
        if plane == "axial":
            bg_slice = bg[:, :, idx]
            hm_slice = diff[:, :, idx]
            title = f"Axial z={idx}"
        elif plane == "coronal":
            bg_slice = bg[:, idx, :]
            hm_slice = diff[:, idx, :]
            title = f"Coronal y={idx}"
        else:
            bg_slice = bg[idx, :, :]
            hm_slice = diff[idx, :, :]
            title = f"Sagittal x={idx}"

        ax.imshow(bg_slice.T, cmap="gray", origin="lower")
        im = ax.imshow(hm_slice.T, cmap="coolwarm", alpha=0.65, origin="lower",
                       vmin=-vmax, vmax=vmax)
        ax.set_title(title, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])

    # Colorbar
    cbar = fig.colorbar(im, ax=axes, shrink=0.75, location="right", pad=0.02)
    cbar.set_label("Attention difference (AD − CN)", fontsize=10)

    plt.suptitle(
        "Discriminative attention: regions more attended for AD than CN\n"
        "(warm = AD > CN, cool = CN > AD)",
        fontsize=11, y=1.02,
    )
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[saved] {out_path.name}")


def main():
    cn_up = np.load(RESULTS / "vit_roll_AGG_CN.npy")
    ad_up = np.load(RESULTS / "vit_roll_AGG_AD.npy")
    diff = np.load(RESULTS / "vit_roll_AGG_diff.npy")

    bg = load_background()

    plot_paper_figure(bg, cn_up, ad_up,
                      RESULTS / "vit_paper_main.png")
    plot_diff_figure(bg, diff,
                     RESULTS / "vit_paper_diff.png")

    print("\nSlices selected to capture the medial-temporal / hippocampal region")
    print("where the peak AD>CN difference occurs (voxel ~(39, 72, 40)).")


if __name__ == "__main__":
    main()

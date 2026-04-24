#!/usr/bin/env python3
"""
Per-fold ViT attention figures — one image per fold to verify
visual consistency of the medial-temporal focus across folds.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.ndimage import zoom

SCRIPT_DIR = Path(__file__).resolve().parent
EXP = SCRIPT_DIR.parent
RESULTS = SCRIPT_DIR / "results_new"


def normalize(vol, percentile=(5, 99)):
    lo, hi = np.percentile(vol, percentile)
    return np.clip((vol - lo) / (hi - lo + 1e-6), 0, 1)


def upsample_to_128(grid8):
    t = torch.from_numpy(grid8).float().unsqueeze(0).unsqueeze(0)
    t = F.interpolate(t, size=(128, 128, 128), mode="trilinear", align_corners=False)
    return t.squeeze(0).squeeze(0).numpy()


def load_background():
    test_df = pd.read_csv(EXP / "data/combined_trajectory/test.csv")
    cn_row = test_df[test_df["label"] == 0].iloc[10]
    img = nib.load(cn_row["scan_path"])
    vol = img.get_fdata().astype(np.float32)
    zf = [128 / s for s in vol.shape]
    return normalize(zoom(vol, zf, order=1))


def plot_fold(bg, cn_up, ad_up, fold, out_path):
    slices = [("axial", 40), ("axial", 50), ("coronal", 72)]
    cn_norm = normalize(cn_up, percentile=(5, 99))
    ad_norm = normalize(ad_up, percentile=(5, 99))

    fig, axes = plt.subplots(2, 3, figsize=(11, 7.5))
    labels_row = ["CN", "AD"]
    heatmaps = [cn_norm, ad_norm]

    for r in range(2):
        for c, (plane, idx) in enumerate(slices):
            ax = axes[r, c]
            if plane == "axial":
                bg_slice = bg[:, :, idx]
                hm_slice = heatmaps[r][:, :, idx]
                title = f"Axial z={idx}"
            else:
                bg_slice = bg[:, idx, :]
                hm_slice = heatmaps[r][:, idx, :]
                title = f"Coronal y={idx}"
            ax.imshow(bg_slice.T, cmap="gray", origin="lower")
            ax.imshow(hm_slice.T, cmap="hot", alpha=0.55, origin="lower", vmin=0, vmax=1)
            if r == 0:
                ax.set_title(title, fontsize=11)
            if c == 0:
                ax.set_ylabel(labels_row[r], fontsize=11, fontweight="bold")
            ax.set_xticks([])
            ax.set_yticks([])

    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    sm = plt.cm.ScalarMappable(cmap="hot",
                                norm=plt.Normalize(vmin=0, vmax=1))
    fig.colorbar(sm, cax=cbar_ax, label="Attention")

    plt.suptitle(
        f"ViT attention rollout — Fold {fold} (seed 42, 4 cases/class)",
        fontsize=11, y=1.0,
    )
    plt.tight_layout(rect=[0, 0, 0.92, 0.96])
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[saved] {out_path.name}")


def main():
    bg = load_background()
    for fold in range(5):
        cn_grid = np.load(RESULTS / f"vit_roll_fold{fold}_CN_mean.npy")
        ad_grid = np.load(RESULTS / f"vit_roll_fold{fold}_AD_mean.npy")
        cn_up = upsample_to_128(cn_grid)
        ad_up = upsample_to_128(ad_grid)
        plot_fold(bg, cn_up, ad_up, fold, RESULTS / f"vit_per_fold_{fold}.png")


if __name__ == "__main__":
    main()

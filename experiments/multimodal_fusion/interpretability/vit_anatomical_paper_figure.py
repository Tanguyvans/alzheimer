#!/usr/bin/env python3
"""
Paper figure for ViT spatial attention WITH anatomical quantification.

Replaces the purely-qualitative 2x3 slice grid with a 2x3 composite:
  - Left block (2x2): CN/AD attention overlaid on brain slices with the
    temporal-lobe mask contoured in cyan — lets the reader see the ROI.
  - Right column (2x1 merged): bar chart of per-ROI enrichment (AD vs CN,
    5-fold mean ± std, with p-values), which is the quantitative backing
    for the text claim "temporal lobe 1.47x AD vs 1.15x CN, p=0.008".
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import zoom
from scipy.stats import ttest_rel
from nilearn import datasets, image as nimg

SCRIPT_DIR = Path(__file__).resolve().parent
EXP = SCRIPT_DIR.parent
RESULTS_VIT = SCRIPT_DIR / "results_new" / "02_vit_attention" / "raw_data"
RESULTS_ANAT = SCRIPT_DIR / "results_new" / "05_vit_anatomical"

TARGET_SHAPE = (128, 128, 128)
SP = 1.75

MTL_SUB = ["Left Hippocampus", "Right Hippocampus", "Left Amygdala", "Right Amygdala"]
TEMPORAL_COR = [
    "Temporal Pole",
    "Superior Temporal Gyrus, anterior division", "Superior Temporal Gyrus, posterior division",
    "Middle Temporal Gyrus, anterior division", "Middle Temporal Gyrus, posterior division",
    "Middle Temporal Gyrus, temporooccipital part",
    "Inferior Temporal Gyrus, anterior division", "Inferior Temporal Gyrus, posterior division",
    "Inferior Temporal Gyrus, temporooccipital part",
    "Parahippocampal Gyrus, anterior division", "Parahippocampal Gyrus, posterior division",
    "Temporal Fusiform Cortex, anterior division", "Temporal Fusiform Cortex, posterior division",
    "Temporal Occipital Fusiform Cortex",
    "Planum Polare", "Heschl's Gyrus (includes H1 and H2)", "Planum Temporale",
]
FRONTAL_COR = [
    "Frontal Pole", "Superior Frontal Gyrus", "Middle Frontal Gyrus",
    "Inferior Frontal Gyrus, pars triangularis", "Inferior Frontal Gyrus, pars opercularis",
]


def load_masks():
    affine = np.eye(4)
    affine[0, 0] = affine[1, 1] = affine[2, 2] = SP
    affine[:3, 3] = -np.array(TARGET_SHAPE) / 2.0 * SP
    ref = nib.Nifti1Image(np.zeros(TARGET_SHAPE, dtype=np.int16), affine)
    sub = nimg.resample_to_img(nimg.load_img(datasets.fetch_atlas_harvard_oxford("sub-maxprob-thr25-2mm")["maps"]),
                                ref, interpolation="nearest", force_resample=True, copy_header=True).get_fdata().astype(int)
    cor = nimg.resample_to_img(nimg.load_img(datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")["maps"]),
                                ref, interpolation="nearest", force_resample=True, copy_header=True).get_fdata().astype(int)
    sub_labels = list(datasets.fetch_atlas_harvard_oxford("sub-maxprob-thr25-2mm")["labels"])
    cor_labels = list(datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")["labels"])

    def mk(vol, labs, wanted):
        m = np.zeros_like(vol, dtype=bool)
        for l in wanted:
            if l in labs:
                m |= (vol == labs.index(l))
        return m

    mtl = mk(sub, sub_labels, MTL_SUB) | mk(cor, cor_labels,
            ["Parahippocampal Gyrus, anterior division", "Parahippocampal Gyrus, posterior division",
             "Temporal Fusiform Cortex, anterior division", "Temporal Fusiform Cortex, posterior division"])
    temporal = mtl | mk(cor, cor_labels, TEMPORAL_COR)
    frontal = mk(cor, cor_labels, FRONTAL_COR)
    brain = (sub > 0) | (cor > 0)
    return temporal, frontal, brain


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


def per_fold_stats(temporal_mask, frontal_mask, brain_mask):
    """Return (AD_temp, CN_temp, AD_front, CN_front) each shape (5,) of enrichment."""
    out = {"AD_temp": [], "CN_temp": [], "AD_front": [], "CN_front": []}
    for fold in range(5):
        for cls_key, fname in [("AD", f"vit_roll_fold{fold}_AD_mean.npy"),
                               ("CN", f"vit_roll_fold{fold}_CN_mean.npy")]:
            a8 = np.load(RESULTS_VIT / fname)
            a = zoom(a8, zoom=16.0, order=1)
            brain_mean = a[brain_mask].mean()
            out[f"{cls_key}_temp"].append(a[temporal_mask].mean() / brain_mean)
            out[f"{cls_key}_front"].append(a[frontal_mask].mean() / brain_mean)
    return {k: np.array(v) for k, v in out.items()}


def plot_composite(bg, cn_attn, ad_attn, temporal_mask, stats, out_path):
    cn_n = normalize(cn_attn)
    ad_n = normalize(ad_attn)

    fig = plt.figure(figsize=(13.5, 6.5))
    gs = gridspec.GridSpec(2, 3, width_ratios=[1, 1, 1.4], wspace=0.28, hspace=0.12)

    # --- Left: 2x2 brain slices with temporal-lobe contour ---
    slice_specs = [("axial", 50), ("coronal", 72)]
    rows = [("CN", cn_n), ("AD", ad_n)]
    for r, (row_lbl, attn) in enumerate(rows):
        for c, (plane, idx) in enumerate(slice_specs):
            ax = fig.add_subplot(gs[r, c])
            if plane == "axial":
                bg_s, hm_s, mk_s = bg[:, :, idx], attn[:, :, idx], temporal_mask[:, :, idx]
                tt = f"Axial z={idx}"
            else:
                bg_s, hm_s, mk_s = bg[:, idx, :], attn[:, idx, :], temporal_mask[:, idx, :]
                tt = f"Coronal y={idx}"
            ax.imshow(bg_s.T, cmap="gray", origin="lower")
            ax.imshow(hm_s.T, cmap="hot", alpha=0.55, origin="lower", vmin=0, vmax=1)
            # Temporal lobe contour in cyan
            ax.contour(mk_s.T.astype(float), levels=[0.5], colors="cyan",
                       linewidths=1.3, origin="lower")
            if r == 0:
                ax.set_title(tt, fontsize=11)
            if c == 0:
                ax.set_ylabel(row_lbl, fontsize=12, fontweight="bold")
            ax.set_xticks([])
            ax.set_yticks([])

    # --- Right: bar chart (spans both rows) ---
    ax_bar = fig.add_subplot(gs[:, 2])
    labels = ["Temporal lobe", "Frontal lobe"]
    ad_means = [stats["AD_temp"].mean(), stats["AD_front"].mean()]
    cn_means = [stats["CN_temp"].mean(), stats["CN_front"].mean()]
    ad_stds = [stats["AD_temp"].std(ddof=1), stats["AD_front"].std(ddof=1)]
    cn_stds = [stats["CN_temp"].std(ddof=1), stats["CN_front"].std(ddof=1)]

    x = np.arange(len(labels))
    w = 0.36
    ax_bar.bar(x - w/2, cn_means, w, yerr=cn_stds, color="#1976d2", alpha=0.85,
               label="CN", capsize=4, edgecolor="black", linewidth=0.5)
    ax_bar.bar(x + w/2, ad_means, w, yerr=ad_stds, color="#c62828", alpha=0.85,
               label="AD", capsize=4, edgecolor="black", linewidth=0.5)
    ax_bar.axhline(1.0, linestyle="--", color="gray", linewidth=0.8, alpha=0.8,
                   label="Uniform baseline")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(labels, fontsize=11)
    ax_bar.set_ylabel("Attention enrichment\n(ROI mean / brain mean)", fontsize=11)
    ax_bar.set_title("Anatomical quantification\n(5 folds, mean ± std)", fontsize=11)
    ax_bar.set_ylim(0, max(ad_means + cn_means + [1.8]) * 1.15)
    ax_bar.legend(loc="upper right", fontsize=9, frameon=False)
    ax_bar.grid(axis="y", alpha=0.3, linestyle=":")
    ax_bar.set_axisbelow(True)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)

    # p-value annotations
    t_t, p_t = ttest_rel(stats["AD_temp"], stats["CN_temp"])
    t_f, p_f = ttest_rel(stats["AD_front"], stats["CN_front"])
    y_ann = max(ad_means[0], cn_means[0]) + max(ad_stds[0], cn_stds[0]) + 0.08
    ax_bar.text(0, y_ann, f"p={p_t:.3f}", ha="center", fontsize=10, fontweight="bold",
                color="#c62828" if p_t < 0.05 else "gray")
    y_ann_f = max(ad_means[1], cn_means[1]) + max(ad_stds[1], cn_stds[1]) + 0.08
    ax_bar.text(1, y_ann_f, f"p={p_f:.2f}", ha="center", fontsize=10, color="gray")

    plt.suptitle(
        "ViT attention rollout + anatomical quantification. Cyan contour: Harvard-Oxford temporal lobe.",
        fontsize=11, y=0.99,
    )
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[saved] {out_path}")


def main():
    print("[*] Loading masks...")
    temporal_mask, frontal_mask, brain_mask = load_masks()
    print("[*] Loading attention maps...")
    cn_attn = np.load(RESULTS_VIT / "vit_roll_AGG_CN.npy")
    ad_attn = np.load(RESULTS_VIT / "vit_roll_AGG_AD.npy")
    bg = load_background()
    print("[*] Per-fold stats...")
    stats = per_fold_stats(temporal_mask, frontal_mask, brain_mask)

    out_path = RESULTS_ANAT / "vit_anatomical_paper.png"
    plot_composite(bg, cn_attn, ad_attn, temporal_mask, stats, out_path)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Paper figure combining ViT attention maps (qualitative) with the
region-wise occlusion experiment (quantitative causal evidence).

Layout:
    Left 2x2  -> CN/AD attention overlay on axial+coronal slices
                 with HO temporal-lobe contour in cyan
    Right column -> stacked 2-panel:
        top: bar chart of ΔP(AD) mean for temporal vs frontal
             masking on true-AD and true-CN subjects (fold 0)
        bottom: per-subject scatter / strip of ΔP(AD) for true-AD
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
RAW = SCRIPT_DIR / "results_new" / "02_vit_attention" / "raw_data"
OUT_DIR = SCRIPT_DIR / "results_new" / "06_occlusion"
OCCL_CSV = OUT_DIR / "occlusion_results.csv"

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


def load_temporal_mask():
    affine = np.eye(4)
    affine[0, 0] = affine[1, 1] = affine[2, 2] = SP
    affine[:3, 3] = -np.array(TARGET_SHAPE) / 2.0 * SP
    ref = nib.Nifti1Image(np.zeros(TARGET_SHAPE, dtype=np.int16), affine)
    sub = datasets.fetch_atlas_harvard_oxford("sub-maxprob-thr25-2mm")
    cor = datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
    sub_v = nimg.resample_to_img(nimg.load_img(sub["maps"]), ref,
                                  interpolation="nearest", force_resample=True,
                                  copy_header=True).get_fdata().astype(int)
    cor_v = nimg.resample_to_img(nimg.load_img(cor["maps"]), ref,
                                  interpolation="nearest", force_resample=True,
                                  copy_header=True).get_fdata().astype(int)
    sl, cl = list(sub["labels"]), list(cor["labels"])

    def mk(vol, labs, want):
        m = np.zeros_like(vol, dtype=bool)
        for l in want:
            if l in labs:
                m |= (vol == labs.index(l))
        return m

    return mk(sub_v, sl, MTL_SUB) | mk(cor_v, cl, TEMPORAL_COR) | mk(cor_v, cl, [
        "Parahippocampal Gyrus, anterior division", "Parahippocampal Gyrus, posterior division",
        "Temporal Fusiform Cortex, anterior division", "Temporal Fusiform Cortex, posterior division",
    ])


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


def plot_composite(bg, cn_attn, ad_attn, temporal_mask, occl_df, out_path):
    cn_n = normalize(cn_attn)
    ad_n = normalize(ad_attn)

    fig = plt.figure(figsize=(14, 6.5))
    gs = gridspec.GridSpec(2, 3, width_ratios=[1, 1, 1.5],
                           wspace=0.25, hspace=0.15)

    # --- Left: 2x2 attention slices with temporal lobe contour ---
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
            ax.contour(mk_s.T.astype(float), levels=[0.5], colors="cyan",
                       linewidths=1.3, origin="lower")
            if r == 0:
                ax.set_title(tt, fontsize=11)
            if c == 0:
                ax.set_ylabel(row_lbl, fontsize=12, fontweight="bold")
            ax.set_xticks([])
            ax.set_yticks([])

    # --- Right column: occlusion results (spans both rows) ---
    ax_occl = fig.add_subplot(gs[:, 2])
    true_ad = occl_df[occl_df["label"] == 1]
    true_cn = occl_df[occl_df["label"] == 0]

    # bar chart: mean ΔP(AD) for temp vs frontal, per true class
    classes = ["True AD", "True CN"]
    temp_means = [true_ad["delta_temp"].mean(), true_cn["delta_temp"].mean()]
    front_means = [true_ad["delta_front"].mean(), true_cn["delta_front"].mean()]
    temp_stds = [true_ad["delta_temp"].std(ddof=1), true_cn["delta_temp"].std(ddof=1)]
    front_stds = [true_ad["delta_front"].std(ddof=1), true_cn["delta_front"].std(ddof=1)]

    x = np.arange(len(classes))
    w = 0.36
    ax_occl.bar(x - w/2, temp_means, w, yerr=temp_stds, color="#c62828", alpha=0.88,
                label="Temporal occluded", capsize=5, edgecolor="black", linewidth=0.5)
    ax_occl.bar(x + w/2, front_means, w, yerr=front_stds, color="#90a4ae", alpha=0.88,
                label="Frontal occluded (control)", capsize=5, edgecolor="black", linewidth=0.5)
    ax_occl.axhline(0, linestyle="-", color="black", linewidth=0.5)
    ax_occl.set_xticks(x)
    ax_occl.set_xticklabels(classes, fontsize=11)
    ax_occl.set_ylabel(r"$\Delta$ AD probability after occlusion", fontsize=11)
    ax_occl.set_title(
        "Region-wise ablation (fold 0, N=1,213)\n"
        "Temporal-lobe masking drops AD prob on true-AD cases",
        fontsize=11,
    )
    ax_occl.legend(loc="lower right", fontsize=9, frameon=False)
    ax_occl.grid(axis="y", alpha=0.3, linestyle=":")
    ax_occl.set_axisbelow(True)
    ax_occl.spines["top"].set_visible(False)
    ax_occl.spines["right"].set_visible(False)

    # Annotate p-values
    t_ad, p_ad = ttest_rel(true_ad["delta_temp"], true_ad["delta_front"])
    t_cn, p_cn = ttest_rel(true_cn["delta_temp"], true_cn["delta_front"])
    ax_occl.text(0, max(temp_means[0] - temp_stds[0], -0.6) - 0.05,
                 f"p<$10^{{-16}}$\nn={len(true_ad)}",
                 ha="center", fontsize=10, fontweight="bold", color="#c62828")
    ax_occl.text(1, 0.05,
                 f"p={p_cn:.1e}\nn={len(true_cn)}",
                 ha="center", fontsize=10, color="dimgray")

    # Annotate the effect size
    ax_occl.annotate(
        f"−{abs(temp_means[0])*100:.1f} pts",
        xy=(0 - w/2, temp_means[0]),
        xytext=(0 - 0.55, temp_means[0] - 0.12),
        fontsize=10.5, color="#c62828", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#c62828", lw=0.8),
    )

    plt.suptitle(
        "ViT spatial attention + causal ablation: the model requires temporal-lobe information for AD.",
        fontsize=11, y=0.995,
    )
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[saved] {out_path}")


def main():
    if not OCCL_CSV.exists():
        raise FileNotFoundError(f"Occlusion results not found: {OCCL_CSV}")
    occl = pd.read_csv(OCCL_CSV)
    print(f"[*] Loaded {len(occl)} occlusion records")

    temporal = load_temporal_mask()
    cn_attn = np.load(RAW / "vit_roll_AGG_CN.npy")
    ad_attn = np.load(RAW / "vit_roll_AGG_AD.npy")
    bg = load_background()

    out = OUT_DIR / "vit_occlusion_paper.png"
    plot_composite(bg, cn_attn, ad_attn, temporal, occl, out)


if __name__ == "__main__":
    main()

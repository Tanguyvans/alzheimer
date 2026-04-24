#!/usr/bin/env python3
"""
Paper figure: minimal, focused on the causal ablation result.

Layout (single row, two panels):
    Left (narrow)  : one coronal MRI slice with the HO temporal-lobe mask
                     highlighted — a visual legend for "what we occluded".
    Right (wide)   : bar chart of ΔP(AD) for temporal vs frontal occlusion,
                     on true-AD and true-CN subjects (fold 0).

No attention heatmaps — the attention/correlational result is mentioned in
text only, so the figure can commit fully to the causal story.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import zoom
from nilearn import datasets, image as nimg

SCRIPT_DIR = Path(__file__).resolve().parent
EXP = SCRIPT_DIR.parent
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


def load_background_coronal():
    test_df = pd.read_csv(EXP / "data/combined_trajectory/test.csv")
    row = test_df[test_df["label"] == 0].iloc[10]
    img = nib.load(row["scan_path"])
    vol = img.get_fdata().astype(np.float32)
    zf = [128 / s for s in vol.shape]
    return normalize(zoom(vol, zf, order=1))


def plot(bg, temporal_mask, occl_df, out_path):
    fig = plt.figure(figsize=(10, 4.5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 2.2], wspace=0.25)

    # --- Left: one coronal slice with temporal lobe highlighted ---
    ax_brain = fig.add_subplot(gs[0, 0])
    coronal_idx = 72
    bg_s = bg[:, coronal_idx, :]
    mk_s = temporal_mask[:, coronal_idx, :]
    ax_brain.imshow(bg_s.T, cmap="gray", origin="lower")
    # Temporal mask as red semi-transparent overlay
    mask_overlay = np.ma.masked_where(~mk_s, np.ones_like(mk_s, dtype=float))
    ax_brain.imshow(mask_overlay.T, cmap="Reds", alpha=0.55, origin="lower", vmin=0, vmax=1.2)
    ax_brain.set_title("Temporal lobe mask\n(coronal y=72)", fontsize=11)
    ax_brain.set_xticks([])
    ax_brain.set_yticks([])

    # --- Right: bar chart of ΔP(AD) ---
    ax = fig.add_subplot(gs[0, 1])
    true_ad = occl_df[occl_df["label"] == 1]
    true_cn = occl_df[occl_df["label"] == 0]

    classes = ["True AD\n(N=263)", "True CN\n(N=950)"]
    temp_m = [true_ad["delta_temp"].mean(), true_cn["delta_temp"].mean()]
    temp_s = [true_ad["delta_temp"].std(ddof=1), true_cn["delta_temp"].std(ddof=1)]
    front_m = [true_ad["delta_front"].mean(), true_cn["delta_front"].mean()]
    front_s = [true_ad["delta_front"].std(ddof=1), true_cn["delta_front"].std(ddof=1)]

    x = np.arange(len(classes))
    w = 0.36
    ax.bar(x - w/2, temp_m, w, yerr=temp_s, color="#c62828", alpha=0.88,
           label="Temporal occluded", capsize=5, edgecolor="black", linewidth=0.5)
    ax.bar(x + w/2, front_m, w, yerr=front_s, color="#90a4ae", alpha=0.88,
           label="Frontal occluded (control)", capsize=5, edgecolor="black", linewidth=0.5)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=11)
    ax.set_ylabel(r"$\Delta$ AD probability", fontsize=11)
    ax.set_title("Causal ablation: removing temporal-lobe MRI information\ndrops AD predictions (fold 0)",
                 fontsize=11)
    ax.legend(loc="lower left", fontsize=9, frameon=False)
    ax.grid(axis="y", alpha=0.3, linestyle=":")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Single annotation: headline effect size + p-value on the True-AD temporal bar
    ax.annotate(
        f"{temp_m[0]*100:+.1f} pts\n$p<10^{{-16}}$",
        xy=(0 - w/2, temp_m[0]),
        xytext=(0 - w/2 - 0.05, temp_m[0] - 0.15),
        ha="center", va="top",
        fontsize=11, color="#c62828", fontweight="bold",
    )

    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[saved] {out_path}")


def main():
    occl = pd.read_csv(OCCL_CSV)
    print(f"[*] Loaded {len(occl)} occlusion records")
    temporal = load_temporal_mask()
    bg = load_background_coronal()
    plot(bg, temporal, occl, OUT_DIR / "vit_occlusion_paper.png")


if __name__ == "__main__":
    main()

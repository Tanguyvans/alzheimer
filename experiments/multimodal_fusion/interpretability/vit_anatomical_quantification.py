#!/usr/bin/env python3
"""
Quantitative anatomical validation of ViT attention maps.

Computes the fraction of attention mass falling within AD-relevant regions
(medial temporal lobe) vs a neutral control region (frontal lobe), using
the Harvard-Oxford atlas (MNI152).

The model's input space is 128^3 at 1.75mm isotropic, center-cropped on
the MNI template. We resample the HO atlas to match this grid and build
three ROIs: MTL (hippocampus + parahippocampal + amygdala), frontal lobe
(superior + middle + inferior frontal gyri), and whole-brain (all labeled
voxels).

For each of the 5 folds, we compute the mean attention per ROI (normalized
by ROI volume) and the AD/CN ratio. A paired t-test across folds quantifies
whether MTL enrichment is statistically higher for AD than CN.
"""

import sys
from pathlib import Path

import numpy as np
import nibabel as nib
from nilearn import datasets, image
from scipy.stats import ttest_rel
from scipy.ndimage import zoom
import matplotlib.pyplot as plt

REPO = Path("/home/tanguy/medical/alzheimer")
EXP = REPO / "experiments" / "multimodal_fusion"
RAW = EXP / "interpretability" / "results_new" / "02_vit_attention" / "raw_data"
OUT = EXP / "interpretability" / "results_new" / "05_vit_anatomical"
OUT.mkdir(exist_ok=True, parents=True)

# ---- Model input space: 128^3 @ 1.75mm, centered on MNI template origin -----
TARGET_SHAPE = (128, 128, 128)
TARGET_SPACING = 1.75  # mm

# ROI definitions from Harvard-Oxford atlases
MTL_SUB_LABELS = [
    "Left Hippocampus", "Right Hippocampus",
    "Left Amygdala", "Right Amygdala",
]  # from sub-maxprob
MTL_COR_LABELS = [
    "Parahippocampal Gyrus, anterior division",
    "Parahippocampal Gyrus, posterior division",
    "Temporal Fusiform Cortex, anterior division",
    "Temporal Fusiform Cortex, posterior division",
]
TEMPORAL_COR_LABELS = [
    "Temporal Pole",
    "Superior Temporal Gyrus, anterior division",
    "Superior Temporal Gyrus, posterior division",
    "Middle Temporal Gyrus, anterior division",
    "Middle Temporal Gyrus, posterior division",
    "Middle Temporal Gyrus, temporooccipital part",
    "Inferior Temporal Gyrus, anterior division",
    "Inferior Temporal Gyrus, posterior division",
    "Inferior Temporal Gyrus, temporooccipital part",
    "Parahippocampal Gyrus, anterior division",
    "Parahippocampal Gyrus, posterior division",
    "Temporal Fusiform Cortex, anterior division",
    "Temporal Fusiform Cortex, posterior division",
    "Temporal Occipital Fusiform Cortex",
    "Planum Polare",
    "Heschl's Gyrus (includes H1 and H2)",
    "Planum Temporale",
]
FRONTAL_COR_LABELS = [
    "Frontal Pole",
    "Superior Frontal Gyrus",
    "Middle Frontal Gyrus",
    "Inferior Frontal Gyrus, pars triangularis",
    "Inferior Frontal Gyrus, pars opercularis",
]


def build_target_affine():
    """Affine for a 128^3 @ 1.75mm volume centered at MNI origin (0,0,0)."""
    affine = np.eye(4)
    affine[0, 0] = TARGET_SPACING
    affine[1, 1] = TARGET_SPACING
    affine[2, 2] = TARGET_SPACING
    # Center the volume at MNI origin
    center_vox = np.array(TARGET_SHAPE) / 2.0
    affine[:3, 3] = -center_vox * TARGET_SPACING
    return affine


def fetch_and_resample_atlases(target_affine, target_shape):
    """Fetch HO subcortical + cortical atlases, resample to model grid."""
    print("[*] Fetching Harvard-Oxford atlases (nilearn cache)...")
    sub = datasets.fetch_atlas_harvard_oxford("sub-maxprob-thr25-2mm")
    cor = datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")

    sub_img = image.load_img(sub["maps"])
    cor_img = image.load_img(cor["maps"])
    ref_img = nib.Nifti1Image(
        np.zeros(target_shape, dtype=np.int16), target_affine
    )
    sub_resampled = image.resample_to_img(sub_img, ref_img, interpolation="nearest", force_resample=True, copy_header=True)
    cor_resampled = image.resample_to_img(cor_img, ref_img, interpolation="nearest", force_resample=True, copy_header=True)

    return (
        sub_resampled.get_fdata().astype(np.int16),
        cor_resampled.get_fdata().astype(np.int16),
        list(sub["labels"]),
        list(cor["labels"]),
    )


def build_mask(vol, labels_list, wanted_labels):
    """Boolean mask for wanted_labels (by string match) from an atlas volume."""
    indices = [labels_list.index(l) for l in wanted_labels if l in labels_list]
    mask = np.zeros_like(vol, dtype=bool)
    for idx in indices:
        mask |= (vol == idx)
    return mask


def roi_attention_fraction(attn, mask, brain_mask):
    """Fraction of total brain-mass attention that falls inside the ROI.

    Normalizes by brain mask so we measure concentration, not raw values.
    """
    brain_total = attn[brain_mask].sum()
    if brain_total <= 0:
        return np.nan
    return attn[mask].sum() / brain_total


def roi_mean_per_voxel(attn, mask):
    """Mean attention per voxel inside the ROI (for enrichment ratio)."""
    if mask.sum() == 0:
        return np.nan
    return attn[mask].mean()


def compute_per_fold(rois, brain_mask):
    """For each fold, compute ROI fractions for AD and CN mean maps.

    rois: dict of {roi_name: bool mask (128^3)}
    """
    records = []
    for fold in range(5):
        ad8 = np.load(RAW / f"vit_roll_fold{fold}_AD_mean.npy")
        cn8 = np.load(RAW / f"vit_roll_fold{fold}_CN_mean.npy")
        ad = zoom(ad8, zoom=16.0, order=1)
        cn = zoom(cn8, zoom=16.0, order=1)
        for cls_name, attn in [("AD", ad), ("CN", cn)]:
            rec = {"fold": fold, "class": cls_name}
            brain_mean = roi_mean_per_voxel(attn, brain_mask)
            for roi_name, mask in rois.items():
                rec[f"{roi_name}_frac"] = roi_attention_fraction(attn, mask, brain_mask)
                roi_mean = roi_mean_per_voxel(attn, mask)
                rec[f"{roi_name}_enrich"] = roi_mean / brain_mean if brain_mean > 0 else np.nan
            records.append(rec)
    return records


def summarize(records, roi_names):
    """Aggregate across folds and run paired tests AD vs CN for each ROI."""
    ad_recs = sorted([r for r in records if r["class"] == "AD"], key=lambda r: r["fold"])
    cn_recs = sorted([r for r in records if r["class"] == "CN"], key=lambda r: r["fold"])

    print("\n=== Per-fold fractions ===")
    hdr = f"{'fold':<5}{'class':<5}" + "".join([f"{n+'_frac':<14}" for n in roi_names])
    print(hdr)
    for r in records:
        row = f"{r['fold']:<5}{r['class']:<5}"
        for n in roi_names:
            row += f"{r[f'{n}_frac']:<14.4f}"
        print(row)

    print("\n=== Aggregated fractions (mean±std, paired-t AD vs CN) ===")
    for n in roi_names:
        ad_v = np.array([r[f"{n}_frac"] for r in ad_recs])
        cn_v = np.array([r[f"{n}_frac"] for r in cn_recs])
        t, p = ttest_rel(ad_v, cn_v)
        print(f"  {n+'_frac':<20} AD={ad_v.mean():.4f}±{ad_v.std(ddof=1):.4f}  "
              f"CN={cn_v.mean():.4f}±{cn_v.std(ddof=1):.4f}  "
              f"Δ(AD-CN)={ad_v.mean()-cn_v.mean():+.4f}  t={t:+.3f}  p={p:.3e}")

    print("\n=== Enrichment (attention per voxel / brain mean) ===")
    for n in roi_names:
        ad_v = np.array([r[f"{n}_enrich"] for r in ad_recs])
        cn_v = np.array([r[f"{n}_enrich"] for r in cn_recs])
        t, p = ttest_rel(ad_v, cn_v)
        print(f"  {n+'_enrich':<20} AD={ad_v.mean():.3f}±{ad_v.std(ddof=1):.3f}  "
              f"CN={cn_v.mean():.3f}±{cn_v.std(ddof=1):.3f}  "
              f"t={t:+.3f}  p={p:.3e}")

    # AGG map analysis (the one used in the paper figure)
    print("\n=== AGG map analysis (the paper figure's aggregated attention) ===")
    for cls_name, fname in [("AD", "vit_roll_AGG_AD.npy"),
                            ("CN", "vit_roll_AGG_CN.npy"),
                            ("AD-CN_diff", "vit_roll_AGG_diff.npy")]:
        try:
            attn = np.load(RAW / fname)
            if attn.shape != (128, 128, 128):
                continue
            print(f"  [{cls_name}]")
            brain_m = attn[brain_mask_global].sum() if attn[brain_mask_global].sum() != 0 else 1
            for n, mask in rois_global.items():
                frac = attn[mask].sum() / brain_m if cls_name != "AD-CN_diff" else attn[mask].sum()
                mean_in = attn[mask].mean()
                brain_mean = attn[brain_mask_global].mean()
                enrich = mean_in / brain_mean if brain_mean != 0 else np.nan
                print(f"    {n:<12} frac={frac:+.4f} mean={mean_in:+.6f} enrich={enrich:+.3f}")
        except FileNotFoundError:
            pass


def plot_masks_overlay(mtl_mask, frontal_mask, brain_mask, out_path):
    """Sanity-check: show masks on three mid-slices."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    mid = [d // 2 for d in mtl_mask.shape]
    for i, (ax, ax_name, idx) in enumerate(zip(axes, ["Sagittal", "Coronal", "Axial"], mid)):
        if i == 0:
            bg = brain_mask[idx, :, :].astype(float)
            m = mtl_mask[idx, :, :].astype(float)
            f = frontal_mask[idx, :, :].astype(float)
        elif i == 1:
            bg = brain_mask[:, idx, :].astype(float)
            m = mtl_mask[:, idx, :].astype(float)
            f = frontal_mask[:, idx, :].astype(float)
        else:
            bg = brain_mask[:, :, idx].astype(float)
            m = mtl_mask[:, :, idx].astype(float)
            f = frontal_mask[:, :, idx].astype(float)
        ax.imshow(bg.T, cmap="gray", alpha=0.3, origin="lower")
        ax.imshow(np.ma.masked_equal(m, 0).T, cmap="autumn", alpha=0.8, origin="lower")
        ax.imshow(np.ma.masked_equal(f, 0).T, cmap="winter", alpha=0.8, origin="lower")
        ax.set_title(f"{ax_name} (slice {idx})")
        ax.axis("off")
    fig.suptitle("ROI masks: MTL (red) vs Frontal (blue) vs brain (gray)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()


def main():
    target_affine = build_target_affine()

    sub_vol, cor_vol, sub_labels, cor_labels = fetch_and_resample_atlases(
        target_affine, TARGET_SHAPE
    )
    print(f"[*] Resampled atlases to {TARGET_SHAPE}")

    mtl_mask = build_mask(sub_vol, sub_labels, MTL_SUB_LABELS) | \
               build_mask(cor_vol, cor_labels, MTL_COR_LABELS)
    temporal_mask = mtl_mask | build_mask(cor_vol, cor_labels, TEMPORAL_COR_LABELS)
    frontal_mask = build_mask(cor_vol, cor_labels, FRONTAL_COR_LABELS)
    # brain mask = only cortical + subcortical labeled voxels (exclude WM/CSF/ventricles to avoid dilution)
    brain_mask = (sub_vol > 0) | (cor_vol > 0)

    rois = {
        "MTL": mtl_mask,
        "temporal": temporal_mask,
        "frontal": frontal_mask,
    }
    global rois_global, brain_mask_global
    rois_global = rois
    brain_mask_global = brain_mask

    for name, m in rois.items():
        print(f"[*] {name}: {m.sum()} vox ({m.sum()/brain_mask.sum()*100:.2f}% of brain)")

    plot_masks_overlay(mtl_mask, frontal_mask, brain_mask, OUT / "roi_masks_sanity.png")

    records = compute_per_fold(rois, brain_mask)
    summarize(records, list(rois.keys()))

    # Save records
    import pandas as pd
    pd.DataFrame(records).to_csv(OUT / "vit_roi_per_fold.csv", index=False)
    print(f"\n[*] Per-fold CSV: {OUT / 'vit_roi_per_fold.csv'}")
    print("[*] Done.")


if __name__ == "__main__":
    main()

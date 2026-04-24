#!/usr/bin/env python3
"""
Causal occlusion analysis for ViT attention.

Tests whether the model CAUSALLY uses temporal-lobe information for AD
prediction (as opposed to merely attending to it). For each test subject,
we replace all voxels inside a Harvard-Oxford ROI with the subject's brain
mean intensity and measure the change in the model's AD probability vs
baseline.

Two ROI masks are tested:
  - Temporal lobe (AD-relevant, hypothesized-causal)
  - Frontal lobe (neutral control)

Strong result would be: temporal masking drops AD probability substantially
more than frontal masking on true-AD subjects, establishing a causal link.
Paired-t across subjects controls for inter-subject variability.

Usage:
    python occlusion_analysis.py --folds 0 1 2 3 4 [--limit N]
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import nibabel as nib
import pandas as pd
import torch
from torch.utils.data import DataLoader
from scipy.stats import ttest_rel, wilcoxon
from nilearn import datasets, image as nimg

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))
from dataset import MultiModalDataset  # noqa: E402
from interpretability.ft_attention import (  # noqa: E402
    TABULAR_FEATURES, build_paper_model, regenerate_fold_splits,
)

REPO = Path("/home/tanguy/medical/alzheimer")
EXP = REPO / "experiments" / "multimodal_fusion"
OUT = EXP / "interpretability" / "results_new" / "06_occlusion"
OUT.mkdir(exist_ok=True, parents=True)

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


def build_masks():
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

    temporal = mk(sub_v, sl, MTL_SUB) | mk(cor_v, cl, TEMPORAL_COR) | \
               mk(cor_v, cl, [
                   "Parahippocampal Gyrus, anterior division",
                   "Parahippocampal Gyrus, posterior division",
                   "Temporal Fusiform Cortex, anterior division",
                   "Temporal Fusiform Cortex, posterior division",
               ])
    frontal = mk(cor_v, cl, FRONTAL_COR)
    brain = (sub_v > 0) | (cor_v > 0)
    return (torch.from_numpy(temporal).bool(),
            torch.from_numpy(frontal).bool(),
            torch.from_numpy(brain).bool())


def occlude(mri, mask_gpu, brain_mask_gpu):
    """Replace voxels inside mask with per-subject brain-mean intensity.

    mri: (B, 1, 128, 128, 128) float tensor on GPU
    mask_gpu: (128, 128, 128) bool tensor on GPU
    brain_mask_gpu: (128, 128, 128) bool tensor on GPU
    """
    B = mri.shape[0]
    out = mri.clone()
    # per-sample brain mean over non-background voxels (broader than HO brain)
    for b in range(B):
        vol = mri[b, 0]
        brain_vals = vol[brain_mask_gpu]
        if brain_vals.numel() == 0:
            mean_val = vol.mean()
        else:
            mean_val = brain_vals.mean()
        out[b, 0][mask_gpu] = mean_val
    return out


@torch.no_grad()
def run_fold(fold, seed, device, temporal_mask, frontal_mask, brain_mask, limit=None):
    ckpt = EXP / "cv_results" / f"seed_{seed}" / f"fold_{fold}" / "model.pth"
    scaler_path = EXP / "cv_results" / f"seed_{seed}" / f"fold_{fold}" / "scaler.pkl"
    _, _, test_df = regenerate_fold_splits(seed, fold)
    if limit:
        test_df = test_df.head(limit)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    tmp_csv = OUT / f"_tmp_fold{fold}.csv"
    test_df.to_csv(tmp_csv, index=False)
    ds = MultiModalDataset(
        str(tmp_csv),
        tabular_features=TABULAR_FEATURES,
        target_shape=TARGET_SHAPE,
        augment=False,
        normalize_tabular=True,
        scaler=scaler,
        use_paper_preprocessing=True,
        target_spacing=1.75,
    )
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=6, pin_memory=True)

    model = build_paper_model()
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(sd, strict=False)
    model = model.to(device).eval()

    temp_gpu = temporal_mask.to(device)
    front_gpu = frontal_mask.to(device)
    brain_gpu = brain_mask.to(device)

    records = []
    global_idx = 0
    for mri, tab, y in loader:
        mri = mri.to(device, non_blocking=True)
        tab = tab.to(device, non_blocking=True)
        # Baseline
        out = model(mri, tab)
        logits = out[0] if isinstance(out, tuple) else out
        p_base = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        # Temporal occlusion
        mri_t = occlude(mri, temp_gpu, brain_gpu)
        out = model(mri_t, tab)
        logits = out[0] if isinstance(out, tuple) else out
        p_temp = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        # Frontal occlusion
        mri_f = occlude(mri, front_gpu, brain_gpu)
        out = model(mri_f, tab)
        logits = out[0] if isinstance(out, tuple) else out
        p_front = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()

        for i in range(len(y)):
            records.append({
                "fold": fold,
                "subject_id": test_df.iloc[global_idx]["subject_id"],
                "label": int(y[i]),
                "p_base": float(p_base[i]),
                "p_temp_occl": float(p_temp[i]),
                "p_front_occl": float(p_front[i]),
                "delta_temp": float(p_temp[i] - p_base[i]),
                "delta_front": float(p_front[i] - p_base[i]),
            })
            global_idx += 1
        print(f"  fold {fold}: {global_idx}/{len(test_df)} done", end="\r")

    tmp_csv.unlink()
    print()
    return records


def summarize(all_records):
    df = pd.DataFrame(all_records)
    df.to_csv(OUT / "occlusion_results.csv", index=False)
    print(f"\n[*] Saved per-subject results: {OUT / 'occlusion_results.csv'}  (N={len(df)})")

    for label_name, label in [("true-AD", 1), ("true-CN", 0)]:
        sub = df[df["label"] == label]
        n = len(sub)
        dt = sub["delta_temp"].values
        df_v = sub["delta_front"].values
        t, p = ttest_rel(dt, df_v)
        w, wp = wilcoxon(dt, df_v)
        print(f"\n=== {label_name} (N={n}) ===")
        print(f"  Δ AD-prob (temporal occluded):  mean={dt.mean():+.4f}  median={np.median(dt):+.4f}  std={dt.std(ddof=1):.4f}")
        print(f"  Δ AD-prob (frontal occluded):   mean={df_v.mean():+.4f}  median={np.median(df_v):+.4f}  std={df_v.std(ddof=1):.4f}")
        print(f"  paired-t(Δtemp vs Δfront): t={t:+.3f} p={p:.3e}")
        print(f"  Wilcoxon:                  W={w:.1f} p={wp:.3e}")
        print(f"  Δtemp more negative than Δfront: {(dt < df_v).sum()}/{n} subjects")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--folds", type=int, nargs="+", default=[0])
    ap.add_argument("--limit", type=int, default=None,
                    help="Limit subjects per fold (for quick test)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Device: {device}")

    print("[*] Building ROI masks from Harvard-Oxford atlas...")
    temporal, frontal, brain = build_masks()
    print(f"    Temporal: {temporal.sum().item()} vox ({temporal.float().mean().item()*100:.2f}% of 128³)")
    print(f"    Frontal:  {frontal.sum().item()} vox ({frontal.float().mean().item()*100:.2f}%)")
    print(f"    Brain:    {brain.sum().item()} vox ({brain.float().mean().item()*100:.2f}%)")

    all_records = []
    for fold in args.folds:
        print(f"\n===== FOLD {fold} =====")
        recs = run_fold(fold, args.seed, device, temporal, frontal, brain, args.limit)
        all_records.extend(recs)

    summarize(all_records)
    print("\n[*] Done.")


if __name__ == "__main__":
    main()

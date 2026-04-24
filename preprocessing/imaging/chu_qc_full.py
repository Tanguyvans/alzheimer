#!/usr/bin/env python3
"""
Full visual+quantitative QC for CHU preprocessed scans.

For each skull-stripped subject:
  - Measures alignment vs MNI brain mask (Dice).
  - Measures intensity similarity vs MNI T1 template (normalized correlation
    over voxels belonging to both brain masks).
  - Measures brain volume (mL) and intensity percentiles.
  - Flags outliers with pass/warn/fail thresholds.
  - Writes one PNG per subject showing 9 slices (3 sag + 3 cor + 3 ax)
    overlaid on the MNI template — so misalignments are easy to spot.
  - Writes a summary CSV and an overview PNG.

Usage:
    python preprocessing/imaging/chu_qc_full.py \
        --work /home/tanguy/Desktop/irm_chu_work
"""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MNI_DIR = PROJECT_ROOT / (
    "mni_template/mni_icbm152_nlin_sym_09a_nifti/mni_icbm152_nlin_sym_09a"
)
MNI_T1 = MNI_DIR / "mni_icbm152_t1_tal_nlin_sym_09a.nii"
MNI_MASK = MNI_DIR / "mni_icbm152_t1_tal_nlin_sym_09a_mask.nii"

# Thresholds (empirical, typical for ANTs SyN registration)
DICE_PASS = 0.92
DICE_WARN = 0.85
NCC_PASS = 0.55
NCC_WARN = 0.35
VOL_MIN, VOL_MAX = 1100, 1900  # mL, adult range with meningeal voxels


def load_arr(p: Path) -> np.ndarray:
    return sitk.GetArrayFromImage(sitk.ReadImage(str(p)))


def resample_to(ref_img_path: Path, moving_path: Path) -> np.ndarray:
    """Resample moving to ref grid (both should already be MNI space but may
    differ in bounding box / spacing slightly)."""
    ref = sitk.ReadImage(str(ref_img_path))
    mov = sitk.ReadImage(str(moving_path))
    if ref.GetSize() == mov.GetSize() and ref.GetSpacing() == mov.GetSpacing():
        return sitk.GetArrayFromImage(mov)
    resampled = sitk.Resample(mov, ref, sitk.Transform(),
                              sitk.sitkLinear, 0.0, mov.GetPixelID())
    return sitk.GetArrayFromImage(resampled)


def dice(a: np.ndarray, b: np.ndarray) -> float:
    num = 2.0 * np.logical_and(a, b).sum()
    den = a.sum() + b.sum()
    return num / den if den > 0 else 0.0


def ncc(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    a = a[mask].astype(np.float64)
    b = b[mask].astype(np.float64)
    a -= a.mean()
    b -= b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else 0.0


def percentile_vmax(arr, p=99):
    nz = arr[arr > 0]
    return float(np.percentile(nz, p)) if nz.size else 1.0


def three_slice_indices(shape, axis, frac=(0.4, 0.5, 0.6)):
    n = shape[axis]
    return [int(n * f) for f in frac]


def _collect_views(subj_arr, tmpl_arr):
    """Return list of (axis_name, idx, t_view, s_view) for the 3x3 grid."""
    out = []
    for axis in [2, 1, 0]:  # sag (x), cor (y), ax (z)
        name_axis = ["axial", "coronal", "sagittal"][axis]
        idxs = three_slice_indices(subj_arr.shape, axis)
        for idx in idxs:
            if axis == 2:
                t_view = np.flipud(tmpl_arr[:, :, idx])
                s_view = np.flipud(subj_arr[:, :, idx])
            elif axis == 1:
                t_view = np.flipud(tmpl_arr[:, idx, :])
                s_view = np.flipud(subj_arr[:, idx, :])
            else:
                t_view = tmpl_arr[idx, :, :]
                s_view = subj_arr[idx, :, :]
            out.append((name_axis, idx, t_view, s_view))
    return out


def make_subject_qc_png(name, subj_arr, tmpl_arr, out_overlay, out_clean):
    """Two 3x3 PNGs: one with MNI overlay, one with just the scan."""
    views = _collect_views(subj_arr, tmpl_arr)
    vmax_t = percentile_vmax(tmpl_arr)
    vmax_s = percentile_vmax(subj_arr)

    # --- overlay version ---
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    for i, (axis_name, idx, t_view, s_view) in enumerate(views):
        ax = axes[i // 3, i % 3]
        ax.imshow(t_view, cmap="gray", vmin=0, vmax=vmax_t)
        mask = s_view > 0
        overlay = np.zeros((*s_view.shape, 4))
        overlay[..., 0] = 1.0
        overlay[..., 3] = mask * 0.4
        ax.imshow(overlay)
        ax.contour(mask.astype(np.uint8), levels=[0.5],
                   colors="yellow", linewidths=0.6)
        ax.set_axis_off()
        ax.set_title(f"{axis_name} @ {idx}", fontsize=8)
    fig.suptitle(f"{name} — overlay on MNI", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_overlay, dpi=85, bbox_inches="tight")
    plt.close(fig)

    # --- clean version (just the skull-stripped scan) ---
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    for i, (axis_name, idx, _, s_view) in enumerate(views):
        ax = axes[i // 3, i % 3]
        ax.imshow(s_view, cmap="gray", vmin=0, vmax=vmax_s)
        ax.set_axis_off()
        ax.set_title(f"{axis_name} @ {idx}", fontsize=8)
    fig.suptitle(f"{name} — skull-stripped", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_clean, dpi=85, bbox_inches="tight")
    plt.close(fig)


def classify(dice_val, ncc_val, vol_ml):
    flags = []
    if dice_val < DICE_WARN:
        flags.append("FAIL-dice")
    elif dice_val < DICE_PASS:
        flags.append("WARN-dice")
    if ncc_val < NCC_WARN:
        flags.append("FAIL-ncc")
    elif ncc_val < NCC_PASS:
        flags.append("WARN-ncc")
    if vol_ml < VOL_MIN or vol_ml > VOL_MAX:
        flags.append("WARN-vol")
    return "PASS" if not flags else ",".join(flags)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True, type=Path)
    args = ap.parse_args()

    skull = args.work / "skull"
    qc = args.work / "qc"
    qc_per_subj = qc / "per_subject"
    qc.mkdir(parents=True, exist_ok=True)
    qc_per_subj.mkdir(parents=True, exist_ok=True)

    print(f"Loading MNI template + mask ({MNI_T1.name})")
    tmpl_arr = load_arr(MNI_T1)
    mni_mask_arr = load_arr(MNI_MASK) > 0.5
    print(f"  MNI shape={tmpl_arr.shape}  brain voxels={mni_mask_arr.sum()}")

    # Find all subjects
    subjects = []
    for d in sorted(skull.iterdir()):
        if not d.is_dir():
            continue
        fs = sorted(d.glob("*_skull_stripped.nii.gz"))
        if fs:
            subjects.append((d.name, fs[0]))
    print(f"{len(subjects)} subjects to QC\n")

    rows = []
    for name, p in subjects:
        subj_arr = load_arr(p)
        # Sanity: shapes match MNI grid?
        same_grid = subj_arr.shape == tmpl_arr.shape
        if not same_grid:
            # Resample to MNI grid for fair comparison
            subj_arr_on_mni = resample_to(MNI_T1, p)
        else:
            subj_arr_on_mni = subj_arr

        subj_mask = subj_arr_on_mni > 0
        d_val = dice(subj_mask, mni_mask_arr)

        intersect = subj_mask & mni_mask_arr
        n_val = ncc(subj_arr_on_mni, tmpl_arr, intersect) if intersect.any() else 0.0

        vol_ml = subj_mask.sum() * 0.001  # 1mm iso
        p99 = float(np.percentile(subj_arr[subj_arr > 0], 99))
        flag = classify(d_val, n_val, vol_ml)

        rows.append({
            "subject": name,
            "shape_matches_mni": same_grid,
            "dice_vs_mni_mask": round(d_val, 4),
            "ncc_vs_mni_t1": round(n_val, 4),
            "brain_vol_ml": round(vol_ml, 1),
            "intensity_p99": round(p99, 1),
            "flag": flag,
        })

        print(f"  {name:<10}  dice={d_val:.3f}  ncc={n_val:.3f}  "
              f"vol={vol_ml:.0f} mL  p99={p99:.0f}  -> {flag}")

        # Per-subject QC PNGs (overlay + clean)
        out_overlay = qc_per_subj / f"{name}_overlay.png"
        out_clean = qc_per_subj / f"{name}_clean.png"
        make_subject_qc_png(name, subj_arr_on_mni, tmpl_arr,
                            out_overlay, out_clean)

    # Write CSV
    csv_path = qc / "chu_qc_full.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Summary
    print(f"\nWrote CSV: {csv_path}")
    print(f"Wrote {len(rows)} per-subject PNGs in {qc_per_subj}/")
    n_pass = sum(1 for r in rows if r["flag"] == "PASS")
    print(f"\nRESULT: {n_pass}/{len(rows)} PASS")
    for r in rows:
        if r["flag"] != "PASS":
            print(f"  {r['subject']}: {r['flag']}  "
                  f"(dice={r['dice_vs_mni_mask']}, ncc={r['ncc_vs_mni_t1']}, "
                  f"vol={r['brain_vol_ml']} mL)")


if __name__ == "__main__":
    main()

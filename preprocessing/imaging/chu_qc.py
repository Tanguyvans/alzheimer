#!/usr/bin/env python3
"""
Visual QC for CHU preprocessed scans.

Produces two artifacts in <work>/qc/:
  - chu_qc_grid.png : one row per subject, 3 orthogonal slices (sag/cor/ax)
                      of the skull-stripped scan.
  - chu_qc_overlay.png : same slices but overlaid on the MNI template to
                         visually confirm registration alignment.

Usage:
    python preprocessing/imaging/chu_qc.py --work /home/tanguy/Desktop/irm_chu_work
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_TEMPLATE = PROJECT_ROOT / (
    "mni_template/mni_icbm152_nlin_sym_09a_nifti/"
    "mni_icbm152_nlin_sym_09a/mni_icbm152_t1_tal_nlin_sym_09a.nii"
)


def load_volume(p: Path) -> np.ndarray:
    img = sitk.ReadImage(str(p))
    arr = sitk.GetArrayFromImage(img)  # (z, y, x)
    return arr


def percentile_clip(arr, lo=1, hi=99):
    nz = arr[arr > 0]
    if nz.size == 0:
        return arr
    vmin, vmax = np.percentile(nz, [lo, hi])
    return np.clip(arr, 0, vmax), vmin, vmax


def three_views(arr):
    """Return central sagittal, coronal, axial slices."""
    z, y, x = arr.shape
    sag = arr[:, :, x // 2]
    cor = arr[:, y // 2, :]
    axi = arr[z // 2, :, :]
    # Orient so superior is up for sag/cor (flip z)
    return np.flipud(sag), np.flipud(cor), axi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True, type=Path)
    ap.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    args = ap.parse_args()

    skull = args.work / "skull"
    out = args.work / "qc"
    out.mkdir(parents=True, exist_ok=True)

    subjects = sorted([d for d in skull.iterdir() if d.is_dir()])
    files = []
    for d in subjects:
        niftis = sorted(d.glob("*_skull_stripped.nii.gz"))
        if niftis:
            files.append((d.name, niftis[0]))
    if not files:
        print("No skull-stripped files found.", file=sys.stderr)
        sys.exit(1)

    print(f"{len(files)} subjects to QC")

    # --- Grid 1: skull-stripped only ---
    n = len(files)
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    if n == 1:
        axes = axes[np.newaxis, :]
    for i, (name, p) in enumerate(files):
        arr = load_volume(p)
        clipped, vmin, vmax = percentile_clip(arr)
        sag, cor, axi = three_views(clipped)
        for j, (title, view) in enumerate(zip(["sag", "cor", "ax"],
                                              [sag, cor, axi])):
            ax = axes[i, j]
            ax.imshow(view, cmap="gray", vmin=0, vmax=vmax)
            ax.set_axis_off()
            if i == 0:
                ax.set_title(title, fontsize=10)
            if j == 0:
                ax.text(-5, view.shape[0] // 2, name, rotation=90,
                        va="center", ha="right", fontsize=9)
    plt.tight_layout()
    grid_path = out / "chu_qc_grid.png"
    plt.savefig(grid_path, dpi=90, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {grid_path}")

    # --- Grid 2: overlay on MNI template ---
    tmpl_arr = load_volume(args.template)
    tmpl_clip, _, tvmax = percentile_clip(tmpl_arr)
    t_sag, t_cor, t_axi = three_views(tmpl_clip)

    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    if n == 1:
        axes = axes[np.newaxis, :]
    for i, (name, p) in enumerate(files):
        arr = load_volume(p)
        clipped, _, vmax = percentile_clip(arr)
        sag, cor, axi = three_views(clipped)
        for j, (t_view, s_view) in enumerate(zip([t_sag, t_cor, t_axi],
                                                 [sag, cor, axi])):
            ax = axes[i, j]
            ax.imshow(t_view, cmap="gray", vmin=0, vmax=tvmax)
            mask = s_view > 0
            overlay = np.zeros((*s_view.shape, 4))
            overlay[..., 0] = 1.0           # red
            overlay[..., 3] = mask * 0.45   # alpha where brain
            ax.imshow(overlay)
            ax.set_axis_off()
            if i == 0:
                ax.set_title(["sag", "cor", "ax"][j], fontsize=10)
            if j == 0:
                ax.text(-5, t_view.shape[0] // 2, name, rotation=90,
                        va="center", ha="right", fontsize=9)
    plt.tight_layout()
    overlay_path = out / "chu_qc_overlay.png"
    plt.savefig(overlay_path, dpi=90, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {overlay_path}")

    # --- Summary stats ---
    print("\nPer-subject stats:")
    print(f'{"subject":<10} {"vol_ml":<8} {"p99":<8} {"nz%":<6}')
    for name, p in files:
        arr = load_volume(p)
        nz = arr > 0
        # 1mm iso → each voxel is 1 mm^3 = 0.001 mL
        vol_ml = nz.sum() * 0.001
        p99 = np.percentile(arr[nz], 99) if nz.any() else 0
        print(f"{name:<10} {vol_ml:<8.1f} {p99:<8.1f} {nz.mean()*100:<6.1f}")


if __name__ == "__main__":
    main()

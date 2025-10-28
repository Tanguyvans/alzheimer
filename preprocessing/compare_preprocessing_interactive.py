#!/usr/bin/env python3
"""
Interactive comparison viewer for preprocessing methods: SynthStrip vs NPPY

Shows side-by-side views with synchronized sliders to explore all slices.

Usage:
    python3 compare_preprocessing_interactive.py --patient-id 035_S_6948
"""

import argparse
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
from pathlib import Path


def load_scan(scan_path: Path):
    """Load NIfTI scan and return 3D array"""
    nifti_img = nib.load(scan_path)
    return nifti_img.get_fdata()


def main():
    parser = argparse.ArgumentParser(description='Interactive preprocessing comparison viewer')
    parser.add_argument('--patient-id', default='035_S_6948',
                       help='Patient ID to visualize')
    parser.add_argument('--skull-dir', default='/Volumes/KINGSTON/ADNI-skull',
                       help='Directory with SynthStrip preprocessed scans')
    parser.add_argument('--nppy-dir', default='/Volumes/KINGSTON/ADNI_nppy',
                       help='Directory with NPPY preprocessed scans')

    args = parser.parse_args()

    skull_dir = Path(args.skull_dir)
    nppy_dir = Path(args.nppy_dir)
    patient_id = args.patient_id

    print(f"Loading scans for patient: {patient_id}")

    # Find SynthStrip scan
    skull_patient_dir = skull_dir / patient_id
    skull_files = [f for f in skull_patient_dir.glob("*.nii.gz") if not f.name.startswith('.')]

    if len(skull_files) == 0:
        print(f"✗ No SynthStrip scan found for {patient_id}")
        return

    skull_scan_path = skull_files[0]
    print(f"✓ SynthStrip: {skull_scan_path.name}")

    # Find NPPY scan
    nppy_patient_dir = nppy_dir / patient_id
    nppy_files = list(nppy_patient_dir.glob("*_mni_norm.nii.gz"))

    if len(nppy_files) == 0:
        print(f"✗ No NPPY scan found for {patient_id}")
        return

    nppy_scan_path = nppy_files[0]
    print(f"✓ NPPY: {nppy_scan_path.name}")

    # Load scans
    print("\nLoading volumes...")
    skull_data = load_scan(skull_scan_path)
    nppy_data = load_scan(nppy_scan_path)

    print(f"  SynthStrip shape: {skull_data.shape}")
    print(f"  NPPY shape: {nppy_data.shape}")

    # Print intensity statistics
    skull_min, skull_max = skull_data.min(), skull_data.max()
    skull_mean, skull_std = skull_data[skull_data > 0].mean(), skull_data[skull_data > 0].std()

    nppy_min, nppy_max = nppy_data.min(), nppy_data.max()
    nppy_mean, nppy_std = nppy_data[nppy_data > 0].mean(), nppy_data[nppy_data > 0].std()

    print(f"\n  SynthStrip intensity: [{skull_min:.2f}, {skull_max:.2f}], mean={skull_mean:.2f}, std={skull_std:.2f}")
    print(f"  NPPY intensity: [{nppy_min:.2f}, {nppy_max:.2f}], mean={nppy_mean:.2f}, std={nppy_std:.2f}")

    # Create figure with 2 rows (SynthStrip top, NPPY bottom) and 3 columns (axial, coronal, sagittal)
    fig = plt.figure(figsize=(18, 12))

    # Title
    fig.suptitle(f'Interactive Preprocessing Comparison: {patient_id}\n'
                 f'Top Row: SynthStrip+ANTs | Bottom Row: NPPY',
                 fontsize=16, fontweight='bold')

    # Create subplots
    # Row 1: SynthStrip
    ax_skull_axial = plt.subplot(2, 3, 1)
    ax_skull_coronal = plt.subplot(2, 3, 2)
    ax_skull_sagittal = plt.subplot(2, 3, 3)

    # Row 2: NPPY
    ax_nppy_axial = plt.subplot(2, 3, 4)
    ax_nppy_coronal = plt.subplot(2, 3, 5)
    ax_nppy_sagittal = plt.subplot(2, 3, 6)

    # Initial slice positions (middle of each dimension)
    skull_z_init = skull_data.shape[2] // 2
    skull_y_init = skull_data.shape[1] // 2
    skull_x_init = skull_data.shape[0] // 2

    nppy_z_init = nppy_data.shape[2] // 2
    nppy_y_init = nppy_data.shape[1] // 2
    nppy_x_init = nppy_data.shape[0] // 2

    # Display initial slices - SynthStrip
    img_skull_axial = ax_skull_axial.imshow(skull_data[:, :, skull_z_init].T, cmap='gray', origin='lower')
    ax_skull_axial.set_title('SynthStrip - Axial', fontweight='bold')
    ax_skull_axial.axis('off')

    img_skull_coronal = ax_skull_coronal.imshow(skull_data[:, skull_y_init, :].T, cmap='gray', origin='lower')
    ax_skull_coronal.set_title('SynthStrip - Coronal', fontweight='bold')
    ax_skull_coronal.axis('off')

    img_skull_sagittal = ax_skull_sagittal.imshow(skull_data[skull_x_init, :, :], cmap='gray', origin='lower')
    ax_skull_sagittal.set_title('SynthStrip - Sagittal', fontweight='bold')
    ax_skull_sagittal.axis('off')

    # Display initial slices - NPPY
    img_nppy_axial = ax_nppy_axial.imshow(nppy_data[:, :, nppy_z_init].T, cmap='gray', origin='lower')
    ax_nppy_axial.set_title('NPPY - Axial', fontweight='bold')
    ax_nppy_axial.axis('off')

    img_nppy_coronal = ax_nppy_coronal.imshow(nppy_data[:, nppy_y_init, :].T, cmap='gray', origin='lower')
    ax_nppy_coronal.set_title('NPPY - Coronal', fontweight='bold')
    ax_nppy_coronal.axis('off')

    img_nppy_sagittal = ax_nppy_sagittal.imshow(nppy_data[nppy_x_init, :, :], cmap='gray', origin='lower')
    ax_nppy_sagittal.set_title('NPPY - Sagittal', fontweight='bold')
    ax_nppy_sagittal.axis('off')

    # Add sliders at the bottom
    plt.subplots_adjust(bottom=0.25)

    # Axial slider (z-axis)
    ax_slider_axial = plt.axes([0.15, 0.15, 0.7, 0.02])
    slider_axial = Slider(
        ax_slider_axial,
        'Axial Slice',
        0,
        min(skull_data.shape[2], nppy_data.shape[2]) - 1,
        valinit=min(skull_z_init, nppy_z_init),
        valstep=1
    )

    # Coronal slider (y-axis)
    ax_slider_coronal = plt.axes([0.15, 0.10, 0.7, 0.02])
    slider_coronal = Slider(
        ax_slider_coronal,
        'Coronal Slice',
        0,
        min(skull_data.shape[1], nppy_data.shape[1]) - 1,
        valinit=min(skull_y_init, nppy_y_init),
        valstep=1
    )

    # Sagittal slider (x-axis)
    ax_slider_sagittal = plt.axes([0.15, 0.05, 0.7, 0.02])
    slider_sagittal = Slider(
        ax_slider_sagittal,
        'Sagittal Slice',
        0,
        min(skull_data.shape[0], nppy_data.shape[0]) - 1,
        valinit=min(skull_x_init, nppy_x_init),
        valstep=1
    )

    # Update functions for sliders
    def update_axial(val):
        pos = int(slider_axial.val)
        if pos < skull_data.shape[2]:
            img_skull_axial.set_array(skull_data[:, :, pos].T)
        if pos < nppy_data.shape[2]:
            img_nppy_axial.set_array(nppy_data[:, :, pos].T)
        fig.canvas.draw_idle()

    def update_coronal(val):
        pos = int(slider_coronal.val)
        if pos < skull_data.shape[1]:
            img_skull_coronal.set_array(skull_data[:, pos, :].T)
        if pos < nppy_data.shape[1]:
            img_nppy_coronal.set_array(nppy_data[:, pos, :].T)
        fig.canvas.draw_idle()

    def update_sagittal(val):
        pos = int(slider_sagittal.val)
        if pos < skull_data.shape[0]:
            img_skull_sagittal.set_array(skull_data[pos, :, :])
        if pos < nppy_data.shape[0]:
            img_nppy_sagittal.set_array(nppy_data[pos, :, :])
        fig.canvas.draw_idle()

    # Connect sliders to update functions
    slider_axial.on_changed(update_axial)
    slider_coronal.on_changed(update_coronal)
    slider_sagittal.on_changed(update_sagittal)

    # Add intensity statistics text
    stats_text = (
        f"SynthStrip: range=[{skull_min:.1f}, {skull_max:.1f}], mean={skull_mean:.1f}, std={skull_std:.1f}\n"
        f"NPPY: range=[{nppy_min:.1f}, {nppy_max:.1f}], mean={nppy_mean:.1f}, std={nppy_std:.1f}"
    )
    fig.text(0.5, 0.23, stats_text, ha='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    print("\n✓ Interactive viewer opened. Use sliders to explore different slices.")
    print("  Close the window when done.")

    plt.show()


if __name__ == '__main__':
    main()

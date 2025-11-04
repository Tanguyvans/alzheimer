#!/usr/bin/env python3
"""
Compare Preprocessing Stages: ADNI_nifti vs ADNI_skull vs ADNI_nppy

Shows the same patient across all three preprocessing stages side-by-side.
Navigate through different patients using arrow keys.

Usage:
    python compare_preprocessing.py --patient 002_S_0295

Keyboard shortcuts:
    - Left Arrow: Previous patient
    - Right Arrow: Next patient
    - Up/Down: Change slice
"""

import argparse
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from pathlib import Path
import numpy as np


class PreprocessingComparisonViewer:
    """Compare preprocessing stages for the same patient"""

    def __init__(self, nifti_base, skull_base, nppy_base, patient_ids=None):
        self.nifti_base = Path(nifti_base)
        self.skull_base = Path(skull_base)
        self.nppy_base = Path(nppy_base)

        # Get list of patients
        if patient_ids:
            self.patient_ids = patient_ids
        else:
            # Get all patients from ADNI_nifti
            self.patient_ids = sorted([d.name for d in self.nifti_base.iterdir() if d.is_dir() and not d.name.startswith('.')])

        self.current_patient_idx = 0

        # Try to load first patient
        loaded = False
        for idx in range(len(self.patient_ids)):
            if self.load_patient(idx):
                loaded = True
                break

        if not loaded:
            raise RuntimeError("No valid patients could be loaded")

        # Create figure with 3 rows x 3 columns
        self.fig = plt.figure(figsize=(20, 12))
        self.fig.suptitle('', fontsize=14, weight='bold')

        # Create subplots: 3 rows (NIFTI, SKULL, NPPY) x 3 columns (Axial, Coronal, Sagittal)
        self.axes = []
        self.images = []

        row_labels = ['ADNI_nifti (Raw)', 'ADNI_skull (SynthStrip+ANTs)', 'ADNI_nppy (NPPY)']
        col_labels = ['Axial', 'Coronal', 'Sagittal']

        for i, row_label in enumerate(row_labels):
            row_axes = []
            row_images = []
            for j, col_label in enumerate(col_labels):
                ax = plt.subplot(3, 3, i*3 + j + 1)
                ax.set_title(f'{row_label}\n{col_label}')
                ax.axis('off')
                row_axes.append(ax)
                row_images.append(None)
            self.axes.append(row_axes)
            self.images.append(row_images)

        plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.12, hspace=0.3, wspace=0.1)

        # Add slice slider
        ax_slider = plt.axes([0.15, 0.05, 0.7, 0.03])
        self.slider = Slider(ax_slider, 'Slice', 0, 100, valinit=50, valstep=1)
        self.slider.on_changed(self.update_slice)

        # Navigation instructions
        nav_text = "Navigation: ← → (change patient) | ↑ ↓ (change slice) | Slider (scroll through slices)"
        self.fig.text(0.5, 0.01, nav_text, ha='center', va='bottom', fontsize=10, style='italic')

        # Connect keyboard
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

        # Initial display
        self.display_patient()

    def find_scan_file(self, patient_dir, pattern='*.nii.gz'):
        """Find first scan matching pattern in patient directory"""
        scans = list(patient_dir.glob(pattern))
        scans = [s for s in scans if not s.name.startswith('.')]
        return scans[0] if scans else None

    def load_patient(self, patient_idx):
        """Load all three preprocessing stages for a patient"""
        if patient_idx < 0 or patient_idx >= len(self.patient_ids):
            return False

        patient_id = self.patient_ids[patient_idx]
        print(f"\nLoading patient {patient_idx + 1}/{len(self.patient_ids)}: {patient_id}")

        # Find scan files
        nifti_dir = self.nifti_base / patient_id
        skull_dir = self.skull_base / patient_id
        nppy_dir = self.nppy_base / patient_id

        # ADNI_nifti: any .nii.gz file
        nifti_scan = self.find_scan_file(nifti_dir) if nifti_dir.exists() else None

        # ADNI_skull: *_registered_skull_stripped.nii.gz
        skull_scan = self.find_scan_file(skull_dir, '*_registered_skull_stripped.nii.gz') if skull_dir.exists() else None

        # ADNI_nppy: *_mni_norm.nii.gz
        nppy_scan = self.find_scan_file(nppy_dir, '*_mni_norm.nii.gz') if nppy_dir.exists() else None

        # Check if at least one scan exists
        if not any([nifti_scan, skull_scan, nppy_scan]):
            print(f"  ⚠️  No scans found for {patient_id}")
            return False

        # Load the scans
        self.nifti_data = None
        self.skull_data = None
        self.nppy_data = None

        try:
            if nifti_scan:
                self.nifti_data = nib.load(str(nifti_scan)).get_fdata()
                print(f"  ✓ NIFTI: {nifti_scan.name} - Shape: {self.nifti_data.shape}, Range: [{self.nifti_data.min():.1f}, {self.nifti_data.max():.1f}]")
            else:
                print(f"  ✗ NIFTI: Not found")

            if skull_scan:
                self.skull_data = nib.load(str(skull_scan)).get_fdata()
                print(f"  ✓ SKULL: {skull_scan.name} - Shape: {self.skull_data.shape}, Range: [{self.skull_data.min():.1f}, {self.skull_data.max():.1f}]")
            else:
                print(f"  ✗ SKULL: Not found")

            if nppy_scan:
                self.nppy_data = nib.load(str(nppy_scan)).get_fdata()
                print(f"  ✓ NPPY: {nppy_scan.name} - Shape: {self.nppy_data.shape}, Range: [{self.nppy_data.min():.1f}, {self.nppy_data.max():.1f}]")
            else:
                print(f"  ✗ NPPY: Not found")

            self.current_patient_idx = patient_idx
            self.current_patient_id = patient_id
            return True

        except Exception as e:
            print(f"  ❌ Error loading: {e}")
            return False

    def display_patient(self):
        """Display all three preprocessing stages"""
        # Get max dimension across all axes to set slider range
        max_dim = 0
        for data in [self.nifti_data, self.skull_data, self.nppy_data]:
            if data is not None:
                max_dim = max(max_dim, max(data.shape))

        if max_dim == 0:
            return

        # Set slider range based on smallest dimension to avoid out of bounds
        min_dim = float('inf')
        for data in [self.nifti_data, self.skull_data, self.nppy_data]:
            if data is not None:
                min_dim = min(min_dim, min(data.shape))

        self.slider.valmax = min_dim - 1
        self.slider.set_val(min_dim // 2)
        self.slider.ax.set_xlim(0, min_dim - 1)

        # Update title
        self.fig.suptitle(f'Patient: {self.current_patient_id} ({self.current_patient_idx + 1}/{len(self.patient_ids)})',
                         fontsize=14, weight='bold')

        self.update_slice(self.slider.val)

    def update_slice(self, val):
        """Update displayed slice - slider controls axial slice position"""
        pos = int(val)

        data_list = [self.nifti_data, self.skull_data, self.nppy_data]
        row_labels = ['ADNI_nifti (Raw)', 'ADNI_skull (SynthStrip+ANTs)', 'ADNI_nppy (NPPY)']
        col_labels = ['Axial', 'Coronal', 'Sagittal']

        for row_idx, data in enumerate(data_list):
            if data is None:
                # Show blank
                for col_idx in range(3):
                    if self.images[row_idx][col_idx]:
                        self.images[row_idx][col_idx].remove()
                        self.images[row_idx][col_idx] = None
                    # Update title to show "Not found"
                    ax = self.axes[row_idx][col_idx]
                    ax.set_title(f'{row_labels[row_idx]}\n{col_labels[col_idx]}\nNot found', fontsize=9)
                continue

            # Ensure slice is within bounds
            z_pos = min(pos, data.shape[2] - 1)
            y_pos = min(pos, data.shape[1] - 1)  # Use slider pos for coronal too
            x_pos = min(pos, data.shape[0] - 1)  # Use slider pos for sagittal too

            # Get slices - use same position for all views
            axial = data[:, :, z_pos].T
            coronal = data[:, y_pos, :].T
            sagittal = data[x_pos, :, :]

            slices = [axial, coronal, sagittal]

            for col_idx, slice_data in enumerate(slices):
                ax = self.axes[row_idx][col_idx]

                # Remove old image
                if self.images[row_idx][col_idx]:
                    self.images[row_idx][col_idx].remove()

                # Display new image
                vmin, vmax = data.min(), data.max()
                img = ax.imshow(slice_data, cmap='gray', origin='lower', aspect='equal', vmin=vmin, vmax=vmax)
                self.images[row_idx][col_idx] = img

                # Update title with slice info and total slices
                view_name = col_labels[col_idx]
                if col_idx == 0:  # Axial
                    slice_info = f"z={z_pos}/{data.shape[2]-1} ({data.shape[2]} slices)"
                elif col_idx == 1:  # Coronal
                    slice_info = f"y={y_pos}/{data.shape[1]-1} ({data.shape[1]} slices)"
                else:  # Sagittal
                    slice_info = f"x={x_pos}/{data.shape[0]-1} ({data.shape[0]} slices)"

                ax.set_title(f'{row_labels[row_idx]}\n{view_name}\n{slice_info}', fontsize=8)

        self.fig.canvas.draw_idle()

    def change_patient(self, direction):
        """Change to next/previous patient"""
        new_idx = self.current_patient_idx + direction

        # Try to load patients in the direction
        for _ in range(len(self.patient_ids)):
            if 0 <= new_idx < len(self.patient_ids):
                if self.load_patient(new_idx):
                    self.display_patient()
                    return
                new_idx += direction
            else:
                break

        if direction > 0:
            print(f"Already at last patient ({self.current_patient_idx + 1}/{len(self.patient_ids)})")
        else:
            print(f"Already at first patient (1/{len(self.patient_ids)})")

    def on_key_press(self, event):
        """Handle keyboard events"""
        if event.key == 'right':
            self.change_patient(1)
        elif event.key == 'left':
            self.change_patient(-1)
        elif event.key == 'up':
            new_val = min(self.slider.val + 1, self.slider.valmax)
            self.slider.set_val(new_val)
        elif event.key == 'down':
            new_val = max(self.slider.val - 1, 0)
            self.slider.set_val(new_val)

    def show(self):
        """Display the viewer"""
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Compare preprocessing stages for ADNI patients',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare all patients
  python compare_preprocessing.py

  # Start at specific patient
  python compare_preprocessing.py --patient 002_S_0295

  # Compare specific list of patients
  python compare_preprocessing.py --patient-list experiments/cn_mci_ad_3dhcct/required_patients.txt

Keyboard shortcuts:
  Left Arrow  : Previous patient
  Right Arrow : Next patient
  Up Arrow    : Next slice (scroll through all views)
  Down Arrow  : Previous slice (scroll through all views)
  Slider      : Scroll through slices
        """
    )

    parser.add_argument('--nifti-dir', type=str, default='/Volumes/KINGSTON/ADNI_nifti',
                       help='ADNI_nifti directory')
    parser.add_argument('--skull-dir', type=str, default='/Volumes/KINGSTON/ADNI-skull',
                       help='ADNI-skull directory (SynthStrip+ANTs)')
    parser.add_argument('--nppy-dir', type=str, default='/Volumes/KINGSTON/ADNI_nppy',
                       help='ADNI_nppy directory (NPPY)')
    parser.add_argument('--patient', type=str, default=None,
                       help='Start at specific patient ID')
    parser.add_argument('--patient-list', type=str, default=None,
                       help='Text file with patient IDs to compare (one per line)')

    args = parser.parse_args()

    # Load patient list if provided
    patient_ids = None
    if args.patient_list:
        with open(args.patient_list, 'r') as f:
            patient_ids = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(patient_ids)} patients from {args.patient_list}")

    # Create viewer
    viewer = PreprocessingComparisonViewer(
        args.nifti_dir,
        args.skull_dir,
        args.nppy_dir,
        patient_ids
    )

    # Jump to specific patient if requested
    if args.patient:
        try:
            idx = viewer.patient_ids.index(args.patient)
            viewer.load_patient(idx)
            viewer.display_patient()
        except ValueError:
            print(f"⚠️  Patient {args.patient} not found in list")

    viewer.show()


if __name__ == '__main__':
    main()

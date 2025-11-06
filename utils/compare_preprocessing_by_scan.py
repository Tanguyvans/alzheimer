#!/usr/bin/env python3
"""
Compare Preprocessing Stages by Individual Scan: ADNI_nifti vs ADNI_skull vs ADNI_nppy

Shows individual MRI scans across all three preprocessing stages side-by-side.
Navigate through different scans using arrow keys.

Usage:
    python compare_preprocessing_by_scan.py --scan-list experiments/cn_mci_ad_3dhcct/required_scans.txt

Keyboard shortcuts:
    - Left Arrow: Previous scan
    - Right Arrow: Next scan
    - Up/Down: Change slice
"""

import argparse
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from pathlib import Path
import numpy as np


class ScanComparisonViewer:
    """Compare preprocessing stages for individual MRI scans"""

    def __init__(self, nifti_scans, skull_base, nppy_base):
        self.skull_base = Path(skull_base)
        self.nppy_base = Path(nppy_base)

        # Load scan list with full paths
        self.nifti_scans = [Path(scan) for scan in nifti_scans]
        self.current_scan_idx = 0

        # Try to load first scan
        loaded = False
        for idx in range(len(self.nifti_scans)):
            if self.load_scan(idx):
                loaded = True
                break

        if not loaded:
            raise RuntimeError("No valid scans could be loaded")

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
        nav_text = "Navigation: ← → (change scan) | ↑ ↓ (change slice) | Slider (scroll through slices)"
        self.fig.text(0.5, 0.01, nav_text, ha='center', va='bottom', fontsize=10, style='italic')

        # Connect keyboard
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

        # Initial display
        self.display_scan()

    def find_skull_scan(self, patient_id):
        """Find corresponding skull-stripped scan"""
        skull_dir = self.skull_base / patient_id
        if not skull_dir.exists():
            return None

        scans = list(skull_dir.glob('*_registered_skull_stripped.nii.gz'))
        scans = [s for s in scans if not s.name.startswith('.')]
        return scans[0] if scans else None

    def find_nppy_scan(self, patient_id, nifti_scan_name):
        """Find corresponding NPPY scan by matching scan base name"""
        nppy_dir = self.nppy_base / patient_id
        if not nppy_dir.exists():
            return None

        # Extract base name from NIfTI scan (remove .0_I#####_#####.nii.gz suffix)
        # Example: MPRAGE_Repeat_2009-01-21_10_42_57.0_I134226_134226.nii.gz
        #       -> MPRAGE_Repeat_2009-01-21_10_42_57
        nifti_base = nifti_scan_name.replace('.nii.gz', '').replace('.nii', '')
        if '.0_I' in nifti_base:
            nifti_base = nifti_base.split('.0_I')[0]

        # Find NPPY scan with matching base name
        # NPPY names: {base}_mni_norm.nii.gz
        expected_nppy = f"{nifti_base}_mni_norm.nii.gz"
        nppy_path = nppy_dir / expected_nppy

        if nppy_path.exists():
            return nppy_path

        # Fallback: try to find any matching scan
        scans = list(nppy_dir.glob('*_mni_norm.nii.gz'))
        scans = [s for s in scans if not s.name.startswith('.')]
        for scan in scans:
            if nifti_base in scan.name:
                return scan

        return None

    def load_scan(self, scan_idx):
        """Load all three preprocessing stages for a scan"""
        if scan_idx < 0 or scan_idx >= len(self.nifti_scans):
            return False

        nifti_scan = self.nifti_scans[scan_idx]
        patient_id = nifti_scan.parent.name

        print(f"\nLoading scan {scan_idx + 1}/{len(self.nifti_scans)}")
        print(f"  Patient: {patient_id}")
        print(f"  Scan: {nifti_scan.name}")

        # Find corresponding scans
        skull_scan = self.find_skull_scan(patient_id)
        nppy_scan = self.find_nppy_scan(patient_id, nifti_scan.name)

        # Load the scans
        self.nifti_data = None
        self.skull_data = None
        self.nppy_data = None

        try:
            if nifti_scan.exists():
                self.nifti_data = nib.load(str(nifti_scan)).get_fdata()
                print(f"  ✓ NIFTI: Shape={self.nifti_data.shape}, Range=[{self.nifti_data.min():.1f}, {self.nifti_data.max():.1f}]")
            else:
                print(f"  ✗ NIFTI: Not found")

            if skull_scan:
                self.skull_data = nib.load(str(skull_scan)).get_fdata()
                print(f"  ✓ SKULL: Shape={self.skull_data.shape}, Range=[{self.skull_data.min():.1f}, {self.skull_data.max():.1f}]")
            else:
                print(f"  ✗ SKULL: Not found")

            if nppy_scan:
                self.nppy_data = nib.load(str(nppy_scan)).get_fdata()
                print(f"  ✓ NPPY: Shape={self.nppy_data.shape}, Range=[{self.nppy_data.min():.1f}, {self.nppy_data.max():.1f}]")
            else:
                print(f"  ✗ NPPY: Not found")

            self.current_scan_idx = scan_idx
            self.current_patient_id = patient_id
            self.current_scan_name = nifti_scan.name
            return True

        except Exception as e:
            print(f"  ❌ Error loading: {e}")
            return False

    def display_scan(self):
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

        # Update title with scan info
        self.fig.suptitle(f'Scan {self.current_scan_idx + 1}/{len(self.nifti_scans)}: {self.current_patient_id} - {self.current_scan_name}',
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
            y_pos = min(pos, data.shape[1] - 1)
            x_pos = min(pos, data.shape[0] - 1)

            # Get slices
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

                # Update title with slice info
                view_name = col_labels[col_idx]
                if col_idx == 0:  # Axial
                    slice_info = f"z={z_pos}/{data.shape[2]-1} ({data.shape[2]} slices)"
                elif col_idx == 1:  # Coronal
                    slice_info = f"y={y_pos}/{data.shape[1]-1} ({data.shape[1]} slices)"
                else:  # Sagittal
                    slice_info = f"x={x_pos}/{data.shape[0]-1} ({data.shape[0]} slices)"

                ax.set_title(f'{row_labels[row_idx]}\n{view_name}\n{slice_info}\n{data.shape}', fontsize=8)

        self.fig.canvas.draw_idle()

    def change_scan(self, direction):
        """Change to next/previous scan"""
        new_idx = self.current_scan_idx + direction

        # Try to load scans in the direction
        for _ in range(len(self.nifti_scans)):
            if 0 <= new_idx < len(self.nifti_scans):
                if self.load_scan(new_idx):
                    self.display_scan()
                    return
                new_idx += direction
            else:
                break

        if direction > 0:
            print(f"Already at last scan ({self.current_scan_idx + 1}/{len(self.nifti_scans)})")
        else:
            print(f"Already at first scan (1/{len(self.nifti_scans)})")

    def on_key_press(self, event):
        """Handle keyboard events"""
        if event.key == 'right':
            self.change_scan(1)
        elif event.key == 'left':
            self.change_scan(-1)
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
        description='Compare preprocessing stages for individual ADNI scans',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare scans from scan list
  python compare_preprocessing_by_scan.py --scan-list experiments/cn_mci_ad_3dhcct/required_scans.txt

Keyboard shortcuts:
  Left Arrow  : Previous scan
  Right Arrow : Next scan
  Up Arrow    : Next slice (scroll through all views)
  Down Arrow  : Previous slice (scroll through all views)
  Slider      : Scroll through slices
        """
    )

    parser.add_argument('--skull-dir', type=str, default='/Volumes/KINGSTON/ADNI-skull',
                       help='ADNI-skull directory (SynthStrip+ANTs)')
    parser.add_argument('--nppy-dir', type=str, default='/Volumes/KINGSTON/ADNI_nppy',
                       help='ADNI_nppy directory (NPPY)')
    parser.add_argument('--scan-list', type=str, required=True,
                       help='Text file with full scan paths (one per line)')

    args = parser.parse_args()

    # Load scan list
    with open(args.scan_list, 'r') as f:
        nifti_scans = [line.strip() for line in f if line.strip()]

    print(f"Loaded {len(nifti_scans)} scans from {args.scan_list}")

    # Create viewer
    viewer = ScanComparisonViewer(
        nifti_scans,
        args.skull_dir,
        args.nppy_dir
    )

    viewer.show()


if __name__ == '__main__':
    main()

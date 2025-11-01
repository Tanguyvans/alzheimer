#!/usr/bin/env python3
"""
Interactive MRI Viewer for Multiple Scans

Allows browsing through a list of scans using keyboard arrows.
Based on visualize.py but supports navigation between multiple MRI files.

Usage:
    python visualize_scan_list.py --scan-list experiments/cn_mci_ad_3dhcct/required_scans.txt

Keyboard shortcuts:
    - Left Arrow: Previous scan
    - Right Arrow: Next scan
    - Up/Down: Change slice within current scan
"""

import argparse
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from pathlib import Path


class MultiScanViewer:
    """Interactive viewer for navigating multiple MRI scans"""

    def __init__(self, scan_paths):
        self.scan_paths = [Path(p) for p in scan_paths]
        self.current_scan_idx = -1
        self.current_slice = None
        self.data = None
        self.img = None

        # Load first available scan
        loaded = False
        for idx in range(len(self.scan_paths)):
            if self.load_scan(idx):
                loaded = True
                break

        if not loaded:
            raise RuntimeError("No valid scans could be loaded")

        # Create figure
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(1, 3, figsize=(18, 6))
        plt.subplots_adjust(bottom=0.20, top=0.92)

        # Initial slice positions
        self.z_pos = self.data.shape[2] // 2
        self.y_pos = self.data.shape[1] // 2
        self.x_pos = self.data.shape[0] // 2

        # Create initial images
        self.img1 = self.ax1.imshow(self.data[:, :, self.z_pos].T, cmap='gray', origin='lower', aspect='equal')
        self.img2 = self.ax2.imshow(self.data[:, self.y_pos, :].T, cmap='gray', origin='lower', aspect='equal')
        self.img3 = self.ax3.imshow(self.data[self.x_pos, :, :], cmap='gray', origin='lower', aspect='equal')

        self.ax1.set_title('Axial')
        self.ax2.set_title('Coronal')
        self.ax3.set_title('Sagittal')

        # Add slice slider
        ax_slider = plt.axes([0.1, 0.08, 0.65, 0.03])
        self.slider = Slider(ax_slider, 'Slice', 0, self.data.shape[2]-1, valinit=self.z_pos, valstep=1)
        self.slider.on_changed(self.update_slice)

        # Add scan info text
        self.info_text = self.fig.text(0.5, 0.97, '', ha='center', va='top', fontsize=10, weight='bold')
        self.update_title()

        # Connect keyboard events
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

        # Add navigation instructions
        nav_text = "Navigation: ← → (change scan) | ↑ ↓ (change slice) | Slider (fine control)"
        self.fig.text(0.5, 0.02, nav_text, ha='center', va='bottom', fontsize=9, style='italic')

    def load_scan(self, scan_idx):
        """Load a scan by index"""
        if 0 <= scan_idx < len(self.scan_paths):
            scan_path = self.scan_paths[scan_idx]

            if not scan_path.exists():
                print(f"⚠️  Scan not found: {scan_path}")
                return False

            print(f"Loading [{scan_idx + 1}/{len(self.scan_paths)}]: {scan_path.name}")

            try:
                self.img = nib.load(str(scan_path))
                self.data = self.img.get_fdata()
                self.current_scan_idx = scan_idx

                # Print scan info
                print(f"  Shape: {self.data.shape}")
                print(f"  Spacing: {self.img.header.get_zooms()}")
                print(f"  Value range: [{self.data.min():.2f}, {self.data.max():.2f}]")

                return True

            except Exception as e:
                print(f"❌ Error loading scan: {e}")
                return False
        return False

    def update_title(self):
        """Update the title with current scan info"""
        scan_path = self.scan_paths[self.current_scan_idx]
        patient_id = scan_path.parent.name
        scan_name = scan_path.name

        title = f"Scan {self.current_scan_idx + 1}/{len(self.scan_paths)} | Patient: {patient_id} | {scan_name}"
        self.info_text.set_text(title)

    def update_slice(self, val):
        """Update the displayed slice"""
        pos = int(self.slider.val)
        self.img1.set_array(self.data[:, :, pos].T)
        self.img2.set_array(self.data[:, pos, :].T)
        self.img3.set_array(self.data[pos, :, :])
        self.fig.canvas.draw_idle()

    def change_scan(self, direction):
        """Change to next/previous scan, skipping missing/corrupted scans"""
        start_idx = self.current_scan_idx
        new_idx = start_idx + direction
        max_attempts = len(self.scan_paths)  # Prevent infinite loop
        attempts = 0

        while 0 <= new_idx < len(self.scan_paths) and attempts < max_attempts:
            if self.load_scan(new_idx):
                # Reset slice to middle
                self.z_pos = self.data.shape[2] // 2
                self.y_pos = self.data.shape[1] // 2
                self.x_pos = self.data.shape[0] // 2

                # Update slider range
                self.slider.valmax = self.data.shape[2] - 1
                self.slider.set_val(self.z_pos)
                self.slider.ax.set_xlim(0, self.data.shape[2] - 1)

                # Update images
                self.img1.set_array(self.data[:, :, self.z_pos].T)
                self.img2.set_array(self.data[:, self.y_pos, :].T)
                self.img3.set_array(self.data[self.x_pos, :, :])

                # Update color limits
                vmin, vmax = self.data.min(), self.data.max()
                self.img1.set_clim(vmin, vmax)
                self.img2.set_clim(vmin, vmax)
                self.img3.set_clim(vmin, vmax)

                self.update_title()
                self.fig.canvas.draw_idle()
                return

            # Try next scan in the same direction
            new_idx += direction
            attempts += 1

        # Reached end or beginning
        if new_idx >= len(self.scan_paths):
            print(f"Already at last scan ({self.current_scan_idx + 1}/{len(self.scan_paths)})")
        elif new_idx < 0:
            print(f"Already at first scan (1/{len(self.scan_paths)})")
        else:
            print(f"No more valid scans in that direction")

    def on_key_press(self, event):
        """Handle keyboard events"""
        if event.key == 'right':
            self.change_scan(1)  # Next scan
        elif event.key == 'left':
            self.change_scan(-1)  # Previous scan
        elif event.key == 'up':
            # Increase slice
            new_val = min(self.slider.val + 1, self.slider.valmax)
            self.slider.set_val(new_val)
        elif event.key == 'down':
            # Decrease slice
            new_val = max(self.slider.val - 1, 0)
            self.slider.set_val(new_val)

    def show(self):
        """Display the viewer"""
        plt.show()


def load_scan_list(scan_list_file):
    """Load list of scan paths from file, filtering out missing scans"""
    scans = []
    missing_count = 0

    with open(scan_list_file, 'r') as f:
        for line in f:
            scan_path = line.strip()
            if scan_path:
                if Path(scan_path).exists():
                    scans.append(scan_path)
                else:
                    missing_count += 1

    if missing_count > 0:
        print(f"⚠️  Skipped {missing_count} missing scans")

    return scans


def main():
    parser = argparse.ArgumentParser(
        description='Interactive viewer for multiple MRI scans',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # View all scans from cn_mci_ad_3dhcct experiment
  python visualize_scan_list.py --scan-list experiments/cn_mci_ad_3dhcct/required_scans.txt

  # View first 10 scans only
  python visualize_scan_list.py --scan-list experiments/cn_mci_ad_3dhcct/required_scans.txt --limit 10

Keyboard shortcuts:
  Left Arrow  : Previous scan
  Right Arrow : Next scan
  Up Arrow    : Next slice
  Down Arrow  : Previous slice
  Slider      : Fine slice control
        """
    )

    parser.add_argument('--scan-list', type=str, required=True,
                       help='Text file with scan paths (one per line)')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of scans to load (for testing)')
    parser.add_argument('--start-index', type=int, default=0,
                       help='Start at specific scan index')

    args = parser.parse_args()

    # Load scan list
    print(f"Loading scan list from: {args.scan_list}")
    scan_paths = load_scan_list(args.scan_list)

    if args.limit:
        scan_paths = scan_paths[:args.limit]

    if args.start_index > 0:
        if args.start_index >= len(scan_paths):
            print(f"❌ Start index {args.start_index} exceeds number of scans ({len(scan_paths)})")
            return
        scan_paths = scan_paths[args.start_index:]

    print(f"Loaded {len(scan_paths)} scan paths")

    if len(scan_paths) == 0:
        print("❌ No scans to display")
        return

    # Create and show viewer
    viewer = MultiScanViewer(scan_paths)
    viewer.show()


if __name__ == '__main__':
    main()

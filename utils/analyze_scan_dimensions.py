#!/usr/bin/env python3
"""
Analyze and visualize dimension distribution of MRI scans

Creates histogram and cumulative distribution plots showing how many scans
fall into different dimension ranges (0-20, 20-40, etc.).

Usage:
    python utils/analyze_scan_dimensions.py \
        --scan-list experiments/cn_mci_ad_3dhcct/required_scans.txt \
        --output experiments/cn_mci_ad_3dhcct/dimension_distribution.png
"""

import argparse
import nibabel as nib
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def analyze_dimensions(scan_list_file: Path, output_plot: Path = None, bin_size: int = 20):
    """
    Analyze dimension distribution of MRI scans

    Args:
        scan_list_file: Path to text file with scan paths
        output_plot: Path to save plot (optional)
        bin_size: Size of bins for histogram (default: 20)
    """
    # Load scan list
    logger.info(f"Loading scan list from: {scan_list_file}")
    with open(scan_list_file, 'r') as f:
        scans = [Path(line.strip()) for line in f if line.strip()]

    logger.info(f"Analyzing dimension distribution for {len(scans)} scans...")

    # Collect minimum dimensions
    min_dimensions = []
    errors = 0

    for scan_path in tqdm(scans, desc="Loading scans"):
        try:
            img = nib.load(str(scan_path))
            data = img.get_fdata()

            # Squeeze out singleton dimensions (e.g., 4D with shape[3]==1 -> 3D)
            data = np.squeeze(data)
            shape = data.shape

            # Get minimum dimension (should be 3D after squeeze)
            if len(shape) == 3:
                min_dim = min(shape)
            elif len(shape) == 4:
                # Still 4D after squeeze (e.g., fMRI with multiple volumes)
                min_dim = min(shape[:3])
            else:
                # Unexpected dimensions
                min_dim = min(shape) if len(shape) > 0 else 0

            min_dimensions.append(min_dim)

        except Exception as e:
            logger.error(f"Error loading {scan_path.name}: {e}")
            min_dimensions.append(0)
            errors += 1

    if errors > 0:
        logger.warning(f"Failed to load {errors} scans")

    # Create bins: 0-20, 20-40, 40-60, ..., up to max
    max_dim = max(min_dimensions)
    max_bin = ((max_dim // bin_size) + 1) * bin_size
    bins = list(range(0, max_bin + bin_size, bin_size))
    hist, bin_edges = np.histogram(min_dimensions, bins=bins)

    # Print distribution table
    logger.info("="*80)
    logger.info("MINIMUM DIMENSION DISTRIBUTION")
    logger.info("="*80)
    print(f"{'Range':<15} {'Count':<10} {'Percentage':<12} {'Cumulative'}")
    print(f"{'-'*80}")

    cumulative = 0
    for i in range(len(hist)):
        count = hist[i]
        cumulative += count
        pct = (count / len(scans)) * 100
        cum_pct = (cumulative / len(scans)) * 100
        print(f"{bin_edges[i]:3d}-{bin_edges[i+1]:3d}       "
              f"{count:5d}      {pct:5.1f}%        {cum_pct:5.1f}%")

    print(f"{'-'*80}")
    print(f"Total:        {len(scans):5d}      100.0%")
    logger.info("="*80)

    # Statistics
    logger.info("\nDimension Statistics:")
    logger.info(f"  Min: {min(min_dimensions)}")
    logger.info(f"  Max: {max(min_dimensions)}")
    logger.info(f"  Mean: {np.mean(min_dimensions):.1f}")
    logger.info(f"  Median: {np.median(min_dimensions):.1f}")
    logger.info(f"  Std: {np.std(min_dimensions):.1f}")

    # Filter analysis at common thresholds
    logger.info("\nFilter Analysis:")
    for threshold in [50, 80, 100, 120, 150]:
        kept = sum(1 for d in min_dimensions if d >= threshold)
        pct = (kept / len(scans)) * 100
        logger.info(f"  min_dim >= {threshold:3d}: {kept:5d} scans ({pct:5.1f}%)")

    # Create visualization if output path provided
    if output_plot:
        create_plot(hist, bin_edges, min_dimensions, len(scans), output_plot, bin_size)
        logger.info(f"\n✓ Saved plot to: {output_plot}")

    return min_dimensions, hist, bin_edges


def create_plot(hist, bin_edges, min_dimensions, total_scans, output_path, bin_size):
    """Create and save visualization plots"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Histogram
    ax1.bar(range(len(hist)), hist, width=0.8, edgecolor='black', alpha=0.7, color='steelblue')
    ax1.set_xlabel('Minimum Dimension Range', fontsize=12)
    ax1.set_ylabel('Number of Scans', fontsize=12)
    ax1.set_title('Distribution of Minimum Dimensions\n(ADNI_nifti Scans)',
                  fontsize=14, weight='bold')
    ax1.set_xticks(range(len(hist)))
    ax1.set_xticklabels([f'{int(bin_edges[i])}-{int(bin_edges[i+1])}'
                         for i in range(len(hist))], rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for i, v in enumerate(hist):
        if v > 0:
            ax1.text(i, v + total_scans * 0.01, str(v),
                    ha='center', va='bottom', fontsize=9)

    # Add threshold line at 100
    threshold_idx = 100 // bin_size
    if threshold_idx < len(hist):
        ax1.axvline(x=threshold_idx, color='red', linestyle='--',
                   linewidth=2, alpha=0.7, label='Threshold (min_dim=100)')
        ax1.text(threshold_idx, ax1.get_ylim()[1] * 0.95,
                'min_dim=100\n(filter threshold)',
                rotation=90, va='top', ha='right', color='red',
                fontsize=10, weight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax1.legend()

    # Cumulative distribution
    cumulative_pct = np.cumsum(hist) / total_scans * 100
    ax2.plot(range(len(cumulative_pct)), cumulative_pct,
            marker='o', linewidth=2, markersize=6, color='steelblue')
    ax2.fill_between(range(len(cumulative_pct)), 0, cumulative_pct, alpha=0.3)
    ax2.set_xlabel('Minimum Dimension Range', fontsize=12)
    ax2.set_ylabel('Cumulative Percentage (%)', fontsize=12)
    ax2.set_title('Cumulative Distribution\n(Percentage of scans with min_dim in range)',
                  fontsize=14, weight='bold')
    ax2.set_xticks(range(len(hist)))
    ax2.set_xticklabels([f'{int(bin_edges[i])}-{int(bin_edges[i+1])}'
                         for i in range(len(hist))], rotation=45, ha='right')
    ax2.grid(alpha=0.3)
    ax2.set_ylim([0, 105])

    # Add horizontal line showing filter cutoff
    if threshold_idx > 0 and threshold_idx < len(cumulative_pct):
        filter_pct = cumulative_pct[threshold_idx - 1]
        kept_pct = 100 - filter_pct

        ax2.axhline(y=filter_pct, color='red', linestyle='--', linewidth=2, alpha=0.5)
        ax2.axvline(x=threshold_idx, color='red', linestyle='--', linewidth=2, alpha=0.5)
        ax2.text(threshold_idx + 0.3, filter_pct + 2,
                f'{filter_pct:.1f}% removed\n{kept_pct:.1f}% kept',
                color='red', fontsize=10, weight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def scan_directory(input_dir: Path) -> list:
    """Find all NIfTI scans in directory"""
    logger.info(f"Scanning directory: {input_dir}")
    scans = []

    for patient_dir in tqdm(sorted(input_dir.iterdir()), desc="Finding scans"):
        if not patient_dir.is_dir() or patient_dir.name.startswith('.'):
            continue

        for scan in patient_dir.glob('*.nii.gz'):
            if not scan.name.startswith('.'):
                scans.append(scan)

    logger.info(f"Found {len(scans)} scans in {input_dir}")
    return scans


def main():
    parser = argparse.ArgumentParser(
        description='Analyze and visualize dimension distribution of MRI scans',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze all scans in a directory
  python utils/analyze_scan_dimensions.py \
    --input-dir /Volumes/KINGSTON/ADNI_nifti \
    --output adni_dimension_distribution.png

  # Analyze specific scan list
  python utils/analyze_scan_dimensions.py \
    --scan-list experiments/cn_mci_ad_3dhcct/required_scans.txt \
    --output experiments/cn_mci_ad_3dhcct/dimension_distribution.png

  # Use different bin size (e.g., 10 instead of 20)
  python utils/analyze_scan_dimensions.py \
    --input-dir /Volumes/KINGSTON/ADNI_nifti \
    --output dimension_dist.png \
    --bin-size 10
        """
    )

    parser.add_argument('--input-dir', type=str, default=None,
                       help='Input directory with NIfTI scans (scans all *.nii.gz files)')
    parser.add_argument('--scan-list', type=str, default=None,
                       help='Text file with scan paths (one per line)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for plot (optional)')
    parser.add_argument('--bin-size', type=int, default=20,
                       help='Bin size for histogram (default: 20)')

    args = parser.parse_args()

    if not args.input_dir and not args.scan_list:
        logger.error("Either --input-dir or --scan-list must be provided")
        return

    if args.input_dir and args.scan_list:
        logger.error("Cannot use both --input-dir and --scan-list")
        return

    # Get scan paths
    if args.input_dir:
        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            logger.error(f"Input directory not found: {input_dir}")
            return

        # Find all scans
        scans = scan_directory(input_dir)

        # Create temporary scan list file
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            for scan in scans:
                f.write(f"{scan}\n")
            temp_file = Path(f.name)

        scan_list_file = temp_file
    else:
        scan_list_file = Path(args.scan_list)
        if not scan_list_file.exists():
            logger.error(f"Scan list file not found: {scan_list_file}")
            return

    output_plot = Path(args.output) if args.output else None

    analyze_dimensions(scan_list_file, output_plot, args.bin_size)

    # Clean up temp file if created
    if args.input_dir:
        temp_file.unlink()


if __name__ == '__main__':
    main()

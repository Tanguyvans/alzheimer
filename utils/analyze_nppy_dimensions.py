#!/usr/bin/env python3
"""
Analyze NPPY Dimension Distribution and Quality

Analyzes the dimension distribution of NPPY-preprocessed scans to identify
which dimensions correlate with good vs poor preprocessing quality.

This helps determine which scan dimensions to keep vs filter out for training.

Usage:
    # Analyze all NPPY scans from required patients
    python utils/analyze_nppy_dimensions.py \
        --patient-list experiments/cn_mci_ad_3dhcct/required_patients.txt \
        --nifti-dir /Volumes/KINGSTON/ADNI_nifti \
        --nppy-dir /Volumes/KINGSTON/ADNI_nppy \
        --output nppy_dimension_analysis.png

    # Also save dimension statistics
    python utils/analyze_nppy_dimensions.py \
        --patient-list experiments/cn_mci_ad_3dhcct/required_patients.txt \
        --nifti-dir /Volumes/KINGSTON/ADNI_nifti \
        --nppy-dir /Volumes/KINGSTON/ADNI_nppy \
        --output nppy_dimension_analysis.png \
        --stats-output nppy_dimension_stats.txt
"""

import argparse
import nibabel as nib
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def find_scan_file(patient_dir, pattern='*.nii.gz'):
    """Find first scan matching pattern in patient directory"""
    if not patient_dir.exists():
        return None
    scans = list(patient_dir.glob(pattern))
    scans = [s for s in scans if not s.name.startswith('.')]
    return scans[0] if scans else None


def get_scan_dimensions(scan_path):
    """Get dimensions of a scan"""
    try:
        img = nib.load(str(scan_path))
        data = img.get_fdata()
        data = np.squeeze(data)
        return data.shape
    except Exception as e:
        logger.error(f"Error loading {scan_path.name}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description='Analyze NPPY dimension distribution and quality',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze all NPPY scans from required patients
  python utils/analyze_nppy_dimensions.py \\
    --patient-list experiments/cn_mci_ad_3dhcct/required_patients.txt \\
    --nifti-dir /Volumes/KINGSTON/ADNI_nifti \\
    --nppy-dir /Volumes/KINGSTON/ADNI_nppy \\
    --output nppy_dimension_analysis.png

  # Also save statistics
  python utils/analyze_nppy_dimensions.py \\
    --patient-list experiments/cn_mci_ad_3dhcct/required_patients.txt \\
    --nifti-dir /Volumes/KINGSTON/ADNI_nifti \\
    --nppy-dir /Volumes/KINGSTON/ADNI_nppy \\
    --output nppy_dimension_analysis.png \\
    --stats-output nppy_dimension_stats.txt
        """
    )

    parser.add_argument('--patient-list', type=str, required=True,
                       help='Text file with patient IDs (one per line)')
    parser.add_argument('--nifti-dir', type=str, required=True,
                       help='ADNI_nifti directory (for original dimensions)')
    parser.add_argument('--nppy-dir', type=str, required=True,
                       help='ADNI_nppy directory (for preprocessed scans)')
    parser.add_argument('--output', type=str, required=True,
                       help='Output plot file (e.g., nppy_dimension_analysis.png)')
    parser.add_argument('--stats-output', type=str, default=None,
                       help='Optional statistics output file')

    args = parser.parse_args()

    # Load patient list
    logger.info(f"Loading patient list from: {args.patient_list}")
    with open(args.patient_list, 'r') as f:
        patient_ids = [line.strip() for line in f if line.strip()]

    logger.info(f"Found {len(patient_ids)} patients to analyze")

    nifti_base = Path(args.nifti_dir)
    nppy_base = Path(args.nppy_dir)

    # Collect dimension data
    logger.info("Analyzing scan dimensions...")
    dimension_data = []
    dimension_counter = Counter()
    failed_loads = 0

    for patient_id in tqdm(patient_ids, desc="Analyzing patients"):
        # Find original NIFTI scan
        nifti_dir = nifti_base / patient_id
        nifti_scan = find_scan_file(nifti_dir)

        if not nifti_scan:
            failed_loads += 1
            continue

        # Get original dimensions
        nifti_shape = get_scan_dimensions(nifti_scan)
        if nifti_shape is None:
            failed_loads += 1
            continue

        # Find NPPY scan
        nppy_dir = nppy_base / patient_id
        nppy_scan = find_scan_file(nppy_dir, '*_mni_norm.nii.gz')

        nppy_exists = nppy_scan is not None
        nppy_shape = get_scan_dimensions(nppy_scan) if nppy_exists else None

        dimension_data.append({
            'patient_id': patient_id,
            'nifti_shape': nifti_shape,
            'nppy_shape': nppy_shape,
            'nppy_exists': nppy_exists,
            'scan_name': nifti_scan.name
        })

        dimension_counter[nifti_shape] += 1

    logger.info(f"Successfully analyzed {len(dimension_data)} patients")
    logger.info(f"Failed to load {failed_loads} patients")

    # Group by dimension
    dimension_groups = defaultdict(list)
    for entry in dimension_data:
        dimension_groups[entry['nifti_shape']].append(entry)

    # Sort dimensions by frequency
    sorted_dimensions = sorted(dimension_counter.items(), key=lambda x: x[1], reverse=True)

    # Print statistics
    logger.info("="*80)
    logger.info("DIMENSION DISTRIBUTION")
    logger.info("="*80)
    logger.info(f"{'Dimension':<20} {'Count':<10} {'Percentage':<12} {'NPPY Success'}")
    logger.info("-"*80)

    stats_lines = []
    stats_lines.append("="*80)
    stats_lines.append("NPPY DIMENSION ANALYSIS")
    stats_lines.append("="*80)
    stats_lines.append(f"Total patients analyzed: {len(dimension_data)}")
    stats_lines.append(f"Unique dimensions found: {len(dimension_counter)}")
    stats_lines.append("")
    stats_lines.append(f"{'Dimension':<20} {'Count':<10} {'Percentage':<12} {'NPPY Success'}")
    stats_lines.append("-"*80)

    for dim, count in sorted_dimensions:
        pct = 100 * count / len(dimension_data)
        # Count NPPY successes for this dimension
        nppy_success = sum(1 for e in dimension_groups[dim] if e['nppy_exists'])
        nppy_rate = 100 * nppy_success / count if count > 0 else 0

        line = f"{str(dim):<20} {count:<10} {pct:>6.2f}%      {nppy_success}/{count} ({nppy_rate:.1f}%)"
        logger.info(line)
        stats_lines.append(line)

    # Create visualization
    logger.info("\nCreating visualization...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Dimension frequency
    dims_str = [str(d) for d, _ in sorted_dimensions[:10]]  # Top 10
    counts = [c for _, c in sorted_dimensions[:10]]
    colors = plt.cm.viridis(np.linspace(0, 1, len(dims_str)))

    ax1.barh(dims_str, counts, color=colors)
    ax1.set_xlabel('Number of Scans', fontsize=12)
    ax1.set_ylabel('Original Scan Dimensions', fontsize=12)
    ax1.set_title('NPPY Input Dimension Distribution\n(Top 10 Most Common)', fontsize=14, weight='bold')
    ax1.grid(axis='x', alpha=0.3)

    # Add count labels
    for i, (dim_str, count) in enumerate(zip(dims_str, counts)):
        ax1.text(count, i, f' {count}', va='center', fontsize=10)

    # Plot 2: Dimension vs NPPY success rate
    success_rates = []
    for dim, count in sorted_dimensions[:10]:
        nppy_success = sum(1 for e in dimension_groups[dim] if e['nppy_exists'])
        success_rate = 100 * nppy_success / count if count > 0 else 0
        success_rates.append(success_rate)

    colors_success = ['green' if r == 100 else 'orange' if r >= 90 else 'red' for r in success_rates]

    ax2.barh(dims_str, success_rates, color=colors_success)
    ax2.set_xlabel('NPPY Success Rate (%)', fontsize=12)
    ax2.set_ylabel('Original Scan Dimensions', fontsize=12)
    ax2.set_title('NPPY Preprocessing Success Rate by Dimension', fontsize=14, weight='bold')
    ax2.set_xlim(0, 105)
    ax2.grid(axis='x', alpha=0.3)

    # Add percentage labels
    for i, rate in enumerate(success_rates):
        ax2.text(rate, i, f' {rate:.1f}%', va='center', fontsize=10)

    plt.tight_layout()

    # Save plot
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"\nPlot saved to: {output_path}")

    # Save statistics if requested
    if args.stats_output:
        stats_lines.append("")
        stats_lines.append("="*80)
        stats_lines.append("DETAILED BREAKDOWN BY DIMENSION")
        stats_lines.append("="*80)

        for dim, count in sorted_dimensions:
            stats_lines.append(f"\n{str(dim)} ({count} scans):")
            stats_lines.append("-"*40)

            entries = dimension_groups[dim]
            nppy_success = [e for e in entries if e['nppy_exists']]
            nppy_failed = [e for e in entries if not e['nppy_exists']]

            stats_lines.append(f"  NPPY Success: {len(nppy_success)}/{count}")
            stats_lines.append(f"  NPPY Failed:  {len(nppy_failed)}/{count}")

            if nppy_failed:
                stats_lines.append("\n  Failed patients:")
                for entry in nppy_failed[:5]:  # Show first 5
                    stats_lines.append(f"    - {entry['patient_id']}: {entry['scan_name']}")
                if len(nppy_failed) > 5:
                    stats_lines.append(f"    ... and {len(nppy_failed) - 5} more")

        stats_path = Path(args.stats_output)
        with open(stats_path, 'w') as f:
            f.write('\n'.join(stats_lines))
        logger.info(f"Statistics saved to: {stats_path}")

    logger.info("="*80)
    logger.info("ANALYSIS COMPLETE")
    logger.info("="*80)


if __name__ == '__main__':
    main()

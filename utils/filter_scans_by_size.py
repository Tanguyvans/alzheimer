#!/usr/bin/env python3
"""
Filter scans by file size to exclude problematic high-resolution acquisitions.

Analyzes scans from required_scans.txt and identifies those with unusually large
file sizes (>20MB) which often indicate newer "Accelerated_Sagittal_MPRAGE__MSV22"
protocol that doesn't work well with NPPY preprocessing.

Usage:
    python filter_scans_by_size.py --scan-list experiments/cn_mci_ad_3dhcct/required_scans.txt
    python filter_scans_by_size.py --scan-list experiments/cn_mci_ad_3dhcct/required_scans.txt --max-size 20
    python filter_scans_by_size.py --scan-list experiments/cn_mci_ad_3dhcct/required_scans.txt --output filtered_scans.txt
"""

import argparse
from pathlib import Path
import statistics


def analyze_scan_sizes(scan_paths):
    """Analyze file sizes of scans and return statistics"""
    scan_info = []

    for scan_path in scan_paths:
        scan_path_obj = Path(scan_path)
        if scan_path_obj.exists():
            size_mb = scan_path_obj.stat().st_size / (1024**2)
            patient_id = scan_path_obj.parent.name
            scan_info.append({
                'path': scan_path,
                'patient_id': patient_id,
                'filename': scan_path_obj.name,
                'size_mb': size_mb
            })
        else:
            print(f"Warning: File not found: {scan_path}")

    return scan_info


def print_statistics(scan_info):
    """Print file size statistics"""
    sizes = [s['size_mb'] for s in scan_info]

    print(f"\n{'='*80}")
    print("FILE SIZE STATISTICS")
    print(f"{'='*80}")
    print(f"Total scans: {len(sizes)}")
    print(f"Mean size: {statistics.mean(sizes):.2f} MB")
    print(f"Median size: {statistics.median(sizes):.2f} MB")
    print(f"Std dev: {statistics.stdev(sizes):.2f} MB")
    print(f"Min size: {min(sizes):.2f} MB")
    print(f"Max size: {max(sizes):.2f} MB")

    # Calculate outlier threshold
    mean = statistics.mean(sizes)
    std = statistics.stdev(sizes)
    threshold_2std = mean + 2 * std

    print(f"\nOutlier threshold (mean + 2*std): {threshold_2std:.2f} MB")

    return threshold_2std


def filter_scans_by_size(scan_info, max_size_mb):
    """Filter scans by maximum file size"""
    good_scans = []
    rejected_scans = []

    for scan in scan_info:
        if scan['size_mb'] <= max_size_mb:
            good_scans.append(scan)
        else:
            rejected_scans.append(scan)

    return good_scans, rejected_scans


def main():
    parser = argparse.ArgumentParser(
        description='Filter scans by file size to exclude problematic high-resolution acquisitions',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--scan-list', type=str, required=True,
                       help='Text file with scan paths (one per line)')
    parser.add_argument('--max-size', type=float, default=20.0,
                       help='Maximum file size in MB (default: 20.0)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file for filtered scans (default: append _filtered to input)')
    parser.add_argument('--output-rejected', type=str, default=None,
                       help='Output file for rejected scans (default: append _rejected to input)')
    parser.add_argument('--show-top', type=int, default=20,
                       help='Number of largest scans to display')

    args = parser.parse_args()

    # Read scan list
    with open(args.scan_list, 'r') as f:
        scan_paths = [line.strip() for line in f if line.strip()]

    print(f"Analyzing {len(scan_paths)} scans from: {args.scan_list}")

    # Analyze file sizes
    scan_info = analyze_scan_sizes(scan_paths)

    # Print statistics
    threshold_2std = print_statistics(scan_info)

    # Show largest scans
    scan_info_sorted = sorted(scan_info, key=lambda x: x['size_mb'], reverse=True)

    print(f"\n{'='*80}")
    print(f"TOP {args.show_top} LARGEST SCANS")
    print(f"{'='*80}")
    print(f"{'Patient':<15} {'Size (MB)':<12} {'Filename':<50}")
    print('-'*80)

    for scan in scan_info_sorted[:args.show_top]:
        filename = scan['filename'][:47] + '...' if len(scan['filename']) > 50 else scan['filename']
        print(f"{scan['patient_id']:<15} {scan['size_mb']:>10.1f}  {filename}")

    # Filter scans
    good_scans, rejected_scans = filter_scans_by_size(scan_info, args.max_size)

    print(f"\n{'='*80}")
    print(f"FILTERING RESULTS (max size: {args.max_size} MB)")
    print(f"{'='*80}")
    print(f"Scans kept: {len(good_scans)} ({100*len(good_scans)/len(scan_info):.1f}%)")
    print(f"Scans rejected: {len(rejected_scans)} ({100*len(rejected_scans)/len(scan_info):.1f}%)")

    # Determine output filenames
    if args.output is None:
        input_path = Path(args.scan_list)
        output_path = input_path.parent / f"{input_path.stem}_filtered.txt"
    else:
        output_path = Path(args.output)

    if args.output_rejected is None:
        input_path = Path(args.scan_list)
        rejected_path = input_path.parent / f"{input_path.stem}_rejected.txt"
    else:
        rejected_path = Path(args.output_rejected)

    # Save filtered scans
    with open(output_path, 'w') as f:
        for scan in good_scans:
            f.write(f"{scan['path']}\n")

    print(f"\n✓ Saved {len(good_scans)} filtered scans to: {output_path}")

    # Save rejected scans
    with open(rejected_path, 'w') as f:
        for scan in rejected_scans:
            f.write(f"{scan['path']}\n")

    print(f"✓ Saved {len(rejected_scans)} rejected scans to: {rejected_path}")

    # Show which patients were affected
    if rejected_scans:
        print(f"\n{'='*80}")
        print("REJECTED SCANS BY PATIENT")
        print(f"{'='*80}")

        rejected_by_patient = {}
        for scan in rejected_scans:
            patient_id = scan['patient_id']
            if patient_id not in rejected_by_patient:
                rejected_by_patient[patient_id] = []
            rejected_by_patient[patient_id].append(scan)

        print(f"Total patients affected: {len(rejected_by_patient)}")
        print(f"\nTop patients with rejected scans:")

        # Sort by size
        for patient_id, patient_scans in sorted(rejected_by_patient.items(),
                                                key=lambda x: x[1][0]['size_mb'],
                                                reverse=True)[:10]:
            scan = patient_scans[0]
            print(f"  {patient_id}: {scan['size_mb']:.1f} MB - {scan['filename'][:50]}")

        print(f"\n⚠️  WARNING: {len(rejected_by_patient)} patients now have NO scans!")
        print("These patients need alternative scans to be selected.")
        print("\nRecommendation: Re-run 00_get_required_patients.py with --max-file-size flag")


if __name__ == '__main__':
    main()

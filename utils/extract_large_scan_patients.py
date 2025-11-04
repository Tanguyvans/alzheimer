#!/usr/bin/env python3
"""
Extract patient IDs from large scans (>20MB) for quality inspection.

This creates a patient list file that can be used with compare_preprocessing.py
to visually inspect NPPY quality for problematic high-resolution scans.

Usage:
    python3 utils/extract_large_scan_patients.py \
        --scan-list experiments/cn_mci_ad_3dhcct/required_scans.txt \
        --max-size 20 \
        --output experiments/cn_mci_ad_3dhcct/large_scan_patients.txt
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description='Extract patient IDs from large scans for quality inspection'
    )
    parser.add_argument('--scan-list', type=str, required=True,
                       help='Text file with scan paths (one per line)')
    parser.add_argument('--max-size', type=float, default=20.0,
                       help='Maximum file size in MB (default: 20.0)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file for patient IDs (default: scan_list_large_patients.txt)')

    args = parser.parse_args()

    # Read required scans
    with open(args.scan_list, 'r') as f:
        scans = [line.strip() for line in f if line.strip()]

    print(f"Analyzing {len(scans)} scans from: {args.scan_list}")
    print(f"Looking for scans > {args.max_size} MB\n")

    # Find large scans
    large_scan_patients = []

    for scan_path in scans:
        scan_path_obj = Path(scan_path)
        if scan_path_obj.exists():
            size_mb = scan_path_obj.stat().st_size / (1024**2)
            if size_mb > args.max_size:
                patient_id = scan_path_obj.parent.name
                large_scan_patients.append((patient_id, size_mb, scan_path_obj.name))
        else:
            print(f"Warning: File not found: {scan_path}")

    # Sort by size (largest first)
    large_scan_patients.sort(key=lambda x: x[1], reverse=True)

    # Get unique patient IDs (preserving order by size)
    patient_ids = []
    seen = set()
    for patient_id, size_mb, filename in large_scan_patients:
        if patient_id not in seen:
            patient_ids.append(patient_id)
            seen.add(patient_id)

    # Determine output filename
    if args.output is None:
        input_path = Path(args.scan_list)
        output_file = input_path.parent / f"{input_path.stem}_large_patients.txt"
    else:
        output_file = Path(args.output)

    # Save patient IDs
    with open(output_file, 'w') as f:
        for patient_id in patient_ids:
            f.write(f'{patient_id}\n')

    # Print summary
    print(f"{'='*80}")
    print("LARGE SCAN SUMMARY")
    print(f"{'='*80}")
    print(f"Found {len(large_scan_patients)} scans >{args.max_size}MB from {len(patient_ids)} unique patients")
    print(f"\nTop 10 patients with largest scans:")
    for patient_id, size_mb, filename in large_scan_patients[:10]:
        display_name = filename[:55] + '...' if len(filename) > 58 else filename
        print(f"  {patient_id}: {size_mb:>5.1f} MB - {display_name}")

    print(f"\n✓ Saved {len(patient_ids)} patient IDs to: {output_file}")
    print(f"\nNext step: Inspect NPPY quality with:")
    print(f"  python3 utils/compare_preprocessing.py --patient-list {output_file}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()

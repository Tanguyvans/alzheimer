#!/usr/bin/env python3
"""
Convert 4D NIfTI scans to 3D by squeezing singleton dimensions

Some ADNI scans have shape (256, 256, 170, 1) instead of (256, 256, 170).
This script identifies such scans and creates squeezed 3D versions in-place.

Usage:
    # Process specific scan list
    python preprocessing/squeeze_4d_scans.py \
        --scan-list experiments/cn_mci_ad_3dhcct/scans_to_process.txt

    # Process entire directory
    python preprocessing/squeeze_4d_scans.py \
        --input-dir /Volumes/KINGSTON/ADNI_nifti

    # Dry run (don't modify files, just report)
    python preprocessing/squeeze_4d_scans.py \
        --scan-list scans.txt \
        --dry-run
"""

import argparse
import nibabel as nib
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def squeeze_scan(scan_path: Path, dry_run: bool = False) -> tuple:
    """
    Squeeze 4D scan to 3D if shape[3]==1

    Args:
        scan_path: Path to NIfTI file
        dry_run: If True, don't modify file

    Returns:
        (was_4d, success, original_shape, new_shape)
    """
    try:
        img = nib.load(str(scan_path))
        original_shape = img.shape

        # Check if 4D with singleton last dimension
        if len(original_shape) == 4 and original_shape[3] == 1:
            if dry_run:
                logger.info(f"[DRY RUN] Would squeeze: {scan_path.name} {original_shape} → {original_shape[:3]}")
                return (True, True, original_shape, original_shape[:3])

            # Get data and squeeze
            data = img.get_fdata()
            data_3d = np.squeeze(data)
            new_shape = data_3d.shape

            # Create backup
            backup_path = scan_path.with_suffix('.nii.gz.4d_backup')
            shutil.copy2(scan_path, backup_path)

            # Create new 3D image with same affine and header
            img_3d = nib.Nifti1Image(data_3d, img.affine, img.header)

            # Update header to reflect 3D dimensions
            img_3d.header.set_data_shape(new_shape)

            # Save squeezed version (overwrite original)
            nib.save(img_3d, str(scan_path))

            logger.info(f"✓ Squeezed: {scan_path.name} {original_shape} → {new_shape}")
            return (True, True, original_shape, new_shape)

        elif len(original_shape) == 3:
            # Already 3D, nothing to do
            return (False, True, original_shape, original_shape)

        else:
            # Unexpected dimensions
            logger.warning(f"⚠ Unexpected shape for {scan_path.name}: {original_shape}")
            return (False, False, original_shape, original_shape)

    except Exception as e:
        logger.error(f"✗ Error processing {scan_path.name}: {e}")
        return (False, False, None, None)


def main():
    parser = argparse.ArgumentParser(
        description='Convert 4D NIfTI scans to 3D by squeezing singleton dimensions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process specific scan list
  python preprocessing/squeeze_4d_scans.py \\
    --scan-list experiments/cn_mci_ad_3dhcct/scans_to_process.txt

  # Process entire directory
  python preprocessing/squeeze_4d_scans.py \\
    --input-dir /Volumes/KINGSTON/ADNI_nifti

  # Dry run (report only, don't modify)
  python preprocessing/squeeze_4d_scans.py \\
    --scan-list scans.txt \\
    --dry-run
        """
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--scan-list', type=str, help='Text file with scan paths (one per line)')
    group.add_argument('--input-dir', type=str, help='Directory to recursively search for NIfTI files')

    parser.add_argument('--dry-run', action='store_true',
                       help='Report what would be done without modifying files')

    args = parser.parse_args()

    # Collect scans to process
    scans = []

    if args.scan_list:
        logger.info(f"Loading scan list from: {args.scan_list}")
        scan_list_file = Path(args.scan_list)
        if not scan_list_file.exists():
            logger.error(f"Scan list file not found: {scan_list_file}")
            return

        with open(scan_list_file, 'r') as f:
            for line in f:
                scan_path = Path(line.strip())
                if scan_path.exists():
                    scans.append(scan_path)
                else:
                    logger.warning(f"Scan not found: {scan_path}")

    elif args.input_dir:
        logger.info(f"Scanning directory: {args.input_dir}")
        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            logger.error(f"Input directory not found: {input_dir}")
            return

        # Find all NIfTI files
        for patient_dir in tqdm(sorted(input_dir.iterdir()), desc="Finding scans"):
            if not patient_dir.is_dir() or patient_dir.name.startswith('.'):
                continue

            for scan in patient_dir.glob('*.nii.gz'):
                if not scan.name.startswith('.'):
                    scans.append(scan)

    logger.info(f"Found {len(scans)} scans to process")

    if args.dry_run:
        logger.info("="*80)
        logger.info("DRY RUN MODE - NO FILES WILL BE MODIFIED")
        logger.info("="*80)

    # Process scans
    logger.info("="*80)
    logger.info("PROCESSING SCANS")
    logger.info("="*80)

    squeezed_count = 0
    already_3d_count = 0
    error_count = 0

    for scan_path in tqdm(scans, desc="Processing"):
        was_4d, success, original_shape, new_shape = squeeze_scan(scan_path, dry_run=args.dry_run)

        if success:
            if was_4d:
                squeezed_count += 1
            else:
                already_3d_count += 1
        else:
            error_count += 1

    # Summary
    logger.info("="*80)
    logger.info("SUMMARY")
    logger.info("="*80)
    logger.info(f"Total scans processed: {len(scans)}")
    logger.info(f"  - Squeezed (4D → 3D): {squeezed_count}")
    logger.info(f"  - Already 3D: {already_3d_count}")
    logger.info(f"  - Errors: {error_count}")

    if not args.dry_run and squeezed_count > 0:
        logger.info(f"\n✓ Successfully squeezed {squeezed_count} scans")
        logger.info(f"  Backups saved with .4d_backup extension")
        logger.info(f"  To restore: mv file.nii.gz.4d_backup file.nii.gz")

    logger.info("="*80)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Create tar.gz archive of required baseline ADNI scans

Creates a compressed archive containing only the baseline scans needed
for the cn_mci_ad_3dhcct experiment, making it easy to transfer or backup.

Usage:
    python utils/create_baseline_archive.py \
        --scan-list experiments/cn_mci_ad_3dhcct/required_scans.txt \
        --output ADNI_nifti_baseline.tar.gz
"""

import argparse
import tarfile
from pathlib import Path
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_archive(scan_list_file: Path, output_archive: Path, compression: str = 'gz'):
    """
    Create tar archive of required scans

    Args:
        scan_list_file: Text file with scan paths (one per line)
        output_archive: Path to output tar.gz file
        compression: Compression type ('gz', 'bz2', 'xz', or '' for none)
    """
    # Load scan list
    logger.info(f"Loading scan list from: {scan_list_file}")
    scans = []
    missing_scans = []

    with open(scan_list_file, 'r') as f:
        for line in f:
            scan_path = Path(line.strip())
            if scan_path.exists():
                scans.append(scan_path)
            else:
                missing_scans.append(str(scan_path))

    logger.info(f"Found {len(scans)} scans to archive")

    if missing_scans:
        logger.warning(f"Skipping {len(missing_scans)} missing scans")

    # Calculate total size
    total_size = sum(scan.stat().st_size for scan in scans)
    total_size_gb = total_size / (1024**3)
    logger.info(f"Total uncompressed size: {total_size_gb:.2f} GB")

    # Create tar archive
    mode = f'w:{compression}' if compression else 'w'
    logger.info(f"Creating archive: {output_archive}")
    logger.info(f"Compression: {compression if compression else 'none'}")

    with tarfile.open(output_archive, mode) as tar:
        for scan_path in tqdm(scans, desc="Archiving scans"):
            # Get patient directory and scan file
            patient_id = scan_path.parent.name
            scan_name = scan_path.name

            # Archive structure: ADNI_nifti_baseline/<patient_id>/<scan>.nii.gz
            arcname = f"ADNI_nifti_baseline/{patient_id}/{scan_name}"

            # Add to archive
            tar.add(scan_path, arcname=arcname)

    # Get final archive size
    archive_size = output_archive.stat().st_size
    archive_size_gb = archive_size / (1024**3)
    compression_ratio = (1 - archive_size / total_size) * 100 if total_size > 0 else 0

    logger.info("\n" + "="*80)
    logger.info("ARCHIVE CREATED SUCCESSFULLY")
    logger.info("="*80)
    logger.info(f"Archive: {output_archive}")
    logger.info(f"Scans archived: {len(scans)}")
    logger.info(f"Uncompressed size: {total_size_gb:.2f} GB")
    logger.info(f"Compressed size: {archive_size_gb:.2f} GB")
    logger.info(f"Compression ratio: {compression_ratio:.1f}%")
    logger.info("="*80)

    # Save list of missing scans if any
    if missing_scans:
        missing_file = output_archive.parent / f"{output_archive.stem}_missing.txt"
        with open(missing_file, 'w') as f:
            for scan in missing_scans:
                f.write(f"{scan}\n")
        logger.warning(f"List of missing scans saved to: {missing_file}")

    # Print extraction command
    logger.info("\nTo extract the archive:")
    logger.info(f"  tar -xzf {output_archive.name}")


def main():
    parser = argparse.ArgumentParser(
        description='Create tar.gz archive of required baseline ADNI scans',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create compressed archive of baseline scans
  python utils/create_baseline_archive.py \
    --scan-list experiments/cn_mci_ad_3dhcct/required_scans.txt \
    --output ADNI_nifti_baseline.tar.gz

  # Create archive with bzip2 compression (better compression, slower)
  python utils/create_baseline_archive.py \
    --scan-list experiments/cn_mci_ad_3dhcct/required_scans.txt \
    --output ADNI_nifti_baseline.tar.bz2 \
    --compression bz2

  # Create uncompressed archive (faster, larger)
  python utils/create_baseline_archive.py \
    --scan-list experiments/cn_mci_ad_3dhcct/required_scans.txt \
    --output ADNI_nifti_baseline.tar \
    --compression none
        """
    )

    parser.add_argument('--scan-list', type=str, required=True,
                       help='Text file with scan paths (one per line)')
    parser.add_argument('--output', type=str, required=True,
                       help='Output archive path (e.g., ADNI_nifti_baseline.tar.gz)')
    parser.add_argument('--compression', type=str, default='gz',
                       choices=['gz', 'bz2', 'xz', 'none'],
                       help='Compression type (default: gz)')

    args = parser.parse_args()

    scan_list_file = Path(args.scan_list)
    output_archive = Path(args.output)

    if not scan_list_file.exists():
        logger.error(f"Scan list file not found: {scan_list_file}")
        return

    # Create output directory if needed
    output_archive.parent.mkdir(parents=True, exist_ok=True)

    # Determine compression from extension if not specified
    compression = args.compression
    if compression == 'none':
        compression = ''

    # Create archive
    create_archive(scan_list_file, output_archive, compression)


if __name__ == '__main__':
    main()

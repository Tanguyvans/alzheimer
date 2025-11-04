#!/usr/bin/env python3
"""
Batch preprocessing of ADNI scans using NPPY (Neural Pre-processing Python)

This script processes all NIfTI files in the ADNI dataset using NPPY Docker container.
NPPY performs skull stripping, intensity normalization, and spatial normalization.

Usage:
    python run_nppy_preprocessing.py \
        --input /Volumes/KINGSTON/ADNI_nifti \
        --output /Volumes/KINGSTON/ADNI_nppy \
        --nppy-script ~/nppy_docker.py
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_patient_filter_list(patient_list_file: Path) -> set:
    """
    Load list of patient IDs to process from file

    Args:
        patient_list_file: Path to text file with patient IDs (one per line)

    Returns:
        Set of patient IDs
    """
    with open(patient_list_file, 'r') as f:
        return {line.strip() for line in f if line.strip()}


def load_scan_list(scan_list_file: Path) -> list:
    """
    Load list of exact scan paths to process from file

    Args:
        scan_list_file: Path to text file with full scan paths (one per line)

    Returns:
        List of tuples (patient_id, Path(scan_path))
    """
    scans = []
    with open(scan_list_file, 'r') as f:
        for line in f:
            scan_path = Path(line.strip())
            if scan_path.exists():
                patient_id = scan_path.parent.name
                scans.append((patient_id, scan_path))
            else:
                logger.warning(f"Scan not found: {scan_path}")
    return scans


def find_all_nifti_files(input_dir: Path, patient_filter: set = None):
    """
    Find all NIfTI files in the input directory

    Args:
        input_dir: Root directory containing patient folders
        patient_filter: Optional set of patient IDs to filter (only process these patients)

    Returns:
        List of tuples (patient_id, nifti_path)
    """
    nifti_files = []

    for patient_dir in input_dir.iterdir():
        if not patient_dir.is_dir():
            continue

        # Skip hidden directories
        if patient_dir.name.startswith('.'):
            continue

        patient_id = patient_dir.name

        # Apply patient filter if provided
        if patient_filter and patient_id not in patient_filter:
            continue

        # Find NIfTI files in patient directory
        for nifti_file in patient_dir.glob("*.nii.gz"):
            # Skip hidden files
            if nifti_file.name.startswith('.'):
                continue

            nifti_files.append((patient_id, nifti_file))

    return nifti_files


def process_single_scan(nifti_path: Path, output_dir: Path, nppy_script: Path) -> bool:
    """
    Process a single scan with NPPY

    Args:
        nifti_path: Path to input NIfTI file
        output_dir: Directory for NPPY outputs
        nppy_script: Path to NPPY Docker wrapper script

    Returns:
        True if successful
    """
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run NPPY
    cmd = [
        'python3',
        str(nppy_script),
        '-i', str(nifti_path),
        '-o', str(output_dir) + '/'  # Must end with /
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout per scan
        )

        if result.returncode == 0:
            return True
        else:
            logger.error(f"NPPY failed for {nifti_path.name}")
            logger.error(f"STDERR: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        logger.error(f"NPPY timeout for {nifti_path.name}")
        return False
    except Exception as e:
        logger.error(f"Error processing {nifti_path.name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Batch NPPY preprocessing for ADNI dataset')
    parser.add_argument('--input', type=str, default=None, help='Input directory with ADNI NIfTI files (not needed if using --scan-list)')
    parser.add_argument('--output', required=True, help='Output directory for NPPY results')
    parser.add_argument('--nppy-script', default=os.path.expanduser('~/nppy_docker.py'),
                       help='Path to NPPY Docker wrapper script')
    parser.add_argument('--resume', action='store_true',
                       help='Skip already processed scans')
    parser.add_argument('--patient-list', type=str, default=None,
                       help='Text file with patient IDs to process (one per line). Only these patients will be processed.')
    parser.add_argument('--scan-list', type=str, default=None,
                       help='Text file with exact scan paths to process (one per line). RECOMMENDED for baseline-only processing.')

    args = parser.parse_args()

    output_dir = Path(args.output)
    nppy_script = Path(args.nppy_script)

    # Validate NPPY script
    if not nppy_script.exists():
        logger.error(f"NPPY script not found: {nppy_script}")
        sys.exit(1)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine which mode to use
    if args.scan_list:
        # Mode 1: Use exact scan list (RECOMMENDED)
        logger.info("="*80)
        logger.info("LOADING SCAN LIST")
        logger.info("="*80)

        scan_list_file = Path(args.scan_list)
        if not scan_list_file.exists():
            logger.error(f"Scan list file not found: {scan_list_file}")
            sys.exit(1)

        nifti_files = load_scan_list(scan_list_file)
        logger.info(f"Loaded {len(nifti_files)} scans from {scan_list_file}")

    else:
        # Mode 2: Scan directory with optional patient filter
        if not args.input:
            logger.error("Either --input or --scan-list must be provided")
            sys.exit(1)

        input_dir = Path(args.input)
        if not input_dir.exists():
            logger.error(f"Input directory not found: {input_dir}")
            sys.exit(1)

        # Load patient filter if provided
        patient_filter = None
        if args.patient_list:
            patient_list_file = Path(args.patient_list)
            if not patient_list_file.exists():
                logger.error(f"Patient list file not found: {patient_list_file}")
                sys.exit(1)

            patient_filter = load_patient_filter_list(patient_list_file)
            logger.info(f"Patient filter loaded: {len(patient_filter)} patients from {patient_list_file}")

        # Find all NIfTI files
        logger.info("="*80)
        logger.info("FINDING NIFTI FILES")
        logger.info("="*80)
        if patient_filter:
            logger.info(f"Filtering for {len(patient_filter)} specific patients")
        nifti_files = find_all_nifti_files(input_dir, patient_filter)

    logger.info(f"Found {len(nifti_files)} scans to process")

    # Process each scan
    logger.info("="*80)
    logger.info("RUNNING NPPY PREPROCESSING")
    logger.info("="*80)

    successful = 0
    failed = 0
    skipped = 0

    with tqdm(total=len(nifti_files), desc="Processing scans") as pbar:
        for patient_id, nifti_path in nifti_files:
            pbar.set_description(f"Processing {patient_id}")

            # Create patient output directory
            patient_output_dir = output_dir / patient_id

            # Check if already processed (for resume mode)
            if args.resume:
                # NPPY truncates scan names by removing .0_I#####_##### suffix
                # Example: MPRAGE_Repeat_2006-09-12_12_33_14.0_I24697_24697.nii.gz
                #       -> MPRAGE_Repeat_2006-09-12_12_33_14_mni_norm.nii.gz
                scan_name = nifti_path.stem.replace('.nii', '')
                scan_base = scan_name.split('.0_')[0] if '.0_' in scan_name else scan_name
                expected_output = patient_output_dir / f"{scan_base}_mni_norm.nii.gz"
                if expected_output.exists():
                    logger.debug(f"Skipping {patient_id} (already processed)")
                    skipped += 1
                    pbar.update(1)
                    continue

            # Process scan
            success = process_single_scan(nifti_path, patient_output_dir, nppy_script)

            if success:
                successful += 1
            else:
                failed += 1

            pbar.update(1)

    # Final summary
    logger.info("="*80)
    logger.info("NPPY PREPROCESSING COMPLETE")
    logger.info("="*80)
    logger.info(f"Total scans: {len(nifti_files)}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Skipped (already processed): {skipped}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("="*80)

    # Log failed scans
    if failed > 0:
        logger.warning(f"{failed} scans failed to process. Check logs above for details.")


if __name__ == '__main__':
    main()

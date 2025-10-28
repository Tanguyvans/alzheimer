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


def find_all_nifti_files(input_dir: Path):
    """
    Find all NIfTI files in the input directory

    Args:
        input_dir: Root directory containing patient folders

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
    parser.add_argument('--input', required=True, help='Input directory with ADNI NIfTI files')
    parser.add_argument('--output', required=True, help='Output directory for NPPY results')
    parser.add_argument('--nppy-script', default=os.path.expanduser('~/nppy_docker.py'),
                       help='Path to NPPY Docker wrapper script')
    parser.add_argument('--resume', action='store_true',
                       help='Skip already processed scans')

    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    nppy_script = Path(args.nppy_script)

    # Validate paths
    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        sys.exit(1)

    if not nppy_script.exists():
        logger.error(f"NPPY script not found: {nppy_script}")
        sys.exit(1)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all NIfTI files
    logger.info("="*80)
    logger.info("FINDING NIFTI FILES")
    logger.info("="*80)
    nifti_files = find_all_nifti_files(input_dir)
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
                expected_output = patient_output_dir / f"{nifti_path.stem.replace('.nii', '')}_mni_norm.nii.gz"
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

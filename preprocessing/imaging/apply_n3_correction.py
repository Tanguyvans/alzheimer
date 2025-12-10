#!/usr/bin/env python3
"""
Apply N3 Bias Field Correction to ADNI MPRAGE scans

According to ADNI documentation, N3 bias correction should be applied
before further preprocessing. This script applies N3 correction to raw
MPRAGE scans to prepare them for NPPY preprocessing.

Usage:
    python preprocessing/apply_n3_correction.py \
        --input /Volumes/KINGSTON/ADNI_nifti \
        --output /Volumes/KINGSTON/ADNI_nifti_n3 \
        --scan-list experiments/cn_mci_ad_3dhcct/required_scans.txt
"""

import argparse
import nibabel as nib
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging
import SimpleITK as sitk

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def apply_n3_correction(input_path: Path, output_path: Path) -> bool:
    """
    Apply N3 bias field correction using SimpleITK

    Args:
        input_path: Path to input NIfTI file
        output_path: Path to save corrected NIfTI file

    Returns:
        True if successful
    """
    try:
        # Read image with SimpleITK
        image = sitk.ReadImage(str(input_path))

        # Cast to float for N4 (improved version of N3)
        image = sitk.Cast(image, sitk.sitkFloat32)

        # Apply N4BiasFieldCorrection (improved version of N3)
        # N4 is the successor to N3 and produces better results
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrector.SetMaximumNumberOfIterations([50, 50, 50, 50])
        corrector.SetConvergenceThreshold(0.001)

        corrected = corrector.Execute(image)

        # Write corrected image
        sitk.WriteImage(corrected, str(output_path))

        return True

    except Exception as e:
        logger.error(f"Error applying N3 correction to {input_path.name}: {e}")
        return False


def load_scan_list(scan_list_file: Path) -> list:
    """Load list of scan paths from file"""
    scans = []
    with open(scan_list_file, 'r') as f:
        for line in f:
            scan_path = Path(line.strip())
            if scan_path.exists():
                scans.append(scan_path)
            else:
                logger.warning(f"Scan not found: {scan_path}")
    return scans


def find_all_scans(input_dir: Path) -> list:
    """Find all NIfTI scans in directory"""
    scans = []
    for patient_dir in input_dir.iterdir():
        if not patient_dir.is_dir() or patient_dir.name.startswith('.'):
            continue

        for scan in patient_dir.glob('*.nii.gz'):
            if not scan.name.startswith('.'):
                scans.append(scan)

    return scans


def main():
    parser = argparse.ArgumentParser(
        description='Apply N3 bias field correction to ADNI MPRAGE scans',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Correct all scans
  python preprocessing/apply_n3_correction.py \
    --input /Volumes/KINGSTON/ADNI_nifti \
    --output /Volumes/KINGSTON/ADNI_nifti_n3

  # Correct only required scans for experiment
  python preprocessing/apply_n3_correction.py \
    --input /Volumes/KINGSTON/ADNI_nifti \
    --output /Volumes/KINGSTON/ADNI_nifti_n3 \
    --scan-list experiments/cn_mci_ad_3dhcct/required_scans.txt

  # Resume interrupted processing
  python preprocessing/apply_n3_correction.py \
    --input /Volumes/KINGSTON/ADNI_nifti \
    --output /Volumes/KINGSTON/ADNI_nifti_n3 \
    --resume
        """
    )

    parser.add_argument('--input', type=str, required=True,
                       help='Input directory with raw MPRAGE scans')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory for N3-corrected scans')
    parser.add_argument('--scan-list', type=str, default=None,
                       help='Text file with scan paths to process (optional)')
    parser.add_argument('--resume', action='store_true',
                       help='Skip already processed scans')

    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        return

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get list of scans to process
    if args.scan_list:
        logger.info(f"Loading scan list from: {args.scan_list}")
        scans = load_scan_list(Path(args.scan_list))
    else:
        logger.info(f"Finding all scans in: {input_dir}")
        scans = find_all_scans(input_dir)

    logger.info(f"Found {len(scans)} scans to process")

    # Process each scan
    logger.info("="*80)
    logger.info("APPLYING N3 BIAS FIELD CORRECTION")
    logger.info("="*80)

    successful = 0
    failed = 0
    skipped = 0

    with tqdm(total=len(scans), desc="Processing") as pbar:
        for scan_path in scans:
            # Create output path maintaining patient directory structure
            patient_id = scan_path.parent.name
            patient_output_dir = output_dir / patient_id
            patient_output_dir.mkdir(parents=True, exist_ok=True)

            output_path = patient_output_dir / scan_path.name

            pbar.set_description(f"Processing {patient_id}")

            # Skip if already processed (resume mode)
            if args.resume and output_path.exists():
                skipped += 1
                pbar.update(1)
                continue

            # Apply N3 correction
            success = apply_n3_correction(scan_path, output_path)

            if success:
                successful += 1
            else:
                failed += 1

            pbar.update(1)

    # Summary
    logger.info("="*80)
    logger.info("N3 BIAS CORRECTION COMPLETE")
    logger.info("="*80)
    logger.info(f"Total scans: {len(scans)}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Skipped (already processed): {skipped}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("="*80)

    if failed > 0:
        logger.warning(f"{failed} scans failed. Check logs above for details.")

    logger.info(f"\nNext step: Run NPPY preprocessing on N3-corrected scans:")
    logger.info(f"  python preprocessing/pipeline_2_nppy/run_nppy_preprocessing.py \\")
    logger.info(f"    --input {output_dir} \\")
    logger.info(f"    --output /Volumes/KINGSTON/ADNI_nppy")


if __name__ == '__main__':
    main()

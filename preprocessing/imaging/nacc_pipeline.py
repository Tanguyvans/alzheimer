#!/usr/bin/env python3
"""
NACC MRI Preprocessing Pipeline

This script processes NACC NIfTI files through the standard preprocessing pipeline:
1. N4 bias correction + MNI registration
2. Skull stripping with SynthStrip (optional next step)

Usage:
    # Step 1: Register to MNI (N4 + registration)
    python nacc_pipeline.py --step register \
        --input /Volumes/KINGSTON/NACC-nifti \
        --output /Volumes/KINGSTON/NACC-registered \
        --template mni_template/mni_icbm152_nlin_sym_09a_nifti/mni_icbm152_nlin_sym_09a/mni_icbm152_t1_tal_nlin_sym_09a.nii

    # Step 2: Skull stripping
    python nacc_pipeline.py --step skull \
        --input /Volumes/KINGSTON/NACC-registered \
        --output /Volumes/KINGSTON/NACC-skull

    # Resume interrupted processing
    python nacc_pipeline.py --step register --resume \
        --input /Volumes/KINGSTON/NACC-nifti \
        --output /Volumes/KINGSTON/NACC-registered
"""

import os
import sys
import argparse
import signal
import json
import time
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# Global flag for graceful shutdown
should_stop = False


def signal_handler(signum, frame):
    """Handle Ctrl+C for graceful shutdown."""
    global should_stop
    print("\n\n[STOP] Stopping after current file completes...")
    print("Press Ctrl+C again to force quit")
    should_stop = True


def load_progress(progress_file):
    """Load processing progress from JSON file."""
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            return json.load(f)
    return {
        'processed': [],
        'failed': [],
        'last_updated': None,
        'total_files': 0
    }


def save_progress(progress_file, progress_data):
    """Save processing progress to JSON file."""
    progress_data['last_updated'] = datetime.now().isoformat()
    with open(progress_file, 'w') as f:
        json.dump(progress_data, f, indent=2)


def find_nacc_nifti_files(input_dir: Path):
    """
    Find all NIfTI files in NACC directory structure.

    NACC structure:
    /NACC-nifti/{subject_id}/{scan_file}.nii.gz

    Args:
        input_dir: Root NACC directory

    Returns:
        List of tuples (subject_id, nifti_path)
    """
    nifti_files = []

    for subject_dir in sorted(input_dir.iterdir()):
        if not subject_dir.is_dir() or subject_dir.name.startswith('.'):
            continue

        subject_id = subject_dir.name

        # Find .nii or .nii.gz files in subject directory
        for nifti_file in subject_dir.glob('*.nii*'):
            if nifti_file.name.startswith('.'):
                continue
            nifti_files.append((subject_id, nifti_file))

    return nifti_files


def find_registered_files(input_dir: Path):
    """Find all registered NIfTI files for skull stripping."""
    registered_files = []

    for subject_dir in input_dir.iterdir():
        if not subject_dir.is_dir() or subject_dir.name.startswith('.'):
            continue

        subject_id = subject_dir.name

        for nifti_file in subject_dir.glob('*_registered.nii.gz'):
            if nifti_file.name.startswith('.'):
                continue
            registered_files.append((subject_id, nifti_file))

    return sorted(registered_files, key=lambda x: x[0])


def process_nifti_file(input_path: str, output_path: str, template_path: str, subject_id: str) -> bool:
    """
    Apply N4 bias correction and register to MNI template.

    Args:
        input_path: Path to input NIfTI file
        output_path: Path for output registered NIfTI
        template_path: Path to MNI template
        subject_id: Subject identifier for logging

    Returns:
        True if successful
    """
    import ants
    import SimpleITK as sitk

    try:
        # Step 1: N4 Bias Field Correction using SimpleITK
        logger.info(f"      N4 bias correction...")
        image = sitk.ReadImage(input_path)
        image = sitk.Cast(image, sitk.sitkFloat32)

        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrector.SetMaximumNumberOfIterations([50, 50, 50, 50])
        corrector.SetConvergenceThreshold(0.001)
        corrected = corrector.Execute(image)

        # Save temporary N4-corrected file
        temp_n4_path = output_path.replace('.nii.gz', '_temp_n4.nii.gz')
        sitk.WriteImage(corrected, temp_n4_path)

        # Step 2: Register to MNI using ANTs
        logger.info(f"      MNI registration...")
        moving = ants.image_read(temp_n4_path)
        fixed = ants.image_read(template_path)

        # Perform registration (affine + SyN)
        registration = ants.registration(
            fixed=fixed,
            moving=moving,
            type_of_transform='SyN',
            verbose=False
        )

        # Save registered image
        ants.image_write(registration['warpedmovout'], output_path)

        # Clean up temp file
        if os.path.exists(temp_n4_path):
            os.remove(temp_n4_path)

        return True

    except Exception as e:
        logger.error(f"      Error processing {subject_id}: {str(e)}")
        # Clean up temp file on error
        temp_n4_path = output_path.replace('.nii.gz', '_temp_n4.nii.gz')
        if os.path.exists(temp_n4_path):
            os.remove(temp_n4_path)
        return False


def run_registration(args):
    """Step 1: N4 bias correction + MNI registration."""
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    template_path = args.template

    output_dir.mkdir(parents=True, exist_ok=True)

    # Progress tracking
    progress_file = output_dir / 'registration_progress.json'
    progress = load_progress(progress_file)

    # Find NACC NIfTI files
    logger.info(f"Searching for NIfTI files in {input_dir}...")
    all_files = find_nacc_nifti_files(input_dir)
    progress['total_files'] = len(all_files)

    # Count unique subjects
    unique_subjects = len(set(f[0] for f in all_files))
    logger.info(f"Found {len(all_files)} scans from {unique_subjects} subjects")

    # Filter already processed
    if args.resume:
        files_to_process = [(subj, path) for subj, path in all_files
                           if subj not in progress['processed']]
        logger.info(f"Resume: {len(progress['processed'])} processed, "
                   f"{len(files_to_process)} remaining")
    else:
        files_to_process = all_files
        progress['processed'] = []
        progress['failed'] = []

    if not files_to_process:
        logger.info("All files already processed!")
        return

    logger.info(f"\nStarting N4 + MNI registration...")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Template: {template_path}")
    logger.info("-" * 60)

    start_time = time.time()
    processed_count = len(progress['processed'])

    for i, (subject_id, nifti_path) in enumerate(files_to_process, 1):
        if should_stop:
            logger.info("Stopped by user. Progress saved.")
            break

        logger.info(f"\n[{processed_count + 1}/{progress['total_files']}] {subject_id}")
        logger.info(f"   File: {nifti_path.name}")

        try:
            # Create output directory per subject
            subject_output_dir = output_dir / subject_id
            subject_output_dir.mkdir(parents=True, exist_ok=True)

            # Output filename
            output_filename = nifti_path.stem
            if output_filename.endswith('.nii'):
                output_filename = output_filename[:-4]
            output_filename = f"{output_filename}_registered.nii.gz"
            output_file = subject_output_dir / output_filename

            if output_file.exists():
                logger.info(f"   [SKIP] Already processed")
                if subject_id not in progress['processed']:
                    progress['processed'].append(subject_id)
                processed_count += 1
            else:
                logger.info(f"   Processing...")
                file_start = time.time()

                success = process_nifti_file(
                    str(nifti_path),
                    str(output_file),
                    template_path,
                    subject_id
                )

                file_time = time.time() - file_start

                if success:
                    progress['processed'].append(subject_id)
                    processed_count += 1
                    logger.info(f"   [OK] Success ({file_time:.1f}s)")
                else:
                    progress['failed'].append(subject_id)
                    logger.info(f"   [FAIL] Failed")

            save_progress(progress_file, progress)

            # Progress stats
            elapsed = time.time() - start_time
            avg_time = elapsed / i if i > 0 else 0
            remaining = len(files_to_process) - i
            eta = remaining * avg_time / 60
            completion = (processed_count / progress['total_files']) * 100
            logger.info(f"   Progress: {completion:.1f}% | ETA: {eta:.1f} min")

        except Exception as e:
            logger.error(f"   [ERROR] {str(e)}")
            progress['failed'].append(subject_id)
            save_progress(progress_file, progress)

    # Summary
    total_time = time.time() - start_time
    logger.info("\n" + "=" * 60)
    logger.info("REGISTRATION COMPLETE")
    logger.info(f"Processed: {len(progress['processed'])}")
    logger.info(f"Failed: {len(progress['failed'])}")
    logger.info(f"Time: {total_time/60:.1f} minutes")
    logger.info("=" * 60)


def run_skull_stripping(args):
    """Step 2: Skull stripping with SynthStrip."""
    from preprocessing.imaging.skull_stripping import synthstrip_skull_strip, setup_synthstrip_docker

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Progress tracking
    progress_file = output_dir / 'skull_stripping_progress.json'
    progress = load_progress(progress_file)

    # Setup SynthStrip
    logger.info("Setting up SynthStrip Docker...")
    if not setup_synthstrip_docker():
        logger.error("Failed to setup SynthStrip Docker")
        return
    logger.info("SynthStrip ready")

    # Find registered files
    logger.info(f"Searching for registered files in {input_dir}...")
    all_files = find_registered_files(input_dir)
    progress['total_files'] = len(all_files)
    logger.info(f"Found {len(all_files)} registered files")

    # Filter already processed
    if args.resume:
        files_to_process = [(s, p) for s, p in all_files
                           if s not in progress['processed']]
        logger.info(f"Resume: {len(progress['processed'])} processed, "
                   f"{len(files_to_process)} remaining")
    else:
        files_to_process = all_files
        progress['processed'] = []
        progress['failed'] = []

    if not files_to_process:
        logger.info("All files already processed!")
        return

    logger.info(f"\nStarting skull stripping...")
    logger.info("-" * 60)

    start_time = time.time()
    processed_count = len(progress['processed'])

    for i, (subject_id, nifti_path) in enumerate(files_to_process, 1):
        if should_stop:
            logger.info("Stopped by user. Progress saved.")
            break

        logger.info(f"\n[{processed_count + 1}/{progress['total_files']}] {subject_id}")

        try:
            subject_output_dir = output_dir / subject_id
            subject_output_dir.mkdir(parents=True, exist_ok=True)

            output_filename = nifti_path.name.replace('_registered.nii.gz', '_skull_stripped.nii.gz')
            output_file = subject_output_dir / output_filename

            if output_file.exists():
                logger.info(f"   [SKIP] Already processed")
                if subject_id not in progress['processed']:
                    progress['processed'].append(subject_id)
                processed_count += 1
            else:
                logger.info(f"   Skull stripping...")
                file_start = time.time()

                success = synthstrip_skull_strip(str(nifti_path), str(output_file), subject_id)
                file_time = time.time() - file_start

                if success:
                    progress['processed'].append(subject_id)
                    processed_count += 1
                    logger.info(f"   [OK] Success ({file_time:.1f}s)")
                else:
                    progress['failed'].append(subject_id)
                    logger.info(f"   [FAIL] Failed")

            save_progress(progress_file, progress)

        except Exception as e:
            logger.error(f"   [ERROR] {str(e)}")
            progress['failed'].append(subject_id)
            save_progress(progress_file, progress)

    # Summary
    total_time = time.time() - start_time
    logger.info("\n" + "=" * 60)
    logger.info("SKULL STRIPPING COMPLETE")
    logger.info(f"Processed: {len(progress['processed'])}")
    logger.info(f"Failed: {len(progress['failed'])}")
    logger.info(f"Time: {total_time/60:.1f} minutes")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='NACC MRI Preprocessing Pipeline')
    parser.add_argument('--step', required=True, choices=['register', 'skull'],
                       help='Processing step: register or skull')
    parser.add_argument('--input', required=True, help='Input directory')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--template', default='mni_template/mni_icbm152_nlin_sym_09a_nifti/mni_icbm152_nlin_sym_09a/mni_icbm152_t1_tal_nlin_sym_09a.nii',
                       help='MNI template path (for register step)')
    parser.add_argument('--resume', action='store_true', help='Resume from previous progress')

    args = parser.parse_args()

    # Setup signal handler
    signal.signal(signal.SIGINT, signal_handler)

    # Verify input/output paths exist
    if not os.path.exists(args.input):
        logger.error(f"Input directory not found: {args.input}")
        return

    if args.step == 'register':
        run_registration(args)
    elif args.step == 'skull':
        run_skull_stripping(args)


if __name__ == '__main__':
    main()

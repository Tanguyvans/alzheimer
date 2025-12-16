#!/usr/bin/env python3
"""
OASIS-3 MRI Preprocessing Pipeline

This script processes OASIS-3 NIfTI files through the standard preprocessing pipeline:
1. Find T1-weighted NIfTI files (OASIS data is already in NIfTI format)
2. N4 bias correction + MNI registration
3. Skull stripping with SynthStrip
4. Convert to NPY for ML training

Usage:
    # Step 1: Register to MNI (N4 + registration)
    python oasis_pipeline.py --step register \
        --input /Volumes/KINGSTON/OASIS \
        --output /Volumes/KINGSTON/OASIS-registered \
        --template mni_template/mni_icbm152_nlin_sym_09a_nifti/mni_icbm152_nlin_sym_09a/mni_icbm152_t1_tal_nlin_sym_09a.nii

    # Step 2: Skull stripping
    python oasis_pipeline.py --step skull \
        --input /Volumes/KINGSTON/OASIS-registered \
        --output /Volumes/KINGSTON/OASIS-skull

    # Step 3: Convert to NPY
    python oasis_pipeline.py --step npy \
        --input /Volumes/KINGSTON/OASIS-skull \
        --output /Volumes/KINGSTON/OASIS-npy \
        --labels data/oasis/oasis_all.csv
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


def find_oasis_nifti_files(input_dir: Path, sequence_preference: list = None):
    """
    Find ALL NIfTI files in OASIS directory structure.

    OASIS structure:
    /OASIS/{subject}/{sequence}/{date}/{scan_id}/{nifti_file}.nii

    Collects ALL scans from ALL sequences and ALL dates.
    Filtering/sorting can be done later.

    Args:
        input_dir: Root OASIS directory
        sequence_preference: Not used (kept for compatibility)

    Returns:
        List of tuples (subject_id, session_id, nifti_path, sequence_type)
        where session_id = {subject}_{seq}_{date}_{scan_id} for unique identification
    """
    nifti_files = []
    subject_dirs = sorted([d for d in input_dir.iterdir()
                          if d.is_dir() and not d.name.startswith('.')])

    for subject_dir in subject_dirs:
        subject_id = subject_dir.name

        # Iterate through ALL sequence folders
        for seq_dir in subject_dir.iterdir():
            if not seq_dir.is_dir() or seq_dir.name.startswith('.'):
                continue

            sequence_name = seq_dir.name

            # Iterate through ALL date folders
            for date_dir in seq_dir.iterdir():
                if not date_dir.is_dir() or date_dir.name.startswith('.'):
                    continue

                date_id = date_dir.name

                # Iterate through ALL scan folders
                for scan_dir in date_dir.iterdir():
                    if not scan_dir.is_dir() or scan_dir.name.startswith('.'):
                        continue

                    scan_id = scan_dir.name

                    # Find .nii or .nii.gz files
                    for nifti_file in scan_dir.glob('*.nii*'):
                        if nifti_file.name.startswith('.'):
                            continue

                        # Create unique session ID: subject_seq_date_scanid
                        session_id = f"{subject_id}_{sequence_name}_{date_id}_{scan_id}"
                        nifti_files.append((subject_id, session_id, nifti_file, sequence_name))
                        break  # Take first NIfTI in this scan folder

    return nifti_files


def clean_oasis_filename(filename: str) -> str:
    """
    Clean OASIS filename by removing anonymized date/time.

    Input:  OAS30037_MPRAGE_GRAPPA2_1970-01-01_00_00_00.0_I11249201_registered.nii.gz
    Output: OAS30037_MPRAGE_GRAPPA2_I11249201_registered.nii.gz

    Removes patterns like: _1970-01-01_00_00_00.0 or _YYYY-MM-DD_HH_MM_SS.0
    """
    import re
    # Pattern: _YYYY-MM-DD_HH_MM_SS.0 (date_time with underscores)
    pattern = r'_\d{4}-\d{2}-\d{2}_\d{2}_\d{2}_\d{2}\.\d'
    cleaned = re.sub(pattern, '', filename)
    return cleaned


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


def find_skull_stripped_files(input_dir: Path):
    """Find all skull-stripped NIfTI files for NPY conversion."""
    skull_files = []

    for subject_dir in input_dir.iterdir():
        if not subject_dir.is_dir() or subject_dir.name.startswith('.'):
            continue

        subject_id = subject_dir.name

        for nifti_file in subject_dir.glob('*_skull_stripped.nii.gz'):
            if nifti_file.name.startswith('.'):
                continue
            skull_files.append((subject_id, nifti_file))

    return sorted(skull_files, key=lambda x: x[0])


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
        import os
        if os.path.exists(temp_n4_path):
            os.remove(temp_n4_path)

        return True

    except Exception as e:
        logger.error(f"      Error processing {subject_id}: {str(e)}")
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

    # Find OASIS NIfTI files (ALL images, not just one per subject)
    logger.info(f"Searching for NIfTI files in {input_dir}...")
    all_files = find_oasis_nifti_files(input_dir)
    progress['total_files'] = len(all_files)

    # Count unique subjects
    unique_subjects = len(set(f[0] for f in all_files))
    logger.info(f"Found {len(all_files)} scans from {unique_subjects} subjects")

    # Filter already processed (using session_id for uniqueness)
    if args.resume:
        files_to_process = [(subj, sess, path, seq) for subj, sess, path, seq in all_files
                           if sess not in progress['processed']]
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

    for i, (subject_id, session_id, nifti_path, sequence) in enumerate(files_to_process, 1):
        if should_stop:
            logger.info("Stopped by user. Progress saved.")
            break

        logger.info(f"\n[{processed_count + 1}/{progress['total_files']}] {session_id}")
        logger.info(f"   Subject: {subject_id} | Sequence: {sequence}")
        logger.info(f"   File: {nifti_path.name}")

        try:
            # Create output directory per subject
            subject_output_dir = output_dir / subject_id
            subject_output_dir.mkdir(parents=True, exist_ok=True)

            # Output filename includes session_id for uniqueness
            output_filename = f"{session_id}_registered.nii.gz"
            output_file = subject_output_dir / output_filename

            if output_file.exists():
                logger.info(f"   [SKIP] Already processed")
                if session_id not in progress['processed']:
                    progress['processed'].append(session_id)
                processed_count += 1
            else:
                logger.info(f"   Processing...")
                file_start = time.time()

                success = process_nifti_file(
                    str(nifti_path),
                    str(output_file),
                    template_path,
                    session_id
                )

                file_time = time.time() - file_start

                if success:
                    progress['processed'].append(session_id)
                    processed_count += 1
                    logger.info(f"   [OK] Success ({file_time:.1f}s)")
                else:
                    progress['failed'].append(session_id)
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

            # Clean filename: remove anonymized date/time part
            # Input:  OAS30037_MPRAGE_GRAPPA2_1970-01-01_00_00_00.0_I11249201_registered.nii.gz
            # Output: OAS30037_MPRAGE_GRAPPA2_I11249201_skull_stripped.nii.gz
            clean_name = clean_oasis_filename(nifti_path.name)
            output_filename = clean_name.replace('_registered.nii.gz', '_skull_stripped.nii.gz')
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


def run_npy_conversion(args):
    """Step 3: Convert to NPY format for ML training."""
    import numpy as np
    import nibabel as nib
    import pandas as pd
    from scipy.ndimage import zoom

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    labels_file = Path(args.labels) if args.labels else None

    target_shape = tuple(map(int, args.target_shape.split(',')))

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load labels if provided
    labels_df = None
    if labels_file and labels_file.exists():
        labels_df = pd.read_csv(labels_file)
        # Get first visit diagnosis per subject
        subject_labels = labels_df.groupby('Subject')['DX'].first().to_dict()
        logger.info(f"Loaded labels for {len(subject_labels)} subjects")

    # Find skull-stripped files
    logger.info(f"Searching for skull-stripped files in {input_dir}...")
    all_files = find_skull_stripped_files(input_dir)
    logger.info(f"Found {len(all_files)} files")

    # Progress tracking
    progress_file = output_dir / 'npy_conversion_progress.json'
    progress = load_progress(progress_file)

    if args.resume:
        files_to_process = [(s, p) for s, p in all_files
                           if s not in progress['processed']]
    else:
        files_to_process = all_files
        progress['processed'] = []
        progress['failed'] = []

    progress['total_files'] = len(all_files)

    logger.info(f"\nConverting to NPY (target shape: {target_shape})...")
    logger.info("-" * 60)

    start_time = time.time()

    for i, (subject_id, nifti_path) in enumerate(files_to_process, 1):
        if should_stop:
            logger.info("Stopped by user.")
            break

        logger.info(f"[{i}/{len(files_to_process)}] {subject_id}")

        try:
            # Get label if available
            label = None
            if labels_df is not None:
                label = subject_labels.get(subject_id)

            # Load NIfTI
            img = nib.load(str(nifti_path))
            data = img.get_fdata()

            # Resize to target shape
            zoom_factors = [t / s for t, s in zip(target_shape, data.shape)]
            data_resized = zoom(data, zoom_factors, order=1)

            # Normalize
            data_norm = (data_resized - data_resized.min()) / (data_resized.max() - data_resized.min() + 1e-8)

            # Create output directory by label if available
            if label:
                label_dir = output_dir / label
                label_dir.mkdir(parents=True, exist_ok=True)
                output_file = label_dir / f"{subject_id}.npy"
            else:
                output_file = output_dir / f"{subject_id}.npy"

            # Save NPY
            np.save(str(output_file), data_norm.astype(np.float32))

            progress['processed'].append(subject_id)
            logger.info(f"   [OK] {output_file.name} (label: {label})")

        except Exception as e:
            logger.error(f"   [ERROR] {str(e)}")
            progress['failed'].append(subject_id)

        save_progress(progress_file, progress)

    # Summary
    total_time = time.time() - start_time
    logger.info("\n" + "=" * 60)
    logger.info("NPY CONVERSION COMPLETE")
    logger.info(f"Processed: {len(progress['processed'])}")
    logger.info(f"Failed: {len(progress['failed'])}")
    logger.info(f"Time: {total_time/60:.1f} minutes")
    logger.info(f"Target shape: {target_shape}")
    logger.info("=" * 60)

    # Count by label
    if labels_df is not None:
        for label_dir in output_dir.iterdir():
            if label_dir.is_dir() and not label_dir.name.startswith('.'):
                count = len(list(label_dir.glob('*.npy')))
                logger.info(f"  {label_dir.name}: {count} files")


def main():
    parser = argparse.ArgumentParser(description='OASIS-3 MRI Preprocessing Pipeline')
    parser.add_argument('--step', required=True, choices=['register', 'skull', 'npy'],
                       help='Processing step: register, skull, or npy')
    parser.add_argument('--input', required=True, help='Input directory')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--template', default='mni_template/mni_icbm152_nlin_sym_09a_nifti/mni_icbm152_nlin_sym_09a/mni_icbm152_t1_tal_nlin_sym_09a.nii',
                       help='MNI template path (for register step)')
    parser.add_argument('--labels', default=None, help='Labels CSV file (for npy step)')
    parser.add_argument('--target-shape', default='192,192,192',
                       help='Target shape for NPY conversion (default: 192,192,192)')
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
    elif args.step == 'npy':
        run_npy_conversion(args)


if __name__ == '__main__':
    main()
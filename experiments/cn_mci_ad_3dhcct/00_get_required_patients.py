#!/usr/bin/env python3
"""
Identify which patients are needed for cn_mci_ad_3dhcct training.
This allows us to preprocess only the required scans with NPPY.

Usage:
    python 00_get_required_patients.py --config config.yaml

Output:
    - required_patients.txt: List of PTID (patient IDs) needed for training
    - Prints summary of required patients
"""

import sys
import argparse
import logging
from pathlib import Path
import yaml
import pandas as pd
import nibabel as nib
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def check_scan_dimensions(scan_path: Path, min_dim: int = 100) -> tuple:
    """
    Check if scan meets dimension requirements.

    IMPORTANT: ALL three dimensions must be >= min_dim (not just the minimum).
    This filters out localizer scans like (240, 256, 3) where only one dimension is too small.

    Returns:
        (is_valid, shape, reason)
        - is_valid: True if scan has 3D shape with ALL dimensions >= min_dim
        - shape: tuple of scan dimensions
        - reason: description of why scan is invalid (or "ok")
    """
    try:
        img = nib.load(str(scan_path))
        data = img.get_fdata()
        data = np.squeeze(data)  # Remove singleton dimensions
        shape = data.shape

        # Check if 3D
        if len(shape) != 3:
            return (False, shape, f"not_3d_{len(shape)}d")

        # Check EACH dimension individually
        for i, dim_size in enumerate(shape):
            if dim_size < min_dim:
                return (False, shape, f"dim[{i}]={dim_size}<{min_dim}")

        return (True, shape, "ok")
    except Exception as e:
        return (False, None, f"load_error: {e}")


def check_scan_file_size(scan_path: Path, max_size_mb: float = 20.0) -> tuple:
    """
    Check if scan file size is within acceptable range.

    Large files (>20MB) are often "Accelerated_Sagittal_MPRAGE__MSV22" protocol
    from 2024-2025 that don't work well with NPPY preprocessing.

    Returns:
        (is_valid, size_mb, reason)
        - is_valid: True if file size is <= max_size_mb
        - size_mb: file size in MB
        - reason: description of why scan is invalid (or "ok")
    """
    try:
        size_mb = scan_path.stat().st_size / (1024**2)

        if size_mb > max_size_mb:
            return (False, size_mb, f"file_size={size_mb:.1f}MB>{max_size_mb}MB")

        return (True, size_mb, "ok")
    except Exception as e:
        return (False, 0.0, f"size_check_error: {e}")


def check_exact_dimensions(scan_path: Path, target_dims_list: list = None) -> tuple:
    """
    Check if scan has exact target dimensions for high-quality NPPY output.

    Accepts multiple dimension sets (e.g., [(192, 192, 160), (240, 256, 160), ...]).
    Each dimension set can be in any orientation.

    Good quality dimensions (visually verified):
    - (160, 192, 192) ✓ GOOD (original, 7.83% variance retention)
    - (160, 240, 256) ✓ GOOD
    - (180, 256, 256) ✓ GOOD
    - (176, 240, 256) ✓ GOOD
    - (208, 240, 256) ✓ GOOD
    - (128, 240, 256) ✓ GOOD
    - (120, 240, 256) ✓ GOOD
    - (230, 230, 240) ✓ GOOD (WARNING: head may be rotated!)

    Poor quality dimensions:
    - (256, 256, 166) ✗ BLURRY
    - (256, 256, 170) ✗ BLURRY
    - (256, 256, 208) ✗ BLURRY
    - (256, 256, 211) ✗ BLURRY (but some exceptions exist)

    Returns:
        (is_valid, shape, reason)
        - is_valid: True if scan dimensions match any target in list (any orientation)
        - shape: tuple of scan dimensions
        - reason: description of why scan is invalid (or "ok")
    """
    if target_dims_list is None:
        target_dims_list = [(192, 192, 160)]

    try:
        img = nib.load(str(scan_path))
        data = img.get_fdata()
        data = np.squeeze(data)  # Remove singleton dimensions
        shape = tuple(data.shape)

        # Check if 3D
        if len(shape) != 3:
            return (False, shape, f"not_3d_{len(shape)}d")

        # Check if sorted dimensions match any target (allows any orientation)
        sorted_shape = tuple(sorted(shape))

        for target_dims in target_dims_list:
            sorted_target = tuple(sorted(target_dims))
            if sorted_shape == sorted_target:
                return (True, shape, "ok")

        # No match found
        return (False, shape, f"dim={shape}!=any_target_dims")
    except Exception as e:
        return (False, None, f"load_error: {e}")


def identify_required_patients(dxsum_csv: str, nifti_dir: str, output_file: str = "required_patients.txt",
                               output_scans_file: str = "required_scans.txt", blacklist_file: str = None,
                               min_dim: int = 100, max_size_mb: float = 20.0,
                               exact_dims_list: list = None):
    """
    Identify stable CN, MCI, and AD patients from ADNI diagnosis data.
    Maps each patient to their BASELINE scan (first visit only).
    Filters out scans with bad dimensions, large file sizes (>20MB), and blacklisted scans.

    If exact_dims_list is specified, only scans matching one of the dimension sets are selected.
    This ensures high-quality NPPY output (e.g., exact_dims_list=[(192, 192, 160), (240, 256, 160)]).

    Returns list of PTID (patient IDs) and exact scan paths needed for the experiment.
    """
    logger.info("="*80)
    logger.info("IDENTIFYING REQUIRED PATIENTS FOR CN_MCI_AD_3DHCCT")
    logger.info("="*80)
    if exact_dims_list:
        logger.info(f"Target dimensions (exact match, {len(exact_dims_list)} sets):")
        for dims in exact_dims_list:
            logger.info(f"  - {dims}")
    else:
        logger.info(f"Minimum dimension threshold: {min_dim}")
    logger.info(f"Maximum file size: {max_size_mb} MB")

    # Load blacklist if provided (optional, dimension checking is primary filter)
    blacklist = set()
    if blacklist_file and Path(blacklist_file).exists():
        logger.info(f"\nLoading blacklist from: {blacklist_file}")
        with open(blacklist_file, 'r') as f:
            blacklist = {line.strip() for line in f if line.strip()}
        logger.info(f"Loaded {len(blacklist)} blacklisted scans")
    else:
        logger.info(f"\nNo blacklist provided - using dimension checking only")

    # Load diagnosis data
    dx_df = pd.read_csv(dxsum_csv)
    logger.info(f"Loaded {len(dx_df):,} diagnosis records for {dx_df['RID'].nunique():,} patients")

    # Group by patient
    patient_diagnoses = dx_df.groupby('RID')['DIAGNOSIS'].apply(lambda x: x.unique().tolist()).reset_index()
    patient_diagnoses.columns = ['RID', 'all_diagnoses']

    # Stable CN: all diagnoses are 1
    stable_cn = patient_diagnoses[patient_diagnoses['all_diagnoses'].apply(lambda x: x == [1])]
    logger.info(f"Stable CN patients: {len(stable_cn):,}")

    # Stable MCI: all diagnoses are 2
    stable_mci = patient_diagnoses[patient_diagnoses['all_diagnoses'].apply(lambda x: x == [2])]
    logger.info(f"Stable MCI patients: {len(stable_mci):,}")

    # Stable AD: all diagnoses are 3
    stable_ad = patient_diagnoses[patient_diagnoses['all_diagnoses'].apply(lambda x: x == [3])]
    logger.info(f"Stable AD patients: {len(stable_ad):,}")

    # Get all stable RIDs
    all_stable_rids = set(stable_cn['RID']) | set(stable_mci['RID']) | set(stable_ad['RID'])

    # Get stable patients with their baseline visit
    stable_patients_df = dx_df[dx_df['RID'].isin(all_stable_rids)].copy()

    # Get baseline visit for each patient (earliest EXAMDATE)
    baseline_visits = stable_patients_df.sort_values('EXAMDATE').groupby('RID').first().reset_index()

    logger.info(f"\nTotal required patients: {len(baseline_visits)}")
    logger.info(f"  CN: {len(stable_cn)}")
    logger.info(f"  MCI: {len(stable_mci)}")
    logger.info(f"  AD: {len(stable_ad)}")

    # Map to actual scan files
    logger.info(f"\n{'='*80}")
    logger.info("MAPPING TO BASELINE SCANS IN NIFTI DIRECTORY")
    logger.info("="*80)

    nifti_path = Path(nifti_dir)
    required_ptids = []
    required_scans = []
    missing_scans = []
    rejected_scans = []
    dimension_failures = []

    for _, row in baseline_visits.iterrows():
        ptid = row['PTID']
        patient_folder = nifti_path / ptid

        if not patient_folder.exists():
            missing_scans.append(ptid)
            continue

        # Find all NIfTI files for this patient
        scans = list(patient_folder.glob('*.nii.gz'))
        scans = [s for s in scans if not s.name.startswith('.') and '.4d_backup' not in s.name]  # Skip hidden files and backups

        if not scans:
            missing_scans.append(ptid)
            continue

        # Try scans in order until we find one that passes all checks
        # Sort scans by file size (prefer smaller files first - more likely to work with NPPY)
        scans_with_size = []
        for scan_path in scans:
            try:
                size_mb = scan_path.stat().st_size / (1024**2)
                scans_with_size.append((scan_path, size_mb))
            except:
                scans_with_size.append((scan_path, 999))  # Put failed size checks at end

        scans_sorted = sorted(scans_with_size, key=lambda x: x[1])

        scan_found = False
        failed_reasons = []

        for scan_path, size_mb in scans_sorted:
            # Check if blacklisted (supports both patient ID and full scan path)
            if ptid in blacklist or str(scan_path) in blacklist:
                failed_reasons.append(f"{scan_path.name}: blacklisted")
                logger.debug(f"Skipping blacklisted scan: {ptid} - {scan_path.name}")
                continue

            # Check file size
            is_size_valid, actual_size_mb, size_reason = check_scan_file_size(scan_path, max_size_mb)
            if not is_size_valid:
                failed_reasons.append(f"{scan_path.name}: {size_reason}")
                logger.debug(f"Skipping large scan: {ptid} - {scan_path.name} - {size_reason}")
                dimension_failures.append((ptid, scan_path.name, f"{actual_size_mb:.1f}MB", size_reason))
                continue

            # Check dimensions
            if exact_dims_list:
                # Use exact dimension matching for NPPY quality
                is_valid, shape, reason = check_exact_dimensions(scan_path, exact_dims_list)
            else:
                # Use minimum dimension threshold
                is_valid, shape, reason = check_scan_dimensions(scan_path, min_dim)

            if not is_valid:
                failed_reasons.append(f"{scan_path.name}: {reason} (shape={shape})")
                logger.debug(f"Skipping bad dimension scan: {ptid} - {scan_path.name} - {reason}")
                dimension_failures.append((ptid, scan_path.name, shape, reason))
                continue

            # Found a good scan!
            required_ptids.append(ptid)
            required_scans.append(str(scan_path))
            scan_found = True
            logger.debug(f"Selected scan: {ptid} - {scan_path.name} - {actual_size_mb:.1f}MB - shape={shape}")
            break

        if not scan_found:
            # All scans for this patient failed checks
            rejected_scans.append((ptid, f"All {len(scans)} scans rejected: {'; '.join(failed_reasons[:3])}"))
            logger.warning(f"All scans rejected for {ptid} ({len(scans)} scans checked)")

    logger.info(f"\nFound scans: {len(required_scans)}")
    logger.info(f"Missing scans: {len(missing_scans)}")
    logger.info(f"Rejected scans (bad dimensions, large file size, or blacklisted): {len(rejected_scans)}")
    logger.info(f"Total failures (dim/size issues): {len(dimension_failures)}")

    # Save patient IDs to file
    with open(output_file, 'w') as f:
        for ptid in required_ptids:
            f.write(f"{ptid}\n")

    # Save exact scan paths to file
    with open(output_scans_file, 'w') as f:
        for scan_path in required_scans:
            f.write(f"{scan_path}\n")

    logger.info(f"\n✓ Saved {len(required_ptids)} patient IDs to: {output_file}")
    logger.info(f"✓ Saved {len(required_scans)} scan paths to: {output_scans_file}")
    logger.info(f"\nNext step: Run NPPY preprocessing on these {len(required_scans)} scans")
    logger.info(f"  Use: preprocessing/pipeline_2_nppy/run_nppy_preprocessing.py --scan-list {output_scans_file}")

    if missing_scans:
        logger.warning(f"\n⚠️  {len(missing_scans)} patients have no NIfTI scans in {nifti_dir}")

    if dimension_failures:
        logger.info(f"\n📊 Dimension failures by reason:")
        from collections import Counter
        reasons = Counter([reason for _, _, _, reason in dimension_failures])
        for reason, count in reasons.most_common():
            logger.info(f"  {reason}: {count} scans")

    return required_ptids, required_scans


def main():
    parser = argparse.ArgumentParser(
        description='Identify required patients and baseline scans for cn_mci_ad_3dhcct experiment',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to YAML config file')
    parser.add_argument('--nifti-dir', type=str, default=None,
                       help='NIfTI directory (default: from config)')
    parser.add_argument('--output-patients', type=str, default='required_patients.txt',
                       help='Output file for patient IDs')
    parser.add_argument('--output-scans', type=str, default='required_scans.txt',
                       help='Output file for scan paths')
    parser.add_argument('--blacklist', type=str, default=None,
                       help='Optional blacklist file to filter out bad quality scans')
    parser.add_argument('--min-dim', type=int, default=100,
                       help='Minimum dimension threshold for scans (default: 100)')
    parser.add_argument('--max-size', type=float, default=20.0,
                       help='Maximum file size in MB (default: 20.0, filters out high-res scans)')
    parser.add_argument('--exact-dims', type=str,
                       default='192,192,160;240,256,160;256,256,180;240,256,176;240,256,208;240,256,128;240,256,120;230,230,240',
                       help='Exact dimensions for high-quality NPPY (visually verified). Multiple sets separated by semicolon. Set to "none" to disable.')

    args = parser.parse_args()

    # Parse exact_dims argument (now supports multiple dimension sets)
    exact_dims_list = None
    if args.exact_dims and args.exact_dims.lower() != 'none':
        try:
            exact_dims_list = []
            for dims_str in args.exact_dims.split(';'):
                dims = tuple(map(int, dims_str.split(',')))
                if len(dims) != 3:
                    logger.error(f"Invalid dimension format: {dims_str}. Each dimension set must have 3 values.")
                    sys.exit(1)
                exact_dims_list.append(dims)
        except ValueError as e:
            logger.error(f"Invalid --exact-dims format: {args.exact_dims}. Expected format: 192,192,160;240,256,160")
            logger.error(f"Error: {e}")
            sys.exit(1)

    # Load config
    config = load_config(args.config)
    dxsum_csv = config['data']['dxsum_csv']

    # Get NIfTI directory - either from arg or derive from skull_dir in config
    if args.nifti_dir:
        nifti_dir = args.nifti_dir
    else:
        # Derive from skull_dir: /Volumes/KINGSTON/ADNI-skull -> /Volumes/KINGSTON/ADNI_nifti
        skull_dir = config['data']['skull_dir']
        nifti_dir = str(Path(skull_dir).parent / 'ADNI_nifti')

    logger.info(f"dxsum.csv: {dxsum_csv}")
    logger.info(f"NIfTI directory: {nifti_dir}")
    if args.blacklist:
        logger.info(f"Blacklist: {args.blacklist}")
    else:
        logger.info(f"Blacklist: None (dimension and size checking only)")
    if exact_dims_list:
        logger.info(f"Exact dimensions: {len(exact_dims_list)} dimension sets")
        for dims in exact_dims_list:
            logger.info(f"  - {dims}")
    else:
        logger.info(f"Min dimension: {args.min_dim}")
    logger.info(f"Max file size: {args.max_size} MB\n")

    # Identify required patients and scans
    identify_required_patients(dxsum_csv, nifti_dir, args.output_patients, args.output_scans,
                              args.blacklist, args.min_dim, args.max_size, exact_dims_list)


if __name__ == '__main__':
    main()

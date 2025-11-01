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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def identify_required_patients(dxsum_csv: str, nifti_dir: str, output_file: str = "required_patients.txt",
                               output_scans_file: str = "required_scans.txt"):
    """
    Identify stable CN, MCI, and AD patients from ADNI diagnosis data.
    Maps each patient to their BASELINE scan (first visit only).

    Returns list of PTID (patient IDs) and exact scan paths needed for the experiment.
    """
    logger.info("="*80)
    logger.info("IDENTIFYING REQUIRED PATIENTS FOR CN_MCI_AD_3DHCCT")
    logger.info("="*80)

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

    for _, row in baseline_visits.iterrows():
        ptid = row['PTID']
        patient_folder = nifti_path / ptid

        if not patient_folder.exists():
            missing_scans.append(ptid)
            continue

        # Find all NIfTI files for this patient
        scans = list(patient_folder.glob('*.nii.gz'))
        scans = [s for s in scans if not s.name.startswith('.')]  # Skip hidden files

        if not scans:
            missing_scans.append(ptid)
            continue

        # Use the first scan (baseline - they should all be from same visit after DICOM conversion)
        scan_path = scans[0]
        required_ptids.append(ptid)
        required_scans.append(str(scan_path))

    logger.info(f"Found scans: {len(required_scans)}")
    logger.info(f"Missing scans: {len(missing_scans)}")

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

    args = parser.parse_args()

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
    logger.info(f"NIfTI directory: {nifti_dir}\n")

    # Identify required patients and scans
    identify_required_patients(dxsum_csv, nifti_dir, args.output_patients, args.output_scans)


if __name__ == '__main__':
    main()

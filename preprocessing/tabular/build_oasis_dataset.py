#!/usr/bin/env python3
"""
Build OASIS3 master dataset from UDS CSV files.

Merges multiple UDS forms and creates diagnosis labels compatible with ADNI.

IMPORTANT: Only labels as 'AD' if PROBAD=1 or POSSAD=1 (true Alzheimer's).
Other dementias (DLB, vascular, FTD, etc.) are labeled as 'Other_Dementia'.

Usage:
    python preprocessing/tabular/build_oasis_dataset.py
    python preprocessing/tabular/build_oasis_dataset.py --mri-dir /path/to/OASIS-nifti
"""

import argparse
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent.parent
OASIS_DIR = BASE_DIR / 'data' / 'oasis'


def load_uds_file(filename: str) -> pd.DataFrame:
    """Load a UDS CSV file"""
    filepath = OASIS_DIR / filename
    if filepath.exists():
        df = pd.read_csv(filepath)
        logger.info(f"Loaded {filename}: {len(df)} rows, {len(df.columns)} cols")
        return df
    else:
        logger.warning(f"File not found: {filename}")
        return pd.DataFrame()


def determine_diagnosis(row) -> str:
    """
    Determine diagnosis label from UDS diagnosis fields.

    IMPORTANT: Only returns 'AD' for Probable or Possible Alzheimer's Disease.
    Other dementias are labeled as 'Other_Dementia'.

    Returns: 'CN', 'MCI', 'AD', 'Other_Dementia', or 'Other'
    """
    # Check for dementia first (most severe)
    if row.get('DEMENTED') == 1:
        # Only label as AD if it's specifically Alzheimer's
        if row.get('PROBAD') == 1 or row.get('POSSAD') == 1:
            return 'AD'
        else:
            # Other dementia types (DLB, vascular, FTD, etc.)
            return 'Other_Dementia'

    # Check for MCI (any MCI subtype)
    mci_fields = ['MCIAMEM', 'MCIAPLUS', 'MCINON1', 'MCINON2']
    for field in mci_fields:
        if row.get(field) == 1:
            return 'MCI'

    # Check for normal cognition
    if row.get('NORMCOG') == 1:
        return 'CN'

    return 'Other'


def get_subjects_with_mri(mri_dir: Path) -> set:
    """Get list of subjects that have MRI data available"""
    if not mri_dir.exists():
        logger.warning(f"MRI directory not found: {mri_dir}")
        return set()

    subjects = set()
    for folder in mri_dir.iterdir():
        if folder.is_dir() and folder.name.startswith('OAS'):
            subjects.add(folder.name)

    logger.info(f"Found {len(subjects)} subjects with MRI data")
    return subjects


def build_oasis_dataset(mri_dir: Path = None):
    """Build master OASIS dataset from UDS files

    Args:
        mri_dir: Optional path to OASIS MRI directory. If provided, also creates
                 a filtered dataset with only subjects that have MRI data.

    Outputs:
        - oasis_tabular.csv: All clinical visits (for tabular analysis)
        - oasis_mri.csv: Only subjects with MRI (for imaging analysis) - if mri_dir provided
        - oasis_all_full.csv: Full dataset including Other_Dementia and Other
    """

    logger.info("=" * 60)
    logger.info("BUILDING OASIS3 MASTER DATASET")
    logger.info("=" * 60)

    # Load all UDS files
    diagnoses = load_uds_file('OASIS3_UDSd1_diagnoses.csv')
    cognitive = load_uds_file('OASIS3_UDSc1_cognitive_assessments.csv')
    demo_participant = load_uds_file('OASIS3_UDSa1_participant_demo.csv')
    demo_cs = load_uds_file('OASIS3_UDSa2_cs_demo.csv')
    health = load_uds_file('OASIS3_UDSa5_health_history.csv')
    physical = load_uds_file('OASIS3_UDSb1_physical_eval.csv')
    cdr = load_uds_file('OASIS3_UDSb4_cdr.csv')
    gds = load_uds_file('OASIS3_UDSb6_gds.csv')

    # Start with diagnoses as base (has the labels)
    if diagnoses.empty:
        logger.error("No diagnosis data found!")
        return None

    # Create visit key for merging
    merge_keys = ['OASISID', 'days_to_visit']

    # Start building master dataframe - include PROBAD and POSSAD for AD determination
    dx_cols = ['OASISID', 'OASIS_session_label', 'days_to_visit', 'age at visit',
               'NORMCOG', 'DEMENTED', 'MCIAMEM', 'MCIAPLUS', 'MCINON1', 'MCINON2',
               'PROBAD', 'POSSAD']
    # Only include columns that exist
    dx_cols = [c for c in dx_cols if c in diagnoses.columns]
    master = diagnoses[dx_cols].copy()

    # Determine diagnosis
    master['DX'] = master.apply(determine_diagnosis, axis=1)

    # Rename for compatibility
    master = master.rename(columns={
        'OASISID': 'Subject',
        'age at visit': 'AGE',
    })

    logger.info(f"\nDiagnosis distribution:")
    logger.info(master['DX'].value_counts().to_string())

    # Merge demographics
    if not demo_cs.empty:
        demo_subset = demo_cs[['OASISID', 'days_to_visit', 'INSEX', 'INEDUC', 'INRACE']].copy()
        demo_subset = demo_subset.rename(columns={
            'OASISID': 'Subject',
            'INSEX': 'PTGENDER',      # 1=Male, 2=Female in OASIS
            'INEDUC': 'PTEDUCAT',     # Years of education
            'INRACE': 'PTRACCAT',     # Race
        })
        master = master.merge(demo_subset, on=['Subject', 'days_to_visit'], how='left')

    if not demo_participant.empty:
        demo_p_subset = demo_participant[['OASISID', 'days_to_visit', 'MARISTAT']].copy()
        demo_p_subset = demo_p_subset.rename(columns={
            'OASISID': 'Subject',
            'MARISTAT': 'PTMARRY',
        })
        master = master.merge(demo_p_subset, on=['Subject', 'days_to_visit'], how='left')

    # Merge cognitive assessments
    if not cognitive.empty:
        cog_cols = ['OASISID', 'days_to_visit', 'ANIMALS', 'tma', 'tmb', 'digfor', 'digback', 'bnt']
        cog_cols = [c for c in cog_cols if c in cognitive.columns]
        cog_subset = cognitive[cog_cols].copy()
        cog_subset = cog_subset.rename(columns={
            'OASISID': 'Subject',
            'ANIMALS': 'CATANIMSC',   # Category fluency (animals)
            'tma': 'TRAASCOR',        # Trail Making A
            'tmb': 'TRABSCOR',        # Trail Making B
            'digfor': 'DSPANFOR',     # Digit span forward
            'digback': 'DSPANBAC',    # Digit span backward
            'bnt': 'BNTTOTAL',        # Boston Naming Test
        })
        master = master.merge(cog_subset, on=['Subject', 'days_to_visit'], how='left')

    # Merge physical measurements
    if not physical.empty:
        phys_subset = physical[['OASISID', 'days_to_visit', 'WEIGHT', 'HEIGHT']].copy()
        phys_subset = phys_subset.rename(columns={
            'OASISID': 'Subject',
            'WEIGHT': 'VSWEIGHT',
            'HEIGHT': 'VSHEIGHT',
        })
        # Handle missing values (999 = not measured)
        phys_subset['VSWEIGHT'] = phys_subset['VSWEIGHT'].replace(999, pd.NA)
        phys_subset['VSHEIGHT'] = phys_subset['VSHEIGHT'].replace(999, pd.NA)
        master = master.merge(phys_subset, on=['Subject', 'days_to_visit'], how='left')

    # Calculate BMI
    if 'VSWEIGHT' in master.columns and 'VSHEIGHT' in master.columns:
        # OASIS: Weight in lbs, Height in inches
        # BMI = (weight_lbs / height_in^2) * 703
        master['BMI'] = (master['VSWEIGHT'] / (master['VSHEIGHT'] ** 2)) * 703

    # Merge health history
    if not health.empty:
        health_cols = ['OASISID', 'days_to_visit', 'ALCOHOL', 'TOBAC100', 'DIABETES',
                       'HYPERTEN', 'HYPERCHO', 'CVHATT', 'DEP2YRS', 'SEIZURES']
        health_cols = [c for c in health_cols if c in health.columns]
        health_subset = health[health_cols].copy()
        health_subset = health_subset.rename(columns={
            'OASISID': 'Subject',
            'ALCOHOL': 'MH14ALCH',
            'TOBAC100': 'MH16SMOK',
            'CVHATT': 'MH4CARD',       # Cardiovascular
            'SEIZURES': 'MH2NEURL',    # Neurological
        })
        master = master.merge(health_subset, on=['Subject', 'days_to_visit'], how='left')

    # Merge GDS (depression)
    if not gds.empty:
        gds_subset = gds[['OASISID', 'days_to_visit', 'GDS']].copy()
        gds_subset = gds_subset.rename(columns={
            'OASISID': 'Subject',
            'GDS': 'BCDEPRES',
        })
        master = master.merge(gds_subset, on=['Subject', 'days_to_visit'], how='left')

    # Filter to CN, MCI, AD only (exclude Other_Dementia and Other)
    master_filtered = master[master['DX'].isin(['CN', 'MCI', 'AD'])].copy()

    # === TABULAR DATASET (all clinical visits) ===
    logger.info(f"\n--- TABULAR Dataset (all clinical visits) ---")
    logger.info(f"Total visits: {len(master_filtered)}")
    logger.info(f"Unique subjects: {master_filtered['Subject'].nunique()}")
    logger.info(f"Class distribution:")
    logger.info(master_filtered['DX'].value_counts().to_string())

    tabular_path = OASIS_DIR / 'oasis_tabular.csv'
    master_filtered.to_csv(tabular_path, index=False)
    logger.info(f"Saved to: {tabular_path}")

    # === MRI DATASET (only subjects with MRI) ===
    if mri_dir is not None:
        mri_subjects = get_subjects_with_mri(mri_dir)
        if mri_subjects:
            master_mri = master_filtered[master_filtered['Subject'].isin(mri_subjects)].copy()

            logger.info(f"\n--- MRI Dataset (subjects with MRI only) ---")
            logger.info(f"Total visits: {len(master_mri)}")
            logger.info(f"Unique subjects: {master_mri['Subject'].nunique()}")
            logger.info(f"Class distribution:")
            logger.info(master_mri['DX'].value_counts().to_string())

            mri_path = OASIS_DIR / 'oasis_mri.csv'
            master_mri.to_csv(mri_path, index=False)
            logger.info(f"Saved to: {mri_path}")

    # === FULL DATASET (including Other_Dementia and Other) ===
    logger.info(f"\n--- Full Diagnosis Breakdown (all categories) ---")
    logger.info(master['DX'].value_counts().to_string())

    full_path = OASIS_DIR / 'oasis_all_full.csv'
    master.to_csv(full_path, index=False)
    logger.info(f"Full dataset saved to: {full_path}")

    logger.info(f"\nColumns: {list(master_filtered.columns)}")

    return master_filtered


def main():
    parser = argparse.ArgumentParser(description='Build OASIS3 master dataset')
    parser.add_argument('--mri-dir', type=str, default=None,
                        help='Path to OASIS MRI directory (to filter subjects with MRI only)')
    args = parser.parse_args()

    mri_dir = Path(args.mri_dir) if args.mri_dir else None
    build_oasis_dataset(mri_dir=mri_dir)


if __name__ == '__main__':
    main()

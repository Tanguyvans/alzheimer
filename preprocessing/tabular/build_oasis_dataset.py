#!/usr/bin/env python3
"""
Build OASIS3 master dataset from UDS CSV files.

Merges multiple UDS forms and creates diagnosis labels compatible with ADNI.

Usage:
    python preprocessing/tabular/build_oasis_dataset.py
"""

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

    Returns: 'CN', 'MCI', 'AD', or 'Other'
    """
    # Check for dementia first (most severe)
    if row.get('DEMENTED') == 1:
        return 'AD'

    # Check for MCI (any MCI subtype)
    mci_fields = ['MCIAMEM', 'MCIAPLUS', 'MCINON1', 'MCINON2']
    for field in mci_fields:
        if row.get(field) == 1:
            return 'MCI'

    # Check for normal cognition
    if row.get('NORMCOG') == 1:
        return 'CN'

    return 'Other'


def build_oasis_dataset():
    """Build master OASIS dataset from UDS files"""

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

    # Start building master dataframe
    master = diagnoses[['OASISID', 'OASIS_session_label', 'days_to_visit', 'age at visit',
                        'NORMCOG', 'DEMENTED', 'MCIAMEM', 'MCIAPLUS', 'MCINON1', 'MCINON2']].copy()

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

    # Filter to CN, MCI, AD only
    master_filtered = master[master['DX'].isin(['CN', 'MCI', 'AD'])].copy()

    logger.info(f"\n--- Final Dataset ---")
    logger.info(f"Total visits: {len(master_filtered)}")
    logger.info(f"Unique subjects: {master_filtered['Subject'].nunique()}")
    logger.info(f"\nClass distribution:")
    logger.info(master_filtered['DX'].value_counts().to_string())
    logger.info(f"\nColumns: {list(master_filtered.columns)}")

    # Save
    output_path = OASIS_DIR / 'oasis_all.csv'
    master_filtered.to_csv(output_path, index=False)
    logger.info(f"\nSaved to: {output_path}")

    # Also save full dataset (including 'Other')
    full_path = OASIS_DIR / 'oasis_all_full.csv'
    master.to_csv(full_path, index=False)
    logger.info(f"Full dataset saved to: {full_path}")

    return master_filtered


if __name__ == '__main__':
    build_oasis_dataset()

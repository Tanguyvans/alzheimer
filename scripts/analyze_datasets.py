#!/usr/bin/env python3
"""
Analyze Datasets - Verify statistics for ADNI, OASIS, and NACC

Usage:
    python scripts/analyze_datasets.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
ADNI_SKULL_DIR = Path("/Volumes/KINGSTON/ADNI-skull")
OASIS_SKULL_DIR = Path("/Volumes/KINGSTON/OASIS-registered")


def compute_trajectory_categories(df, id_col, dx_col, date_col, dx_map):
    """
    Compute patient trajectory categories.

    Args:
        df: DataFrame with diagnosis data
        id_col: Column name for patient ID
        dx_col: Column name for diagnosis
        date_col: Column name for date
        dx_map: Dict mapping diagnosis codes to names (e.g., {1: 'CN', 2: 'MCI', 3: 'AD'})

    Returns:
        Dict with category counts
    """
    df = df.dropna(subset=[dx_col]).copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.sort_values([id_col, date_col])

    # Get sequences
    patient_sequences = df.groupby(id_col)[dx_col].apply(list).reset_index()

    # Define categories based on dx_map keys
    codes = sorted(dx_map.keys())
    cn_code, mci_code, ad_code = codes[0], codes[1], codes[2]

    categories = {
        f'Stable {dx_map[cn_code]}': 0,
        f'Stable {dx_map[mci_code]}': 0,
        f'Stable {dx_map[ad_code]}': 0,
        f'{dx_map[cn_code]} → {dx_map[mci_code]}': 0,
        f'{dx_map[mci_code]} → {dx_map[ad_code]}': 0,
        f'{dx_map[cn_code]} → {dx_map[ad_code]}': 0,
        f'{dx_map[cn_code]} → {dx_map[mci_code]} → {dx_map[ad_code]}': 0,
        'Other': 0
    }

    for _, row in patient_sequences.iterrows():
        seq = [int(x) for x in row[dx_col]]
        unique = set(seq)

        if unique == {cn_code}:
            categories[f'Stable {dx_map[cn_code]}'] += 1
        elif unique == {mci_code}:
            categories[f'Stable {dx_map[mci_code]}'] += 1
        elif unique == {ad_code}:
            categories[f'Stable {dx_map[ad_code]}'] += 1
        elif unique == {cn_code, mci_code} and seq[0] == cn_code:
            categories[f'{dx_map[cn_code]} → {dx_map[mci_code]}'] += 1
        elif unique == {mci_code, ad_code} and seq[0] == mci_code:
            categories[f'{dx_map[mci_code]} → {dx_map[ad_code]}'] += 1
        elif unique == {cn_code, ad_code} and seq[0] == cn_code:
            categories[f'{dx_map[cn_code]} → {dx_map[ad_code]}'] += 1
        elif cn_code in unique and mci_code in unique and ad_code in unique:
            categories[f'{dx_map[cn_code]} → {dx_map[mci_code]} → {dx_map[ad_code]}'] += 1
        else:
            categories['Other'] += 1

    return categories, len(patient_sequences)


def compute_transition_matrix(df, id_col, dx_col, date_col, dx_map):
    """
    Compute visit-to-visit transition matrix.

    Returns:
        Dict with transition counts and percentages
    """
    df = df.dropna(subset=[dx_col]).copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.sort_values([id_col, date_col])

    labels = list(dx_map.values())
    transitions = {f'{f}->{t}': 0 for f in labels for t in labels}

    for _, group in df.groupby(id_col):
        diagnoses = group[dx_col].tolist()
        for i in range(len(diagnoses) - 1):
            from_dx = dx_map.get(int(diagnoses[i]))
            to_dx = dx_map.get(int(diagnoses[i+1]))
            if from_dx and to_dx:
                transitions[f'{from_dx}->{to_dx}'] += 1

    return transitions, labels


def print_trajectory_analysis(categories, total, transitions, labels):
    """Print trajectory categories and transition matrix."""
    print(f"\n  Patient Trajectory Categories:")
    for cat, count in categories.items():
        if count > 0:
            pct = 100 * count / total
            print(f"    {cat}: {count:,} ({pct:.1f}%)")

    print(f"\n  Transition Matrix (visit-to-visit):")
    col_width = max(10, max(len(l) for l in labels) + 2)
    print(f"    {'From↓/To→':<12}", end='')
    for label in labels:
        print(f"{label:>{col_width}}", end='')
    print()
    print("    " + "-" * (12 + col_width * len(labels)))

    for from_label in labels:
        total_from = sum(transitions[f'{from_label}->{to}'] for to in labels)
        print(f"    {from_label:<12}", end='')
        for to_label in labels:
            count = transitions[f'{from_label}->{to_label}']
            if total_from > 0:
                pct = 100 * count / total_from
                print(f"{pct:>{col_width-1}.1f}%", end='')
            else:
                print(f"{'N/A':>{col_width}}", end='')
        print()


def analyze_adni():
    """Analyze ADNI dataset statistics."""
    print("=" * 60)
    print("ADNI (Alzheimer's Disease Neuroimaging Initiative)")
    print("=" * 60)

    # Full diagnosis data from dxsum.csv
    dxsum = DATA_DIR / "adni" / "dxsum.csv"
    if dxsum.exists():
        df = pd.read_csv(dxsum)
        print(f"\ndxsum.csv (full diagnosis data):")
        print(f"  Total patients: {df['PTID'].nunique():,}")
        print(f"  Total visits: {len(df):,}")
        print(f"  Visits per patient: {len(df)/df['PTID'].nunique():.1f} avg")

        # Diagnosis distribution
        diag_map = {1: 'CN', 2: 'MCI', 3: 'AD'}
        print(f"\n  Diagnosis (all visits):")
        for code in [1, 2, 3]:
            count = (df['DIAGNOSIS'] == code).sum()
            pct = 100 * count / len(df)
            print(f"    {diag_map[code]}: {count:,} ({pct:.0f}%)")

        # First visit
        first = df.sort_values('EXAMDATE').groupby('PTID').first()
        print(f"\n  First visit diagnosis:")
        for code in [1, 2, 3]:
            count = (first['DIAGNOSIS'] == code).sum()
            print(f"    {diag_map[code]}: {count:,}")

        # Phases
        print(f"\n  Phases:")
        for phase, count in df['PHASE'].value_counts().items():
            print(f"    {phase}: {count:,}")

        # Trajectory analysis
        categories, total = compute_trajectory_categories(
            df, 'RID', 'DIAGNOSIS', 'EXAMDATE', diag_map
        )
        transitions, labels = compute_transition_matrix(
            df, 'RID', 'DIAGNOSIS', 'EXAMDATE', diag_map
        )
        print_trajectory_analysis(categories, total, transitions, labels)

    # MRI scans from ADNI-skull folder
    if ADNI_SKULL_DIR.exists():
        mri_subjects = [d.name for d in ADNI_SKULL_DIR.iterdir()
                       if d.is_dir() and not d.name.startswith('.')]
        mri_count = sum(1 for f in ADNI_SKULL_DIR.rglob('*.nii.gz'))
        print(f"\n  T1 MRI scans: {mri_count:,}")
        print(f"  Patients with MRI: {len(mri_subjects):,}")
        print(f"  MRI per patient: {mri_count/len(mri_subjects):.1f} avg")

    # ML subsets
    adni_cn_ad = DATA_DIR / "adni" / "adni_cn_ad.csv"
    if adni_cn_ad.exists():
        df = pd.read_csv(adni_cn_ad)
        print(f"\n  ML subset (adni_cn_ad.csv): {len(df):,} samples")
        if 'DX' in df.columns:
            for dx, count in df['DX'].value_counts().items():
                print(f"    {dx}: {count:,}")


def analyze_oasis():
    """Analyze OASIS dataset statistics."""
    print("\n" + "=" * 60)
    print("OASIS (Open Access Series of Imaging Studies)")
    print("=" * 60)

    # All visits (full)
    oasis_full = DATA_DIR / "oasis" / "oasis_all_full.csv"
    if oasis_full.exists():
        df = pd.read_csv(oasis_full)
        print(f"\noasis_all_full.csv:")
        print(f"  Total visits: {len(df):,}")
        print(f"  Unique subjects: {df['Subject'].nunique():,}")
        if 'DX' in df.columns:
            print(f"  DX distribution (all visits):")
            for dx, count in df['DX'].value_counts().items():
                print(f"    {dx}: {count:,}")

            # First visit only
            first = df.sort_values('days_to_visit').groupby('Subject').first()
            print(f"\n  First visit only:")
            for dx, count in first['DX'].value_counts().items():
                print(f"    {dx}: {count:,}")

            # Trajectory analysis (for CN, MCI, AD only)
            # Map string DX to numeric codes
            dx_to_code = {'CN': 1, 'MCI': 2, 'AD': 3}
            df_filtered = df[df['DX'].isin(['CN', 'MCI', 'AD'])].copy()
            df_filtered['DX_CODE'] = df_filtered['DX'].map(dx_to_code)

            if len(df_filtered) > 0:
                diag_map = {1: 'CN', 2: 'MCI', 3: 'AD'}
                categories, total = compute_trajectory_categories(
                    df_filtered, 'Subject', 'DX_CODE', 'days_to_visit', diag_map
                )
                transitions, labels = compute_transition_matrix(
                    df_filtered, 'Subject', 'DX_CODE', 'days_to_visit', diag_map
                )
                print(f"\n  (CN/MCI/AD patients only: {total})")
                print_trajectory_analysis(categories, total, transitions, labels)

    # MRI sessions
    oasis_mri = DATA_DIR / "oasis" / "oasis_mri.csv"
    if oasis_mri.exists():
        df = pd.read_csv(oasis_mri)
        print(f"\noasis_mri.csv:")
        print(f"  Total MRI sessions: {len(df):,}")
        print(f"  Unique subjects: {df['Subject'].nunique():,}")

    # T1 scans
    oasis_t1 = DATA_DIR / "oasis" / "oasis-t1_12_16_2025.csv"
    if oasis_t1.exists():
        df = pd.read_csv(oasis_t1)
        print(f"\noasis-t1_12_16_2025.csv:")
        print(f"  Total T1 scans: {len(df):,}")
        print(f"  Unique subjects: {df['Subject'].nunique():,}")


def analyze_nacc():
    """Analyze NACC dataset statistics."""
    print("\n" + "=" * 60)
    print("NACC (National Alzheimer's Coordinating Center)")
    print("=" * 60)

    # Full investigator file
    nacc_full = DATA_DIR / "nacc" / "investigator_ftldlbd_nacc71.csv"
    if nacc_full.exists():
        df = pd.read_csv(nacc_full, low_memory=False)
        print(f"\ninvestigator_ftldlbd_nacc71.csv:")
        print(f"  Total visits: {len(df):,}")
        print(f"  Unique subjects: {df['NACCID'].nunique():,}")

        if 'NACCUDSD' in df.columns:
            print(f"\n  NACCUDSD distribution (all visits):")
            udsd_map = {1: 'CN', 2: 'Impaired-not-MCI', 3: 'MCI', 4: 'Dementia'}
            for code in sorted(df['NACCUDSD'].dropna().unique()):
                count = (df['NACCUDSD'] == code).sum()
                label = udsd_map.get(int(code), f'Code {code}')
                print(f"    {int(code)} ({label}): {count:,}")

            # First visit only
            first = df.sort_values('VISITYR').groupby('NACCID').first()
            print(f"\n  First visit only:")
            for code in sorted(first['NACCUDSD'].dropna().unique()):
                count = (first['NACCUDSD'] == code).sum()
                label = udsd_map.get(int(code), f'Code {code}')
                print(f"    {int(code)} ({label}): {count:,}")

            # Trajectory analysis (CN=1, MCI=3, Dementia=4)
            # Filter to CN, MCI, Dementia only (exclude Impaired-not-MCI)
            df_filtered = df[df['NACCUDSD'].isin([1, 3, 4])].copy()
            # Remap: 1->1 (CN), 3->2 (MCI), 4->3 (Dementia)
            remap = {1: 1, 3: 2, 4: 3}
            df_filtered['DX_CODE'] = df_filtered['NACCUDSD'].map(remap)

            if len(df_filtered) > 0:
                diag_map = {1: 'CN', 2: 'MCI', 3: 'Dementia'}
                categories, total = compute_trajectory_categories(
                    df_filtered, 'NACCID', 'DX_CODE', 'VISITYR', diag_map
                )
                transitions, labels = compute_transition_matrix(
                    df_filtered, 'NACCID', 'DX_CODE', 'VISITYR', diag_map
                )
                print(f"\n  (CN/MCI/Dementia patients only: {total})")
                print_trajectory_analysis(categories, total, transitions, labels)

    # MRI subset
    nacc_mri = DATA_DIR / "nacc" / "nacc_tabular_mri.csv"
    if nacc_mri.exists():
        df = pd.read_csv(nacc_mri)
        print(f"\nnacc_tabular_mri.csv (MRI subset):")
        print(f"  Total samples: {len(df):,}")
        if 'NACCID' in df.columns:
            print(f"  Unique subjects: {df['NACCID'].nunique():,}")
        if 'DX' in df.columns:
            print(f"  DX distribution:")
            for dx, count in df['DX'].value_counts().items():
                print(f"    {dx}: {count:,}")

    # T1 scans
    nacc_t1 = DATA_DIR / "nacc" / "nacc-t1_12_16_2025.csv"
    if nacc_t1.exists():
        df = pd.read_csv(nacc_t1)
        print(f"\nnacc-t1_12_16_2025.csv:")
        print(f"  Total T1 scans: {len(df):,}")


def print_summary():
    """Print summary table."""
    print("\n" + "=" * 60)
    print("SUMMARY TABLE")
    print("=" * 60)

    data = []

    # ADNI - use dxsum.csv for visits and ADNI-skull for MRI
    adni_dxsum = DATA_DIR / "adni" / "dxsum.csv"
    if adni_dxsum.exists():
        df = pd.read_csv(adni_dxsum)
        total_visits = len(df)
        unique_subjects = df['PTID'].nunique()

        # Count MRI from ADNI-skull folder
        mri_count = '-'
        mri_subjects = '-'
        if ADNI_SKULL_DIR.exists():
            mri_subjects = len([d for d in ADNI_SKULL_DIR.iterdir()
                               if d.is_dir() and not d.name.startswith('.')])
            mri_count = sum(1 for f in ADNI_SKULL_DIR.rglob('*.nii.gz'))

        # Diagnosis counts (all visits)
        cn = (df['DIAGNOSIS'] == 1).sum()
        mci = (df['DIAGNOSIS'] == 2).sum()
        ad = (df['DIAGNOSIS'] == 3).sum()

        data.append({
            'Dataset': 'ADNI',
            'Patients': unique_subjects,
            'Total Visits': total_visits,
            'T1 MRI Scans': mri_count,
            'MRI Subjects': mri_subjects,
            'CN (visits)': cn,
            'MCI (visits)': mci,
            'AD (visits)': ad
        })

    # OASIS
    oasis_full = DATA_DIR / "oasis" / "oasis_all_full.csv"
    if oasis_full.exists():
        df = pd.read_csv(oasis_full)
        total_visits = len(df)
        unique_subjects = df['Subject'].nunique()

        # Count MRI from OASIS-registered folder
        mri_count = '-'
        mri_subjects = '-'
        if OASIS_SKULL_DIR.exists():
            mri_subjects = len([d for d in OASIS_SKULL_DIR.iterdir()
                               if d.is_dir() and not d.name.startswith('.')])
            mri_count = sum(1 for f in OASIS_SKULL_DIR.rglob('*.nii.gz'))

        # Diagnosis counts (all visits)
        cn = (df['DX'] == 'CN').sum() if 'DX' in df.columns else '-'
        mci = (df['DX'] == 'MCI').sum() if 'DX' in df.columns else '-'
        ad = (df['DX'] == 'AD').sum() if 'DX' in df.columns else '-'

        data.append({
            'Dataset': 'OASIS',
            'Patients': unique_subjects,
            'Total Visits': total_visits,
            'T1 MRI Scans': mri_count,
            'MRI Subjects': mri_subjects,
            'CN (visits)': cn,
            'MCI (visits)': mci,
            'AD (visits)': ad
        })

    # NACC
    nacc_full = DATA_DIR / "nacc" / "investigator_ftldlbd_nacc71.csv"
    nacc_t1 = DATA_DIR / "nacc" / "nacc-t1_12_16_2025.csv"
    if nacc_full.exists():
        df = pd.read_csv(nacc_full, low_memory=False)
        total_visits = len(df)
        unique_subjects = df['NACCID'].nunique()

        t1_count = '-'
        if nacc_t1.exists():
            t1_count = len(pd.read_csv(nacc_t1))

        # Diagnosis counts (all visits) - NACCUDSD: 1=CN, 3=MCI, 4=Dementia
        cn = (df['NACCUDSD'] == 1).sum()
        mci = (df['NACCUDSD'] == 3).sum()
        dementia = (df['NACCUDSD'] == 4).sum()

        data.append({
            'Dataset': 'NACC',
            'Patients': unique_subjects,
            'Total Visits': total_visits,
            'T1 MRI Scans': t1_count,
            'MRI Subjects': '-',
            'CN (visits)': cn,
            'MCI (visits)': mci,
            'AD (visits)': dementia
        })

    # Print table
    if data:
        df_summary = pd.DataFrame(data)
        print(f"\n{df_summary.to_string(index=False)}")


def main():
    print("Dataset Analysis Report")
    print("Generated for DATASETS.md verification\n")

    analyze_adni()
    analyze_oasis()
    analyze_nacc()
    print_summary()


if __name__ == '__main__':
    main()

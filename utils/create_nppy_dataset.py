#!/usr/bin/env python3
"""
Create tar.gz dataset from NPPY preprocessed scans

Reads the required_patients.txt file and creates a compressed archive
containing the NPPY preprocessed scans for those patients with their
diagnoses (CN, MCI, AD) from dxsum.csv.

Usage:
    python create_nppy_dataset.py \
        --patient-list experiments/cn_mci_ad_3dhcct/required_patients.txt \
        --nppy-dir /Volumes/KINGSTON/ADNI_nppy \
        --dxsum /Volumes/KINGSTON/dxsum.csv \
        --output adni_nppy_dataset.tar.gz
"""

import argparse
import tarfile
from pathlib import Path
from tqdm import tqdm
import sys


def load_scan_list(scan_list_file):
    """Load list of NIfTI scan paths"""
    with open(scan_list_file, 'r') as f:
        scans = [line.strip() for line in f if line.strip()]
    return scans


def get_nppy_path(nifti_path, nppy_base):
    """
    Convert NIfTI scan path to corresponding NPPY path

    Example:
        Input:  /Volumes/KINGSTON/ADNI_nifti/018_S_0043/MPRAGE_Repeat_2009-01-21_10_42_57.0_I134226_134226.nii.gz
        Output: /Volumes/KINGSTON/ADNI_nppy/018_S_0043/MPRAGE_Repeat_2009-01-21_10_42_57_mni_norm.nii.gz
    """
    nifti_path = Path(nifti_path)
    patient_id = nifti_path.parent.name
    scan_name = nifti_path.name

    # Extract base name (remove .0_I#####_#####.nii.gz suffix)
    nifti_base = scan_name.replace('.nii.gz', '').replace('.nii', '')
    if '.0_I' in nifti_base:
        nifti_base = nifti_base.split('.0_I')[0]

    # NPPY output name: {base}_mni_norm.nii.gz
    nppy_name = f"{nifti_base}_mni_norm.nii.gz"
    nppy_path = nppy_base / patient_id / nppy_name

    return nppy_path


def load_diagnosis_data(dxsum_path='/Volumes/KINGSTON/dxsum.csv'):
    """Load diagnosis data to get class labels

    Note: Since we only include stable patients (diagnosis never changes),
    we can take any visit's diagnosis value for each patient.
    """
    import pandas as pd

    dx_df = pd.read_csv(dxsum_path)
    # Map diagnosis: 1=CN, 2=MCI, 3=AD
    dx_df['Group'] = dx_df['DIAGNOSIS'].map({1: 'CN', 2: 'MCI', 3: 'AD'})

    # Get first row for each patient (since stable, all rows have same diagnosis)
    first_rows = dx_df.groupby('PTID').first()

    # Create patient_id -> Group mapping
    patient_to_group = dict(zip(first_rows.index, first_rows['Group']))

    return patient_to_group


def create_dataset_archive(patient_list_file, nppy_dir, output_file, dxsum_path, blacklist_file=None):
    """Create tar.gz archive of NPPY dataset"""

    print("="*60)
    print("CREATING NPPY DATASET ARCHIVE")
    print("="*60)

    # Load blacklist if provided
    blacklist = set()
    if blacklist_file and Path(blacklist_file).exists():
        print(f"\nLoading blacklist from: {blacklist_file}")
        with open(blacklist_file, 'r') as f:
            blacklist = {line.strip() for line in f if line.strip()}
        print(f"Loaded {len(blacklist)} blacklisted patients")

    # Load diagnosis data
    print(f"\nLoading diagnosis data from: {dxsum_path}")
    patient_to_group = load_diagnosis_data(dxsum_path)
    print(f"Loaded diagnosis for {len(patient_to_group)} patients")

    # Load patient list
    print(f"\nLoading patient list from: {patient_list_file}")
    with open(patient_list_file, 'r') as f:
        patients = [line.strip() for line in f if line.strip()]
    print(f"Found {len(patients)} patients in list")

    # Filter out blacklisted patients
    if blacklist:
        patients_before = len(patients)
        patients = [p for p in patients if p not in blacklist]
        filtered = patients_before - len(patients)
        if filtered > 0:
            print(f"Filtered out {filtered} blacklisted patients ({len(patients)} remaining)")

    nppy_base = Path(nppy_dir)
    output_path = Path(output_file)

    # Find NPPY scans for each patient
    print(f"\nFinding NPPY scans in: {nppy_base}")
    nppy_scans = []
    missing_patients = []
    class_counts = {'CN': 0, 'MCI': 0, 'AD': 0, 'Unknown': 0}

    for patient_id in tqdm(patients, desc="Locating NPPY scans"):
        # Get diagnosis first
        diagnosis = patient_to_group.get(patient_id, 'Unknown')

        # Skip patients with unknown diagnosis
        if diagnosis == 'Unknown':
            missing_patients.append((patient_id, "Unknown diagnosis (not in dxsum.csv)"))
            continue

        # Find NPPY directory for this patient
        patient_dir = nppy_base / patient_id

        if not patient_dir.exists():
            missing_patients.append((patient_id, "Directory not found"))
            continue

        # Find NPPY scans in patient directory
        nppy_files = list(patient_dir.glob('*_mni_norm.nii.gz'))
        nppy_files = [f for f in nppy_files if not f.name.startswith('.')]

        if not nppy_files:
            missing_patients.append((patient_id, "No NPPY scans found"))
            continue

        # Take the first NPPY scan for this patient
        nppy_path = nppy_files[0]

        class_counts[diagnosis] += 1
        nppy_scans.append((nppy_path, patient_id, diagnosis))

    print(f"\nFound: {len(nppy_scans)}/{len(patients)} NPPY scans with known diagnosis")
    print(f"\nClass distribution:")
    print(f"  CN:  {class_counts['CN']:4d} scans")
    print(f"  MCI: {class_counts['MCI']:4d} scans")
    print(f"  AD:  {class_counts['AD']:4d} scans")

    if missing_patients:
        print(f"\n⚠️  Warning: {len(missing_patients)} patients without NPPY scans:")
        for patient_id, reason in missing_patients[:5]:
            print(f"  - {patient_id}: {reason}")
        if len(missing_patients) > 5:
            print(f"  ... and {len(missing_patients) - 5} more")

    # Calculate total size
    print("\nCalculating dataset size...")
    total_size = sum(scan_path.stat().st_size for scan_path, _, _ in tqdm(nppy_scans, desc="Computing size"))
    total_size_mb = total_size / (1024 * 1024)
    total_size_gb = total_size / (1024 * 1024 * 1024)

    print(f"Total dataset size: {total_size_mb:.2f} MB ({total_size_gb:.2f} GB)")

    # Auto-generate filename with class counts if not specified
    if output_file == 'adni_nppy_dataset.tar.gz':
        output_file = f"adni_stable_cn{class_counts['CN']}_mci{class_counts['MCI']}_ad{class_counts['AD']}_nppy.tar.gz"
        output_path = Path(output_file)
        print(f"\nAuto-generated filename: {output_file}")

    # Create tar.gz archive
    print(f"\nCreating archive: {output_path}")
    print("This may take several minutes...")

    with tarfile.open(output_path, 'w:gz') as tar:
        for nppy_scan, patient_id, diagnosis in tqdm(nppy_scans, desc="Archiving scans"):
            scan_name = nppy_scan.name

            # Add to archive with structure: patient_id/scan_name
            arcname = f"{patient_id}/{scan_name}"
            tar.add(nppy_scan, arcname=arcname)

    # Check output file size
    output_size = output_path.stat().st_size / (1024 * 1024 * 1024)
    compression_ratio = (1 - output_size * 1024 / total_size_gb) * 100

    print(f"\n✅ Dataset archive created successfully!")
    print(f"   Archive: {output_path}")
    print(f"   Size: {output_size:.2f} GB")
    print(f"   Compression: {compression_ratio:.1f}%")
    print(f"   Scans included: {len(nppy_scans)}")
    print(f"   Missing: {len(missing_patients)}")

    # Create manifest file
    manifest_file = output_path.with_suffix('.txt')
    print(f"\nCreating manifest: {manifest_file}")

    with open(manifest_file, 'w') as f:
        f.write("ADNI NPPY Preprocessed Dataset\n")
        f.write("="*60 + "\n\n")
        f.write(f"Total scans: {len(nppy_scans)}\n")
        f.write(f"Archive size: {output_size:.2f} GB\n")
        f.write(f"Uncompressed size: {total_size_gb:.2f} GB\n")
        f.write(f"Compression ratio: {compression_ratio:.1f}%\n\n")
        f.write("Class distribution:\n")
        f.write(f"  CN (Cognitively Normal):        {class_counts['CN']:4d} scans\n")
        f.write(f"  MCI (Mild Cognitive Impairment): {class_counts['MCI']:4d} scans\n")
        f.write(f"  AD (Alzheimer's Disease):       {class_counts['AD']:4d} scans\n\n")
        f.write("Preprocessing pipeline:\n")
        f.write("  - NPPY (Neural Preprocessing Python)\n")
        f.write("  - End-to-end learned preprocessing\n")
        f.write("  - Spatial and intensity normalization\n\n")
        f.write("Quality filtering applied:\n")
        f.write("  - Dimension whitelist: 176×240×256, 160×192×192, 240×256×208\n")
        f.write("  - File size filter: ≤20MB\n")
        f.write("  - Manual blacklist: 1 patient (073_S_4230)\n")
        f.write("  - Stable diagnosis: baseline visit only\n\n")
        f.write("Scan list:\n")
        f.write("-"*60 + "\n")

        for nppy_scan, patient_id, diagnosis in sorted(nppy_scans):
            scan_name = nppy_scan.name
            f.write(f"{patient_id}/{scan_name} [{diagnosis}]\n")

        if missing_patients:
            f.write("\n\nMissing patients (not in archive):\n")
            f.write("-"*60 + "\n")
            for patient_id, reason in missing_patients:
                f.write(f"{patient_id}: {reason}\n")

    print(f"✅ Manifest created: {manifest_file}")

    print("\n" + "="*60)
    print("DATASET ARCHIVE COMPLETE")
    print("="*60)
    print(f"\nTo extract:")
    print(f"  tar -xzf {output_path.name}")
    print(f"\nTo view contents:")
    print(f"  tar -tzf {output_path.name} | head")


def main():
    parser = argparse.ArgumentParser(
        description='Create tar.gz dataset from NPPY preprocessed scans',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--patient-list', type=str, required=True,
                       help='Path to required_patients.txt file')
    parser.add_argument('--nppy-dir', type=str, default='/Volumes/KINGSTON/ADNI_nppy',
                       help='Path to NPPY preprocessed directory')
    parser.add_argument('--dxsum', type=str, default='/Volumes/KINGSTON/dxsum.csv',
                       help='Path to dxsum.csv diagnosis file')
    parser.add_argument('--blacklist', type=str, default=None,
                       help='Path to blacklist.txt file (optional)')
    parser.add_argument('--output', type=str, default='adni_nppy_dataset.tar.gz',
                       help='Output tar.gz filename (auto-generated if default)')

    args = parser.parse_args()

    # Validate inputs
    patient_list_file = Path(args.patient_list)
    if not patient_list_file.exists():
        print(f"❌ Error: Patient list not found: {patient_list_file}")
        sys.exit(1)

    nppy_dir = Path(args.nppy_dir)
    if not nppy_dir.exists():
        print(f"❌ Error: NPPY directory not found: {nppy_dir}")
        sys.exit(1)

    dxsum_path = Path(args.dxsum)
    if not dxsum_path.exists():
        print(f"❌ Error: Diagnosis file not found: {dxsum_path}")
        sys.exit(1)

    create_dataset_archive(patient_list_file, nppy_dir, args.output, args.dxsum, args.blacklist)


if __name__ == '__main__':
    main()

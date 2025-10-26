#!/usr/bin/env python3
"""
Prepare ADNI Dataset CSV for DenseNet3D Training

Creates a CSV file with paths to skull-stripped NIfTI files (.nii.gz format)
for training the DenseNet3D model.

The script expects skull-stripped NIfTI files organized by diagnosis group.

Usage:
    # For standard AD/CN/MCI folders:
    python prepare_training_data.py --input /Volumes/KINGSTON/ADNI-skull --output data/adni_training.csv

    # For organized progression folders (MCI->AD, etc.):
    python prepare_training_data.py --input /Volumes/KINGSTON/ADNI-MCI-organized --output data/adni_mci_training.csv --organized
"""

import argparse
import logging
from pathlib import Path
import pandas as pd
import nibabel as nib
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_patient_info(filename: str) -> dict:
    """
    Extract patient information from ADNI filename

    Expected format: XXX_S_XXXX_bl_...nii.gz or XXX_S_XXXX_m##_...nii.gz

    Returns:
        dict with 'Subject', 'RID', 'PTID', 'visit'
    """
    parts = filename.replace('.nii.gz', '').replace('.nii', '').split('_')

    if len(parts) >= 3:
        # Extract subject ID (e.g., 002_S_0729)
        subject_id = '_'.join(parts[:3])
        rid = parts[2] if len(parts) > 2 else None

        # Extract visit
        visit = 'baseline'
        if len(parts) > 3:
            if parts[3] == 'bl':
                visit = 'baseline'
            elif parts[3].startswith('m'):
                visit = parts[3]

        return {
            'Subject': subject_id,
            'RID': rid,
            'PTID': subject_id,
            'visit': visit
        }

    return {
        'Subject': filename,
        'RID': None,
        'PTID': filename,
        'visit': 'unknown'
    }


def validate_nifti(file_path: Path) -> bool:
    """
    Validate that file is a readable NIfTI file

    Args:
        file_path: Path to NIfTI file

    Returns:
        True if valid, False otherwise
    """
    try:
        nii = nib.load(str(file_path))
        shape = nii.shape
        # Check reasonable dimensions (skull-stripped brain)
        if len(shape) != 3:
            return False
        if any(d < 10 or d > 512 for d in shape):
            return False
        return True
    except Exception as e:
        logger.warning(f"Invalid NIfTI file {file_path.name}: {e}")
        return False


def find_nifti_files(folder: Path, group: str, validate: bool = True) -> list:
    """
    Find all valid NIfTI files in folder

    Args:
        folder: Path to search
        group: Diagnosis group (AD, CN, MCI)
        validate: Whether to validate NIfTI files

    Returns:
        List of dicts with file info
    """
    nifti_files = []

    # Find .nii.gz files (standard for skull-stripped data)
    for file_path in folder.glob('**/*.nii.gz'):
        # Skip hidden files
        if file_path.name.startswith('.') or file_path.name.startswith('._'):
            continue

        # Validate if requested
        if validate and not validate_nifti(file_path):
            continue

        # Extract patient info
        patient_info = extract_patient_info(file_path.name)

        # Create record
        record = {
            'nii_path': str(file_path.absolute()),
            'Subject': patient_info['Subject'],
            'RID': patient_info['RID'],
            'PTID': patient_info['PTID'],
            'Group': group,
            'visit': patient_info['visit'],
            'file_name': file_path.name
        }

        nifti_files.append(record)

    # Also check for .nii files
    for file_path in folder.glob('**/*.nii'):
        if file_path.name.startswith('.') or file_path.name.startswith('._'):
            continue

        if validate and not validate_nifti(file_path):
            continue

        patient_info = extract_patient_info(file_path.name)

        record = {
            'nii_path': str(file_path.absolute()),
            'Subject': patient_info['Subject'],
            'RID': patient_info['RID'],
            'PTID': patient_info['PTID'],
            'Group': group,
            'visit': patient_info['visit'],
            'file_name': file_path.name
        }

        nifti_files.append(record)

    return nifti_files


def prepare_standard_dataset(input_dir: str, output_csv: str, groups: list = None, validate: bool = True):
    """
    Prepare training dataset from standard AD/CN/MCI folders

    Args:
        input_dir: Root directory containing AD, CN, MCI subfolders
        output_csv: Output CSV path
        groups: List of groups to include
        validate: Whether to validate NIfTI files
    """
    input_path = Path(input_dir)

    if not input_path.exists():
        logger.error(f"Input directory not found: {input_dir}")
        return None

    if groups is None:
        groups = ['AD', 'CN', 'MCI']

    print("\n" + "="*80)
    print("📊 PREPARING ADNI TRAINING DATASET (Skull-Stripped NIfTI)")
    print("="*80 + "\n")
    print(f"Input: {input_dir}")
    print(f"Groups: {', '.join(groups)}\n")

    all_records = []

    # Process each group
    for group in groups:
        group_folder = input_path / group

        if not group_folder.exists():
            logger.warning(f"Group folder not found: {group_folder}")
            continue

        logger.info(f"Processing {group} group...")

        # Find all NIfTI files
        records = find_nifti_files(group_folder, group, validate=validate)

        logger.info(f"  Found {len(records)} valid skull-stripped files")

        all_records.extend(records)

    if not all_records:
        logger.error("No valid NIfTI files found!")
        return None

    # Create DataFrame
    df = pd.DataFrame(all_records)

    # Summary statistics
    print("\n" + "="*80)
    print("📈 DATASET SUMMARY")
    print("="*80 + "\n")

    print(f"Total scans: {len(df):,}")

    print(f"\nScans per group:")
    group_counts = df['Group'].value_counts().sort_index()
    for group, count in group_counts.items():
        percentage = (count / len(df) * 100)
        print(f"  {group:6s}: {count:5,d} scans ({percentage:5.1f}%)")

    print(f"\nUnique patients: {df['Subject'].nunique():,}")

    # Visit distribution
    print(f"\nVisit distribution:")
    visit_counts = df['visit'].value_counts()
    for visit, count in visit_counts.head(5).items():
        print(f"  {visit:12s}: {count:5,d}")

    # Save to CSV
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)

    print(f"\n✅ Dataset saved to: {output_path}")

    # Show sample
    print("\n" + "="*80)
    print("📋 SAMPLE RECORDS")
    print("="*80 + "\n")
    print(df[['file_name', 'Subject', 'Group', 'visit']].head(3).to_string(index=False))

    print("\n" + "="*80 + "\n")

    return df


def prepare_organized_dataset(input_dir: str, output_csv: str, validate: bool = True):
    """
    Prepare dataset from organized MCI progression folders

    Args:
        input_dir: Root directory containing MCI->AD, MCI->CN, etc. folders
        output_csv: Output CSV path
        validate: Whether to validate NIfTI files
    """
    input_path = Path(input_dir)

    if not input_path.exists():
        logger.error(f"Input directory not found: {input_dir}")
        return None

    print("\n" + "="*80)
    print("📊 PREPARING ORGANIZED MCI PROGRESSION DATASET")
    print("="*80 + "\n")
    print(f"Input: {input_dir}\n")

    all_records = []

    # Folder to group mapping
    progression_to_group = {
        'AD->AD': 'AD',
        'CN->CN': 'CN',
        'MCI->AD': 'MCI',
        'MCI->CN': 'MCI',
        'MCI->MCI': 'MCI'
    }

    # Process each progression folder
    for progression_folder in sorted(input_path.iterdir()):
        if not progression_folder.is_dir():
            continue

        progression_name = progression_folder.name

        if progression_name not in progression_to_group:
            logger.warning(f"Unknown progression folder: {progression_name}")
            continue

        group = progression_to_group[progression_name]

        logger.info(f"Processing {progression_name}...")

        # Find all NIfTI files
        records = find_nifti_files(progression_folder, group, validate=validate)

        # Add progression info
        for record in records:
            record['progression'] = progression_name

        logger.info(f"  Found {len(records)} valid files")

        all_records.extend(records)

    if not all_records:
        logger.error("No valid NIfTI files found!")
        return None

    # Create DataFrame
    df = pd.DataFrame(all_records)

    # Summary
    print("\n" + "="*80)
    print("📈 DATASET SUMMARY")
    print("="*80 + "\n")

    print(f"Total scans: {len(df):,}")

    print(f"\nScans per group:")
    group_counts = df['Group'].value_counts().sort_index()
    for group, count in group_counts.items():
        percentage = (count / len(df) * 100)
        print(f"  {group:6s}: {count:5,d} ({percentage:5.1f}%)")

    print(f"\nScans per progression:")
    prog_counts = df['progression'].value_counts()
    for prog, count in sorted(prog_counts.items()):
        percentage = (count / len(df) * 100)
        print(f"  {prog:10s}: {count:5,d} ({percentage:5.1f}%)")

    print(f"\nUnique patients: {df['Subject'].nunique():,}")

    # Save to CSV
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)

    print(f"\n✅ Dataset saved to: {output_path}")

    # Show sample
    print("\n" + "="*80)
    print("📋 SAMPLE RECORDS")
    print("="*80 + "\n")
    print(df[['file_name', 'Subject', 'Group', 'progression', 'visit']].head(3).to_string(index=False))

    print("\n" + "="*80 + "\n")

    return df


def main():
    parser = argparse.ArgumentParser(
        description='Prepare skull-stripped ADNI dataset CSV for training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard AD/CN/MCI folders:
  python prepare_training_data.py --input /Volumes/KINGSTON/ADNI-skull --output data/adni_training.csv

  # Organized MCI progression folders:
  python prepare_training_data.py --input /Volumes/KINGSTON/ADNI-MCI-organized --output data/adni_mci.csv --organized
        """
    )

    parser.add_argument('--input', '-i', type=str, required=True,
                       help='Input directory with skull-stripped NIfTI files')
    parser.add_argument('--output', '-o', type=str, default='data/adni_training_data.csv',
                       help='Output CSV file path')
    parser.add_argument('--organized', action='store_true',
                       help='Use organized MCI progression format (MCI->AD, etc.)')
    parser.add_argument('--groups', type=str, nargs='+', default=['AD', 'CN', 'MCI'],
                       help='Groups to include (default: AD CN MCI)')
    parser.add_argument('--no-validate', action='store_true',
                       help='Skip NIfTI file validation (faster but less safe)')

    args = parser.parse_args()

    try:
        validate = not args.no_validate

        if args.organized:
            df = prepare_organized_dataset(args.input, args.output, validate=validate)
        else:
            df = prepare_standard_dataset(args.input, args.output, args.groups, validate=validate)

        if df is None:
            return 1

        return 0

    except Exception as e:
        logger.error(f"Failed to prepare dataset: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())

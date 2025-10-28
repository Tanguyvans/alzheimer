#!/usr/bin/env python3
"""
Prepare CN vs MCI vs AD Dataset for 3-Class Classification

This script creates train/val/test splits for stable CN vs stable MCI vs stable AD classification.
It uses ONLY patients who remain CN, MCI, or AD throughout their visits (stable diagnosis).

Classification task:
- CN (Cognitively Normal): label = 0
- MCI (Mild Cognitive Impairment): label = 1
- AD (Alzheimer's Disease): label = 2

Usage:
    python 01_prepare_dataset.py --config config.yaml
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import Dict
import yaml
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datetime import datetime
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CNMCIADDatasetPreparator:
    """Prepare stable CN vs MCI vs AD dataset (3-class)"""

    def __init__(self, dxsum_csv: str, skull_dir: str, output_dir: str):
        self.dxsum_csv = dxsum_csv
        self.skull_dir = Path(skull_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.dx_df = None
        self.stable_patients_df = None
        self.dataset_df = None

    def load_diagnosis_data(self):
        """Load ADNI diagnosis data"""
        logger.info("="*80)
        logger.info("LOADING DIAGNOSIS DATA")
        logger.info("="*80)

        self.dx_df = pd.read_csv(self.dxsum_csv)
        logger.info(f"Loaded {len(self.dx_df):,} diagnosis records for {self.dx_df['RID'].nunique():,} patients")

        # Diagnosis codes: 1=CN, 2=MCI, 3=AD

    def identify_stable_patients(self):
        """Identify patients with stable CN, MCI, or AD diagnosis"""
        logger.info("\n" + "="*80)
        logger.info("IDENTIFYING STABLE PATIENTS")
        logger.info("="*80)

        # Group by patient
        patient_diagnoses = self.dx_df.groupby('RID')['DIAGNOSIS'].apply(lambda x: x.unique().tolist()).reset_index()
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

        # Create stable patients dataframe
        stable_cn_df = self.dx_df[self.dx_df['RID'].isin(stable_cn['RID'])].copy()
        stable_cn_df['group'] = 'CN'
        stable_cn_df['label'] = 0

        stable_mci_df = self.dx_df[self.dx_df['RID'].isin(stable_mci['RID'])].copy()
        stable_mci_df['group'] = 'MCI'
        stable_mci_df['label'] = 1

        stable_ad_df = self.dx_df[self.dx_df['RID'].isin(stable_ad['RID'])].copy()
        stable_ad_df['group'] = 'AD'
        stable_ad_df['label'] = 2

        self.stable_patients_df = pd.concat([stable_cn_df, stable_mci_df, stable_ad_df], ignore_index=True)

        logger.info(f"\nTotal stable patients: {self.stable_patients_df['RID'].nunique():,}")
        logger.info(f"  CN: {stable_cn_df['RID'].nunique():,}")
        logger.info(f"  MCI: {stable_mci_df['RID'].nunique():,}")
        logger.info(f"  AD: {stable_ad_df['RID'].nunique():,}")

    def find_baseline_mri_scans(self):
        """Map patients to their baseline MRI scans"""
        import nibabel as nib

        logger.info("\n" + "="*80)
        logger.info("MAPPING TO BASELINE MRI SCANS")
        logger.info("="*80)

        # Get baseline visit for each patient (first visit)
        baseline_visits = self.stable_patients_df.sort_values('EXAMDATE').groupby('RID').first().reset_index()

        scan_paths = []
        missing_patients = []
        corrupted_files = []

        for _, row in baseline_visits.iterrows():
            ptid = row['PTID']
            patient_folder = self.skull_dir / ptid

            if not patient_folder.exists():
                missing_patients.append(ptid)
                continue

            # Find all skull-stripped scans for this patient
            scans = list(patient_folder.glob('*_registered_skull_stripped.nii.gz'))

            if not scans:
                missing_patients.append(ptid)
                continue

            # Validate the scan file is not corrupted
            scan_path = scans[0]

            # Quick check: file must be larger than 1KB
            if scan_path.stat().st_size < 1024:
                corrupted_files.append(f"{ptid}: {scan_path.name} - File too small ({scan_path.stat().st_size} bytes)")
                continue

            # Optional: Full validation (slower but more thorough)
            # Uncomment to validate file can be loaded
            # try:
            #     img = nib.load(str(scan_path))
            #     data = img.get_fdata()
            #     if data.size == 0:
            #         raise ValueError("Empty scan data")
            # except Exception as e:
            #     corrupted_files.append(f"{ptid}: {scan_path.name} - {str(e)}")
            #     continue

            scan_paths.append({
                'PTID': ptid,
                'RID': row['RID'],
                'PHASE': row['PHASE'],
                'group': row['group'],
                'label': row['label'],
                'EXAMDATE': row['EXAMDATE'],
                'scan_path': str(scan_path)
            })

        self.dataset_df = pd.DataFrame(scan_paths)

        logger.info(f"\nMRI mapping results:")
        logger.info(f"  Successfully mapped: {len(self.dataset_df):,} patients")
        logger.info(f"  Missing/no scans: {len(missing_patients):,} patients")
        logger.info(f"  Corrupted files: {len(corrupted_files):,} files")

        if len(self.dataset_df) > 0:
            logger.info(f"\nClass distribution:")
            logger.info(f"  CN: {len(self.dataset_df[self.dataset_df['label'] == 0]):,} ({100*len(self.dataset_df[self.dataset_df['label'] == 0])/len(self.dataset_df):.1f}%)")
            logger.info(f"  MCI: {len(self.dataset_df[self.dataset_df['label'] == 1]):,} ({100*len(self.dataset_df[self.dataset_df['label'] == 1])/len(self.dataset_df):.1f}%)")
            logger.info(f"  AD: {len(self.dataset_df[self.dataset_df['label'] == 2]):,} ({100*len(self.dataset_df[self.dataset_df['label'] == 2])/len(self.dataset_df):.1f}%)")

        if len(missing_patients) > 0 and len(missing_patients) <= 10:
            logger.info(f"\nMissing patients: {missing_patients}")

        if len(corrupted_files) > 0:
            logger.warning(f"\nCorrupted/empty files detected:")
            for cf in corrupted_files[:10]:  # Show first 10
                logger.warning(f"  - {cf}")
            if len(corrupted_files) > 10:
                logger.warning(f"  ... and {len(corrupted_files) - 10} more")

        return self.dataset_df

    def create_train_val_test_split(self, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42):
        """Create stratified train/val/test split"""
        logger.info("\n" + "="*80)
        logger.info("CREATING TRAIN/VAL/TEST SPLIT")
        logger.info("="*80)
        logger.info(f"Split ratios: {train_ratio*100:.1f}% train / {val_ratio*100:.1f}% val / {test_ratio*100:.1f}% test")
        logger.info(f"Random seed: {random_seed}")

        # Stratified split
        train_val, test = train_test_split(
            self.dataset_df,
            test_size=test_ratio,
            stratify=self.dataset_df['label'],
            random_state=random_seed
        )

        val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
        train, val = train_test_split(
            train_val,
            test_size=val_ratio_adjusted,
            stratify=train_val['label'],
            random_state=random_seed
        )

        # Add split column
        train['split'] = 'train'
        val['split'] = 'val'
        test['split'] = 'test'

        # Print stats
        for split_name, split_df in [('TRAIN', train), ('VAL', val), ('TEST', test)]:
            cn_count = len(split_df[split_df['label'] == 0])
            mci_count = len(split_df[split_df['label'] == 1])
            ad_count = len(split_df[split_df['label'] == 2])
            logger.info(f"\n{split_name} SET:")
            logger.info(f"  Total: {len(split_df)} samples")
            logger.info(f"  CN: {cn_count} ({100*cn_count/len(split_df):.1f}%)")
            logger.info(f"  MCI: {mci_count} ({100*mci_count/len(split_df):.1f}%)")
            logger.info(f"  AD: {ad_count} ({100*ad_count/len(split_df):.1f}%)")

        return train, val, test

    def save_dataset_files(self, train, val, test):
        """Save train/val/test CSV files and metadata"""
        logger.info("\n" + "="*80)
        logger.info("SAVING DATASET FILES")
        logger.info("="*80)

        # Save splits
        train.to_csv(self.output_dir / 'train.csv', index=False)
        val.to_csv(self.output_dir / 'val.csv', index=False)
        test.to_csv(self.output_dir / 'test.csv', index=False)

        # Save complete dataset
        complete = pd.concat([train, val, test], ignore_index=True)
        complete.to_csv(self.output_dir / 'dataset_complete.csv', index=False)

        # Save metadata
        metadata = {
            'creation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'dataset_type': 'cn_mci_ad_stable',
            'description': '3-class classification (CN vs MCI vs AD) using stable patients only',
            'dxsum_path': str(self.dxsum_csv),
            'skull_dir': str(self.skull_dir),
            'total_samples': len(complete),
            'train_samples': len(train),
            'val_samples': len(val),
            'test_samples': len(test),
            'cn_count': len(complete[complete['label'] == 0]),
            'mci_count': len(complete[complete['label'] == 1]),
            'ad_count': len(complete[complete['label'] == 2])
        }

        with open(self.output_dir / 'dataset_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"✓ Saved train.csv: {self.output_dir / 'train.csv'}")
        logger.info(f"✓ Saved val.csv: {self.output_dir / 'val.csv'}")
        logger.info(f"✓ Saved test.csv: {self.output_dir / 'test.csv'}")
        logger.info(f"✓ Saved dataset_complete.csv: {self.output_dir / 'dataset_complete.csv'}")
        logger.info(f"✓ Saved dataset_metadata.json: {self.output_dir / 'dataset_metadata.json'}")

    def run(self, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42):
        """Run complete dataset preparation pipeline"""
        self.load_diagnosis_data()
        self.identify_stable_patients()
        self.find_baseline_mri_scans()

        if len(self.dataset_df) == 0:
            logger.error("No valid samples found! Cannot create dataset.")
            sys.exit(1)

        train, val, test = self.create_train_val_test_split(train_ratio, val_ratio, test_ratio, random_seed)
        self.save_dataset_files(train, val, test)

        logger.info("\n" + "="*80)
        logger.info("DATASET PREPARATION COMPLETE")
        logger.info("="*80)
        logger.info(f"\nDataset summary:")
        logger.info(f"  Total patients: {len(self.dataset_df)}")
        logger.info(f"  CN: {len(self.dataset_df[self.dataset_df['label'] == 0])} ({100*len(self.dataset_df[self.dataset_df['label'] == 0])/len(self.dataset_df):.1f}%)")
        logger.info(f"  MCI: {len(self.dataset_df[self.dataset_df['label'] == 1])} ({100*len(self.dataset_df[self.dataset_df['label'] == 1])/len(self.dataset_df):.1f}%)")
        logger.info(f"  AD: {len(self.dataset_df[self.dataset_df['label'] == 2])} ({100*len(self.dataset_df[self.dataset_df['label'] == 2])/len(self.dataset_df):.1f}%)")
        logger.info(f"\nFiles saved to: {self.output_dir}")


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description='Prepare CN vs MCI vs AD dataset (3-class)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to YAML config file')

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Get parameters from config
    dxsum_csv = config['data']['dxsum_csv']
    skull_dir = config['data']['skull_dir']
    splits_dir = config['data']['splits_dir']

    train_ratio = config['dataset_preparation']['train_ratio']
    val_ratio = config['dataset_preparation']['val_ratio']
    test_ratio = config['dataset_preparation']['test_ratio']
    random_seed = config['dataset_preparation']['random_seed']

    logger.info("Initializing CNMCIADDatasetPreparator...")
    logger.info(f"  dxsum.csv: {dxsum_csv}")
    logger.info(f"  skull_dir: {skull_dir}")
    logger.info(f"  output_dir: {splits_dir}\n")

    # Run preparation
    preparator = CNMCIADDatasetPreparator(dxsum_csv, skull_dir, splits_dir)
    preparator.run(train_ratio, val_ratio, test_ratio, random_seed)


if __name__ == '__main__':
    main()

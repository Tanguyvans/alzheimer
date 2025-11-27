#!/usr/bin/env python3
"""
Prepare 4-Class MRI Dataset for CN | MCI stable | MCI→AD | AD Classification

Uses DXSUM longitudinal data to identify patient trajectories:
- CN: Baseline CN, stayed CN
- MCI_stable: Baseline MCI, stayed MCI
- MCI_to_AD: Baseline MCI, converted to AD
- AD: Baseline AD

Usage:
    python 01_prepare_dataset.py --config config.yaml
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, Tuple
import yaml
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datetime import datetime
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CLASS_NAMES = ['CN', 'MCI_stable', 'MCI_to_AD', 'AD']


class FourClassDatasetPreparator:
    """Prepare 4-class dataset using longitudinal diagnosis trajectories"""

    def __init__(self, dxsum_csv: str, skull_dir: str, output_dir: str):
        self.dxsum_csv = dxsum_csv
        self.skull_dir = Path(skull_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.dx_df = None
        self.trajectories = None
        self.dataset_df = None

    def load_diagnosis_data(self):
        """Load ADNI diagnosis data"""
        logger.info("=" * 80)
        logger.info("LOADING DIAGNOSIS DATA")
        logger.info("=" * 80)

        self.dx_df = pd.read_csv(self.dxsum_csv)
        logger.info(f"Loaded {len(self.dx_df):,} diagnosis records")
        logger.info(f"Unique patients: {self.dx_df['PTID'].nunique():,}")

        # DIAGNOSIS codes: 1=CN, 2=MCI, 3=AD
        self.dx_df = self.dx_df[self.dx_df['DIAGNOSIS'].notna()].copy()
        self.dx_df['DIAGNOSIS'] = self.dx_df['DIAGNOSIS'].astype(int)

    def identify_trajectories(self):
        """Identify patient trajectories from baseline to last diagnosis"""
        logger.info("\n" + "=" * 80)
        logger.info("IDENTIFYING PATIENT TRAJECTORIES")
        logger.info("=" * 80)

        # Get baseline diagnosis (first visit)
        baseline = self.dx_df[self.dx_df['VISCODE'] == 'bl'][['PTID', 'DIAGNOSIS', 'EXAMDATE']].copy()
        baseline = baseline.rename(columns={'DIAGNOSIS': 'BL_DX', 'EXAMDATE': 'BL_DATE'})

        # Get last diagnosis (most recent)
        self.dx_df['EXAMDATE'] = pd.to_datetime(self.dx_df['EXAMDATE'])
        last = self.dx_df.sort_values('EXAMDATE').groupby('PTID').last()[['DIAGNOSIS', 'EXAMDATE']].reset_index()
        last = last.rename(columns={'DIAGNOSIS': 'LAST_DX', 'EXAMDATE': 'LAST_DATE'})

        # Merge
        self.trajectories = baseline.merge(last, on='PTID', how='inner')

        # Assign 4-class labels
        def assign_class(row):
            bl = row['BL_DX']
            last = row['LAST_DX']

            if bl == 1 and last == 1:
                return 0  # CN
            elif bl == 2 and last == 2:
                return 1  # MCI_stable
            elif bl == 2 and last == 3:
                return 2  # MCI_to_AD
            elif bl == 3:
                return 3  # AD
            else:
                return -1  # Exclude (CN→MCI, CN→AD, etc.)

        self.trajectories['label'] = self.trajectories.apply(assign_class, axis=1)

        # Filter valid classes
        valid = self.trajectories[self.trajectories['label'] >= 0].copy()

        logger.info(f"\nTrajectory distribution (all patients):")
        for label, name in enumerate(CLASS_NAMES):
            count = sum(valid['label'] == label)
            logger.info(f"  {name}: {count}")

        self.trajectories = valid

    def find_mri_scans(self):
        """Map patients to their MRI scans"""
        logger.info("\n" + "=" * 80)
        logger.info("MAPPING TO MRI SCANS")
        logger.info("=" * 80)

        scan_paths = []
        missing_patients = []

        for _, row in self.trajectories.iterrows():
            ptid = row['PTID']
            patient_folder = self.skull_dir / ptid

            if not patient_folder.exists():
                missing_patients.append(ptid)
                continue

            # Find skull-stripped scans
            scans = list(patient_folder.glob('*_registered_skull_stripped.nii.gz'))
            if not scans:
                scans = list(patient_folder.glob('*_mni_norm.nii.gz'))
            if not scans:
                scans = list(patient_folder.glob('*.nii.gz'))

            if not scans:
                missing_patients.append(ptid)
                continue

            # Use first scan (baseline)
            scan_path = scans[0]

            # Validate file size
            if scan_path.stat().st_size < 1024:
                logger.warning(f"Skipping small file: {ptid}")
                continue

            scan_paths.append({
                'PTID': ptid,
                'label': row['label'],
                'group': CLASS_NAMES[row['label']],
                'BL_DX': row['BL_DX'],
                'LAST_DX': row['LAST_DX'],
                'scan_path': str(scan_path)
            })

        self.dataset_df = pd.DataFrame(scan_paths)

        logger.info(f"\nMRI mapping results:")
        logger.info(f"  Successfully mapped: {len(self.dataset_df):,} patients")
        logger.info(f"  Missing/no scans: {len(missing_patients):,} patients")

        if len(self.dataset_df) > 0:
            logger.info(f"\nClass distribution (with MRI):")
            for label, name in enumerate(CLASS_NAMES):
                count = sum(self.dataset_df['label'] == label)
                pct = 100 * count / len(self.dataset_df)
                logger.info(f"  {name}: {count} ({pct:.1f}%)")

    def create_splits(self, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
        """Create stratified train/val/test splits"""
        logger.info("\n" + "=" * 80)
        logger.info("CREATING TRAIN/VAL/TEST SPLITS")
        logger.info("=" * 80)

        # Stratified split
        train_val, test = train_test_split(
            self.dataset_df,
            test_size=test_ratio,
            stratify=self.dataset_df['label'],
            random_state=seed
        )

        val_ratio_adj = val_ratio / (train_ratio + val_ratio)
        train, val = train_test_split(
            train_val,
            test_size=val_ratio_adj,
            stratify=train_val['label'],
            random_state=seed
        )

        # Log stats
        for name, df in [('TRAIN', train), ('VAL', val), ('TEST', test)]:
            logger.info(f"\n{name} SET: {len(df)} samples")
            for label, class_name in enumerate(CLASS_NAMES):
                count = sum(df['label'] == label)
                logger.info(f"  {class_name}: {count}")

        return train, val, test

    def save_dataset(self, train, val, test):
        """Save dataset files"""
        logger.info("\n" + "=" * 80)
        logger.info("SAVING DATASET FILES")
        logger.info("=" * 80)

        train.to_csv(self.output_dir / 'train.csv', index=False)
        val.to_csv(self.output_dir / 'val.csv', index=False)
        test.to_csv(self.output_dir / 'test.csv', index=False)

        # Complete dataset
        complete = pd.concat([train, val, test], ignore_index=True)
        complete.to_csv(self.output_dir / 'dataset_complete.csv', index=False)

        # Metadata
        metadata = {
            'creation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'dataset_type': '4-class (CN | MCI_stable | MCI_to_AD | AD)',
            'total_samples': len(complete),
            'train_samples': len(train),
            'val_samples': len(val),
            'test_samples': len(test),
            'class_counts': {
                name: int(sum(complete['label'] == i))
                for i, name in enumerate(CLASS_NAMES)
            }
        }

        with open(self.output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved to: {self.output_dir}")

    def run(self, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
        """Run complete pipeline"""
        self.load_diagnosis_data()
        self.identify_trajectories()
        self.find_mri_scans()

        if len(self.dataset_df) == 0:
            logger.error("No valid samples found!")
            return

        train, val, test = self.create_splits(train_ratio, val_ratio, test_ratio, seed)
        self.save_dataset(train, val, test)

        logger.info("\n" + "=" * 80)
        logger.info("DATASET PREPARATION COMPLETE")
        logger.info("=" * 80)


def load_config(config_path: str) -> Dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='Prepare 4-class MRI dataset')
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()

    config = load_config(args.config)

    dxsum_csv = config['data']['dxsum_csv']
    skull_dir = config['data']['skull_dir']
    splits_dir = config['data']['splits_dir']

    prep = config['dataset_preparation']

    logger.info(f"DXSUM: {dxsum_csv}")
    logger.info(f"MRI dir: {skull_dir}")
    logger.info(f"Output: {splits_dir}")

    preparator = FourClassDatasetPreparator(dxsum_csv, skull_dir, splits_dir)
    preparator.run(
        train_ratio=prep['train_ratio'],
        val_ratio=prep['val_ratio'],
        test_ratio=prep['test_ratio'],
        seed=prep['random_seed']
    )


if __name__ == '__main__':
    main()

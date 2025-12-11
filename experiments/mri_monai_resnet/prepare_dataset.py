#!/usr/bin/env python3
"""
Unified Dataset Preparation for MRI Classification

Supports multiple classification tasks:
- cn_ad: CN vs AD (binary, 2 classes)
- cn_mci_ad: CN vs MCI vs AD (3 classes)
- cn_mcis_mcic_ad: CN vs MCI-Stable vs MCI-Converting vs AD (4 classes)

Usage:
    # CN vs AD (binary)
    python prepare_dataset.py --task cn_ad --dxsum data/adni/dxsum.csv --skull-dir /path/to/skull --output data/cn_ad

    # CN vs MCI vs AD (3-class)
    python prepare_dataset.py --task cn_mci_ad --dxsum data/adni/dxsum.csv --skull-dir /path/to/skull --output data/cn_mci_ad

    # 4-class with MCI conversion
    python prepare_dataset.py --task cn_mcis_mcic_ad --dxsum data/adni/dxsum.csv --skull-dir /path/to/skull --output data/cn_mcis_mcic_ad
"""

import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datetime import datetime
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Classification task definitions
TASK_CONFIGS = {
    'cn_ad': {
        'name': 'CN vs AD',
        'num_classes': 2,
        'classes': ['CN', 'AD'],
        'description': 'Binary classification: Cognitively Normal vs Alzheimer\'s Disease'
    },
    'cn_mci_ad': {
        'name': 'CN vs MCI vs AD',
        'num_classes': 3,
        'classes': ['CN', 'MCI', 'AD'],
        'description': '3-class: CN vs Mild Cognitive Impairment vs AD'
    },
    'cn_mcis_mcic_ad': {
        'name': 'CN vs MCIs vs MCIc vs AD',
        'num_classes': 4,
        'classes': ['CN', 'MCIs', 'MCIc', 'AD'],
        'description': '4-class: CN vs Stable MCI vs Converting MCI vs AD'
    },
}


class DatasetPreparator:
    """Unified dataset preparation for different classification tasks"""

    def __init__(self, dxsum_csv: str, skull_dir: str, output_dir: str, task: str):
        self.dxsum_csv = dxsum_csv
        self.skull_dir = Path(skull_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.task = task
        self.task_config = TASK_CONFIGS[task]

        self.dx_df = None
        self.dataset_df = None

    def load_diagnosis_data(self):
        """Load ADNI diagnosis data"""
        logger.info("="*60)
        logger.info(f"LOADING DIAGNOSIS DATA FOR: {self.task_config['name']}")
        logger.info("="*60)

        self.dx_df = pd.read_csv(self.dxsum_csv)
        logger.info(f"Loaded {len(self.dx_df):,} records for {self.dx_df['RID'].nunique():,} patients")

        # Show diagnosis distribution
        dx_counts = self.dx_df['DIAGNOSIS'].value_counts().sort_index()
        logger.info(f"\nDiagnosis distribution (1=CN, 2=MCI, 3=AD):")
        for dx, count in dx_counts.items():
            logger.info(f"  {dx}: {count:,}")

    def identify_patients(self):
        """Identify patients based on the classification task"""
        logger.info("\n" + "="*60)
        logger.info("IDENTIFYING PATIENTS")
        logger.info("="*60)

        if self.task == 'cn_ad':
            return self._identify_cn_ad_patients()
        elif self.task == 'cn_mci_ad':
            return self._identify_cn_mci_ad_patients()
        elif self.task == 'cn_mcis_mcic_ad':
            return self._identify_4class_patients()

    def _identify_cn_ad_patients(self):
        """CN vs AD: stable CN and stable AD patients"""
        patient_diagnoses = self.dx_df.groupby('RID')['DIAGNOSIS'].apply(lambda x: set(x.tolist())).reset_index()

        # Stable CN: only diagnosis 1
        stable_cn = patient_diagnoses[patient_diagnoses['DIAGNOSIS'] == {1}]['RID'].tolist()

        # Stable AD: only diagnosis 3
        stable_ad = patient_diagnoses[patient_diagnoses['DIAGNOSIS'] == {3}]['RID'].tolist()

        logger.info(f"Stable CN patients: {len(stable_cn)}")
        logger.info(f"Stable AD patients: {len(stable_ad)}")

        # Create dataframe
        patients = []

        cn_df = self.dx_df[self.dx_df['RID'].isin(stable_cn)].drop_duplicates('RID')
        for _, row in cn_df.iterrows():
            patients.append({'RID': row['RID'], 'PTID': row['PTID'], 'group': 'CN', 'label': 0})

        ad_df = self.dx_df[self.dx_df['RID'].isin(stable_ad)].drop_duplicates('RID')
        for _, row in ad_df.iterrows():
            patients.append({'RID': row['RID'], 'PTID': row['PTID'], 'group': 'AD', 'label': 1})

        return pd.DataFrame(patients)

    def _identify_cn_mci_ad_patients(self):
        """CN vs MCI vs AD: stable patients in each category"""
        patient_diagnoses = self.dx_df.groupby('RID')['DIAGNOSIS'].apply(lambda x: set(x.tolist())).reset_index()

        stable_cn = patient_diagnoses[patient_diagnoses['DIAGNOSIS'] == {1}]['RID'].tolist()
        stable_mci = patient_diagnoses[patient_diagnoses['DIAGNOSIS'] == {2}]['RID'].tolist()
        stable_ad = patient_diagnoses[patient_diagnoses['DIAGNOSIS'] == {3}]['RID'].tolist()

        logger.info(f"Stable CN patients: {len(stable_cn)}")
        logger.info(f"Stable MCI patients: {len(stable_mci)}")
        logger.info(f"Stable AD patients: {len(stable_ad)}")

        patients = []

        for rid in stable_cn:
            row = self.dx_df[self.dx_df['RID'] == rid].iloc[0]
            patients.append({'RID': rid, 'PTID': row['PTID'], 'group': 'CN', 'label': 0})

        for rid in stable_mci:
            row = self.dx_df[self.dx_df['RID'] == rid].iloc[0]
            patients.append({'RID': rid, 'PTID': row['PTID'], 'group': 'MCI', 'label': 1})

        for rid in stable_ad:
            row = self.dx_df[self.dx_df['RID'] == rid].iloc[0]
            patients.append({'RID': rid, 'PTID': row['PTID'], 'group': 'AD', 'label': 2})

        return pd.DataFrame(patients)

    def _identify_4class_patients(self):
        """CN vs MCIs vs MCIc vs AD: includes MCI conversion tracking"""
        # Sort by date to track progression
        self.dx_df['EXAMDATE'] = pd.to_datetime(self.dx_df['EXAMDATE'])
        dx_sorted = self.dx_df.sort_values(['RID', 'EXAMDATE'])

        patient_diagnoses = dx_sorted.groupby('RID')['DIAGNOSIS'].apply(list).reset_index()
        patient_diagnoses.columns = ['RID', 'dx_sequence']

        # Get first PTID for each RID
        ptid_map = self.dx_df.drop_duplicates('RID').set_index('RID')['PTID'].to_dict()

        patients = []

        for _, row in patient_diagnoses.iterrows():
            rid = row['RID']
            ptid = ptid_map.get(rid)
            dx_seq = row['dx_sequence']
            unique_dx = set(dx_seq)

            # Stable CN: only 1
            if unique_dx == {1}:
                patients.append({'RID': rid, 'PTID': ptid, 'group': 'CN', 'label': 0})

            # Stable MCI: only 2 (never progressed)
            elif unique_dx == {2}:
                patients.append({'RID': rid, 'PTID': ptid, 'group': 'MCIs', 'label': 1})

            # Converting MCI: started as 2, progressed to 3
            elif 2 in unique_dx and 3 in unique_dx:
                # Check if started with MCI
                if dx_seq[0] == 2:
                    patients.append({'RID': rid, 'PTID': ptid, 'group': 'MCIc', 'label': 2})

            # Stable AD: only 3
            elif unique_dx == {3}:
                patients.append({'RID': rid, 'PTID': ptid, 'group': 'AD', 'label': 3})

        result = pd.DataFrame(patients)

        for group in ['CN', 'MCIs', 'MCIc', 'AD']:
            count = len(result[result['group'] == group])
            logger.info(f"{group} patients: {count}")

        return result

    def find_mri_scans(self, patients_df):
        """Map patients to their MRI scan files"""
        logger.info("\n" + "="*60)
        logger.info("MAPPING TO MRI SCANS")
        logger.info("="*60)

        scan_data = []
        missing = []

        for _, row in patients_df.iterrows():
            ptid = row['PTID']
            patient_folder = self.skull_dir / ptid

            if not patient_folder.exists():
                missing.append(ptid)
                continue

            # Find skull-stripped or preprocessed scans
            scans = list(patient_folder.glob('*_registered_skull_stripped.nii.gz'))
            if not scans:
                scans = list(patient_folder.glob('*_mni_norm.nii.gz'))
            if not scans:
                scans = list(patient_folder.glob('*.nii.gz'))

            if not scans:
                missing.append(ptid)
                continue

            # Validate scan size
            scan_path = scans[0]
            if scan_path.stat().st_size < 1024:
                logger.warning(f"Small file: {scan_path}")
                continue

            scan_data.append({
                'PTID': ptid,
                'RID': row['RID'],
                'group': row['group'],
                'label': row['label'],
                'scan_path': str(scan_path)
            })

        self.dataset_df = pd.DataFrame(scan_data)

        logger.info(f"\nMapped: {len(self.dataset_df)} patients")
        logger.info(f"Missing: {len(missing)} patients")

        # Class distribution
        logger.info("\nClass distribution:")
        for group in self.task_config['classes']:
            count = len(self.dataset_df[self.dataset_df['group'] == group])
            pct = 100 * count / len(self.dataset_df) if len(self.dataset_df) > 0 else 0
            label = self.dataset_df[self.dataset_df['group'] == group]['label'].iloc[0] if count > 0 else '?'
            logger.info(f"  {group} (label={label}): {count} ({pct:.1f}%)")

        return self.dataset_df

    def create_splits(self, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
        """Create stratified train/val/test splits"""
        logger.info("\n" + "="*60)
        logger.info("CREATING TRAIN/VAL/TEST SPLITS")
        logger.info("="*60)
        logger.info(f"Ratios: {train_ratio:.0%} / {val_ratio:.0%} / {test_ratio:.0%}")

        # Stratified split
        train_val, test = train_test_split(
            self.dataset_df,
            test_size=test_ratio,
            stratify=self.dataset_df['label'],
            random_state=seed
        )

        val_adjusted = val_ratio / (train_ratio + val_ratio)
        train, val = train_test_split(
            train_val,
            test_size=val_adjusted,
            stratify=train_val['label'],
            random_state=seed
        )

        # Add split column
        train = train.copy()
        val = val.copy()
        test = test.copy()
        train['split'] = 'train'
        val['split'] = 'val'
        test['split'] = 'test'

        # Print stats
        for name, df in [('TRAIN', train), ('VAL', val), ('TEST', test)]:
            logger.info(f"\n{name}: {len(df)} samples")
            for group in self.task_config['classes']:
                count = len(df[df['group'] == group])
                logger.info(f"  {group}: {count} ({100*count/len(df):.1f}%)")

        return train, val, test

    def save_dataset(self, train, val, test):
        """Save dataset files"""
        logger.info("\n" + "="*60)
        logger.info("SAVING DATASET")
        logger.info("="*60)

        # Save CSVs
        train.to_csv(self.output_dir / 'train.csv', index=False)
        val.to_csv(self.output_dir / 'val.csv', index=False)
        test.to_csv(self.output_dir / 'test.csv', index=False)

        complete = pd.concat([train, val, test], ignore_index=True)
        complete.to_csv(self.output_dir / 'dataset_complete.csv', index=False)

        # Save metadata
        metadata = {
            'task': self.task,
            'task_name': self.task_config['name'],
            'num_classes': self.task_config['num_classes'],
            'classes': self.task_config['classes'],
            'description': self.task_config['description'],
            'created': datetime.now().isoformat(),
            'total_samples': len(complete),
            'train_samples': len(train),
            'val_samples': len(val),
            'test_samples': len(test),
            'class_distribution': {
                group: len(complete[complete['group'] == group])
                for group in self.task_config['classes']
            }
        }

        with open(self.output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved to: {self.output_dir}")
        logger.info(f"  train.csv: {len(train)} samples")
        logger.info(f"  val.csv: {len(val)} samples")
        logger.info(f"  test.csv: {len(test)} samples")

    def run(self, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
        """Run full preparation pipeline"""
        self.load_diagnosis_data()
        patients_df = self.identify_patients()
        self.find_mri_scans(patients_df)

        if len(self.dataset_df) == 0:
            logger.error("No valid samples found!")
            return

        train, val, test = self.create_splits(train_ratio, val_ratio, test_ratio, seed)
        self.save_dataset(train, val, test)

        logger.info("\n" + "="*60)
        logger.info("PREPARATION COMPLETE")
        logger.info("="*60)


def main():
    parser = argparse.ArgumentParser(description='Prepare MRI dataset for classification')

    parser.add_argument('--task', type=str, required=True,
                       choices=['cn_ad', 'cn_mci_ad', 'cn_mcis_mcic_ad'],
                       help='Classification task')
    parser.add_argument('--dxsum', type=str, required=True,
                       help='Path to dxsum.csv')
    parser.add_argument('--skull-dir', type=str, required=True,
                       help='Path to skull-stripped MRI directory')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory for dataset files')
    parser.add_argument('--train-ratio', type=float, default=0.7)
    parser.add_argument('--val-ratio', type=float, default=0.15)
    parser.add_argument('--test-ratio', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    logger.info(f"Task: {args.task}")
    logger.info(f"dxsum: {args.dxsum}")
    logger.info(f"skull_dir: {args.skull_dir}")
    logger.info(f"output: {args.output}")

    preparator = DatasetPreparator(args.dxsum, args.skull_dir, args.output, args.task)
    preparator.run(args.train_ratio, args.val_ratio, args.test_ratio, args.seed)


if __name__ == '__main__':
    main()

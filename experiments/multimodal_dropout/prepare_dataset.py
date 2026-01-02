#!/usr/bin/env python3
"""
Dataset Preparation for Multimodal Dropout Training

Creates unified dataset with:
- All MRI scans (from ADNI, OASIS, NACC)
- Tabular data where available (marked with has_tabular flag)

Usage:
    python prepare_dataset.py --config configs/config_nacc.yaml
"""

import argparse
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import numpy as np
import yaml
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Default directories
DEFAULT_ADNI_MRI_DIR = Path("/home/tanguy/medical/ADNI-skull")
DEFAULT_OASIS_MRI_DIR = Path("/home/tanguy/medical/OASIS-skull")
DEFAULT_NACC_MRI_DIR = Path("/home/tanguy/medical/NACC-skull")

# NACC tabular features (mapped to common names)
NACC_TABULAR_FEATURES = [
    'AGE', 'PTGENDER', 'PTEDUCAT', 'PTMARRY',
    'VSWEIGHT', 'BMI',
    'MH14ALCH', 'MH16SMOK', 'MH4CARD', 'MHPSYCH', 'MH2NEURL',
    'CATANIMSC', 'TRAASCOR', 'TRABSCOR', 'TRABERRCOM',
    'CLOCKSCOR', 'BNTTOTAL', 'DSPANFOR', 'DSPANBAC'
]


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        raw_config = yaml.safe_load(f)

    # Handle nested config structure
    if 'experiment' in raw_config:
        exp = raw_config.get('experiment', {})
        data = raw_config.get('data', {})
        training = raw_config.get('training', {})

        config = {
            'task': exp.get('task', 'cn_ad_trajectory'),
            'dataset': exp.get('dataset', 'nacc'),
            'output_dir': data.get('output_dir', 'data/multimodal_dropout'),
            'seed': training.get('seed', 42),
            'train_ratio': data.get('train_ratio', 0.7),
            'val_ratio': data.get('val_ratio', 0.15),
            'test_ratio': data.get('test_ratio', 0.15),
            'nacc_mri_dir': data.get('nacc_mri_dir', str(DEFAULT_NACC_MRI_DIR)),
            'adni_mri_dir': data.get('adni_mri_dir', str(DEFAULT_ADNI_MRI_DIR)),
            'oasis_mri_dir': data.get('oasis_mri_dir', str(DEFAULT_OASIS_MRI_DIR)),
            'tabular_features': data.get('tabular_features', NACC_TABULAR_FEATURES),
        }
    else:
        config = raw_config
        config.setdefault('seed', 42)
        config.setdefault('train_ratio', 0.7)
        config.setdefault('val_ratio', 0.15)
        config.setdefault('test_ratio', 0.15)
        config.setdefault('tabular_features', NACC_TABULAR_FEATURES)

    return config


class MultimodalDatasetPreparator:
    """Prepare unified multimodal dataset with MRI + optional tabular data."""

    def __init__(self, config: dict):
        self.config = config
        self.task = config['task']
        self.dataset = config['dataset']
        self.output_dir = PROJECT_ROOT / config['output_dir']
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tabular_features = config['tabular_features']

    def scan_mri_folder(self, mri_dir: Path, dataset_name: str) -> pd.DataFrame:
        """Scan MRI folder and return dataframe of available scans."""
        logger.info(f"Scanning MRI directory: {mri_dir}")

        if not mri_dir.exists():
            logger.warning(f"{dataset_name} MRI directory not found: {mri_dir}")
            return pd.DataFrame()

        # Check what's in the directory
        subdirs = list(mri_dir.iterdir())[:5]
        logger.info(f"  First 5 items in dir: {[s.name for s in subdirs]}")

        scans = []
        # Search recursively for .nii.gz files
        for scan_file in mri_dir.glob("**/*.nii.gz"):
            # Extract subject ID from parent folder name (e.g., NACC531822)
            subject_id = scan_file.parent.name
            scans.append({
                'subject_id': subject_id,
                'scan_path': str(scan_file),
                'dataset': dataset_name
            })

        logger.info(f"{dataset_name}: Found {len(scans)} MRI scans")

        # Keep only first scan per subject (avoid duplicates from multiple visits)
        if len(scans) > 0:
            df = pd.DataFrame(scans)
            df = df.drop_duplicates(subset='subject_id', keep='first')
            logger.info(f"{dataset_name}: {len(df)} unique subjects after deduplication")
            return df

        return pd.DataFrame(scans)

    def load_nacc_diagnosis(self) -> pd.DataFrame:
        """Load NACC diagnosis data from raw UDS file."""
        nacc_uds_path = DATA_DIR / "nacc" / "investigator_ftldlbd_nacc71.csv"
        if not nacc_uds_path.exists():
            raise FileNotFoundError(f"NACC UDS file not found: {nacc_uds_path}")

        df = pd.read_csv(nacc_uds_path, low_memory=False)

        # Map diagnosis codes (NACCUDSD: 1=Normal, 2=Impaired, 3=MCI, 4=Dementia)
        diag_map = {1: 'CN', 2: 'Impaired', 3: 'MCI', 4: 'AD'}
        df['DX_raw'] = df['NACCUDSD'].map(diag_map)

        # For trajectory task: identify patients who ever reached AD
        if self.task == 'cn_ad_trajectory':
            patients_with_ad = set(df[df['DX_raw'] == 'AD']['NACCID'].unique())

            def assign_trajectory(row):
                if row['DX_raw'] == 'AD':
                    return 'AD_trajectory'
                elif row['DX_raw'] == 'MCI' and row['NACCID'] in patients_with_ad:
                    return 'AD_trajectory'  # MCI converter
                elif row['DX_raw'] == 'CN':
                    return 'CN'
                return None

            df['DX'] = df.apply(assign_trajectory, axis=1)
        else:
            df['DX'] = df['DX_raw']

        # Get first visit per patient
        df = df.sort_values('NACCVNUM')
        first_visits = df.groupby('NACCID').first().reset_index()
        first_visits = first_visits[first_visits['DX'].notna()]
        first_visits = first_visits.rename(columns={'NACCID': 'subject_id'})

        return first_visits[['subject_id', 'DX']]

    def load_nacc_tabular(self) -> pd.DataFrame:
        """Load NACC tabular data."""
        tabular_path = DATA_DIR / "nacc" / "nacc_tabular_mri.csv"
        if not tabular_path.exists():
            logger.warning(f"NACC tabular file not found: {tabular_path}")
            return pd.DataFrame()

        df = pd.read_csv(tabular_path)
        logger.info(f"  Tabular file columns: {list(df.columns[:10])}...")

        # Try to find subject ID column (could be NACCID, Subject, subject_id, or similar)
        subject_col = None
        for col_name in ['NACCID', 'Subject', 'subject_id', 'Subject_ID', 'ID', 'SubjectID']:
            if col_name in df.columns:
                subject_col = col_name
                break

        if subject_col is None:
            logger.warning(f"No subject ID column found in tabular file. Columns: {list(df.columns)}")
            return pd.DataFrame()

        # Rename to subject_id
        if subject_col != 'subject_id':
            df = df.rename(columns={subject_col: 'subject_id'})
            logger.info(f"  Renamed '{subject_col}' to 'subject_id'")

        # Keep only relevant columns
        available_features = [c for c in self.tabular_features if c in df.columns]
        keep_cols = ['subject_id'] + available_features
        df = df[keep_cols]

        logger.info(f"Loaded tabular data for {len(df)} NACC subjects")
        logger.info(f"  Features available: {len(available_features)}/{len(self.tabular_features)}")
        if len(available_features) < len(self.tabular_features):
            missing = set(self.tabular_features) - set(available_features)
            logger.info(f"  Missing features: {missing}")

        return df

    def build_dataset(self) -> pd.DataFrame:
        """Build unified dataset with MRI and optional tabular data."""
        logger.info("=" * 60)
        logger.info(f"BUILDING MULTIMODAL DATASET")
        logger.info(f"Task: {self.task}")
        logger.info(f"Dataset: {self.dataset}")
        logger.info("=" * 60)

        all_samples = []

        # Process NACC
        if self.dataset in ['nacc', 'combined']:
            nacc_mri_dir = Path(self.config.get('nacc_mri_dir', DEFAULT_NACC_MRI_DIR))
            nacc_scans = self.scan_mri_folder(nacc_mri_dir, 'NACC')

            if len(nacc_scans) > 0:
                # Load diagnosis
                nacc_dx = self.load_nacc_diagnosis()
                nacc_merged = nacc_scans.merge(nacc_dx, on='subject_id', how='inner')

                # Load tabular data
                nacc_tabular = self.load_nacc_tabular()

                if len(nacc_tabular) > 0:
                    # Merge with tabular (left join - keep all MRI samples)
                    nacc_merged = nacc_merged.merge(nacc_tabular, on='subject_id', how='left')
                    # Mark which samples have tabular data
                    nacc_merged['has_tabular'] = nacc_merged['subject_id'].isin(nacc_tabular['subject_id'])
                else:
                    nacc_merged['has_tabular'] = False
                    for feat in self.tabular_features:
                        nacc_merged[feat] = np.nan

                # Filter by task labels
                label_map = {'CN': 0, 'AD_trajectory': 1} if self.task == 'cn_ad_trajectory' else {'CN': 0, 'AD': 1}
                nacc_merged = nacc_merged[nacc_merged['DX'].isin(label_map.keys())]
                nacc_merged['label'] = nacc_merged['DX'].map(label_map)

                all_samples.append(nacc_merged)

                with_tab = nacc_merged['has_tabular'].sum()
                logger.info(f"NACC: {len(nacc_merged)} samples ({with_tab} with tabular)")

        # Combine all samples
        if len(all_samples) == 0:
            raise ValueError("No samples found!")

        df = pd.concat(all_samples, ignore_index=True)

        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("DATASET SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total samples: {len(df)}")
        logger.info(f"  With tabular: {df['has_tabular'].sum()} ({100*df['has_tabular'].mean():.1f}%)")
        logger.info(f"  MRI only: {(~df['has_tabular']).sum()} ({100*(~df['has_tabular']).mean():.1f}%)")

        for label, count in df['label'].value_counts().sort_index().items():
            label_name = 'CN' if label == 0 else 'AD_trajectory'
            logger.info(f"  {label_name} (label={label}): {count} ({100*count/len(df):.1f}%)")

        return df

    def create_splits(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Create train/val/test splits with stratification."""
        train_ratio = self.config['train_ratio']
        val_ratio = self.config['val_ratio']
        test_ratio = self.config['test_ratio']
        seed = self.config['seed']

        # Stratify by label and has_tabular
        df['stratify_key'] = df['label'].astype(str) + '_' + df['has_tabular'].astype(str)

        # Split: train vs (val + test)
        train_df, temp_df = train_test_split(
            df,
            test_size=(val_ratio + test_ratio),
            random_state=seed,
            stratify=df['stratify_key']
        )

        # Split: val vs test
        relative_test = test_ratio / (val_ratio + test_ratio)
        val_df, test_df = train_test_split(
            temp_df,
            test_size=relative_test,
            random_state=seed,
            stratify=temp_df['stratify_key']
        )

        # Drop stratify key
        for split_df in [train_df, val_df, test_df]:
            split_df.drop('stratify_key', axis=1, inplace=True)

        logger.info("\n" + "=" * 60)
        logger.info("SPLITS")
        logger.info("=" * 60)
        for name, split_df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
            with_tab = split_df['has_tabular'].sum()
            logger.info(f"{name}: {len(split_df)} samples ({with_tab} with tabular)")

        return {'train': train_df, 'val': val_df, 'test': test_df}

    def save_dataset(self, splits: Dict[str, pd.DataFrame]):
        """Save splits to CSV files."""
        # Columns to save
        base_cols = ['subject_id', 'scan_path', 'dataset', 'DX', 'label', 'has_tabular']
        all_cols = base_cols + self.tabular_features

        for split_name, df in splits.items():
            # Ensure all columns exist
            for col in all_cols:
                if col not in df.columns:
                    df[col] = np.nan

            output_path = self.output_dir / f"{split_name}.csv"
            df[all_cols].to_csv(output_path, index=False)
            logger.info(f"Saved {split_name}.csv: {len(df)} samples")

        # Save all combined
        all_df = pd.concat(splits.values(), ignore_index=True)
        all_df[all_cols].to_csv(self.output_dir / "all.csv", index=False)

        # Save metadata
        metadata = {
            'task': self.task,
            'dataset': self.dataset,
            'tabular_features': self.tabular_features,
            'num_features': len(self.tabular_features),
            'total_samples': len(all_df),
            'samples_with_tabular': int(all_df['has_tabular'].sum()),
            'samples_mri_only': int((~all_df['has_tabular']).sum()),
            'train_samples': len(splits['train']),
            'val_samples': len(splits['val']),
            'test_samples': len(splits['test']),
            'created': datetime.now().isoformat()
        }

        with open(self.output_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        # Save config
        with open(self.output_dir / "config.yaml", 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)

        logger.info(f"\nSaved to: {self.output_dir}")

    def run(self):
        """Run full preparation pipeline."""
        df = self.build_dataset()
        splits = self.create_splits(df)
        self.save_dataset(splits)
        logger.info("\n" + "=" * 60)
        logger.info("PREPARATION COMPLETE")
        logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Prepare multimodal dropout dataset')
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    args = parser.parse_args()

    config = load_config(args.config)
    logger.info(f"Loaded config: {args.config}")

    preparator = MultimodalDatasetPreparator(config)
    preparator.run()


if __name__ == '__main__':
    main()

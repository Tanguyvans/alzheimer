#!/usr/bin/env python3
"""
Unified Dataset Preparation for MRI Classification

Supports:
- Multiple datasets: ADNI, OASIS, or combined
- Multiple tasks: cn_ad, cn_mci_ad, cn_mcis_mcic_ad
- YAML config files (like xgboost_tabular)

Usage:
    # With YAML config (recommended)
    python prepare_dataset.py --config configs/cn_ad_adni.yaml

    # With command line args
    python prepare_dataset.py --task cn_ad --dataset adni --output data/cn_ad_adni
"""

import argparse
import logging
import re
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import yaml
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Default MRI directories
DEFAULT_ADNI_MRI_DIR = Path("/Volumes/KINGSTON/ADNI-skull")
DEFAULT_OASIS_MRI_DIR = Path("/Volumes/KINGSTON/OASIS-registered")

# Classification task definitions
TASK_CONFIGS = {
    'cn_ad': {
        'name': 'CN vs AD',
        'num_classes': 2,
        'classes': ['CN', 'AD'],
        'label_map': {'CN': 0, 'AD': 1},
        'description': 'Binary classification: Cognitively Normal vs Alzheimer\'s Disease'
    },
    'cn_ad_trajectory': {
        'name': 'CN vs AD Trajectory',
        'num_classes': 2,
        'classes': ['CN', 'AD_trajectory'],
        'label_map': {'CN': 0, 'AD_trajectory': 1},
        'description': 'Binary classification: CN vs AD trajectory (AD + MCI converters)'
    },
    'cn_mci_ad': {
        'name': 'CN vs MCI vs AD',
        'num_classes': 3,
        'classes': ['CN', 'MCI', 'AD'],
        'label_map': {'CN': 0, 'MCI': 1, 'AD': 2},
        'description': '3-class: CN vs Mild Cognitive Impairment vs AD'
    },
    'cn_mcis_mcic_ad': {
        'name': 'CN vs MCIs vs MCIc vs AD',
        'num_classes': 4,
        'classes': ['CN', 'MCIs', 'MCIc', 'AD'],
        'label_map': {'CN': 0, 'MCIs': 1, 'MCIc': 2, 'AD': 3},
        'description': '4-class: CN vs Stable MCI vs Converting MCI vs AD'
    },
}


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Set defaults
    config.setdefault('seed', 42)
    config.setdefault('train_ratio', 0.7)
    config.setdefault('val_ratio', 0.15)
    config.setdefault('test_ratio', 0.15)
    config.setdefault('adni_mri_dir', str(DEFAULT_ADNI_MRI_DIR))
    config.setdefault('oasis_mri_dir', str(DEFAULT_OASIS_MRI_DIR))

    return config


class DatasetPreparator:
    """Unified dataset preparation for MRI classification"""

    def __init__(self, config: dict):
        self.config = config
        self.task = config['task']
        self.dataset = config['dataset']
        self.output_dir = PROJECT_ROOT / config['output_dir']
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.task_config = TASK_CONFIGS[self.task]
        self.dataset_df = None

    def load_adni_diagnosis(self) -> pd.DataFrame:
        """Load ADNI diagnosis data (first visit per patient).

        For 'cn_ad_trajectory' task, also identifies MCI patients who converted to AD.
        """
        dxsum_path = DATA_DIR / "adni" / "dxsum.csv"
        if not dxsum_path.exists():
            raise FileNotFoundError(f"ADNI dxsum.csv not found at {dxsum_path}")

        df = pd.read_csv(dxsum_path)
        df['EXAMDATE'] = pd.to_datetime(df['EXAMDATE'])

        # Map diagnosis codes
        diag_map = {1: 'CN', 2: 'MCI', 3: 'AD'}
        df['DX_raw'] = df['DIAGNOSIS'].map(diag_map)

        # For trajectory task: identify MCI converters (patients who ever reached AD)
        if self.task == 'cn_ad_trajectory':
            # Find patients who ever had AD diagnosis
            patients_with_ad = set(df[df['DX_raw'] == 'AD']['PTID'].unique())

            # Sort and take first visit
            df_first = df.sort_values(['PTID', 'EXAMDATE']).groupby('PTID').first().reset_index()
            df_first['DX'] = df_first['DX_raw']

            # Remap: AD and MCI converters -> AD_trajectory, CN stays CN
            def assign_trajectory(row):
                if row['PTID'] in patients_with_ad:
                    return 'AD_trajectory'
                elif row['DX'] == 'CN':
                    return 'CN'
                else:
                    return None  # MCI non-converters excluded

            df_first['DX'] = df_first.apply(assign_trajectory, axis=1)
            df_first['subject_id'] = df_first['PTID']

            result = df_first[['subject_id', 'DX', 'RID']].dropna(subset=['DX'])
            n_ad_traj = (result['DX'] == 'AD_trajectory').sum()
            logger.info(f"ADNI: {len(result)} patients for trajectory task ({n_ad_traj} AD trajectory)")
            return result
        else:
            # Standard: just take first visit
            df = df.sort_values(['PTID', 'EXAMDATE']).groupby('PTID').first().reset_index()
            df['DX'] = df['DX_raw']
            df['subject_id'] = df['PTID']

            logger.info(f"ADNI: {len(df)} patients with first-visit diagnosis")
            return df[['subject_id', 'DX', 'RID']].dropna(subset=['DX'])

    def load_oasis_diagnosis(self) -> pd.DataFrame:
        """Load OASIS diagnosis data (first visit per patient).

        For 'cn_ad_trajectory' task, also identifies MCI patients who converted to AD.
        """
        oasis_path = DATA_DIR / "oasis" / "oasis_all_full.csv"
        if not oasis_path.exists():
            raise FileNotFoundError(f"OASIS data not found at {oasis_path}")

        df = pd.read_csv(oasis_path)

        # For trajectory task: identify patients who ever reached AD
        if self.task == 'cn_ad_trajectory':
            # Find patients who ever had AD diagnosis
            patients_with_ad = set(df[df['DX'] == 'AD']['Subject'].unique())

            # Sort and take first visit
            if 'days_to_visit' in df.columns:
                df_first = df.sort_values(['Subject', 'days_to_visit']).groupby('Subject').first().reset_index()
            else:
                df_first = df.groupby('Subject').first().reset_index()

            # Remap: AD and MCI converters -> AD_trajectory, CN stays CN
            def assign_trajectory(row):
                if row['Subject'] in patients_with_ad:
                    return 'AD_trajectory'
                elif row['DX'] == 'CN':
                    return 'CN'
                else:
                    return None  # MCI non-converters excluded

            df_first['DX'] = df_first.apply(assign_trajectory, axis=1)
            df_first['subject_id'] = df_first['Subject']

            result = df_first[['subject_id', 'DX']].dropna(subset=['DX'])
            n_ad_traj = (result['DX'] == 'AD_trajectory').sum()
            logger.info(f"OASIS: {len(result)} patients for trajectory task ({n_ad_traj} AD trajectory)")
            return result
        else:
            # Standard: just take first visit
            if 'days_to_visit' in df.columns:
                df = df.sort_values(['Subject', 'days_to_visit']).groupby('Subject').first().reset_index()
            else:
                df = df.groupby('Subject').first().reset_index()

            df['subject_id'] = df['Subject']

            logger.info(f"OASIS: {len(df)} patients with first-visit diagnosis")
            return df[['subject_id', 'DX']].dropna(subset=['DX'])

    def scan_mri_folder(self, mri_dir: Path, source: str) -> pd.DataFrame:
        """Scan MRI folder and return DataFrame with scan paths."""
        if not mri_dir.exists():
            logger.warning(f"{source} MRI directory not found: {mri_dir}")
            return pd.DataFrame()

        scans = []

        for subject_dir in mri_dir.iterdir():
            if not subject_dir.is_dir() or subject_dir.name.startswith('.'):
                continue

            subject_id = subject_dir.name

            # Find first valid scan
            scan_files = sorted(subject_dir.glob('*.nii.gz'))
            for scan_file in scan_files:
                if scan_file.name.startswith('._'):
                    continue
                if scan_file.stat().st_size < 1024:
                    continue

                # Take first valid scan only
                scans.append({
                    'subject_id': subject_id,
                    'scan_path': str(scan_file),
                })
                break

        logger.info(f"{source}: Found {len(scans)} patients with MRI scans")
        return pd.DataFrame(scans)

    def build_dataset(self) -> pd.DataFrame:
        """Build dataset by matching MRI scans with diagnosis labels."""
        logger.info("=" * 60)
        logger.info(f"BUILDING DATASET: {self.task_config['name']}")
        logger.info(f"Dataset: {self.dataset}")
        logger.info("=" * 60)

        valid_classes = set(self.task_config['classes'])
        label_map = self.task_config['label_map']
        all_data = []

        # Process ADNI
        if self.dataset in ['adni', 'combined']:
            adni_mri_dir = Path(self.config.get('adni_mri_dir', DEFAULT_ADNI_MRI_DIR))
            adni_scans = self.scan_mri_folder(adni_mri_dir, 'ADNI')

            if len(adni_scans) > 0:
                adni_dx = self.load_adni_diagnosis()
                adni_merged = adni_scans.merge(adni_dx, on='subject_id', how='inner')
                adni_merged = adni_merged[adni_merged['DX'].isin(valid_classes)]
                adni_merged['source'] = 'ADNI'
                all_data.append(adni_merged)
                logger.info(f"ADNI: {len(adni_merged)} samples after filtering")

        # Process OASIS
        if self.dataset in ['oasis', 'combined']:
            oasis_mri_dir = Path(self.config.get('oasis_mri_dir', DEFAULT_OASIS_MRI_DIR))
            oasis_scans = self.scan_mri_folder(oasis_mri_dir, 'OASIS')

            if len(oasis_scans) > 0:
                oasis_dx = self.load_oasis_diagnosis()
                oasis_merged = oasis_scans.merge(oasis_dx, on='subject_id', how='inner')
                oasis_merged = oasis_merged[oasis_merged['DX'].isin(valid_classes)]
                oasis_merged['source'] = 'OASIS'
                all_data.append(oasis_merged)
                logger.info(f"OASIS: {len(oasis_merged)} samples after filtering")

        if not all_data:
            raise ValueError("No data found. Check MRI directories.")

        # Combine
        df = pd.concat(all_data, ignore_index=True)
        df['group'] = df['DX']
        df['label'] = df['DX'].map(label_map)

        # Log statistics
        logger.info(f"\nDataset Summary:")
        logger.info(f"  Total: {len(df)} samples")
        for source in df['source'].unique():
            count = (df['source'] == source).sum()
            logger.info(f"  {source}: {count}")
        for cls in self.task_config['classes']:
            count = (df['DX'] == cls).sum()
            pct = 100 * count / len(df) if len(df) > 0 else 0
            logger.info(f"  {cls} (label={label_map[cls]}): {count} ({pct:.1f}%)")

        self.dataset_df = df
        return df

    def create_splits(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Create stratified train/val/test splits."""
        logger.info("\n" + "=" * 60)
        logger.info("CREATING SPLITS")
        logger.info("=" * 60)

        train_ratio = self.config['train_ratio']
        val_ratio = self.config['val_ratio']
        test_ratio = self.config['test_ratio']
        seed = self.config['seed']

        logger.info(f"Ratios: {train_ratio:.0%} / {val_ratio:.0%} / {test_ratio:.0%}")

        # Stratified split
        train_df, temp_df = train_test_split(
            self.dataset_df,
            test_size=(val_ratio + test_ratio),
            random_state=seed,
            stratify=self.dataset_df['label']
        )

        relative_test_ratio = test_ratio / (val_ratio + test_ratio)
        val_df, test_df = train_test_split(
            temp_df,
            test_size=relative_test_ratio,
            random_state=seed,
            stratify=temp_df['label']
        )

        # Log stats
        for name, df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
            logger.info(f"\n{name}: {len(df)} samples")
            for cls in self.task_config['classes']:
                count = (df['group'] == cls).sum()
                logger.info(f"  {cls}: {count}")

        return train_df, val_df, test_df

    def save_dataset(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
        """Save dataset files."""
        logger.info("\n" + "=" * 60)
        logger.info("SAVING DATASET")
        logger.info("=" * 60)

        # Save CSVs
        train_df.to_csv(self.output_dir / 'train.csv', index=False)
        val_df.to_csv(self.output_dir / 'val.csv', index=False)
        test_df.to_csv(self.output_dir / 'test.csv', index=False)

        complete = pd.concat([train_df, val_df, test_df], ignore_index=True)
        complete.to_csv(self.output_dir / 'all.csv', index=False)

        # Save metadata
        metadata = {
            'task': self.task,
            'task_name': self.task_config['name'],
            'dataset': self.dataset,
            'num_classes': self.task_config['num_classes'],
            'classes': self.task_config['classes'],
            'description': self.task_config['description'],
            'created': datetime.now().isoformat(),
            'seed': self.config['seed'],
            'total_samples': len(complete),
            'train_samples': len(train_df),
            'val_samples': len(val_df),
            'test_samples': len(test_df),
            'class_distribution': {
                cls: int((complete['group'] == cls).sum())
                for cls in self.task_config['classes']
            },
            'source_distribution': {
                src: int((complete['source'] == src).sum())
                for src in complete['source'].unique()
            }
        }

        with open(self.output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        # Save config copy
        with open(self.output_dir / 'config.yaml', 'w') as f:
            yaml.dump(self.config, f)

        logger.info(f"\nSaved to: {self.output_dir}")
        logger.info(f"  train.csv: {len(train_df)} samples")
        logger.info(f"  val.csv: {len(val_df)} samples")
        logger.info(f"  test.csv: {len(test_df)} samples")
        logger.info(f"  all.csv: {len(complete)} samples")
        logger.info(f"  metadata.json")
        logger.info(f"  config.yaml")

    def run(self):
        """Run full preparation pipeline."""
        self.build_dataset()

        if len(self.dataset_df) == 0:
            logger.error("No valid samples found!")
            return

        train_df, val_df, test_df = self.create_splits()
        self.save_dataset(train_df, val_df, test_df)

        logger.info("\n" + "=" * 60)
        logger.info("PREPARATION COMPLETE")
        logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Prepare MRI dataset for classification')

    # Config file (recommended)
    parser.add_argument('--config', type=str, help='Path to YAML config file')

    # Command line args (alternative)
    parser.add_argument('--task', type=str, choices=list(TASK_CONFIGS.keys()),
                        help='Classification task')
    parser.add_argument('--dataset', type=str, choices=['adni', 'oasis', 'combined'],
                        help='Dataset to use')
    parser.add_argument('--output', type=str, help='Output directory')
    parser.add_argument('--adni-mri-dir', type=str, help='ADNI MRI directory')
    parser.add_argument('--oasis-mri-dir', type=str, help='OASIS MRI directory')
    parser.add_argument('--train-ratio', type=float, default=0.7)
    parser.add_argument('--val-ratio', type=float, default=0.15)
    parser.add_argument('--test-ratio', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    # Load config
    if args.config:
        config = load_config(args.config)
        logger.info(f"Loaded config: {args.config}")
    else:
        if not args.task or not args.dataset or not args.output:
            parser.error("Either --config or (--task, --dataset, --output) are required")

        config = {
            'task': args.task,
            'dataset': args.dataset,
            'output_dir': args.output,
            'train_ratio': args.train_ratio,
            'val_ratio': args.val_ratio,
            'test_ratio': args.test_ratio,
            'seed': args.seed,
        }
        if args.adni_mri_dir:
            config['adni_mri_dir'] = args.adni_mri_dir
        if args.oasis_mri_dir:
            config['oasis_mri_dir'] = args.oasis_mri_dir

    logger.info(f"Task: {config['task']}")
    logger.info(f"Dataset: {config['dataset']}")
    logger.info(f"Output: {config['output_dir']}")

    preparator = DatasetPreparator(config)
    preparator.run()


if __name__ == '__main__':
    main()

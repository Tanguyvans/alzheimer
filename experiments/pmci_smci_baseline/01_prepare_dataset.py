"""
Dataset Preparation: pMCI vs sMCI Binary Classification (Single MRI per Patient)

This script prepares a dataset for predicting MCI-to-AD conversion using:
- Binary labels: pMCI (converts to AD) vs sMCI (stays stable)
- Single baseline MRI scan per patient (closest to diagnosis date)
- Stratified train/validation/test splits

Usage:
    # Local machine
    python pmci_smci_single_mri.py

    # Cluster with custom paths
    python pmci_smci_single_mri.py \
        --dxsum-path /path/to/dxsum.csv \
        --skull-dir /path/to/ADNI-skull \
        --output-dir /path/to/output

Author: ADNI MCI Prediction Pipeline
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from datetime import datetime
import json
import argparse


class MCIDatasetPreparator:
    """Prepare pMCI vs sMCI binary classification dataset using single baseline MRI"""

    def __init__(self, dxsum_path, skull_dir, output_dir):
        """
        Args:
            dxsum_path: Path to dxsum.csv file
            skull_dir: Path to ADNI-skull directory containing patient folders
            output_dir: Path to save output files (dataset splits, metadata)
        """
        self.dxsum_path = Path(dxsum_path)
        self.skull_dir = Path(skull_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Initializing MCIDatasetPreparator...")
        print(f"  dxsum.csv: {self.dxsum_path}")
        print(f"  skull_dir: {self.skull_dir}")
        print(f"  output_dir: {self.output_dir}")

    def load_diagnosis_data(self):
        """Load and parse diagnosis summary data"""
        print("\n" + "="*80)
        print("LOADING DIAGNOSIS DATA")
        print("="*80)

        self.df = pd.read_csv(self.dxsum_path)
        print(f"Loaded {len(self.df):,} diagnosis records for {self.df['PTID'].nunique():,} patients")

        # Convert EXAMDATE to datetime
        self.df['EXAMDATE'] = pd.to_datetime(self.df['EXAMDATE'])

        return self.df

    def identify_mci_patients(self):
        """Identify MCI patients at baseline and classify as pMCI or sMCI"""
        print("\n" + "="*80)
        print("IDENTIFYING MCI PATIENTS")
        print("="*80)

        # Get baseline diagnosis
        baseline = self.df[self.df['VISCODE'] == 'bl'].copy()
        mci_baseline = baseline[baseline['DIAGNOSIS'] == 2]

        print(f"Found {len(mci_baseline):,} patients with MCI at baseline")

        # Track progression for each MCI patient
        mci_labels = []

        for _, row in mci_baseline.iterrows():
            ptid = row['PTID']
            baseline_date = row['EXAMDATE']

            # Get all follow-up visits for this patient
            patient_visits = self.df[self.df['PTID'] == ptid].sort_values('EXAMDATE')

            # Check if patient converted to AD (diagnosis = 3)
            converted = (patient_visits['DIAGNOSIS'] == 3).any()

            # Find conversion date if applicable
            conversion_date = None
            if converted:
                ad_visits = patient_visits[patient_visits['DIAGNOSIS'] == 3]
                conversion_date = ad_visits['EXAMDATE'].min()
                months_to_conversion = (conversion_date - baseline_date).days / 30.44
            else:
                months_to_conversion = None

            mci_labels.append({
                'PTID': ptid,
                'RID': row['RID'],
                'PHASE': row['PHASE'],
                'baseline_date': baseline_date,
                'label': 1 if converted else 0,  # 1 = pMCI, 0 = sMCI
                'label_name': 'pMCI' if converted else 'sMCI',
                'conversion_date': conversion_date,
                'months_to_conversion': months_to_conversion,
                'total_visits': len(patient_visits)
            })

        self.mci_labels_df = pd.DataFrame(mci_labels)

        print(f"\nClassification results:")
        print(f"  pMCI (progressive): {(self.mci_labels_df['label'] == 1).sum():,} patients ({(self.mci_labels_df['label'] == 1).sum() / len(self.mci_labels_df) * 100:.1f}%)")
        print(f"  sMCI (stable): {(self.mci_labels_df['label'] == 0).sum():,} patients ({(self.mci_labels_df['label'] == 0).sum() / len(self.mci_labels_df) * 100:.1f}%)")

        # Conversion time statistics
        pmci_df = self.mci_labels_df[self.mci_labels_df['label'] == 1]
        if len(pmci_df) > 0:
            print(f"\nConversion time statistics (pMCI patients):")
            print(f"  Mean: {pmci_df['months_to_conversion'].mean():.1f} months")
            print(f"  Median: {pmci_df['months_to_conversion'].median():.1f} months")
            print(f"  Min: {pmci_df['months_to_conversion'].min():.1f} months")
            print(f"  Max: {pmci_df['months_to_conversion'].max():.1f} months")

        return self.mci_labels_df

    def find_baseline_mri_scans(self):
        """Map patient IDs to their baseline MRI scan files"""
        print("\n" + "="*80)
        print("MAPPING TO MRI FILES")
        print("="*80)

        scan_paths = []
        missing_patients = []

        for _, row in self.mci_labels_df.iterrows():
            ptid = row['PTID']
            baseline_date = row['baseline_date']

            # Check if patient folder exists
            patient_dir = self.skull_dir / ptid

            if not patient_dir.exists():
                missing_patients.append(ptid)
                continue

            # Find all scans for this patient
            scans = list(patient_dir.glob("*.nii.gz"))

            if len(scans) == 0:
                missing_patients.append(ptid)
                continue

            # Parse scan dates from filenames (format: MP-RAGE_YYYY-MM-DD_HH_MM_SS.0_...)
            scan_info = []
            for scan_path in scans:
                try:
                    filename = scan_path.name
                    # Extract date from filename
                    date_str = filename.split('_')[1]  # YYYY-MM-DD
                    scan_date = datetime.strptime(date_str, '%Y-%m-%d')

                    scan_info.append({
                        'path': scan_path,
                        'date': scan_date,
                        'days_from_baseline': abs((scan_date - baseline_date.to_pydatetime()).days)
                    })
                except Exception as e:
                    print(f"Warning: Could not parse date from {filename}: {e}")
                    continue

            if len(scan_info) == 0:
                missing_patients.append(ptid)
                continue

            # Select scan closest to baseline date
            scan_info.sort(key=lambda x: x['days_from_baseline'])
            best_scan = scan_info[0]

            scan_paths.append({
                'PTID': ptid,
                'label': row['label'],
                'label_name': row['label_name'],
                'baseline_date': baseline_date,
                'scan_date': best_scan['date'],
                'days_from_baseline': best_scan['days_from_baseline'],
                'scan_path': str(best_scan['path']),
                'months_to_conversion': row['months_to_conversion'],
                'RID': row['RID'],
                'PHASE': row['PHASE']
            })

        self.dataset_df = pd.DataFrame(scan_paths)

        print(f"\nMRI mapping results:")
        print(f"  Successfully mapped: {len(self.dataset_df):,} patients")
        print(f"  Missing/no scans: {len(missing_patients):,} patients")
        print(f"  Average days from baseline to scan: {self.dataset_df['days_from_baseline'].mean():.1f} days")
        print(f"  Max days from baseline: {self.dataset_df['days_from_baseline'].max():.0f} days")

        if len(missing_patients) > 0:
            print(f"\nSample missing patients: {missing_patients[:5]}")

        return self.dataset_df

    def create_train_val_test_split(self, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42):
        """Create stratified train/val/test split"""
        print("\n" + "="*80)
        print("CREATING TRAIN/VAL/TEST SPLIT")
        print("="*80)

        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"

        np.random.seed(random_seed)

        # Separate by class
        pmci_df = self.dataset_df[self.dataset_df['label'] == 1].copy()
        smci_df = self.dataset_df[self.dataset_df['label'] == 0].copy()

        # Shuffle
        pmci_df = pmci_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
        smci_df = smci_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

        # Split each class
        def split_class(df, train_r, val_r, test_r):
            n = len(df)
            train_end = int(n * train_r)
            val_end = train_end + int(n * val_r)

            train = df.iloc[:train_end].copy()
            val = df.iloc[train_end:val_end].copy()
            test = df.iloc[val_end:].copy()

            return train, val, test

        pmci_train, pmci_val, pmci_test = split_class(pmci_df, train_ratio, val_ratio, test_ratio)
        smci_train, smci_val, smci_test = split_class(smci_df, train_ratio, val_ratio, test_ratio)

        # Combine and add split column
        train_df = pd.concat([pmci_train, smci_train]).sample(frac=1, random_state=random_seed).reset_index(drop=True)
        val_df = pd.concat([pmci_val, smci_val]).sample(frac=1, random_state=random_seed).reset_index(drop=True)
        test_df = pd.concat([pmci_test, smci_test]).sample(frac=1, random_state=random_seed).reset_index(drop=True)

        train_df['split'] = 'train'
        val_df['split'] = 'val'
        test_df['split'] = 'test'

        print(f"Split ratios: {train_ratio:.1%} train / {val_ratio:.1%} val / {test_ratio:.1%} test")
        print(f"Random seed: {random_seed}")
        print()

        print("TRAIN SET:")
        print(f"  Total: {len(train_df):,} samples")
        print(f"  pMCI: {(train_df['label'] == 1).sum():,} ({(train_df['label'] == 1).sum() / len(train_df) * 100:.1f}%)")
        print(f"  sMCI: {(train_df['label'] == 0).sum():,} ({(train_df['label'] == 0).sum() / len(train_df) * 100:.1f}%)")

        print("\nVAL SET:")
        print(f"  Total: {len(val_df):,} samples")
        print(f"  pMCI: {(val_df['label'] == 1).sum():,} ({(val_df['label'] == 1).sum() / len(val_df) * 100:.1f}%)")
        print(f"  sMCI: {(val_df['label'] == 0).sum():,} ({(val_df['label'] == 0).sum() / len(val_df) * 100:.1f}%)")

        print("\nTEST SET:")
        print(f"  Total: {len(test_df):,} samples")
        print(f"  pMCI: {(test_df['label'] == 1).sum():,} ({(test_df['label'] == 1).sum() / len(test_df) * 100:.1f}%)")
        print(f"  sMCI: {(test_df['label'] == 0).sum():,} ({(test_df['label'] == 0).sum() / len(test_df) * 100:.1f}%)")

        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        return train_df, val_df, test_df

    def save_dataset(self):
        """Save dataset splits and metadata to CSV files"""
        print("\n" + "="*80)
        print("SAVING DATASET FILES")
        print("="*80)

        # Save individual splits
        train_path = self.output_dir / "train.csv"
        val_path = self.output_dir / "val.csv"
        test_path = self.output_dir / "test.csv"

        self.train_df.to_csv(train_path, index=False)
        self.val_df.to_csv(val_path, index=False)
        self.test_df.to_csv(test_path, index=False)

        print(f"✓ Saved train.csv: {train_path}")
        print(f"✓ Saved val.csv: {val_path}")
        print(f"✓ Saved test.csv: {test_path}")

        # Save complete dataset
        complete_df = pd.concat([self.train_df, self.val_df, self.test_df])
        complete_path = self.output_dir / "dataset_complete.csv"
        complete_df.to_csv(complete_path, index=False)
        print(f"✓ Saved dataset_complete.csv: {complete_path}")

        # Save metadata
        metadata = {
            'creation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'dataset_type': 'pmci_smci_single_mri',
            'description': 'Binary classification (pMCI vs sMCI) using single baseline MRI per patient',
            'dxsum_path': str(self.dxsum_path),
            'skull_dir': str(self.skull_dir),
            'total_samples': len(complete_df),
            'train_samples': len(self.train_df),
            'val_samples': len(self.val_df),
            'test_samples': len(self.test_df),
            'pmci_count': int((complete_df['label'] == 1).sum()),
            'smci_count': int((complete_df['label'] == 0).sum()),
            'conversion_rate': float((complete_df['label'] == 1).sum() / len(complete_df)),
            'class_balance_ratio': float(max((complete_df['label'] == 1).sum(), (complete_df['label'] == 0).sum()) /
                                         min((complete_df['label'] == 1).sum(), (complete_df['label'] == 0).sum())),
        }

        metadata_path = self.output_dir / "dataset_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✓ Saved dataset_metadata.json: {metadata_path}")

        print("\n" + "="*80)
        print("DATASET PREPARATION COMPLETE")
        print("="*80)
        print(f"\nDataset summary:")
        print(f"  Total patients: {len(complete_df):,}")
        print(f"  pMCI: {metadata['pmci_count']:,} ({metadata['conversion_rate']*100:.1f}%)")
        print(f"  sMCI: {metadata['smci_count']:,}")
        print(f"  Class balance: {metadata['class_balance_ratio']:.2f}:1")
        print(f"\nFiles saved to: {self.output_dir}")

        return metadata


def load_config(config_path: str):
    """Load configuration from YAML file"""
    import yaml
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    """Main execution function with command line arguments"""
    parser = argparse.ArgumentParser(
        description='Step 1: Prepare pMCI vs sMCI dataset using single baseline MRI per patient',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration YAML file'
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Extract parameters from config
    dxsum_path = config['data']['dxsum_csv']
    skull_dir = config['data']['skull_dir']
    output_dir = config['data']['splits_dir']
    train_ratio = config['dataset_preparation']['train_ratio']
    val_ratio = config['dataset_preparation']['val_ratio']
    test_ratio = config['dataset_preparation']['test_ratio']
    random_seed = config['dataset_preparation']['random_seed']

    # Create preparator
    preparator = MCIDatasetPreparator(
        dxsum_path=dxsum_path,
        skull_dir=skull_dir,
        output_dir=output_dir
    )

    # Execute pipeline
    preparator.load_diagnosis_data()
    preparator.identify_mci_patients()
    preparator.find_baseline_mri_scans()
    preparator.create_train_val_test_split(
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_seed=random_seed
    )
    preparator.save_dataset()


if __name__ == '__main__':
    main()

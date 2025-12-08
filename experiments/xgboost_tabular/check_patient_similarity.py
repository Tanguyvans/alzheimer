#!/usr/bin/env python3
"""
Check patient similarity between train and test sets.

Detects potential data leakage by finding similar patients across splits.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def check_similarity(train_csv: str, test_csv: str, feature_cols: list = None):
    """
    Check similarity between train and test patients.

    Args:
        train_csv: Path to training data
        test_csv: Path to test data
        feature_cols: Features to use for similarity (None = auto-detect numeric)
    """
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    print("=" * 70)
    print("PATIENT SIMILARITY ANALYSIS")
    print("=" * 70)

    # 1. Check for duplicate Subject IDs
    print("\n1. DUPLICATE SUBJECT IDs")
    print("-" * 40)
    train_subjects = set(train_df['Subject'].unique()) if 'Subject' in train_df.columns else set()
    test_subjects = set(test_df['Subject'].unique()) if 'Subject' in test_df.columns else set()

    overlap = train_subjects & test_subjects
    if overlap:
        print(f"WARNING: {len(overlap)} subjects appear in BOTH train and test!")
        print(f"Overlapping subjects: {list(overlap)[:10]}...")
    else:
        print(f"OK: No overlapping subjects")
        print(f"   Train: {len(train_subjects)} unique subjects")
        print(f"   Test: {len(test_subjects)} unique subjects")

    # 2. Feature-based similarity using k-NN
    print("\n2. FEATURE-BASED SIMILARITY (k-NN)")
    print("-" * 40)

    # Auto-detect numeric features
    if feature_cols is None:
        exclude = ['Subject', 'PTID', 'label', 'y_true', 'y_pred', 'y_proba', 'RID', 'VISCODE']
        feature_cols = [c for c in train_df.columns
                       if train_df[c].dtype in ['float64', 'int64']
                       and c not in exclude]

    # Use common features
    common_features = [f for f in feature_cols if f in train_df.columns and f in test_df.columns]
    print(f"Using {len(common_features)} features: {common_features[:5]}...")

    X_train = train_df[common_features].fillna(0).values
    X_test = test_df[common_features].fillna(0).values

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Find nearest neighbor in train set for each test sample
    nn = NearestNeighbors(n_neighbors=1, metric='euclidean')
    nn.fit(X_train_scaled)
    distances, indices = nn.kneighbors(X_test_scaled)

    distances = distances.flatten()

    print(f"\nDistance to nearest train sample for each test sample:")
    print(f"   Min distance:  {distances.min():.4f}")
    print(f"   Mean distance: {distances.mean():.4f}")
    print(f"   Max distance:  {distances.max():.4f}")
    print(f"   Std:           {distances.std():.4f}")

    # Flag suspiciously similar samples (distance < 0.1)
    threshold = 0.5
    suspicious = distances < threshold
    n_suspicious = suspicious.sum()

    if n_suspicious > 0:
        print(f"\nWARNING: {n_suspicious} test samples have very similar train samples (dist < {threshold})")

        # Show details of suspicious pairs
        suspicious_idx = np.where(suspicious)[0][:5]
        print("\nTop suspicious pairs:")
        for i in suspicious_idx:
            train_idx = indices[i, 0]
            print(f"   Test[{i}] <-> Train[{train_idx}]: dist = {distances[i]:.4f}")
            if 'Subject' in test_df.columns:
                print(f"      Test subject: {test_df.iloc[i]['Subject']}")
                print(f"      Train subject: {train_df.iloc[train_idx]['Subject']}")
    else:
        print(f"\nOK: No test samples have suspiciously similar train samples (all dist >= {threshold})")

    # 3. Plot distance distribution
    print("\n3. SAVING DISTANCE DISTRIBUTION PLOT")
    print("-" * 40)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Histogram
    axes[0].hist(distances, bins=30, edgecolor='black', alpha=0.7)
    axes[0].axvline(threshold, color='red', linestyle='--', label=f'Threshold ({threshold})')
    axes[0].set_xlabel('Distance to nearest train sample')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Test-to-Train Distance Distribution')
    axes[0].legend()

    # Sorted distances
    axes[1].plot(np.sort(distances), marker='.', markersize=2)
    axes[1].axhline(threshold, color='red', linestyle='--', label=f'Threshold ({threshold})')
    axes[1].set_xlabel('Test sample (sorted)')
    axes[1].set_ylabel('Distance to nearest train sample')
    axes[1].set_title('Sorted Distances')
    axes[1].legend()

    plt.tight_layout()
    output_path = Path(train_csv).parent / 'patient_similarity_analysis.png'
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")

    # 4. Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    if len(overlap) == 0 and n_suspicious == 0:
        print("No data leakage detected")
    else:
        print("Potential issues found - review above warnings")

    return distances, overlap


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, default='results/cn_ad_mci_ad/train_data.csv')
    parser.add_argument('--test', type=str, default='results/cn_ad_mci_ad/test_data.csv')
    args = parser.parse_args()

    # Check if files exist, otherwise try predictions
    train_path = Path(args.train)
    test_path = Path(args.test)

    if not train_path.exists():
        # Try to find any saved data
        results_dir = Path('results/cn_ad_mci_ad')
        csvs = list(results_dir.glob('*.csv'))
        print(f"Available CSVs: {[c.name for c in csvs]}")
        print("\nUsage: python check_patient_similarity.py --train <train.csv> --test <test.csv>")
    else:
        check_similarity(args.train, args.test)

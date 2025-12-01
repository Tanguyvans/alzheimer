#!/usr/bin/env python3
"""
Plot confidence distribution comparison: CN vs MCI Stable vs AD

This script visualizes how the binary CN vs AD model classifies patients
from each group, showing the confidence distribution.

Usage:
    python plot_confidence_comparison.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_confidence_comparison(
    binary_predictions_csv: str = 'results/cn_ad_mci_ad/predictions.csv',
    mci_predictions_csv: str = 'results/mci_stable_analysis/mci_stable_predictions.csv',
    output_path: str = 'results/confidence_cn_mci_ad_comparison.png'
):
    """
    Create confidence distribution comparison plot for CN, MCI Stable, and AD.

    Args:
        binary_predictions_csv: Path to binary model predictions (CN vs AD)
        mci_predictions_csv: Path to MCI stable predictions
        output_path: Output path for the figure
    """
    # Load binary model predictions (CN vs AD)
    df_binary = pd.read_csv(binary_predictions_csv)
    df_binary['prob_CN'] = 1 - df_binary['y_proba']
    df_binary['prob_AD'] = df_binary['y_proba']

    # Load MCI stable predictions
    df_mci = pd.read_csv(mci_predictions_csv)
    df_mci['prob_CN'] = 1 - df_mci['AD_probability']
    df_mci['prob_AD'] = df_mci['AD_probability']

    # Separate CN and AD
    cn_df = df_binary[df_binary['y_true'] == 0]
    ad_df = df_binary[df_binary['y_true'] == 1]

    # Create figure with 3 columns
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. CN patients
    ax1 = axes[0]
    ax1.hist(cn_df['prob_CN'] * 100, bins=20, color='#2ecc71', edgecolor='black', alpha=0.7)
    ax1.axvline(50, color='red', linestyle='--', linewidth=2, label='Decision boundary')
    ax1.axvline(cn_df['prob_CN'].mean()*100, color='blue', linestyle='--', linewidth=2,
                label=f"Mean: {cn_df['prob_CN'].mean()*100:.1f}%")
    ax1.set_xlabel('Confidence for CN (%)', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title(f'CN Patients (n={len(cn_df)})\nMean conf: {cn_df["prob_CN"].mean()*100:.1f}%',
                  fontsize=12, fontweight='bold', color='#2ecc71')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 100)

    # 2. MCI Stable patients
    ax2 = axes[1]
    ax2.hist(df_mci['prob_AD'] * 100, bins=20, color='#3498db', edgecolor='black', alpha=0.7)
    ax2.axvline(50, color='red', linestyle='--', linewidth=2, label='Decision boundary')
    ax2.axvline(df_mci['prob_AD'].mean()*100, color='blue', linestyle='--', linewidth=2,
                label=f"Mean: {df_mci['prob_AD'].mean()*100:.1f}%")
    ax2.set_xlabel('AD Probability (%)', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title(f'MCI Stable (n={len(df_mci)})\nMean AD prob: {df_mci["prob_AD"].mean()*100:.1f}%',
                  fontsize=12, fontweight='bold', color='#3498db')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 100)

    # 3. AD patients
    ax3 = axes[2]
    ax3.hist(ad_df['prob_AD'] * 100, bins=20, color='#9b59b6', edgecolor='black', alpha=0.7)
    ax3.axvline(50, color='red', linestyle='--', linewidth=2, label='Decision boundary')
    ax3.axvline(ad_df['prob_AD'].mean()*100, color='blue', linestyle='--', linewidth=2,
                label=f"Mean: {ad_df['prob_AD'].mean()*100:.1f}%")
    ax3.set_xlabel('Confidence for AD (%)', fontsize=12)
    ax3.set_ylabel('Count', fontsize=12)
    ax3.set_title(f'AD Patients (n={len(ad_df)})\nMean conf: {ad_df["prob_AD"].mean()*100:.1f}%',
                  fontsize=12, fontweight='bold', color='#9b59b6')
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 100)

    plt.suptitle('Confidence Distribution: CN vs MCI Stable vs AD', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_path}")

    # Print summary statistics
    print("\n" + "=" * 70)
    print("CONFIDENCE COMPARISON: CN vs MCI Stable vs AD")
    print("=" * 70)

    print(f"\n{'Group':<15} {'N':>6} {'Mean Conf':>12} {'Median':>10} {'≥90%':>10}")
    print("-" * 55)

    # CN
    cn_conf = cn_df['prob_CN']
    print(f"{'CN':<15} {len(cn_df):>6} {cn_conf.mean()*100:>11.1f}% {cn_conf.median()*100:>9.1f}% {(cn_conf >= 0.9).sum()/len(cn_df)*100:>9.1f}%")

    # MCI - show as AD probability
    mci_ad_prob = df_mci['prob_AD']
    print(f"{'MCI Stable':<15} {len(df_mci):>6} {mci_ad_prob.mean()*100:>11.1f}% {mci_ad_prob.median()*100:>9.1f}% {(mci_ad_prob >= 0.9).sum()/len(df_mci)*100:>9.1f}%")

    # AD
    ad_conf = ad_df['prob_AD']
    print(f"{'AD':<15} {len(ad_df):>6} {ad_conf.mean()*100:>11.1f}% {ad_conf.median()*100:>9.1f}% {(ad_conf >= 0.9).sum()/len(ad_df)*100:>9.1f}%")

    print("\n(MCI Stable shown as AD probability - higher = more AD-like)")


if __name__ == '__main__':
    plot_confidence_comparison()

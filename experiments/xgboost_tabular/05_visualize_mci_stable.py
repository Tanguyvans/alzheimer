#!/usr/bin/env python3
"""
Create confusion matrix visualization for MCI-stable predictions

Usage:
    python 05_visualize_mci_stable.py \
        --predictions-csv mci_stable_predictions.csv \
        --output-dir models
"""

import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_mci_confusion_matrix(predictions_df, output_dir):
    """
    Create confusion matrix showing MCI-stable classification distribution

    Since we don't have ground truth labels for MCI-stable patients,
    we create a distribution matrix showing how the model classified them.
    """
    logger.info("Creating MCI-stable classification visualization...")

    # Get predictions
    y_pred = predictions_df['predicted_label'].values
    y_proba = predictions_df['predicted_AD_proba'].values

    # Count classifications
    n_cn = (y_pred == 0).sum()
    n_ad = (y_pred == 1).sum()
    total = len(y_pred)

    # Probability bins
    bins = [0, 0.3, 0.5, 0.7, 1.0]
    bin_labels = ['CN-like\n(<30%)', 'Mild\n(30-50%)', 'Moderate\n(50-70%)', 'AD-like\n(>70%)']
    bin_counts = []

    for i in range(len(bins) - 1):
        count = ((y_proba >= bins[i]) & (y_proba < bins[i+1])).sum()
        if i == len(bins) - 2:  # Last bin includes upper bound
            count = ((y_proba >= bins[i]) & (y_proba <= bins[i+1])).sum()
        bin_counts.append(count)

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Plot 1: Binary classification (CN vs AD)
    binary_data = np.array([[n_cn], [n_ad]])

    sns.set(font_scale=1.3)
    sns.heatmap(binary_data, annot=True, fmt='d', cmap='RdYlGn_r',
                cbar_kws={'label': 'Number of Patients'},
                xticklabels=['MCI-Stable\nPatients'],
                yticklabels=['Classified as CN', 'Classified as AD'],
                annot_kws={'size': 20, 'weight': 'bold'},
                ax=ax1, vmin=0, vmax=total)

    ax1.set_title('MCI-Stable Binary Classification',
                  fontsize=16, fontweight='bold', pad=20)

    # Add percentages
    ax1.text(0.5, -0.15,
             f'CN: {n_cn} ({n_cn/total*100:.1f}%) | AD: {n_ad} ({n_ad/total*100:.1f}%)',
             ha='center', va='center', transform=ax1.transAxes,
             fontsize=13, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 2: Probability distribution bins
    bin_data = np.array(bin_counts).reshape(4, 1)

    sns.heatmap(bin_data, annot=True, fmt='d', cmap='RdYlGn_r',
                cbar_kws={'label': 'Number of Patients'},
                xticklabels=['MCI-Stable\nPatients'],
                yticklabels=bin_labels,
                annot_kws={'size': 18, 'weight': 'bold'},
                ax=ax2, vmin=0, vmax=total/2)

    ax2.set_title('AD Probability Distribution',
                  fontsize=16, fontweight='bold', pad=20)

    # Add interpretation
    interpretation = (
        f"Mean AD Probability: {y_proba.mean():.3f} ± {y_proba.std():.3f}\n"
        f"Median AD Probability: {np.median(y_proba):.3f}"
    )
    ax2.text(0.5, -0.15, interpretation,
             ha='center', va='center', transform=ax2.transAxes,
             fontsize=11, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    plt.tight_layout()

    # Save figure
    output_path = output_dir / 'mci_stable_confusion_matrix.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"✓ MCI-stable confusion matrix saved to {output_path}")
    plt.close()

    sns.reset_orig()  # Reset seaborn settings

    # Print statistics
    logger.info("\n" + "="*80)
    logger.info("MCI-STABLE CLASSIFICATION SUMMARY")
    logger.info("="*80)
    logger.info(f"Total MCI-stable patients: {total}")
    logger.info(f"  Classified as CN: {n_cn} ({n_cn/total*100:.1f}%)")
    logger.info(f"  Classified as AD: {n_ad} ({n_ad/total*100:.1f}%)")
    logger.info(f"\nProbability Distribution:")
    for label, count in zip(bin_labels, bin_counts):
        logger.info(f"  {label.replace(chr(10), ' ')}: {count} ({count/total*100:.1f}%)")
    logger.info(f"\nMean AD Probability: {y_proba.mean():.3f} ± {y_proba.std():.3f}")
    logger.info(f"Median AD Probability: {np.median(y_proba):.3f}")
    logger.info("="*80)


def main():
    parser = argparse.ArgumentParser(description='Visualize MCI-stable predictions')
    parser.add_argument('--predictions-csv', type=str,
                        default='mci_stable_predictions.csv',
                        help='CSV with MCI-stable predictions')
    parser.add_argument('--output-dir', type=str, default='models',
                        help='Output directory for plots')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load predictions
    logger.info(f"Loading predictions from {args.predictions_csv}...")
    predictions_df = pd.read_csv(args.predictions_csv)
    logger.info(f"Loaded {len(predictions_df)} MCI-stable patient samples")
    logger.info(f"Unique patients: {predictions_df['PTID'].nunique()}")

    # Create visualization
    create_mci_confusion_matrix(predictions_df, output_dir)

    logger.info("\n✓ Visualization completed!")


if __name__ == '__main__':
    main()

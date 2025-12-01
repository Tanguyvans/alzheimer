#!/usr/bin/env python3
"""
Step 3: Plot converter detection results.

Shows prediction accuracy for each patient trajectory:
- CN stable
- MCI stable
- MCI -> AD (converters)
- AD stable

Usage:
    python 03_plot_converter_detection.py
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUTPUT_DIR = Path(__file__).parent / "results"


def plot_converter_detection(predictions_csv: str = None, output_dir: str = None):
    """
    Create visualization of converter detection accuracy.

    Args:
        predictions_csv: Path to test predictions CSV
        output_dir: Output directory for figures
    """
    output_dir = Path(output_dir) if output_dir else OUTPUT_DIR
    predictions_path = predictions_csv or output_dir / 'test_predictions.csv'

    # Load predictions
    df = pd.read_csv(predictions_path)

    # Calculate accuracy by trajectory
    trajectories = ['CN_stable', 'MCI_stable', 'MCI_to_AD', 'AD_stable']
    results = {}

    print("=" * 60)
    print("PREDICTION ACCURACY BY PATIENT TRAJECTORY")
    print("=" * 60)
    print()

    for traj in trajectories:
        subset = df[df['trajectory'] == traj]
        if len(subset) > 0:
            correct = int(subset['correct'].sum())
            total = len(subset)
            accuracy = correct / total * 100
            results[traj] = {'correct': correct, 'total': total, 'accuracy': accuracy}
            print(f"{traj:15s}: {correct:3d}/{total:3d} correct ({accuracy:.1f}%)")

    print()

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Left plot: Bar chart showing correct/incorrect predictions ---
    ax1 = axes[0]

    categories = ['CN\n(stable)', 'MCI\n(stable)', 'MCI→AD\n(converters)', 'AD\n(stable)']
    correct = [results[t]['correct'] for t in trajectories]
    incorrect = [results[t]['total'] - results[t]['correct'] for t in trajectories]

    x = np.arange(len(categories))
    width = 0.6

    # Stacked bar
    bars1 = ax1.bar(x, correct, width, label='Correct', color='#2ecc71', edgecolor='black')
    bars2 = ax1.bar(x, incorrect, width, bottom=correct, label='Incorrect', color='#e74c3c', edgecolor='black')

    # Add count labels on bars
    for i, (c, inc) in enumerate(zip(correct, incorrect)):
        total = c + inc
        ax1.text(i, c/2, str(c), ha='center', va='center', fontsize=14, fontweight='bold', color='white')
        if inc > 2:
            ax1.text(i, c + inc/2, str(inc), ha='center', va='center', fontsize=12, fontweight='bold', color='white')

    # Add accuracy on top
    for i, traj in enumerate(trajectories):
        total = results[traj]['total']
        acc = results[traj]['accuracy']
        ax1.text(i, total + 2, f'{acc:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax1.set_ylabel('Number of Patients', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, fontsize=11)
    ax1.set_title('Prediction Accuracy by Patient Trajectory', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.set_ylim(0, max([r['total'] for r in results.values()]) + 15)
    ax1.grid(axis='y', alpha=0.3)

    # Highlight MCI→AD with a box
    rect = mpatches.FancyBboxPatch((1.6, -5), 0.8, max([r['total'] for r in results.values()]) + 5,
                                    boxstyle="round,pad=0.05",
                                    facecolor='none', edgecolor='#3498db',
                                    linewidth=3, linestyle='--')
    ax1.add_patch(rect)

    # --- Right plot: Focus on MCI→AD converters ---
    ax2 = axes[1]

    # Pie chart for MCI→AD
    mci_correct = results['MCI_to_AD']['correct']
    mci_total = results['MCI_to_AD']['total']
    mci_incorrect = mci_total - mci_correct

    colors = ['#2ecc71', '#e74c3c']
    explode = (0.05, 0.05)

    wedges, texts, autotexts = ax2.pie(
        [mci_correct, mci_incorrect],
        labels=['Correctly predicted\nas AD', 'Incorrectly predicted\nas CN/MCI'],
        autopct=lambda p: f'{int(p*mci_total/100)}\n({p:.1f}%)',
        colors=colors,
        explode=explode,
        startangle=90,
        textprops={'fontsize': 11},
        wedgeprops={'edgecolor': 'black', 'linewidth': 1.5}
    )
    autotexts[0].set_fontsize(14)
    autotexts[0].set_fontweight('bold')
    autotexts[1].set_fontsize(12)

    ax2.set_title(f'MCI → AD Converters Detection\n(n={mci_total} patients)',
                  fontsize=14, fontweight='bold', color='#3498db')

    # Add annotation
    ax2.annotate(f'{results["MCI_to_AD"]["accuracy"]:.1f}% of patients who\nconverted from MCI to AD\nwere correctly identified!',
                xy=(0.5, -0.15), xycoords='axes fraction',
                ha='center', fontsize=12,
                bbox=dict(boxstyle='round', facecolor='#ecf0f1', edgecolor='#3498db', linewidth=2))

    plt.suptitle('XGBoost with Longitudinal Features: Converter Detection',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    output_path = output_dir / 'converter_detection.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_path}")
    print()
    print("=== Key Insight ===")
    print(f"The model correctly identifies {results['MCI_to_AD']['accuracy']:.1f}% of MCI→AD converters,")
    print(f"but only {results['MCI_stable']['accuracy']:.1f}% of stable MCI patients.")
    print("This shows longitudinal features help detect patients at risk of conversion!")


def main():
    parser = argparse.ArgumentParser(description='Plot converter detection results')
    parser.add_argument('--predictions', type=str, default=None, help='Path to predictions CSV')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory')
    args = parser.parse_args()

    plot_converter_detection(args.predictions, args.output_dir)


if __name__ == '__main__':
    main()

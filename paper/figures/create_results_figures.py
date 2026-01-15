#!/usr/bin/env python3
"""
Generate paper figures for Alzheimer's multimodal classification results.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['figure.dpi'] = 150

# Colors
COLORS = {
    'multimodal': '#2E86AB',      # Blue
    'mri_only': '#A23B72',         # Purple/Pink
    'tabular_only': '#F18F01',     # Orange
    'cn': '#28A745',               # Green
    'ad': '#DC3545',               # Red
    'ad_trajectory': '#FFC107',    # Yellow
    'mci_to_ad': '#6F42C1',        # Purple
}


def load_results():
    """Load all experiment results."""
    base_path = Path(__file__).parent.parent.parent / "experiments"

    results = {}

    # Multimodal fusion
    mm_path = base_path / "multimodal_fusion/cv_results/cv_summary.json"
    if mm_path.exists():
        with open(mm_path) as f:
            results['multimodal'] = json.load(f)

    # MRI only
    mri_path = base_path / "mri_vit_ad/cv_results/cv_summary.json"
    if mri_path.exists():
        with open(mri_path) as f:
            results['mri_only'] = json.load(f)

    # Tabular only (if available)
    tab_path = base_path / "ablation_tabular_only/cv_results_cn_ad_trajectory/cv_results.json"
    if tab_path.exists():
        with open(tab_path) as f:
            results['tabular_only'] = json.load(f)

    return results


def plot_model_comparison(results, output_path):
    """Create bar chart comparing model performance."""

    models = []
    accuracies = []
    acc_stds = []
    balanced_accs = []
    bal_stds = []
    colors = []

    if 'multimodal' in results:
        models.append('Multimodal\n(MRI + Tabular)')
        accuracies.append(results['multimodal']['overall_accuracy_mean'])
        acc_stds.append(results['multimodal']['overall_accuracy_std'])
        balanced_accs.append(results['multimodal']['balanced_accuracy_mean'])
        bal_stds.append(results['multimodal']['balanced_accuracy_std'])
        colors.append(COLORS['multimodal'])

    if 'mri_only' in results:
        models.append('MRI Only\n(ViT)')
        accuracies.append(results['mri_only']['overall_accuracy_mean'])
        acc_stds.append(results['mri_only']['overall_accuracy_std'])
        balanced_accs.append(results['mri_only']['balanced_accuracy_mean'])
        bal_stds.append(results['mri_only']['balanced_accuracy_std'])
        colors.append(COLORS['mri_only'])

    if 'tabular_only' in results:
        models.append('Tabular Only\n(FT-Transformer)')
        accuracies.append(results['tabular_only']['traj_accuracy'])
        acc_stds.append(results['tabular_only']['traj_accuracy_std'])
        balanced_accs.append(results['tabular_only']['traj_balanced_accuracy'])
        bal_stds.append(results['tabular_only']['traj_balanced_accuracy_std'])
        colors.append(COLORS['tabular_only'])

    if len(models) == 0:
        print("No results to plot")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(models))
    width = 0.35

    bars1 = ax.bar(x - width/2, accuracies, width, yerr=acc_stds,
                   label='Accuracy', color=colors, alpha=0.9, capsize=5)
    bars2 = ax.bar(x + width/2, balanced_accs, width, yerr=bal_stds,
                   label='Balanced Accuracy', color=colors, alpha=0.5, capsize=5,
                   hatch='///')

    ax.set_ylabel('Performance (%)')
    ax.set_title('Model Comparison: CN vs AD Classification')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend(loc='lower right')
    ax.set_ylim(0, 100)

    # Add value labels
    for bar, val in zip(bars1, accuracies):
        ax.annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                   xytext=(0, 3), textcoords='offset points', ha='center', va='bottom',
                   fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_subgroup_performance(results, output_path):
    """Create bar chart showing performance per subgroup."""

    if 'multimodal' not in results or 'subgroup_summary' not in results['multimodal']:
        print("No subgroup data available")
        return

    subgroups = results['multimodal']['subgroup_summary']

    # Order: CN, MCI_to_AD, AD_trajectory, AD
    order = ['CN', 'MCI_to_AD', 'AD_trajectory', 'AD']
    labels = ['CN\n(Cognitively Normal)', 'MCI→AD\n(Converters)',
              'AD Trajectory\n(Progressive)', 'AD\n(Alzheimer\'s)']

    accs = []
    stds = []
    samples = []
    colors = [COLORS['cn'], COLORS['mci_to_ad'], COLORS['ad_trajectory'], COLORS['ad']]

    for sg in order:
        if sg in subgroups:
            accs.append(subgroups[sg]['accuracy_mean'])
            stds.append(subgroups[sg]['accuracy_std'])
            samples.append(subgroups[sg]['avg_samples_per_fold'])

    fig, ax = plt.subplots(figsize=(9, 5))

    x = np.arange(len(order))
    bars = ax.bar(x, accs, yerr=stds, color=colors, alpha=0.85, capsize=5,
                  edgecolor='black', linewidth=0.5)

    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Multimodal Fusion: Performance by Clinical Subgroup')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 105)

    # Add value labels with sample counts
    for bar, val, std, n in zip(bars, accs, stds, samples):
        ax.annotate(f'{val:.1f}%\n(n≈{int(n)})',
                   xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                   xytext=(0, 5), textcoords='offset points', ha='center', va='bottom',
                   fontsize=9, fontweight='bold')

    # Add horizontal line at overall accuracy
    overall = results['multimodal']['overall_accuracy_mean']
    ax.axhline(y=overall, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.annotate(f'Overall: {overall:.1f}%', xy=(len(order)-0.5, overall+1),
               fontsize=9, color='gray')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_metrics_comparison(results, output_path):
    """Create comprehensive metrics comparison (Acc, Bal Acc, Sens, Spec, AUC)."""

    if 'tabular_only' not in results:
        print("Waiting for tabular results...")
        return

    metrics = ['Accuracy', 'Balanced\nAccuracy', 'Sensitivity', 'Specificity', 'AUC']

    # Extract metrics for each model
    model_data = {}

    if 'multimodal' in results:
        # Multimodal doesn't have all metrics in summary, use what's available
        model_data['Multimodal'] = {
            'values': [
                results['multimodal']['overall_accuracy_mean'],
                results['multimodal']['balanced_accuracy_mean'],
                None, None, None  # No sens/spec/auc in current format
            ],
            'color': COLORS['multimodal']
        }

    if 'tabular_only' in results:
        tab = results['tabular_only']
        model_data['Tabular Only'] = {
            'values': [
                tab.get('traj_accuracy', 0),
                tab.get('traj_balanced_accuracy', 0),
                tab.get('traj_sensitivity', 0),
                tab.get('traj_specificity', 0),
                tab.get('traj_auc', 0) * 100  # Convert to percentage
            ],
            'color': COLORS['tabular_only']
        }

    # Filter models with complete data
    complete_models = {k: v for k, v in model_data.items()
                      if all(x is not None for x in v['values'])}

    if len(complete_models) < 1:
        print("Not enough complete data for metrics comparison")
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(metrics))
    width = 0.25
    multiplier = 0

    for model_name, data in complete_models.items():
        offset = width * multiplier
        bars = ax.bar(x + offset, data['values'], width, label=model_name,
                     color=data['color'], alpha=0.85)
        multiplier += 1

    ax.set_ylabel('Score (%)')
    ax.set_title('Comprehensive Model Comparison')
    ax.set_xticks(x + width * (len(complete_models) - 1) / 2)
    ax.set_xticklabels(metrics)
    ax.legend(loc='lower right')
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def print_results_table(results):
    """Print LaTeX-ready results table."""

    print("\n" + "="*60)
    print("RESULTS SUMMARY (LaTeX Table)")
    print("="*60)

    print("\n\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Model Performance Comparison}")
    print("\\begin{tabular}{lccc}")
    print("\\toprule")
    print("Model & Accuracy (\\%) & Balanced Acc (\\%) & AUC \\\\")
    print("\\midrule")

    if 'multimodal' in results:
        mm = results['multimodal']
        print(f"Multimodal Fusion & {mm['overall_accuracy_mean']:.1f} $\\pm$ {mm['overall_accuracy_std']:.1f} & "
              f"{mm['balanced_accuracy_mean']:.1f} $\\pm$ {mm['balanced_accuracy_std']:.1f} & - \\\\")

    if 'mri_only' in results:
        mri = results['mri_only']
        print(f"MRI Only (ViT) & {mri['overall_accuracy_mean']:.1f} $\\pm$ {mri['overall_accuracy_std']:.1f} & "
              f"{mri['balanced_accuracy_mean']:.1f} $\\pm$ {mri['balanced_accuracy_std']:.1f} & - \\\\")

    if 'tabular_only' in results:
        tab = results['tabular_only']
        print(f"Tabular Only & {tab['traj_accuracy']:.1f} $\\pm$ {tab['traj_accuracy_std']:.1f} & "
              f"{tab['traj_balanced_accuracy']:.1f} $\\pm$ {tab['traj_balanced_accuracy_std']:.1f} & "
              f"{tab['traj_auc']:.3f} $\\pm$ {tab['traj_auc_std']:.3f} \\\\")

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")
    print()


def main():
    output_dir = Path(__file__).parent

    print("Loading results...")
    results = load_results()

    print(f"Found results for: {list(results.keys())}")

    # Generate plots
    print("\nGenerating figures...")

    plot_model_comparison(results, output_dir / "model_comparison.png")
    plot_subgroup_performance(results, output_dir / "subgroup_performance.png")
    plot_metrics_comparison(results, output_dir / "metrics_comparison.png")

    # Print table
    print_results_table(results)

    print("\nDone!")


if __name__ == "__main__":
    main()

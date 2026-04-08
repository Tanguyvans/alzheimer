#!/usr/bin/env python3
"""
Multi-seed analysis: aggregate results across seeds, DeLong tests, GradCAM.

Loads predictions from seed_* subdirectories, computes mean +/- std
for all metrics, runs DeLong tests on mean probabilities, and GradCAM from
the best-AUC seed. Methods with fewer than EXPECTED_SEEDS seeds are marked
"(1 seed)" in the heatmap.

Usage:
    python analyze_multi_seed.py                  # CPU analyses only
    python analyze_multi_seed.py --gradcam        # include GradCAM (needs GPU)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from sklearn.metrics import (confusion_matrix, roc_curve, auc,
                             roc_auc_score, accuracy_score, balanced_accuracy_score)
import argparse
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# ── Paths ──
BASE = Path(__file__).parent
MLP_DIR = BASE / "resnet3d_mlp"
XGB_DIR = BASE / "resnet3d_xgboost"
REPORT_DIR = BASE / "report_multi_seed"
REPORT_DIR.mkdir(exist_ok=True)

MAX_SEED = 20   # scan up to seed_20
EXPECTED_SEEDS = 5  # number of seeds expected per method


# ═══════════════════════════════════════════════════════
# Load predictions across all seeds
# ═══════════════════════════════════════════════════════

def load_all_seeds():
    """Load predictions for all seeds per method (supports different seed counts)."""
    all_preds = {}   # method -> list of proba arrays
    seed_counts = {} # method -> list of seed indices
    y_true = None

    def _add(method, seed, proba, yt):
        nonlocal y_true
        y_true = yt
        if method not in all_preds:
            all_preds[method] = []
            seed_counts[method] = []
        if len(all_preds[method]) >= EXPECTED_SEEDS:
            return  # cap at EXPECTED_SEEDS
        all_preds[method].append(proba)
        seed_counts[method].append(seed)

    # MLP early fusion
    for seed in range(MAX_SEED):
        p = MLP_DIR / "results_early" / f"seed_{seed}"
        if (p / "y_true_test.npy").exists():
            yt = np.load(p / "y_true_test.npy")
            _add("MLP Early", seed, np.load(p / "y_proba_test.npy"), yt)

    # MLP late fusion
    for seed in range(MAX_SEED):
        p = MLP_DIR / "results_late_fusion" / f"seed_{seed}"
        if (p / "y_true_test.npy").exists():
            yt = np.load(p / "y_true_test.npy")
            # MRI only (MLP) skipped — redundant with MRI only
            _add("Tab only (MLP)", seed, np.load(p / "y_proba_tab_test.npy"), yt)
            _add("MLP Late Avg", seed, np.load(p / "y_proba_avg_test.npy"), yt)
            _add("MLP Late Wt", seed, np.load(p / "y_proba_weighted_test.npy"), yt)
            _add("MLP Late Stack", seed, np.load(p / "y_proba_stacking_test.npy"), yt)

    # XGB early fusion
    for seed in range(MAX_SEED):
        p = XGB_DIR / "results_finetuned" / f"seed_{seed}"
        if (p / "y_true_test.npy").exists():
            yt = np.load(p / "y_true_test.npy")
            _add("XGB Early", seed, np.load(p / "y_proba_test.npy"), yt)

    # XGB late fusion
    for seed in range(MAX_SEED):
        p = XGB_DIR / "results_late_fusion" / f"seed_{seed}"
        if (p / "y_true_test.npy").exists():
            yt = np.load(p / "y_true_test.npy")
            _add("MRI only", seed, np.load(p / "y_proba_mri_test.npy"), yt)
            _add("Tab only (XGB)", seed, np.load(p / "y_proba_tab_test.npy"), yt)
            _add("XGB Late Avg", seed, np.load(p / "y_proba_avg_test.npy"), yt)
            _add("XGB Late Wt", seed, np.load(p / "y_proba_weighted_test.npy"), yt)
            _add("XGB Late Stack", seed, np.load(p / "y_proba_stacking_test.npy"), yt)

    for method in all_preds:
        seeds = seed_counts[method]
        print(f"  {method}: {len(seeds)} seeds ({seeds})")

    return y_true, all_preds, seed_counts


def compute_per_seed_metrics(y_true, all_preds):
    """Compute metrics for each seed and method. Returns DataFrame."""
    rows = []
    for method, proba_list in all_preds.items():
        for seed_idx, y_proba in enumerate(proba_list):
            y_pred = (y_proba >= 0.5).astype(int)
            cm = confusion_matrix(y_true, y_pred)
            tn, fp, fn, tp = cm.ravel()
            rows.append({
                'Method': method,
                'Seed': seed_idx,
                'Accuracy': accuracy_score(y_true, y_pred) * 100,
                'Bal Acc': balanced_accuracy_score(y_true, y_pred) * 100,
                'Sensitivity': tp / (tp + fn) * 100 if (tp + fn) > 0 else 0,
                'Specificity': tn / (tn + fp) * 100 if (tn + fp) > 0 else 0,
                'AUC': roc_auc_score(y_true, y_proba),
            })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════
# 1. Summary Table (mean +/- std)
# ═══════════════════════════════════════════════════════

def generate_summary(metrics_df):
    """Generate mean +/- std summary table."""
    print("\n" + "=" * 80)
    print("SUMMARY TABLE (mean +/- std over seeds)")
    print("=" * 80)

    summary = metrics_df.groupby('Method').agg(
        Acc_mean=('Accuracy', 'mean'), Acc_std=('Accuracy', 'std'),
        BAcc_mean=('Bal Acc', 'mean'), BAcc_std=('Bal Acc', 'std'),
        Sens_mean=('Sensitivity', 'mean'), Sens_std=('Sensitivity', 'std'),
        Spec_mean=('Specificity', 'mean'), Spec_std=('Specificity', 'std'),
        AUC_mean=('AUC', 'mean'), AUC_std=('AUC', 'std'),
        N_seeds=('Seed', 'count'),
    ).sort_values('AUC_mean', ascending=False)

    # Pretty print
    def _fmt(mean, std, is_auc=False):
        """Format mean+/-std, show N/A for std when only 1 seed."""
        if is_auc:
            if np.isnan(std):
                return f"{mean:5.3f}  (N/A)"
            return f"{mean:5.3f}+/-{std:.3f}"
        else:
            if np.isnan(std):
                return f"{mean:5.1f}  (N/A)"
            return f"{mean:5.1f}+/-{std:4.1f}"

    print(f"\n{'Method':<20} {'Acc %':>14} {'Bal Acc %':>14} {'Sens %':>14} {'Spec %':>14} {'AUC':>14} {'N':>3}")
    print("-" * 100)
    for method, row in summary.iterrows():
        print(f"{method:<20} "
              f"{_fmt(row['Acc_mean'], row['Acc_std']):>14} "
              f"{_fmt(row['BAcc_mean'], row['BAcc_std']):>14} "
              f"{_fmt(row['Sens_mean'], row['Sens_std']):>14} "
              f"{_fmt(row['Spec_mean'], row['Spec_std']):>14} "
              f"{_fmt(row['AUC_mean'], row['AUC_std'], is_auc=True):>14} "
              f"{int(row['N_seeds']):3d}")

    # Save detailed per-seed CSV
    metrics_df.to_csv(REPORT_DIR / "per_seed_metrics.csv", index=False)
    summary.to_csv(REPORT_DIR / "summary_table.csv", na_rep='N/A')
    print(f"\nSaved: {REPORT_DIR / 'per_seed_metrics.csv'}")
    print(f"Saved: {REPORT_DIR / 'summary_table.csv'}")

    return summary


# ═══════════════════════════════════════════════════════
# 2. Box plots
# ═══════════════════════════════════════════════════════

def plot_boxplots(metrics_df):
    """Box plots of AUC and Bal Acc across seeds for each method."""
    print("\n" + "=" * 60)
    print("Box Plots")
    print("=" * 60)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Order by median AUC
    order = metrics_df.groupby('Method')['AUC'].median().sort_values(ascending=False).index.tolist()

    sns.boxplot(data=metrics_df, x='AUC', y='Method', order=order, ax=ax1,
                palette='viridis', orient='h')
    sns.stripplot(data=metrics_df, x='AUC', y='Method', order=order, ax=ax1,
                  color='black', size=3, alpha=0.5, orient='h')
    ax1.set_title(f'AUC Distribution ({EXPECTED_SEEDS} seeds)', fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)

    sns.boxplot(data=metrics_df, x='Bal Acc', y='Method', order=order, ax=ax2,
                palette='viridis', orient='h')
    sns.stripplot(data=metrics_df, x='Bal Acc', y='Method', order=order, ax=ax2,
                  color='black', size=3, alpha=0.5, orient='h')
    ax2.set_title(f'Balanced Accuracy Distribution ({EXPECTED_SEEDS} seeds)', fontweight='bold')
    ax2.set_xlabel('Balanced Accuracy (%)')
    ax2.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    fig.savefig(REPORT_DIR / "boxplots.png", dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {REPORT_DIR / 'boxplots.png'}")


# ═══════════════════════════════════════════════════════
# 3. DeLong Test (on mean probabilities)
# ═══════════════════════════════════════════════════════

def compute_midrank(x):
    j = np.argsort(x)
    z = x[j]
    n = len(x)
    rank = np.zeros(n)
    i = 0
    while i < n:
        a = i
        while a < n - 1 and z[a + 1] == z[a]:
            a += 1
        for k in range(i, a + 1):
            rank[j[k]] = 0.5 * (i + a) + 1
        i = a + 1
    return rank


def fastDeLong(predictions_sorted_transposed, label_1_count):
    """Fast DeLong AUC computation (Sun & Xu 2014, corrected)."""
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    k = predictions_sorted_transposed.shape[0]

    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]

    tx = np.empty([k, m], dtype=np.float64)
    ty = np.empty([k, n], dtype=np.float64)
    tz = np.empty([k, m + n], dtype=np.float64)

    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])

    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m

    sx = np.cov(v01) if k > 1 else np.atleast_2d(np.var(v01, axis=1))
    sy = np.cov(v10) if k > 1 else np.atleast_2d(np.var(v10, axis=1))
    S = sx / m + sy / n
    return aucs, S


def delong_test(y_true, y_score1, y_score2):
    order = (-y_true).argsort()
    label_1_count = int(y_true.sum())
    predictions = np.vstack([y_score1, y_score2])
    predictions_sorted = predictions[:, order]
    aucs, S = fastDeLong(predictions_sorted, label_1_count)
    diff = aucs[0] - aucs[1]
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    if var <= 0:
        return aucs[0], aucs[1], 0.0, 1.0
    z = diff / np.sqrt(var)
    p = 2 * stats.norm.sf(abs(z))
    return aucs[0], aucs[1], z, p


def run_delong_analysis(y_true, all_preds, seed_counts=None):
    """DeLong test on mean probabilities across seeds."""
    print("\n" + "=" * 60)
    print("DeLong Test - Pairwise AUC comparison (mean probabilities)")
    print("=" * 60)

    from matplotlib.colors import LinearSegmentedColormap

    # Number of seeds per method
    n_seeds = {}
    for method in all_preds:
        if seed_counts and method in seed_counts:
            n_seeds[method] = len(seed_counts[method])
        else:
            n_seeds[method] = len(all_preds[method])

    # Compute mean probabilities and AUCs
    mean_preds = {}
    auc_values = {}
    for method, proba_list in all_preds.items():
        mean_preds[method] = np.mean(proba_list, axis=0)
        auc_values[method] = roc_auc_score(y_true, mean_preds[method])

    # Logical ordering: unimodal → early fusion → late fusion, then AUC desc
    def _order_key(name):
        if 'only' in name.lower():
            group = 0
        elif 'Early' in name:
            group = 1
        else:
            group = 2
        return (group, -auc_values[name])

    sorted_names = sorted(all_preds.keys(), key=_order_key)
    nc = len(sorted_names)

    # Compute DeLong for ALL pairs (single pass)
    p_matrix = np.full((nc, nc), np.nan)
    for i in range(nc):
        p_matrix[i, i] = 1.0
        for j in range(i + 1, nc):
            _, _, z, p_val = delong_test(y_true, mean_preds[sorted_names[i]], mean_preds[sorted_names[j]])
            p_matrix[i, j] = p_val
            p_matrix[j, i] = p_val

    # Print
    print(f"\n{'Method':<30} {'AUC (mean proba)':>16} {'Seeds':>6}")
    print("-" * 54)
    for name in sorted(sorted_names, key=lambda x: auc_values[x], reverse=True):
        print(f"{name:<30} {auc_values[name]:.4f} {n_seeds[name]:>6}")

    print("\nSignificant differences (p < 0.05):")
    found = False
    for i in range(nc):
        for j in range(i + 1, nc):
            if p_matrix[i, j] < 0.05:
                found = True
                diff = auc_values[sorted_names[i]] - auc_values[sorted_names[j]]
                print(f"  {sorted_names[i]} vs {sorted_names[j]}: p={p_matrix[i,j]:.4f}, AUC diff={diff:+.4f}")
    if not found:
        print("  None - all AUC differences are not statistically significant")

    # Labels: mark incomplete methods
    labels = []
    for name in sorted_names:
        if n_seeds[name] < EXPECTED_SEEDS:
            labels.append(f"{name} (1 seed)")
        else:
            labels.append(name)

    # Annotations
    annot = np.full((nc, nc), "", dtype=object)
    for i in range(nc):
        for j in range(nc):
            if i == j:
                annot[i, j] = f"AUC\n{auc_values[sorted_names[i]]:.3f}"
            else:
                p = p_matrix[i, j]
                if p < 0.001:
                    annot[i, j] = "p<.001\n***"
                elif p < 0.01:
                    annot[i, j] = f"p={p:.3f}\n**"
                elif p < 0.05:
                    annot[i, j] = f"p={p:.3f}\n*"
                else:
                    annot[i, j] = f"p={p:.2f}\nns"

    # Color matrix: -log10(p), diagonal masked
    color_matrix = np.full((nc, nc), np.nan)
    for i in range(nc):
        for j in range(nc):
            if i != j:
                color_matrix[i, j] = -np.log10(max(p_matrix[i, j], 1e-10))

    mask = np.eye(nc, dtype=bool)

    fig, ax = plt.subplots(figsize=(max(10, nc * 1.1), max(9, nc * 1.0)))

    colors_list = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c', '#8e44ad']
    cmap = LinearSegmentedColormap.from_list('significance', colors_list, N=256)

    sns.heatmap(color_matrix, xticklabels=labels, yticklabels=labels,
                annot=annot, fmt='', cmap=cmap, vmin=0, vmax=5,
                mask=mask, square=True, ax=ax,
                cbar_kws={'label': 'Significativite  (-log10 p)', 'shrink': 0.7},
                annot_kws={'size': 8},
                linewidths=1.5, linecolor='white')

    # Diagonal cells: light gray + bold AUC
    for i in range(nc):
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=True,
                     facecolor='#ecf0f1', edgecolor='white', lw=1.5))
        ax.text(i + 0.5, i + 0.5, f"AUC\n{auc_values[sorted_names[i]]:.3f}",
                ha='center', va='center', fontsize=9, fontweight='bold', color='#2c3e50')

    # Group separators (thick lines between unimodal / early / late)
    group_boundaries = []
    prev_group = None
    for i, name in enumerate(sorted_names):
        if 'only' in name.lower():
            g = 0
        elif 'Early' in name:
            g = 1
        else:
            g = 2
        if prev_group is not None and g != prev_group:
            group_boundaries.append(i)
        prev_group = g
    for b in group_boundaries:
        ax.axhline(y=b, color='#2c3e50', linewidth=2.5)
        ax.axvline(x=b, color='#2c3e50', linewidth=2.5)

    ax.set_title(f'Test de DeLong - Comparaison par paires\n'
                 f'(probabilites moyennees sur {EXPECTED_SEEDS} seeds)\n'
                 f'* p<0.05   ** p<0.01   *** p<0.001   ns = non significatif',
                 fontsize=11, fontweight='bold', pad=12)
    plt.xticks(rotation=40, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    fig.savefig(REPORT_DIR / "delong_test.png", dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {REPORT_DIR / 'delong_test.png'}")

    df = pd.DataFrame(p_matrix, index=labels, columns=labels)
    df.to_csv(REPORT_DIR / "delong_pvalues.csv", na_rep='N/A')
    print(f"Saved: {REPORT_DIR / 'delong_pvalues.csv'}")

    return mean_preds


# ═══════════════════════════════════════════════════════
# 4. ROC Curves (mean +/- std band)
# ═══════════════════════════════════════════════════════

def plot_roc_curves(y_true, all_preds):
    """ROC curves with mean +/- std shaded band across seeds."""
    print("\n" + "=" * 60)
    print("ROC Curves (mean +/- std)")
    print("=" * 60)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    mean_fpr = np.linspace(0, 1, 200)

    mlp_methods = [
        ("MRI only (MLP)", "MRI only", "--", 'gray'),
        ("Tab only (MLP)", "Tabular MLP", "--", 'steelblue'),
        ("MLP Early", "Early Fusion", "-", 'darkorange'),
        ("MLP Late Wt", "Late Weighted", "-", 'green'),
        ("MLP Late Stack", "Late Stacking", "-.", 'purple'),
    ]
    xgb_methods = [
        ("MRI only", "MRI only", "--", 'gray'),
        ("Tab only (XGB)", "Tabular XGBoost", "--", 'steelblue'),
        ("XGB Early", "Early Fusion", "-", 'darkorange'),
        ("XGB Late Wt", "Late Weighted", "-", 'green'),
        ("XGB Late Stack", "Late Stacking", "-.", 'purple'),
    ]

    def _plot_method(ax, key, label, ls, color, all_preds, y_true):
        if key not in all_preds:
            return
        tprs = []
        aucs_list = []
        for proba in all_preds[key]:
            fpr, tpr, _ = roc_curve(y_true, proba)
            tprs.append(np.interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            aucs_list.append(auc(fpr, tpr))
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        std_tpr = np.std(tprs, axis=0)
        mean_auc = np.mean(aucs_list)
        std_auc = np.std(aucs_list)

        ax.plot(mean_fpr, mean_tpr, ls=ls, color=color, linewidth=2,
                label=f"{label} ({mean_auc:.3f}+/-{std_auc:.3f})")
        ax.fill_between(mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr,
                        color=color, alpha=0.1)

    for key, label, ls, color in mlp_methods:
        _plot_method(ax1, key, label, ls, color, all_preds, y_true)
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ResNet3D + MLP', fontweight='bold')
    ax1.legend(loc='lower right', fontsize=8)
    ax1.grid(alpha=0.3)

    for key, label, ls, color in xgb_methods:
        _plot_method(ax2, key, label, ls, color, all_preds, y_true)
    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('ResNet3D + XGBoost', fontweight='bold')
    ax2.legend(loc='lower right', fontsize=8)
    ax2.grid(alpha=0.3)

    plt.suptitle(f'ROC Curves (mean +/- std, {EXPECTED_SEEDS} seeds)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(REPORT_DIR / "roc_curves.png", dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {REPORT_DIR / 'roc_curves.png'}")


# ═══════════════════════════════════════════════════════
# 5. Confusion Matrices (from mean probabilities)
# ═══════════════════════════════════════════════════════

def plot_confusion_matrices(y_true, mean_preds):
    """Confusion matrices from mean probabilities."""
    print("\n" + "=" * 60)
    print("Confusion Matrices (mean probabilities)")
    print("=" * 60)

    methods = [
        ("MRI only", "MRI only\n(ResNet3D)"),
        ("Tab only (XGB)", "Tabular only\n(XGBoost)"),
        ("Tab only (MLP)", "Tabular only\n(MLP)"),
        ("XGB Early", "Early Fusion\n(ResNet3D+XGB)"),
        ("MLP Early", "Early Fusion\n(ResNet3D+MLP)"),
        ("XGB Late Wt", "Late Weighted\n(ResNet3D+XGB)"),
        ("MLP Late Wt", "Late Weighted\n(ResNet3D+MLP)"),
        ("XGB Late Stack", "Late Stacking\n(ResNet3D+XGB)"),
    ]

    available = [(key, label) for key, label in methods if key in mean_preds]
    n_methods = len(available)
    ncols = 4
    nrows = (n_methods + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.5 * nrows))
    if nrows == 1:
        axes = axes.reshape(1, -1)

    for idx, (key, label) in enumerate(available):
        row, col = idx // ncols, idx % ncols
        ax = axes[row, col]
        y_pred = (mean_preds[key] >= 0.5).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['CN', 'AD'], yticklabels=['CN', 'AD'],
                    cbar=False, annot_kws={'size': 16, 'weight': 'bold'}, square=True)
        for i in range(2):
            for j in range(2):
                ax.text(j + 0.5, i + 0.75, f'({cm_pct[i, j]:.1f}%)',
                        ha='center', va='center', fontsize=9, color='gray')
        auc_val = roc_auc_score(y_true, mean_preds[key])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True' if col == 0 else '')
        ax.set_title(f'{label}\nAUC={auc_val:.3f}', fontsize=10, fontweight='bold')

    for idx in range(n_methods, nrows * ncols):
        row, col = idx // ncols, idx % ncols
        axes[row, col].set_visible(False)

    n_cn = int((y_true == 0).sum())
    n_ad = int((y_true == 1).sum())
    plt.suptitle(f'Confusion Matrices - Mean Probabilities ({n_cn} CN, {n_ad} AD)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(REPORT_DIR / "confusion_matrices.png", dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {REPORT_DIR / 'confusion_matrices.png'}")


# ═══════════════════════════════════════════════════════
# 6. Interpretability: Integrated Gradients (best seed)
#    - Individual images (5 AD + 5 CN)
#    - Group average maps (AD vs CN)
#    - Difference map (AD - CN)
# ═══════════════════════════════════════════════════════

def _load_model_and_data(all_preds, y_true):
    """Load best-seed model and datasets."""
    import torch
    import yaml
    import importlib.util

    if "MLP Early" not in all_preds:
        print("  No MLP Early predictions found, skipping.")
        return None

    aucs = [roc_auc_score(y_true, p) for p in all_preds["MLP Early"]]
    best_seed = int(np.argmax(aucs))
    print(f"  Best seed: {best_seed} (AUC={aucs[best_seed]:.4f})")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Using device: {device}")

    _spec = importlib.util.spec_from_file_location("model", MLP_DIR / "model.py")
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)

    _spec_ds = importlib.util.spec_from_file_location("dataset", BASE / "multimodal_fusion" / "dataset.py")
    _ds_mod = importlib.util.module_from_spec(_spec_ds)
    _spec_ds.loader.exec_module(_ds_mod)

    with open(MLP_DIR / "config.yaml") as f:
        config = yaml.safe_load(f)

    preproc = config.get('preprocessing', {})
    target_shape = tuple(preproc.get('target_shape', [128, 128, 128]))
    tabular_features = config['data']['tabular_features']

    model_path = MLP_DIR / "results_early" / f"seed_{best_seed}" / "best_model.pth"
    if not model_path.exists():
        print(f"  No model found at {model_path}, skipping.")
        return None

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = _mod.EarlyFusionModel(
        pretrained=False,
        tabular_input_dim=len(tabular_features),
        tabular_hidden_dims=config['model']['tabular']['hidden_dims'],
        fusion_hidden_dims=config['model']['early_fusion']['hidden_dims'],
        num_classes=config['model']['early_fusion']['num_classes'],
        dropout=config['model']['early_fusion']['dropout'],
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    train_dataset = _ds_mod.MultiModalDataset(
        config['data']['train_csv'], tabular_features=tabular_features,
        target_shape=target_shape, augment=False, normalize_tabular=True,
        scaler=None, use_paper_preprocessing=preproc.get('use_paper_preprocessing', True),
        target_spacing=preproc.get('target_spacing', 1.75)
    )
    scaler = train_dataset.get_scaler()

    test_dataset = _ds_mod.MultiModalDataset(
        config['data']['test_csv'], tabular_features=tabular_features,
        target_shape=target_shape, augment=False, normalize_tabular=True,
        scaler=scaler, use_paper_preprocessing=preproc.get('use_paper_preprocessing', True),
        target_spacing=preproc.get('target_spacing', 1.75)
    )

    return {
        'model': model, 'device': device, 'test_dataset': test_dataset,
        'target_shape': target_shape, 'best_seed': best_seed,
    }


def _compute_ig(model, mri_gpu, tabular_gpu, device, n_steps=100):
    """Compute Integrated Gradients attribution for a single sample."""
    import torch
    from scipy.ndimage import gaussian_filter

    baseline_mri = torch.zeros_like(mri_gpu)
    baseline_tab = torch.zeros_like(tabular_gpu)

    ig_grads = torch.zeros_like(mri_gpu)
    for step in range(n_steps):
        alpha = step / n_steps
        interp_mri = (baseline_mri + alpha * (mri_gpu - baseline_mri)).requires_grad_(True)
        interp_tab = (baseline_tab + alpha * (tabular_gpu - baseline_tab)).requires_grad_(True)

        model.zero_grad()
        out = model(interp_mri, interp_tab)
        out[0, 1].backward()
        ig_grads += interp_mri.grad.detach()

    ig_attr = ((mri_gpu - baseline_mri) * ig_grads / n_steps).squeeze().cpu().numpy()
    ig_attr = np.abs(ig_attr)
    ig_attr = gaussian_filter(ig_attr, sigma=2.0)
    ig_attr = (ig_attr - ig_attr.min()) / (ig_attr.max() - ig_attr.min() + 1e-8)
    return ig_attr


def _plot_individual(mri, ig, label, prob, out_path):
    """Save a single patient figure: 3 slices (sagittal, coronal, axial) with IG overlay."""
    d, h, w = mri.shape
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    slice_names = ['Sagittal', 'Coronal', 'Axial']
    slices_mri = [mri[d // 2, :, :], mri[:, h // 2, :], mri[:, :, w // 2]]
    slices_ig = [ig[d // 2, :, :], ig[:, h // 2, :], ig[:, :, w // 2]]

    for col in range(3):
        axes[col].imshow(slices_mri[col].T, cmap='gray', origin='lower', aspect='auto')
        axes[col].imshow(slices_ig[col].T, cmap='hot', alpha=0.5, origin='lower', aspect='auto')
        axes[col].set_title(slice_names[col], fontsize=12, fontweight='bold')
        axes[col].axis('off')

    diag = 'AD' if label == 1 else 'CN'
    fig.suptitle(f'{diag} — p(AD)={prob:.3f}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()


def _plot_group_average(avg_map, mri_template, title, out_path):
    """Plot group-average attribution map overlaid on a template MRI."""
    d, h, w = mri_template.shape
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    slice_names = ['Sagittal', 'Coronal', 'Axial']
    slices_mri = [mri_template[d // 2, :, :], mri_template[:, h // 2, :], mri_template[:, :, w // 2]]
    slices_attr = [avg_map[d // 2, :, :], avg_map[:, h // 2, :], avg_map[:, :, w // 2]]

    for col in range(3):
        axes[col].imshow(slices_mri[col].T, cmap='gray', origin='lower', aspect='auto')
        im = axes[col].imshow(slices_attr[col].T, cmap='hot', alpha=0.55, origin='lower',
                              aspect='auto', vmin=0, vmax=1)
        axes[col].set_title(slice_names[col], fontsize=12, fontweight='bold')
        axes[col].axis('off')

    cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.04)
    cbar.set_label('Mean attribution', fontsize=10)
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()


def _plot_difference_map(diff_map, mri_template, best_seed, out_path):
    """Plot AD - CN difference map (diverging colormap)."""
    d, h, w = mri_template.shape
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    slice_names = ['Sagittal', 'Coronal', 'Axial']
    slices_mri = [mri_template[d // 2, :, :], mri_template[:, h // 2, :], mri_template[:, :, w // 2]]
    slices_diff = [diff_map[d // 2, :, :], diff_map[:, h // 2, :], diff_map[:, :, w // 2]]

    vmax = np.percentile(np.abs(diff_map), 99)
    for col in range(3):
        axes[col].imshow(slices_mri[col].T, cmap='gray', origin='lower', aspect='auto')
        im = axes[col].imshow(slices_diff[col].T, cmap='RdBu_r', alpha=0.6, origin='lower',
                              aspect='auto', vmin=-vmax, vmax=vmax)
        axes[col].set_title(slice_names[col], fontsize=12, fontweight='bold')
        axes[col].axis('off')

    cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.04)
    cbar.set_label('AD - CN attribution', fontsize=10)
    fig.suptitle(f'Differential Attribution Map (AD - CN)\n'
                 f'Red = more important for AD, Blue = more important for CN  (seed={best_seed})',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()


def _plot_summary_figure(ad_examples, cn_examples, avg_ad, avg_cn, diff_map, mri_template, best_seed, out_path):
    """Paper-ready summary: 2 AD + 2 CN individuals, group averages, difference map."""
    fig = plt.figure(figsize=(18, 20))
    # Layout: 6 rows x 3 cols
    # Row 0-1: AD individuals, Row 2-3: CN individuals, Row 4: group avg AD & CN, Row 5: difference
    gs = fig.add_gridspec(6, 3, hspace=0.35, wspace=0.05)

    d, h, w = mri_template.shape

    def _draw_row(row_idx, mri, attr, cmap, alpha, label_text, color):
        slices_mri = [mri[d // 2, :, :], mri[:, h // 2, :], mri[:, :, w // 2]]
        slices_attr = [attr[d // 2, :, :], attr[:, h // 2, :], attr[:, :, w // 2]]
        slice_names = ['Sagittal', 'Coronal', 'Axial']
        for col in range(3):
            ax = fig.add_subplot(gs[row_idx, col])
            ax.imshow(slices_mri[col].T, cmap='gray', origin='lower', aspect='auto')
            ax.imshow(slices_attr[col].T, cmap=cmap, alpha=alpha, origin='lower', aspect='auto')
            ax.axis('off')
            if col == 0:
                ax.set_ylabel(label_text, fontsize=10, fontweight='bold', color=color,
                              rotation=0, labelpad=70, verticalalignment='center')
            if row_idx == 0:
                ax.set_title(slice_names[col], fontsize=12, fontweight='bold')

    # AD individuals (rows 0-1)
    for i, ex in enumerate(ad_examples[:2]):
        _draw_row(i, ex['mri'], ex['ig'], 'hot', 0.5,
                  f"AD #{i+1}\np(AD)={ex['prob']:.2f}", 'darkred')

    # CN individuals (rows 2-3)
    for i, ex in enumerate(cn_examples[:2]):
        _draw_row(2 + i, ex['mri'], ex['ig'], 'hot', 0.5,
                  f"CN #{i+1}\np(AD)={ex['prob']:.2f}", 'darkblue')

    # Group average AD (row 4, left half) + CN (row 4, right... actually use full row for each)
    # Row 4: Group avg AD
    _draw_row(4, mri_template, avg_ad, 'hot', 0.55, 'Group avg\nAD (n=50)', 'darkred')

    # Row 5: Difference map
    vmax = np.percentile(np.abs(diff_map), 99)
    slice_names = ['Sagittal', 'Coronal', 'Axial']
    slices_mri = [mri_template[d // 2, :, :], mri_template[:, h // 2, :], mri_template[:, :, w // 2]]
    slices_diff = [diff_map[d // 2, :, :], diff_map[:, h // 2, :], diff_map[:, :, w // 2]]
    for col in range(3):
        ax = fig.add_subplot(gs[5, col])
        ax.imshow(slices_mri[col].T, cmap='gray', origin='lower', aspect='auto')
        im = ax.imshow(slices_diff[col].T, cmap='RdBu_r', alpha=0.6, origin='lower',
                       aspect='auto', vmin=-vmax, vmax=vmax)
        ax.axis('off')
        if col == 0:
            ax.set_ylabel('AD - CN\ndifference', fontsize=10, fontweight='bold', color='purple',
                          rotation=0, labelpad=70, verticalalignment='center')

    fig.suptitle(f'Integrated Gradients — ResNet3D Early Fusion (seed={best_seed})\n'
                 'Individual examples, group averages, and differential attribution',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()


def interpretability_analysis(all_preds, y_true):
    """Full interpretability: individual IG maps + group averages + difference map."""
    import torch
    from scipy.ndimage import gaussian_filter

    print("\n" + "=" * 60)
    print("Interpretability: Integrated Gradients (best seed)")
    print("=" * 60)

    ctx = _load_model_and_data(all_preds, y_true)
    if ctx is None:
        return

    model = ctx['model']
    device = ctx['device']
    test_dataset = ctx['test_dataset']
    best_seed = ctx['best_seed']

    # Create output directory
    ig_dir = REPORT_DIR / "interpretability"
    ig_dir.mkdir(exist_ok=True)
    (ig_dir / "individual").mkdir(exist_ok=True)

    # ── Phase 1: Classify all test samples ──
    print("\n  Phase 1: Classifying all test samples...")
    sample_info = []
    for idx in range(len(test_dataset)):
        mri, tabular, label_val = test_dataset[idx]
        mri_gpu = mri.unsqueeze(0).to(device)
        tabular_gpu = tabular.unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(mri_gpu, tabular_gpu)
        prob = torch.softmax(output, dim=1)[0, 1].item()
        pred = output.argmax(dim=1).item()
        sample_info.append({'idx': idx, 'label': label_val, 'pred': pred, 'prob': prob})

    # Sort by confidence for picking good examples
    ad_correct = sorted([s for s in sample_info if s['label'] == 1 and s['pred'] == 1],
                        key=lambda x: x['prob'], reverse=True)
    cn_correct = sorted([s for s in sample_info if s['label'] == 0 and s['pred'] == 0],
                        key=lambda x: x['prob'])

    n_individual = 5
    n_group = 50

    print(f"  Found {len(ad_correct)} correct AD, {len(cn_correct)} correct CN")
    print(f"  Will compute IG for: {n_individual} AD + {n_individual} CN individual")
    print(f"                       {min(n_group, len(ad_correct))} AD + {min(n_group, len(cn_correct))} CN for group average")

    # ── Phase 2: Individual examples (5 AD + 5 CN) ──
    print("\n  Phase 2: Individual Integrated Gradients (5 AD + 5 CN)...")
    ad_examples = []
    for i, s in enumerate(ad_correct[:n_individual]):
        print(f"    AD {i+1}/{n_individual} (idx={s['idx']}, p(AD)={s['prob']:.3f})...")
        mri, tabular, label_val = test_dataset[s['idx']]
        mri_gpu = mri.unsqueeze(0).to(device)
        tabular_gpu = tabular.unsqueeze(0).to(device)
        ig = _compute_ig(model, mri_gpu, tabular_gpu, device, n_steps=100)
        ex = {'mri': mri.squeeze().numpy(), 'ig': ig, 'label': label_val, 'prob': s['prob']}
        ad_examples.append(ex)
        _plot_individual(ex['mri'], ig, label_val, s['prob'],
                         ig_dir / "individual" / f"AD_{i+1:02d}.png")

    cn_examples = []
    for i, s in enumerate(cn_correct[:n_individual]):
        print(f"    CN {i+1}/{n_individual} (idx={s['idx']}, p(AD)={s['prob']:.3f})...")
        mri, tabular, label_val = test_dataset[s['idx']]
        mri_gpu = mri.unsqueeze(0).to(device)
        tabular_gpu = tabular.unsqueeze(0).to(device)
        ig = _compute_ig(model, mri_gpu, tabular_gpu, device, n_steps=100)
        ex = {'mri': mri.squeeze().numpy(), 'ig': ig, 'label': label_val, 'prob': s['prob']}
        cn_examples.append(ex)
        _plot_individual(ex['mri'], ig, label_val, s['prob'],
                         ig_dir / "individual" / f"CN_{i+1:02d}.png")

    print(f"  Saved {n_individual} AD + {n_individual} CN individual maps")

    # ── Phase 3: Group averages (50 AD + 50 CN) ──
    n_ad_group = min(n_group, len(ad_correct))
    n_cn_group = min(n_group, len(cn_correct))
    target_shape = ctx['target_shape']

    print(f"\n  Phase 3: Group average IG ({n_ad_group} AD + {n_cn_group} CN)...")
    # Use a reference MRI as template (first CN)
    mri_template = cn_examples[0]['mri']

    # Accumulate AD attributions
    ad_accum = np.zeros(target_shape, dtype=np.float64)
    for i, s in enumerate(ad_correct[:n_ad_group]):
        if i < n_individual:
            # Reuse already computed
            ad_accum += ad_examples[i]['ig']
        else:
            if (i + 1) % 10 == 0:
                print(f"    AD group: {i+1}/{n_ad_group}...")
            mri, tabular, label_val = test_dataset[s['idx']]
            mri_gpu = mri.unsqueeze(0).to(device)
            tabular_gpu = tabular.unsqueeze(0).to(device)
            ig = _compute_ig(model, mri_gpu, tabular_gpu, device, n_steps=50)
            ad_accum += ig

    avg_ad = ad_accum / n_ad_group
    avg_ad = (avg_ad - avg_ad.min()) / (avg_ad.max() - avg_ad.min() + 1e-8)

    # Accumulate CN attributions
    cn_accum = np.zeros(target_shape, dtype=np.float64)
    for i, s in enumerate(cn_correct[:n_cn_group]):
        if i < n_individual:
            cn_accum += cn_examples[i]['ig']
        else:
            if (i + 1) % 10 == 0:
                print(f"    CN group: {i+1}/{n_cn_group}...")
            mri, tabular, label_val = test_dataset[s['idx']]
            mri_gpu = mri.unsqueeze(0).to(device)
            tabular_gpu = tabular.unsqueeze(0).to(device)
            ig = _compute_ig(model, mri_gpu, tabular_gpu, device, n_steps=50)
            cn_accum += ig

    avg_cn = cn_accum / n_cn_group
    avg_cn = (avg_cn - avg_cn.min()) / (avg_cn.max() - avg_cn.min() + 1e-8)

    # Difference map (AD - CN) on raw accumulations before normalization
    diff_raw = (ad_accum / n_ad_group) - (cn_accum / n_cn_group)
    diff_map = gaussian_filter(diff_raw, sigma=2.0)

    # ── Phase 4: Save group plots ──
    print("\n  Phase 4: Saving group average and difference maps...")
    _plot_group_average(avg_ad, mri_template,
                        f'Group Average Attribution — AD patients (n={n_ad_group}, seed={best_seed})',
                        ig_dir / "group_average_AD.png")
    _plot_group_average(avg_cn, mri_template,
                        f'Group Average Attribution — CN patients (n={n_cn_group}, seed={best_seed})',
                        ig_dir / "group_average_CN.png")
    _plot_difference_map(diff_map, mri_template, best_seed,
                         ig_dir / "group_difference_AD_minus_CN.png")

    # Summary figure
    _plot_summary_figure(ad_examples, cn_examples, avg_ad, avg_cn, diff_map,
                         mri_template, best_seed,
                         ig_dir / "summary_figure.png")

    # Save raw numpy arrays for further analysis
    np.save(ig_dir / "group_avg_AD.npy", avg_ad)
    np.save(ig_dir / "group_avg_CN.npy", avg_cn)
    np.save(ig_dir / "group_diff_AD_minus_CN.npy", diff_map)

    print(f"\n  All interpretability outputs saved to {ig_dir}/")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multi-seed analysis')
    parser.add_argument('--gradcam', action='store_true', help='Include interpretability maps (Integrated Gradients + GradCAM, needs GPU)')
    args = parser.parse_args()

    print(f"ResNet3D Fusion - Multi-Seed Analysis ({EXPECTED_SEEDS} seeds)")
    print("=" * 60)

    # Load all seeds
    print("Loading predictions...")
    y_true, all_preds, seed_counts = load_all_seeds()
    if y_true is None:
        print("ERROR: No predictions found!")
        exit(1)

    # 1. Per-seed metrics & summary
    metrics_df = compute_per_seed_metrics(y_true, all_preds)
    generate_summary(metrics_df)

    # 2. Box plots
    plot_boxplots(metrics_df)

    # 3. DeLong test (on mean probabilities)
    mean_preds = run_delong_analysis(y_true, all_preds, seed_counts)

    # 4. ROC curves with std bands
    plot_roc_curves(y_true, all_preds)

    # 5. Confusion matrices
    plot_confusion_matrices(y_true, mean_preds)

    # 6. Interpretability (Integrated Gradients + GradCAM)
    if args.gradcam:
        interpretability_analysis(all_preds, y_true)
    else:
        print("\nSkipping interpretability (use --gradcam to include)")

    print("\n" + "=" * 60)
    print(f"All outputs saved to {REPORT_DIR}/")
    print("=" * 60)

#!/usr/bin/env python3
"""
Tabular data analysis for Alzheimer's datasets.

Usage:
    python analyze.py --config configs/adni.yaml
    python analyze.py --config configs/oasis.yaml
    python analyze.py --config configs/combined.yaml
"""

import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent


def load_config(config_path: str) -> dict:
    """Load YAML config file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_data(config: dict) -> pd.DataFrame:
    """Load dataset based on config"""
    if config.get('dataset') == 'combined':
        # Load and combine ADNI + OASIS
        adni_path = PROJECT_ROOT / config['adni_csv']
        oasis_path = PROJECT_ROOT / config['oasis_csv']

        df_adni = pd.read_csv(adni_path)
        df_oasis = pd.read_csv(oasis_path)

        # Standardize ADNI diagnosis to 3-class
        if 'CLASS_4' in df_adni.columns:
            df_adni['DX'] = df_adni['CLASS_4'].map({
                'CN': 'CN',
                'MCI_stable': 'MCI',
                'MCI_to_AD': 'MCI',
                'AD': 'AD'
            })
        df_adni['source'] = 'ADNI'
        df_oasis['source'] = 'OASIS'

        # Combine
        df = pd.concat([df_adni, df_oasis], ignore_index=True)
        config['diagnosis_col'] = 'DX'
        logger.info(f"Loaded combined: {len(df_adni)} ADNI + {len(df_oasis)} OASIS = {len(df)} total")
    else:
        input_path = PROJECT_ROOT / config['input_csv']
        df = pd.read_csv(input_path)
        logger.info(f"Loaded {len(df)} samples from {config['name']}")

    return df


def get_all_features(config: dict) -> list:
    """Get flat list of all features"""
    features = []
    for category, feature_list in config.get('features', {}).items():
        features.extend(feature_list)
    return features


def calculate_age(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate AGE if not present"""
    if 'AGE' not in df.columns and 'PTDOBYY' in df.columns:
        if 'EXAMDATE' in df.columns:
            df['EXAMDATE'] = pd.to_datetime(df['EXAMDATE'])
            df['AGE'] = df['EXAMDATE'].dt.year - df['PTDOBYY']
        else:
            df['AGE'] = 2010 - df['PTDOBYY']
    return df


def calculate_bmi(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate BMI if not present"""
    if 'BMI' not in df.columns:
        if 'VSWEIGHT' in df.columns and 'VSHEIGHT' in df.columns:
            # Check if height is in cm or inches
            avg_height = df['VSHEIGHT'].median()
            if avg_height > 100:  # cm
                df['BMI'] = df['VSWEIGHT'] / ((df['VSHEIGHT'] / 100) ** 2)
            else:  # inches
                df['BMI'] = (df['VSWEIGHT'] / (df['VSHEIGHT'] ** 2)) * 703
    return df


def generate_summary(df: pd.DataFrame, config: dict, output_dir: Path):
    """Generate dataset summary markdown"""
    diagnosis_col = config.get('diagnosis_col', 'DX')
    subject_col = config.get('subject_col', 'Subject')

    summary = f"# {config['name']} Dataset Summary\n\n"
    summary += f"## Overview\n\n"
    summary += f"- **Total samples**: {len(df):,}\n"

    if subject_col in df.columns:
        n_subjects = df[subject_col].nunique()
        avg_visits = len(df) / n_subjects
        summary += f"- **Unique subjects**: {n_subjects:,}\n"
        summary += f"- **Average visits per subject**: {avg_visits:.2f}\n"

    summary += f"\n## Class Distribution\n\n"
    summary += "| Class | Count | Percentage |\n"
    summary += "|-------|-------|------------|\n"

    if diagnosis_col in df.columns:
        class_counts = df[diagnosis_col].value_counts()
        for cls, count in class_counts.items():
            pct = count / len(df) * 100
            summary += f"| {cls} | {count:,} | {pct:.1f}% |\n"

    # Feature availability
    all_features = get_all_features(config)
    available = [f for f in all_features if f in df.columns]
    missing = [f for f in all_features if f not in df.columns]

    summary += f"\n## Feature Availability\n\n"
    summary += f"- **Available features**: {len(available)}/{len(all_features)}\n"
    if missing:
        summary += f"- **Missing features**: {', '.join(missing)}\n"

    # Missing values per feature
    summary += f"\n## Missing Values\n\n"
    summary += "| Feature | Missing | Percentage |\n"
    summary += "|---------|---------|------------|\n"

    for feature in available:
        n_missing = df[feature].isna().sum()
        pct_missing = n_missing / len(df) * 100
        if n_missing > 0:
            summary += f"| {feature} | {n_missing:,} | {pct_missing:.1f}% |\n"

    # Save summary
    with open(output_dir / 'dataset_summary.md', 'w') as f:
        f.write(summary)

    logger.info(f"Saved dataset summary to {output_dir / 'dataset_summary.md'}")


def plot_class_distribution(df: pd.DataFrame, config: dict, output_dir: Path):
    """Plot class distribution"""
    diagnosis_col = config.get('diagnosis_col', 'DX')
    subject_col = config.get('subject_col', 'Subject')

    if diagnosis_col not in df.columns:
        logger.warning(f"Diagnosis column '{diagnosis_col}' not found")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Sample-level distribution
    class_counts = df[diagnosis_col].value_counts()
    colors = plt.cm.Set2(np.linspace(0, 1, len(class_counts)))

    ax1 = axes[0]
    bars = ax1.bar(class_counts.index, class_counts.values, color=colors)
    ax1.set_xlabel('Diagnosis')
    ax1.set_ylabel('Number of Samples')
    ax1.set_title('Sample Distribution')
    for bar, count in zip(bars, class_counts.values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                f'{count:,}', ha='center', va='bottom', fontsize=10)

    # Subject-level distribution (if available)
    ax2 = axes[1]
    if subject_col in df.columns:
        subject_classes = df.groupby(subject_col)[diagnosis_col].first().value_counts()
        bars = ax2.bar(subject_classes.index, subject_classes.values, color=colors)
        ax2.set_xlabel('Diagnosis')
        ax2.set_ylabel('Number of Subjects')
        ax2.set_title('Subject Distribution')
        for bar, count in zip(bars, subject_classes.values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    f'{count:,}', ha='center', va='bottom', fontsize=10)
    else:
        ax2.text(0.5, 0.5, 'Subject column not available',
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Subject Distribution')

    plt.suptitle(f"{config['name']} - Class Distribution", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'class_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved class distribution plot")


def plot_feature_distributions(df: pd.DataFrame, config: dict, output_dir: Path):
    """Plot feature distributions by class"""
    diagnosis_col = config.get('diagnosis_col', 'DX')

    if diagnosis_col not in df.columns:
        return

    all_features = get_all_features(config)
    available_features = [f for f in all_features if f in df.columns]

    if not available_features:
        logger.warning("No features available for distribution plots")
        return

    # Select key numeric features
    numeric_features = []
    for f in available_features:
        if df[f].dtype in ['int64', 'float64'] and df[f].nunique() > 5:
            numeric_features.append(f)

    if len(numeric_features) == 0:
        return

    n_features = min(12, len(numeric_features))
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten() if n_features > 1 else [axes]

    classes = df[diagnosis_col].dropna().unique()
    colors = plt.cm.Set2(np.linspace(0, 1, len(classes)))

    for idx, feature in enumerate(numeric_features[:n_features]):
        ax = axes[idx]

        for i, cls in enumerate(sorted(classes)):
            data = df[df[diagnosis_col] == cls][feature].dropna()
            if len(data) > 0:
                ax.hist(data, bins=30, alpha=0.5, label=cls, color=colors[i], density=True)

        ax.set_xlabel(feature)
        ax.set_ylabel('Density')
        ax.set_title(feature)
        ax.legend(fontsize=8)

    # Hide empty subplots
    for idx in range(n_features, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(f"{config['name']} - Feature Distributions by Class", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved feature distributions plot")


def plot_missing_values(df: pd.DataFrame, config: dict, output_dir: Path):
    """Plot missing values heatmap"""
    all_features = get_all_features(config)
    available_features = [f for f in all_features if f in df.columns]

    if not available_features:
        return

    # Calculate missing percentage
    missing_pct = df[available_features].isna().sum() / len(df) * 100
    missing_pct = missing_pct.sort_values(ascending=False)

    # Only show features with some missing values
    missing_pct = missing_pct[missing_pct > 0]

    if len(missing_pct) == 0:
        logger.info("No missing values found")
        return

    fig, ax = plt.subplots(figsize=(10, max(6, len(missing_pct) * 0.4)))

    colors = plt.cm.Reds(missing_pct.values / 100)
    bars = ax.barh(missing_pct.index, missing_pct.values, color=colors)

    ax.set_xlabel('Missing (%)')
    ax.set_title(f"{config['name']} - Missing Values by Feature")
    ax.set_xlim(0, 100)

    for bar, pct in zip(bars, missing_pct.values):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f'{pct:.1f}%', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / 'missing_values.png', dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved missing values plot")


def plot_correlation_matrix(df: pd.DataFrame, config: dict, output_dir: Path):
    """Plot correlation matrix"""
    all_features = get_all_features(config)
    available_features = [f for f in all_features if f in df.columns]

    # Select numeric features
    numeric_features = [f for f in available_features
                       if df[f].dtype in ['int64', 'float64']]

    if len(numeric_features) < 2:
        return

    # Compute correlation
    corr_matrix = df[numeric_features].corr()

    fig, ax = plt.subplots(figsize=(12, 10))

    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
                cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                square=True, ax=ax, annot_kws={'size': 8})

    ax.set_title(f"{config['name']} - Feature Correlation Matrix")

    plt.tight_layout()
    plt.savefig(output_dir / 'correlation_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved correlation matrix")


def generate_feature_stats(df: pd.DataFrame, config: dict, output_dir: Path):
    """Generate feature statistics by class"""
    diagnosis_col = config.get('diagnosis_col', 'DX')
    all_features = get_all_features(config)
    available_features = [f for f in all_features if f in df.columns]

    if diagnosis_col not in df.columns or not available_features:
        return

    stats_data = []
    classes = sorted(df[diagnosis_col].dropna().unique())

    for feature in available_features:
        if df[feature].dtype not in ['int64', 'float64']:
            continue

        row = {'feature': feature}

        # Overall stats
        row['overall_mean'] = df[feature].mean()
        row['overall_std'] = df[feature].std()
        row['missing_pct'] = df[feature].isna().sum() / len(df) * 100

        # Per-class stats
        class_data = []
        for cls in classes:
            data = df[df[diagnosis_col] == cls][feature].dropna()
            row[f'{cls}_mean'] = data.mean()
            row[f'{cls}_std'] = data.std()
            row[f'{cls}_n'] = len(data)
            class_data.append(data)

        # ANOVA p-value (if >2 classes)
        if len(classes) > 2:
            valid_data = [d.values for d in class_data if len(d) > 1]
            if len(valid_data) >= 2:
                try:
                    _, p_value = stats.f_oneway(*valid_data)
                    row['anova_pvalue'] = p_value
                except:
                    row['anova_pvalue'] = np.nan
        elif len(classes) == 2:
            # t-test for 2 classes
            if len(class_data[0]) > 1 and len(class_data[1]) > 1:
                try:
                    _, p_value = stats.ttest_ind(class_data[0], class_data[1])
                    row['ttest_pvalue'] = p_value
                except:
                    row['ttest_pvalue'] = np.nan

        stats_data.append(row)

    stats_df = pd.DataFrame(stats_data)
    stats_df.to_csv(output_dir / 'feature_stats.csv', index=False)

    logger.info(f"Saved feature statistics to {output_dir / 'feature_stats.csv'}")


def plot_boxplots(df: pd.DataFrame, config: dict, output_dir: Path):
    """Plot boxplots for key features by class"""
    diagnosis_col = config.get('diagnosis_col', 'DX')

    if diagnosis_col not in df.columns:
        return

    # Key cognitive features
    key_features = ['AGE', 'TRAASCOR', 'TRABSCOR', 'CATANIMSC', 'BNTTOTAL', 'DSPANFOR']
    available = [f for f in key_features if f in df.columns]

    if len(available) < 2:
        return

    n_features = len(available)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
    axes = axes.flatten()

    for idx, feature in enumerate(available):
        ax = axes[idx]
        df_plot = df[[diagnosis_col, feature]].dropna()

        sns.boxplot(data=df_plot, x=diagnosis_col, y=feature, ax=ax, palette='Set2')
        ax.set_title(feature)
        ax.set_xlabel('')

    for idx in range(len(available), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(f"{config['name']} - Feature Boxplots by Class", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_boxplots.png', dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved feature boxplots")


def _load_dxsum_transitions(dxsum_path: Path):
    """Load diagnosis transitions from ADNI dxsum.csv"""
    from collections import defaultdict

    df = pd.read_csv(dxsum_path)
    dx_map = {1: 'CN', 2: 'MCI', 3: 'AD'}

    # Filter to valid diagnoses and sort by patient/date
    df_valid = df[df['DIAGNOSIS'].isin([1, 2, 3])].copy()
    df_valid['DX_NAME'] = df_valid['DIAGNOSIS'].map(dx_map)
    df_valid = df_valid.sort_values(['PTID', 'EXAMDATE'])

    # Get first and last diagnosis per patient
    transitions = df_valid.groupby('PTID').agg({
        'DX_NAME': ['first', 'last'],
        'EXAMDATE': 'count'
    }).reset_index()
    transitions.columns = ['PTID', 'FIRST_DX', 'LAST_DX', 'N_VISITS']

    # Only patients with >1 visit
    multi_visit = transitions[transitions['N_VISITS'] > 1]

    first_dx = multi_visit['FIRST_DX']
    last_dx = multi_visit['LAST_DX']
    n_subjects = len(multi_visit)

    # Count visit-to-visit transitions
    visit_transitions = defaultdict(int)
    for ptid, group in df_valid.groupby('PTID'):
        dxs = group['DX_NAME'].tolist()
        for i in range(len(dxs) - 1):
            if dxs[i] != dxs[i+1]:
                visit_transitions[(dxs[i], dxs[i+1])] += 1

    return first_dx, last_dx, n_subjects, dict(visit_transitions)


def _generate_dxsum_transition_summary(first_dx, last_dx, n_subjects, visit_transitions, config, output_dir):
    """Generate summary and plots for dxsum transitions"""
    class_order = ['CN', 'MCI', 'AD']
    colors = {'CN': '#66c2a5', 'MCI': '#fc8d62', 'AD': '#8da0cb'}

    # Create transition dataframe
    trans_df = pd.DataFrame({'first': first_dx.values, 'last': last_dx.values})

    # Generate markdown summary
    summary = f"\n## Diagnosis Transitions (from dxsum.csv)\n\n"
    summary += f"**Total subjects with multiple visits**: {n_subjects}\n\n"

    # First-to-last transitions by starting diagnosis
    summary += "### First → Last Diagnosis\n\n"
    for start_dx in class_order:
        subset = trans_df[trans_df['first'] == start_dx]
        if len(subset) == 0:
            continue

        stable = (subset['last'] == start_dx).sum()
        summary += f"#### {start_dx} ({len(subset)} patients)\n"
        summary += f"- **Stable ({start_dx} → {start_dx})**: {stable} ({stable/len(subset)*100:.1f}%)\n"

        for end_dx in class_order:
            if end_dx != start_dx:
                count = (subset['last'] == end_dx).sum()
                if count > 0:
                    pct = count / len(subset) * 100
                    summary += f"- **{start_dx} → {end_dx}**: {count} ({pct:.1f}%)\n"
        summary += "\n"

    # Visit-by-visit transitions
    summary += "### Visit-to-Visit Transitions (all occurrences)\n\n"
    summary += "**Progression (expected):**\n"
    summary += f"- CN → MCI: {visit_transitions.get(('CN', 'MCI'), 0)} occurrences\n"
    summary += f"- CN → AD: {visit_transitions.get(('CN', 'AD'), 0)} occurrences\n"
    summary += f"- MCI → AD: {visit_transitions.get(('MCI', 'AD'), 0)} occurrences\n\n"

    summary += "**Reversion (rare but possible):**\n"
    summary += f"- MCI → CN: {visit_transitions.get(('MCI', 'CN'), 0)} occurrences\n\n"

    summary += "**Clinically unlikely:**\n"
    summary += f"- AD → CN: {visit_transitions.get(('AD', 'CN'), 0)} occurrences\n"
    summary += f"- AD → MCI: {visit_transitions.get(('AD', 'MCI'), 0)} occurrences\n\n"

    # Create visualization - transition heatmap
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Transition matrix heatmap (first → last)
    ax1 = axes[0]
    matrix = np.zeros((3, 3))
    for i, from_dx in enumerate(class_order):
        subset = trans_df[trans_df['first'] == from_dx]
        total = len(subset)
        for j, to_dx in enumerate(class_order):
            count = (subset['last'] == to_dx).sum()
            matrix[i, j] = count / total * 100 if total > 0 else 0

    im = ax1.imshow(matrix, cmap='Blues', aspect='auto', vmin=0, vmax=100)
    ax1.set_xticks(range(3))
    ax1.set_yticks(range(3))
    ax1.set_xticklabels(class_order, fontsize=11)
    ax1.set_yticklabels(class_order, fontsize=11)
    ax1.set_xlabel('Final Diagnosis', fontsize=12)
    ax1.set_ylabel('Initial Diagnosis', fontsize=12)
    ax1.set_title(f'Transition Matrix (%)\n{n_subjects} subjects', fontsize=12, fontweight='bold')

    # Add text annotations
    for i in range(3):
        subset = trans_df[trans_df['first'] == class_order[i]]
        for j in range(3):
            count = (subset['last'] == class_order[j]).sum()
            pct = matrix[i, j]
            color = 'white' if pct > 50 else 'black'
            ax1.text(j, i, f'{count}\n({pct:.0f}%)', ha='center', va='center',
                    fontsize=10, color=color, fontweight='bold')

    # Right: Progression bar chart (cleaner)
    ax2 = axes[1]

    # Group transitions by type
    progressions = [
        ('CN → MCI', visit_transitions.get(('CN', 'MCI'), 0), '#3498db'),
        ('CN → AD', visit_transitions.get(('CN', 'AD'), 0), '#2980b9'),
        ('MCI → AD', visit_transitions.get(('MCI', 'AD'), 0), '#1a5276'),
    ]
    reversions = [
        ('MCI → CN', visit_transitions.get(('MCI', 'CN'), 0), '#27ae60'),
        ('AD → MCI', visit_transitions.get(('AD', 'MCI'), 0), '#f39c12'),
        ('AD → CN', visit_transitions.get(('AD', 'CN'), 0), '#e74c3c'),
    ]

    all_trans = progressions + reversions
    labels = [t[0] for t in all_trans]
    values = [t[1] for t in all_trans]
    bar_colors = [t[2] for t in all_trans]

    y_pos = np.arange(len(labels))
    bars = ax2.barh(y_pos, values, color=bar_colors, alpha=0.85, height=0.7)

    # Add count labels
    for bar, val in zip(bars, values):
        ax2.text(bar.get_width() + max(values)*0.02, bar.get_y() + bar.get_height()/2,
                f'{int(val)}', ha='left', va='center', fontsize=10, fontweight='bold')

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels, fontsize=11)
    ax2.set_xlabel('Number of Transitions', fontsize=12)
    ax2.set_title('Visit-to-Visit Changes', fontsize=12, fontweight='bold')
    ax2.set_xlim(0, max(values) * 1.15)

    # Add separator line between progressions and reversions
    ax2.axhline(y=2.5, color='gray', linestyle='--', alpha=0.5)
    ax2.text(max(values)*0.5, 1, 'Progression ↓', ha='center', va='center',
            fontsize=9, color='#2c3e50', style='italic')
    ax2.text(max(values)*0.5, 4, 'Reversion ↑', ha='center', va='center',
            fontsize=9, color='#2c3e50', style='italic')

    ax2.invert_yaxis()
    ax2.grid(axis='x', alpha=0.3)

    plt.suptitle(f"{config['name']} - Disease Progression", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'transition_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Save CSV
    transitions_data = []
    for start_dx in class_order:
        subset = trans_df[trans_df['first'] == start_dx]
        for end_dx in class_order:
            count = (subset['last'] == end_dx).sum()
            pct = count / len(subset) * 100 if len(subset) > 0 else 0
            transitions_data.append({
                'From': start_dx,
                'To': end_dx,
                'Count': count,
                'Percentage': pct
            })
    pd.DataFrame(transitions_data).to_csv(output_dir / 'transition_matrix.csv', index=False)

    # Save visit transitions CSV
    visit_trans_data = [{'From': k[0], 'To': k[1], 'Count': v} for k, v in visit_transitions.items()]
    pd.DataFrame(visit_trans_data).to_csv(output_dir / 'visit_transitions.csv', index=False)

    # Append to summary
    with open(output_dir / 'dataset_summary.md', 'a') as f:
        f.write(summary)

    logger.info(f"Saved transition matrix (using dxsum.csv)")


def plot_transition_matrix(df: pd.DataFrame, config: dict, output_dir: Path):
    """Plot diagnosis transition matrix for longitudinal data"""
    diagnosis_col = config.get('diagnosis_col', 'DX')
    subject_col = config.get('subject_col', 'Subject')

    # Check if dxsum.csv is configured for ADNI (real per-visit diagnoses)
    dxsum_path = config.get('dxsum_csv')
    if dxsum_path:
        dxsum_file = PROJECT_ROOT / dxsum_path
        if dxsum_file.exists():
            logger.info(f"Using dxsum.csv for real per-visit diagnosis transitions")
            first_dx, last_dx, n_subjects, visit_transitions = _load_dxsum_transitions(dxsum_file)

            # Generate summary with visit-by-visit transitions
            _generate_dxsum_transition_summary(
                first_dx, last_dx, n_subjects, visit_transitions, config, output_dir
            )
            return

    if subject_col not in df.columns:
        logger.warning("Cannot create transition matrix: missing subject column")
        return

    # Check if ADNI with BL_DX/LAST_DX columns (baseline -> final diagnosis)
    if 'BL_DX' in df.columns and 'LAST_DX' in df.columns:
        logger.info("Using BL_DX -> LAST_DX for ADNI transitions")
        dx_map = {1: 'CN', 2: 'MCI', 3: 'AD'}

        subject_dx = df.groupby(subject_col).agg({
            'BL_DX': 'first',
            'LAST_DX': 'first'
        }).reset_index()

        first_dx = subject_dx['BL_DX'].map(dx_map)
        last_dx = subject_dx['LAST_DX'].map(dx_map)
        n_subjects = len(subject_dx)
    else:
        if diagnosis_col not in df.columns:
            logger.warning("Cannot create transition matrix: missing diagnosis column")
            return

        visits_per_subject = df.groupby(subject_col).size()
        subjects_with_multiple = (visits_per_subject > 1).sum()

        if subjects_with_multiple < 10:
            logger.info("Not enough longitudinal data for transition matrix")
            return

        time_col = None
        for col in ['days_to_visit', 'EXAMDATE', 'VISDATE', 'visit']:
            if col in df.columns:
                time_col = col
                break

        if time_col:
            df_sorted = df.sort_values([subject_col, time_col])
        else:
            df_sorted = df.sort_values(subject_col)

        first_dx = df_sorted.groupby(subject_col)[diagnosis_col].first()
        last_dx = df_sorted.groupby(subject_col)[diagnosis_col].last()

        multi_visit_subjects = visits_per_subject[visits_per_subject > 1].index
        first_dx = first_dx.loc[multi_visit_subjects]
        last_dx = last_dx.loc[multi_visit_subjects]
        n_subjects = len(multi_visit_subjects)

    # Create transition dataframe
    trans_df = pd.DataFrame({'first': first_dx.values, 'last': last_dx.values})

    # Count visit-to-visit transitions for datasets with per-visit data
    from collections import defaultdict
    visit_transitions = defaultdict(int)

    if 'BL_DX' not in df.columns:  # OASIS-style with per-visit DX
        time_col = None
        for col in ['days_to_visit', 'EXAMDATE', 'VISDATE', 'visit']:
            if col in df.columns:
                time_col = col
                break

        if time_col:
            df_sorted = df.sort_values([subject_col, time_col])
        else:
            df_sorted = df.sort_values(subject_col)

        for subj, group in df_sorted.groupby(subject_col):
            dxs = group[diagnosis_col].tolist()
            for i in range(len(dxs) - 1):
                if pd.notna(dxs[i]) and pd.notna(dxs[i+1]) and dxs[i] != dxs[i+1]:
                    visit_transitions[(str(dxs[i]), str(dxs[i+1]))] += 1

    # Count transitions by starting diagnosis
    summary = f"\n## Diagnosis Transitions\n\n"
    summary += f"**Total subjects**: {n_subjects}\n\n"

    # Define order
    class_order = ['CN', 'MCI', 'AD']

    for start_dx in class_order:
        subset = trans_df[trans_df['first'] == start_dx]
        if len(subset) == 0:
            continue

        stable = (subset['last'] == start_dx).sum()
        summary += f"### {start_dx} ({len(subset)} patients)\n"
        summary += f"- **Stable ({start_dx} → {start_dx})**: {stable} ({stable/len(subset)*100:.1f}%)\n"

        for end_dx in class_order:
            if end_dx != start_dx:
                count = (subset['last'] == end_dx).sum()
                if count > 0:
                    pct = count / len(subset) * 100
                    summary += f"- **{start_dx} → {end_dx}**: {count} ({pct:.1f}%)\n"

        summary += "\n"

    # Create visualization - heatmap style (same as dxsum)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Transition matrix heatmap (first → last)
    ax1 = axes[0]
    matrix = np.zeros((3, 3))
    for i, from_dx in enumerate(class_order):
        subset = trans_df[trans_df['first'] == from_dx]
        total = len(subset)
        for j, to_dx in enumerate(class_order):
            count = (subset['last'] == to_dx).sum()
            matrix[i, j] = count / total * 100 if total > 0 else 0

    im = ax1.imshow(matrix, cmap='Blues', aspect='auto', vmin=0, vmax=100)
    ax1.set_xticks(range(3))
    ax1.set_yticks(range(3))
    ax1.set_xticklabels(class_order, fontsize=11)
    ax1.set_yticklabels(class_order, fontsize=11)
    ax1.set_xlabel('Final Diagnosis', fontsize=12)
    ax1.set_ylabel('Initial Diagnosis', fontsize=12)
    ax1.set_title(f'Transition Matrix (%)\n{n_subjects} subjects', fontsize=12, fontweight='bold')

    # Add text annotations
    for i in range(3):
        subset = trans_df[trans_df['first'] == class_order[i]]
        for j in range(3):
            count = (subset['last'] == class_order[j]).sum()
            pct = matrix[i, j]
            color = 'white' if pct > 50 else 'black'
            ax1.text(j, i, f'{count}\n({pct:.0f}%)', ha='center', va='center',
                    fontsize=10, color=color, fontweight='bold')

    # Right: Progression bar chart
    ax2 = axes[1]

    # Group transitions by type
    progressions = [
        ('CN → MCI', visit_transitions.get(('CN', 'MCI'), 0), '#3498db'),
        ('CN → AD', visit_transitions.get(('CN', 'AD'), 0), '#2980b9'),
        ('MCI → AD', visit_transitions.get(('MCI', 'AD'), 0), '#1a5276'),
    ]
    reversions = [
        ('MCI → CN', visit_transitions.get(('MCI', 'CN'), 0), '#27ae60'),
        ('AD → MCI', visit_transitions.get(('AD', 'MCI'), 0), '#f39c12'),
        ('AD → CN', visit_transitions.get(('AD', 'CN'), 0), '#e74c3c'),
    ]

    all_trans = progressions + reversions
    labels = [t[0] for t in all_trans]
    values = [t[1] for t in all_trans]
    bar_colors = [t[2] for t in all_trans]

    y_pos = np.arange(len(labels))
    bars = ax2.barh(y_pos, values, color=bar_colors, alpha=0.85, height=0.7)

    # Add count labels
    max_val = max(values) if max(values) > 0 else 1
    for bar, val in zip(bars, values):
        ax2.text(bar.get_width() + max_val*0.02, bar.get_y() + bar.get_height()/2,
                f'{int(val)}', ha='left', va='center', fontsize=10, fontweight='bold')

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels, fontsize=11)
    ax2.set_xlabel('Number of Transitions', fontsize=12)
    ax2.set_title('Visit-to-Visit Changes', fontsize=12, fontweight='bold')
    ax2.set_xlim(0, max_val * 1.15)

    # Add separator line between progressions and reversions
    ax2.axhline(y=2.5, color='gray', linestyle='--', alpha=0.5)
    ax2.text(max_val*0.5, 1, 'Progression ↓', ha='center', va='center',
            fontsize=9, color='#2c3e50', style='italic')
    ax2.text(max_val*0.5, 4, 'Reversion ↑', ha='center', va='center',
            fontsize=9, color='#2c3e50', style='italic')

    ax2.invert_yaxis()
    ax2.grid(axis='x', alpha=0.3)

    plt.suptitle(f"{config['name']} - Disease Progression", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'transition_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Save CSV
    transitions_data = []
    for start_dx in class_order:
        subset = trans_df[trans_df['first'] == start_dx]
        for end_dx in class_order:
            count = (subset['last'] == end_dx).sum()
            pct = count / len(subset) * 100 if len(subset) > 0 else 0
            transitions_data.append({
                'From': start_dx,
                'To': end_dx,
                'Count': count,
                'Percentage': pct
            })
    pd.DataFrame(transitions_data).to_csv(output_dir / 'transition_matrix.csv', index=False)

    # Save visit transitions CSV
    if visit_transitions:
        visit_trans_data = [{'From': k[0], 'To': k[1], 'Count': v} for k, v in visit_transitions.items()]
        pd.DataFrame(visit_trans_data).to_csv(output_dir / 'visit_transitions.csv', index=False)

    # Append to summary
    with open(output_dir / 'dataset_summary.md', 'a') as f:
        f.write(summary)

    logger.info(f"Saved transition matrix")


def main():
    parser = argparse.ArgumentParser(description='Tabular data analysis')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    logger.info(f"Analyzing {config['name']} dataset")

    # Create output directory
    output_dir = PROJECT_ROOT / config['output_dir']
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df = load_data(config)

    # Preprocess
    df = calculate_age(df)
    df = calculate_bmi(df)

    # Generate analyses
    logger.info("Generating dataset summary...")
    generate_summary(df, config, output_dir)

    logger.info("Plotting class distribution...")
    plot_class_distribution(df, config, output_dir)

    logger.info("Plotting feature distributions...")
    plot_feature_distributions(df, config, output_dir)

    logger.info("Plotting missing values...")
    plot_missing_values(df, config, output_dir)

    logger.info("Plotting correlation matrix...")
    plot_correlation_matrix(df, config, output_dir)

    logger.info("Plotting boxplots...")
    plot_boxplots(df, config, output_dir)

    logger.info("Generating feature statistics...")
    generate_feature_stats(df, config, output_dir)

    logger.info("Plotting transition matrix...")
    plot_transition_matrix(df, config, output_dir)

    logger.info(f"\nAll analyses saved to {output_dir}")


if __name__ == '__main__':
    main()
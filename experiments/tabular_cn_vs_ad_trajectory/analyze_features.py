#!/usr/bin/env python3
"""
Feature Space Analysis: UMAP visualization and SHAP explanations

Usage:
    python analyze_features.py --results-dir results
"""

import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_data(results_dir: Path):
    """Load predictions and data"""
    test_df = pd.read_csv(results_dir / 'test_predictions.csv')
    mcis_df = pd.read_csv(results_dir / 'mcis_predictions.csv')

    with open(results_dir / 'feature_names.json', 'r') as f:
        feature_names = json.load(f)

    scaler = joblib.load(results_dir / 'scaler.pkl')

    return test_df, mcis_df, feature_names, scaler


def plot_umap(results_dir: Path, n_neighbors: int = 15, min_dist: float = 0.1):
    """UMAP visualization of feature space"""
    try:
        import umap
    except ImportError:
        logger.error("UMAP not installed. Run: pip install umap-learn")
        return

    test_df, mcis_df, feature_names, scaler = load_data(results_dir)

    # Get features
    X_test = test_df[feature_names].values
    X_mcis = mcis_df[feature_names].values

    # Combine for UMAP
    X_all = np.vstack([X_test, X_mcis])
    X_all_scaled = scaler.transform(X_all)

    # Create labels for visualization
    test_labels = test_df['label'].values
    groups = []
    for i, l in enumerate(test_labels):
        if l == 0:
            groups.append('CN (test)')
        else:
            # Check if MCIc or AD from group column if available
            if 'group' in test_df.columns:
                g = test_df.iloc[i]['group']
                groups.append(f'{g} (test)')
            else:
                groups.append('AD-traj (test)')

    groups.extend(['MCIs (holdout)'] * len(mcis_df))

    # Fit UMAP
    logger.info("Fitting UMAP...")
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
    embedding = reducer.fit_transform(X_all_scaled)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Color by group
    colors = {'CN (test)': 'green', 'MCIc (test)': 'orange', 'AD (test)': 'red',
              'AD-traj (test)': 'red', 'MCIs (holdout)': 'blue'}

    for group in set(groups):
        mask = np.array(groups) == group
        color = colors.get(group, 'gray')
        axes[0].scatter(embedding[mask, 0], embedding[mask, 1],
                       c=color, label=group, alpha=0.6, s=30)

    axes[0].set_xlabel('UMAP 1')
    axes[0].set_ylabel('UMAP 2')
    axes[0].set_title('Feature Space: All Groups')
    axes[0].legend()

    # Color by AD probability for MCIs
    test_size = len(test_df)
    mcis_proba = mcis_df['AD_trajectory_prob_mean'].values

    # Test set in gray
    axes[1].scatter(embedding[:test_size, 0], embedding[:test_size, 1],
                   c='lightgray', alpha=0.3, s=20, label='Test set')

    # MCIs colored by probability
    scatter = axes[1].scatter(embedding[test_size:, 0], embedding[test_size:, 1],
                              c=mcis_proba, cmap='RdYlGn_r', alpha=0.8, s=40)
    plt.colorbar(scatter, ax=axes[1], label='AD-trajectory Probability')

    axes[1].set_xlabel('UMAP 1')
    axes[1].set_ylabel('UMAP 2')
    axes[1].set_title('MCIs Colored by AD-trajectory Probability')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(results_dir / 'umap_visualization.png', dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"UMAP plot saved to {results_dir / 'umap_visualization.png'}")


def plot_shap(results_dir: Path, max_samples: int = 500):
    """SHAP analysis for model interpretability"""
    try:
        import shap
        import xgboost as xgb
    except ImportError:
        logger.error("SHAP not installed. Run: pip install shap")
        return

    test_df, mcis_df, feature_names, scaler = load_data(results_dir)

    # Load first model
    model = xgb.Booster()
    model.load_model(str(results_dir / 'model_seed_0.json'))

    # Prepare data
    X_test = scaler.transform(test_df[feature_names].values)
    X_mcis = scaler.transform(mcis_df[feature_names].values)

    # Subsample if needed
    if len(X_test) > max_samples:
        idx = np.random.choice(len(X_test), max_samples, replace=False)
        X_test_sample = X_test[idx]
    else:
        X_test_sample = X_test

    if len(X_mcis) > max_samples:
        idx = np.random.choice(len(X_mcis), max_samples, replace=False)
        X_mcis_sample = X_mcis[idx]
    else:
        X_mcis_sample = X_mcis

    # SHAP explainer
    logger.info("Computing SHAP values...")
    explainer = shap.TreeExplainer(model)

    # Test set SHAP
    shap_test = explainer.shap_values(X_test_sample)

    # MCIs SHAP
    shap_mcis = explainer.shap_values(X_mcis_sample)

    # Summary plot - Test set
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_test, X_test_sample, feature_names=feature_names, show=False)
    plt.title('SHAP Summary - Test Set (CN vs AD-trajectory)')
    plt.tight_layout()
    plt.savefig(results_dir / 'shap_summary_test.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Summary plot - MCIs
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_mcis, X_mcis_sample, feature_names=feature_names, show=False)
    plt.title('SHAP Summary - MCI-stable (Holdout)')
    plt.tight_layout()
    plt.savefig(results_dir / 'shap_summary_mcis.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Compare mean absolute SHAP values
    mean_shap_test = np.abs(shap_test).mean(axis=0)
    mean_shap_mcis = np.abs(shap_mcis).mean(axis=0)

    comparison_df = pd.DataFrame({
        'feature': feature_names,
        'importance_test': mean_shap_test,
        'importance_mcis': mean_shap_mcis,
        'diff': mean_shap_mcis - mean_shap_test
    }).sort_values('importance_test', ascending=False)

    comparison_df.to_csv(results_dir / 'shap_comparison.csv', index=False)

    # Plot comparison
    fig, ax = plt.subplots(figsize=(12, 8))
    top_features = comparison_df.head(15)

    x = np.arange(len(top_features))
    width = 0.35

    ax.barh(x - width/2, top_features['importance_test'], width, label='Test Set', color='steelblue')
    ax.barh(x + width/2, top_features['importance_mcis'], width, label='MCIs (OOD)', color='orange')

    ax.set_yticks(x)
    ax.set_yticklabels(top_features['feature'])
    ax.set_xlabel('Mean |SHAP value|')
    ax.set_title('Feature Importance: Test vs MCIs')
    ax.legend()
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(results_dir / 'shap_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"SHAP plots saved to {results_dir}")


def main():
    parser = argparse.ArgumentParser(description='Feature space analysis')
    parser.add_argument('--results-dir', type=str, default='results', help='Results directory')
    parser.add_argument('--umap', action='store_true', help='Run UMAP analysis')
    parser.add_argument('--shap', action='store_true', help='Run SHAP analysis')
    parser.add_argument('--all', action='store_true', help='Run all analyses')

    args = parser.parse_args()
    results_dir = Path(args.results_dir)

    if args.all or args.umap:
        plot_umap(results_dir)

    if args.all or args.shap:
        plot_shap(results_dir)

    if not (args.all or args.umap or args.shap):
        logger.info("No analysis selected. Use --umap, --shap, or --all")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
CN vs AD-Trajectory Ensemble Training with Uncertainty Quantification

Trains multiple XGBoost models with different seeds, then:
1. Evaluates on test set (CN vs AD-trajectory)
2. Applies to MCI-stable (holdout) with uncertainty estimation
3. Generates calibration, UMAP, and SHAP analyses

Usage:
    python train_ensemble.py --config config.yaml
"""

import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, roc_auc_score, roc_curve, auc, brier_score_loss
)
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Features
FEATURES = [
    'AGE', 'PTGENDER', 'PTEDUCAT', 'PTMARRY',
    'VSWEIGHT', 'BMI',
    'MH14ALCH', 'MH17MALI', 'MH16SMOK', 'MH15DRUG', 'MH4CARD',
    'MHPSYCH', 'MH2NEURL', 'MH6HEPAT', 'MH12RENA',
    'TRAASCOR', 'TRABSCOR', 'TRABERRCOM', 'CATANIMSC',
    'CLOCKSCOR', 'BNTTOTAL', 'DSPANFOR', 'DSPANBAC',
    'BCDEPRES',
]


def load_config(config_path: str) -> dict:
    """Load YAML config"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def prepare_data(df: pd.DataFrame, features: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """Prepare features from raw data"""
    df_prep = df.copy()

    # Calculate AGE
    if 'AGE' not in df_prep.columns:
        if 'EXAMDATE' in df_prep.columns:
            df_prep['EXAMDATE'] = pd.to_datetime(df_prep['EXAMDATE'])
            df_prep['AGE'] = df_prep['EXAMDATE'].dt.year - df_prep['PTDOBYY']
        elif 'Acq Date' in df_prep.columns:
            df_prep['Acq Date'] = pd.to_datetime(df_prep['Acq Date'])
            df_prep['AGE'] = df_prep['Acq Date'].dt.year - df_prep['PTDOBYY']
        else:
            df_prep['AGE'] = 2010 - df_prep['PTDOBYY']

    # Calculate BMI
    if 'BMI' not in df_prep.columns and 'VSWEIGHT' in df_prep.columns and 'VSHEIGHT' in df_prep.columns:
        df_prep['BMI'] = df_prep['VSWEIGHT'] / ((df_prep['VSHEIGHT'] / 100) ** 2)

    # Get available features
    available = [f for f in features if f in df_prep.columns]
    missing = [f for f in features if f not in df_prep.columns]
    if missing:
        logger.warning(f"Missing features: {missing}")

    # Impute missing values
    for col in available:
        if df_prep[col].isnull().any():
            df_prep[col] = df_prep[col].fillna(df_prep[col].median())

    return df_prep, available


def create_cn_vs_ad_trajectory_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into:
    - Training data: CN vs (MCIc + AD)
    - Holdout data: MCI_stable (for later analysis)
    """
    # Get subject column
    subject_col = None
    for col in ['Subject', 'PTID', 'RID']:
        if col in df.columns:
            subject_col = col
            break
    if subject_col is None:
        df['Subject'] = range(len(df))
        subject_col = 'Subject'

    # CN: CLASS_4 == 'CN'
    cn_df = df[df['CLASS_4'] == 'CN'].copy()
    cn_df['label'] = 0
    cn_df['group'] = 'CN'
    logger.info(f"CN samples: {len(cn_df)}")

    # AD-trajectory: MCIc + AD
    mcic_df = df[df['CLASS_4'] == 'MCI_to_AD'].copy()
    mcic_df['label'] = 1
    mcic_df['group'] = 'MCIc'

    ad_df = df[df['CLASS_4'] == 'AD'].copy()
    ad_df['label'] = 1
    ad_df['group'] = 'AD'

    ad_trajectory_df = pd.concat([mcic_df, ad_df], ignore_index=True)
    logger.info(f"AD-trajectory samples: {len(ad_trajectory_df)} (MCIc: {len(mcic_df)}, AD: {len(ad_df)})")

    # Training data
    train_data = pd.concat([cn_df, ad_trajectory_df], ignore_index=True)

    # Holdout: MCI_stable
    mcis_df = df[df['CLASS_4'] == 'MCI_stable'].copy()
    mcis_df['group'] = 'MCIs'
    logger.info(f"MCI-stable (holdout): {len(mcis_df)}")

    return train_data, mcis_df


def patient_level_split(df: pd.DataFrame, train_ratio: float, val_ratio: float,
                        test_ratio: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data at patient level to avoid leakage"""
    subject_col = 'Subject' if 'Subject' in df.columns else 'PTID'

    # Get unique patients per class
    cn_patients = df[df['label'] == 0][subject_col].unique()
    ad_patients = df[df['label'] == 1][subject_col].unique()

    def split_patients(patients, seed):
        train_p, temp_p = train_test_split(patients, test_size=(val_ratio + test_ratio), random_state=seed)
        val_p, test_p = train_test_split(temp_p, test_size=test_ratio / (val_ratio + test_ratio), random_state=seed)
        return train_p, val_p, test_p

    cn_train, cn_val, cn_test = split_patients(cn_patients, seed)
    ad_train, ad_val, ad_test = split_patients(ad_patients, seed)

    train_patients = list(cn_train) + list(ad_train)
    val_patients = list(cn_val) + list(ad_val)
    test_patients = list(cn_test) + list(ad_test)

    train_df = df[df[subject_col].isin(train_patients)]
    val_df = df[df[subject_col].isin(val_patients)]
    test_df = df[df[subject_col].isin(test_patients)]

    logger.info(f"Split (seed={seed}): Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    return train_df, val_df, test_df


def train_single_model(X_train: np.ndarray, y_train: np.ndarray,
                       X_val: np.ndarray, y_val: np.ndarray,
                       feature_names: List[str], config: dict, seed: int) -> xgb.Booster:
    """Train a single XGBoost model"""
    xgb_config = config.get('xgboost', {})

    # Class weights
    class_counts = np.bincount(y_train)
    scale_pos_weight = class_counts[0] / class_counts[1]

    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'max_depth': xgb_config.get('max_depth', 6),
        'learning_rate': xgb_config.get('learning_rate', 0.1),
        'subsample': xgb_config.get('subsample', 0.8),
        'colsample_bytree': xgb_config.get('colsample_bytree', 0.8),
        'scale_pos_weight': scale_pos_weight,
        'random_state': seed,
        'tree_method': 'hist',
    }

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)

    model = xgb.train(
        params, dtrain,
        num_boost_round=xgb_config.get('num_boost_round', 200),
        evals=[(dval, 'val')],
        early_stopping_rounds=xgb_config.get('early_stopping_rounds', 20),
        verbose_eval=False
    )

    return model


def train_ensemble(X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray,
                   feature_names: List[str], config: dict) -> List[xgb.Booster]:
    """Train ensemble of models with different seeds"""
    ensemble_config = config.get('ensemble', {})
    n_seeds = ensemble_config.get('n_seeds', 10)
    base_seed = ensemble_config.get('base_seed', 42)

    models = []
    for i in range(n_seeds):
        seed = base_seed + i
        logger.info(f"Training model {i+1}/{n_seeds} (seed={seed})")
        model = train_single_model(X_train, y_train, X_val, y_val, feature_names, config, seed)
        models.append(model)

    return models


def predict_ensemble(models: List[xgb.Booster], X: np.ndarray,
                     feature_names: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get ensemble predictions with uncertainty.

    Returns:
        mean_proba: Mean probability across models
        std_proba: Standard deviation (uncertainty)
        all_proba: All individual predictions (n_samples, n_models)
    """
    dmatrix = xgb.DMatrix(X, feature_names=feature_names)

    all_proba = np.array([model.predict(dmatrix) for model in models])  # (n_models, n_samples)
    all_proba = all_proba.T  # (n_samples, n_models)

    mean_proba = all_proba.mean(axis=1)
    std_proba = all_proba.std(axis=1)

    return mean_proba, std_proba, all_proba


def evaluate_ensemble(models: List[xgb.Booster], X: np.ndarray, y: np.ndarray,
                      feature_names: List[str], dataset_name: str = 'Test') -> Dict:
    """Evaluate ensemble on labeled data"""
    mean_proba, std_proba, all_proba = predict_ensemble(models, X, feature_names)
    y_pred = (mean_proba >= 0.5).astype(int)

    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y, y_pred),
        'auc_roc': roc_auc_score(y, mean_proba),
        'brier_score': brier_score_loss(y, mean_proba),
        'mean_uncertainty': std_proba.mean(),
        'confusion_matrix': confusion_matrix(y, y_pred).tolist(),
    }

    logger.info(f"\n{'='*60}")
    logger.info(f"{dataset_name.upper()} RESULTS (Ensemble of {len(models)} models)")
    logger.info(f"{'='*60}")
    logger.info(f"Accuracy: {metrics['accuracy']*100:.2f}%")
    logger.info(f"Balanced Accuracy: {metrics['balanced_accuracy']*100:.2f}%")
    logger.info(f"AUC-ROC: {metrics['auc_roc']:.4f}")
    logger.info(f"Brier Score: {metrics['brier_score']:.4f}")
    logger.info(f"Mean Uncertainty (std): {metrics['mean_uncertainty']:.4f}")
    logger.info(f"\nConfusion Matrix:\n{confusion_matrix(y, y_pred)}")
    logger.info(f"\n{classification_report(y, y_pred, target_names=['CN', 'AD-trajectory'])}")

    return metrics, mean_proba, std_proba, all_proba


def analyze_mci_stable(models: List[xgb.Booster], X_mcis: np.ndarray,
                       mcis_df: pd.DataFrame, feature_names: List[str],
                       output_dir: Path, config: dict):
    """Analyze MCI-stable predictions with uncertainty"""
    logger.info(f"\n{'='*60}")
    logger.info("MCI-STABLE ANALYSIS (Out-of-Distribution)")
    logger.info(f"{'='*60}")

    mean_proba, std_proba, all_proba = predict_ensemble(models, X_mcis, feature_names)
    y_pred = (mean_proba >= 0.5).astype(int)

    # Classification results
    n_cn_like = sum(y_pred == 0)
    n_ad_like = sum(y_pred == 1)

    logger.info(f"Total MCI-stable samples: {len(y_pred)}")
    logger.info(f"Classified as CN-like: {n_cn_like} ({n_cn_like/len(y_pred)*100:.1f}%)")
    logger.info(f"Classified as AD-trajectory: {n_ad_like} ({n_ad_like/len(y_pred)*100:.1f}%)")
    logger.info(f"\nMean AD-trajectory probability: {mean_proba.mean():.3f} +/- {std_proba.mean():.3f}")
    logger.info(f"Mean uncertainty (ensemble std): {std_proba.mean():.4f}")

    # Confidence bins
    thresholds = config.get('analysis', {}).get('confidence_threshold', {})
    low_thresh = thresholds.get('low', 0.3)
    high_thresh = thresholds.get('high', 0.7)

    bins = [
        (0, low_thresh, f'Confident CN-like (<{low_thresh})'),
        (low_thresh, 0.5, f'Uncertain CN-like ({low_thresh}-0.5)'),
        (0.5, high_thresh, f'Uncertain AD-like (0.5-{high_thresh})'),
        (high_thresh, 1.0, f'Confident AD-like (>{high_thresh})'),
    ]

    logger.info(f"\nConfidence distribution:")
    for low, high, label in bins:
        count = sum((mean_proba >= low) & (mean_proba < high))
        logger.info(f"  {label}: {count} ({count/len(mean_proba)*100:.1f}%)")

    # Save predictions
    results_df = mcis_df.copy()
    results_df['AD_trajectory_prob_mean'] = mean_proba
    results_df['AD_trajectory_prob_std'] = std_proba
    results_df['predicted_class'] = ['AD-trajectory' if p == 1 else 'CN-like' for p in y_pred]
    results_df['confidence'] = np.where(
        mean_proba < low_thresh, 'high_CN',
        np.where(mean_proba < 0.5, 'uncertain_CN',
        np.where(mean_proba < high_thresh, 'uncertain_AD', 'high_AD'))
    )

    # Add individual model predictions
    for i in range(all_proba.shape[1]):
        results_df[f'prob_model_{i}'] = all_proba[:, i]

    results_df.to_csv(output_dir / 'mcis_predictions.csv', index=False)

    # Summary stats
    summary = {
        'total_samples': len(y_pred),
        'cn_like_count': int(n_cn_like),
        'ad_like_count': int(n_ad_like),
        'cn_like_percent': round(n_cn_like/len(y_pred)*100, 2),
        'ad_like_percent': round(n_ad_like/len(y_pred)*100, 2),
        'mean_prob': round(float(mean_proba.mean()), 4),
        'std_prob': round(float(mean_proba.std()), 4),
        'mean_uncertainty': round(float(std_proba.mean()), 4),
        'high_uncertainty_count': int(sum(std_proba > 0.1)),
    }

    with open(output_dir / 'mcis_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    return mean_proba, std_proba, results_df


def plot_calibration(y_true: np.ndarray, y_proba: np.ndarray,
                     output_dir: Path, n_bins: int = 10):
    """Plot calibration curve"""
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins, strategy='uniform')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Calibration curve
    axes[0].plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
    axes[0].plot(prob_pred, prob_true, 'o-', color='steelblue', label='Model')
    axes[0].set_xlabel('Mean Predicted Probability')
    axes[0].set_ylabel('Fraction of Positives')
    axes[0].set_title('Calibration Curve')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Histogram of predictions
    axes[1].hist(y_proba[y_true == 0], bins=20, alpha=0.7, label='CN (true)', color='green')
    axes[1].hist(y_proba[y_true == 1], bins=20, alpha=0.7, label='AD-trajectory (true)', color='red')
    axes[1].axvline(0.5, color='black', linestyle='--', label='Decision boundary')
    axes[1].set_xlabel('Predicted Probability')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Prediction Distribution by True Class')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'calibration.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Calculate ECE (Expected Calibration Error)
    ece = np.abs(prob_true - prob_pred).mean()
    logger.info(f"Expected Calibration Error (ECE): {ece:.4f}")

    return ece


def plot_confidence_comparison(test_proba: np.ndarray, test_labels: np.ndarray,
                               mcis_proba: np.ndarray, output_dir: Path):
    """Compare confidence distributions across groups"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Separate test set by class
    cn_test_proba = test_proba[test_labels == 0]
    ad_test_proba = test_proba[test_labels == 1]

    # Histograms
    axes[0, 0].hist(cn_test_proba, bins=20, alpha=0.7, color='green', edgecolor='black')
    axes[0, 0].axvline(0.5, color='red', linestyle='--')
    axes[0, 0].set_title('CN Test Set')
    axes[0, 0].set_xlabel('AD-trajectory Probability')
    axes[0, 0].set_ylabel('Count')

    axes[0, 1].hist(ad_test_proba, bins=20, alpha=0.7, color='red', edgecolor='black')
    axes[0, 1].axvline(0.5, color='red', linestyle='--')
    axes[0, 1].set_title('AD-trajectory Test Set')
    axes[0, 1].set_xlabel('AD-trajectory Probability')

    axes[1, 0].hist(mcis_proba, bins=20, alpha=0.7, color='orange', edgecolor='black')
    axes[1, 0].axvline(0.5, color='red', linestyle='--')
    axes[1, 0].set_title('MCI-stable (Holdout - OOD)')
    axes[1, 0].set_xlabel('AD-trajectory Probability')
    axes[1, 0].set_ylabel('Count')

    # Combined box plot
    data = [cn_test_proba, ad_test_proba, mcis_proba]
    bp = axes[1, 1].boxplot(data, labels=['CN\n(test)', 'AD-traj\n(test)', 'MCIs\n(holdout)'],
                            patch_artist=True)
    colors = ['green', 'red', 'orange']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[1, 1].axhline(0.5, color='red', linestyle='--', label='Decision boundary')
    axes[1, 1].set_ylabel('AD-trajectory Probability')
    axes[1, 1].set_title('Confidence Comparison')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'confidence_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_uncertainty_analysis(test_proba: np.ndarray, test_std: np.ndarray, test_labels: np.ndarray,
                              mcis_proba: np.ndarray, mcis_std: np.ndarray, output_dir: Path):
    """Analyze uncertainty differences between test and OOD data"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Scatter: probability vs uncertainty (test)
    colors_test = ['green' if l == 0 else 'red' for l in test_labels]
    axes[0].scatter(test_proba, test_std, c=colors_test, alpha=0.6, s=30)
    axes[0].set_xlabel('AD-trajectory Probability')
    axes[0].set_ylabel('Uncertainty (Ensemble Std)')
    axes[0].set_title('Test Set: Probability vs Uncertainty')
    axes[0].axvline(0.5, color='black', linestyle='--', alpha=0.5)

    # Scatter: probability vs uncertainty (MCIs)
    axes[1].scatter(mcis_proba, mcis_std, c='orange', alpha=0.6, s=30)
    axes[1].set_xlabel('AD-trajectory Probability')
    axes[1].set_ylabel('Uncertainty (Ensemble Std)')
    axes[1].set_title('MCI-stable (OOD): Probability vs Uncertainty')
    axes[1].axvline(0.5, color='black', linestyle='--', alpha=0.5)

    # Uncertainty comparison
    axes[2].boxplot([test_std, mcis_std], labels=['Test Set', 'MCIs (OOD)'])
    axes[2].set_ylabel('Uncertainty (Ensemble Std)')
    axes[2].set_title('Uncertainty Comparison')

    plt.tight_layout()
    plt.savefig(output_dir / 'uncertainty_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Statistical comparison
    from scipy import stats
    stat, pval = stats.mannwhitneyu(test_std, mcis_std, alternative='two-sided')
    logger.info(f"Uncertainty comparison (Mann-Whitney U): p={pval:.4f}")
    logger.info(f"  Test set mean uncertainty: {test_std.mean():.4f}")
    logger.info(f"  MCIs mean uncertainty: {mcis_std.mean():.4f}")


def plot_results(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray,
                 metrics: Dict, output_dir: Path, model: xgb.Booster = None):
    """Generate standard result plots"""
    # Confusion Matrix
    cm = np.array(metrics['confusion_matrix'])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['CN', 'AD-trajectory'], yticklabels=['CN', 'AD-trajectory'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='steelblue', linewidth=2, label=f'ROC (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / 'roc_curve.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Feature importance (from first model)
    if model is not None:
        importance = model.get_score(importance_type='gain')
        if importance:
            sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
            top_n = min(15, len(sorted_imp))
            features = [x[0] for x in sorted_imp[:top_n]][::-1]
            values = [x[1] for x in sorted_imp[:top_n]][::-1]

            plt.figure(figsize=(10, 8))
            colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(features)))
            plt.barh(features, values, color=colors)
            plt.xlabel('Importance (Gain)')
            plt.title('Top Feature Importance')
            plt.tight_layout()
            plt.savefig(output_dir / 'feature_importance.png', dpi=150, bbox_inches='tight')
            plt.close()


def main():
    parser = argparse.ArgumentParser(description='CN vs AD-trajectory ensemble training')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Setup paths
    project_root = Path(__file__).parent.parent.parent
    output_dir = project_root / config.get('output_dir', 'experiments/tabular_cn_vs_ad_trajectory/results')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)

    # Load data
    input_csv = project_root / config['data']['input_csv']
    logger.info(f"Loading data from {input_csv}")
    df = pd.read_csv(input_csv)
    logger.info(f"Loaded {len(df)} samples")

    # Prepare features
    df, feature_names = prepare_data(df, FEATURES)
    logger.info(f"Using {len(feature_names)} features")

    # Create CN vs AD-trajectory dataset
    train_data, mcis_df = create_cn_vs_ad_trajectory_dataset(df)

    # Prepare MCIs features
    mcis_df, _ = prepare_data(mcis_df, feature_names)

    # Split training data
    split_config = config.get('split', {})
    train_df, val_df, test_df = patient_level_split(
        train_data,
        split_config.get('train_ratio', 0.7),
        split_config.get('val_ratio', 0.15),
        split_config.get('test_ratio', 0.15),
        config.get('ensemble', {}).get('base_seed', 42)
    )

    # Prepare arrays
    X_train = train_df[feature_names].values
    y_train = train_df['label'].values
    X_val = val_df[feature_names].values
    y_val = val_df['label'].values
    X_test = test_df[feature_names].values
    y_test = test_df['label'].values
    X_mcis = mcis_df[feature_names].values

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    X_mcis_scaled = scaler.transform(X_mcis)

    # Train ensemble
    logger.info("\n" + "="*60)
    logger.info("TRAINING ENSEMBLE")
    logger.info("="*60)
    models = train_ensemble(X_train_scaled, y_train, X_val_scaled, y_val, feature_names, config)

    # Evaluate on test set
    test_metrics, test_proba, test_std, _ = evaluate_ensemble(
        models, X_test_scaled, y_test, feature_names, 'Test'
    )

    # Analyze MCI-stable
    mcis_proba, mcis_std, mcis_results = analyze_mci_stable(
        models, X_mcis_scaled, mcis_df, feature_names, output_dir, config
    )

    # Calibration analysis
    logger.info("\n" + "="*60)
    logger.info("CALIBRATION ANALYSIS")
    logger.info("="*60)
    ece = plot_calibration(y_test, test_proba, output_dir)
    test_metrics['ece'] = ece

    # Confidence comparison
    logger.info("\n" + "="*60)
    logger.info("CONFIDENCE COMPARISON")
    logger.info("="*60)
    plot_confidence_comparison(test_proba, y_test, mcis_proba, output_dir)

    # Uncertainty analysis
    logger.info("\n" + "="*60)
    logger.info("UNCERTAINTY ANALYSIS")
    logger.info("="*60)
    plot_uncertainty_analysis(test_proba, test_std, y_test, mcis_proba, mcis_std, output_dir)

    # Standard plots
    y_pred = (test_proba >= 0.5).astype(int)
    plot_results(y_test, y_pred, test_proba, test_metrics, output_dir, models[0])

    # Save artifacts
    joblib.dump(scaler, output_dir / 'scaler.pkl')
    for i, model in enumerate(models):
        model.save_model(str(output_dir / f'model_seed_{i}.json'))

    with open(output_dir / 'feature_names.json', 'w') as f:
        json.dump(feature_names, f, indent=2)

    with open(output_dir / 'test_metrics.json', 'w') as f:
        json.dump(test_metrics, f, indent=2, default=str)

    # Save test predictions
    test_results = test_df.copy()
    test_results['prob_mean'] = test_proba
    test_results['prob_std'] = test_std
    test_results['predicted'] = y_pred
    test_results.to_csv(output_dir / 'test_predictions.csv', index=False)

    logger.info(f"\n{'='*60}")
    logger.info("COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"All results saved to {output_dir}")


if __name__ == '__main__':
    main()

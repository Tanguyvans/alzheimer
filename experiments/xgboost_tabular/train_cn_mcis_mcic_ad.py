#!/usr/bin/env python3
"""
Train XGBoost for multiclass classification.

If DX column available (4-class): CN | MCI stable | MCI→AD | AD
If only Group column (3-class): CN | MCI | AD

Usage:
    python train_cn_mcis_mcic_ad.py \
        --input-csv /path/to/clinical_data.csv \
        --output-dir results/multiclass \
        --seed 42
"""

import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, roc_auc_score, roc_curve, auc
)
from sklearn.preprocessing import StandardScaler
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Will be set dynamically based on data
CLASS_NAMES = None
CLASS_MAPPING = None
NUM_CLASSES = None


# Fair features: Exclude diagnostic criteria and potential confounds
# Removed: MMSCORE, CDGLOBAL, BCFAQ (diagnostic), PTTLANG, VSHEIGHT, PTHAND, PTRACCAT (confounds)
CLINICAL_FEATURES = [
    # Demographics (legitimate predictors)
    'AGE', 'PTGENDER', 'PTEDUCAT', 'PTMARRY',
    # Physical measurements
    'VSWEIGHT', 'BMI',
    # Medical history
    'MH14ALCH', 'MH17MALI', 'MH16SMOK', 'MH15DRUG', 'MH4CARD',
    'MHPSYCH', 'MH2NEURL', 'MH6HEPAT', 'MH12RENA',
    # Neuropsychological tests (core cognitive measures)
    'TRAASCOR', 'TRABSCOR', 'TRABERRCOM', 'CATANIMSC',
    'CLOCKSCOR', 'BNTTOTAL', 'DSPANFOR', 'DSPANBAC',
    # Clinical assessments
    'BCDEPRES',
]


def prepare_data(df: pd.DataFrame):
    """Prepare features and create multiclass labels"""
    global CLASS_NAMES, CLASS_MAPPING, NUM_CLASSES

    df_prep = df.copy()

    # Calculate AGE
    if 'EXAMDATE' in df_prep.columns:
        df_prep['EXAMDATE'] = pd.to_datetime(df_prep['EXAMDATE'])
        df_prep['exam_year'] = df_prep['EXAMDATE'].dt.year
        df_prep['AGE'] = df_prep['exam_year'] - df_prep['PTDOBYY']
    elif 'Acq Date' in df_prep.columns:
        df_prep['Acq Date'] = pd.to_datetime(df_prep['Acq Date'])
        df_prep['acq_year'] = df_prep['Acq Date'].dt.year
        df_prep['AGE'] = df_prep['acq_year'] - df_prep['PTDOBYY']
    else:
        df_prep['AGE'] = 2010 - df_prep['PTDOBYY']

    # Calculate BMI
    df_prep['BMI'] = df_prep['VSWEIGHT'] / ((df_prep['VSHEIGHT'] / 100) ** 2)

    # Check if we have DX column for 4-class or just Group for 3-class
    has_dx = 'DX' in df_prep.columns
    has_group = 'Group' in df_prep.columns

    if has_dx and has_group:
        # 4-class: CN | MCI stable | MCI→AD | AD
        CLASS_NAMES = ['CN', 'MCI_stable', 'MCI_to_AD', 'AD']
        CLASS_MAPPING = {'CN': 0, 'MCI_stable': 1, 'MCI_to_AD': 2, 'AD': 3}
        NUM_CLASSES = 4
        logger.info("Using 4-class classification: CN | MCI stable | MCI→AD | AD")

        def assign_label(row):
            group = row['Group']
            dx = row['DX']
            if group == 'CN' and dx == 'CN':
                return 0  # CN
            elif group == 'MCI' and dx == 'MCI':
                return 1  # MCI stable
            elif group == 'MCI' and dx == 'AD':
                return 2  # MCI→AD (converter)
            elif group == 'AD' or dx == 'AD':
                return 3  # AD
            elif group == 'CN' and dx == 'MCI':
                return 1  # CN→MCI (treat as MCI stable)
            else:
                return -1
        df_prep['label'] = df_prep.apply(assign_label, axis=1)
    elif has_group:
        # 3-class: CN | MCI | AD
        CLASS_NAMES = ['CN', 'MCI', 'AD']
        CLASS_MAPPING = {'CN': 0, 'MCI': 1, 'AD': 2}
        NUM_CLASSES = 3
        logger.info("Using 3-class classification: CN | MCI | AD")

        df_prep['label'] = df_prep['Group'].map(CLASS_MAPPING)
    else:
        raise ValueError("No diagnosis column found (expected 'Group' or 'DX')")

    # Remove unknown labels
    df_prep = df_prep[df_prep['label'] >= 0].copy()

    # Get available features
    available_features = [f for f in CLINICAL_FEATURES if f in df_prep.columns]
    missing_features = [f for f in CLINICAL_FEATURES if f not in df_prep.columns]
    if missing_features:
        logger.warning(f"Missing features: {missing_features}")

    # Select features
    df_features = df_prep[available_features + ['label', 'Subject']].copy()

    # Impute missing values with median
    for col in available_features:
        if df_features[col].isnull().any():
            median_val = df_features[col].median()
            df_features[col] = df_features[col].fillna(median_val)

    logger.info(f"Prepared {len(df_features)} samples with {len(available_features)} features")
    for i, name in enumerate(CLASS_NAMES):
        count = sum(df_features['label'] == i)
        logger.info(f"  Class {i} ({name}): {count} samples")

    return df_features, available_features


def patient_level_split(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """Split data at patient level to prevent data leakage"""
    # Get unique patients per class
    patient_classes = {}
    for label in range(NUM_CLASSES):
        patient_classes[label] = df[df['label'] == label]['Subject'].unique()
        logger.info(f"Unique patients - {CLASS_NAMES[label]}: {len(patient_classes[label])}")

    def split_patients(patients):
        if len(patients) < 3:
            # Not enough patients to split properly
            return patients, np.array([]), np.array([])
        train_p, temp_p = train_test_split(patients, test_size=(val_ratio + test_ratio), random_state=seed)
        if len(temp_p) < 2:
            return train_p, temp_p, np.array([])
        val_p, test_p = train_test_split(temp_p, test_size=test_ratio / (val_ratio + test_ratio), random_state=seed)
        return train_p, val_p, test_p

    train_patients = []
    val_patients = []
    test_patients = []

    for label in range(NUM_CLASSES):
        train_p, val_p, test_p = split_patients(patient_classes[label])
        train_patients.extend(train_p)
        val_patients.extend(val_p)
        test_patients.extend(test_p)

    train_df = df[df['Subject'].isin(train_patients)]
    val_df = df[df['Subject'].isin(val_patients)]
    test_df = df[df['Subject'].isin(test_patients)]

    logger.info(f"Train: {len(train_df)} samples, Val: {len(val_df)} samples, Test: {len(test_df)} samples")

    return train_df, val_df, test_df


def train_xgboost(X_train, y_train, X_val, y_val, feature_names=None):
    """Train XGBoost for multiclass classification"""
    params = {
        'objective': 'multi:softprob',
        'num_class': NUM_CLASSES,
        'eval_metric': 'mlogloss',
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'tree_method': 'hist',
    }

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=200,
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=20,
        verbose_eval=10
    )

    logger.info(f"Best iteration: {model.best_iteration}")

    return model


def evaluate_model(model, X, y, feature_names, dataset_name='Test'):
    """Evaluate model and return metrics"""
    dmatrix = xgb.DMatrix(X, feature_names=feature_names)
    y_proba = model.predict(dmatrix)
    y_pred = np.argmax(y_proba, axis=1)

    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y, y_pred),
        'confusion_matrix': confusion_matrix(y, y_pred).tolist(),
        'classification_report': classification_report(y, y_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    }

    # Calculate macro AUC-ROC if possible
    try:
        from sklearn.preprocessing import label_binarize
        y_bin = label_binarize(y, classes=list(range(NUM_CLASSES)))
        metrics['auc_roc_macro'] = roc_auc_score(y_bin, y_proba, average='macro', multi_class='ovr')
    except Exception as e:
        logger.warning(f"Could not calculate AUC-ROC: {e}")
        metrics['auc_roc_macro'] = None

    logger.info(f"\n{'='*60}")
    logger.info(f"{dataset_name.upper()} RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"Accuracy: {metrics['accuracy']*100:.2f}%")
    logger.info(f"Balanced Accuracy: {metrics['balanced_accuracy']*100:.2f}%")
    if metrics['auc_roc_macro']:
        logger.info(f"AUC-ROC (macro): {metrics['auc_roc_macro']:.4f}")
    logger.info(f"\nConfusion Matrix:\n{confusion_matrix(y, y_pred)}")
    logger.info(f"\nClassification Report:\n{classification_report(y, y_pred, target_names=CLASS_NAMES, zero_division=0)}")

    return metrics, y_pred, y_proba


def plot_results(y_true, y_pred, y_proba, metrics, output_dir, model=None):
    """Generate and save plots"""
    output_dir = Path(output_dir)
    from sklearn.preprocessing import label_binarize

    # Confusion Matrix
    cm = np.array(metrics['confusion_matrix'])
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {NUM_CLASSES}-Class Classification')
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Normalized Confusion Matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Normalized Confusion Matrix - {NUM_CLASSES}-Class Classification')
    plt.savefig(output_dir / 'confusion_matrix_normalized.png', dpi=150, bbox_inches='tight')
    plt.close()

    # ROC Curve (One-vs-Rest for multiclass)
    y_bin = label_binarize(y_true, classes=list(range(NUM_CLASSES)))
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']
    plt.figure(figsize=(10, 8))
    for i, (class_name, color) in enumerate(zip(CLASS_NAMES, colors)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, linewidth=2,
                 label=f'{class_name} (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curves - {NUM_CLASSES}-Class (One-vs-Rest)', fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / 'roc_curve.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Per-class metrics bar chart
    report = metrics['classification_report']
    classes = CLASS_NAMES
    precision = [report[c]['precision'] for c in classes]
    recall = [report[c]['recall'] for c in classes]
    f1 = [report[c]['f1-score'] for c in classes]

    x = np.arange(len(classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width, precision, width, label='Precision')
    ax.bar(x, recall, width, label='Recall')
    ax.bar(x + width, f1, width, label='F1-Score')

    ax.set_xlabel('Class')
    ax.set_ylabel('Score')
    ax.set_title(f'Per-Class Metrics - {NUM_CLASSES}-Class Classification')
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    plt.savefig(output_dir / 'per_class_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Feature Importance
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
            plt.xlabel('Importance (Gain)', fontsize=12)
            plt.title(f'Top Feature Importance - XGBoost ({NUM_CLASSES}-class)', fontsize=14)
            plt.tight_layout()
            plt.savefig(output_dir / 'feature_importance.png', dpi=150, bbox_inches='tight')
            plt.close()

    logger.info(f"Plots saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Train multiclass XGBoost classifier (3 or 4 classes)')
    parser.add_argument('--input-csv', type=str, required=True, help='Path to clinical data CSV')
    parser.add_argument('--output-dir', type=str, default='results/multiclass', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and prepare data
    logger.info(f"Loading data from {args.input_csv}")
    df = pd.read_csv(args.input_csv)
    logger.info(f"Loaded {len(df)} samples")

    df_features, feature_names = prepare_data(df)

    # Patient-level split
    train_df, val_df, test_df = patient_level_split(df_features, seed=args.seed)

    # Prepare arrays
    X_train = train_df[feature_names].values
    y_train = train_df['label'].values
    X_val = val_df[feature_names].values
    y_val = val_df['label'].values
    X_test = test_df[feature_names].values
    y_test = test_df['label'].values

    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    model = train_xgboost(X_train_scaled, y_train, X_val_scaled, y_val, feature_names)

    # Evaluate
    test_metrics, y_pred, y_proba = evaluate_model(model, X_test_scaled, y_test, feature_names, 'Test')

    # Save model and artifacts
    model.save_model(str(output_dir / 'xgboost_model.json'))
    joblib.dump(scaler, output_dir / 'scaler.pkl')

    with open(output_dir / 'feature_names.json', 'w') as f:
        json.dump(feature_names, f, indent=2)

    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(test_metrics, f, indent=2, default=str)

    with open(output_dir / 'class_mapping.json', 'w') as f:
        json.dump(CLASS_MAPPING, f, indent=2)

    # Save predictions
    pred_dict = {'y_true': y_test, 'y_pred': y_pred}
    for i, class_name in enumerate(CLASS_NAMES):
        pred_dict[f'y_proba_{class_name}'] = y_proba[:, i]
    pred_df = pd.DataFrame(pred_dict)
    pred_df.to_csv(output_dir / 'predictions.csv', index=False)

    # Feature importance
    importance = model.get_score(importance_type='gain')
    importance_df = pd.DataFrame({
        'feature': list(importance.keys()),
        'importance': list(importance.values())
    }).sort_values('importance', ascending=False)
    importance_df.to_csv(output_dir / 'feature_importance.csv', index=False)

    # Plot results
    plot_results(y_test, y_pred, y_proba, test_metrics, output_dir, model)

    logger.info(f"\nAll results saved to {output_dir}")
    logger.info(f"Final Test Accuracy: {test_metrics['accuracy']*100:.2f}%")
    logger.info(f"Final Test Balanced Accuracy: {test_metrics['balanced_accuracy']*100:.2f}%")


if __name__ == '__main__':
    main()

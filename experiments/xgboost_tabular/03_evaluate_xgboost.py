#!/usr/bin/env python3
"""
Evaluate trained XGBoost model on test set

Usage:
    python 03_evaluate_xgboost.py \
        --model-dir models \
        --test-csv data/splits/test.csv \
        --output-csv predictions.csv
"""

import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
import json
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model_and_scaler(model_dir: Path):
    """Load trained model and scaler"""
    logger.info(f"Loading model from {model_dir}...")

    # Load model
    model_path = model_dir / 'xgboost_model.json'
    model = xgb.Booster()
    model.load_model(str(model_path))

    # Load scaler
    scaler_path = model_dir / 'scaler.pkl'
    scaler = joblib.load(scaler_path)

    # Load feature names
    feature_names_path = model_dir / 'feature_names.json'
    with open(feature_names_path, 'r') as f:
        feature_names = json.load(f)

    logger.info("✓ Model, scaler, and feature names loaded successfully")

    return model, scaler, feature_names


def evaluate_model(model, scaler, X, y, feature_names, output_dir: Path):
    """Comprehensive model evaluation"""

    # Scale features
    X_scaled = scaler.transform(X)

    # Predict
    dmatrix = xgb.DMatrix(X_scaled, feature_names=feature_names)
    y_proba = model.predict(dmatrix)
    y_pred = (y_proba >= 0.5).astype(int)

    # Calculate metrics
    accuracy = accuracy_score(y, y_pred)
    balanced_acc = balanced_accuracy_score(y, y_pred)
    auc_score = roc_auc_score(y, y_proba)

    logger.info("\n" + "="*80)
    logger.info("TEST SET RESULTS")
    logger.info("="*80)
    logger.info(f"Samples: {len(y)}")
    logger.info(f"  CN: {(y==0).sum()}")
    logger.info(f"  AD+MCI-to-AD: {(y==1).sum()}")
    logger.info(f"\nAccuracy: {accuracy*100:.2f}%")
    logger.info(f"Balanced Accuracy: {balanced_acc*100:.2f}%")
    logger.info(f"AUC-ROC: {auc_score:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    logger.info(f"\nConfusion Matrix:")
    logger.info(cm)

    # Classification report
    logger.info(f"\nClassification Report:")
    report = classification_report(y, y_pred, target_names=['CN', 'AD+MCI-to-AD'])
    logger.info(report)

    # Per-class metrics
    tn, fp, fn, tp = cm.ravel()
    cn_precision = tn / (tn + fn) if (tn + fn) > 0 else 0
    cn_recall = tn / (tn + fp) if (tn + fp) > 0 else 0
    ad_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    ad_recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    logger.info(f"\nDetailed per-class metrics:")
    logger.info(f"  CN: Precision={cn_precision*100:.2f}%, Recall={cn_recall*100:.2f}%")
    logger.info(f"  AD+MCI-to-AD: Precision={ad_precision*100:.2f}%, Recall={ad_recall*100:.2f}%")

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.set(font_scale=1.2)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['CN', 'AD+MCI-to-AD'],
                yticklabels=['CN', 'AD+MCI-to-AD'],
                annot_kws={'size': 16})
    plt.title('Confusion Matrix - XGBoost CN vs AD+MCI-to-AD', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=14, fontweight='bold')

    # Add accuracy text
    plt.text(0.5, -0.15, f'Accuracy: {accuracy*100:.2f}% | Balanced Accuracy: {balanced_acc*100:.2f}%',
             ha='center', va='center', transform=plt.gca().transAxes,
             fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    logger.info(f"\n✓ Confusion matrix saved to {output_dir / 'confusion_matrix.png'}")
    plt.close()
    sns.reset_orig()  # Reset seaborn settings

    # Plot ROC curve
    fpr, tpr, thresholds = roc_curve(y, y_proba)
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, linewidth=3, label=f'XGBoost (AUC = {auc_score:.4f})', color='#2E86AB')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier', alpha=0.5)
    plt.fill_between(fpr, tpr, alpha=0.2, color='#2E86AB')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=14, fontweight='bold')
    plt.title('ROC Curve - XGBoost CN vs AD+MCI-to-AD', fontsize=16, fontweight='bold', pad=20)
    plt.legend(loc='lower right', fontsize=12)
    plt.grid(alpha=0.3, linestyle='--', linewidth=0.5)

    # Add accuracy metrics as text
    plt.text(0.6, 0.2, f'Accuracy: {accuracy*100:.2f}%\nBalanced Acc: {balanced_acc*100:.2f}%',
             fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_dir / 'roc_curve.png', dpi=300, bbox_inches='tight')
    logger.info(f"✓ ROC curve saved to {output_dir / 'roc_curve.png'}")
    plt.close()

    return {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'auc_roc': auc_score,
        'predictions': y_pred,
        'probabilities': y_proba
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate XGBoost model')
    parser.add_argument('--model-dir', type=str, required=True,
                        help='Directory containing trained model')
    parser.add_argument('--test-csv', type=str, required=True,
                        help='Path to test CSV')
    parser.add_argument('--output-csv', type=str, default='predictions.csv',
                        help='Output CSV with predictions')

    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    output_dir = model_dir  # Save plots in same directory as model

    # Load model
    model, scaler, feature_names = load_model_and_scaler(model_dir)

    # Load test data
    logger.info(f"Loading test data from {args.test_csv}...")
    test_df = pd.read_csv(args.test_csv)

    X_test = test_df.drop('label', axis=1)
    y_test = test_df['label']

    logger.info(f"Test samples: {len(X_test)}")

    # Evaluate
    results = evaluate_model(model, scaler, X_test, y_test, feature_names, output_dir)

    # Save predictions
    predictions_df = test_df.copy()
    predictions_df['predicted_label'] = results['predictions']
    predictions_df['predicted_CN_proba'] = 1 - results['probabilities']
    predictions_df['predicted_AD_proba'] = results['probabilities']
    predictions_df['predicted_class'] = predictions_df['predicted_label'].map({0: 'CN', 1: 'AD+MCI-to-AD'})
    predictions_df['true_class'] = predictions_df['label'].map({0: 'CN', 1: 'AD+MCI-to-AD'})

    predictions_df.to_csv(args.output_csv, index=False)
    logger.info(f"\n✓ Predictions saved to {args.output_csv}")

    logger.info("\n" + "="*80)
    logger.info("EVALUATION COMPLETED!")
    logger.info(f"Test Accuracy: {results['accuracy']*100:.2f}%")
    logger.info(f"Test Balanced Accuracy: {results['balanced_accuracy']*100:.2f}%")
    logger.info(f"Test AUC-ROC: {results['auc_roc']:.4f}")
    logger.info("="*80)


if __name__ == '__main__':
    main()

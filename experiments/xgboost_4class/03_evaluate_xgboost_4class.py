#!/usr/bin/env python3
"""
Evaluate XGBoost 4-class model on test set

Generates:
- 4x4 confusion matrix
- Per-class metrics
- Predictions CSV

Usage:
    python 03_evaluate_xgboost_4class.py \
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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    balanced_accuracy_score, accuracy_score
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model_and_artifacts(model_dir: Path):
    """Load trained model, scaler, and feature names"""
    logger.info(f"Loading model from {model_dir}...")

    # Load model
    model_path = model_dir / 'xgboost_model_4class.json'
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


def evaluate_model(model, X_test, y_test, class_names):
    """
    Evaluate model on test set

    Args:
        model: Trained XGBoost model
        X_test, y_test: Test data
        class_names: List of class names

    Returns:
        Predictions, probabilities, metrics
    """
    logger.info("\n" + "="*80)
    logger.info("TEST SET RESULTS")
    logger.info("="*80)

    # Predict
    dtest = xgb.DMatrix(X_test)
    y_proba = model.predict(dtest)
    y_pred = np.argmax(y_proba, axis=1)

    # Overall metrics
    accuracy = accuracy_score(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)

    logger.info(f"\nSamples: {len(y_test)}")
    logger.info(f"Accuracy: {accuracy*100:.2f}%")
    logger.info(f"Balanced Accuracy: {balanced_acc*100:.2f}%")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    logger.info(f"\nConfusion Matrix:")
    logger.info(cm)

    # Classification report
    logger.info(f"\nClassification Report:")
    report = classification_report(y_test, y_pred, target_names=class_names, digits=4)
    logger.info(report)

    # Per-class accuracy
    logger.info(f"\nPer-class metrics:")
    for i, name in enumerate(class_names):
        class_mask = y_test == i
        if class_mask.sum() > 0:
            class_acc = (y_pred[class_mask] == i).mean()
            logger.info(f"  {name}: Accuracy={class_acc*100:.2f}% ({class_mask.sum()} samples)")

    logger.info("="*80)

    return y_pred, y_proba, accuracy, balanced_acc, cm


def plot_confusion_matrix(cm, class_names, output_path):
    """
    Plot 4x4 confusion matrix with professional styling

    Args:
        cm: Confusion matrix
        class_names: List of class names
        output_path: Path to save figure
    """
    logger.info(f"\nCreating confusion matrix visualization...")

    plt.figure(figsize=(12, 10))
    sns.set(font_scale=1.4)

    # Normalize confusion matrix (percentage)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    # Create heatmap with both counts and percentages
    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f'{cm[i, j]}\n({cm_norm[i, j]:.1f}%)'

    sns.heatmap(
        cm_norm,
        annot=annot,
        fmt='',
        cmap='Blues',
        cbar_kws={'label': 'Percentage (%)'},
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5,
        linecolor='gray',
        square=True
    )

    plt.xlabel('Predicted Label', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=16, fontweight='bold')
    plt.title('4-Class Confusion Matrix: CN | MCI-stable | MCI→AD | AD',
              fontsize=18, fontweight='bold', pad=20)

    # Add overall accuracy as subtitle
    accuracy = cm.diagonal().sum() / cm.sum()
    plt.text(0.5, -0.1,
             f'Overall Accuracy: {accuracy*100:.2f}%',
             ha='center', va='center', transform=plt.gca().transAxes,
             fontsize=14, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"✓ Confusion matrix saved to {output_path}")
    plt.close()

    sns.reset_orig()


def plot_per_class_performance(cm, class_names, output_path):
    """
    Plot per-class precision, recall, F1-score

    Args:
        cm: Confusion matrix
        class_names: List of class names
        output_path: Path to save figure
    """
    logger.info(f"\nCreating per-class performance visualization...")

    # Calculate metrics from confusion matrix
    precision = []
    recall = []
    f1_score = []

    for i in range(len(class_names)):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

        precision.append(prec * 100)
        recall.append(rec * 100)
        f1_score.append(f1 * 100)

    # Plot
    x = np.arange(len(class_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.bar(x - width, precision, width, label='Precision', color='#2E86AB', alpha=0.8)
    ax.bar(x, recall, width, label='Recall', color='#A23B72', alpha=0.8)
    ax.bar(x + width, f1_score, width, label='F1-Score', color='#F18F01', alpha=0.8)

    ax.set_xlabel('Class', fontsize=14, fontweight='bold')
    ax.set_ylabel('Score (%)', fontsize=14, fontweight='bold')
    ax.set_title('Per-Class Performance Metrics', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3, axis='y')
    ax.set_ylim(0, 105)

    # Add value labels on bars
    for i, (p, r, f) in enumerate(zip(precision, recall, f1_score)):
        ax.text(i - width, p + 2, f'{p:.1f}', ha='center', fontsize=10)
        ax.text(i, r + 2, f'{r:.1f}', ha='center', fontsize=10)
        ax.text(i + width, f + 2, f'{f:.1f}', ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"✓ Per-class performance plot saved to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Evaluate XGBoost 4-class model')
    parser.add_argument('--model-dir', type=str, required=True)
    parser.add_argument('--test-csv', type=str, required=True)
    parser.add_argument('--output-csv', type=str, default='predictions_4class.csv')

    args = parser.parse_args()

    model_dir = Path(args.model_dir)

    # Load model and artifacts
    model, scaler, feature_names = load_model_and_artifacts(model_dir)

    # Load test data
    logger.info(f"\nLoading test data from {args.test_csv}...")
    test_df = pd.read_csv(args.test_csv)

    X_test = test_df.drop('label', axis=1).values
    y_test = test_df['label'].values

    logger.info(f"Test samples: {len(X_test)}")

    # Normalize features
    X_test_scaled = scaler.transform(X_test)

    # Evaluate
    class_names = ['CN', 'MCI-stable', 'MCI→AD', 'AD']
    y_pred, y_proba, accuracy, balanced_acc, cm = evaluate_model(
        model, X_test_scaled, y_test, class_names
    )

    # Plot confusion matrix
    cm_path = model_dir / 'confusion_matrix_4class.png'
    plot_confusion_matrix(cm, class_names, cm_path)

    # Plot per-class performance
    perf_path = model_dir / 'per_class_performance.png'
    plot_per_class_performance(cm, class_names, perf_path)

    # Save predictions
    predictions_df = test_df.copy()
    predictions_df['predicted_label'] = y_pred
    predictions_df['predicted_class'] = [class_names[p] for p in y_pred]

    for i, name in enumerate(class_names):
        predictions_df[f'prob_{name}'] = y_proba[:, i]

    predictions_df.to_csv(args.output_csv, index=False)
    logger.info(f"\n✓ Predictions saved to {args.output_csv}")

    # Final summary
    logger.info("\n" + "="*80)
    logger.info("EVALUATION COMPLETED!")
    logger.info(f"Test Accuracy: {accuracy*100:.2f}%")
    logger.info(f"Test Balanced Accuracy: {balanced_acc*100:.2f}%")
    logger.info("="*80)


if __name__ == '__main__':
    main()

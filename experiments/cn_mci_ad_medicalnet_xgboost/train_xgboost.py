#!/usr/bin/env python3
"""
Train XGBoost on MedicalNet ResNet-18 extracted features

This script:
1. Loads extracted features from CSV
2. Trains XGBoost classifier
3. Evaluates on validation and test sets
4. Saves model and predictions
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix
import argparse
import logging
from pathlib import Path
import joblib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_features(csv_path: str):
    """Load features from CSV"""
    df = pd.read_csv(csv_path)

    # Separate features and labels
    feature_cols = [col for col in df.columns if col.startswith('feature_')]
    X = df[feature_cols].values
    y = df['label'].values

    return X, y, df


def train_xgboost(
    train_csv: str,
    val_csv: str,
    test_csv: str,
    output_dir: str = 'results'
):
    """
    Train XGBoost on extracted features

    Args:
        train_csv: Path to training features CSV
        val_csv: Path to validation features CSV
        test_csv: Path to test features CSV
        output_dir: Directory to save results
    """
    logger.info("="*80)
    logger.info("TRAINING XGBOOST ON MEDICALNET FEATURES")
    logger.info("="*80)

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info("\nLoading features...")
    X_train, y_train, train_df = load_features(train_csv)
    X_val, y_val, val_df = load_features(val_csv)
    X_test, y_test, test_df = load_features(test_csv)

    logger.info(f"Train set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    logger.info(f"Val set: {X_val.shape[0]} samples")
    logger.info(f"Test set: {X_test.shape[0]} samples")

    # Class distribution
    label_names = {0: 'CN', 1: 'MCI', 2: 'AD'}
    logger.info("\nTrain class distribution:")
    for label in sorted(np.unique(y_train)):
        count = (y_train == label).sum()
        pct = 100 * count / len(y_train)
        logger.info(f"  {label_names[label]}: {count} ({pct:.1f}%)")

    # Calculate scale_pos_weight for class imbalance
    class_counts = np.bincount(y_train)
    scale_pos_weight = class_counts[0] / class_counts  # Ratio relative to CN

    logger.info("\n" + "="*80)
    logger.info("TRAINING XGBOOST MODEL")
    logger.info("="*80)

    # XGBoost parameters
    params = {
        'objective': 'multi:softmax',
        'num_class': 3,
        'max_depth': 4,                  # Shallow trees (prevent overfitting)
        'learning_rate': 0.1,
        'n_estimators': 300,
        'subsample': 0.8,                # Random sampling
        'colsample_bytree': 0.8,         # Random feature sampling
        'min_child_weight': 3,           # Regularization
        'gamma': 0.1,                    # Regularization
        'reg_alpha': 0.1,                # L1 regularization
        'reg_lambda': 1.0,               # L2 regularization
        'random_state': 42,
        'n_jobs': -1
    }

    logger.info("XGBoost parameters:")
    for key, value in params.items():
        logger.info(f"  {key}: {value}")

    # Train model with early stopping
    model = xgb.XGBClassifier(**params)

    logger.info("\nTraining with early stopping...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        eval_metric=['mlogloss', 'merror'],
        verbose=True
    )

    # Save model
    model_path = output_dir / 'xgboost_model.json'
    model.save_model(str(model_path))
    logger.info(f"\nModel saved to {model_path}")

    # Also save with joblib for easy loading
    joblib.dump(model, output_dir / 'xgboost_model.pkl')

    # Evaluate on all sets
    logger.info("\n" + "="*80)
    logger.info("EVALUATION RESULTS")
    logger.info("="*80)

    for name, X, y, df in [('Train', X_train, y_train, train_df),
                            ('Val', X_val, y_val, val_df),
                            ('Test', X_test, y_test, test_df)]:

        # Predictions
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)

        # Metrics
        acc = accuracy_score(y, y_pred)
        balanced_acc = balanced_accuracy_score(y, y_pred)

        logger.info(f"\n{name} Set:")
        logger.info(f"  Accuracy: {acc*100:.2f}%")
        logger.info(f"  Balanced Accuracy: {balanced_acc*100:.2f}%")

        # Confusion matrix
        cm = confusion_matrix(y, y_pred)
        logger.info(f"\n  Confusion Matrix:")
        logger.info(f"  {cm}")

        # Classification report
        logger.info(f"\n  Classification Report:")
        report = classification_report(y, y_pred, target_names=['CN', 'MCI', 'AD'])
        logger.info(f"\n{report}")

        # Save predictions
        pred_df = df.copy()
        pred_df['predicted_label'] = y_pred
        pred_df['predicted_CN_proba'] = y_proba[:, 0]
        pred_df['predicted_MCI_proba'] = y_proba[:, 1]
        pred_df['predicted_AD_proba'] = y_proba[:, 2]

        pred_path = output_dir / f'{name.lower()}_predictions.csv'
        pred_df.to_csv(pred_path, index=False)
        logger.info(f"\n  Predictions saved to {pred_path}")

    # Feature importance
    logger.info("\n" + "="*80)
    logger.info("TOP 20 MOST IMPORTANT FEATURES")
    logger.info("="*80)

    importance = model.feature_importances_
    feature_names = [f'feature_{i}' for i in range(len(importance))]

    # Sort by importance
    indices = np.argsort(importance)[::-1]

    for i in range(min(20, len(importance))):
        idx = indices[i]
        logger.info(f"  {i+1}. {feature_names[idx]}: {importance[idx]:.4f}")

    # Save feature importance
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)

    importance_df.to_csv(output_dir / 'feature_importance.csv', index=False)
    logger.info(f"\nFeature importance saved to {output_dir / 'feature_importance.csv'}")


def main():
    parser = argparse.ArgumentParser(description='Train XGBoost on MedicalNet features')
    parser.add_argument('--train', type=str, required=True,
                       help='Path to training features CSV')
    parser.add_argument('--val', type=str, required=True,
                       help='Path to validation features CSV')
    parser.add_argument('--test', type=str, required=True,
                       help='Path to test features CSV')
    parser.add_argument('--output', type=str, default='results',
                       help='Output directory')

    args = parser.parse_args()

    train_xgboost(
        train_csv=args.train,
        val_csv=args.val,
        test_csv=args.test,
        output_dir=args.output
    )


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Train XGBoost for 4-class classification with Inverse Frequency Weighting

Classes:
    0: CN
    1: MCI-stable
    2: MCI→AD
    3: AD

Usage:
    python 02_train_xgboost_4class.py \
        --train-csv data/splits/train.csv \
        --val-csv data/splits/val.csv \
        --output-dir models
"""

import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
import json
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_sample_weights(y_train: np.ndarray, boost_mci_to_ad: float = 1.5) -> np.ndarray:
    """
    Compute sample weights using Inverse Frequency Weighting with MCI→AD boost

    Formula: weight_i = N_total / (N_classes * N_samples_in_class_i)

    BOOST: MCI→AD weight is multiplied by boost_mci_to_ad factor (default 1.5x)

    This gives higher weight to minority classes (MCI→AD, AD)
    and lower weight to majority classes (CN, MCI-stable)

    Args:
        y_train: Training labels
        boost_mci_to_ad: Multiplicative factor for MCI→AD class weight (default 1.5)

    Returns:
        Sample weights for each training sample
    """
    # Count samples per class
    unique_classes, class_counts = np.unique(y_train, return_counts=True)
    n_total = len(y_train)
    n_classes = len(unique_classes)

    logger.info("\nClass distribution in training set:")
    class_names = {0: 'CN', 1: 'MCI-stable', 2: 'MCI→AD', 3: 'AD'}

    # Compute inverse frequency weights
    class_weights = {}
    for cls, count in zip(unique_classes, class_counts):
        weight = n_total / (n_classes * count)

        # BOOST MCI→AD weight (class 2)
        if cls == 2:
            weight *= boost_mci_to_ad
            logger.info(f"  {class_names[cls]} ({cls}): {count} samples, base_weight: {weight/boost_mci_to_ad:.3f} → BOOSTED: {weight:.3f} (×{boost_mci_to_ad})")
        else:
            logger.info(f"  {class_names[cls]} ({cls}): {count} samples, weight: {weight:.3f}")

        class_weights[cls] = weight

    # Assign weight to each sample based on its class
    sample_weights = np.array([class_weights[y] for y in y_train])

    logger.info(f"\nSample weights statistics:")
    logger.info(f"  Min: {sample_weights.min():.3f}")
    logger.info(f"  Max: {sample_weights.max():.3f}")
    logger.info(f"  Mean: {sample_weights.mean():.3f}")

    return sample_weights


def train_xgboost_multiclass(
    X_train, y_train, X_val, y_val,
    sample_weights=None,
    params=None
):
    """
    Train XGBoost for multiclass classification

    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        sample_weights: Per-sample weights (inverse frequency)
        params: XGBoost parameters

    Returns:
        Trained XGBoost Booster
    """
    if params is None:
        params = {
            'objective': 'multi:softprob',  # Multiclass with probability
            'num_class': 4,                 # 4 classes
            'eval_metric': 'mlogloss',      # Multiclass log loss
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1,
            'gamma': 0,
            'reg_alpha': 0,
            'reg_lambda': 1,
            'random_state': 42,
            'n_jobs': -1,
            'tree_method': 'hist',
        }

    logger.info("\nTraining XGBoost 4-class classifier...")
    logger.info(f"Parameters: {params}")

    # Create DMatrix with sample weights
    dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weights)
    dval = xgb.DMatrix(X_val, label=y_val)

    # Training
    evals = [(dtrain, 'train'), (dval, 'val')]
    evals_result = {}

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=200,
        evals=evals,
        early_stopping_rounds=20,
        verbose_eval=10,
        evals_result=evals_result
    )

    logger.info(f"\n✓ Training completed!")
    logger.info(f"Best iteration: {model.best_iteration}")
    logger.info(f"Best score: {model.best_score:.4f}")

    return model


def evaluate_on_validation(model, X_val, y_val):
    """
    Evaluate model on validation set

    Args:
        model: Trained XGBoost model
        X_val, y_val: Validation data

    Returns:
        Predictions and metrics
    """
    logger.info("\n" + "="*80)
    logger.info("VALIDATION SET RESULTS")
    logger.info("="*80)

    # Predict
    dval = xgb.DMatrix(X_val)
    y_proba = model.predict(dval)
    y_pred = np.argmax(y_proba, axis=1)

    # Metrics
    accuracy = (y_pred == y_val).mean()
    balanced_acc = balanced_accuracy_score(y_val, y_pred)

    logger.info(f"\nAccuracy: {accuracy*100:.2f}%")
    logger.info(f"Balanced Accuracy: {balanced_acc*100:.2f}%")

    # Confusion matrix
    cm = confusion_matrix(y_val, y_pred)
    logger.info(f"\nConfusion Matrix:")
    logger.info(cm)

    # Classification report
    class_names = ['CN', 'MCI-stable', 'MCI→AD', 'AD']
    logger.info(f"\nClassification Report:")
    logger.info(classification_report(y_val, y_pred, target_names=class_names))

    logger.info("="*80)

    return y_pred, y_proba


def main():
    parser = argparse.ArgumentParser(description='Train XGBoost 4-class')
    parser.add_argument('--train-csv', type=str, required=True)
    parser.add_argument('--val-csv', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='models')
    parser.add_argument('--boost-mci-to-ad', type=float, default=1.5,
                        help='Boost factor for MCI→AD class weight (default: 1.5)')

    args = parser.parse_args()

    # Load data
    logger.info("Loading data...")
    train_df = pd.read_csv(args.train_csv)
    val_df = pd.read_csv(args.val_csv)

    # Separate features and labels
    X_train = train_df.drop('label', axis=1).values
    y_train = train_df['label'].values
    X_val = val_df.drop('label', axis=1).values
    y_val = val_df['label'].values

    feature_names = train_df.drop('label', axis=1).columns.tolist()

    logger.info(f"Train: {len(X_train)} samples, {len(feature_names)} features")
    logger.info(f"Val:   {len(X_val)} samples")

    # Normalize features
    logger.info("\nNormalizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Compute sample weights (Inverse Frequency Weighting with MCI→AD boost)
    sample_weights = compute_sample_weights(y_train, boost_mci_to_ad=args.boost_mci_to_ad)

    # Train XGBoost
    model = train_xgboost_multiclass(
        X_train_scaled, y_train,
        X_val_scaled, y_val,
        sample_weights=sample_weights
    )

    # Evaluate on validation
    y_pred, y_proba = evaluate_on_validation(model, X_val_scaled, y_val)

    # Save model and artifacts
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / 'xgboost_model_4class.json'
    model.save_model(str(model_path))
    logger.info(f"\n✓ Model saved to {model_path}")

    scaler_path = output_dir / 'scaler.pkl'
    joblib.dump(scaler, scaler_path)
    logger.info(f"✓ Scaler saved to {scaler_path}")

    feature_names_path = output_dir / 'feature_names.json'
    with open(feature_names_path, 'w') as f:
        json.dump(feature_names, f)
    logger.info(f"✓ Feature names saved to {feature_names_path}")

    # Save feature importance
    importance = model.get_score(importance_type='weight')
    importance_df = pd.DataFrame([
        {'feature': k, 'importance': v}
        for k, v in importance.items()
    ]).sort_values('importance', ascending=False)

    importance_path = output_dir / 'feature_importance.csv'
    importance_df.to_csv(importance_path, index=False)
    logger.info(f"✓ Feature importance saved to {importance_path}")

    logger.info("\nTop 10 most important features:")
    for idx, row in importance_df.head(10).iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.2f}")

    # Final summary
    accuracy = (y_pred == y_val).mean()
    balanced_acc = balanced_accuracy_score(y_val, y_pred)

    logger.info("\n" + "="*80)
    logger.info("TRAINING COMPLETED!")
    logger.info(f"Validation Accuracy: {accuracy*100:.2f}%")
    logger.info(f"Validation Balanced Accuracy: {balanced_acc*100:.2f}%")
    logger.info("="*80)


if __name__ == '__main__':
    main()

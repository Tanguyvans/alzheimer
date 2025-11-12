#!/usr/bin/env python3
"""
Train XGBoost for CN vs AD+MCI-to-AD binary classification

Usage:
    python 02_train_xgboost.py \
        --train-csv data/splits/train.csv \
        --val-csv data/splits/val.csv \
        --output-dir models \
        --tune-hyperparameters
"""

import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data(train_csv: str, val_csv: str):
    """Load and prepare training and validation data"""
    logger.info("Loading data...")

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)

    # Separate features and labels
    X_train = train_df.drop('label', axis=1)
    y_train = train_df['label']

    X_val = val_df.drop('label', axis=1)
    y_val = val_df['label']

    logger.info(f"Train: {len(X_train)} samples (CN: {(y_train==0).sum()}, AD: {(y_train==1).sum()})")
    logger.info(f"Val:   {len(X_val)} samples (CN: {(y_val==0).sum()}, AD: {(y_val==1).sum()})")

    return X_train, y_train, X_val, y_val


def normalize_features(X_train, X_val, X_test=None):
    """Standardize features using StandardScaler"""
    logger.info("Normalizing features...")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_val_scaled, X_test_scaled, scaler

    return X_train_scaled, X_val_scaled, scaler


def train_xgboost(
    X_train,
    y_train,
    X_val,
    y_val,
    params: dict = None,
    use_class_weights: bool = True
):
    """
    Train XGBoost classifier

    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        params: XGBoost parameters
        use_class_weights: Whether to use class weights for imbalance
    """
    logger.info("Training XGBoost classifier...")

    # Compute class weights
    if use_class_weights:
        class_counts = np.bincount(y_train)
        scale_pos_weight = class_counts[0] / class_counts[1]  # CN count / AD count
        logger.info(f"Using scale_pos_weight: {scale_pos_weight:.2f}")
    else:
        scale_pos_weight = 1.0

    # Default parameters (optimized for tabular data)
    if params is None:
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 200,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1,
            'gamma': 0,
            'reg_alpha': 0,
            'reg_lambda': 1,
            'scale_pos_weight': scale_pos_weight,
            'random_state': 42,
            'n_jobs': -1,
            'tree_method': 'hist',  # Faster training
        }

    # Create DMatrix for faster training
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    # Training with early stopping
    evals = [(dtrain, 'train'), (dval, 'val')]

    logger.info("Starting training...")
    logger.info(f"Parameters: {params}")

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=params.pop('n_estimators', 200),
        evals=evals,
        early_stopping_rounds=20,
        verbose_eval=10
    )

    logger.info(f"✓ Training completed!")
    logger.info(f"Best iteration: {model.best_iteration}")
    logger.info(f"Best score: {model.best_score:.4f}")

    return model


def evaluate_model(model, X, y, dataset_name='Test'):
    """Evaluate model and print metrics"""
    # Predict
    dmatrix = xgb.DMatrix(X)
    y_proba = model.predict(dmatrix)
    y_pred = (y_proba >= 0.5).astype(int)

    # Calculate metrics
    accuracy = accuracy_score(y, y_pred)
    balanced_acc = balanced_accuracy_score(y, y_pred)

    logger.info(f"\n{'='*80}")
    logger.info(f"{dataset_name.upper()} SET RESULTS")
    logger.info(f"{'='*80}")
    logger.info(f"Accuracy: {accuracy*100:.2f}%")
    logger.info(f"Balanced Accuracy: {balanced_acc*100:.2f}%")

    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    logger.info(f"\nConfusion Matrix:")
    logger.info(cm)

    # Classification report
    logger.info(f"\nClassification Report:")
    report = classification_report(y, y_pred, target_names=['CN', 'AD+MCI-to-AD'])
    logger.info(report)

    return {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'predictions': y_pred,
        'probabilities': y_proba
    }


def tune_hyperparameters(X_train, y_train, X_val, y_val):
    """
    Simple grid search for hyperparameter tuning

    Returns best parameters
    """
    logger.info("\n" + "="*80)
    logger.info("HYPERPARAMETER TUNING")
    logger.info("="*80)

    # Define parameter grid
    param_grid = {
        'max_depth': [3, 6, 9],
        'learning_rate': [0.01, 0.1, 0.3],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
    }

    best_score = 0
    best_params = None

    # Compute class weight
    class_counts = np.bincount(y_train)
    scale_pos_weight = class_counts[0] / class_counts[1]

    # Grid search
    total_combinations = np.prod([len(v) for v in param_grid.values()])
    logger.info(f"Testing {total_combinations} parameter combinations...")

    current = 0
    for max_depth in param_grid['max_depth']:
        for lr in param_grid['learning_rate']:
            for subsample in param_grid['subsample']:
                for colsample in param_grid['colsample_bytree']:
                    current += 1

                    params = {
                        'objective': 'binary:logistic',
                        'eval_metric': 'logloss',
                        'max_depth': max_depth,
                        'learning_rate': lr,
                        'subsample': subsample,
                        'colsample_bytree': colsample,
                        'scale_pos_weight': scale_pos_weight,
                        'random_state': 42,
                        'n_jobs': -1,
                        'tree_method': 'hist',
                    }

                    dtrain = xgb.DMatrix(X_train, label=y_train)
                    dval = xgb.DMatrix(X_val, label=y_val)

                    # Train with early stopping
                    model = xgb.train(
                        params,
                        dtrain,
                        num_boost_round=200,
                        evals=[(dval, 'val')],
                        early_stopping_rounds=20,
                        verbose_eval=False
                    )

                    # Evaluate
                    y_proba = model.predict(dval)
                    y_pred = (y_proba >= 0.5).astype(int)
                    score = balanced_accuracy_score(y_val, y_pred)

                    logger.info(f"[{current}/{total_combinations}] depth={max_depth}, lr={lr}, subsample={subsample}, colsample={colsample} → Bal Acc: {score*100:.2f}%")

                    if score > best_score:
                        best_score = score
                        best_params = params.copy()
                        best_params['n_estimators'] = model.best_iteration

    logger.info(f"\n✓ Best parameters found!")
    logger.info(f"Best balanced accuracy: {best_score*100:.2f}%")
    logger.info(f"Best parameters: {best_params}")

    return best_params


def main():
    parser = argparse.ArgumentParser(description='Train XGBoost for AD classification')
    parser.add_argument('--train-csv', type=str, required=True)
    parser.add_argument('--val-csv', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='models')
    parser.add_argument('--tune-hyperparameters', action='store_true',
                        help='Perform hyperparameter tuning')
    parser.add_argument('--use-class-weights', action='store_true', default=True,
                        help='Use class weights for imbalanced data')

    args = parser.parse_args()

    # Load data
    X_train, y_train, X_val, y_val = load_data(args.train_csv, args.val_csv)

    # Normalize features
    X_train_scaled, X_val_scaled, scaler = normalize_features(X_train, X_val)

    # Hyperparameter tuning (optional)
    if args.tune_hyperparameters:
        best_params = tune_hyperparameters(X_train_scaled, y_train, X_val_scaled, y_val)
    else:
        best_params = None

    # Train final model
    model = train_xgboost(
        X_train_scaled,
        y_train,
        X_val_scaled,
        y_val,
        params=best_params,
        use_class_weights=args.use_class_weights
    )

    # Evaluate on validation set
    val_results = evaluate_model(model, X_val_scaled, y_val, 'Validation')

    # Save model and scaler
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / 'xgboost_model.json'
    model.save_model(str(model_path))
    logger.info(f"\n✓ Model saved to {model_path}")

    scaler_path = output_dir / 'scaler.pkl'
    joblib.dump(scaler, scaler_path)
    logger.info(f"✓ Scaler saved to {scaler_path}")

    # Save feature names
    feature_names = list(X_train.columns)
    feature_names_path = output_dir / 'feature_names.json'
    with open(feature_names_path, 'w') as f:
        json.dump(feature_names, f, indent=2)
    logger.info(f"✓ Feature names saved to {feature_names_path}")

    # Save feature importance
    importance_dict = model.get_score(importance_type='gain')
    importance_df = pd.DataFrame({
        'feature': list(importance_dict.keys()),
        'importance': list(importance_dict.values())
    }).sort_values('importance', ascending=False)

    importance_path = output_dir / 'feature_importance.csv'
    importance_df.to_csv(importance_path, index=False)
    logger.info(f"✓ Feature importance saved to {importance_path}")

    logger.info(f"\nTop 10 most important features:")
    for idx, row in importance_df.head(10).iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.2f}")

    logger.info("\n" + "="*80)
    logger.info("TRAINING COMPLETED!")
    logger.info(f"Validation Accuracy: {val_results['accuracy']*100:.2f}%")
    logger.info(f"Validation Balanced Accuracy: {val_results['balanced_accuracy']*100:.2f}%")
    logger.info("="*80)


if __name__ == '__main__':
    main()

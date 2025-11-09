#!/usr/bin/env python3
"""
Make predictions using ensemble of models

This script:
1. Loads all trained models
2. Gets predictions from each model
3. Averages probability outputs
4. Evaluates ensemble performance
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'cn_mci_ad_medicalnet'))

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import logging
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix
from model_resnet3d import resnet18
from model_seresnet3d import seresnet18
from dataset import ADNIDataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_ensemble_models(checkpoints_dir: str, num_models: int, device, use_se: bool = True):
    """Load all models in the ensemble"""
    models = []
    checkpoints_dir = Path(checkpoints_dir)

    for i in range(1, num_models + 1):
        checkpoint_path = checkpoints_dir / f'model_{i}_best.pth'

        if not checkpoint_path.exists():
            logger.warning(f"Checkpoint not found: {checkpoint_path}")
            continue

        # Load SEResNet or vanilla ResNet
        if use_se:
            model = seresnet18(num_classes=2, in_channels=1, use_se=True)
        else:
            model = resnet18(num_classes=2, in_channels=1)

        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model = model.to(device)
        model.eval()
        models.append(model)

        logger.info(f"Loaded model {i} from {checkpoint_path}")

    return models


@torch.no_grad()
def predict_ensemble(models, dataloader, device):
    """Get ensemble predictions"""
    all_labels = []
    all_probs = []

    for images, labels in tqdm(dataloader, desc="Predicting"):
        images = images.to(device)
        all_labels.extend(labels.numpy())

        # Get predictions from all models
        batch_probs = []
        for model in models:
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            batch_probs.append(probs.cpu().numpy())

        # Average probabilities across models
        avg_probs = np.mean(batch_probs, axis=0)
        all_probs.extend(avg_probs)

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_preds = np.argmax(all_probs, axis=1)

    return all_labels, all_preds, all_probs


def evaluate_ensemble(
    checkpoints_dir: str,
    test_csv: str,
    num_models: int,
    batch_size: int = 4,
    num_workers: int = 4
):
    """Evaluate ensemble on test set"""
    logger.info("="*80)
    logger.info("ENSEMBLE EVALUATION")
    logger.info("="*80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load models
    logger.info(f"\nLoading {num_models} models...")
    models = load_ensemble_models(checkpoints_dir, num_models, device)
    logger.info(f"Loaded {len(models)} models successfully")

    if len(models) == 0:
        logger.error("No models loaded. Exiting.")
        return

    # Load test data
    test_dataset = ADNIDataset(test_csv, target_shape=(192, 192, 192))
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # Get predictions
    logger.info(f"\nMaking predictions on {len(test_dataset)} test samples...")
    labels, preds, probs = predict_ensemble(models, test_loader, device)

    # Calculate metrics
    acc = accuracy_score(labels, preds)
    balanced_acc = balanced_accuracy_score(labels, preds)

    logger.info("\n" + "="*80)
    logger.info("ENSEMBLE RESULTS")
    logger.info("="*80)
    logger.info(f"Number of models: {len(models)}")
    logger.info(f"Test samples: {len(labels)}")
    logger.info(f"Accuracy: {acc*100:.2f}%")
    logger.info(f"Balanced Accuracy: {balanced_acc*100:.2f}%")

    # Confusion matrix
    cm = confusion_matrix(labels, preds)
    logger.info(f"\nConfusion Matrix:")
    logger.info(f"{cm}")

    # Classification report
    logger.info(f"\nClassification Report:")
    report = classification_report(labels, preds, target_names=['CN', 'AD'])
    logger.info(f"\n{report}")

    # Save predictions
    df = pd.read_csv(test_csv)
    df['predicted_label'] = preds
    df['predicted_CN_proba'] = probs[:, 0]
    df['predicted_AD_proba'] = probs[:, 1]

    output_path = Path(checkpoints_dir).parent / 'ensemble_predictions.csv'
    df.to_csv(output_path, index=False)
    logger.info(f"\nPredictions saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate ensemble of models')
    parser.add_argument('--checkpoints-dir', type=str, required=True,
                       help='Directory with model checkpoints')
    parser.add_argument('--test-csv', type=str, required=True,
                       help='Test CSV')
    parser.add_argument('--num-models', type=int, default=5,
                       help='Number of models in ensemble')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of dataloader workers')

    args = parser.parse_args()

    evaluate_ensemble(
        checkpoints_dir=args.checkpoints_dir,
        test_csv=args.test_csv,
        num_models=args.num_models,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )


if __name__ == '__main__':
    main()

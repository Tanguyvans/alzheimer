#!/usr/bin/env python3
"""
Evaluate 3D Vision Transformer on Test Set

Usage:
    python predict_vit3d.py \
        --checkpoint checkpoints_vit3d/best_model.pth \
        --test-csv data/splits_nppy/test.csv \
        --model-size small \
        --batch-size 4
"""

import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score

from dataset import AlzheimerDataset
from model_vit3d import vit3d_small, vit3d_base, vit3d_large

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_model(model, dataloader, device):
    """Evaluate model and return predictions"""
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Predicting"):
            images = images.to(device)

            # Forward pass
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

            # Get predictions
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def main():
    parser = argparse.ArgumentParser(description='Evaluate 3D-ViT on test set')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--test-csv', type=str, required=True,
                        help='Path to test CSV')
    parser.add_argument('--model-size', type=str, default='small',
                        choices=['small', 'base', 'large'],
                        help='Model size (must match checkpoint)')
    parser.add_argument('--patch-size', type=int, default=16,
                        help='Patch size (must match checkpoint)')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size for inference')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of dataloader workers')
    parser.add_argument('--output-csv', type=str, default='vit3d_predictions.csv',
                        help='Output CSV file for predictions')

    args = parser.parse_args()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Create model
    logger.info(f"Creating 3D-ViT-{args.model_size.capitalize()}/16 model...")
    if args.model_size == 'small':
        model = vit3d_small(num_classes=2, img_size=192, patch_size=args.patch_size)
    elif args.model_size == 'base':
        model = vit3d_base(num_classes=2, img_size=192, patch_size=args.patch_size)
    else:
        model = vit3d_large(num_classes=2, img_size=192, patch_size=args.patch_size)

    # Load checkpoint
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    if 'val_acc' in checkpoint:
        logger.info(f"Checkpoint validation accuracy: {checkpoint['val_acc']:.2f}%")

    # Create dataset
    logger.info("Loading test dataset...")
    test_dataset = AlzheimerDataset(args.test_csv, augment=False)
    logger.info(f"Test samples: {len(test_dataset)}")

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Evaluate
    logger.info("\n" + "="*80)
    logger.info("EVALUATION RESULTS")
    logger.info("="*80)

    labels, preds, probs = evaluate_model(model, test_loader, device)

    # Calculate metrics
    accuracy = 100. * (preds == labels).sum() / len(labels)
    balanced_acc = 100. * balanced_accuracy_score(labels, preds)

    logger.info(f"Test samples: {len(labels)}")
    logger.info(f"Accuracy: {accuracy:.2f}%")
    logger.info(f"Balanced Accuracy: {balanced_acc:.2f}%")

    # Confusion matrix
    logger.info("\nConfusion Matrix:")
    cm = confusion_matrix(labels, preds)
    logger.info(cm)

    # Classification report
    logger.info("\nClassification Report:")
    target_names = ['CN', 'AD']
    report = classification_report(labels, preds, target_names=target_names)
    logger.info("\n" + report)

    # Save predictions
    logger.info(f"\nSaving predictions to {args.output_csv}...")

    # Load original CSV to get scan paths
    df = pd.read_csv(args.test_csv)

    # Add predictions
    df['predicted_label'] = preds
    df['predicted_CN_proba'] = probs[:, 0]
    df['predicted_AD_proba'] = probs[:, 1]

    # Add predicted class name
    df['predicted_class'] = df['predicted_label'].map({0: 'CN', 1: 'AD'})
    df['true_class'] = df['label'].map({0: 'CN', 1: 'AD'})

    # Save
    df.to_csv(args.output_csv, index=False)
    logger.info(f"✓ Predictions saved!")

    logger.info("\n" + "="*80)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Step 2: Train Siamese Network for longitudinal MRI analysis.

Trains a Siamese network to compare baseline and follow-up MRI scans
and predict conversion to Alzheimer's disease.

Usage:
    python 02_train.py
    python 02_train.py --epochs 100 --batch-size 4
"""

import argparse
import logging
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt

from model import SiameseNetwork, WeightedSiameseNetwork
from dataset import PairedMRIDataset, RandomAugmentation3D

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
EXPERIMENT_DIR = Path(__file__).parent
DATA_DIR = EXPERIMENT_DIR / "data"
RESULTS_DIR = EXPERIMENT_DIR / "results"


class EarlyStopping:
    """Early stopping to prevent overfitting."""

    def __init__(self, patience=15, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
        return self.early_stop


def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for batch in loader:
        baseline = batch['baseline'].to(device)
        followup = batch['followup'].to(device)
        labels = batch['label'].to(device)
        time_delta = batch['time_delta'].to(device)

        optimizer.zero_grad()
        logits = model(baseline, followup, time_delta)
        loss = criterion(logits, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)

    return avg_loss, accuracy, balanced_acc


def validate(model, loader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            baseline = batch['baseline'].to(device)
            followup = batch['followup'].to(device)
            labels = batch['label'].to(device)
            time_delta = batch['time_delta'].to(device)

            logits = model(baseline, followup, time_delta)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of conversion
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)

    # AUC if binary classification
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.0

    return avg_loss, accuracy, balanced_acc, auc, all_preds, all_labels


def plot_training_curves(history, output_path):
    """Plot training curves."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Loss
    axes[0].plot(history['train_loss'], label='Train', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(history['train_acc'], label='Train', linewidth=2)
    axes[1].plot(history['val_acc'], label='Val', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Balanced Accuracy
    axes[2].plot(history['train_bal_acc'], label='Train', linewidth=2)
    axes[2].plot(history['val_bal_acc'], label='Val', linewidth=2)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Balanced Accuracy')
    axes[2].set_title('Balanced Accuracy')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved training curves to {output_path}")


def plot_confusion_matrix(y_true, y_pred, output_path):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap='Blues')

    labels = ['Non-Converter', 'Converter']
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    # Add text annotations
    for i in range(len(labels)):
        for j in range(len(labels)):
            text = ax.text(j, i, cm[i, j], ha='center', va='center',
                          fontsize=20, fontweight='bold',
                          color='white' if cm[i, j] > cm.max()/2 else 'black')

    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_title('Siamese Network - Converter Detection', fontsize=14, fontweight='bold')

    # Add colorbar
    plt.colorbar(im)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved confusion matrix to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Train Siamese Network')
    parser.add_argument('--pairs-csv', type=str, default=None, help='Path to pairs CSV')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience')
    parser.add_argument('--target-shape', type=int, nargs=3, default=[64, 64, 64], help='Target volume shape')
    parser.add_argument('--embedding-dim', type=int, default=256, help='Embedding dimension')
    parser.add_argument('--weighted', action='store_true', help='Use time-weighted model')
    parser.add_argument('--num-workers', type=int, default=4, help='Data loading workers')
    args = parser.parse_args()

    # Create output directory
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else
                         'mps' if torch.backends.mps.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Find pairs CSV
    pairs_csv = args.pairs_csv or DATA_DIR / 'pairs.csv'
    if not Path(pairs_csv).exists():
        logger.error(f"Pairs CSV not found: {pairs_csv}")
        logger.error("Run 01_prepare_pairs.py first to create the paired dataset")
        return

    # Load and split data
    import pandas as pd
    from sklearn.model_selection import train_test_split

    pairs_df = pd.read_csv(pairs_csv)
    logger.info(f"Loaded {len(pairs_df)} pairs")

    # Stratified split
    labels = pairs_df['is_converter'].values
    train_idx, temp_idx = train_test_split(
        np.arange(len(pairs_df)), test_size=0.3, stratify=labels, random_state=42
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, stratify=labels[temp_idx], random_state=42
    )

    # Save splits
    for name, idx in [('train', train_idx), ('val', val_idx), ('test', test_idx)]:
        split_df = pairs_df.iloc[idx]
        split_path = DATA_DIR / f'{name}_pairs.csv'
        split_df.to_csv(split_path, index=False)
        logger.info(f"{name}: {len(split_df)} pairs (converters: {split_df['is_converter'].sum()})")

    # Create datasets
    target_shape = tuple(args.target_shape)

    train_dataset = PairedMRIDataset(
        DATA_DIR / 'train_pairs.csv',
        transform=RandomAugmentation3D(flip_prob=0.5, noise_std=0.01),
        target_shape=target_shape
    )

    val_dataset = PairedMRIDataset(
        DATA_DIR / 'val_pairs.csv',
        transform=None,
        target_shape=target_shape
    )

    test_dataset = PairedMRIDataset(
        DATA_DIR / 'test_pairs.csv',
        transform=None,
        target_shape=target_shape
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    # Create model
    ModelClass = WeightedSiameseNetwork if args.weighted else SiameseNetwork
    model = ModelClass(
        in_channels=1,
        base_channels=32,
        embedding_dim=args.embedding_dim,
        num_classes=2  # Binary: converter vs non-converter
    ).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {total_params:,} (trainable: {trainable_params:,})")

    # Class weights for imbalanced data
    n_converters = labels.sum()
    n_non_converters = len(labels) - n_converters
    class_weights = torch.tensor([n_converters, n_non_converters], dtype=torch.float32)
    class_weights = class_weights / class_weights.sum()
    class_weights = class_weights.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    early_stopping = EarlyStopping(patience=args.patience)

    # Training history
    history = {
        'train_loss': [], 'train_acc': [], 'train_bal_acc': [],
        'val_loss': [], 'val_acc': [], 'val_bal_acc': [], 'val_auc': []
    }

    best_val_bal_acc = 0
    best_epoch = 0

    logger.info("Starting training...")
    logger.info(f"Epochs: {args.epochs}, Batch size: {args.batch_size}, LR: {args.lr}")

    for epoch in range(args.epochs):
        # Train
        train_loss, train_acc, train_bal_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_loss, val_acc, val_bal_acc, val_auc, _, _ = validate(
            model, val_loader, criterion, device
        )

        scheduler.step()

        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_bal_acc'].append(train_bal_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_bal_acc'].append(val_bal_acc)
        history['val_auc'].append(val_auc)

        # Log progress
        logger.info(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.3f}, Bal: {train_bal_acc:.3f} | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.3f}, Bal: {val_bal_acc:.3f}, AUC: {val_auc:.3f}"
        )

        # Save best model
        if val_bal_acc > best_val_bal_acc:
            best_val_bal_acc = val_bal_acc
            best_epoch = epoch + 1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_bal_acc': val_bal_acc,
                'val_auc': val_auc,
            }, RESULTS_DIR / 'best_model.pt')
            logger.info(f"  -> New best model (balanced acc: {val_bal_acc:.3f})")

        # Early stopping
        if early_stopping(val_bal_acc):
            logger.info(f"Early stopping at epoch {epoch+1}")
            break

    # Load best model for test evaluation
    checkpoint = torch.load(RESULTS_DIR / 'best_model.pt', map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Test evaluation
    test_loss, test_acc, test_bal_acc, test_auc, test_preds, test_labels = validate(
        model, test_loader, criterion, device
    )

    logger.info("\n" + "=" * 60)
    logger.info("TEST RESULTS")
    logger.info("=" * 60)
    logger.info(f"Accuracy: {test_acc:.3f}")
    logger.info(f"Balanced Accuracy: {test_bal_acc:.3f}")
    logger.info(f"AUC: {test_auc:.3f}")

    # Plot training curves
    plot_training_curves(history, RESULTS_DIR / 'training_curves.png')

    # Plot confusion matrix
    plot_confusion_matrix(test_labels, test_preds, RESULTS_DIR / 'confusion_matrix.png')

    # Save results
    results = {
        'best_epoch': best_epoch,
        'best_val_bal_acc': float(best_val_bal_acc),
        'test_accuracy': float(test_acc),
        'test_balanced_accuracy': float(test_bal_acc),
        'test_auc': float(test_auc),
        'config': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'target_shape': args.target_shape,
            'embedding_dim': args.embedding_dim,
            'weighted': args.weighted,
        },
        'timestamp': datetime.now().isoformat()
    }

    with open(RESULTS_DIR / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to {RESULTS_DIR}")

    # Print summary
    print("\n" + "=" * 60)
    print("SIAMESE NETWORK TRAINING COMPLETE")
    print("=" * 60)
    print(f"\nBest epoch: {best_epoch}")
    print(f"Best validation balanced accuracy: {best_val_bal_acc:.1%}")
    print(f"\nTest Results:")
    print(f"  Accuracy: {test_acc:.1%}")
    print(f"  Balanced Accuracy: {test_bal_acc:.1%}")
    print(f"  AUC: {test_auc:.3f}")


if __name__ == '__main__':
    main()

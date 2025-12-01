#!/usr/bin/env python3
"""
Step 2: Train Siamese Network for longitudinal MRI analysis.

Trains a Siamese network to compare baseline and follow-up MRI scans
and predict conversion to Alzheimer's disease.

Usage:
    python 02_train.py --config config.yaml
"""

import argparse
import logging
import json
import yaml
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

from model import SiameseNetwork, WeightedSiameseNetwork
from dataset import PairedMRIDataset, RandomAugmentation3D

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

EXPERIMENT_DIR = Path(__file__).parent


def load_config(config_path: str = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = EXPERIMENT_DIR / "config.yaml"

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def get_device(config: dict) -> torch.device:
    """Get compute device."""
    device_str = config['hardware']['device']

    if device_str == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
            logger.info("Using Apple MPS")
        else:
            device = torch.device('cpu')
            logger.info("Using CPU")
    elif device_str == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif device_str == 'mps' and torch.backends.mps.is_available():
        device = torch.device('mps')
        logger.info("Using Apple MPS")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU")

    return device


class EarlyStopping:
    """Early stopping to prevent overfitting."""

    def __init__(self, patience: int = 15, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, score: float) -> bool:
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


def train_epoch(model, loader, criterion, optimizer, device, max_grad_norm: float = 1.0):
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
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
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.0

    return avg_loss, accuracy, balanced_acc, auc, all_preds, all_labels


def plot_training_curves(history: dict, output_path: Path):
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


def plot_confusion_matrix(y_true, y_pred, class_names: list, output_path: Path):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap='Blues')

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, fontsize=11)
    ax.set_yticklabels(class_names, fontsize=11)

    # Add text annotations
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            text_color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                   fontsize=20, fontweight='bold', color=text_color)

    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_title('Siamese Network - Converter Detection', fontsize=14, fontweight='bold')

    # Add accuracy annotation
    accuracy = np.trace(cm) / cm.sum()
    ax.annotate(f'Accuracy: {accuracy:.1%}',
                xy=(0.02, 0.98), xycoords='axes fraction',
                fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray'))

    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved confusion matrix to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Train Siamese Network')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file path')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Setup paths
    data_dir = EXPERIMENT_DIR / config['data']['pairs_dir']
    results_dir = EXPERIMENT_DIR / config['data']['results_dir']
    results_dir.mkdir(parents=True, exist_ok=True)

    # Device
    device = get_device(config)

    # Check pairs CSV
    pairs_csv = EXPERIMENT_DIR / config['data']['pairs_csv']
    if not pairs_csv.exists():
        logger.error(f"Pairs CSV not found: {pairs_csv}")
        logger.error("Run 01_prepare_pairs.py --config config.yaml first")
        return

    # Load and split data
    pairs_df = pd.read_csv(pairs_csv)
    logger.info(f"Loaded {len(pairs_df)} pairs")

    # Stratified split
    seed = config['training']['seed']
    train_ratio = config['splits']['train_ratio']
    val_ratio = config['splits']['val_ratio']
    test_ratio = config['splits']['test_ratio']

    labels = pairs_df['is_converter'].values
    train_idx, temp_idx = train_test_split(
        np.arange(len(pairs_df)),
        test_size=(val_ratio + test_ratio),
        stratify=labels,
        random_state=seed
    )
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=test_ratio / (val_ratio + test_ratio),
        stratify=labels[temp_idx],
        random_state=seed
    )

    # Save splits
    for name, idx in [('train', train_idx), ('val', val_idx), ('test', test_idx)]:
        split_df = pairs_df.iloc[idx]
        split_path = data_dir / f'{name}_pairs.csv'
        split_df.to_csv(split_path, index=False)
        logger.info(f"{name}: {len(split_df)} pairs (converters: {split_df['is_converter'].sum()})")

    # Create datasets
    target_shape = tuple(config['model']['target_shape'])
    aug_config = config['augmentation']

    train_transform = RandomAugmentation3D(
        flip_prob=aug_config['random_flip_prob'],
        noise_std=aug_config['noise_std']
    ) if aug_config['enabled'] else None

    train_dataset = PairedMRIDataset(
        data_dir / 'train_pairs.csv',
        transform=train_transform,
        normalize=aug_config['normalize'],
        target_shape=target_shape
    )

    val_dataset = PairedMRIDataset(
        data_dir / 'val_pairs.csv',
        transform=None,
        normalize=aug_config['normalize'],
        target_shape=target_shape
    )

    test_dataset = PairedMRIDataset(
        data_dir / 'test_pairs.csv',
        transform=None,
        normalize=aug_config['normalize'],
        target_shape=target_shape
    )

    # Create dataloaders
    batch_size = config['training']['batch_size']
    num_workers = config['hardware']['num_workers']

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=config['hardware']['pin_memory']
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=config['hardware']['pin_memory']
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=config['hardware']['pin_memory']
    )

    # Create model
    model_config = config['model']
    ModelClass = WeightedSiameseNetwork if model_config['use_time_weighting'] else SiameseNetwork
    model = ModelClass(
        in_channels=model_config['in_channels'],
        base_channels=model_config['base_channels'],
        embedding_dim=model_config['embedding_dim'],
        num_classes=model_config['num_classes'],
        dropout=model_config['dropout']
    ).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {model_config['name']}")
    logger.info(f"Parameters: {total_params:,} (trainable: {trainable_params:,})")

    # Class weights for imbalanced data
    if config['training']['use_weighted_loss']:
        n_converters = labels.sum()
        n_non_converters = len(labels) - n_converters
        class_weights = torch.tensor([n_converters, n_non_converters], dtype=torch.float32)
        class_weights = class_weights / class_weights.sum()
        class_weights = class_weights.to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    # Optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config['training']['epochs'],
        eta_min=config['training']['lr_min']
    )

    # Early stopping
    early_stopping = EarlyStopping(
        patience=config['callbacks']['early_stopping']['patience'],
        min_delta=config['callbacks']['early_stopping']['min_delta']
    )

    # Training history
    history = {
        'train_loss': [], 'train_acc': [], 'train_bal_acc': [],
        'val_loss': [], 'val_acc': [], 'val_bal_acc': [], 'val_auc': []
    }

    best_val_bal_acc = 0
    best_epoch = 0
    epochs = config['training']['epochs']
    max_grad_norm = config['training']['max_grad_norm']

    logger.info("=" * 60)
    logger.info("STARTING TRAINING")
    logger.info("=" * 60)
    logger.info(f"Epochs: {epochs}, Batch size: {batch_size}, LR: {config['training']['learning_rate']}")

    for epoch in range(epochs):
        # Train
        train_loss, train_acc, train_bal_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, max_grad_norm
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
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.4f}, Bal: {train_bal_acc:.3f} | "
            f"Val Loss: {val_loss:.4f}, Bal: {val_bal_acc:.3f}, AUC: {val_auc:.3f}"
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
                'config': config
            }, results_dir / 'best_model.pt')
            logger.info(f"  -> New best model (balanced acc: {val_bal_acc:.3f})")

        # Early stopping
        if config['callbacks']['early_stopping']['enabled'] and early_stopping(val_bal_acc):
            logger.info(f"Early stopping at epoch {epoch+1}")
            break

    # Load best model for test evaluation
    checkpoint = torch.load(results_dir / 'best_model.pt', map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Test evaluation
    test_loss, test_acc, test_bal_acc, test_auc, test_preds, test_labels = validate(
        model, test_loader, criterion, device
    )

    logger.info("\n" + "=" * 60)
    logger.info("TEST RESULTS")
    logger.info("=" * 60)
    logger.info(f"Accuracy: {test_acc:.4f}")
    logger.info(f"Balanced Accuracy: {test_bal_acc:.4f}")
    logger.info(f"AUC: {test_auc:.4f}")

    # Classification report
    class_names = config['classes']['names']
    report = classification_report(test_labels, test_preds, target_names=class_names, digits=4)
    logger.info(f"\nClassification Report:\n{report}")

    # Plot results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    if config['evaluation']['plot_training_curves']:
        plot_training_curves(history, results_dir / f'training_curves_{timestamp}.png')

    if config['evaluation']['plot_confusion_matrix']:
        plot_confusion_matrix(test_labels, test_preds, class_names, results_dir / f'confusion_matrix_{timestamp}.png')

    # Save results
    results = {
        'best_epoch': best_epoch,
        'best_val_bal_acc': float(best_val_bal_acc),
        'test_accuracy': float(test_acc),
        'test_balanced_accuracy': float(test_bal_acc),
        'test_auc': float(test_auc),
        'config': config,
        'timestamp': timestamp
    }

    with open(results_dir / f'results_{timestamp}.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"\nResults saved to {results_dir}")

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

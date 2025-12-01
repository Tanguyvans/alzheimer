#!/usr/bin/env python3
"""
MedTransformer Training Script

4-Class Alzheimer's Classification: CN | MCI_stable | MCI_to_AD | AD
Using multi-view 2D slices with Vision Transformer backbone.

Reference: "MedTransformer: Accurate AD Diagnosis for 3D MRI Images
through 2D Vision Transformers" (arXiv 2024)

Usage:
    python train.py --config config.yaml
"""

import os
import sys
import yaml
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
from tqdm import tqdm
import matplotlib.pyplot as plt

from model import MedTransformer, MedTransformerLite
from dataset import get_dataloaders, CLASS_NAMES

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def plot_confusion_matrix(cm, class_names, output_path):
    """Plot and save confusion matrix as PNG."""
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, cmap='Blues')

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, fontsize=11, rotation=45, ha='right')
    ax.set_yticklabels(class_names, fontsize=11)

    # Add text annotations
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            val = cm[i, j]
            text_color = 'white' if val > cm.max() / 2 else 'black'
            ax.text(j, i, str(val), ha='center', va='center',
                   fontsize=14, fontweight='bold', color=text_color)

    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_title('MedTransformer - 4-Class Confusion Matrix', fontsize=14, fontweight='bold')

    # Add colorbar
    plt.colorbar(im, ax=ax)

    # Add accuracy annotation
    accuracy = np.trace(cm) / cm.sum()
    ax.annotate(f'Accuracy: {accuracy:.1%}',
                xy=(0.02, 0.98), xycoords='axes fraction',
                fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray'))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved confusion matrix to {output_path}")


def get_device(config: dict) -> torch.device:
    """Get compute device"""
    device_str = config['hardware']['device']
    if device_str == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif device_str == 'mps' and torch.backends.mps.is_available():
        device = torch.device('mps')
        logger.info("Using Apple MPS")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU")
    return device


def build_model(config: dict, device: torch.device) -> nn.Module:
    """Build MedTransformer model"""
    model_config = config['model']
    model_name = model_config.get('name', 'MedTransformer')

    kwargs = {
        'num_classes': model_config['num_classes'],
        'num_slices': model_config['num_slices'],
        'embed_dim': model_config['embed_dim'],
        'num_heads': model_config['num_heads'],
        'num_layers': model_config['num_layers'],
        'dropout': model_config['dropout'],
        'backbone': model_config['backbone'],
        'pretrained': model_config['pretrained'],
        'use_cross_attention': model_config.get('use_cross_attention', True)
    }

    if model_name == 'MedTransformerLite':
        model = MedTransformerLite(**kwargs)
    else:
        model = MedTransformer(**kwargs)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {model_name}")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")

    return model.to(device)


def get_optimizer(model: nn.Module, config: dict) -> optim.Optimizer:
    """Create optimizer"""
    train_config = config['training']

    if train_config['optimizer'] == 'AdamW':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=train_config['learning_rate'],
            weight_decay=train_config['weight_decay']
        )
    else:
        optimizer = optim.Adam(
            model.parameters(),
            lr=train_config['learning_rate'],
            weight_decay=train_config['weight_decay']
        )

    return optimizer


def get_scheduler(optimizer: optim.Optimizer, config: dict, steps_per_epoch: int):
    """Create learning rate scheduler with warmup"""
    train_config = config['training']

    warmup_epochs = train_config.get('warmup_epochs', 5)
    total_epochs = train_config['epochs']
    lr_min = train_config.get('lr_min', 1e-6)

    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = total_epochs * steps_per_epoch

    # Linear warmup
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_steps
    )

    # Cosine annealing after warmup
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_steps - warmup_steps,
        eta_min=lr_min
    )

    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps]
    )

    return scheduler


def train_epoch(
    model: nn.Module,
    train_loader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    scaler: GradScaler,
    device: torch.device,
    config: dict,
    epoch: int
) -> dict:
    """Train for one epoch"""
    model.train()

    accumulation_steps = config['training'].get('accumulation_steps', 1)
    use_amp = config['hardware'].get('mixed_precision', True) and device.type == 'cuda'

    running_loss = 0.0
    all_preds = []
    all_labels = []

    optimizer.zero_grad()

    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
    for batch_idx, (axial, coronal, sagittal, labels) in enumerate(pbar):
        # Move to device
        axial = axial.to(device)
        coronal = coronal.to(device)
        sagittal = sagittal.to(device)
        labels = labels.to(device)

        # Forward pass with mixed precision
        with autocast(enabled=use_amp):
            outputs = model(axial, coronal, sagittal)
            loss = criterion(outputs, labels)
            loss = loss / accumulation_steps

        # Backward pass
        scaler.scale(loss).backward()

        # Gradient accumulation
        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

        # Track metrics
        running_loss += loss.item() * accumulation_steps
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        # Update progress bar
        pbar.set_postfix({
            'loss': f"{running_loss / (batch_idx + 1):.4f}",
            'lr': f"{scheduler.get_last_lr()[0]:.2e}"
        })

    # Compute metrics
    avg_loss = running_loss / len(train_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)

    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data_loader,
    criterion: nn.Module,
    device: torch.device,
    config: dict,
    prefix: str = "Val"
) -> dict:
    """Evaluate model"""
    model.eval()

    use_amp = config['hardware'].get('mixed_precision', True) and device.type == 'cuda'

    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    pbar = tqdm(data_loader, desc=f"[{prefix}]")
    for axial, coronal, sagittal, labels in pbar:
        axial = axial.to(device)
        coronal = coronal.to(device)
        sagittal = sagittal.to(device)
        labels = labels.to(device)

        with autocast(enabled=use_amp):
            outputs = model(axial, coronal, sagittal)
            loss = criterion(outputs, labels)

        running_loss += loss.item()
        probs = torch.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    # Compute metrics
    avg_loss = running_loss / len(data_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds)

    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'predictions': np.array(all_preds),
        'labels': np.array(all_labels),
        'probabilities': np.array(all_probs)
    }


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    epoch: int,
    metrics: dict,
    config: dict,
    filepath: str
):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'metrics': metrics,
        'config': config
    }
    torch.save(checkpoint, filepath)
    logger.info(f"Saved checkpoint: {filepath}")


def train(config: dict):
    """Main training function"""
    # Setup
    set_seed(config['training']['seed'])
    device = get_device(config)

    # Create directories
    checkpoints_dir = Path(config['data']['checkpoints_dir'])
    logs_dir = Path(config['data']['logs_dir'])
    results_dir = Path(config['data']['results_dir'])

    for d in [checkpoints_dir, logs_dir, results_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Data loaders
    train_loader, val_loader, test_loader, class_weights = get_dataloaders(
        config['data']['train_csv'],
        config['data']['val_csv'],
        config['data']['test_csv'],
        config
    )

    # Model
    model = build_model(config, device)

    # Loss function
    if config['training'].get('use_weighted_loss', True):
        class_weights = class_weights.to(device)
        logger.info(f"Using weighted loss: {class_weights.tolist()}")
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    # Optimizer and scheduler
    optimizer = get_optimizer(model, config)
    steps_per_epoch = len(train_loader) // config['training'].get('accumulation_steps', 1)
    scheduler = get_scheduler(optimizer, config, steps_per_epoch)

    # Mixed precision scaler
    scaler = GradScaler(enabled=config['hardware'].get('mixed_precision', True) and device.type == 'cuda')

    # Training history
    history = {
        'train_loss': [], 'train_acc': [], 'train_bal_acc': [],
        'val_loss': [], 'val_acc': [], 'val_bal_acc': []
    }

    # Early stopping
    early_stop_config = config['callbacks']['early_stopping']
    best_metric = 0 if early_stop_config['mode'] == 'max' else float('inf')
    patience_counter = 0
    best_epoch = 0

    # Checkpoint config
    checkpoint_config = config['callbacks']['checkpoint']
    top_k_checkpoints = []

    logger.info("\n" + "=" * 60)
    logger.info("Starting Training")
    logger.info("=" * 60)

    for epoch in range(1, config['training']['epochs'] + 1):
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            scaler, device, config, epoch
        )

        # Validate
        val_metrics = evaluate(model, val_loader, criterion, device, config, "Val")

        # Log metrics
        logger.info(f"\nEpoch {epoch}/{config['training']['epochs']}")
        logger.info(f"  Train - Loss: {train_metrics['loss']:.4f}, "
                   f"Acc: {train_metrics['accuracy']:.4f}, "
                   f"BalAcc: {train_metrics['balanced_accuracy']:.4f}")
        logger.info(f"  Val   - Loss: {val_metrics['loss']:.4f}, "
                   f"Acc: {val_metrics['accuracy']:.4f}, "
                   f"BalAcc: {val_metrics['balanced_accuracy']:.4f}")

        # Update history
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['train_bal_acc'].append(train_metrics['balanced_accuracy'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_bal_acc'].append(val_metrics['balanced_accuracy'])

        # Monitor metric
        monitor_metric = val_metrics[early_stop_config['monitor'].replace('val_', '')]

        # Check for improvement
        improved = False
        if early_stop_config['mode'] == 'max':
            if monitor_metric > best_metric:
                improved = True
        else:
            if monitor_metric < best_metric:
                improved = True

        if improved:
            best_metric = monitor_metric
            best_epoch = epoch
            patience_counter = 0

            # Save best model
            best_path = checkpoints_dir / 'best_model.pt'
            save_checkpoint(model, optimizer, scheduler, epoch, val_metrics, config, str(best_path))
        else:
            patience_counter += 1

        # Save top-k checkpoints
        checkpoint_metric = val_metrics[checkpoint_config['monitor'].replace('val_', '')]
        checkpoint_path = checkpoints_dir / f'epoch_{epoch:03d}_bal_acc_{checkpoint_metric:.4f}.pt'

        top_k_checkpoints.append((checkpoint_metric, epoch, str(checkpoint_path)))
        top_k_checkpoints.sort(key=lambda x: x[0], reverse=(checkpoint_config['mode'] == 'max'))

        if len(top_k_checkpoints) <= checkpoint_config['save_top_k']:
            save_checkpoint(model, optimizer, scheduler, epoch, val_metrics, config, str(checkpoint_path))
        else:
            # Save new checkpoint and remove worst
            save_checkpoint(model, optimizer, scheduler, epoch, val_metrics, config, str(checkpoint_path))
            _, _, worst_path = top_k_checkpoints.pop()
            if Path(worst_path).exists():
                Path(worst_path).unlink()

        # Save last checkpoint
        if checkpoint_config.get('save_last', True):
            last_path = checkpoints_dir / 'last_model.pt'
            save_checkpoint(model, optimizer, scheduler, epoch, val_metrics, config, str(last_path))

        # Early stopping
        if early_stop_config['enabled'] and patience_counter >= early_stop_config['patience']:
            logger.info(f"\nEarly stopping at epoch {epoch}. Best epoch: {best_epoch}")
            break

    # Final evaluation on test set
    logger.info("\n" + "=" * 60)
    logger.info("Final Evaluation on Test Set")
    logger.info("=" * 60)

    # Load best model
    best_checkpoint = torch.load(checkpoints_dir / 'best_model.pt', map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint['model_state_dict'])

    test_metrics = evaluate(model, test_loader, criterion, device, config, "Test")

    logger.info(f"\nTest Results (Best model from epoch {best_epoch}):")
    logger.info(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"  Balanced Accuracy: {test_metrics['balanced_accuracy']:.4f}")
    logger.info(f"  Precision: {test_metrics['precision']:.4f}")
    logger.info(f"  Recall: {test_metrics['recall']:.4f}")
    logger.info(f"  F1 Score: {test_metrics['f1']:.4f}")

    logger.info(f"\nConfusion Matrix:")
    logger.info(f"  {CLASS_NAMES}")
    for i, row in enumerate(test_metrics['confusion_matrix']):
        logger.info(f"  {CLASS_NAMES[i]}: {row}")

    # Per-class metrics
    logger.info(f"\nClassification Report:")
    report = classification_report(
        test_metrics['labels'],
        test_metrics['predictions'],
        target_names=CLASS_NAMES,
        digits=4
    )
    logger.info(f"\n{report}")

    # Plot and save confusion matrix
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    cm_path = results_dir / f'confusion_matrix_{timestamp}.png'
    plot_confusion_matrix(test_metrics['confusion_matrix'], CLASS_NAMES, cm_path)

    # Save results
    results = {
        'config': config,
        'best_epoch': best_epoch,
        'history': history,
        'test_metrics': {
            'accuracy': float(test_metrics['accuracy']),
            'balanced_accuracy': float(test_metrics['balanced_accuracy']),
            'precision': float(test_metrics['precision']),
            'recall': float(test_metrics['recall']),
            'f1': float(test_metrics['f1']),
            'confusion_matrix': test_metrics['confusion_matrix'].tolist()
        }
    }

    results_path = results_dir / f'results_{timestamp}.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\nResults saved to: {results_path}")

    # Save predictions
    if config['evaluation'].get('save_predictions', True):
        predictions_path = results_dir / f'predictions_{timestamp}.npz'
        np.savez(
            predictions_path,
            predictions=test_metrics['predictions'],
            labels=test_metrics['labels'],
            probabilities=test_metrics['probabilities']
        )
        logger.info(f"Predictions saved to: {predictions_path}")

    return test_metrics


def main():
    parser = argparse.ArgumentParser(description='Train MedTransformer')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded config from: {args.config}")

    # Run training
    train(config)


if __name__ == '__main__':
    main()

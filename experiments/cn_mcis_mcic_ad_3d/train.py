#!/usr/bin/env python3
"""
Train 3D ResNet for 4-Class MRI Classification

Classes: CN | MCI_stable | MCI_to_AD | AD

Usage:
    python train.py --config config.yaml
"""

import argparse
import logging
from pathlib import Path
import yaml
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix,
    classification_report, roc_auc_score
)
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from dataset import get_dataloaders, CLASS_NAMES

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_resnet3d(num_classes=4, pretrained_path=None):
    """
    Get 3D ResNet model

    Uses torchvision-style 3D ResNet or MedicalNet weights
    """
    try:
        # Try to use MONAI's ResNet
        from monai.networks.nets import resnet18
        model = resnet18(
            pretrained=False,
            spatial_dims=3,
            n_input_channels=1,
            num_classes=num_classes
        )
        logger.info("Using MONAI ResNet18")
    except ImportError:
        # Fallback to simple 3D CNN
        logger.info("MONAI not available, using simple 3D CNN")
        model = Simple3DCNN(num_classes=num_classes)

    # Load pretrained weights if available
    if pretrained_path and Path(pretrained_path).exists():
        logger.info(f"Loading pretrained weights from {pretrained_path}")
        state_dict = torch.load(pretrained_path, map_location='cpu')
        # Remove FC layer weights (different num_classes)
        state_dict = {k: v for k, v in state_dict.items() if 'fc' not in k and 'classifier' not in k}
        model.load_state_dict(state_dict, strict=False)

    return model


class Simple3DCNN(nn.Module):
    """Simple 3D CNN for when MONAI is not available"""

    def __init__(self, num_classes=4):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: 192 -> 96
            nn.Conv3d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),

            # Block 2: 96 -> 48
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),

            # Block 3: 48 -> 24
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),

            # Block 4: 24 -> 12
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),

            # Block 5: 12 -> 6
            nn.Conv3d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def train_epoch(model, loader, criterion, optimizer, scaler, device, use_amp=True):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    pbar = tqdm(loader, desc='Training')
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with autocast('cuda', enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    bal_acc = balanced_accuracy_score(all_labels, all_preds)

    return avg_loss, acc, bal_acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []

    for images, labels in tqdm(loader, desc='Evaluating'):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item()
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        preds = outputs.argmax(dim=1).cpu().numpy()

        all_probs.extend(probs)
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    bal_acc = balanced_accuracy_score(all_labels, all_preds)

    return avg_loss, acc, bal_acc, np.array(all_preds), np.array(all_labels), np.array(all_probs)


def plot_results(y_true, y_pred, y_proba, output_dir):
    """Generate evaluation plots"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - 4-Class Classification')
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Normalized confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Normalized Confusion Matrix')
    plt.savefig(output_dir / 'confusion_matrix_normalized.png', dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Train 4-class 3D MRI classifier')
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Seed
    seed = config['training']['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Data
    train_loader, val_loader, test_loader, class_weights = get_dataloaders(
        train_csv=config['data']['train_csv'],
        val_csv=config['data']['val_csv'],
        test_csv=config['data']['test_csv'],
        batch_size=config['training']['batch_size'],
        target_shape=(config['model']['image_size'],) * 3,
        num_workers=config['hardware']['num_workers'],
        pin_memory=config['hardware']['pin_memory']
    )

    # Model
    model = get_resnet3d(
        num_classes=config['model']['num_classes'],
        pretrained_path=config['model'].get('pretrained_path')
    )
    model = model.to(device)

    # Loss with class weights
    if config['training']['use_weighted_loss']:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        logger.info(f"Using weighted loss: {class_weights.tolist()}")
    else:
        criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['epochs'],
        eta_min=config['training']['lr_min']
    )

    # Mixed precision
    scaler = GradScaler('cuda')
    use_amp = config['hardware']['mixed_precision'] and device.type == 'cuda'

    # Output dirs
    checkpoint_dir = Path(config['data']['checkpoints_dir'])
    results_dir = Path(config['data']['results_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    best_val_acc = 0
    patience = config['callbacks']['early_stopping']['patience']
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'val_bal_acc': []}

    logger.info("\n" + "=" * 80)
    logger.info("STARTING TRAINING")
    logger.info("=" * 80)

    for epoch in range(config['training']['epochs']):
        logger.info(f"\nEpoch {epoch + 1}/{config['training']['epochs']}")

        # Train
        train_loss, train_acc, train_bal_acc = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device, use_amp
        )

        # Validate
        val_loss, val_acc, val_bal_acc, _, _, _ = evaluate(model, val_loader, criterion, device)

        scheduler.step()

        # Log
        logger.info(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, Bal Acc: {train_bal_acc:.4f}")
        logger.info(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, Bal Acc: {val_bal_acc:.4f}")

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['val_bal_acc'].append(val_bal_acc)

        # Save best model
        if val_bal_acc > best_val_acc:
            best_val_acc = val_bal_acc
            torch.save(model.state_dict(), checkpoint_dir / 'best_model.pth')
            logger.info(f"Saved best model (bal_acc: {val_bal_acc:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch + 1}")
            break

    # Load best model
    model.load_state_dict(torch.load(checkpoint_dir / 'best_model.pth'))

    # Final evaluation on test set
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATING ON TEST SET")
    logger.info("=" * 80)

    test_loss, test_acc, test_bal_acc, y_pred, y_true, y_proba = evaluate(
        model, test_loader, criterion, device
    )

    logger.info(f"\nTest Results:")
    logger.info(f"  Accuracy: {test_acc:.4f}")
    logger.info(f"  Balanced Accuracy: {test_bal_acc:.4f}")

    # Classification report
    logger.info(f"\nClassification Report:")
    logger.info(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

    # Save results
    results = {
        'test_accuracy': float(test_acc),
        'test_balanced_accuracy': float(test_bal_acc),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
        'classification_report': classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True)
    }

    with open(results_dir / 'metrics.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Plots
    plot_results(y_true, y_pred, y_proba, results_dir)

    # Save training history
    with open(results_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    logger.info(f"\nResults saved to: {results_dir}")


if __name__ == '__main__':
    main()

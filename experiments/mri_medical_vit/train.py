#!/usr/bin/env python3
"""
Training script for 3D Vision Transformer

Usage:
    python train.py --config config.yaml
    python train.py --config config.yaml --test-only
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
import logging
from tqdm import tqdm
import argparse
import json
from datetime import datetime
import math

from model import get_vit_model, ViT3DClassifier, HybridViT, VIT_CONFIGS
from dataset import get_dataloaders

# Optional imports
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    from sklearn.metrics import classification_report, confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_cosine_schedule_with_warmup(optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
    """Cosine learning rate schedule with linear warmup"""

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Linear warmup
            return epoch / warmup_epochs
        else:
            # Cosine decay
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return min_lr + 0.5 * (1 - min_lr) * (1 + math.cos(math.pi * progress))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class Trainer:
    """Trainer for 3D Vision Transformer"""

    def __init__(self, config: Dict):
        self.config = config
        self.device = self._setup_device()
        self.set_seed(config['training']['seed'])

        # Create directories
        self.checkpoint_dir = Path(config['data']['checkpoints_dir'])
        self.results_dir = Path(config['data']['results_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.epochs_without_improvement = 0

        # Initialize wandb
        self.use_wandb = config['wandb']['enabled'] and WANDB_AVAILABLE
        if self.use_wandb:
            self._init_wandb()

    def _setup_device(self) -> torch.device:
        """Setup compute device"""
        device_str = self.config['hardware']['device']

        if device_str == 'cuda' and torch.cuda.is_available():
            device = torch.device('cuda')
            logger.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
        elif device_str == 'mps' and torch.backends.mps.is_available():
            device = torch.device('mps')
            logger.info("Using Apple Metal (MPS)")
        else:
            device = torch.device('cpu')
            logger.info("Using CPU")

        return device

    def set_seed(self, seed: int):
        """Set random seeds"""
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _init_wandb(self):
        """Initialize Weights & Biases"""
        wandb.init(
            project=self.config['wandb']['project'],
            name=self.config['wandb']['run_name'],
            config=self.config,
        )

    def build_model(self) -> nn.Module:
        """Build the ViT model"""
        logger.info("="*60)
        logger.info("BUILDING VISION TRANSFORMER")
        logger.info("="*60)

        cfg = self.config['model']
        model_type = cfg.get('model_type', 'vit')

        # Check for pretrained weights
        pretrained_path = None
        if cfg.get('use_pretrained', False) and cfg.get('pretrained_path'):
            pretrained_path = Path(cfg['pretrained_path'])
            if not pretrained_path.exists():
                logger.warning(f"Pretrained weights not found at {pretrained_path}")
                pretrained_path = None

        if model_type == 'hybrid':
            model = HybridViT(
                num_classes=cfg['num_classes'],
                in_channels=cfg['in_channels'],
                image_size=cfg['image_size'],
                vit_config=cfg['architecture'],
                dropout=cfg['dropout'],
            )
        elif model_type == 'vit_classifier':
            model = ViT3DClassifier(
                architecture=cfg['architecture'],
                num_classes=cfg['num_classes'],
                in_channels=cfg['in_channels'],
                image_size=cfg['image_size'],
                patch_size=cfg['patch_size'],
                dropout=cfg['dropout'],
                pretrained_path=str(pretrained_path) if pretrained_path else None,
                pool=cfg.get('pool', 'cls'),
            )
        else:
            model = get_vit_model(
                architecture=cfg['architecture'],
                num_classes=cfg['num_classes'],
                in_channels=cfg['in_channels'],
                image_size=cfg['image_size'],
                patch_size=cfg['patch_size'],
                dropout=cfg['dropout'],
                pretrained_path=str(pretrained_path) if pretrained_path else None,
            )

        model = model.to(self.device)

        # Log model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        logger.info(f"Model type: {model_type}")
        logger.info(f"Architecture: {cfg['architecture']}")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Device: {self.device}")

        return model

    def build_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Build dataloaders"""
        logger.info("\n" + "="*60)
        logger.info("BUILDING DATALOADERS")
        logger.info("="*60)

        cfg = self.config
        image_size = cfg['model']['image_size']

        train_loader, val_loader, test_loader = get_dataloaders(
            train_csv=cfg['data']['train_csv'],
            val_csv=cfg['data']['val_csv'],
            test_csv=cfg['data']['test_csv'],
            batch_size=cfg['training']['batch_size'],
            target_shape=(image_size, image_size, image_size),
            num_workers=cfg['hardware']['num_workers'],
            pin_memory=cfg['hardware']['pin_memory'],
            augment=cfg['augmentation']['enabled'],
        )

        return train_loader, val_loader, test_loader

    def build_optimizer(self, model: nn.Module) -> Tuple[optim.Optimizer, optim.lr_scheduler._LRScheduler]:
        """Build optimizer and scheduler"""
        cfg = self.config['training']

        # Separate parameters for different learning rates (optional)
        # ViT benefits from layer-wise learning rate decay
        optimizer = optim.AdamW(
            model.parameters(),
            lr=cfg['learning_rate'],
            weight_decay=cfg['weight_decay'],
            betas=(0.9, 0.999)
        )

        # Scheduler with warmup
        warmup_epochs = cfg.get('warmup_epochs', 5)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            warmup_epochs=warmup_epochs,
            total_epochs=cfg['epochs'],
            min_lr=cfg['lr_min'] / cfg['learning_rate']
        )

        logger.info(f"Optimizer: AdamW (lr={cfg['learning_rate']}, wd={cfg['weight_decay']})")
        logger.info(f"Scheduler: CosineWithWarmup (warmup={warmup_epochs} epochs)")

        return optimizer, scheduler

    def build_criterion(self, train_loader: DataLoader) -> nn.Module:
        """Build loss function"""
        if self.config['training']['use_weighted_loss']:
            labels = [label for _, label in train_loader.dataset]
            class_counts = np.bincount(labels)
            class_weights = 1.0 / class_counts
            class_weights = class_weights / class_weights.sum() * len(class_counts)
            class_weights = torch.FloatTensor(class_weights).to(self.device)

            criterion = nn.CrossEntropyLoss(
                weight=class_weights,
                label_smoothing=self.config['training'].get('label_smoothing', 0.0)
            )
            logger.info(f"Loss: Weighted CrossEntropyLoss")
        else:
            criterion = nn.CrossEntropyLoss(
                label_smoothing=self.config['training'].get('label_smoothing', 0.0)
            )
            logger.info("Loss: CrossEntropyLoss")

        return criterion

    def train_epoch(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        optimizer: optim.Optimizer,
        criterion: nn.Module
    ) -> Dict[str, float]:
        """Train for one epoch"""
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        grad_clip = self.config['training'].get('gradient_clip', 0)

        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch+1} [Train]")
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            optimizer.zero_grad()

            # Forward pass - handle different model outputs
            outputs = model(images)
            if isinstance(outputs, tuple):
                outputs = outputs[0]  # MONAI ViT returns (logits, hidden_states)

            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()

            # Gradient clipping
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })

        return {
            'loss': total_loss / len(train_loader),
            'accuracy': 100. * correct / total
        }

    @torch.no_grad()
    def validate(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        criterion: nn.Module,
        split_name: str = 'val'
    ) -> Dict[str, float]:
        """Validate model"""
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        all_preds = []
        all_labels = []

        pbar = tqdm(dataloader, desc=f"Epoch {self.current_epoch+1} [{split_name.upper()}]")
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = model(images)
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })

        # Balanced accuracy
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        class_accs = []
        for cls in np.unique(all_labels):
            mask = all_labels == cls
            class_acc = (all_preds[mask] == all_labels[mask]).mean()
            class_accs.append(class_acc)

        return {
            'loss': total_loss / len(dataloader),
            'accuracy': 100. * correct / total,
            'balanced_accuracy': 100. * np.mean(class_accs),
            'predictions': all_preds,
            'labels': all_labels
        }

    def train(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler._LRScheduler,
        criterion: nn.Module
    ):
        """Main training loop"""
        logger.info("\n" + "="*60)
        logger.info("STARTING TRAINING")
        logger.info("="*60)

        num_epochs = self.config['training']['epochs']
        patience = self.config['callbacks']['early_stopping']['patience']

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            # Train
            train_metrics = self.train_epoch(model, train_loader, optimizer, criterion)

            # Validate
            val_metrics = self.validate(model, val_loader, criterion, 'val')

            # Update scheduler
            scheduler.step()

            # Log
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(f"\nEpoch {epoch+1}/{num_epochs}")
            logger.info(f"  Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.2f}%")
            logger.info(f"  Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.2f}%, "
                       f"Balanced: {val_metrics['balanced_accuracy']:.2f}%")
            logger.info(f"  LR: {current_lr:.6f}")

            if self.use_wandb:
                wandb.log({
                    'epoch': epoch + 1,
                    'train/loss': train_metrics['loss'],
                    'train/accuracy': train_metrics['accuracy'],
                    'val/loss': val_metrics['loss'],
                    'val/accuracy': val_metrics['accuracy'],
                    'val/balanced_accuracy': val_metrics['balanced_accuracy'],
                    'lr': current_lr
                })

            # Save best
            if val_metrics['accuracy'] > self.best_val_acc:
                self.best_val_acc = val_metrics['accuracy']
                self.epochs_without_improvement = 0
                self._save_checkpoint(model, optimizer, 'best.pth')
                logger.info(f"  -> New best model! (val_acc: {self.best_val_acc:.2f}%)")
            else:
                self.epochs_without_improvement += 1

            # Early stopping
            if self.config['callbacks']['early_stopping']['enabled']:
                if self.epochs_without_improvement >= patience:
                    logger.info(f"\nEarly stopping after {epoch+1} epochs")
                    break

        self._save_checkpoint(model, optimizer, 'last.pth')

        logger.info("\n" + "="*60)
        logger.info("TRAINING COMPLETE")
        logger.info(f"Best validation accuracy: {self.best_val_acc:.2f}%")
        logger.info("="*60)

    def test(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        criterion: nn.Module
    ) -> Dict[str, float]:
        """Evaluate on test set"""
        logger.info("\n" + "="*60)
        logger.info("EVALUATING ON TEST SET")
        logger.info("="*60)

        # Load best model
        best_path = self.checkpoint_dir / 'best.pth'
        if best_path.exists():
            checkpoint = torch.load(best_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded best model from {best_path}")

        test_metrics = self.validate(model, test_loader, criterion, 'test')

        logger.info(f"\nTest Results:")
        logger.info(f"  Loss: {test_metrics['loss']:.4f}")
        logger.info(f"  Accuracy: {test_metrics['accuracy']:.2f}%")
        logger.info(f"  Balanced Accuracy: {test_metrics['balanced_accuracy']:.2f}%")

        self._save_results(test_metrics)

        if self.use_wandb:
            wandb.log({
                'test/loss': test_metrics['loss'],
                'test/accuracy': test_metrics['accuracy'],
                'test/balanced_accuracy': test_metrics['balanced_accuracy']
            })

        return test_metrics

    def _save_checkpoint(self, model: nn.Module, optimizer: optim.Optimizer, filename: str):
        """Save checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'config': self.config
        }
        torch.save(checkpoint, self.checkpoint_dir / filename)

    def _save_results(self, metrics: Dict):
        """Save test results"""
        results = {
            'loss': float(metrics['loss']),
            'accuracy': float(metrics['accuracy']),
            'balanced_accuracy': float(metrics['balanced_accuracy']),
            'timestamp': datetime.now().isoformat()
        }

        with open(self.results_dir / 'test_metrics.json', 'w') as f:
            json.dump(results, f, indent=2)

        if SKLEARN_AVAILABLE:
            labels = metrics['labels']
            preds = metrics['predictions']
            class_names = ['CN', 'MCI', 'AD']

            report = classification_report(labels, preds, target_names=class_names)
            logger.info(f"\nClassification Report:\n{report}")

            with open(self.results_dir / 'classification_report.txt', 'w') as f:
                f.write(report)

            cm = confusion_matrix(labels, preds)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=class_names, yticklabels=class_names)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Confusion Matrix - ViT')
            plt.savefig(self.results_dir / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
            plt.close()


def load_config(config_path: str) -> Dict:
    """Load config from YAML"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='Train 3D Vision Transformer')
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--test-only', action='store_true')
    args = parser.parse_args()

    config = load_config(args.config)

    trainer = Trainer(config)

    model = trainer.build_model()
    train_loader, val_loader, test_loader = trainer.build_dataloaders()
    optimizer, scheduler = trainer.build_optimizer(model)
    criterion = trainer.build_criterion(train_loader)

    if args.test_only:
        trainer.test(model, test_loader, criterion)
    else:
        trainer.train(model, train_loader, val_loader, optimizer, scheduler, criterion)
        trainer.test(model, test_loader, criterion)

    if trainer.use_wandb:
        wandb.finish()


if __name__ == '__main__':
    main()

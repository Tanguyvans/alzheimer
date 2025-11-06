#!/usr/bin/env python3
"""
Training script for VECNN Model on ADNI data

Based on: "Vision transformer-equipped Convolutional Neural Networks for
automated Alzheimer's disease diagnosis using 3D MRI scans"
PMC11682981 (November 2024)
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
from typing import Dict
import logging
from tqdm import tqdm
import argparse
from datetime import datetime

# Import custom modules
from dataset import get_dataloaders
from model_vecnn import vecnn_small

# Optional: Weights & Biases
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Logging will be local only.")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Trainer:
    """Trainer for VECNN model"""

    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device(config['hardware']['device'] if torch.cuda.is_available() else 'cpu')

        # Set random seeds
        self.set_seed(config['training']['seed'])

        # Create output directories
        self.checkpoint_dir = Path(config['data']['checkpoints_dir'])
        self.logs_dir = Path(config['data']['logs_dir'])
        self.results_dir = Path(config['data']['results_dir'])

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Initialize model, optimizer, dataloaders
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.scaler = None  # For mixed precision

        # Training state
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0

        # Initialize wandb if enabled
        self.use_wandb = config['wandb']['enabled'] and WANDB_AVAILABLE
        if self.use_wandb:
            self.init_wandb()

    def set_seed(self, seed: int):
        """Set random seeds for reproducibility"""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        logger.info(f"Random seed set to {seed}")

    def init_wandb(self):
        """Initialize Weights & Biases"""
        wandb.init(
            project=self.config['wandb']['project'],
            entity=self.config['wandb']['entity'],
            name=self.config['wandb']['run_name'],
            config=self.config,
            tags=self.config['wandb']['tags'],
            notes=self.config['wandb']['notes']
        )
        logger.info("Weights & Biases initialized")

    def build_model(self):
        """Build VECNN model"""
        logger.info("="*80)
        logger.info("BUILDING VECNN MODEL")
        logger.info("="*80)

        self.model = vecnn_small(
            num_classes=self.config['model']['num_classes'],
            in_channels=self.config['model']['num_channels']
        )
        self.model = self.model.to(self.device)

        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        logger.info(f"Model: VECNN")
        logger.info(f"  Total parameters: {total_params:,}")
        logger.info(f"  Trainable parameters: {trainable_params:,}")
        logger.info(f"  Device: {self.device}")

        if self.use_wandb:
            wandb.watch(self.model, log='all', log_freq=100)

    def build_dataloaders(self):
        """Build train/val/test dataloaders"""
        logger.info("\n" + "="*80)
        logger.info("BUILDING DATALOADERS")
        logger.info("="*80)

        self.train_loader, self.val_loader, self.test_loader = get_dataloaders(
            train_csv=self.config['data']['train_csv'],
            val_csv=self.config['data']['val_csv'],
            test_csv=self.config['data']['test_csv'],
            batch_size=self.config['training']['batch_size'],
            target_shape=(self.config['model']['image_size'],) * 3,
            num_workers=self.config['hardware']['num_workers'],
            pin_memory=self.config['hardware']['pin_memory']
        )

    def build_optimizer(self):
        """Build optimizer and learning rate scheduler"""
        logger.info("\n" + "="*80)
        logger.info("BUILDING OPTIMIZER")
        logger.info("="*80)

        lr = self.config['training']['learning_rate']
        weight_decay = self.config['training']['weight_decay']

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        logger.info(f"Optimizer: AdamW")
        logger.info(f"  Learning rate: {lr}")
        logger.info(f"  Weight decay: {weight_decay}")

        # Cosine Annealing scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config['training']['epochs'],
            eta_min=self.config['training']['lr_min']
        )
        logger.info(f"LR Scheduler: CosineAnnealingLR (T_max={self.config['training']['epochs']}, eta_min={self.config['training']['lr_min']})")

        # Mixed precision scaler
        if self.config['hardware']['mixed_precision']:
            self.scaler = torch.cuda.amp.GradScaler()
            logger.info("Mixed precision training enabled (FP16)")

    def build_criterion(self):
        """Build loss criterion"""
        if self.config['training']['use_weighted_loss']:
            # Calculate class weights from training data
            labels = [label for _, label in self.train_loader.dataset]
            class_counts = np.bincount(labels)
            class_weights = 1.0 / class_counts
            class_weights = class_weights / class_weights.sum()
            class_weights = torch.FloatTensor(class_weights).to(self.device)

            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
            logger.info(f"Loss: Weighted CrossEntropyLoss")
            logger.info(f"  Class weights: {class_weights.cpu().numpy()}")
        else:
            self.criterion = nn.CrossEntropyLoss()
            logger.info(f"Loss: CrossEntropyLoss")

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1} [Train]")
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            # Mixed precision training
            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    logits = self.model(images)
                    loss = self.criterion(logits, labels)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(images)
                loss = self.criterion(logits, labels)
                loss.backward()
                self.optimizer.step()

            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100*correct/total:.2f}%'})

        # Calculate epoch metrics
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100 * correct / total

        return {'loss': avg_loss, 'accuracy': accuracy}

    @torch.no_grad()
    def validate(self, dataloader: DataLoader, split_name: str = 'val') -> Dict[str, float]:
        """Validate on val or test set"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        all_preds = []
        all_labels = []

        pbar = tqdm(dataloader, desc=f"Epoch {self.current_epoch+1} [{split_name.upper()}]")
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            logits = self.model(images)
            loss = self.criterion(logits, labels)

            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100*correct/total:.2f}%'})

        # Calculate metrics
        avg_loss = total_loss / len(dataloader)
        accuracy = 100 * correct / total

        # Calculate balanced accuracy (for imbalanced datasets)
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        # Per-class accuracy
        unique_labels = np.unique(all_labels)
        class_accuracies = []
        for label in unique_labels:
            mask = all_labels == label
            class_acc = (all_preds[mask] == all_labels[mask]).mean()
            class_accuracies.append(class_acc)

        balanced_acc = 100 * np.mean(class_accuracies)

        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'balanced_accuracy': balanced_acc
        }

    def train(self):
        """Main training loop"""
        logger.info("\n" + "="*80)
        logger.info("STARTING TRAINING")
        logger.info("="*80)

        num_epochs = self.config['training']['epochs']
        early_stopping_patience = self.config['callbacks']['early_stopping']['patience']

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            # Train
            train_metrics = self.train_epoch()

            # Validate
            val_metrics = self.validate(self.val_loader, 'val')

            # Log metrics
            logger.info(f"\nEpoch {epoch+1}/{num_epochs}")
            logger.info(f"  Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['accuracy']:.2f}%")
            logger.info(f"  Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.2f}% | Balanced Acc: {val_metrics['balanced_accuracy']:.2f}%")

            if self.use_wandb:
                wandb.log({
                    'epoch': epoch + 1,
                    'train/loss': train_metrics['loss'],
                    'train/accuracy': train_metrics['accuracy'],
                    'val/loss': val_metrics['loss'],
                    'val/accuracy': val_metrics['accuracy'],
                    'val/balanced_accuracy': val_metrics['balanced_accuracy'],
                    'lr': self.optimizer.param_groups[0]['lr']
                })

            # Learning rate scheduling
            self.scheduler.step()

            # Save best model
            if val_metrics['accuracy'] > self.best_val_acc:
                self.best_val_acc = val_metrics['accuracy']
                self.best_val_loss = val_metrics['loss']
                self.epochs_without_improvement = 0
                self.save_checkpoint('best_model.pth', is_best=True)
                logger.info(f"  ✓ New best model saved! (val_acc: {self.best_val_acc:.2f}%)")
            else:
                self.epochs_without_improvement += 1

            # Early stopping
            if self.config['callbacks']['early_stopping']['enabled']:
                if self.epochs_without_improvement >= early_stopping_patience:
                    logger.info(f"\nEarly stopping triggered after {epoch+1} epochs")
                    break

            # Save checkpoint periodically
            if epoch % 10 == 0 or epoch == num_epochs - 1:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pth')

        logger.info("\n" + "="*80)
        logger.info("TRAINING COMPLETE")
        logger.info("="*80)
        logger.info(f"Best validation accuracy: {self.best_val_acc:.2f}%")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")

    def save_checkpoint(self, filename: str, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }

        checkpoint_path = self.checkpoint_dir / filename
        torch.save(checkpoint, checkpoint_path)

        if is_best:
            # Also save as 'best.pth'
            torch.save(checkpoint, self.checkpoint_dir / 'best.pth')

    def test(self):
        """Evaluate on test set"""
        logger.info("\n" + "="*80)
        logger.info("EVALUATING ON TEST SET")
        logger.info("="*80)

        # Load best model
        best_model_path = self.checkpoint_dir / 'best.pth'
        if best_model_path.exists():
            checkpoint = torch.load(best_model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded best model from {best_model_path}")

        test_metrics = self.validate(self.test_loader, 'test')

        logger.info(f"\nTest Results:")
        logger.info(f"  Loss: {test_metrics['loss']:.4f}")
        logger.info(f"  Accuracy: {test_metrics['accuracy']:.2f}%")
        logger.info(f"  Balanced Accuracy: {test_metrics['balanced_accuracy']:.2f}%")

        if self.use_wandb:
            wandb.log({
                'test/loss': test_metrics['loss'],
                'test/accuracy': test_metrics['accuracy'],
                'test/balanced_accuracy': test_metrics['balanced_accuracy']
            })

        return test_metrics


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='Train VECNN model on ADNI data')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    parser.add_argument('--test-only', action='store_true',
                       help='Only run testing (requires trained model)')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Create trainer
    trainer = Trainer(config)

    # Build components
    trainer.build_model()
    trainer.build_dataloaders()
    trainer.build_optimizer()
    trainer.build_criterion()

    if args.test_only:
        # Test only
        trainer.test()
    else:
        # Train and test
        trainer.train()
        trainer.test()

    if trainer.use_wandb:
        wandb.finish()


if __name__ == '__main__':
    main()

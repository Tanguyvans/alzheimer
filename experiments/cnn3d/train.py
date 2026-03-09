#!/usr/bin/env python3
"""
Training script for 3D CNN Alzheimer's Classification

Usage:
    python train.py --config config.yaml
    python train.py --config config.yaml --test-only
"""

import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from typing import Dict
import logging
from tqdm import tqdm
import argparse
import json
from datetime import datetime

from model import CNN3DClassifier
from dataset import get_dataloaders

try:
    from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
    import matplotlib.pyplot as plt
    import seaborn as sns
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance."""

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, weight: torch.Tensor = None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class CNN3DTrainer:
    """Trainer for 3D CNN model"""

    def __init__(self, config: Dict):
        self.config = config
        self.device = self._setup_device()
        self.set_seed(config['training']['seed'])

        # Build experiment name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.timestamp = timestamp

        exp_cfg = config.get('experiment', {})
        arch = config['model']['architecture']
        task = exp_cfg.get('task', 'cn_ad')
        dataset = exp_cfg.get('dataset', 'unknown')

        self.run_name = f"{task}_{dataset}_{arch}_{timestamp}"

        self.checkpoint_dir = Path(config['data']['checkpoints_dir']) / self.run_name
        self.results_dir = Path(config['data']['results_dir']) / self.run_name

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Experiment: {self.run_name}")
        logger.info(f"Results: {self.results_dir}")

        # Training state
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.epochs_without_improvement = 0

        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [], 'val_balanced_acc': [],
            'lr': []
        }

    def _setup_device(self) -> torch.device:
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
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def build_dataloaders(self):
        logger.info("=" * 60)
        logger.info("BUILDING DATALOADERS")
        logger.info("=" * 60)

        cfg = self.config
        image_size = cfg['model']['image_size']
        preproc = cfg.get('preprocessing', {})

        train_loader, val_loader, test_loader = get_dataloaders(
            train_csv=cfg['data']['train_csv'],
            val_csv=cfg['data']['val_csv'],
            test_csv=cfg['data']['test_csv'],
            batch_size=cfg['training']['batch_size'],
            target_shape=(image_size, image_size, image_size),
            num_workers=cfg['hardware']['num_workers'],
            pin_memory=cfg['hardware']['pin_memory'],
            augment=True,
            use_paper_preprocessing=preproc.get('use_paper_preprocessing', True),
            target_spacing=preproc.get('target_spacing', 1.75),
        )

        return train_loader, val_loader, test_loader

    def build_model(self) -> nn.Module:
        logger.info("=" * 60)
        logger.info("BUILDING MODEL")
        logger.info("=" * 60)

        cfg = self.config['model']
        model = CNN3DClassifier(
            architecture=cfg['architecture'],
            num_classes=cfg['num_classes'],
            in_channels=cfg['in_channels'],
            dropout=cfg.get('dropout', 0.1),
            classifier_dropout=cfg.get('classifier_dropout', 0.5),
        )
        model = model.to(self.device)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")

        return model

    def build_optimizer(self, model: nn.Module):
        cfg = self.config['training']

        optimizer = optim.AdamW(
            model.parameters(),
            lr=cfg['learning_rate'],
            weight_decay=cfg['weight_decay']
        )
        logger.info(f"Optimizer: AdamW (lr={cfg['learning_rate']}, wd={cfg['weight_decay']})")

        main_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg['epochs'] - cfg.get('warmup_epochs', 0),
            eta_min=cfg['lr_min']
        )

        warmup_epochs = cfg.get('warmup_epochs', 0)
        if warmup_epochs > 0:
            warmup_scheduler = optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs
            )
            scheduler = optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, main_scheduler],
                milestones=[warmup_epochs]
            )
            logger.info(f"Scheduler: CosineAnnealingLR with {warmup_epochs} warmup epochs")
        else:
            scheduler = main_scheduler
            logger.info("Scheduler: CosineAnnealingLR")

        return optimizer, scheduler

    def build_criterion(self, train_loader: DataLoader) -> nn.Module:
        cfg = self.config['training']
        loss_type = cfg.get('loss_type', 'cross_entropy')
        use_weighted = cfg.get('use_weighted_loss', True)
        label_smoothing = cfg.get('label_smoothing', 0.0)

        class_weights = None
        if use_weighted:
            labels = [label for _, label in train_loader.dataset]
            class_counts = np.bincount(labels)
            class_weights = 1.0 / class_counts
            class_weights = class_weights / class_weights.sum() * len(class_counts)
            class_weights = torch.FloatTensor(class_weights).to(self.device)

        if loss_type == 'focal':
            criterion = FocalLoss(
                alpha=cfg.get('focal_alpha', 0.75),
                gamma=cfg.get('focal_gamma', 2.0),
                weight=class_weights
            )
            logger.info(f"Loss: FocalLoss (weighted={use_weighted})")
        else:
            criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
            logger.info(f"Loss: CrossEntropyLoss (weighted={use_weighted}, label_smoothing={label_smoothing})")

        return criterion

    def train_epoch(self, model, train_loader, optimizer, criterion):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        grad_clip = self.config['training'].get('gradient_clip', 1.0)

        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch+1} [Train]")
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
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
    def validate(self, model, dataloader, criterion, split_name='val'):
        model.eval()
        total_loss, correct, total = 0.0, 0, 0
        all_preds, all_labels = [], []

        pbar = tqdm(dataloader, desc=f"Epoch {self.current_epoch+1} [{split_name.upper()}]")
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = model(images)
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

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        balanced_acc = balanced_accuracy_score(all_labels, all_preds) * 100 if SKLEARN_AVAILABLE else 0

        return {
            'loss': total_loss / len(dataloader),
            'accuracy': 100. * correct / total,
            'balanced_accuracy': balanced_acc,
            'predictions': all_preds,
            'labels': all_labels
        }

    def train(self, model, train_loader, val_loader, optimizer, scheduler, criterion):
        logger.info("=" * 60)
        logger.info("STARTING TRAINING")
        logger.info("=" * 60)

        num_epochs = self.config['training']['epochs']
        patience = self.config['callbacks']['early_stopping']['patience']
        min_epochs = self.config['callbacks']['early_stopping'].get('min_epochs', 0)

        # Init W&B
        use_wandb = self.config.get('wandb', {}).get('enabled', False) and WANDB_AVAILABLE
        if use_wandb:
            wandb_cfg = self.config['wandb']
            wandb.init(
                project=wandb_cfg.get('project', 'alzheimer-cnn3d'),
                name=wandb_cfg.get('name', self.run_name),
                config=self.config
            )

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            train_metrics = self.train_epoch(model, train_loader, optimizer, criterion)
            val_metrics = self.validate(model, val_loader, criterion, 'val')
            scheduler.step()

            current_lr = optimizer.param_groups[0]['lr']

            # Record history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['val_balanced_acc'].append(val_metrics['balanced_accuracy'])
            self.history['lr'].append(current_lr)

            logger.info(f"\nEpoch {epoch+1}/{num_epochs}")
            logger.info(f"  Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.2f}%")
            logger.info(f"  Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.2f}%, "
                        f"Balanced: {val_metrics['balanced_accuracy']:.2f}%")
            logger.info(f"  LR: {current_lr:.6f}")

            if use_wandb:
                wandb.log({
                    'train_loss': train_metrics['loss'],
                    'train_accuracy': train_metrics['accuracy'],
                    'val_loss': val_metrics['loss'],
                    'val_accuracy': val_metrics['accuracy'],
                    'val_balanced_accuracy': val_metrics['balanced_accuracy'],
                    'lr': current_lr,
                    'epoch': epoch + 1
                })

            # Save best model
            if val_metrics['accuracy'] > self.best_val_acc:
                self.best_val_acc = val_metrics['accuracy']
                self.epochs_without_improvement = 0
                self._save_checkpoint(model, optimizer, 'best.pth')
                logger.info(f"  -> New best model! (val_acc: {self.best_val_acc:.2f}%)")
            else:
                self.epochs_without_improvement += 1

            # Early stopping
            if self.config['callbacks']['early_stopping']['enabled']:
                if epoch >= min_epochs and self.epochs_without_improvement >= patience:
                    logger.info(f"\nEarly stopping triggered after {epoch+1} epochs")
                    break

        self._save_checkpoint(model, optimizer, 'last.pth')
        self._plot_training_curves()

        if use_wandb:
            wandb.finish()

        logger.info("=" * 60)
        logger.info("TRAINING COMPLETE")
        logger.info(f"Best validation accuracy: {self.best_val_acc:.2f}%")
        logger.info("=" * 60)

    @torch.no_grad()
    def test_with_tta(self, model, dataloader, criterion, num_augmentations=8):
        """Test-Time Augmentation with spatial flips"""
        model.eval()
        total_loss, correct, total = 0.0, 0, 0
        all_preds, all_labels = [], []

        def apply_tta_augmentation(mri, aug_idx):
            if aug_idx == 0:
                return mri
            elif aug_idx == 1:
                return torch.flip(mri, dims=[2])
            elif aug_idx == 2:
                return torch.flip(mri, dims=[3])
            elif aug_idx == 3:
                return torch.flip(mri, dims=[4])
            elif aug_idx == 4:
                return torch.flip(mri, dims=[2, 3])
            elif aug_idx == 5:
                return torch.flip(mri, dims=[2, 4])
            elif aug_idx == 6:
                return torch.flip(mri, dims=[3, 4])
            elif aug_idx == 7:
                return torch.flip(mri, dims=[2, 3, 4])
            return mri

        pbar = tqdm(dataloader, desc=f"Test with TTA ({num_augmentations} augs)")
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            all_logits = []
            for aug_idx in range(min(num_augmentations, 8)):
                aug_images = apply_tta_augmentation(images, aug_idx)
                outputs = model(aug_images)
                all_logits.append(outputs)

            avg_logits = torch.stack(all_logits).mean(dim=0)
            loss = criterion(avg_logits, labels)

            total_loss += loss.item()
            _, predicted = avg_logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        balanced_acc = balanced_accuracy_score(all_labels, all_preds) * 100 if SKLEARN_AVAILABLE else 0

        return {
            'loss': total_loss / len(dataloader),
            'accuracy': 100. * correct / total,
            'balanced_accuracy': balanced_acc,
            'predictions': all_preds,
            'labels': all_labels
        }

    def test(self, model, test_loader, criterion):
        logger.info("=" * 60)
        logger.info("EVALUATING ON TEST SET")
        logger.info("=" * 60)

        best_path = self.checkpoint_dir / 'best.pth'
        if best_path.exists():
            checkpoint = torch.load(best_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded best model from {best_path}")

        use_tta = self.config['training'].get('use_tta', False)
        tta_augmentations = self.config['training'].get('tta_augmentations', 8)

        if use_tta:
            logger.info(f"Using TTA with {tta_augmentations} augmentations")
            test_metrics = self.test_with_tta(model, test_loader, criterion, tta_augmentations)
        else:
            test_metrics = self.validate(model, test_loader, criterion, 'test')

        logger.info(f"\nTest Results:")
        logger.info(f"  Accuracy: {test_metrics['accuracy']:.2f}%")
        logger.info(f"  Balanced Accuracy: {test_metrics['balanced_accuracy']:.2f}%")

        self._save_results(test_metrics)
        return test_metrics

    def _save_checkpoint(self, model, optimizer, filename):
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'config': self.config
        }
        torch.save(checkpoint, self.checkpoint_dir / filename)

    def _plot_training_curves(self):
        if not SKLEARN_AVAILABLE or len(self.history['train_loss']) == 0:
            return

        epochs = range(1, len(self.history['train_loss']) + 1)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Loss
        axes[0, 0].plot(epochs, self.history['train_loss'], 'b-', label='Train', linewidth=2)
        axes[0, 0].plot(epochs, self.history['val_loss'], 'r-', label='Val', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Accuracy
        axes[0, 1].plot(epochs, self.history['train_acc'], 'b-', label='Train', linewidth=2)
        axes[0, 1].plot(epochs, self.history['val_acc'], 'r-', label='Val', linewidth=2)
        axes[0, 1].plot(epochs, self.history['val_balanced_acc'], 'g--', label='Val Balanced', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].set_title('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Learning rate
        axes[1, 0].plot(epochs, self.history['lr'], 'purple', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)

        # Val loss vs accuracy
        ax4 = axes[1, 1]
        ax4_acc = ax4.twinx()
        line1, = ax4.plot(epochs, self.history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        line2, = ax4_acc.plot(epochs, self.history['val_acc'], 'b-', label='Val Acc', linewidth=2)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Loss', color='r')
        ax4_acc.set_ylabel('Accuracy (%)', color='b')
        ax4.set_title('Val Loss vs Accuracy')
        ax4.legend([line1, line2], ['Val Loss', 'Val Acc'], loc='center right')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.results_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
        plt.close()

        with open(self.results_dir / 'training_history.json', 'w') as f:
            json.dump(self.history, f, indent=2)

        logger.info(f"Training curves saved to {self.results_dir / 'training_curves.png'}")

    def save_config(self):
        config_path = self.results_dir / 'config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        logger.info(f"Config saved to: {config_path}")

    def _save_results(self, metrics: Dict):
        results = {
            'run_name': self.run_name,
            'accuracy': float(metrics['accuracy']),
            'balanced_accuracy': float(metrics['balanced_accuracy']),
            'timestamp': datetime.now().isoformat(),
            'architecture': self.config['model']['architecture'],
            'epochs_trained': len(self.history['train_loss']),
            'best_val_acc': float(self.best_val_acc)
        }

        with open(self.results_dir / 'test_metrics.json', 'w') as f:
            json.dump(results, f, indent=2)

        if SKLEARN_AVAILABLE:
            labels = metrics['labels']
            preds = metrics['predictions']
            class_names = ['CN', 'AD']

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
            plt.title('Confusion Matrix - CNN 3D')
            plt.savefig(self.results_dir / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
            plt.close()


def load_config(config_path: str) -> Dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='Train 3D CNN for AD Classification')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file')
    parser.add_argument('--test-only', action='store_true', help='Only run test')
    args = parser.parse_args()

    config = load_config(args.config)

    trainer = CNN3DTrainer(config)
    train_loader, val_loader, test_loader = trainer.build_dataloaders()
    model = trainer.build_model()
    optimizer, scheduler = trainer.build_optimizer(model)
    criterion = trainer.build_criterion(train_loader)

    trainer.save_config()

    if args.test_only:
        trainer.test(model, test_loader, criterion)
    else:
        trainer.train(model, train_loader, val_loader, optimizer, scheduler, criterion)
        trainer.test(model, test_loader, criterion)


if __name__ == '__main__':
    main()

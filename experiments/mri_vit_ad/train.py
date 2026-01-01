#!/usr/bin/env python3
"""
Training script for 3D Vision Transformer (ViT) for Alzheimer's Classification

Based on: "Training ViT with Limited Data for Alzheimer's Disease Classification" (MICCAI 2024)
GitHub: https://github.com/qasymjomart/ViT_recipe_for_AD

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

from model import ViT3DClassifier
from dataset import get_dataloaders, ADNIDataset

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


class FocalLoss(nn.Module):
    """
    Focal Loss for imbalanced classification.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha: Class weights (tensor of shape [num_classes])
        gamma: Focusing parameter (default 2.0, higher = more focus on hard examples)
        reduction: 'mean', 'sum', or 'none'
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)  # p_t = probability of correct class
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


def get_layer_wise_lr_decay_params(model, base_lr: float, weight_decay: float, lr_decay: float = 0.75):
    """
    Create parameter groups with layer-wise learning rate decay.

    As per the MICCAI 2024 paper:
    - Head gets highest LR (base_lr)
    - Deeper transformer blocks get progressively lower LR
    - Patch embed and pos_embed get lowest LR

    Args:
        model: ViT3DClassifier model
        base_lr: Base learning rate for the head
        weight_decay: Weight decay for regularization
        lr_decay: Decay factor per layer (default 0.75 from paper)

    Returns:
        List of parameter group dicts for optimizer
    """
    param_groups = []

    # Count transformer blocks (should be 12)
    num_blocks = len(model.blocks)

    # Layer ordering from top (highest LR) to bottom (lowest LR):
    # head -> norm -> blocks[11] -> blocks[10] -> ... -> blocks[0] -> embeddings

    # Layer 0: Classification head (highest LR)
    head_params = list(model.head.parameters())
    if head_params:
        param_groups.append({
            'params': head_params,
            'lr': base_lr,
            'weight_decay': weight_decay,
            'layer_name': 'head'
        })

    # Layer 1: Final norm
    norm_params = list(model.norm.parameters())
    if norm_params:
        param_groups.append({
            'params': norm_params,
            'lr': base_lr * lr_decay,
            'weight_decay': weight_decay,
            'layer_name': 'norm'
        })

    # Layers 2 to num_blocks+1: Transformer blocks (reverse order - last block first)
    for i in range(num_blocks - 1, -1, -1):
        block_params = list(model.blocks[i].parameters())
        layer_idx = num_blocks - i + 1  # 2 for block[11], 13 for block[0]
        lr = base_lr * (lr_decay ** layer_idx)

        if block_params:
            param_groups.append({
                'params': block_params,
                'lr': lr,
                'weight_decay': weight_decay,
                'layer_name': f'blocks.{i}'
            })

    # Last layers: Embeddings (lowest LR)
    embed_layer_idx = num_blocks + 2
    embed_lr = base_lr * (lr_decay ** embed_layer_idx)

    embed_params = []
    embed_params.extend([model.cls_token, model.pos_embed])
    embed_params.extend(list(model.patch_embed.parameters()))
    embed_params.append(model.pos_drop.p) if hasattr(model.pos_drop, 'p') else None

    # Filter out non-parameters and None
    embed_params = [p for p in embed_params if isinstance(p, torch.nn.Parameter)]

    if embed_params:
        param_groups.append({
            'params': embed_params,
            'lr': embed_lr,
            'weight_decay': weight_decay,
            'layer_name': 'embeddings'
        })

    # Log LR distribution
    logger.info("Layer-wise learning rate decay:")
    for pg in param_groups:
        logger.info(f"  {pg['layer_name']}: lr={pg['lr']:.2e}")

    return param_groups


class Trainer:
    """Trainer for 3D ViT"""

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
        """Set random seeds for reproducibility"""
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
        """Build the model"""
        logger.info("="*60)
        logger.info("BUILDING MODEL")
        logger.info("="*60)

        cfg = self.config['model']

        # Check for pretrained weights
        pretrained_path = None
        if cfg['use_pretrained']:
            pretrained_path = Path(cfg['pretrained_path'])
            if not pretrained_path.exists():
                logger.warning(f"Pretrained weights not found at {pretrained_path}")
                logger.warning("Training from scratch. Download from:")
                logger.warning("  https://github.com/qasymjomart/ViT_recipe_for_AD")
                pretrained_path = None

        image_size = cfg['image_size']

        model = ViT3DClassifier(
            architecture=cfg['architecture'],
            num_classes=cfg['num_classes'],
            in_channels=cfg['in_channels'],
            image_size=image_size,  # Pass as int, not tuple
            pretrained_path=str(pretrained_path) if pretrained_path else None,
            dropout=cfg['dropout'],
            classifier_dropout=cfg['classifier_dropout'],
            drop_path_rate=cfg.get('drop_path_rate', 0.1),
        )

        model = model.to(self.device)

        # Log model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        logger.info(f"Architecture: {cfg['architecture']}")
        logger.info(f"Image size: {image_size}x{image_size}x{image_size}")
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

        # Get preprocessing settings (default to paper settings if not specified)
        preproc = cfg.get('preprocessing', {})
        use_paper_preprocessing = preproc.get('use_paper_preprocessing', True)
        target_spacing = preproc.get('target_spacing', 1.75)

        train_loader, val_loader, test_loader = get_dataloaders(
            train_csv=cfg['data']['train_csv'],
            val_csv=cfg['data']['val_csv'],
            test_csv=cfg['data']['test_csv'],
            batch_size=cfg['training']['batch_size'],
            target_shape=(image_size, image_size, image_size),
            num_workers=cfg['hardware']['num_workers'],
            pin_memory=cfg['hardware']['pin_memory'],
            augment=cfg['augmentation']['enabled'],
            use_paper_preprocessing=use_paper_preprocessing,
            target_spacing=target_spacing,
        )

        return train_loader, val_loader, test_loader

    def build_optimizer(self, model: nn.Module) -> Tuple[optim.Optimizer, optim.lr_scheduler._LRScheduler]:
        """Build optimizer and scheduler with warmup and optional layer-wise LR decay"""
        cfg = self.config['training']

        # Check if layer-wise LR decay is enabled
        use_layer_lr_decay = cfg.get('layer_wise_lr_decay', 0.0)

        if use_layer_lr_decay > 0:
            # Use layer-wise learning rate decay (paper uses 0.75)
            param_groups = get_layer_wise_lr_decay_params(
                model,
                base_lr=cfg['learning_rate'],
                weight_decay=cfg['weight_decay'],
                lr_decay=use_layer_lr_decay
            )
            optimizer = optim.AdamW(param_groups)
            logger.info(f"Optimizer: AdamW with layer-wise LR decay (factor={use_layer_lr_decay})")
        else:
            # Standard optimizer - all parameters with same LR
            optimizer = optim.AdamW(
                model.parameters(),
                lr=cfg['learning_rate'],
                weight_decay=cfg['weight_decay']
            )
            logger.info(f"Optimizer: AdamW (lr={cfg['learning_rate']}, weight_decay={cfg['weight_decay']})")

        # Main scheduler (cosine annealing)
        main_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg['epochs'] - cfg.get('warmup_epochs', 0),
            eta_min=cfg['lr_min']
        )

        # Warmup scheduler
        warmup_epochs = cfg.get('warmup_epochs', 0)
        if warmup_epochs > 0:
            warmup_scheduler = optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=warmup_epochs
            )
            scheduler = optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, main_scheduler],
                milestones=[warmup_epochs]
            )
            logger.info(f"Scheduler: CosineAnnealingLR with {warmup_epochs} warmup epochs")
        else:
            scheduler = main_scheduler
            logger.info(f"Scheduler: CosineAnnealingLR")

        return optimizer, scheduler

    def build_criterion(self, train_loader: DataLoader) -> nn.Module:
        """Build loss function with optional focal loss for imbalanced data"""
        training_cfg = self.config['training']
        use_focal = training_cfg.get('use_focal_loss', False)
        focal_gamma = training_cfg.get('focal_gamma', 2.0)

        # Calculate class weights if weighted loss is enabled
        class_weights = None
        if training_cfg['use_weighted_loss']:
            labels = [label for _, label in train_loader.dataset]
            labels_np = np.array(labels)
            class_counts = np.bincount(labels_np)
            n_samples = len(labels_np)
            n_classes = len(class_counts)

            # Balanced class weights (sklearn formula)
            class_weights = n_samples / (n_classes * class_counts)
            class_weights = torch.FloatTensor(class_weights).to(self.device)

            logger.info(f"Class distribution: {dict(enumerate(class_counts))}")
            logger.info(f"Class weights (balanced): {class_weights.cpu().numpy()}")

        # Choose loss function
        if use_focal:
            criterion = FocalLoss(alpha=class_weights, gamma=focal_gamma)
            logger.info(f"Loss: FocalLoss (gamma={focal_gamma}, weighted={class_weights is not None})")
        elif class_weights is not None:
            criterion = nn.CrossEntropyLoss(
                weight=class_weights,
                label_smoothing=training_cfg.get('label_smoothing', 0.0)
            )
            logger.info("Loss: Weighted CrossEntropyLoss")
        else:
            criterion = nn.CrossEntropyLoss(
                label_smoothing=training_cfg.get('label_smoothing', 0.0)
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
        """Train for one epoch with gradient accumulation"""
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        # Gradient accumulation settings
        accumulation_steps = self.config['training'].get('gradient_accumulation_steps', 1)
        grad_clip = self.config['training'].get('gradient_clip', 1.0)

        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch+1} [Train]")
        for step, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            # Scale loss for gradient accumulation
            loss = loss / accumulation_steps
            loss.backward()

            # Track metrics (use unscaled loss for logging)
            total_loss += loss.item() * accumulation_steps
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Update weights every accumulation_steps
            if (step + 1) % accumulation_steps == 0 or (step + 1) == len(train_loader):
                # Gradient clipping (important for transformers)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                optimizer.step()
                optimizer.zero_grad()

            pbar.set_postfix({
                'loss': f'{loss.item() * accumulation_steps:.4f}',
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

        # Calculate balanced accuracy
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
        freeze_epochs = self.config['training'].get('freeze_backbone_epochs', 0)

        # Freeze backbone for initial epochs if specified
        if freeze_epochs > 0 and hasattr(model, 'freeze_backbone'):
            model.freeze_backbone()
            logger.info(f"Backbone frozen for first {freeze_epochs} epochs")

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            # Unfreeze backbone after freeze_epochs
            if epoch == freeze_epochs and freeze_epochs > 0 and hasattr(model, 'unfreeze_backbone'):
                model.unfreeze_backbone()
                # Reset optimizer for full model training
                for param_group in optimizer.param_groups:
                    param_group['lr'] = self.config['training']['learning_rate']

            # Train
            train_metrics = self.train_epoch(model, train_loader, optimizer, criterion)

            # Validate
            val_metrics = self.validate(model, val_loader, criterion, 'val')

            # Update scheduler
            scheduler.step()

            # Get current LR
            current_lr = optimizer.param_groups[0]['lr']

            # Log metrics
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

            # Save best model (monitor balanced accuracy for imbalanced datasets)
            monitor_metric = self.config['callbacks']['early_stopping'].get('monitor', 'val_accuracy')
            if monitor_metric == 'val_balanced_accuracy':
                current_metric = val_metrics['balanced_accuracy']
                metric_name = 'balanced_acc'
            else:
                current_metric = val_metrics['accuracy']
                metric_name = 'val_acc'

            if current_metric > self.best_val_acc:
                self.best_val_acc = current_metric
                self.epochs_without_improvement = 0
                self._save_checkpoint(model, optimizer, 'best.pth')
                logger.info(f"  -> New best model! ({metric_name}: {self.best_val_acc:.2f}%)")
            else:
                self.epochs_without_improvement += 1

            # Early stopping
            if self.config['callbacks']['early_stopping']['enabled']:
                if self.epochs_without_improvement >= patience:
                    logger.info(f"\nEarly stopping triggered after {epoch+1} epochs")
                    break

        # Save final model
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

        # Save results
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
        """Save test results and generate plots"""
        # Save metrics
        results = {
            'loss': float(metrics['loss']),
            'accuracy': float(metrics['accuracy']),
            'balanced_accuracy': float(metrics['balanced_accuracy']),
            'timestamp': datetime.now().isoformat()
        }

        with open(self.results_dir / 'test_metrics.json', 'w') as f:
            json.dump(results, f, indent=2)

        if SKLEARN_AVAILABLE:
            # Classification report
            labels = metrics['labels']
            preds = metrics['predictions']

            # Dynamic class names based on num_classes
            num_classes = self.config['model']['num_classes']
            if num_classes == 2:
                class_names = ['CN', 'AD_trajectory']
            elif num_classes == 3:
                class_names = ['CN', 'MCI', 'AD']
            elif num_classes == 4:
                class_names = ['CN', 'MCIs', 'MCIc', 'AD']
            else:
                class_names = [f'Class_{i}' for i in range(num_classes)]

            report = classification_report(labels, preds, target_names=class_names)
            logger.info(f"\nClassification Report:\n{report}")

            with open(self.results_dir / 'classification_report.txt', 'w') as f:
                f.write(report)

            # Confusion matrix
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
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='Train 3D ViT for Alzheimer Classification')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file path')
    parser.add_argument('--test-only', action='store_true', help='Only run test evaluation')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Create trainer
    trainer = Trainer(config)

    # Build components
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

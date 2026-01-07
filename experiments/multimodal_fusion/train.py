#!/usr/bin/env python3
"""
Training script for Multi-Modal Fusion (MRI + Tabular)

Usage:
    python train.py --config config.yaml
"""

import os
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

from model import MultiModalFusion, build_model
from dataset import get_multimodal_dataloaders

import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.

    FL(p) = -alpha * (1-p)^gamma * log(p)

    Args:
        alpha: Weight for the positive class (default: 0.75 for minority)
        gamma: Focusing parameter (default: 2.0)
        weight: Optional class weights tensor
    """
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, weight: torch.Tensor = None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)

        # Apply alpha weighting based on class
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)

        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


try:
    from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
    import matplotlib.pyplot as plt
    import seaborn as sns
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MultiModalTrainer:
    """Trainer for Multi-Modal Fusion model"""

    def __init__(self, config: Dict):
        self.config = config
        self.device = self._setup_device()
        self.set_seed(config['training']['seed'])

        # Build descriptive experiment name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.timestamp = timestamp

        # Extract config info for naming
        exp_cfg = config.get('experiment', {})
        model_cfg = config.get('model', {})

        dataset = exp_cfg.get('dataset', 'unknown')
        task = exp_cfg.get('task', 'cn_ad_trajectory')
        backbone = model_cfg.get('backbone', {}).get('type', 'resnet')
        tabular_type = model_cfg.get('tabular', {}).get('type', 'mlp')
        fusion_method = model_cfg.get('fusion', {}).get('method', 'concat')

        # Create descriptive run name: task_dataset_backbone_tabular_fusion_timestamp
        self.run_name = f"{task}_{dataset}_{backbone}_{tabular_type}_{fusion_method}_{timestamp}"

        self.checkpoint_dir = Path(config['data']['checkpoints_dir']) / self.run_name
        base_results_dir = Path(config['data']['results_dir'])
        self.results_dir = base_results_dir / self.run_name

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Experiment: {self.run_name}")
        logger.info(f"Results will be saved to: {self.results_dir}")

        # Training state
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.epochs_without_improvement = 0

        # Training history for plotting curves
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_balanced_acc': [],
            'lr': []
        }

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

    def build_dataloaders(self):
        """Build multi-modal dataloaders"""
        logger.info("=" * 60)
        logger.info("BUILDING DATALOADERS")
        logger.info("=" * 60)

        cfg = self.config
        # Get image_size from backbone config or legacy vit config
        if 'backbone' in cfg['model']:
            image_size = cfg['model']['backbone'].get('image_size', 128)
        else:
            image_size = cfg['model']['vit']['image_size']
        preproc = cfg.get('preprocessing', {})

        train_loader, val_loader, test_loader, scaler = get_multimodal_dataloaders(
            train_csv=cfg['data']['train_csv'],
            val_csv=cfg['data']['val_csv'],
            test_csv=cfg['data']['test_csv'],
            tabular_features=cfg['data']['tabular_features'],
            batch_size=cfg['training']['batch_size'],
            target_shape=(image_size, image_size, image_size),
            num_workers=cfg['hardware']['num_workers'],
            pin_memory=cfg['hardware']['pin_memory'],
            augment=True,
            use_paper_preprocessing=preproc.get('use_paper_preprocessing', True),
            target_spacing=preproc.get('target_spacing', 1.75)
        )

        self.scaler = scaler
        return train_loader, val_loader, test_loader

    def build_model(self) -> nn.Module:
        """Build multi-modal fusion model"""
        logger.info("=" * 60)
        logger.info("BUILDING MODEL")
        logger.info("=" * 60)

        num_tabular_features = len(self.config['data']['tabular_features'])
        model = build_model(self.config, num_tabular_features)
        model = model.to(self.device)

        # Log parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")

        return model

    def build_optimizer(self, model: nn.Module):
        """Build optimizer and scheduler with optional layer-wise LR decay"""
        cfg = self.config['training']
        base_lr = cfg['learning_rate']
        weight_decay = cfg['weight_decay']
        lr_decay = cfg.get('layer_wise_lr_decay', 0.0)

        if lr_decay > 0 and hasattr(model, 'vit') and model.backbone_type == 'vit':
            # Layer-wise LR decay for ViT fine-tuning (not applicable to ResNet)
            param_groups = []
            vit = model.vit

            # Fusion head (highest LR)
            fusion_params = list(model.fusion.parameters()) + list(model.classifier.parameters())
            if hasattr(model, 'tabular_encoder'):
                fusion_params += list(model.tabular_encoder.parameters())
            param_groups.append({'params': fusion_params, 'lr': base_lr})

            # ViT head and norm
            if hasattr(vit, 'head'):
                param_groups.append({'params': list(vit.head.parameters()), 'lr': base_lr * lr_decay})
            if hasattr(vit, 'norm'):
                param_groups.append({'params': list(vit.norm.parameters()), 'lr': base_lr * lr_decay})

            # ViT blocks (deeper = lower LR)
            if hasattr(vit, 'blocks'):
                num_blocks = len(vit.blocks)
                for i, block in enumerate(reversed(list(vit.blocks))):
                    layer_lr = base_lr * (lr_decay ** (i + 2))
                    param_groups.append({'params': list(block.parameters()), 'lr': layer_lr})

            # Embeddings (lowest LR)
            embed_params = []
            if hasattr(vit, 'patch_embed'):
                embed_params += list(vit.patch_embed.parameters())
            if hasattr(vit, 'cls_token'):
                embed_params.append(vit.cls_token)
            if hasattr(vit, 'pos_embed'):
                embed_params.append(vit.pos_embed)
            if embed_params:
                embed_lr = base_lr * (lr_decay ** (num_blocks + 3))
                param_groups.append({'params': embed_params, 'lr': embed_lr})

            optimizer = optim.AdamW(param_groups, weight_decay=weight_decay)
            logger.info(f"Optimizer: AdamW with layer-wise LR decay ({lr_decay})")
        else:
            optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=base_lr,
                weight_decay=weight_decay
            )
            logger.info(f"Optimizer: AdamW (lr={base_lr}, wd={weight_decay})")

        scheduler = optim.lr_scheduler.CosineAnnealingLR(
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
                schedulers=[warmup_scheduler, scheduler],
                milestones=[warmup_epochs]
            )
            logger.info(f"Scheduler: CosineAnnealingLR with {warmup_epochs} warmup epochs")

        return optimizer, scheduler

    def build_criterion(self, train_loader: DataLoader) -> nn.Module:
        """Build loss function based on config"""
        loss_type = self.config['training'].get('loss_type', 'cross_entropy')
        use_weighted = self.config['training'].get('use_weighted_loss', True)

        # Calculate class weights if needed
        class_weights = None
        if use_weighted:
            labels = [label for _, _, label in train_loader.dataset]
            class_counts = np.bincount(labels)
            class_weights = 1.0 / class_counts
            class_weights = class_weights / class_weights.sum() * len(class_counts)
            class_weights = torch.FloatTensor(class_weights).to(self.device)

        # Label smoothing (regularization technique)
        label_smoothing = self.config['training'].get('label_smoothing', 0.0)

        if loss_type == 'focal':
            gamma = self.config['training'].get('focal_gamma', 2.0)
            alpha = self.config['training'].get('focal_alpha', 0.75)
            criterion = FocalLoss(alpha=alpha, gamma=gamma, weight=class_weights)
            logger.info(f"Loss: FocalLoss (alpha={alpha}, gamma={gamma}, weighted={use_weighted})")
        elif loss_type == 'cross_entropy' or loss_type == 'ce':
            criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
            if use_weighted:
                logger.info(f"Loss: Weighted CrossEntropyLoss (weights={class_weights.cpu().numpy()}, label_smoothing={label_smoothing})")
            else:
                logger.info(f"Loss: CrossEntropyLoss (label_smoothing={label_smoothing})")
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

        return criterion

    def train_epoch(self, model, train_loader, optimizer, criterion):
        """Train for one epoch with optional auxiliary losses"""
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        grad_clip = self.config['training'].get('gradient_clip', 1.0)

        # Check if auxiliary losses should be used
        use_aux_loss = self.config['training'].get('use_auxiliary_loss', False)
        aux_weight = self.config['training'].get('auxiliary_loss_weight', 0.3)

        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch+1} [Train]")
        for mri, tabular, labels in pbar:
            mri = mri.to(self.device)
            tabular = tabular.to(self.device)
            labels = labels.to(self.device)

            optimizer.zero_grad()

            # Forward pass with or without auxiliary outputs
            if use_aux_loss and hasattr(model, 'use_auxiliary_losses') and model.use_auxiliary_losses:
                outputs_dict = model(mri, tabular, return_auxiliary=True)
                outputs = outputs_dict['logits']
                mri_logits = outputs_dict['mri_logits']
                tab_logits = outputs_dict['tab_logits']

                # Main loss + auxiliary losses
                main_loss = criterion(outputs, labels)
                mri_loss = criterion(mri_logits, labels)
                tab_loss = criterion(tab_logits, labels)
                loss = main_loss + aux_weight * (mri_loss + tab_loss)
            else:
                outputs = model(mri, tabular)
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
        """Validate model"""
        model.eval()
        total_loss, correct, total = 0.0, 0, 0
        all_preds, all_labels = [], []

        pbar = tqdm(dataloader, desc=f"Epoch {self.current_epoch+1} [{split_name.upper()}]")
        for mri, tabular, labels in pbar:
            mri = mri.to(self.device)
            tabular = tabular.to(self.device)
            labels = labels.to(self.device)

            outputs = model(mri, tabular)
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
        """Main training loop"""
        logger.info("=" * 60)
        logger.info("STARTING TRAINING")
        logger.info("=" * 60)

        num_epochs = self.config['training']['epochs']
        patience = self.config['callbacks']['early_stopping']['patience']

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
                if self.epochs_without_improvement >= patience:
                    logger.info(f"\nEarly stopping triggered after {epoch+1} epochs")
                    break

        self._save_checkpoint(model, optimizer, 'last.pth')

        # Plot and save training curves
        self._plot_training_curves()

        logger.info("=" * 60)
        logger.info("TRAINING COMPLETE")
        logger.info(f"Best validation accuracy: {self.best_val_acc:.2f}%")
        logger.info("=" * 60)

    @torch.no_grad()
    def test_with_tta(self, model, dataloader, criterion, num_augmentations=8):
        """
        Test-Time Augmentation (TTA)

        Applies random augmentations at test time and averages predictions
        for improved robustness.
        """
        model.eval()
        total_loss, correct, total = 0.0, 0, 0
        all_preds, all_labels = [], []

        # TTA augmentations for 3D MRI
        def apply_tta_augmentation(mri, aug_idx):
            """Apply different augmentations based on index"""
            if aug_idx == 0:
                return mri  # Original
            elif aug_idx == 1:
                return torch.flip(mri, dims=[2])  # Flip depth
            elif aug_idx == 2:
                return torch.flip(mri, dims=[3])  # Flip height
            elif aug_idx == 3:
                return torch.flip(mri, dims=[4])  # Flip width
            elif aug_idx == 4:
                return torch.flip(mri, dims=[2, 3])  # Flip depth+height
            elif aug_idx == 5:
                return torch.flip(mri, dims=[2, 4])  # Flip depth+width
            elif aug_idx == 6:
                return torch.flip(mri, dims=[3, 4])  # Flip height+width
            elif aug_idx == 7:
                return torch.flip(mri, dims=[2, 3, 4])  # Flip all
            else:
                return mri

        pbar = tqdm(dataloader, desc=f"Test with TTA ({num_augmentations} augs)")
        for mri, tabular, labels in pbar:
            mri = mri.to(self.device)
            tabular = tabular.to(self.device)
            labels = labels.to(self.device)

            # Aggregate predictions from multiple augmentations
            all_logits = []
            for aug_idx in range(min(num_augmentations, 8)):
                aug_mri = apply_tta_augmentation(mri, aug_idx)
                outputs = model(aug_mri, tabular)
                all_logits.append(outputs)

            # Average logits across augmentations
            avg_logits = torch.stack(all_logits).mean(dim=0)
            loss = criterion(avg_logits, labels)

            total_loss += loss.item()
            _, predicted = avg_logits.max(1)
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

    def test(self, model, test_loader, criterion):
        """Evaluate on test set with optional TTA"""
        logger.info("=" * 60)
        logger.info("EVALUATING ON TEST SET")
        logger.info("=" * 60)

        # Load best model
        best_path = self.checkpoint_dir / 'best.pth'
        if best_path.exists():
            checkpoint = torch.load(best_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded best model from {best_path}")

        # Check if TTA is enabled
        use_tta = self.config['training'].get('use_tta', False)
        tta_augmentations = self.config['training'].get('tta_augmentations', 8)

        if use_tta:
            logger.info(f"Using Test-Time Augmentation with {tta_augmentations} augmentations")
            test_metrics = self.test_with_tta(model, test_loader, criterion, tta_augmentations)
        else:
            test_metrics = self.validate(model, test_loader, criterion, 'test')

        logger.info(f"\nTest Results:")
        logger.info(f"  Accuracy: {test_metrics['accuracy']:.2f}%")
        logger.info(f"  Balanced Accuracy: {test_metrics['balanced_accuracy']:.2f}%")

        self._save_results(test_metrics)
        return test_metrics

    def _save_checkpoint(self, model, optimizer, filename):
        """Save checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'config': self.config
        }
        torch.save(checkpoint, self.checkpoint_dir / filename)

    def _plot_training_curves(self):
        """Plot and save training/validation curves"""
        if not SKLEARN_AVAILABLE or len(self.history['train_loss']) == 0:
            return

        epochs = range(1, len(self.history['train_loss']) + 1)

        # Create figure with 2x2 subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: Loss curves
        ax1 = axes[0, 0]
        ax1.plot(epochs, self.history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        ax1.plot(epochs, self.history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Accuracy curves
        ax2 = axes[0, 1]
        ax2.plot(epochs, self.history['train_acc'], 'b-', label='Train Acc', linewidth=2)
        ax2.plot(epochs, self.history['val_acc'], 'r-', label='Val Acc', linewidth=2)
        ax2.plot(epochs, self.history['val_balanced_acc'], 'g--', label='Val Balanced Acc', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Learning rate
        ax3 = axes[1, 0]
        ax3.plot(epochs, self.history['lr'], 'purple', linewidth=2)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.set_title('Learning Rate Schedule')
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3)

        # Plot 4: Combined loss and accuracy (dual y-axis)
        ax4 = axes[1, 1]
        ax4_acc = ax4.twinx()

        line1, = ax4.plot(epochs, self.history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        line2, = ax4_acc.plot(epochs, self.history['val_acc'], 'b-', label='Val Acc', linewidth=2)

        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Loss', color='r')
        ax4_acc.set_ylabel('Accuracy (%)', color='b')
        ax4.set_title('Validation Loss vs Accuracy')
        ax4.tick_params(axis='y', labelcolor='r')
        ax4_acc.tick_params(axis='y', labelcolor='b')

        lines = [line1, line2]
        labels = [l.get_label() for l in lines]
        ax4.legend(lines, labels, loc='center right')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.results_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
        plt.close()

        # Also save history as JSON
        history_path = self.results_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)

        logger.info(f"Training curves saved to {self.results_dir / 'training_curves.png'}")

    def save_config(self):
        """Save config to results directory at start of training"""
        config_path = self.results_dir / 'config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        logger.info(f"Config saved to: {config_path}")

    def _save_results(self, metrics: Dict):
        """Save test results"""
        exp_cfg = self.config.get('experiment', {})
        model_cfg = self.config.get('model', {})

        results = {
            'run_name': self.run_name,
            'accuracy': float(metrics['accuracy']),
            'balanced_accuracy': float(metrics['balanced_accuracy']),
            'timestamp': datetime.now().isoformat(),
            # Experiment info
            'experiment_name': exp_cfg.get('name', 'multimodal_fusion'),
            'task': exp_cfg.get('task', 'cn_ad_trajectory'),
            'dataset': exp_cfg.get('dataset', 'unknown'),
            # Model info
            'backbone': model_cfg.get('backbone', {}).get('type', 'resnet'),
            'tabular_encoder': model_cfg.get('tabular', {}).get('type', 'mlp'),
            'fusion_method': model_cfg.get('fusion', {}).get('method', 'concat'),
            # Training info
            'epochs_trained': len(self.history['train_loss']),
            'best_val_acc': float(self.best_val_acc)
        }

        with open(self.results_dir / 'test_metrics.json', 'w') as f:
            json.dump(results, f, indent=2)

        if SKLEARN_AVAILABLE:
            labels = metrics['labels']
            preds = metrics['predictions']
            class_names = ['CN', 'AD_trajectory']

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
            plt.title('Confusion Matrix - Multi-Modal Fusion')
            plt.savefig(self.results_dir / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
            plt.close()


def load_config(config_path: str) -> Dict:
    """Load configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='Train Multi-Modal Fusion Model')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file')
    parser.add_argument('--test-only', action='store_true', help='Only test')
    args = parser.parse_args()

    config = load_config(args.config)

    trainer = MultiModalTrainer(config)
    train_loader, val_loader, test_loader = trainer.build_dataloaders()
    model = trainer.build_model()
    optimizer, scheduler = trainer.build_optimizer(model)
    criterion = trainer.build_criterion(train_loader)

    # Save config to results directory
    trainer.save_config()

    if args.test_only:
        trainer.test(model, test_loader, criterion)
    else:
        trainer.train(model, train_loader, val_loader, optimizer, scheduler, criterion)
        trainer.test(model, test_loader, criterion)


if __name__ == '__main__':
    main()

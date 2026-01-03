#!/usr/bin/env python3
"""
Training script for Multi-Modal Fusion with Modality Dropout

Features:
- Handles MRI + Tabular data with optional modality dropout
- Robust to missing modalities at inference time
- Uses FocalLoss for class imbalance

Usage:
    python train.py --config configs/config_nacc.yaml
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

from model import ModalityDropoutFusion, build_model
from dataset import get_dropout_dataloaders

# Project root for resolving relative paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
# Experiment directory (where this script lives)
EXPERIMENT_DIR = Path(__file__).parent

try:
    from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
    import matplotlib.pyplot as plt
    import seaborn as sns
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """Focal Loss for imbalanced classification."""
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class VectorScalingLoss(nn.Module):
    """
    Vector Scaling (VS) Loss for class-imbalanced classification.

    Based on: "Class-Balanced Deep Learning with Adaptive Vector Scaling Loss
    for Dementia Stage Detection" (PMC10924683)

    Combines additive and multiplicative logit adjustments:
    - Additive: l_k = τ * log(π_k) where π_k is class prior
    - Multiplicative: Δ_k = (N_k / N_max)^γ

    Args:
        class_counts: Number of samples per class
        tau: Additive adjustment strength [-1, 2], default 1.0
        gamma: Multiplicative adjustment strength [0, 0.5], default 0.3
        label_smoothing: Label smoothing factor [0, 0.2], default 0.1
    """
    def __init__(self, class_counts, tau=1.0, gamma=0.3, label_smoothing=0.1, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.num_classes = len(class_counts)

        # Calculate class priors
        total_samples = sum(class_counts)
        class_priors = np.array(class_counts) / total_samples

        # Additive adjustment: l_k = τ * log(π_k)
        # Higher tau = more adjustment for rare classes
        additive_adj = tau * np.log(class_priors + 1e-8)
        self.register_buffer('additive_adj', torch.FloatTensor(additive_adj))

        # Multiplicative adjustment: Δ_k = (N_k / N_max)^γ
        # Scales logits inversely to class frequency
        max_count = max(class_counts)
        multiplicative_adj = (np.array(class_counts) / max_count) ** gamma
        self.register_buffer('multiplicative_adj', torch.FloatTensor(multiplicative_adj))

        logger.info(f"VS Loss initialized:")
        logger.info(f"  tau={tau}, gamma={gamma}, label_smoothing={label_smoothing}")
        logger.info(f"  Additive adjustments: {additive_adj}")
        logger.info(f"  Multiplicative adjustments: {multiplicative_adj}")

    def forward(self, inputs, targets):
        # Apply multiplicative scaling to logits
        scaled_logits = inputs * self.multiplicative_adj.unsqueeze(0)

        # Apply additive adjustment (logit adjustment)
        adjusted_logits = scaled_logits + self.additive_adj.unsqueeze(0)

        # Cross entropy with label smoothing
        loss = nn.functional.cross_entropy(
            adjusted_logits,
            targets,
            reduction=self.reduction,
            label_smoothing=self.label_smoothing
        )

        return loss


class VSFocalLoss(nn.Module):
    """
    Combined Vector Scaling + Focal Loss.
    Best of both worlds: VS handles class imbalance, Focal focuses on hard examples.
    """
    def __init__(self, class_counts, tau=1.0, gamma_vs=0.3, gamma_focal=2.0,
                 label_smoothing=0.1, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.gamma_focal = gamma_focal
        self.num_classes = len(class_counts)

        # VS adjustments
        total_samples = sum(class_counts)
        class_priors = np.array(class_counts) / total_samples
        additive_adj = tau * np.log(class_priors + 1e-8)
        self.register_buffer('additive_adj', torch.FloatTensor(additive_adj))

        max_count = max(class_counts)
        multiplicative_adj = (np.array(class_counts) / max_count) ** gamma_vs
        self.register_buffer('multiplicative_adj', torch.FloatTensor(multiplicative_adj))

        logger.info(f"VS-Focal Loss: tau={tau}, gamma_vs={gamma_vs}, gamma_focal={gamma_focal}")

    def forward(self, inputs, targets):
        # Apply VS adjustments
        scaled_logits = inputs * self.multiplicative_adj.unsqueeze(0)
        adjusted_logits = scaled_logits + self.additive_adj.unsqueeze(0)

        # Compute focal loss on adjusted logits
        ce_loss = nn.functional.cross_entropy(
            adjusted_logits, targets, reduction='none',
            label_smoothing=self.label_smoothing
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma_focal) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


def get_layer_wise_lr_decay_params(model, base_lr: float, weight_decay: float, lr_decay: float = 0.75):
    """
    Create parameter groups with layer-wise learning rate decay.

    Based on mri_vit_ad strategy:
    - Head/fusion gets highest LR (base_lr)
    - Deeper transformer blocks get progressively lower LR
    """
    param_groups = []

    # Get the MRI backbone
    backbone = model.mri_backbone
    num_blocks = len(backbone.blocks)

    # Layer 0: Classification head + fusion layers (highest LR)
    head_params = []
    head_params.extend(list(model.classifier.parameters()))
    head_params.extend(list(model.fusion.parameters()) if hasattr(model.fusion, 'parameters') else [])
    head_params.extend(list(model.tabular_encoder.parameters()))
    if model.use_modality_embeddings:
        head_params.extend([model.mri_available_embed, model.mri_missing_embed,
                           model.tab_available_embed, model.tab_missing_embed])

    if head_params:
        param_groups.append({
            'params': [p for p in head_params if p.requires_grad],
            'lr': base_lr,
            'weight_decay': weight_decay,
            'layer_name': 'head_fusion'
        })

    # Layer 1: Backbone norm
    norm_params = list(backbone.norm.parameters())
    if norm_params:
        param_groups.append({
            'params': norm_params,
            'lr': base_lr * lr_decay,
            'weight_decay': weight_decay,
            'layer_name': 'backbone_norm'
        })

    # Layers 2+: Transformer blocks (reverse order)
    for i in range(num_blocks - 1, -1, -1):
        block_params = list(backbone.blocks[i].parameters())
        layer_idx = num_blocks - i + 1
        lr = base_lr * (lr_decay ** layer_idx)

        if block_params:
            param_groups.append({
                'params': block_params,
                'lr': lr,
                'weight_decay': weight_decay,
                'layer_name': f'blocks.{i}'
            })

    # Embeddings (lowest LR)
    embed_layer_idx = num_blocks + 2
    embed_lr = base_lr * (lr_decay ** embed_layer_idx)

    embed_params = [backbone.cls_token, backbone.pos_embed]
    embed_params.extend(list(backbone.patch_embed.parameters()))
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
    """Trainer for Multimodal Dropout Model"""

    def __init__(self, config: Dict):
        self.config = config
        self.device = self._setup_device()
        self.set_seed(config['training']['seed'])

        # Create output directories
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = config.get('experiment', {}).get('name', 'multimodal_dropout')
        self.run_name = f"{exp_name}_{timestamp}"

        # Resolve paths relative to experiment directory
        self.checkpoint_dir = EXPERIMENT_DIR / config['data']['checkpoints_dir'] / self.run_name
        self.results_dir = EXPERIMENT_DIR / config['data']['results_dir'] / self.run_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self.current_epoch = 0
        self.best_val_metric = 0.0
        self.epochs_without_improvement = 0

        # History
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [], 'val_balanced_acc': [],
            'lr': []
        }

        logger.info(f"Run name: {self.run_name}")
        logger.info(f"Results: {self.results_dir}")

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
        """Build dataloaders with modality dropout."""
        cfg = self.config
        image_size = cfg['model']['backbone'].get('image_size', 128)
        preproc = cfg.get('preprocessing', {})
        dropout_cfg = cfg.get('modality_dropout', {})

        # Resolve paths relative to project root
        train_csv = PROJECT_ROOT / cfg['data']['train_csv']
        val_csv = PROJECT_ROOT / cfg['data']['val_csv']
        test_csv = PROJECT_ROOT / cfg['data']['test_csv']

        train_loader, val_loader, test_loader, scaler = get_dropout_dataloaders(
            train_csv=str(train_csv),
            val_csv=str(val_csv),
            test_csv=str(test_csv),
            tabular_features=cfg['data']['tabular_features'],
            batch_size=cfg['training']['batch_size'],
            target_shape=(image_size, image_size, image_size),
            num_workers=cfg['hardware']['num_workers'],
            pin_memory=cfg['hardware']['pin_memory'],
            mri_dropout_rate=dropout_cfg.get('mri_dropout_rate', 0.2),
            tabular_dropout_rate=dropout_cfg.get('tabular_dropout_rate', 0.2),
            use_paper_preprocessing=preproc.get('use_paper_preprocessing', True),
            target_spacing=preproc.get('target_spacing', 1.75)
        )

        self.scaler = scaler
        return train_loader, val_loader, test_loader

    def build_model(self) -> nn.Module:
        """Build model."""
        logger.info("=" * 60)
        logger.info("BUILDING MODEL")
        logger.info("=" * 60)

        num_features = len(self.config['data']['tabular_features'])
        model = build_model(self.config, num_features)
        model = model.to(self.device)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")

        return model

    def build_optimizer(self, model: nn.Module):
        """Build optimizer and scheduler with optional layer-wise LR decay."""
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
                filter(lambda p: p.requires_grad, model.parameters()),
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
        """Build loss function with VS Loss, Focal Loss, or standard CE."""
        training_cfg = self.config['training']
        loss_type = training_cfg.get('loss_type', 'vs_focal')  # vs, vs_focal, focal, ce
        label_smoothing = training_cfg.get('label_smoothing', 0.1)

        # Get class distribution
        labels = train_loader.dataset.get_labels()
        class_counts = list(np.bincount(labels))
        logger.info(f"Class distribution: {dict(enumerate(class_counts))}")

        # VS Loss parameters
        vs_tau = training_cfg.get('vs_tau', 1.0)  # [-1, 2]
        vs_gamma = training_cfg.get('vs_gamma', 0.3)  # [0, 0.5]
        focal_gamma = training_cfg.get('focal_gamma', 2.0)

        if loss_type == 'vs':
            # Pure Vector Scaling Loss
            criterion = VectorScalingLoss(
                class_counts=class_counts,
                tau=vs_tau,
                gamma=vs_gamma,
                label_smoothing=label_smoothing
            )
            logger.info(f"Loss: VectorScalingLoss (tau={vs_tau}, gamma={vs_gamma}, smoothing={label_smoothing})")

        elif loss_type == 'vs_focal':
            # Combined VS + Focal Loss (recommended)
            criterion = VSFocalLoss(
                class_counts=class_counts,
                tau=vs_tau,
                gamma_vs=vs_gamma,
                gamma_focal=focal_gamma,
                label_smoothing=label_smoothing
            )
            logger.info(f"Loss: VSFocalLoss (tau={vs_tau}, gamma_vs={vs_gamma}, "
                       f"gamma_focal={focal_gamma}, smoothing={label_smoothing})")

        elif loss_type == 'focal':
            # Standard Focal Loss with class weights
            n_samples = len(labels)
            n_classes = len(class_counts)
            class_weights = n_samples / (n_classes * np.array(class_counts))
            class_weights = torch.FloatTensor(class_weights).to(self.device)
            criterion = FocalLoss(alpha=class_weights, gamma=focal_gamma)
            logger.info(f"Loss: FocalLoss (gamma={focal_gamma}, weighted=True)")

        else:  # 'ce' or default
            # Standard CrossEntropy with optional class weights
            if training_cfg.get('use_weighted_loss', False):
                n_samples = len(labels)
                n_classes = len(class_counts)
                class_weights = n_samples / (n_classes * np.array(class_counts))
                class_weights = torch.FloatTensor(class_weights).to(self.device)
                criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
                logger.info(f"Loss: Weighted CrossEntropyLoss (smoothing={label_smoothing})")
            else:
                criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
                logger.info(f"Loss: CrossEntropyLoss (smoothing={label_smoothing})")

        return criterion

    def train_epoch(self, model, train_loader, optimizer, criterion):
        """Train for one epoch."""
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        grad_clip = self.config['training'].get('gradient_clip', 1.0)

        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch+1} [Train]")
        for mri, tabular, modality_mask, labels in pbar:
            mri = mri.to(self.device)
            tabular = tabular.to(self.device)
            modality_mask = modality_mask.to(self.device)
            labels = labels.to(self.device)

            optimizer.zero_grad()
            outputs = model(mri, tabular, modality_mask)
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
        """Validate model."""
        model.eval()
        total_loss, correct, total = 0.0, 0, 0
        all_preds, all_labels = [], []

        pbar = tqdm(dataloader, desc=f"Epoch {self.current_epoch+1} [{split_name.upper()}]")
        for mri, tabular, modality_mask, labels in pbar:
            mri = mri.to(self.device)
            tabular = tabular.to(self.device)
            modality_mask = modality_mask.to(self.device)
            labels = labels.to(self.device)

            outputs = model(mri, tabular, modality_mask)
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
        """Main training loop."""
        logger.info("=" * 60)
        logger.info("STARTING TRAINING")
        logger.info("=" * 60)

        num_epochs = self.config['training']['epochs']
        patience = self.config['callbacks']['early_stopping']['patience']
        monitor = self.config['callbacks']['early_stopping'].get('monitor', 'val_balanced_accuracy')

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
            if monitor == 'val_balanced_accuracy':
                current_metric = val_metrics['balanced_accuracy']
            else:
                current_metric = val_metrics['accuracy']

            if current_metric > self.best_val_metric:
                self.best_val_metric = current_metric
                self.epochs_without_improvement = 0
                self._save_checkpoint(model, optimizer, 'best.pth')
                logger.info(f"  -> New best model! ({monitor}: {self.best_val_metric:.2f}%)")
            else:
                self.epochs_without_improvement += 1

            # Early stopping
            if self.config['callbacks']['early_stopping']['enabled']:
                if self.epochs_without_improvement >= patience:
                    logger.info(f"\nEarly stopping after {epoch+1} epochs")
                    break

        self._save_checkpoint(model, optimizer, 'last.pth')
        self._plot_training_curves()

        logger.info("=" * 60)
        logger.info("TRAINING COMPLETE")
        logger.info(f"Best {monitor}: {self.best_val_metric:.2f}%")
        logger.info("=" * 60)

    def test(self, model, test_loader, criterion):
        """Evaluate on test set."""
        logger.info("=" * 60)
        logger.info("EVALUATING ON TEST SET")
        logger.info("=" * 60)

        best_path = self.checkpoint_dir / 'best.pth'
        if best_path.exists():
            checkpoint = torch.load(best_path, map_location=self.device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded best model from {best_path}")

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
            'best_val_metric': self.best_val_metric,
            'config': self.config
        }
        torch.save(checkpoint, self.checkpoint_dir / filename)

    def _plot_training_curves(self):
        if not SKLEARN_AVAILABLE or len(self.history['train_loss']) == 0:
            return

        epochs = range(1, len(self.history['train_loss']) + 1)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Loss
        axes[0, 0].plot(epochs, self.history['train_loss'], 'b-', label='Train')
        axes[0, 0].plot(epochs, self.history['val_loss'], 'r-', label='Val')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Accuracy
        axes[0, 1].plot(epochs, self.history['train_acc'], 'b-', label='Train')
        axes[0, 1].plot(epochs, self.history['val_acc'], 'r-', label='Val')
        axes[0, 1].plot(epochs, self.history['val_balanced_acc'], 'g--', label='Val Balanced')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].set_title('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Learning rate
        axes[1, 0].plot(epochs, self.history['lr'], 'purple')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('LR')
        axes[1, 0].set_title('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)

        # Empty or additional plot
        axes[1, 1].axis('off')

        plt.tight_layout()
        plt.savefig(self.results_dir / 'training_curves.png', dpi=150)
        plt.close()

        with open(self.results_dir / 'training_history.json', 'w') as f:
            json.dump(self.history, f, indent=2)

    def _save_results(self, metrics):
        results = {
            'accuracy': float(metrics['accuracy']),
            'balanced_accuracy': float(metrics['balanced_accuracy']),
            'timestamp': datetime.now().isoformat()
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

            cm = confusion_matrix(labels, preds)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=class_names, yticklabels=class_names)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Confusion Matrix')
            plt.savefig(self.results_dir / 'confusion_matrix.png', dpi=150)
            plt.close()


def load_config(config_path: str) -> Dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='Train Multimodal Dropout Model')
    parser.add_argument('--config', type=str, required=True, help='Config file')
    parser.add_argument('--test-only', action='store_true', help='Only test')
    args = parser.parse_args()

    config = load_config(args.config)

    trainer = Trainer(config)
    train_loader, val_loader, test_loader = trainer.build_dataloaders()
    model = trainer.build_model()
    optimizer, scheduler = trainer.build_optimizer(model)
    criterion = trainer.build_criterion(train_loader)

    # Save config
    with open(trainer.results_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    if args.test_only:
        trainer.test(model, test_loader, criterion)
    else:
        trainer.train(model, train_loader, val_loader, optimizer, scheduler, criterion)
        trainer.test(model, test_loader, criterion)


if __name__ == '__main__':
    main()

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

        # Create directories
        self.checkpoint_dir = Path(config['data']['checkpoints_dir'])
        self.results_dir = Path(config['data']['results_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.epochs_without_improvement = 0

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
        """Build loss function with class weights"""
        if self.config['training']['use_weighted_loss']:
            labels = [label for _, _, label in train_loader.dataset]
            class_counts = np.bincount(labels)
            class_weights = 1.0 / class_counts
            class_weights = class_weights / class_weights.sum() * len(class_counts)
            class_weights = torch.FloatTensor(class_weights).to(self.device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            logger.info(f"Loss: Weighted CrossEntropyLoss (weights={class_weights.cpu().numpy()})")
        else:
            criterion = nn.CrossEntropyLoss()
            logger.info("Loss: CrossEntropyLoss")
        return criterion

    def train_epoch(self, model, train_loader, optimizer, criterion):
        """Train for one epoch"""
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        grad_clip = self.config['training'].get('gradient_clip', 1.0)

        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch+1} [Train]")
        for mri, tabular, labels in pbar:
            mri = mri.to(self.device)
            tabular = tabular.to(self.device)
            labels = labels.to(self.device)

            optimizer.zero_grad()
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

        logger.info("=" * 60)
        logger.info("TRAINING COMPLETE")
        logger.info(f"Best validation accuracy: {self.best_val_acc:.2f}%")
        logger.info("=" * 60)

    def test(self, model, test_loader, criterion):
        """Evaluate on test set"""
        logger.info("=" * 60)
        logger.info("EVALUATING ON TEST SET")
        logger.info("=" * 60)

        # Load best model
        best_path = self.checkpoint_dir / 'best.pth'
        if best_path.exists():
            checkpoint = torch.load(best_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded best model from {best_path}")

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

    def _save_results(self, metrics: Dict):
        """Save test results"""
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

    if args.test_only:
        trainer.test(model, test_loader, criterion)
    else:
        trainer.train(model, train_loader, val_loader, optimizer, scheduler, criterion)
        trainer.test(model, test_loader, criterion)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Train ensemble of ResNet-18 models on balanced subsets

This script:
1. Trains separate ResNet-18 models on each balanced subset
2. Saves each model checkpoint
3. Evaluates individual models and ensemble on validation/test sets
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'cn_mci_ad_medicalnet'))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import logging
from model_resnet3d import resnet18, load_pretrained_weights as load_pretrained_resnet
from model_seresnet3d import seresnet18, load_pretrained_weights as load_pretrained_seresnet
from dataset import ADNIDataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EnsembleTrainer:
    """Train ensemble of models"""

    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = []
        self.checkpoints_dir = Path(config['checkpoints_dir'])
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    def train_single_model(
        self,
        train_csv: str,
        val_csv: str,
        model_id: int,
        pretrained_path: str = None,
        use_se: bool = True
    ):
        """Train a single model on a balanced subset"""
        logger.info(f"\n{'='*80}")
        logger.info(f"TRAINING MODEL {model_id}")
        logger.info(f"{'='*80}")

        # Build model (with or without SE blocks)
        if use_se:
            logger.info("Using SEResNet-18 (with SE blocks)")
            model = seresnet18(num_classes=3, in_channels=1, use_se=True)
            if pretrained_path and Path(pretrained_path).exists():
                model = load_pretrained_seresnet(model, pretrained_path, num_classes=3)
        else:
            logger.info("Using vanilla ResNet-18")
            model = resnet18(num_classes=3, in_channels=1)
            if pretrained_path and Path(pretrained_path).exists():
                model = load_pretrained_resnet(model, pretrained_path, num_classes=3)

        model = model.to(self.device)

        # Build dataloaders
        train_dataset = ADNIDataset(train_csv, target_shape=(192, 192, 192))
        val_dataset = ADNIDataset(val_csv, target_shape=(192, 192, 192))

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config['num_workers'],
            pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers'],
            pin_memory=True
        )

        # Optimizer and loss
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )

        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config['epochs'],
            eta_min=1e-6
        )

        # Weighted loss for balance
        labels = [label for _, label in train_dataset]
        class_counts = np.bincount(labels)
        class_weights = 1.0 / class_counts
        class_weights = class_weights / class_weights.sum()
        class_weights = torch.FloatTensor(class_weights).to(self.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Training loop
        best_val_acc = 0.0
        patience_counter = 0

        for epoch in range(self.config['epochs']):
            # Train
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config['epochs']} [Train]")
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                logits = model(images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()

                pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100*train_correct/train_total:.2f}%'})

            train_acc = 100 * train_correct / train_total

            # Validate
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    logits = model(images)
                    loss = criterion(logits, labels)

                    val_loss += loss.item()
                    _, predicted = torch.max(logits, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

            val_acc = 100 * val_correct / val_total

            scheduler.step()

            logger.info(f"Epoch {epoch+1}: Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                checkpoint_path = self.checkpoints_dir / f'model_{model_id}_best.pth'
                torch.save(model.state_dict(), checkpoint_path)
                logger.info(f"  ✓ Best model saved (val_acc: {best_val_acc:.2f}%)")
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= self.config['patience']:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

        logger.info(f"Model {model_id} training complete. Best val acc: {best_val_acc:.2f}%")
        return best_val_acc

    def train_ensemble(
        self,
        subset_dir: str,
        val_csv: str,
        num_models: int,
        pretrained_path: str = None,
        use_se: bool = True
    ):
        """Train all models in the ensemble"""
        logger.info("="*80)
        logger.info("TRAINING ENSEMBLE OF MODELS")
        logger.info("="*80)
        logger.info(f"Architecture: {'SEResNet-18' if use_se else 'ResNet-18'}")

        subset_dir = Path(subset_dir)
        best_val_accs = []

        for i in range(1, num_models + 1):
            train_csv = subset_dir / f'train_subset_{i}.csv'

            if not train_csv.exists():
                logger.error(f"Subset CSV not found: {train_csv}")
                continue

            val_acc = self.train_single_model(
                train_csv=str(train_csv),
                val_csv=val_csv,
                model_id=i,
                pretrained_path=pretrained_path,
                use_se=use_se
            )
            best_val_accs.append(val_acc)

        logger.info("\n" + "="*80)
        logger.info("ENSEMBLE TRAINING COMPLETE")
        logger.info("="*80)
        logger.info(f"Individual model validation accuracies:")
        for i, acc in enumerate(best_val_accs, 1):
            logger.info(f"  Model {i}: {acc:.2f}%")
        logger.info(f"Average: {np.mean(best_val_accs):.2f}% ± {np.std(best_val_accs):.2f}%")


def main():
    parser = argparse.ArgumentParser(description='Train ensemble of ResNet-18 models')
    parser.add_argument('--subset-dir', type=str, required=True,
                       help='Directory with balanced subset CSVs')
    parser.add_argument('--val-csv', type=str, required=True,
                       help='Validation CSV')
    parser.add_argument('--num-models', type=int, default=5,
                       help='Number of models in ensemble')
    parser.add_argument('--pretrained', type=str, default=None,
                       help='Path to pretrained weights')
    parser.add_argument('--checkpoints-dir', type=str, default='checkpoints',
                       help='Directory to save model checkpoints')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Max epochs per model')
    parser.add_argument('--learning-rate', type=float, default=0.0001,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.0001,
                       help='Weight decay')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of dataloader workers')
    parser.add_argument('--use-se', action='store_true', default=True,
                       help='Use SE blocks (SEResNet-18)')
    parser.add_argument('--no-se', dest='use_se', action='store_false',
                       help='Use vanilla ResNet-18 without SE blocks')

    args = parser.parse_args()

    config = {
        'checkpoints_dir': args.checkpoints_dir,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'patience': args.patience,
        'num_workers': args.num_workers
    }

    trainer = EnsembleTrainer(config)
    trainer.train_ensemble(
        subset_dir=args.subset_dir,
        val_csv=args.val_csv,
        num_models=args.num_models,
        pretrained_path=args.pretrained,
        use_se=args.use_se
    )


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Train DenseNet3D with CLIP for Alzheimer's Disease Classification

This script trains a DenseNet3D model with CLIP-based pretraining for
AD/CN/MCI classification using the ADNI dataset.

Features:
- DenseNet3D backbone with CLIP initialization
- PyTorch Lightning training framework
- Patient-aware train/val/test split
- Class balancing with weighted sampling
- TensorBoard logging
- Model checkpointing
- Early stopping

Usage:
    python train_densenet3d_clip.py --config config.yaml
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split

# Add ADNI_unimodal_models to path (from project root)
repo_path = Path(__file__).parent.parent.parent / 'ADNI_unimodal_models'
sys.path.insert(0, str(repo_path))

from encoders.image_encoders import DenseNet3D
from models.classifier import CustomClassifier
from models.model import UnimodalModel
from datasets import CustomDataset, MMClassificationCollator, get_adni_dataset_new
from seeder import seed_worker

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ADNIDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for ADNI dataset

    Handles data loading, preprocessing, and splitting with patient-aware
    stratification to prevent data leakage.
    """

    def __init__(
        self,
        data_path: str,
        config: Dict,
        batch_size: int = 8,
        num_workers: int = 4,
        seed: int = 42
    ):
        super().__init__()
        self.data_path = data_path
        self.config = config
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.classes = None

    def setup(self, stage: Optional[str] = None):
        """Setup train/val/test datasets"""
        logger.info("Setting up ADNI datasets...")

        # Load and split data using patient-aware split
        datasets = get_adni_dataset_new(
            data_path=self.data_path,
            cat_cols=self.config.get('cat_cols', []),
            cont_cols=self.config.get('cont_cols', []),
            target_column=self.config.get('target_column', 'Group'),
            subject_column=self.config.get('subject_column', 'Subject'),
            seed=self.seed,
            normalize=self.config.get('normalize', False),
            make_dummies=self.config.get('make_dummies', True),
            patient_based_split=True
        )

        train_df, val_df, test_df, train_labels, val_labels, test_labels, self.classes = datasets

        # Create datasets
        self.train_dataset = CustomDataset(train_df, train_labels, self.config)
        self.val_dataset = CustomDataset(val_df, val_labels, self.config)
        self.test_dataset = CustomDataset(test_df, test_labels, self.config)

        logger.info(f"Train: {len(self.train_dataset)}, Val: {len(self.val_dataset)}, Test: {len(self.test_dataset)}")
        logger.info(f"Classes: {self.classes}")

    def train_dataloader(self):
        """Create training dataloader with weighted sampling"""
        # Get class distribution
        labels = [self.train_dataset[i]['label'] for i in range(len(self.train_dataset))]
        class_counts = np.bincount(labels)
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[labels]

        sampler = torch.utils.data.WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

        collator = MMClassificationCollator(
            image_processing=self.config.get('image_processing'),
            tabular_processing=self.config.get('tabular_processing')
        )

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=collator,
            worker_init_fn=seed_worker,
            persistent_workers=True if self.num_workers > 0 else False
        )

    def val_dataloader(self):
        """Create validation dataloader"""
        collator = MMClassificationCollator(
            image_processing=self.config.get('image_processing'),
            tabular_processing=self.config.get('tabular_processing')
        )

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collator,
            worker_init_fn=seed_worker,
            persistent_workers=True if self.num_workers > 0 else False
        )

    def test_dataloader(self):
        """Create test dataloader"""
        collator = MMClassificationCollator(
            image_processing=self.config.get('image_processing'),
            tabular_processing=self.config.get('tabular_processing')
        )

        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collator,
            worker_init_fn=seed_worker,
            persistent_workers=True if self.num_workers > 0 else False
        )


class DenseNet3DCLIP(pl.LightningModule):
    """
    PyTorch Lightning module for DenseNet3D with CLIP

    Implements training, validation, and testing logic with comprehensive
    metrics tracking and logging.
    """

    def __init__(
        self,
        num_classes: int,
        classes: list,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        freeze_encoder: bool = False,
        hidden_dim: int = 0,
        dropout_rate: float = 0.0
    ):
        super().__init__()
        self.save_hyperparameters()

        self.num_classes = num_classes
        self.classes = classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # Create encoder
        logger.info("Initializing DenseNet3D encoder...")
        self.image_encoder = DenseNet3D(
            freeze=freeze_encoder,
            include_head=False
        )

        # Create classifier
        logger.info("Initializing classifier...")
        self.classifier = CustomClassifier(
            hidden_dim=hidden_dim,
            activation_fun=nn.ReLU(),
            num_class=num_classes,
            task="multiclass",
            dropout_rate=dropout_rate
        )

        # Create model
        encoders = {"image": self.image_encoder}
        self.model = UnimodalModel(encoders=encoders, classifier=self.classifier)

        # Loss function with class weights
        self.criterion = nn.CrossEntropyLoss()

        # Metrics storage
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, batch):
        """Forward pass"""
        return self.model(batch)

    def training_step(self, batch, batch_idx):
        """Training step"""
        logits = self(batch)
        labels = batch['label']

        loss = self.criterion(logits, labels)

        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()

        # Log metrics
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/acc', acc, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step"""
        logits = self(batch)
        labels = batch['label']

        loss = self.criterion(logits, labels)

        # Get predictions
        preds = torch.argmax(logits, dim=1)

        # Store for epoch-end metrics
        self.validation_step_outputs.append({
            'loss': loss,
            'preds': preds,
            'labels': labels,
            'logits': logits
        })

        return loss

    def on_validation_epoch_end(self):
        """Compute validation metrics at epoch end"""
        outputs = self.validation_step_outputs

        # Gather predictions and labels
        all_preds = torch.cat([x['preds'] for x in outputs])
        all_labels = torch.cat([x['labels'] for x in outputs])
        all_logits = torch.cat([x['logits'] for x in outputs])

        # Calculate metrics
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        accuracy = accuracy_score(all_labels.cpu(), all_preds.cpu())

        # Per-class metrics
        f1_macro = f1_score(all_labels.cpu(), all_preds.cpu(), average='macro')
        precision = precision_score(all_labels.cpu(), all_preds.cpu(), average='macro')
        recall = recall_score(all_labels.cpu(), all_preds.cpu(), average='macro')

        # Log metrics
        self.log('val/loss', avg_loss, prog_bar=True)
        self.log('val/acc', accuracy, prog_bar=True)
        self.log('val/f1_macro', f1_macro)
        self.log('val/precision', precision)
        self.log('val/recall', recall)

        # Confusion matrix
        cm = confusion_matrix(all_labels.cpu(), all_preds.cpu())
        logger.info(f"\nValidation Confusion Matrix:\n{cm}")

        # Clear outputs
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        """Test step"""
        logits = self(batch)
        labels = batch['label']

        loss = self.criterion(logits, labels)
        preds = torch.argmax(logits, dim=1)

        self.test_step_outputs.append({
            'loss': loss,
            'preds': preds,
            'labels': labels,
            'logits': logits
        })

        return loss

    def on_test_epoch_end(self):
        """Compute test metrics at epoch end"""
        outputs = self.test_step_outputs

        all_preds = torch.cat([x['preds'] for x in outputs])
        all_labels = torch.cat([x['labels'] for x in outputs])

        # Calculate final metrics
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        accuracy = accuracy_score(all_labels.cpu(), all_preds.cpu())
        f1_macro = f1_score(all_labels.cpu(), all_preds.cpu(), average='macro')

        # Log final test metrics
        self.log('test/loss', avg_loss)
        self.log('test/acc', accuracy)
        self.log('test/f1_macro', f1_macro)

        # Print detailed results
        cm = confusion_matrix(all_labels.cpu(), all_preds.cpu())
        logger.info(f"\nTest Results:")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  F1 Score (macro): {f1_macro:.4f}")
        logger.info(f"\nTest Confusion Matrix:\n{cm}")

        self.test_step_outputs.clear()

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler"""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val/loss'
            }
        }


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description='Train DenseNet3D with CLIP for AD classification')

    parser.add_argument('--config', type=str, default='model_3d/configs/training_config.yaml',
                       help='Path to config YAML file')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to ADNI CSV file with nii_path column')
    parser.add_argument('--output', type=str, default='model_3d/outputs/training',
                       help='Output directory for checkpoints and logs')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Maximum number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                       help='Weight decay')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of dataloader workers')
    parser.add_argument('--gpus', type=int, default=0,
                       help='Number of GPUs to use (0 for CPU)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--freeze-encoder', action='store_true',
                       help='Freeze encoder weights')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')

    args = parser.parse_args()

    # Set random seed
    pl.seed_everything(args.seed)

    # Load configuration
    logger.info(f"Loading config from {args.config}")
    config = load_config(args.config)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*80)
    print("🧠 DENSENET3D + CLIP TRAINING FOR ALZHEIMER'S DETECTION")
    print("="*80 + "\n")

    # Create data module
    logger.info("Creating data module...")
    data_module = ADNIDataModule(
        data_path=args.data,
        config=config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed
    )
    data_module.setup()

    # Create model
    logger.info("Creating model...")
    model = DenseNet3DCLIP(
        num_classes=len(data_module.classes),
        classes=data_module.classes,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        freeze_encoder=args.freeze_encoder
    )

    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir / 'checkpoints',
        filename='densenet3d_clip-{epoch:02d}-{val/acc:.4f}',
        monitor='val/acc',
        mode='max',
        save_top_k=3,
        save_last=True
    )

    early_stop_callback = EarlyStopping(
        monitor='val/loss',
        patience=10,
        mode='min',
        verbose=True
    )

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # Setup logger
    tb_logger = TensorBoardLogger(
        save_dir=output_dir,
        name='tensorboard_logs'
    )

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator='gpu' if args.gpus > 0 else 'cpu',
        devices=args.gpus if args.gpus > 0 else 1,
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
        logger=tb_logger,
        log_every_n_steps=10,
        enable_progress_bar=True,
        deterministic=True
    )

    # Train
    logger.info("Starting training...")
    trainer.fit(model, data_module, ckpt_path=args.resume)

    # Test
    logger.info("Running final test...")
    trainer.test(model, data_module)

    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE")
    print("="*80)
    print(f"\nBest checkpoint: {checkpoint_callback.best_model_path}")
    print(f"TensorBoard logs: {output_dir / 'tensorboard_logs'}")
    print(f"\nTo view training progress:")
    print(f"  tensorboard --logdir {output_dir / 'tensorboard_logs'}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

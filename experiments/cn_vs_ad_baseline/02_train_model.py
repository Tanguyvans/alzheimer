"""
Train DenseNet3D for CN vs AD Binary Classification

This script trains a DenseNet3D model for predicting MCI-to-AD conversion
using single baseline MRI scans.

Features:
- DenseNet3D backbone with CLIP initialization
- Binary classification (CN vs AD)
- PyTorch Lightning training framework
- Class balancing with weighted sampling
- TensorBoard logging
- Model checkpointing and early stopping

Usage:
    # Using config file (recommended)
    python train_densenet3d_binary.py --config config.yaml

    # Using command line arguments
    python train_densenet3d_binary.py \\
        --train-csv outputs/mci_baseline/dataset_splits/train.csv \\
        --val-csv outputs/mci_baseline/dataset_splits/val.csv \\
        --test-csv outputs/mci_baseline/dataset_splits/test.csv \\
        --batch-size 8 \\
        --epochs 50
"""

import sys
import os
import argparse
import logging
from pathlib import Path
from typing import Dict, Optional
import yaml
from dotenv import load_dotenv

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score

# Add current directory to path for local imports
sys.path.insert(0, str(Path(__file__).parent))

# Import local models (self-contained, no external dependencies)
from models import Resnet3D, CustomClassifier, UnimodalModel
from dataset import get_mci_dataloaders

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ResNet3DMCIBinary(pl.LightningModule):
    """
    PyTorch Lightning module for ResNet3D binary MCI classification with MedicalNet pretrained weights
    """

    def __init__(
        self,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        freeze_encoder: bool = False,
        hidden_dim: int = 0,
        dropout_rate: float = 0.3,
        resnet_depth: int = 50
    ):
        super().__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # Create encoder (ResNet3D) with MedicalNet pretrained weights
        logger.info(f"Initializing ResNet3D-{resnet_depth} encoder with MedicalNet pretrained weights...")
        self.image_encoder = Resnet3D(
            depth=resnet_depth,
            freeze=freeze_encoder,
            include_head=False
        )

        # Create binary classifier
        logger.info("Initializing binary classifier...")
        self.classifier = CustomClassifier(
            hidden_dim=hidden_dim,
            activation_fun=nn.ReLU(),
            num_class=2,  # Binary: AD (0) vs CN (1)
            task="multiclass",
            dropout_rate=dropout_rate
        )

        # Create model
        encoders = {"image": self.image_encoder}
        self.model = UnimodalModel(encoders=encoders, classifier=self.classifier)

        # Loss function (with optional class weights)
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
        preds = torch.argmax(logits, dim=1)

        # Get probabilities for AUC
        probs = torch.softmax(logits, dim=1)[:, 1]  # Probability of CN (class 1)

        # Store for epoch-end metrics
        self.validation_step_outputs.append({
            'loss': loss,
            'preds': preds,
            'labels': labels,
            'probs': probs
        })

        return loss

    def on_validation_epoch_end(self):
        """Compute validation metrics at epoch end"""
        outputs = self.validation_step_outputs

        # Gather predictions and labels
        all_preds = torch.cat([x['preds'] for x in outputs]).cpu().numpy()
        all_labels = torch.cat([x['labels'] for x in outputs]).cpu().numpy()
        all_probs = torch.cat([x['probs'] for x in outputs]).cpu().numpy()

        # Calculate metrics
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='binary')
        precision = precision_score(all_labels, all_preds, average='binary', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='binary', zero_division=0)

        # AUC-ROC
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except:
            auc = 0.0

        # Balanced accuracy (important for imbalanced datasets)
        cm = confusion_matrix(all_labels, all_preds)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            balanced_acc = (sensitivity + specificity) / 2
        else:
            balanced_acc = accuracy

        # Log metrics
        self.log('val/loss', avg_loss, prog_bar=True)
        self.log('val/acc', accuracy, prog_bar=True)
        self.log('val/balanced_acc', balanced_acc, prog_bar=True)
        self.log('val/f1', f1)
        self.log('val/precision', precision)
        self.log('val/recall', recall)
        self.log('val/auc', auc)
        self.log('val/sensitivity', sensitivity if 'sensitivity' in locals() else 0)
        self.log('val/specificity', specificity if 'specificity' in locals() else 0)

        # Print confusion matrix
        logger.info(f"\nValidation Confusion Matrix:")
        logger.info(f"  TN={cm[0,0]}, FP={cm[0,1]}")
        logger.info(f"  FN={cm[1,0]}, TP={cm[1,1]}")
        logger.info(f"  Sensitivity (CN recall): {sensitivity if 'sensitivity' in locals() else 0:.4f}")
        logger.info(f"  Specificity (AD recall): {specificity if 'specificity' in locals() else 0:.4f}")

        # Clear outputs
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        """Test step"""
        logits = self(batch)
        labels = batch['label']

        loss = self.criterion(logits, labels)
        preds = torch.argmax(logits, dim=1)
        probs = torch.softmax(logits, dim=1)[:, 1]

        self.test_step_outputs.append({
            'loss': loss,
            'preds': preds,
            'labels': labels,
            'probs': probs
        })

        return loss

    def on_test_epoch_end(self):
        """Compute test metrics at epoch end"""
        outputs = self.test_step_outputs

        all_preds = torch.cat([x['preds'] for x in outputs]).cpu().numpy()
        all_labels = torch.cat([x['labels'] for x in outputs]).cpu().numpy()
        all_probs = torch.cat([x['probs'] for x in outputs]).cpu().numpy()

        # Calculate final metrics
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='binary')
        precision = precision_score(all_labels, all_preds, average='binary', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='binary', zero_division=0)

        try:
            auc = roc_auc_score(all_labels, all_probs)
        except:
            auc = 0.0

        # Confusion matrix and balanced accuracy
        cm = confusion_matrix(all_labels, all_preds)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            balanced_acc = (sensitivity + specificity) / 2
        else:
            balanced_acc = accuracy

        # Log final test metrics
        self.log('test/loss', avg_loss)
        self.log('test/acc', accuracy)
        self.log('test/balanced_acc', balanced_acc)
        self.log('test/f1', f1)
        self.log('test/auc', auc)

        # Print detailed results
        logger.info(f"\n{'='*80}")
        logger.info(f"FINAL TEST RESULTS")
        logger.info(f"{'='*80}")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  Balanced Accuracy: {balanced_acc:.4f}")
        logger.info(f"  F1 Score: {f1:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall: {recall:.4f}")
        logger.info(f"  AUC-ROC: {auc:.4f}")
        logger.info(f"\nConfusion Matrix:")
        logger.info(f"  TN={cm[0,0]}, FP={cm[0,1]}")
        logger.info(f"  FN={cm[1,0]}, TP={cm[1,1]}")
        logger.info(f"\n  Sensitivity (CN recall): {sensitivity:.4f}")
        logger.info(f"  Specificity (AD recall): {specificity:.4f}")
        logger.info(f"{'='*80}\n")

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
            patience=5
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
    # Load environment variables from .env file (if it exists)
    # Check both root directory and experiment directory
    root_env = Path(__file__).parent.parent.parent / '.env'
    local_env = Path(__file__).parent / '.env'

    if root_env.exists():
        load_dotenv(root_env)
        logger.info(f"Loaded environment variables from {root_env}")
    if local_env.exists():
        load_dotenv(local_env, override=True)  # Local .env overrides root .env
        logger.info(f"Loaded environment variables from {local_env} (override)")

    parser = argparse.ArgumentParser(
        description='Train DenseNet3D for CN vs AD binary classification',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Config file (if provided, overrides other arguments)
    parser.add_argument('--config', type=str, default=None,
                       help='Path to YAML config file (recommended)')

    # Data paths
    parser.add_argument('--train-csv', type=str,
                       help='Path to training CSV file')
    parser.add_argument('--val-csv', type=str,
                       help='Path to validation CSV file')
    parser.add_argument('--test-csv', type=str,
                       help='Path to test CSV file')

    # Output
    parser.add_argument('--output', type=str,
                       help='Output directory for checkpoints and logs')

    # Training hyperparameters
    parser.add_argument('--batch-size', type=int,
                       help='Batch size')
    parser.add_argument('--epochs', type=int,
                       help='Maximum number of epochs')
    parser.add_argument('--lr', type=float,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float,
                       help='Weight decay')
    parser.add_argument('--dropout', type=float,
                       help='Dropout rate')

    # Model architecture
    parser.add_argument('--target-shape', type=int, nargs=3,
                       help='Target shape for MRI volumes (D H W)')
    parser.add_argument('--hidden-dim', type=int,
                       help='Hidden dimension for classifier (0=no hidden layer)')
    parser.add_argument('--freeze-encoder', action='store_true',
                       help='Freeze encoder weights (transfer learning)')

    # Training options
    parser.add_argument('--num-workers', type=int, default=None,
                       help='Number of dataloader workers')
    parser.add_argument('--gpus', type=int, default=None,
                       help='Number of GPUs to use (0 for CPU)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--no-weighted-sampling', action='store_true',
                       help='Disable weighted sampling for class balance')

    args = parser.parse_args()

    # Load config file if provided
    if args.config:
        logger.info(f"Loading configuration from {args.config}")
        config = load_config(args.config)

        # Merge config with args (command line args override config)
        # Data paths
        args.train_csv = args.train_csv or config['data']['train_csv']
        args.val_csv = args.val_csv or config['data']['val_csv']
        args.test_csv = args.test_csv or config['data']['test_csv']
        args.output = args.output or config['data']['logs_dir']

        # Model
        args.target_shape = args.target_shape or config['model']['target_shape']
        args.hidden_dim = args.hidden_dim if args.hidden_dim is not None else config['model']['hidden_dim']
        args.dropout = args.dropout if args.dropout is not None else config['model']['dropout']
        args.freeze_encoder = args.freeze_encoder or config['model']['freeze_encoder']
        args.resnet_depth = getattr(args, 'resnet_depth', None) or config['model'].get('resnet_depth', 50)

        # Training
        args.batch_size = args.batch_size or config['training']['batch_size']
        args.epochs = args.epochs or config['training']['epochs']
        args.lr = args.lr or config['training']['learning_rate']
        args.weight_decay = args.weight_decay or config['training']['weight_decay']
        args.seed = args.seed or config['training']['seed']
        args.no_weighted_sampling = not config['training']['use_weighted_sampling']

        # Hardware
        args.gpus = args.gpus if args.gpus is not None else config['hardware']['gpus']
        args.num_workers = args.num_workers or config['hardware']['num_workers']

    # Validate required arguments
    if not args.train_csv or not args.val_csv or not args.test_csv:
        parser.error("--train-csv, --val-csv, and --test-csv are required (or provide --config)")

    # Set defaults if still None
    args.output = args.output or '../../outputs/mci_baseline/training'
    args.batch_size = args.batch_size or 8
    args.epochs = args.epochs or 50
    args.lr = args.lr or 1e-4
    args.weight_decay = args.weight_decay or 1e-5
    args.dropout = args.dropout or 0.3
    args.target_shape = args.target_shape or [96, 96, 96]
    args.hidden_dim = args.hidden_dim if args.hidden_dim is not None else 0
    args.num_workers = args.num_workers or 4

    # Set random seed
    pl.seed_everything(args.seed)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*80)
    print("🧠 RESNET3D TRAINING FOR CN VS AD CLASSIFICATION PREDICTION")
    print("="*80)
    print(f"Task: Binary Classification (CN vs AD)")
    print(f"Model: ResNet3D with MedicalNet pretrained weights")
    print(f"Target Shape: {tuple(args.target_shape)}")
    print(f"Config: {args.config if args.config else 'Command line arguments'}")
    print("="*80 + "\n")

    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_loader, val_loader, test_loader = get_mci_dataloaders(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        test_csv=args.test_csv,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        target_shape=tuple(args.target_shape),
        use_weighted_sampling=not args.no_weighted_sampling
    )

    # Create model
    logger.info("Creating model...")
    resnet_depth = getattr(args, 'resnet_depth', 50)  # Default to ResNet-50
    model = ResNet3DMCIBinary(
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        freeze_encoder=args.freeze_encoder,
        hidden_dim=args.hidden_dim,
        dropout_rate=args.dropout,
        resnet_depth=resnet_depth
    )

    # Setup callbacks
    # Checkpoints go to data/checkpoints/ (separate from logs)
    checkpoint_dir = Path('data/checkpoints')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='resnet3d_mci-{epoch:02d}-{val/balanced_acc:.4f}',
        monitor='val/balanced_acc',
        mode='max',
        save_top_k=3,
        save_last=True
    )

    early_stop_callback = EarlyStopping(
        monitor='val/loss',
        patience=15,
        mode='min',
        verbose=True
    )

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # Setup loggers
    loggers = []

    # TensorBoard logger (always enabled)
    tb_logger = TensorBoardLogger(
        save_dir=output_dir,
        name='tensorboard_logs'
    )
    loggers.append(tb_logger)

    # Weights & Biases logger (optional, based on config or env)
    # Priority: environment variables > config file
    wandb_enabled = config.get('wandb', {}).get('enabled', False)
    if wandb_enabled or os.getenv('WANDB_API_KEY'):
        wandb_project = os.getenv('WANDB_PROJECT') or config.get('wandb', {}).get('project', 'alzheimer-pmci-smci')
        wandb_entity = os.getenv('WANDB_ENTITY') or config.get('wandb', {}).get('entity', None)
        wandb_run_name = config.get('wandb', {}).get('run_name', None)

        wandb_logger = WandbLogger(
            project=wandb_project,
            name=wandb_run_name,
            entity=wandb_entity,
            save_dir=output_dir,
            log_model=config.get('wandb', {}).get('log_model', False),
            tags=config.get('wandb', {}).get('tags', ['resnet3d', 'mci-classification'])
        )
        loggers.append(wandb_logger)
        logger.info(f"Weights & Biases logging enabled: {wandb_project}")
        if wandb_entity:
            logger.info(f"  Entity: {wandb_entity}")

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator='gpu' if args.gpus > 0 else 'cpu',
        devices=args.gpus if args.gpus > 0 else 1,
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
        logger=loggers,
        log_every_n_steps=10,
        enable_progress_bar=True,
        deterministic='warn'  # 'warn' instead of True to allow non-deterministic ops on GPU
    )

    # Train
    logger.info("Starting training...")
    trainer.fit(model, train_loader, val_loader, ckpt_path=args.resume)

    # Test
    logger.info("Running final test...")
    trainer.test(model, test_loader)

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

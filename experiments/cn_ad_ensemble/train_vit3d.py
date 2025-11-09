#!/usr/bin/env python3
"""
Train 3D Vision Transformer for CN vs AD Binary Classification

Usage:
    python train_vit3d.py \
        --train-csv data/splits_nppy/train.csv \
        --val-csv data/splits_nppy/val.csv \
        --model-size small \
        --batch-size 2 \
        --epochs 100 \
        --lr 3e-4 \
        --output-dir checkpoints_vit3d
"""

import argparse
import logging
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import time

from dataset import ADNIDataset
from model_vit3d import vit3d_small, vit3d_base, vit3d_large

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance

    FL(p_t) = -α(1-p_t)^γ * log(p_t)

    Focuses training on hard examples by down-weighting easy examples.
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        """
        Args:
            inputs: (batch_size, num_classes) - logits
            targets: (batch_size,) - class labels
        """
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{running_loss / (batch_idx + 1):.4f}',
            'acc': f'{100. * correct / total:.2f}%'
        })

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device, epoch):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]")
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{running_loss / (batch_idx + 1):.4f}',
                'acc': f'{100. * correct / total:.2f}%'
            })

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total

    # Per-class accuracy
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    cn_mask = all_labels == 0
    ad_mask = all_labels == 1

    cn_acc = 100. * (all_preds[cn_mask] == all_labels[cn_mask]).sum() / cn_mask.sum() if cn_mask.sum() > 0 else 0
    ad_acc = 100. * (all_preds[ad_mask] == all_labels[ad_mask]).sum() / ad_mask.sum() if ad_mask.sum() > 0 else 0

    return epoch_loss, epoch_acc, cn_acc, ad_acc


def main():
    parser = argparse.ArgumentParser(description='Train 3D Vision Transformer for AD Classification')
    parser.add_argument('--train-csv', type=str, required=True,
                        help='Path to training CSV')
    parser.add_argument('--val-csv', type=str, required=True,
                        help='Path to validation CSV')
    parser.add_argument('--model-size', type=str, default='small',
                        choices=['small', 'base', 'large'],
                        help='Model size: small (6 layers), base (12 layers), large (24 layers)')
    parser.add_argument('--patch-size', type=int, default=16,
                        help='Patch size for ViT (default: 16)')
    parser.add_argument('--batch-size', type=int, default=2,
                        help='Batch size (default: 2 for 3D-ViT)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate (default: 3e-4)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='Weight decay (default: 0.05)')
    parser.add_argument('--warmup-epochs', type=int, default=10,
                        help='Number of warmup epochs')
    parser.add_argument('--use-focal-loss', action='store_true',
                        help='Use focal loss instead of cross-entropy')
    parser.add_argument('--focal-alpha', type=float, default=0.25,
                        help='Focal loss alpha parameter')
    parser.add_argument('--focal-gamma', type=float, default=2.0,
                        help='Focal loss gamma parameter')
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience')
    parser.add_argument('--output-dir', type=str, default='checkpoints_vit3d',
                        help='Output directory for checkpoints')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of dataloader workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Create model
    logger.info(f"Creating 3D-ViT-{args.model_size.capitalize()}/16 model...")
    if args.model_size == 'small':
        model = vit3d_small(num_classes=2, img_size=192, patch_size=args.patch_size)
    elif args.model_size == 'base':
        model = vit3d_base(num_classes=2, img_size=192, patch_size=args.patch_size)
    else:
        model = vit3d_large(num_classes=2, img_size=192, patch_size=args.patch_size)

    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # Create datasets
    logger.info("Loading datasets...")
    train_dataset = ADNIDataset(args.train_csv, augment=True)
    val_dataset = ADNIDataset(args.val_csv, augment=False)

    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Loss function
    if args.use_focal_loss:
        logger.info(f"Using Focal Loss (alpha={args.focal_alpha}, gamma={args.focal_gamma})")
        criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
    else:
        logger.info("Using Cross-Entropy Loss")
        criterion = nn.CrossEntropyLoss()

    # Optimizer (AdamW recommended for ViT)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )

    # Learning rate scheduler with warmup
    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            # Linear warmup
            return (epoch + 1) / args.warmup_epochs
        else:
            # Cosine annealing after warmup
            progress = (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)
            return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Training loop
    best_val_acc = 0.0
    patience_counter = 0

    logger.info("\n" + "="*80)
    logger.info("Starting training...")
    logger.info("="*80)

    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # Validate
        val_loss, val_acc, cn_acc, ad_acc = validate(
            model, val_loader, criterion, device, epoch
        )

        # Learning rate step
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        epoch_time = time.time() - epoch_start_time

        # Log results
        logger.info(f"\nEpoch {epoch}/{args.epochs} ({epoch_time:.1f}s)")
        logger.info(f"  LR: {current_lr:.6f}")
        logger.info(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        logger.info(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
        logger.info(f"          CN Acc: {cn_acc:.2f}%, AD Acc: {ad_acc:.2f}%")

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0

            checkpoint_path = output_dir / 'best_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'cn_acc': cn_acc,
                'ad_acc': ad_acc,
            }, checkpoint_path)
            logger.info(f"  ✓ New best model saved! (Val Acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1
            logger.info(f"  Patience: {patience_counter}/{args.patience}")

        # Early stopping
        if patience_counter >= args.patience:
            logger.info(f"\n✗ Early stopping triggered after {epoch} epochs")
            break

    # Save final model
    final_checkpoint_path = output_dir / 'final_model.pth'
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
        'val_loss': val_loss,
    }, final_checkpoint_path)

    logger.info("\n" + "="*80)
    logger.info("Training completed!")
    logger.info(f"Best validation accuracy: {best_val_acc:.2f}%")
    logger.info(f"Model saved to: {output_dir}/")
    logger.info("="*80)

    # Save training history
    history_path = output_dir / 'training_history.npz'
    np.savez(
        history_path,
        train_losses=train_losses,
        train_accs=train_accs,
        val_losses=val_losses,
        val_accs=val_accs
    )
    logger.info(f"Training history saved to: {history_path}")


if __name__ == '__main__':
    main()

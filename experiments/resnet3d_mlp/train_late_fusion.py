#!/usr/bin/env python3
"""
True Late Fusion: ResNet3D (MRI) + MLP (Tabular) — separate predictions, combined.

Pipeline:
  1. Fine-tune ResNet3D + Linear head on MRI → proba_mri
  2. Train MLP on tabular features only → proba_tab
  3. Late fusion: combine probabilities (average, weighted, learned stacking)

Usage:
    python train_late_fusion.py --config config.yaml
"""

import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict
import logging
import argparse
import json
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
import importlib.util

# Import ResNet3DBackbone
_resnet_model_path = Path(__file__).parent / "model.py"
_spec_resnet = importlib.util.spec_from_file_location("resnet3d_mlp_model", _resnet_model_path)
_resnet_module = importlib.util.module_from_spec(_spec_resnet)
_spec_resnet.loader.exec_module(_resnet_module)
ResNet3DBackbone = _resnet_module.ResNet3DBackbone

# Import MultiModalDataset
_mm_dataset_path = Path(__file__).parent.parent / "multimodal_fusion" / "dataset.py"
_spec_mm = importlib.util.spec_from_file_location("multimodal_fusion_dataset", _mm_dataset_path)
_mm_module = importlib.util.module_from_spec(_spec_mm)
_spec_mm.loader.exec_module(_mm_module)
MultiModalDataset = _mm_module.MultiModalDataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_device(config: Dict) -> torch.device:
    device_str = config['hardware']['device']
    if device_str == 'cuda' and torch.cuda.is_available():
        return torch.device('cuda')
    elif device_str == 'mps' and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


class ResNet3DClassifier(nn.Module):
    """ResNet3D + linear head for fine-tuning."""
    def __init__(self, pretrained: bool = True, num_classes: int = 2):
        super().__init__()
        self.backbone = ResNet3DBackbone(pretrained=pretrained)
        self.head = nn.Linear(2048, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)


class TabularMLP(nn.Module):
    """MLP for tabular features only."""
    def __init__(self, input_dim: int = 16, hidden_dims=None, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]

        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def compute_metrics(y_true, y_pred, y_proba) -> Dict:
    acc = accuracy_score(y_true, y_pred) * 100
    bal_acc = balanced_accuracy_score(y_true, y_pred) * 100
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    sensitivity = 100. * tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = 100. * tn / (tn + fp) if (tn + fp) > 0 else 0.0
    try:
        auc = roc_auc_score(y_true, y_proba)
    except:
        auc = 0.5
    return {
        'accuracy': float(acc), 'balanced_accuracy': float(bal_acc),
        'sensitivity': float(sensitivity), 'specificity': float(specificity),
        'auc': float(auc), 'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
    }


def log_metrics(name, metrics):
    logger.info(f"\n{name}:")
    logger.info(f"  Accuracy:     {metrics['accuracy']:.1f}%")
    logger.info(f"  Balanced Acc: {metrics['balanced_accuracy']:.1f}%")
    logger.info(f"  Sensitivity:  {metrics['sensitivity']:.1f}%")
    logger.info(f"  Specificity:  {metrics['specificity']:.1f}%")
    logger.info(f"  AUC:          {metrics['auc']:.3f}")
    logger.info(f"  Confusion:    {metrics['confusion_matrix']}")


def main():
    parser = argparse.ArgumentParser(description='True Late Fusion: ResNet3D + MLP')
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--output-dir', type=str, default='results_late_fusion')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--finetune-epochs', type=int, default=30)
    parser.add_argument('--freeze-epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=0.00005)
    parser.add_argument('--backbone-lr-factor', type=float, default=0.1)
    parser.add_argument('--mlp-epochs', type=int, default=200)
    parser.add_argument('--mlp-lr', type=float, default=0.001)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = setup_device(config)
    preproc = config.get('preprocessing', {})
    target_shape = tuple(preproc.get('target_shape', [128, 128, 128]))
    tabular_features = config['data']['tabular_features']
    batch_size = config['hardware'].get('batch_size', config['training'].get('batch_size', 4))
    num_workers = config['hardware']['num_workers']

    # ── Datasets ──
    logger.info("Loading datasets...")
    train_dataset = MultiModalDataset(
        config['data']['train_csv'], tabular_features=tabular_features,
        target_shape=target_shape, augment=True, normalize_tabular=True,
        scaler=None, use_paper_preprocessing=preproc.get('use_paper_preprocessing', True),
        target_spacing=preproc.get('target_spacing', 1.75)
    )
    tab_scaler = train_dataset.get_scaler()

    val_dataset = MultiModalDataset(
        config['data']['val_csv'], tabular_features=tabular_features,
        target_shape=target_shape, augment=False, normalize_tabular=True,
        scaler=tab_scaler, use_paper_preprocessing=preproc.get('use_paper_preprocessing', True),
        target_spacing=preproc.get('target_spacing', 1.75)
    )
    test_dataset = MultiModalDataset(
        config['data']['test_csv'], tabular_features=tabular_features,
        target_shape=target_shape, augment=False, normalize_tabular=True,
        scaler=tab_scaler, use_paper_preprocessing=preproc.get('use_paper_preprocessing', True),
        target_spacing=preproc.get('target_spacing', 1.75)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    # Class weights
    train_labels = pd.read_csv(config['data']['train_csv'])['label'].tolist()
    class_counts = np.bincount(train_labels)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * len(class_counts)
    class_weights_t = torch.FloatTensor(class_weights).to(device)

    # ═══════════════════════════════════════════════════════
    # BRANCH 1: ResNet3D on MRI
    # ═══════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("BRANCH 1: ResNet3D on MRI")
    logger.info("=" * 60)

    model = ResNet3DClassifier(pretrained=True, num_classes=2).to(device)
    criterion_mri = nn.CrossEntropyLoss(weight=class_weights_t)
    scaler_amp = torch.amp.GradScaler('cuda', enabled=True)

    backbone_params = list(model.backbone.parameters())
    backbone_ids = set(id(p) for p in backbone_params)
    head_params = [p for p in model.parameters() if id(p) not in backbone_ids]

    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': args.lr * args.backbone_lr_factor},
        {'params': head_params, 'lr': args.lr},
    ], weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.finetune_epochs, eta_min=1e-7)

    if args.freeze_epochs > 0:
        for p in model.backbone.parameters():
            p.requires_grad = False
        logger.info(f"Backbone frozen for first {args.freeze_epochs} epochs")

    best_val_bal_acc = 0.0
    best_state = None
    epochs_no_improve = 0

    for epoch in range(args.finetune_epochs):
        if epoch == args.freeze_epochs and args.freeze_epochs > 0:
            for p in model.backbone.parameters():
                p.requires_grad = True
            logger.info(f"Backbone unfrozen at epoch {epoch+1}")

        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for mri, tabular, labels in train_loader:
            mri, labels = mri.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                outputs = model(mri)
                loss = criterion_mri(outputs, labels)
            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler_amp.step(optimizer)
            scaler_amp.update()

            total_loss += loss.item()
            _, pred = outputs.max(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()

        train_acc = 100. * correct / total
        scheduler.step()

        # Validate
        model.eval()
        val_preds, val_labels_list, val_probs = [], [], []
        with torch.no_grad():
            for mri, tabular, labels in val_loader:
                mri, labels = mri.to(device), labels.to(device)
                with torch.amp.autocast('cuda'):
                    outputs = model(mri)
                probs = torch.softmax(outputs.float(), dim=1)
                _, pred = outputs.max(1)
                val_preds.extend(pred.cpu().numpy())
                val_labels_list.extend(labels.cpu().numpy())
                val_probs.extend(probs[:, 1].cpu().numpy())

        val_bal_acc = balanced_accuracy_score(val_labels_list, val_preds) * 100
        val_acc = accuracy_score(val_labels_list, val_preds) * 100
        try:
            val_auc = roc_auc_score(val_labels_list, val_probs)
        except:
            val_auc = 0.5

        frozen_str = " [frozen]" if epoch < args.freeze_epochs else ""
        logger.info(
            f"Epoch {epoch+1}/{args.finetune_epochs}: "
            f"train_acc={train_acc:.1f}%  val_acc={val_acc:.1f}% "
            f"val_bal_acc={val_bal_acc:.1f}% val_auc={val_auc:.3f}{frozen_str}"
        )

        if val_bal_acc > best_val_bal_acc:
            best_val_bal_acc = val_bal_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= 15 and epoch >= 10:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    logger.info(f"Best val balanced accuracy: {best_val_bal_acc:.1f}%")

    # Get MRI probabilities
    def get_mri_probas(loader):
        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for mri, tabular, labels in loader:
                mri = mri.to(device)
                with torch.amp.autocast('cuda'):
                    outputs = model(mri)
                probs = torch.softmax(outputs.float(), dim=1)
                all_probs.extend(probs[:, 1].cpu().numpy())
                all_labels.extend(labels.numpy())
        return np.array(all_probs), np.array(all_labels)

    train_dataset_noaug = MultiModalDataset(
        config['data']['train_csv'], tabular_features=tabular_features,
        target_shape=target_shape, augment=False, normalize_tabular=True,
        scaler=tab_scaler, use_paper_preprocessing=preproc.get('use_paper_preprocessing', True),
        target_spacing=preproc.get('target_spacing', 1.75)
    )
    train_loader_noaug = DataLoader(train_dataset_noaug, batch_size=batch_size, shuffle=False,
                                     num_workers=num_workers, pin_memory=True)

    logger.info("Getting MRI probabilities...")
    proba_mri_train, y_train = get_mri_probas(train_loader_noaug)
    proba_mri_val, y_val = get_mri_probas(val_loader)
    proba_mri_test, y_test = get_mri_probas(test_loader)

    mri_pred = (proba_mri_test >= 0.5).astype(int)
    mri_metrics = compute_metrics(y_test, mri_pred, proba_mri_test)
    log_metrics("BRANCH 1 — MRI only (TEST)", mri_metrics)

    # Free GPU for branch 2
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ═══════════════════════════════════════════════════════
    # BRANCH 2: MLP on tabular only
    # ═══════════════════════════════════════════════════════
    logger.info("\n" + "=" * 60)
    logger.info("BRANCH 2: MLP on Tabular features only")
    logger.info("=" * 60)

    # Extract tabular features
    def get_tabular(loader):
        all_tab, all_labels = [], []
        for mri, tabular, labels in loader:
            all_tab.append(tabular.numpy())
            all_labels.append(labels.numpy())
        return np.concatenate(all_tab), np.concatenate(all_labels)

    X_tab_train, _ = get_tabular(train_loader_noaug)
    X_tab_val, _ = get_tabular(val_loader)
    X_tab_test, _ = get_tabular(test_loader)

    # Tabular dataloaders
    tab_train_ds = TensorDataset(torch.FloatTensor(X_tab_train), torch.LongTensor(y_train))
    tab_val_ds = TensorDataset(torch.FloatTensor(X_tab_val), torch.LongTensor(y_val))
    tab_test_ds = TensorDataset(torch.FloatTensor(X_tab_test), torch.LongTensor(y_test))

    tab_train_loader = DataLoader(tab_train_ds, batch_size=64, shuffle=True)
    tab_val_loader = DataLoader(tab_val_ds, batch_size=64, shuffle=False)
    tab_test_loader = DataLoader(tab_test_ds, batch_size=64, shuffle=False)

    mlp = TabularMLP(input_dim=len(tabular_features), hidden_dims=[128, 64, 32],
                     num_classes=2, dropout=0.3).to(device)
    criterion_tab = nn.CrossEntropyLoss(weight=class_weights_t)
    mlp_optimizer = optim.AdamW(mlp.parameters(), lr=args.mlp_lr, weight_decay=0.01)
    mlp_scheduler = optim.lr_scheduler.CosineAnnealingLR(mlp_optimizer, T_max=args.mlp_epochs, eta_min=1e-6)

    logger.info(f"TabularMLP params: {sum(p.numel() for p in mlp.parameters()):,}")

    best_mlp_bal_acc = 0.0
    best_mlp_state = None
    mlp_no_improve = 0

    for epoch in range(args.mlp_epochs):
        mlp.train()
        correct, total = 0, 0
        for tab, labels in tab_train_loader:
            tab, labels = tab.to(device), labels.to(device)
            mlp_optimizer.zero_grad()
            outputs = mlp(tab)
            loss = criterion_tab(outputs, labels)
            loss.backward()
            mlp_optimizer.step()
            _, pred = outputs.max(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()

        train_acc = 100. * correct / total
        mlp_scheduler.step()

        # Validate
        mlp.eval()
        val_preds, val_labels_list, val_probs = [], [], []
        with torch.no_grad():
            for tab, labels in tab_val_loader:
                tab, labels = tab.to(device), labels.to(device)
                outputs = mlp(tab)
                probs = torch.softmax(outputs, dim=1)
                _, pred = outputs.max(1)
                val_preds.extend(pred.cpu().numpy())
                val_labels_list.extend(labels.cpu().numpy())
                val_probs.extend(probs[:, 1].cpu().numpy())

        val_bal_acc = balanced_accuracy_score(val_labels_list, val_preds) * 100
        val_acc = accuracy_score(val_labels_list, val_preds) * 100
        try:
            val_auc = roc_auc_score(val_labels_list, val_probs)
        except:
            val_auc = 0.5

        if epoch % 20 == 0 or epoch == args.mlp_epochs - 1:
            logger.info(
                f"MLP Epoch {epoch+1}/{args.mlp_epochs}: "
                f"train_acc={train_acc:.1f}%  val_acc={val_acc:.1f}% "
                f"val_bal_acc={val_bal_acc:.1f}% val_auc={val_auc:.3f}"
            )

        if val_bal_acc > best_mlp_bal_acc:
            best_mlp_bal_acc = val_bal_acc
            best_mlp_state = {k: v.cpu().clone() for k, v in mlp.state_dict().items()}
            mlp_no_improve = 0
        else:
            mlp_no_improve += 1

        if mlp_no_improve >= 30 and epoch >= 50:
            logger.info(f"MLP early stopping at epoch {epoch+1}")
            break

    if best_mlp_state is not None:
        mlp.load_state_dict({k: v.to(device) for k, v in best_mlp_state.items()})
    logger.info(f"Best MLP val balanced accuracy: {best_mlp_bal_acc:.1f}%")

    # Get tabular probabilities
    def get_mlp_probas(loader):
        mlp.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for tab, labels in loader:
                tab = tab.to(device)
                outputs = mlp(tab)
                probs = torch.softmax(outputs, dim=1)
                all_probs.extend(probs[:, 1].cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        return np.array(all_probs), np.array(all_labels)

    proba_tab_train, _ = get_mlp_probas(tab_train_loader)
    proba_tab_val, _ = get_mlp_probas(tab_val_loader)
    proba_tab_test, _ = get_mlp_probas(tab_test_loader)

    tab_pred = (proba_tab_test >= 0.5).astype(int)
    tab_metrics = compute_metrics(y_test, tab_pred, proba_tab_test)
    log_metrics("BRANCH 2 — Tabular MLP only (TEST)", tab_metrics)

    # ═══════════════════════════════════════════════════════
    # LATE FUSION: Combine probabilities
    # ═══════════════════════════════════════════════════════
    logger.info("\n" + "=" * 60)
    logger.info("LATE FUSION: Combining predictions")
    logger.info("=" * 60)

    all_fusion_results = {}

    # Method 1: Simple average
    proba_avg = (proba_mri_test + proba_tab_test) / 2
    pred_avg = (proba_avg >= 0.5).astype(int)
    avg_metrics = compute_metrics(y_test, pred_avg, proba_avg)
    log_metrics("LATE FUSION — Simple Average (TEST)", avg_metrics)
    all_fusion_results['simple_average'] = avg_metrics

    # Method 2: Weighted average (sweep on val)
    best_w, best_val_auc = 0.5, 0.0
    for w in np.arange(0.0, 1.05, 0.05):
        proba_w = w * proba_mri_val + (1 - w) * proba_tab_val
        try:
            auc_w = roc_auc_score(y_val, proba_w)
        except:
            auc_w = 0.5
        if auc_w > best_val_auc:
            best_val_auc = auc_w
            best_w = w

    logger.info(f"\nBest weight (MRI): {best_w:.2f} (val AUC: {best_val_auc:.3f})")
    proba_weighted = best_w * proba_mri_test + (1 - best_w) * proba_tab_test
    pred_weighted = (proba_weighted >= 0.5).astype(int)
    weighted_metrics = compute_metrics(y_test, pred_weighted, proba_weighted)
    log_metrics(f"LATE FUSION — Weighted Average w_mri={best_w:.2f} (TEST)", weighted_metrics)
    all_fusion_results['weighted_average'] = {**weighted_metrics, 'weight_mri': float(best_w)}

    # Method 3: Learned stacking
    X_stack_train = np.column_stack([proba_mri_train, proba_tab_train])
    X_stack_val = np.column_stack([proba_mri_val, proba_tab_val])
    X_stack_test = np.column_stack([proba_mri_test, proba_tab_test])

    best_C, best_stack_val_auc = 1.0, 0.0
    for C in [0.01, 0.1, 1.0, 10.0]:
        lr = LogisticRegression(C=C, random_state=args.seed, max_iter=1000)
        lr.fit(X_stack_train, y_train)
        proba_lr_val = lr.predict_proba(X_stack_val)[:, 1]
        try:
            auc_lr = roc_auc_score(y_val, proba_lr_val)
        except:
            auc_lr = 0.5
        if auc_lr > best_stack_val_auc:
            best_stack_val_auc = auc_lr
            best_C = C

    stacker = LogisticRegression(C=best_C, random_state=args.seed, max_iter=1000)
    stacker.fit(X_stack_train, y_train)
    proba_stack = stacker.predict_proba(X_stack_test)[:, 1]
    pred_stack = (proba_stack >= 0.5).astype(int)
    stack_metrics = compute_metrics(y_test, pred_stack, proba_stack)
    logger.info(f"\nStacking weights: MRI={stacker.coef_[0][0]:.3f}, Tab={stacker.coef_[0][1]:.3f}, bias={stacker.intercept_[0]:.3f}")
    log_metrics(f"LATE FUSION — Learned Stacking C={best_C} (TEST)", stack_metrics)
    all_fusion_results['learned_stacking'] = {
        **stack_metrics, 'C': best_C,
        'weight_mri': float(stacker.coef_[0][0]),
        'weight_tab': float(stacker.coef_[0][1]),
    }

    # ═══════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"{'Method':<35} {'Acc':>6} {'BalAcc':>7} {'Sens':>6} {'Spec':>6} {'AUC':>6}")
    logger.info("-" * 70)
    for name, m in [
        ('MRI only (ResNet3D)', mri_metrics),
        ('Tabular only (MLP)', tab_metrics),
        ('Late Fusion: Average', avg_metrics),
        ('Late Fusion: Weighted', weighted_metrics),
        ('Late Fusion: Stacking', stack_metrics),
    ]:
        logger.info(f"{name:<35} {m['accuracy']:5.1f}% {m['balanced_accuracy']:5.1f}% "
                     f"{m['sensitivity']:5.1f}% {m['specificity']:5.1f}% {m['auc']:5.3f}")

    # Save models
    results = {
        'mri_only': mri_metrics,
        'tabular_only': tab_metrics,
        'late_fusion': all_fusion_results,
        'config': {
            'finetune_epochs': args.finetune_epochs,
            'freeze_epochs': args.freeze_epochs,
            'lr': args.lr,
            'mlp_epochs': args.mlp_epochs,
            'mlp_lr': args.mlp_lr,
            'seed': args.seed,
        }
    }
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    torch.save(best_mlp_state, output_dir / 'tabular_mlp.pth')
    if best_state is not None:
        torch.save(best_state, output_dir / 'resnet3d_finetuned.pth')

    # Save predictions for analysis (DeLong, confusion matrices)
    np.save(output_dir / 'y_true_test.npy', y_test)
    np.save(output_dir / 'y_proba_mri_test.npy', proba_mri_test)
    np.save(output_dir / 'y_proba_tab_test.npy', proba_tab_test)
    np.save(output_dir / 'y_proba_avg_test.npy', proba_avg)
    np.save(output_dir / 'y_proba_weighted_test.npy', proba_weighted)
    np.save(output_dir / 'y_proba_stacking_test.npy', proba_stack)
    # Val predictions
    np.save(output_dir / 'y_true_val.npy', y_val)
    np.save(output_dir / 'y_proba_mri_val.npy', proba_mri_val)
    np.save(output_dir / 'y_proba_tab_val.npy', proba_tab_val)

    logger.info(f"\nResults saved to {output_dir}")


if __name__ == '__main__':
    main()

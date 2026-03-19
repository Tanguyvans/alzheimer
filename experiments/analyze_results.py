#!/usr/bin/env python3
"""
Post-training analysis: DeLong test, confusion matrices, XAI (GradCAM + feature importance).

Run AFTER all training scripts have completed (run_all.sh).

Usage:
    python analyze_results.py
"""

import numpy as np
import json
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from sklearn.metrics import confusion_matrix, roc_curve, auc
import importlib.util
import yaml
import xgboost as xgb
from torch.utils.data import DataLoader

# ── Paths ──
BASE = Path(__file__).parent
MLP_DIR = BASE / "resnet3d_mlp"
XGB_DIR = BASE / "resnet3d_xgboost"
REPORT_DIR = BASE / "report"
REPORT_DIR.mkdir(exist_ok=True)

# Import models
_resnet_model_path = MLP_DIR / "model.py"
_spec_resnet = importlib.util.spec_from_file_location("resnet3d_mlp_model", _resnet_model_path)
_resnet_module = importlib.util.module_from_spec(_spec_resnet)
_spec_resnet.loader.exec_module(_resnet_module)
ResNet3DBackbone = _resnet_module.ResNet3DBackbone
EarlyFusionModel = _resnet_module.EarlyFusionModel

# Import dataset
_mm_dataset_path = BASE / "multimodal_fusion" / "dataset.py"
_spec_mm = importlib.util.spec_from_file_location("multimodal_fusion_dataset", _mm_dataset_path)
_mm_module = importlib.util.module_from_spec(_spec_mm)
_spec_mm.loader.exec_module(_mm_module)
MultiModalDataset = _mm_module.MultiModalDataset


# ═══════════════════════════════════════════════════════
# 1. DeLong Test
# ═══════════════════════════════════════════════════════

def compute_midrank(x):
    """Compute midranks for DeLong test."""
    j = np.argsort(x)
    z = x[j]
    n = len(x)
    rank = np.zeros(n)
    i = 0
    while i < n:
        a = i
        while a < n - 1 and z[a + 1] == z[a]:
            a += 1
        for k in range(i, a + 1):
            rank[j[k]] = 0.5 * (i + a) + 1
        i = a + 1
    return rank


def fastDeLong(predictions_sorted_transposed, label_1_count):
    """Fast DeLong AUC computation."""
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    aucs = np.zeros(k)
    score = np.zeros((k, m))

    for j in range(k):
        r = compute_midrank(predictions_sorted_transposed[j, :])
        aucs[j] = (np.sum(r[:m]) - m * (m + 1) / 2.0) / (m * n)
        score[j] = r[:m] - np.arange(1, m + 1)

    # Structural components
    S10 = np.cov(score) if k > 1 else np.atleast_2d(np.var(score, axis=1))
    # For negative examples
    score_n = np.zeros((k, n))
    for j in range(k):
        r_n = compute_midrank(predictions_sorted_transposed[j, :])
        score_n[j] = r_n[m:] - np.arange(1, n + 1)
    S01 = np.cov(score_n) if k > 1 else np.atleast_2d(np.var(score_n, axis=1))

    S = S10 / m + S01 / n
    return aucs, S


def delong_test(y_true, y_score1, y_score2):
    """
    Two-sided DeLong test for two correlated ROC AUCs.
    Returns: auc1, auc2, z_stat, p_value
    """
    order = (-y_true).argsort()  # positives first
    label_1_count = int(y_true.sum())

    predictions = np.vstack([y_score1, y_score2])
    predictions_sorted = predictions[:, order]

    aucs, S = fastDeLong(predictions_sorted, label_1_count)

    # Difference and variance
    diff = aucs[0] - aucs[1]
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]

    if var <= 0:
        return aucs[0], aucs[1], 0.0, 1.0

    z = diff / np.sqrt(var)
    p = 2 * stats.norm.sf(abs(z))
    return aucs[0], aucs[1], z, p


def run_delong_analysis():
    """Load all predictions and run pairwise DeLong tests."""
    print("=" * 60)
    print("DeLong Test — Pairwise AUC comparison")
    print("=" * 60)

    # Load predictions
    preds = {}

    # MLP early fusion
    p = MLP_DIR / "results_early"
    if (p / "y_true_test.npy").exists():
        y_true = np.load(p / "y_true_test.npy")
        preds["MLP Early"] = np.load(p / "y_proba_test.npy")

    # MLP late fusion
    p = MLP_DIR / "results_late_fusion"
    if (p / "y_true_test.npy").exists():
        y_true = np.load(p / "y_true_test.npy")
        preds["MRI only (MLP)"] = np.load(p / "y_proba_mri_test.npy")
        preds["Tab only (MLP)"] = np.load(p / "y_proba_tab_test.npy")
        preds["MLP Late Avg"] = np.load(p / "y_proba_avg_test.npy")
        preds["MLP Late Wt"] = np.load(p / "y_proba_weighted_test.npy")
        preds["MLP Late Stack"] = np.load(p / "y_proba_stacking_test.npy")

    # XGB early fusion
    p = XGB_DIR / "results_finetuned"
    if (p / "y_true_test.npy").exists():
        y_true = np.load(p / "y_true_test.npy")
        preds["XGB Early"] = np.load(p / "y_proba_test.npy")

    # XGB late fusion
    p = XGB_DIR / "results_late_fusion"
    if (p / "y_true_test.npy").exists():
        y_true = np.load(p / "y_true_test.npy")
        preds["MRI only (XGB)"] = np.load(p / "y_proba_mri_test.npy")
        preds["Tab only (XGB)"] = np.load(p / "y_proba_tab_test.npy")
        preds["XGB Late Avg"] = np.load(p / "y_proba_avg_test.npy")
        preds["XGB Late Wt"] = np.load(p / "y_proba_weighted_test.npy")
        preds["XGB Late Stack"] = np.load(p / "y_proba_stacking_test.npy")

    if len(preds) < 2:
        print("Not enough predictions found. Run training first.")
        return

    names = list(preds.keys())
    n = len(names)
    p_matrix = np.ones((n, n))
    auc_values = {}

    for i in range(n):
        for j in range(i + 1, n):
            auc1, auc2, z, p_val = delong_test(y_true, preds[names[i]], preds[names[j]])
            p_matrix[i, j] = p_val
            p_matrix[j, i] = p_val
            auc_values[names[i]] = auc1
            auc_values[names[j]] = auc2

    # Print results
    print(f"\n{'Method':<20} {'AUC':>6}")
    print("-" * 28)
    for name in names:
        if name in auc_values:
            print(f"{name:<20} {auc_values[name]:.4f}")

    # Save p-value heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(p_matrix, dtype=bool))

    # Create annotation matrix with significance markers
    annot = np.full_like(p_matrix, "", dtype=object)
    for i in range(n):
        for j in range(n):
            if i == j:
                annot[i, j] = "—"
            else:
                p = p_matrix[i, j]
                if p < 0.001:
                    annot[i, j] = f"{p:.1e}\n***"
                elif p < 0.01:
                    annot[i, j] = f"{p:.3f}\n**"
                elif p < 0.05:
                    annot[i, j] = f"{p:.3f}\n*"
                else:
                    annot[i, j] = f"{p:.3f}"

    sns.heatmap(p_matrix, xticklabels=names, yticklabels=names,
                annot=annot, fmt='', cmap='RdYlGn', vmin=0, vmax=0.1,
                mask=mask, square=True, ax=ax,
                cbar_kws={'label': 'p-value'})
    ax.set_title('DeLong Test — Pairwise p-values\n(* p<0.05, ** p<0.01, *** p<0.001)', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    fig.savefig(REPORT_DIR / "delong_test.png", dpi=200)
    plt.close()
    print(f"\nSaved: {REPORT_DIR / 'delong_test.png'}")

    # Save CSV
    import pandas as pd
    df = pd.DataFrame(p_matrix, index=names, columns=names)
    df.to_csv(REPORT_DIR / "delong_pvalues.csv")
    print(f"Saved: {REPORT_DIR / 'delong_pvalues.csv'}")

    return y_true, preds


# ═══════════════════════════════════════════════════════
# 2. Confusion Matrices
# ═══════════════════════════════════════════════════════

def plot_confusion_matrices(y_true, preds):
    """Plot confusion matrices for all methods."""
    print("\n" + "=" * 60)
    print("Confusion Matrices")
    print("=" * 60)

    # Select key methods (not all, to keep readable)
    methods = [
        ("MRI only (MLP)", "MRI only\n(MLP branch)"),
        ("Tab only (MLP)", "Tabular only\n(MLP)"),
        ("MLP Early", "Early Fusion\n(ResNet3D+MLP)"),
        ("MLP Late Wt", "Late Fusion Weighted\n(ResNet3D+MLP)"),
        ("MRI only (XGB)", "MRI only\n(XGB branch)"),
        ("Tab only (XGB)", "Tabular only\n(XGBoost)"),
        ("XGB Early", "Early Fusion\n(ResNet3D+XGB)"),
        ("XGB Late Wt", "Late Fusion Weighted\n(ResNet3D+XGB)"),
    ]

    available = [(key, label) for key, label in methods if key in preds]
    n_methods = len(available)

    if n_methods == 0:
        print("No predictions available.")
        return

    ncols = 4
    nrows = (n_methods + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    if nrows == 1:
        axes = axes.reshape(1, -1)

    for idx, (key, label) in enumerate(available):
        row, col = idx // ncols, idx % ncols
        ax = axes[row, col]

        y_pred = (preds[key] >= 0.5).astype(int)
        cm = confusion_matrix(y_true, y_pred)

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['CN', 'AD'], yticklabels=['CN', 'AD'],
                    cbar=False, annot_kws={'size': 14})
        ax.set_xlabel('Predicted', fontsize=10)
        ax.set_ylabel('True', fontsize=10)
        ax.set_title(label, fontsize=10, fontweight='bold')

    # Hide empty axes
    for idx in range(n_methods, nrows * ncols):
        row, col = idx // ncols, idx % ncols
        axes[row, col].set_visible(False)

    plt.suptitle('Confusion Matrices — Test Set', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(REPORT_DIR / "confusion_matrices.png", dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {REPORT_DIR / 'confusion_matrices.png'}")


# ═══════════════════════════════════════════════════════
# 3. ROC Curves
# ═══════════════════════════════════════════════════════

def plot_roc_curves(y_true, preds):
    """Plot ROC curves for all methods."""
    print("\n" + "=" * 60)
    print("ROC Curves")
    print("=" * 60)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # ResNet3D + MLP methods
    mlp_methods = [
        ("MRI only (MLP)", "MRI only", "--"),
        ("Tab only (MLP)", "Tabular only (MLP)", "--"),
        ("MLP Early", "Early Fusion", "-"),
        ("MLP Late Avg", "Late Avg", "-."),
        ("MLP Late Wt", "Late Weighted", "-"),
        ("MLP Late Stack", "Late Stacking", "-."),
    ]

    for key, label, ls in mlp_methods:
        if key in preds:
            fpr, tpr, _ = roc_curve(y_true, preds[key])
            auc_val = auc(fpr, tpr)
            ax1.plot(fpr, tpr, ls, label=f"{label} (AUC={auc_val:.3f})", linewidth=2)

    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax1.set_xlabel('False Positive Rate', fontsize=12)
    ax1.set_ylabel('True Positive Rate', fontsize=12)
    ax1.set_title('ResNet3D + MLP', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=9)
    ax1.grid(alpha=0.3)

    # ResNet3D + XGBoost methods
    xgb_methods = [
        ("MRI only (XGB)", "MRI only", "--"),
        ("Tab only (XGB)", "Tabular only (XGBoost)", "--"),
        ("XGB Early", "Early Fusion", "-"),
        ("XGB Late Avg", "Late Avg", "-."),
        ("XGB Late Wt", "Late Weighted", "-"),
        ("XGB Late Stack", "Late Stacking", "-."),
    ]

    for key, label, ls in xgb_methods:
        if key in preds:
            fpr, tpr, _ = roc_curve(y_true, preds[key])
            auc_val = auc(fpr, tpr)
            ax2.plot(fpr, tpr, ls, label=f"{label} (AUC={auc_val:.3f})", linewidth=2)

    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax2.set_xlabel('False Positive Rate', fontsize=12)
    ax2.set_ylabel('True Positive Rate', fontsize=12)
    ax2.set_title('ResNet3D + XGBoost', fontsize=14, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=9)
    ax2.grid(alpha=0.3)

    plt.suptitle('ROC Curves — Test Set', fontsize=16, fontweight='bold')
    plt.tight_layout()
    fig.savefig(REPORT_DIR / "roc_curves.png", dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {REPORT_DIR / 'roc_curves.png'}")


# ═══════════════════════════════════════════════════════
# 4. XGBoost Feature Importance
# ═══════════════════════════════════════════════════════

def plot_xgboost_importance():
    """Plot feature importance for XGBoost models."""
    print("\n" + "=" * 60)
    print("XGBoost Feature Importance")
    print("=" * 60)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 1. Tabular-only XGBoost (from late fusion)
    xgb_tab_path = XGB_DIR / "results_late_fusion" / "xgboost_tabular.json"
    if xgb_tab_path.exists():
        model_tab = xgb.Booster()
        model_tab.load_model(str(xgb_tab_path))
        importance = model_tab.get_score(importance_type='gain')

        if importance:
            features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
            names, values = zip(*features)
            colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(names)))

            axes[0].barh(range(len(names)), values, color=colors)
            axes[0].set_yticks(range(len(names)))
            axes[0].set_yticklabels(names, fontsize=9)
            axes[0].invert_yaxis()
            axes[0].set_xlabel('Gain', fontsize=11)
            axes[0].set_title('Tabular XGBoost\n(Late Fusion Branch)', fontsize=12, fontweight='bold')
            axes[0].grid(axis='x', alpha=0.3)
        print(f"  Loaded tabular XGBoost: {len(importance)} features")

    # 2. Early fusion XGBoost (embeddings + tabular)
    xgb_early_path = XGB_DIR / "results_finetuned" / "xgboost_model.json"
    if xgb_early_path.exists():
        model_early = xgb.Booster()
        model_early.load_model(str(xgb_early_path))
        importance = model_early.get_score(importance_type='gain')

        if importance:
            # Separate CNN features from tabular features
            cnn_gain = sum(v for k, v in importance.items() if k.startswith('cnn_'))
            tab_features = {k: v for k, v in importance.items() if not k.startswith('cnn_')}
            n_cnn = sum(1 for k in importance if k.startswith('cnn_'))

            # Show tabular features + aggregated CNN
            combined = {'CNN embeddings\n(aggregated)': cnn_gain}
            combined.update(tab_features)
            features = sorted(combined.items(), key=lambda x: x[1], reverse=True)
            names, values = zip(*features)

            colors = ['#e74c3c' if 'CNN' in n else '#3498db' for n in names]

            axes[1].barh(range(len(names)), values, color=colors)
            axes[1].set_yticks(range(len(names)))
            axes[1].set_yticklabels(names, fontsize=9)
            axes[1].invert_yaxis()
            axes[1].set_xlabel('Gain', fontsize=11)
            axes[1].set_title('Early Fusion XGBoost\n(CNN embeddings + Tabular)', fontsize=12, fontweight='bold')
            axes[1].grid(axis='x', alpha=0.3)

            # Legend
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor='#e74c3c', label=f'CNN ({n_cnn} features)'),
                               Patch(facecolor='#3498db', label='Tabular')]
            axes[1].legend(handles=legend_elements, loc='lower right', fontsize=9)
            print(f"  Loaded early fusion XGBoost: {n_cnn} CNN + {len(tab_features)} tabular features")

    plt.suptitle('XGBoost Feature Importance (Gain)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(REPORT_DIR / "xgboost_feature_importance.png", dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {REPORT_DIR / 'xgboost_feature_importance.png'}")


# ═══════════════════════════════════════════════════════
# 5. GradCAM for ResNet3D
# ═══════════════════════════════════════════════════════

def gradcam_resnet3d():
    """Generate GradCAM heatmaps for ResNet3D predictions."""
    print("\n" + "=" * 60)
    print("GradCAM — ResNet3D Explainability")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load config
    with open(MLP_DIR / "config.yaml") as f:
        config = yaml.safe_load(f)

    preproc = config.get('preprocessing', {})
    target_shape = tuple(preproc.get('target_shape', [128, 128, 128]))
    tabular_features = config['data']['tabular_features']

    # Try to load MLP early fusion model (best AUC)
    model_path = MLP_DIR / "results_early" / "best_model.pth"
    if not model_path.exists():
        print("  No early fusion model found, skipping GradCAM.")
        return

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = EarlyFusionModel(
        pretrained=False,
        tabular_input_dim=len(tabular_features),
        tabular_hidden_dims=config['model']['tabular']['hidden_dims'],
        fusion_hidden_dims=config['model']['early_fusion']['hidden_dims'],
        num_classes=config['model']['early_fusion']['num_classes'],
        dropout=config['model']['early_fusion']['dropout'],
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Hook for GradCAM on last conv layer (layer4 of ResNet50)
    activations = {}
    gradients = {}

    def forward_hook(module, input, output):
        activations['value'] = output.detach()

    def backward_hook(module, grad_input, grad_output):
        gradients['value'] = grad_output[0].detach()

    # Register hooks on the last layer of the ResNet backbone
    target_layer = model.backbone.net.layer4
    fwd_handle = target_layer.register_forward_hook(forward_hook)
    bwd_handle = target_layer.register_full_backward_hook(backward_hook)

    # Load test dataset
    # Use train scaler
    train_dataset = MultiModalDataset(
        config['data']['train_csv'], tabular_features=tabular_features,
        target_shape=target_shape, augment=False, normalize_tabular=True,
        scaler=None, use_paper_preprocessing=preproc.get('use_paper_preprocessing', True),
        target_spacing=preproc.get('target_spacing', 1.75)
    )
    scaler = train_dataset.get_scaler()

    test_dataset = MultiModalDataset(
        config['data']['test_csv'], tabular_features=tabular_features,
        target_shape=target_shape, augment=False, normalize_tabular=True,
        scaler=scaler, use_paper_preprocessing=preproc.get('use_paper_preprocessing', True),
        target_spacing=preproc.get('target_spacing', 1.75)
    )

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    # Collect examples: TP, TN, FP, FN
    examples = {'TP': [], 'TN': [], 'FP': [], 'FN': []}
    target_counts = {'TP': 3, 'TN': 3, 'FP': 1, 'FN': 1}

    print("  Collecting GradCAM examples...")
    for idx, (mri, tabular, label) in enumerate(test_loader):
        # Check if we have enough
        if all(len(examples[k]) >= target_counts[k] for k in examples):
            break

        mri_gpu = mri.to(device).requires_grad_(True)
        tabular_gpu = tabular.to(device)
        label_val = label.item()

        # Forward pass
        output = model(mri_gpu, tabular_gpu)
        pred = output.argmax(dim=1).item()
        prob = torch.softmax(output, dim=1)[0, 1].item()

        # Classify example
        if pred == 1 and label_val == 1:
            cat = 'TP'
        elif pred == 0 and label_val == 0:
            cat = 'TN'
        elif pred == 1 and label_val == 0:
            cat = 'FP'
        else:
            cat = 'FN'

        if len(examples[cat]) >= target_counts[cat]:
            continue

        # Backward for GradCAM (target class = predicted class)
        model.zero_grad()
        target_score = output[0, pred]
        target_score.backward()

        # Compute GradCAM
        grads = gradients['value']  # (1, C, D, H, W)
        acts = activations['value']  # (1, C, D, H, W)
        weights = grads.mean(dim=[2, 3, 4], keepdim=True)  # GAP over spatial dims
        cam = (weights * acts).sum(dim=1, keepdim=True)  # (1, 1, D, H, W)
        cam = torch.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        # Upsample to input size
        from scipy.ndimage import zoom
        zoom_factors = [s / c for s, c in zip(target_shape, cam.shape)]
        cam_upsampled = zoom(cam, zoom_factors, order=1)
        cam_upsampled = (cam_upsampled - cam_upsampled.min()) / (cam_upsampled.max() - cam_upsampled.min() + 1e-8)

        examples[cat].append({
            'mri': mri.squeeze().numpy(),
            'cam': cam_upsampled,
            'label': label_val,
            'pred': pred,
            'prob': prob,
            'idx': idx,
        })

    fwd_handle.remove()
    bwd_handle.remove()

    # Plot GradCAM examples
    all_examples = []
    for cat in ['TP', 'TN', 'FP', 'FN']:
        all_examples.extend([(cat, ex) for ex in examples[cat]])

    n_examples = len(all_examples)
    if n_examples == 0:
        print("  No examples collected.")
        return

    fig, axes = plt.subplots(n_examples, 3, figsize=(12, 3.5 * n_examples))
    if n_examples == 1:
        axes = axes.reshape(1, -1)

    slice_names = ['Axial', 'Coronal', 'Sagittal']

    for row, (cat, ex) in enumerate(all_examples):
        mri = ex['mri']
        cam = ex['cam']
        d, h, w = mri.shape
        true_label = 'AD' if ex['label'] == 1 else 'CN'
        pred_label = 'AD' if ex['pred'] == 1 else 'CN'

        slices_mri = [mri[d // 2, :, :], mri[:, h // 2, :], mri[:, :, w // 2]]
        slices_cam = [cam[d // 2, :, :], cam[:, h // 2, :], cam[:, :, w // 2]]

        for col in range(3):
            ax = axes[row, col]
            ax.imshow(slices_mri[col].T, cmap='gray', origin='lower', aspect='auto')
            ax.imshow(slices_cam[col].T, cmap='jet', alpha=0.4, origin='lower', aspect='auto')
            ax.axis('off')

            if col == 0:
                color = 'green' if cat in ('TP', 'TN') else 'red'
                ax.set_ylabel(f'{cat}\nTrue={true_label}, Pred={pred_label}\np(AD)={ex["prob"]:.2f}',
                              fontsize=9, fontweight='bold', color=color, rotation=0, labelpad=100,
                              verticalalignment='center')
            if row == 0:
                ax.set_title(slice_names[col], fontsize=12, fontweight='bold')

    plt.suptitle('GradCAM — ResNet3D Early Fusion\n(Red = high attention regions)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(REPORT_DIR / "gradcam_examples.png", dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {REPORT_DIR / 'gradcam_examples.png'}")

    # Free GPU
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


# ═══════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════

if __name__ == '__main__':
    print("ResNet3D Fusion — Post-training Analysis")
    print("=" * 60)

    # 1. DeLong + load predictions
    result = run_delong_analysis()
    if result is not None:
        y_true, preds = result

        # 2. Confusion matrices
        plot_confusion_matrices(y_true, preds)

        # 3. ROC curves
        plot_roc_curves(y_true, preds)

    # 4. XGBoost feature importance
    plot_xgboost_importance()

    # 5. GradCAM
    gradcam_resnet3d()

    print("\n" + "=" * 60)
    print(f"All outputs saved to {REPORT_DIR}/")
    print("=" * 60)

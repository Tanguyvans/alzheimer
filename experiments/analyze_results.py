#!/usr/bin/env python3
"""
Post-training analysis: DeLong test, confusion matrices, XAI (GradCAM + feature importance).

Run AFTER all training scripts have completed (run_all.sh).

Usage:
    python analyze_results.py                  # CPU analyses only
    python analyze_results.py --gradcam        # include GradCAM (needs GPU)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from pathlib import Path
from scipy import stats
from sklearn.metrics import (confusion_matrix, roc_curve, auc,
                             roc_auc_score, accuracy_score, balanced_accuracy_score)
import argparse
import warnings
warnings.filterwarnings('ignore')

# ── Paths ──
BASE = Path(__file__).parent
MLP_DIR = BASE / "resnet3d_mlp"
XGB_DIR = BASE / "resnet3d_xgboost"
REPORT_DIR = BASE / "report"
REPORT_DIR.mkdir(exist_ok=True)


# ═══════════════════════════════════════════════════════
# Load predictions
# ═══════════════════════════════════════════════════════

def load_predictions():
    """Load all y_true and y_proba arrays from saved .npy files."""
    preds = {}
    y_true = None

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

    return y_true, preds


# ═══════════════════════════════════════════════════════
# 1. DeLong Test
# ═══════════════════════════════════════════════════════

def compute_midrank(x):
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
    """Fast DeLong AUC computation (Sun & Xu 2014)."""
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    k = predictions_sorted_transposed.shape[0]

    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]

    tx = np.empty([k, m], dtype=np.float64)
    ty = np.empty([k, n], dtype=np.float64)
    tz = np.empty([k, m + n], dtype=np.float64)

    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])

    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m

    sx = np.cov(v01) if k > 1 else np.atleast_2d(np.var(v01, axis=1))
    sy = np.cov(v10) if k > 1 else np.atleast_2d(np.var(v10, axis=1))
    S = sx / m + sy / n
    return aucs, S


def delong_test(y_true, y_score1, y_score2):
    """Two-sided DeLong test for two correlated ROC AUCs."""
    order = (-y_true).argsort()
    label_1_count = int(y_true.sum())
    predictions = np.vstack([y_score1, y_score2])
    predictions_sorted = predictions[:, order]
    aucs, S = fastDeLong(predictions_sorted, label_1_count)
    diff = aucs[0] - aucs[1]
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    if var <= 0:
        return aucs[0], aucs[1], 0.0, 1.0
    z = diff / np.sqrt(var)
    p = 2 * stats.norm.sf(abs(z))
    return aucs[0], aucs[1], z, p


def _order_key(name, auc_values):
    """Sort key: unimodal (0) → early (1) → late (2), then by AUC desc."""
    if 'only' in name.lower():
        group = 0
    elif 'Early' in name:
        group = 1
    else:
        group = 2
    return (group, -auc_values[name])


def run_delong_analysis(y_true, preds):
    """Run pairwise DeLong tests and generate heatmap."""
    print("=" * 60)
    print("DeLong Test - Pairwise AUC comparison")
    print("=" * 60)

    # Compute AUCs
    auc_values = {name: roc_auc_score(y_true, preds[name]) for name in preds}

    # Sort: unimodal → early → late, then by AUC
    sorted_names = sorted(preds.keys(), key=lambda x: _order_key(x, auc_values))
    n = len(sorted_names)

    # Pairwise DeLong
    p_matrix = np.ones((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            _, _, z, p_val = delong_test(y_true, preds[sorted_names[i]], preds[sorted_names[j]])
            p_matrix[i, j] = p_val
            p_matrix[j, i] = p_val

    # Print
    print(f"\n{'Method':<20} {'AUC':>6}")
    print("-" * 28)
    for name in sorted(sorted_names, key=lambda x: auc_values[x], reverse=True):
        print(f"{name:<20} {auc_values[name]:.4f}")

    print("\nSignificant differences (p < 0.05):")
    found = False
    for i in range(n):
        for j in range(i + 1, n):
            if p_matrix[i, j] < 0.05:
                found = True
                diff = auc_values[sorted_names[i]] - auc_values[sorted_names[j]]
                print(f"  {sorted_names[i]} vs {sorted_names[j]}: p={p_matrix[i,j]:.4f}, AUC diff={diff:+.4f}")
    if not found:
        print("  None")

    # ── Heatmap ──
    annot = np.full((n, n), "", dtype=object)
    for i in range(n):
        for j in range(n):
            if i == j:
                annot[i, j] = f"AUC\n{auc_values[sorted_names[i]]:.3f}"
            else:
                p = p_matrix[i, j]
                if p < 0.001:
                    annot[i, j] = "p<.001\n***"
                elif p < 0.01:
                    annot[i, j] = f"p={p:.3f}\n**"
                elif p < 0.05:
                    annot[i, j] = f"p={p:.3f}\n*"
                else:
                    annot[i, j] = f"p={p:.2f}\nns"

    color_matrix = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            if i != j:
                color_matrix[i, j] = -np.log10(max(p_matrix[i, j], 1e-10))

    fig, ax = plt.subplots(figsize=(max(10, n * 1.1), max(9, n * 1.0)))

    cmap = LinearSegmentedColormap.from_list(
        'significance', ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c', '#8e44ad'], N=256)

    sns.heatmap(color_matrix, xticklabels=sorted_names, yticklabels=sorted_names,
                annot=annot, fmt='', cmap=cmap, vmin=0, vmax=5,
                mask=np.eye(n, dtype=bool), square=True, ax=ax,
                cbar_kws={'label': 'Significativite  (-log10 p)', 'shrink': 0.7},
                annot_kws={'size': 8},
                linewidths=1.5, linecolor='white')

    # Diagonal: light gray + bold AUC
    for i in range(n):
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=True,
                     facecolor='#ecf0f1', edgecolor='white', lw=1.5))
        ax.text(i + 0.5, i + 0.5, f"AUC\n{auc_values[sorted_names[i]]:.3f}",
                ha='center', va='center', fontsize=9, fontweight='bold', color='#2c3e50')

    # Group separators (unimodal / early / late)
    prev_group = None
    for i, name in enumerate(sorted_names):
        g = 0 if 'only' in name.lower() else (1 if 'Early' in name else 2)
        if prev_group is not None and g != prev_group:
            ax.axhline(y=i, color='#2c3e50', linewidth=2.5)
            ax.axvline(x=i, color='#2c3e50', linewidth=2.5)
        prev_group = g

    ax.set_title('Test de DeLong - Comparaison par paires\n'
                 '* p<0.05   ** p<0.01   *** p<0.001   ns = non significatif',
                 fontsize=11, fontweight='bold', pad=12)
    plt.xticks(rotation=40, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    fig.savefig(REPORT_DIR / "delong_test.png", dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {REPORT_DIR / 'delong_test.png'}")

    df = pd.DataFrame(p_matrix, index=sorted_names, columns=sorted_names)
    df.to_csv(REPORT_DIR / "delong_pvalues.csv")
    print(f"Saved: {REPORT_DIR / 'delong_pvalues.csv'}")


# ═══════════════════════════════════════════════════════
# 2. Confusion Matrices
# ═══════════════════════════════════════════════════════

def plot_confusion_matrices(y_true, preds):
    print("\n" + "=" * 60)
    print("Confusion Matrices")
    print("=" * 60)

    methods = [
        ("MRI only (XGB)", "MRI only\n(ResNet3D)"),
        ("Tab only (XGB)", "Tabular only\n(XGBoost)"),
        ("Tab only (MLP)", "Tabular only\n(MLP)"),
        ("XGB Early", "Early Fusion\n(ResNet3D+XGB)"),
        ("MLP Early", "Early Fusion\n(ResNet3D+MLP)"),
        ("XGB Late Wt", "Late Weighted\n(ResNet3D+XGB)"),
        ("MLP Late Wt", "Late Weighted\n(ResNet3D+MLP)"),
        ("XGB Late Stack", "Late Stacking\n(ResNet3D+XGB)"),
    ]

    available = [(key, label) for key, label in methods if key in preds]
    n_methods = len(available)
    ncols = 4
    nrows = (n_methods + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.5 * nrows))
    if nrows == 1:
        axes = axes.reshape(1, -1)

    for idx, (key, label) in enumerate(available):
        row, col = idx // ncols, idx % ncols
        ax = axes[row, col]
        y_pred = (preds[key] >= 0.5).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['CN', 'AD'], yticklabels=['CN', 'AD'],
                    cbar=False, annot_kws={'size': 16, 'weight': 'bold'}, square=True)
        for i in range(2):
            for j in range(2):
                ax.text(j + 0.5, i + 0.75, f'({cm_pct[i, j]:.1f}%)',
                        ha='center', va='center', fontsize=9, color='gray')

        auc_val = roc_auc_score(y_true, preds[key])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True' if col == 0 else '')
        ax.set_title(f'{label}\nAUC={auc_val:.3f}', fontsize=10, fontweight='bold')

    for idx in range(n_methods, nrows * ncols):
        row, col = idx // ncols, idx % ncols
        axes[row, col].set_visible(False)

    n_cn = int((y_true == 0).sum())
    n_ad = int((y_true == 1).sum())
    plt.suptitle(f'Confusion Matrices - Test Set ({n_cn} CN, {n_ad} AD)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(REPORT_DIR / "confusion_matrices.png", dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {REPORT_DIR / 'confusion_matrices.png'}")


# ═══════════════════════════════════════════════════════
# 3. ROC Curves
# ═══════════════════════════════════════════════════════

def plot_roc_curves(y_true, preds):
    print("\n" + "=" * 60)
    print("ROC Curves")
    print("=" * 60)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    def _plot_methods(ax, method_list, title):
        for key, label, ls, color in method_list:
            if key in preds:
                fpr, tpr, _ = roc_curve(y_true, preds[key])
                auc_val = auc(fpr, tpr)
                ax.plot(fpr, tpr, ls=ls, color=color, label=f"{label} ({auc_val:.3f})", linewidth=2)
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(title, fontweight='bold')
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(alpha=0.3)

    _plot_methods(ax1, [
        ("MRI only (MLP)", "MRI only", "--", 'gray'),
        ("Tab only (MLP)", "Tabular MLP", "--", 'steelblue'),
        ("MLP Early", "Early Fusion", "-", 'darkorange'),
        ("MLP Late Wt", "Late Weighted", "-", 'green'),
        ("MLP Late Stack", "Late Stacking", "-.", 'purple'),
    ], 'ResNet3D + MLP')

    _plot_methods(ax2, [
        ("MRI only (XGB)", "MRI only", "--", 'gray'),
        ("Tab only (XGB)", "Tabular XGBoost", "--", 'steelblue'),
        ("XGB Early", "Early Fusion", "-", 'darkorange'),
        ("XGB Late Wt", "Late Weighted", "-", 'green'),
        ("XGB Late Stack", "Late Stacking", "-.", 'purple'),
    ], 'ResNet3D + XGBoost')

    plt.suptitle('ROC Curves - Test Set', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(REPORT_DIR / "roc_curves.png", dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {REPORT_DIR / 'roc_curves.png'}")


# ═══════════════════════════════════════════════════════
# 4. XGBoost Feature Importance
# ═══════════════════════════════════════════════════════

def plot_xgboost_importance():
    print("\n" + "=" * 60)
    print("XGBoost Feature Importance")
    print("=" * 60)

    try:
        import xgboost as xgb
    except ImportError:
        print("xgboost not installed, skipping")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Tabular-only XGBoost
    xgb_tab_path = XGB_DIR / "results_late_fusion" / "xgboost_tabular.json"
    if xgb_tab_path.exists():
        model_tab = xgb.Booster()
        model_tab.load_model(str(xgb_tab_path))
        importance = model_tab.get_score(importance_type='gain')
        if importance:
            features = sorted(importance.items(), key=lambda x: x[1])
            feat_names, values = zip(*features)
            colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(feat_names)))
            axes[0].barh(range(len(feat_names)), values, color=colors)
            axes[0].set_yticks(range(len(feat_names)))
            axes[0].set_yticklabels(feat_names, fontsize=9)
            axes[0].set_xlabel('Gain')
            axes[0].set_title('Tabular XGBoost\n(Late Fusion Branch)', fontweight='bold')
            axes[0].grid(axis='x', alpha=0.3)
        print(f"  Tabular XGBoost: {len(importance)} features")

    # Early fusion XGBoost (embeddings + tabular)
    xgb_early_path = XGB_DIR / "results_finetuned" / "xgboost_model.json"
    if xgb_early_path.exists():
        model_early = xgb.Booster()
        model_early.load_model(str(xgb_early_path))
        importance = model_early.get_score(importance_type='gain')
        if importance:
            cnn_gain = sum(v for k, v in importance.items() if k.startswith('cnn_'))
            tab_features = {k: v for k, v in importance.items() if not k.startswith('cnn_')}
            n_cnn = sum(1 for k in importance if k.startswith('cnn_'))

            combined = {'CNN embeddings\n(2048-d aggregate)': cnn_gain}
            combined.update(tab_features)
            features = sorted(combined.items(), key=lambda x: x[1])
            feat_names, values = zip(*features)
            colors = ['#e74c3c' if 'CNN' in n else '#3498db' for n in feat_names]

            axes[1].barh(range(len(feat_names)), values, color=colors)
            axes[1].set_yticks(range(len(feat_names)))
            axes[1].set_yticklabels(feat_names, fontsize=9)
            axes[1].set_xlabel('Gain')
            axes[1].set_title('Early Fusion XGBoost\n(CNN emb + Tabular)', fontweight='bold')
            axes[1].grid(axis='x', alpha=0.3)

            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor='#e74c3c', label=f'CNN ({n_cnn} features)'),
                               Patch(facecolor='#3498db', label='Tabular')]
            axes[1].legend(handles=legend_elements, loc='lower right', fontsize=9)
            print(f"  Early fusion XGBoost: {n_cnn} CNN + {len(tab_features)} tabular features")

    plt.suptitle('XGBoost Feature Importance (Gain)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(REPORT_DIR / "xgboost_feature_importance.png", dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {REPORT_DIR / 'xgboost_feature_importance.png'}")


# ═══════════════════════════════════════════════════════
# 5. GradCAM for ResNet3D
# ═══════════════════════════════════════════════════════

def gradcam_resnet3d():
    print("\n" + "=" * 60)
    print("GradCAM - ResNet3D Explainability")
    print("=" * 60)

    import torch
    import yaml
    import importlib.util
    from scipy.ndimage import zoom

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    _spec = importlib.util.spec_from_file_location("model", MLP_DIR / "model.py")
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)

    _spec_ds = importlib.util.spec_from_file_location("dataset", BASE / "multimodal_fusion" / "dataset.py")
    _ds_mod = importlib.util.module_from_spec(_spec_ds)
    _spec_ds.loader.exec_module(_ds_mod)

    with open(MLP_DIR / "config.yaml") as f:
        config = yaml.safe_load(f)

    preproc = config.get('preprocessing', {})
    target_shape = tuple(preproc.get('target_shape', [128, 128, 128]))
    tabular_features = config['data']['tabular_features']

    model_path = MLP_DIR / "results_early" / "best_model.pth"
    if not model_path.exists():
        print("  No early fusion model found, skipping GradCAM.")
        return

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = _mod.EarlyFusionModel(
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

    activations = {}
    gradients = {}

    def forward_hook(module, input, output):
        activations['value'] = output.detach()

    def backward_hook(module, grad_input, grad_output):
        gradients['value'] = grad_output[0].detach()

    target_layer = model.backbone.net.layer4
    fwd_handle = target_layer.register_forward_hook(forward_hook)
    bwd_handle = target_layer.register_full_backward_hook(backward_hook)

    train_dataset = _ds_mod.MultiModalDataset(
        config['data']['train_csv'], tabular_features=tabular_features,
        target_shape=target_shape, augment=False, normalize_tabular=True,
        scaler=None, use_paper_preprocessing=preproc.get('use_paper_preprocessing', True),
        target_spacing=preproc.get('target_spacing', 1.75)
    )
    scaler = train_dataset.get_scaler()

    test_dataset = _ds_mod.MultiModalDataset(
        config['data']['test_csv'], tabular_features=tabular_features,
        target_shape=target_shape, augment=False, normalize_tabular=True,
        scaler=scaler, use_paper_preprocessing=preproc.get('use_paper_preprocessing', True),
        target_spacing=preproc.get('target_spacing', 1.75)
    )

    examples = {'TP': [], 'TN': [], 'FP': [], 'FN': []}
    target_counts = {'TP': 3, 'TN': 3, 'FP': 2, 'FN': 2}

    print("  Collecting GradCAM examples...")
    for idx in range(len(test_dataset)):
        if all(len(examples[k]) >= target_counts[k] for k in examples):
            break

        mri, tabular, label_val = test_dataset[idx]
        mri_gpu = mri.unsqueeze(0).to(device).requires_grad_(True)
        tabular_gpu = tabular.unsqueeze(0).to(device)

        model.zero_grad()
        output = model(mri_gpu, tabular_gpu)
        pred = output.argmax(dim=1).item()
        prob = torch.softmax(output, dim=1)[0, 1].item()

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

        output[0, 1].backward()
        grads = gradients['value']
        acts = activations['value']
        weights = grads.mean(dim=[2, 3, 4], keepdim=True)
        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = torch.relu(cam).squeeze().cpu().numpy()

        zoom_factors = [s / c for s, c in zip(target_shape, cam.shape)]
        cam_up = zoom(cam, zoom_factors, order=1)
        cam_up = (cam_up - cam_up.min()) / (cam_up.max() - cam_up.min() + 1e-8)

        examples[cat].append({
            'mri': mri.squeeze().numpy(),
            'cam': cam_up,
            'label': label_val,
            'pred': pred,
            'prob': prob,
        })

    fwd_handle.remove()
    bwd_handle.remove()

    all_examples = []
    for cat in ['TP', 'TN', 'FP', 'FN']:
        all_examples.extend([(cat, ex) for ex in examples[cat]])

    n_ex = len(all_examples)
    if n_ex == 0:
        print("  No examples collected.")
        return

    fig, axes = plt.subplots(n_ex, 3, figsize=(12, 3.5 * n_ex))
    if n_ex == 1:
        axes = axes.reshape(1, -1)

    slice_names = ['Sagittal', 'Coronal', 'Axial']
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
                ax.set_ylabel(f'{cat}\nTrue={true_label}\nPred={pred_label}\np(AD)={ex["prob"]:.2f}',
                              fontsize=9, fontweight='bold', color=color, rotation=0,
                              labelpad=80, verticalalignment='center')
            if row == 0:
                ax.set_title(slice_names[col], fontsize=12, fontweight='bold')

    plt.suptitle('GradCAM - ResNet3D Early Fusion\n(Red = high AD-related activation)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(REPORT_DIR / "gradcam_examples.png", dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {REPORT_DIR / 'gradcam_examples.png'}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════
# 6. Summary Table
# ═══════════════════════════════════════════════════════

def generate_summary(y_true, preds):
    print("\n" + "=" * 60)
    print("SUMMARY TABLE")
    print("=" * 60)

    rows = []
    for method, y_proba in preds.items():
        y_pred = (y_proba >= 0.5).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        rows.append({
            'Method': method,
            'Acc (%)': accuracy_score(y_true, y_pred) * 100,
            'Bal Acc (%)': balanced_accuracy_score(y_true, y_pred) * 100,
            'Sens (%)': tp / (tp + fn) * 100 if (tp + fn) > 0 else 0,
            'Spec (%)': tn / (tn + fp) * 100 if (tn + fp) > 0 else 0,
            'AUC': roc_auc_score(y_true, y_proba),
        })

    df = pd.DataFrame(rows).sort_values('AUC', ascending=False)
    print(df.to_string(index=False, float_format=lambda x: f'{x:.1f}' if x > 1 else f'{x:.3f}'))
    df.to_csv(REPORT_DIR / "summary_table.csv", index=False)
    print(f"\nSaved: {REPORT_DIR / 'summary_table.csv'}")


# ═══════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze ResNet3D fusion results')
    parser.add_argument('--gradcam', action='store_true', help='Include GradCAM (needs GPU)')
    args = parser.parse_args()

    print("ResNet3D Fusion - Post-training Analysis")
    print("=" * 60)

    print("Loading predictions...")
    y_true, preds = load_predictions()
    print(f"Loaded {len(preds)} methods, {len(y_true)} test samples\n")

    generate_summary(y_true, preds)
    run_delong_analysis(y_true, preds)
    plot_confusion_matrices(y_true, preds)
    plot_roc_curves(y_true, preds)
    plot_xgboost_importance()

    if args.gradcam:
        gradcam_resnet3d()
    else:
        print("\nSkipping GradCAM (use --gradcam to include, needs GPU)")

    print("\n" + "=" * 60)
    print(f"All outputs saved to {REPORT_DIR}/")
    print("=" * 60)

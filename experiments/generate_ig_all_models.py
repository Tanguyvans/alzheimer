#!/usr/bin/env python3
"""
Generate Integrated Gradients visualizations for all 4 models.
Same 5 AD + 5 CN patients across all models for direct comparison.

Output:
    interpretability/
    ├── mlp_early_fusion/   AD_01..05.png, CN_01..05.png
    ├── mlp_late_fusion/    AD_01..05.png, CN_01..05.png
    ├── xgb_early_fusion/   AD_01..05.png, CN_01..05.png
    ├── xgb_late_fusion/    AD_01..05.png, CN_01..05.png
    └── group_comparison.png  (summary figure)

Usage:
    cd experiments/resnet3d_mlp
    python ../generate_ig_all_models.py [--seed 2] [--n-individual 5] [--n-steps 100]
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import yaml
import argparse
import importlib.util
from pathlib import Path
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score

# ── Paths ──
BASE = Path(__file__).parent
MLP_DIR = BASE / "resnet3d_mlp"
XGB_DIR = BASE / "resnet3d_xgboost"
REPORT_DIR = BASE / "report_multi_seed" / "interpretability"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_dataset(config_path):
    """Load test dataset using config from resnet3d_mlp."""
    ds_mod = load_module("dataset", BASE / "multimodal_fusion" / "dataset.py")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    preproc = config.get('preprocessing', {})
    target_shape = tuple(preproc.get('target_shape', [128, 128, 128]))
    tabular_features = config['data']['tabular_features']

    train_dataset = ds_mod.MultiModalDataset(
        config['data']['train_csv'], tabular_features=tabular_features,
        target_shape=target_shape, augment=False, normalize_tabular=True,
        scaler=None, use_paper_preprocessing=preproc.get('use_paper_preprocessing', True),
        target_spacing=preproc.get('target_spacing', 1.75)
    )
    scaler = train_dataset.get_scaler()
    test_dataset = ds_mod.MultiModalDataset(
        config['data']['test_csv'], tabular_features=tabular_features,
        target_shape=target_shape, augment=False, normalize_tabular=True,
        scaler=scaler, use_paper_preprocessing=preproc.get('use_paper_preprocessing', True),
        target_spacing=preproc.get('target_spacing', 1.75)
    )
    return test_dataset, config, target_shape, tabular_features


# ═══════════════════════════════════════════════════════
# Model loaders
# ═══════════════════════════════════════════════════════

def load_mlp_early(seed, device, config, tabular_features):
    """MLP Early Fusion: EarlyFusionModel(mri, tabular) -> logits."""
    model_mod = load_module("model", MLP_DIR / "model.py")
    model_path = MLP_DIR / "results_early" / f"seed_{seed}" / "best_model.pth"
    if not model_path.exists():
        print(f"  [SKIP] {model_path} not found")
        return None

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = model_mod.EarlyFusionModel(
        pretrained=False,
        tabular_input_dim=len(tabular_features),
        tabular_hidden_dims=config['model']['tabular']['hidden_dims'],
        fusion_hidden_dims=config['model']['early_fusion']['hidden_dims'],
        num_classes=config['model']['early_fusion']['num_classes'],
        dropout=config['model']['early_fusion']['dropout'],
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device).eval()

    def forward_fn(mri, tabular):
        return model(mri, tabular)

    return forward_fn, model


def load_resnet3d_classifier(checkpoint_path, device):
    """Load ResNet3DClassifier (backbone + linear head) used by Late Fusion & XGBoost models."""
    model_mod = load_module("model", MLP_DIR / "model.py")

    class ResNet3DClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = model_mod.ResNet3DBackbone(pretrained=False)
            self.head = nn.Linear(2048, 2)

        def forward(self, x):
            features = self.backbone(x)
            return self.head(features)

    model = ResNet3DClassifier()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    # Handle both dict formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.to(device).eval()

    def forward_fn(mri, tabular):
        return model(mri)

    return forward_fn, model


def load_mlp_late(seed, device):
    path = MLP_DIR / "results_late_fusion" / f"seed_{seed}" / "resnet3d_finetuned.pth"
    if not path.exists():
        print(f"  [SKIP] {path} not found")
        return None
    return load_resnet3d_classifier(path, device)


def load_xgb_early(seed, device):
    path = XGB_DIR / "results_finetuned" / f"seed_{seed}" / "finetuned_resnet3d.pth"
    if not path.exists():
        print(f"  [SKIP] {path} not found")
        return None
    return load_resnet3d_classifier(path, device)


def load_xgb_late(seed, device):
    path = XGB_DIR / "results_late_fusion" / f"seed_{seed}" / "resnet3d_finetuned.pth"
    if not path.exists():
        print(f"  [SKIP] {path} not found")
        return None
    return load_resnet3d_classifier(path, device)


# ═══════════════════════════════════════════════════════
# Integrated Gradients
# ═══════════════════════════════════════════════════════

def compute_ig(forward_fn, mri_gpu, tabular_gpu, device, n_steps=100):
    """Compute Integrated Gradients w.r.t. MRI input."""
    baseline_mri = torch.zeros_like(mri_gpu)
    baseline_tab = torch.zeros_like(tabular_gpu)

    ig_grads = torch.zeros_like(mri_gpu)
    for step in range(n_steps):
        alpha = step / n_steps
        interp_mri = (baseline_mri + alpha * (mri_gpu - baseline_mri)).requires_grad_(True)
        interp_tab = baseline_tab + alpha * (tabular_gpu - baseline_tab)

        out = forward_fn(interp_mri, interp_tab)
        out[0, 1].backward()
        ig_grads += interp_mri.grad.detach()
        # Zero grads on any model params
        if hasattr(forward_fn, '__self__'):
            forward_fn.__self__.zero_grad()

    ig_attr = ((mri_gpu - baseline_mri) * ig_grads / n_steps).squeeze().cpu().numpy()
    ig_attr = np.abs(ig_attr)
    ig_attr = gaussian_filter(ig_attr, sigma=2.0)
    ig_attr = (ig_attr - ig_attr.min()) / (ig_attr.max() - ig_attr.min() + 1e-8)
    return ig_attr


# ═══════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════

def plot_individual(mri, ig, label, prob, out_path):
    """Save individual patient: 3 slices with IG overlay."""
    d, h, w = mri.shape
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    slice_names = ['Sagittal', 'Coronal', 'Axial']
    slices_mri = [mri[d // 2, :, :], mri[:, h // 2, :], mri[:, :, w // 2]]
    slices_ig = [ig[d // 2, :, :], ig[:, h // 2, :], ig[:, :, w // 2]]

    for col in range(3):
        axes[col].imshow(slices_mri[col].T, cmap='gray', origin='lower', aspect='auto')
        axes[col].imshow(slices_ig[col].T, cmap='hot', alpha=0.5, origin='lower', aspect='auto')
        axes[col].set_title(slice_names[col], fontsize=12, fontweight='bold')
        axes[col].axis('off')

    diag = 'AD' if label == 1 else 'CN'
    fig.suptitle(f'{diag} — p(AD)={prob:.3f}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()


def plot_model_grid(model_name, ad_results, cn_results, out_dir):
    """Grid figure for one model: 5 AD rows + 5 CN rows, 3 columns (sag/cor/ax)."""
    all_results = ad_results + cn_results
    n = len(all_results)
    if n == 0:
        return

    fig, axes = plt.subplots(n, 3, figsize=(12, 3.2 * n))
    if n == 1:
        axes = axes.reshape(1, -1)

    slice_names = ['Sagittal', 'Coronal', 'Axial']
    for row, res in enumerate(all_results):
        mri = res['mri']
        ig = res['ig']
        d, h, w = mri.shape
        diag = 'AD' if res['label'] == 1 else 'CN'

        slices_mri = [mri[d // 2, :, :], mri[:, h // 2, :], mri[:, :, w // 2]]
        slices_ig = [ig[d // 2, :, :], ig[:, h // 2, :], ig[:, :, w // 2]]

        for col in range(3):
            ax = axes[row, col]
            ax.imshow(slices_mri[col].T, cmap='gray', origin='lower', aspect='auto')
            ax.imshow(slices_ig[col].T, cmap='hot', alpha=0.5, origin='lower', aspect='auto')
            ax.axis('off')
            if col == 0:
                color = 'darkred' if res['label'] == 1 else 'darkblue'
                ax.set_ylabel(f"{diag} #{row + 1 if res['label'] == 1 else row + 1 - len(ad_results)}\n"
                              f"p(AD)={res['prob']:.3f}",
                              fontsize=9, fontweight='bold', color=color, rotation=0,
                              labelpad=65, verticalalignment='center')
            if row == 0:
                ax.set_title(slice_names[col], fontsize=12, fontweight='bold')

    fig.suptitle(f'Integrated Gradients — {model_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(out_dir / f"grid_{model_name.lower().replace(' ', '_')}.png", dpi=200, bbox_inches='tight')
    plt.close()


def plot_cross_model_comparison(all_model_results, patient_indices, test_dataset, out_path):
    """
    Comparison grid: rows = patients, columns = models.
    Shows axial slice with IG overlay for each model on the same patient.
    """
    model_names = list(all_model_results.keys())
    n_models = len(model_names)
    n_patients = len(patient_indices)

    fig, axes = plt.subplots(n_patients, n_models, figsize=(4.5 * n_models, 3.5 * n_patients))
    if n_patients == 1:
        axes = axes.reshape(1, -1)
    if n_models == 1:
        axes = axes.reshape(-1, 1)

    for col, model_name in enumerate(model_names):
        if col < len(model_names):
            axes[0, col].set_title(model_name, fontsize=11, fontweight='bold')

        results = all_model_results[model_name]
        for row, (idx, label) in enumerate(patient_indices):
            ax = axes[row, col]
            # Find this patient's result
            match = [r for r in results if r['idx'] == idx]
            if not match:
                ax.axis('off')
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax.transAxes)
                continue

            res = match[0]
            mri = res['mri']
            ig = res['ig']
            d, h, w = mri.shape
            axial_mri = mri[:, :, w // 2].T
            axial_ig = ig[:, :, w // 2].T

            ax.imshow(axial_mri, cmap='gray', origin='lower', aspect='auto')
            ax.imshow(axial_ig, cmap='hot', alpha=0.5, origin='lower', aspect='auto')
            ax.axis('off')

            if col == 0:
                diag = 'AD' if label == 1 else 'CN'
                color = 'darkred' if label == 1 else 'darkblue'
                ax.set_ylabel(f'{diag}', fontsize=11, fontweight='bold', color=color,
                              rotation=0, labelpad=30, verticalalignment='center')

    fig.suptitle('Integrated Gradients — Cross-Model Comparison (Axial)',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()


# ═══════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Generate IG for all 4 models')
    parser.add_argument('--seed', type=int, default=2, help='Model seed to use')
    parser.add_argument('--n-individual', type=int, default=5, help='Number of AD + CN patients')
    parser.add_argument('--n-steps', type=int, default=100, help='IG interpolation steps')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Seed: {args.seed}, N patients: {args.n_individual} AD + {args.n_individual} CN")
    print(f"IG steps: {args.n_steps}")

    # Load dataset
    print("\nLoading dataset...")
    test_dataset, config, target_shape, tabular_features = load_dataset(MLP_DIR / "config.yaml")
    print(f"  Test samples: {len(test_dataset)}")

    # ── Select patients (same for all models) ──
    # Use MLP Early to classify and pick best examples
    print("\nSelecting patients using MLP Early Fusion...")
    fwd_fn, model = load_mlp_early(args.seed, device, config, tabular_features)

    sample_info = []
    for idx in range(len(test_dataset)):
        mri, tabular, label_val = test_dataset[idx]
        mri_gpu = mri.unsqueeze(0).to(device)
        tabular_gpu = tabular.unsqueeze(0).to(device)
        with torch.no_grad():
            output = fwd_fn(mri_gpu, tabular_gpu)
        prob = torch.softmax(output, dim=1)[0, 1].item()
        pred = output.argmax(dim=1).item()
        sample_info.append({'idx': idx, 'label': int(label_val), 'pred': pred, 'prob': prob})

    # Pick high-confidence correct predictions
    ad_candidates = sorted([s for s in sample_info if s['label'] == 1 and s['pred'] == 1],
                           key=lambda x: x['prob'], reverse=True)
    cn_candidates = sorted([s for s in sample_info if s['label'] == 0 and s['pred'] == 0],
                           key=lambda x: x['prob'])

    n = args.n_individual
    selected_ad = ad_candidates[:n]
    selected_cn = cn_candidates[:n]
    selected_indices = [(s['idx'], s['label']) for s in selected_ad + selected_cn]

    print(f"  Selected {len(selected_ad)} AD + {len(selected_cn)} CN patients")

    del model
    torch.cuda.empty_cache()

    # ── Define models ──
    models = {
        'MLP Early Fusion': lambda: load_mlp_early(args.seed, device, config, tabular_features),
        'MLP Late Fusion': lambda: load_mlp_late(args.seed, device),
        'XGB Early Fusion': lambda: load_xgb_early(args.seed, device),
        'XGB Late Fusion': lambda: load_xgb_late(args.seed, device),
    }

    # ── Compute IG for each model ──
    all_model_results = {}

    for model_name, loader_fn in models.items():
        print(f"\n{'=' * 60}")
        print(f"  {model_name}")
        print('=' * 60)

        result = loader_fn()
        if result is None:
            continue
        fwd_fn, model = result

        model_dir = REPORT_DIR / model_name.lower().replace(' ', '_')
        model_dir.mkdir(parents=True, exist_ok=True)

        model_results = []

        # AD patients
        for i, s in enumerate(selected_ad):
            print(f"  AD {i+1}/{n} (idx={s['idx']}, p(AD)={s['prob']:.3f})...")
            mri, tabular, label_val = test_dataset[s['idx']]
            mri_gpu = mri.unsqueeze(0).to(device)
            tabular_gpu = tabular.unsqueeze(0).to(device)

            ig = compute_ig(fwd_fn, mri_gpu, tabular_gpu, device, n_steps=args.n_steps)
            mri_np = mri.squeeze().numpy()

            plot_individual(mri_np, ig, int(label_val), s['prob'],
                            model_dir / f"AD_{i+1:02d}.png")
            model_results.append({
                'idx': s['idx'], 'mri': mri_np, 'ig': ig,
                'label': int(label_val), 'prob': s['prob'],
            })

        # CN patients
        for i, s in enumerate(selected_cn):
            print(f"  CN {i+1}/{n} (idx={s['idx']}, p(AD)={s['prob']:.3f})...")
            mri, tabular, label_val = test_dataset[s['idx']]
            mri_gpu = mri.unsqueeze(0).to(device)
            tabular_gpu = tabular.unsqueeze(0).to(device)

            ig = compute_ig(fwd_fn, mri_gpu, tabular_gpu, device, n_steps=args.n_steps)
            mri_np = mri.squeeze().numpy()

            plot_individual(mri_np, ig, int(label_val), s['prob'],
                            model_dir / f"CN_{i+1:02d}.png")
            model_results.append({
                'idx': s['idx'], 'mri': mri_np, 'ig': ig,
                'label': int(label_val), 'prob': s['prob'],
            })

        # Per-model grid
        ad_results = [r for r in model_results if r['label'] == 1]
        cn_results = [r for r in model_results if r['label'] == 0]
        plot_model_grid(model_name, ad_results, cn_results, model_dir)

        all_model_results[model_name] = model_results

        del model
        torch.cuda.empty_cache()

    # ── Cross-model comparison ──
    print(f"\n{'=' * 60}")
    print("  Cross-Model Comparison")
    print('=' * 60)
    plot_cross_model_comparison(all_model_results, selected_indices, test_dataset,
                                REPORT_DIR / "cross_model_comparison.png")
    print(f"Saved: {REPORT_DIR / 'cross_model_comparison.png'}")

    print(f"\nAll outputs saved to {REPORT_DIR}/")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Cross-modal analysis — REAL baselines edition.

Replaces the previous auxiliary-classifier analysis (which was misleading
because aux classifiers are degenerate artifacts of multi-task training).

For each of 5 folds (seed 42), loads THREE independently-trained models:
  1. MRI-only   : ablation_mri_only/cv_results/seed_42/fold_X/model.pth
  2. Tab-only   : ablation_tabular_only/cv_results/seed_42/fold_X/model.pth
  3. Fused      : multimodal_fusion/cv_results/seed_42/fold_X/model.pth

Runs each on the same fold-test subjects and compares predictions.

Usage:
    python cross_modal_real_baselines.py [--seed 42]
"""

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = Path("/home/tanguy/medical/alzheimer")
MM = REPO / "experiments" / "multimodal_fusion"
ABL_MRI = REPO / "experiments" / "ablation_mri_only"
ABL_TAB = REPO / "experiments" / "ablation_tabular_only"

sys.path.insert(0, str(MM))
from dataset import MultiModalDataset  # noqa: E402
from interpretability.ft_attention import (  # noqa: E402
    TABULAR_FEATURES, build_paper_model,
)


def regenerate_fold_test(seed, fold, n_folds=5):
    """Get fold's test set deterministically from combined_trajectory/."""
    train_df = pd.read_csv(MM / "data/combined_trajectory/train.csv")
    val_df = pd.read_csv(MM / "data/combined_trajectory/val.csv")
    test_df = pd.read_csv(MM / "data/combined_trajectory/test.csv")
    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    labels = all_df["label"].values
    indices = np.arange(len(all_df))
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fi, (_, tst_idx) in enumerate(skf.split(indices, labels)):
        if fi == fold:
            return all_df.iloc[tst_idx].reset_index(drop=True)
    raise ValueError(f"fold {fold} not found")


def predict_fused(test_df, ckpt_path, scaler, device):
    """Run multimodal Fused model on the fold test."""
    tmp = MM / "interpretability" / "_tmp_fused_test.csv"
    test_df.to_csv(tmp, index=False)
    ds = MultiModalDataset(
        str(tmp), tabular_features=TABULAR_FEATURES,
        target_shape=(128, 128, 128), augment=False,
        normalize_tabular=True, scaler=scaler,
        use_paper_preprocessing=True, target_spacing=1.75,
    )
    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)
    model = build_paper_model()
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(sd, strict=False)
    model = model.to(device).eval()
    preds = []
    probs = []
    with torch.no_grad():
        for mri, tab, _ in loader:
            mri, tab = mri.to(device), tab.to(device)
            out = model(mri, tab)
            logits = out[0] if isinstance(out, tuple) else out
            p = torch.softmax(logits, dim=-1).cpu()
            preds.append(p.argmax(-1).numpy())
            probs.append(p[:, 1].numpy())
    tmp.unlink()
    return np.concatenate(preds), np.concatenate(probs)


def build_tab_model(num_features):
    """Reproduce TabularTrainer's FT-Transformer architecture."""
    # Import the model class from ablation_tabular_only/train_cv.py
    sys.path.insert(0, str(ABL_TAB))
    import importlib
    spec = importlib.util.spec_from_file_location(
        "tab_train_cv", str(ABL_TAB / "train_cv.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # The trainer's build_model uses FTTransformer
    trainer = mod.TabularCVTrainer(
        config={"model": {"tabular": {
            "type": "ft_transformer",
            "embed_dim": 64, "num_heads": 4, "num_layers": 3,
            "dropout": 0.1, "output_dim": 64,
        }, "num_classes": 2},
        "hardware": {"device": "cuda", "num_workers": 4},
        "data": {"tabular_features": TABULAR_FEATURES},
        "training": {"batch_size": 32, "learning_rate": 1e-3, "weight_decay": 0.01,
                     "epochs": 1, "label_smoothing": 0.1, "use_weighted_loss": True,
                     "warmup_epochs": 5, "lr_min": 1e-5},
        "preprocessing": {"tabular_normalize": True, "handle_missing": "median"},
        "callbacks": {"early_stopping": {"enabled": False, "patience": 20, "min_epochs": 40, "monitor": "val_accuracy"}},
        "wandb": {"enabled": False},
        }, fold=0, seed=42, output_dir=Path("/tmp"), use_wandb=False,
    )
    return trainer.build_model(num_features)


def predict_tab(test_df, ckpt_path, scaler_path, device):
    """Run tab-only model on fold test."""
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    X = test_df[TABULAR_FEATURES].apply(lambda c: c.fillna(c.median())).values.astype(np.float32)
    X = scaler.transform(X)
    X = torch.tensor(X, dtype=torch.float32).to(device)
    model = build_tab_model(len(TABULAR_FEATURES))
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(sd)
    model = model.to(device).eval()
    with torch.no_grad():
        logits = model(X)
        p = torch.softmax(logits, dim=-1).cpu()
    return p.argmax(-1).numpy(), p[:, 1].numpy()


def predict_mri_only(test_df, ckpt_path, device):
    """Run MRI-only ViT model on fold test using the same preprocessing
    as ablation_mri_only/train_cv.py:MRIDataset (z-score normalization,
    plain zoom to 128^3, no paper_preprocessing)."""
    sys.path.insert(0, str(REPO / "experiments" / "mri_vit_ad"))
    from model import ViT3DClassifier  # noqa: E402
    sys.path.insert(0, str(ABL_MRI))
    from train_cv import MRIDataset  # noqa: E402

    tmp = MM / "interpretability" / "_tmp_mri_test.csv"
    test_df.to_csv(tmp, index=False)
    ds = MRIDataset(str(tmp), target_size=128)
    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)
    model = ViT3DClassifier(
        architecture="vit_base", num_classes=2, in_channels=1, image_size=128,
    )
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(sd, strict=False)
    model = model.to(device).eval()
    preds = []
    probs = []
    with torch.no_grad():
        for mri, _ in loader:
            mri = mri.to(device)
            logits = model(mri)
            p = torch.softmax(logits, dim=-1).cpu()
            preds.append(p.argmax(-1).numpy())
            probs.append(p[:, 1].numpy())
    tmp.unlink()
    return np.concatenate(preds), np.concatenate(probs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--skip-mri", action="store_true",
                        help="Skip MRI-only (use if ablation_mri_only not trained yet)")
    args = parser.parse_args()
    seed = args.seed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = SCRIPT_DIR / "results_new" / "03_cross_modal"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_dfs = []
    for fold in args.folds:
        print(f"\n===== FOLD {fold} =====")
        test_df = regenerate_fold_test(seed, fold)

        # Fused
        fused_ckpt = MM / "cv_results" / f"seed_{seed}" / f"fold_{fold}" / "model.pth"
        fused_scaler = MM / "cv_results" / f"seed_{seed}" / f"fold_{fold}" / "scaler.pkl"
        with open(fused_scaler, "rb") as f:
            scaler = pickle.load(f)
        fused_pred, fused_prob = predict_fused(test_df, fused_ckpt, scaler, device)
        print(f"  Fused    acc: {100 * accuracy_score(test_df['label'], fused_pred):.2f}%")

        # Tab-only
        tab_ckpt = ABL_TAB / "cv_results" / f"seed_{seed}" / f"fold_{fold}" / "model.pth"
        tab_scaler = ABL_TAB / "cv_results" / f"seed_{seed}" / f"fold_{fold}" / "scaler.pkl"
        tab_pred, tab_prob = predict_tab(test_df, tab_ckpt, tab_scaler, device)
        print(f"  Tab-only acc: {100 * accuracy_score(test_df['label'], tab_pred):.2f}%")

        # MRI-only
        if args.skip_mri:
            mri_pred = np.full(len(test_df), -1)
            mri_prob = np.full(len(test_df), np.nan)
        else:
            mri_ckpt = ABL_MRI / "cv_results" / f"seed_{seed}" / f"fold_{fold}" / "model.pth"
            mri_pred, mri_prob = predict_mri_only(test_df, mri_ckpt, device)
            print(f"  MRI-only acc: {100 * accuracy_score(test_df['label'], mri_pred):.2f}%")

        df = pd.DataFrame({
            "fold": fold,
            "subject_id": test_df["subject_id"].values,
            "label": test_df["label"].values,
            "pred_fused": fused_pred, "prob_fused_ad": fused_prob,
            "pred_mri": mri_pred,     "prob_mri_ad": mri_prob,
            "pred_tab": tab_pred,     "prob_tab_ad": tab_prob,
        })
        all_dfs.append(df)

    full = pd.concat(all_dfs, ignore_index=True)
    full.to_csv(out_dir / "cross_modal_real_predictions.csv", index=False)
    print(f"\n[*] Saved {len(full)} predictions to cross_modal_real_predictions.csv")


if __name__ == "__main__":
    main()

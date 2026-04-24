#!/usr/bin/env python3
"""
Aggregate FT-Transformer attention across multiple folds (seed 42, folds 0-4).

Loads the raw attention tensors saved by ft_attention.py for each fold, computes
per-fold class means, then reports mean +/- std across folds and produces a
consolidated plot with error bars.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))
from interpretability.ft_attention import (  # noqa: E402
    TABULAR_FEATURES, FEATURE_LABELS,
)

REPO = Path("/home/tanguy/medical/alzheimer")
EXP = REPO / "experiments" / "multimodal_fusion"
# Post-reorg layout: raw attention tensors live in 01_ft_attention/raw_data,
# aggregated figures/CSVs go to 01_ft_attention/aggregated.
RAW = SCRIPT_DIR / "results_new" / "01_ft_attention" / "raw_data"
AGG = SCRIPT_DIR / "results_new" / "01_ft_attention" / "aggregated"
RESULTS = RAW


def per_fold_class_means(attn, labels):
    """Return (cn_mean, ad_mean) for last-layer CLS→features (avg over heads)."""
    # attn: (N, layers, heads, L, L)
    cls_last = attn[:, -1, :, 0, 1:].mean(axis=1)  # (N, 16)
    return cls_last[labels == 0].mean(axis=0), cls_last[labels == 1].mean(axis=0)


def per_fold_rollout_means(attn, labels):
    """Attention rollout (Abnar & Zuidema) aggregated per class."""
    N, num_layers, heads, L, _ = attn.shape
    avg = attn.mean(axis=2)
    I = np.eye(L)[None]
    rolled = np.broadcast_to(I, (N, L, L)).copy()
    for li in range(num_layers):
        A = (avg[:, li] + I) / 2.0
        A = A / A.sum(axis=-1, keepdims=True)
        rolled = np.einsum("nij,njk->nik", A, rolled)
    cls_roll = rolled[:, 0, 1:]  # (N, 16)
    return cls_roll[labels == 0].mean(axis=0), cls_roll[labels == 1].mean(axis=0)


def load_fold_labels(seed, fold, which):
    """Read labels for the fold's test set (regenerate splits from all_df)."""
    # Regenerate fold splits deterministically to match train_cv.py
    from sklearn.model_selection import StratifiedKFold
    train_df = pd.read_csv(EXP / "data/combined_trajectory/train.csv")
    val_df = pd.read_csv(EXP / "data/combined_trajectory/val.csv")
    test_df = pd.read_csv(EXP / "data/combined_trajectory/test.csv")
    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    labels = all_df["label"].values
    indices = np.arange(len(all_df))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for fi, (_, tst_idx) in enumerate(skf.split(indices, labels)):
        if fi == fold:
            test_fold_df = all_df.iloc[tst_idx].reset_index(drop=True)
            if which == "traj":
                return test_fold_df["label"].values.astype(int)
            # cn_ad: filter test fold to stable CN_AD subjects
            cn_ad_dir = EXP / "data" / "combined_cn_ad"
            stable_ids = set()
            for n in ["train.csv", "val.csv", "test.csv"]:
                p = cn_ad_dir / n
                if p.exists():
                    stable_ids |= set(pd.read_csv(p)["subject_id"].values)
            filtered = test_fold_df[test_fold_df["subject_id"].isin(stable_ids)]
            return filtered["label"].values.astype(int)
    raise ValueError(f"fold {fold} not found")


def aggregate(seed=42, folds=(0, 1, 2, 3, 4), variant="last_layer", test="traj"):
    cn_all, ad_all = [], []
    for fold in folds:
        attn_path = RESULTS / f"attn_seed{seed}_fold{fold}_{test}.npy"
        if not attn_path.exists():
            print(f"[skip] missing {attn_path.name}")
            continue
        attn = np.load(attn_path)
        labels = load_fold_labels(seed, fold, test)
        if variant == "last_layer":
            cn, ad = per_fold_class_means(attn, labels)
        else:
            cn, ad = per_fold_rollout_means(attn, labels)
        cn_all.append(cn)
        ad_all.append(ad)
        print(f"fold {fold}: CN mean first-3 features = {np.round(cn[:3], 4)}")
    cn_stack = np.stack(cn_all)  # (K, 16)
    ad_stack = np.stack(ad_all)
    return cn_stack, ad_stack


def plot_aggregate(cn_stack, ad_stack, out_path, title_suffix=""):
    cn_mean, cn_std = cn_stack.mean(axis=0), cn_stack.std(axis=0)
    ad_mean, ad_std = ad_stack.mean(axis=0), ad_stack.std(axis=0)
    diff_mean = (ad_stack - cn_stack).mean(axis=0)
    diff_std = (ad_stack - cn_stack).std(axis=0)
    # Signed consistency: number of folds with AD > CN (positive) vs AD < CN (negative)
    diff_signs = np.sign(ad_stack - cn_stack)
    agree_pos = (diff_signs > 0).sum(axis=0)
    agree_neg = (diff_signs < 0).sum(axis=0)
    # consistent = max(agree_pos, agree_neg)
    K = cn_stack.shape[0]

    # Order by |diff_mean| for the diff panel (most discriminative first)
    diff_order = np.argsort(-np.abs(diff_mean))
    # Order by total attention for the attention panel
    total_order = np.argsort(-(cn_mean + ad_mean))

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    # Panel 1: attention per feature per class
    y = np.arange(len(TABULAR_FEATURES))
    lbls = [FEATURE_LABELS[TABULAR_FEATURES[i]] for i in total_order]
    axes[0].barh(y - 0.2, cn_mean[total_order], 0.4, xerr=cn_std[total_order],
                 color="steelblue", label="CN", alpha=0.85, capsize=2)
    axes[0].barh(y + 0.2, ad_mean[total_order], 0.4, xerr=ad_std[total_order],
                 color="coral", label="AD-trajectory", alpha=0.85, capsize=2)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(lbls)
    axes[0].invert_yaxis()
    axes[0].axvline(1.0 / (len(TABULAR_FEATURES) + 1), color="gray", linestyle="--",
                    alpha=0.5, label="Uniform")
    axes[0].set_xlabel("Attention weight (mean +/- std across folds)")
    axes[0].set_title(f"FT-T CLS -> feature attention by class{title_suffix}")
    axes[0].legend(loc="lower right")

    # Panel 2: per-feature AD-CN diff with consistency annotation
    lbls_diff = [FEATURE_LABELS[TABULAR_FEATURES[i]] for i in diff_order]
    vals = diff_mean[diff_order]
    errs = diff_std[diff_order]
    colors = ["firebrick" if v > 0 else "steelblue" for v in vals]
    bars = axes[1].barh(y, vals, xerr=errs, color=colors, alpha=0.85, capsize=3)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(lbls_diff)
    axes[1].invert_yaxis()
    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].set_xlabel("Attention diff (AD - CN), mean +/- std across folds")
    axes[1].set_title(f"Class discrimination per feature ({K} folds)")

    # Annotate consistency (e.g., 4/5)
    for i, idx in enumerate(diff_order):
        k = max(agree_pos[idx], agree_neg[idx])
        axes[1].text(
            vals[i] + (0.001 if vals[i] >= 0 else -0.001),
            i,
            f"{k}/{K}",
            ha="left" if vals[i] >= 0 else "right",
            va="center",
            fontsize=8,
            color="gray",
        )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    # Also return summary dataframe
    df = pd.DataFrame({
        "feature": [FEATURE_LABELS[f] for f in TABULAR_FEATURES],
        "cn_mean": cn_mean,
        "cn_std": cn_std,
        "ad_mean": ad_mean,
        "ad_std": ad_std,
        "diff_mean": diff_mean,
        "diff_std": diff_std,
        "consistency": [f"{max(p, n)}/{K}" for p, n in zip(agree_pos, agree_neg)],
        "agree_pos": agree_pos,
        "agree_neg": agree_neg,
    })
    return df.sort_values("diff_mean", ascending=False)


def main():
    for variant in ("last_layer", "rollout"):
        for test in ("traj", "cn_ad"):
            print(f"\n=== variant={variant} test={test} ===")
            cn_stack, ad_stack = aggregate(
                seed=42, folds=(0, 1, 2, 3, 4), variant=variant, test=test
            )
            out_png = AGG / f"ft_attn_{variant}_AGG_seed42_{test}.png"
            df = plot_aggregate(
                cn_stack, ad_stack, out_png,
                title_suffix=f" (5 folds, seed 42, {test})",
            )
            csv_path = AGG / f"ft_attn_{variant}_AGG_seed42_{test}.csv"
            df.to_csv(csv_path, index=False)
            pd.set_option("display.float_format", lambda x: f"{x:.4f}")
            print(df[["feature", "diff_mean", "diff_std", "consistency"]].to_string(index=False))
            print(f"[saved] {out_png.name}")


if __name__ == "__main__":
    main()

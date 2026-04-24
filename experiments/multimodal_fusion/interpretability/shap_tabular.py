#!/usr/bin/env python3
"""
SHAP analysis of the FT-Transformer tabular branch.

Mirrors ft_attention.py exactly (same paper-grade checkpoints, same split
regeneration, same saved scaler) so that the SHAP ranking can be compared
one-to-one against the attention rollout ranking from that script.

Target: tab_classifier(tabular_encoder(x))  -> (N, 2) logits.
We use GradientShap (shap.GradientExplainer), which is fast on 16 continuous
features and compatible with PyTorch models containing transformer blocks.

Output per fold:
    shap_values_seed{S}_fold{F}_{test}.npy    # (N, 16, 2) signed SHAP per class
    shap_seed{S}_fold{F}_{test}.csv           # ranked feature table

Aggregated across folds:
    shap_AGG_seed{S}_{test}.csv               # mean/std/consistency
    shap_AGG_seed{S}_{test}.png               # bar plot
    compare_shap_vs_rollout_seed{S}_{test}.csv  # Spearman + side-by-side ranks

Usage:
    python shap_tabular.py --seed 42
"""

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch
import torch.nn as nn

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))
from model import MultiModalFusion  # noqa: E402
from ft_attention import (  # noqa: E402
    TABULAR_FEATURES,
    FEATURE_LABELS,
    build_paper_model,
    regenerate_fold_splits,
)

REPO = Path("/home/tanguy/medical/alzheimer")
EXP = REPO / "experiments" / "multimodal_fusion"


class TabularHead(nn.Module):
    """Wrap the tabular branch: x (B, 16) -> logits (B, 2)."""

    def __init__(self, full_model):
        super().__init__()
        self.encoder = full_model.tabular_encoder
        self.clf = full_model.tab_classifier

    def forward(self, x):
        return self.clf(self.encoder(x))


def compute_shap(head, X_bg, X_test, device, n_samples=64):
    """GradientExplainer SHAP for each class.

    Returns array (N, 16, 2): signed contribution per feature per class logit.
    """
    head.eval()
    bg_t = torch.from_numpy(X_bg).float().to(device)
    test_t = torch.from_numpy(X_test).float().to(device)
    explainer = shap.GradientExplainer(head, bg_t)
    # shap 0.49 returns either a list (older API) or an ndarray (N, F, C)
    sv = explainer.shap_values(test_t, nsamples=n_samples)
    if isinstance(sv, list):
        sv = np.stack(sv, axis=-1)  # -> (N, F, C)
    return sv  # (N, 16, 2)


def aggregate_folds(per_fold_dfs, feature_order):
    """Aggregate per-fold (signed diff AD-CN) like ft_attention does for rollout."""
    # per_fold_dfs: list of DataFrames with columns [feature, diff]
    mat = np.stack(
        [df.set_index("feature").loc[feature_order]["diff"].values for df in per_fold_dfs],
        axis=0,
    )  # (n_folds, 16)
    mean = mat.mean(axis=0)
    std = mat.std(axis=0)
    signs = np.sign(mat)
    agree_pos = (signs > 0).sum(axis=0)
    agree_neg = (signs < 0).sum(axis=0)
    consistency = np.maximum(agree_pos, agree_neg)
    return pd.DataFrame({
        "feature": feature_order,
        "label": [FEATURE_LABELS[f] for f in feature_order],
        "diff_mean": mean,
        "diff_std": std,
        "agree_pos": agree_pos,
        "agree_neg": agree_neg,
        "consistency": [f"{c}/{mat.shape[0]}" for c in consistency],
    })


def plot_agg(df_agg, out_path, title):
    order = np.argsort(-df_agg["diff_mean"].values)
    df_s = df_agg.iloc[order].reset_index(drop=True)
    colors = ["coral" if v > 0 else "steelblue" for v in df_s["diff_mean"]]
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(range(len(df_s)), df_s["diff_mean"], xerr=df_s["diff_std"],
            color=colors, ecolor="gray", capsize=3)
    ax.set_yticks(range(len(df_s)))
    ax.set_yticklabels(df_s["label"])
    ax.invert_yaxis()
    ax.axvline(0, color="k", linewidth=0.5)
    ax.set_xlabel("SHAP(AD) - SHAP(CN)  (mean ± std across folds)")
    ax.set_title(title)
    for i, c in enumerate(df_s["consistency"]):
        ax.text(0.0, i, f"  {c}", va="center", ha="left", fontsize=8, color="dimgray")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def compare_with_rollout(shap_agg_df, rollout_csv, out_path):
    """Spearman + side-by-side feature ranking."""
    from scipy.stats import spearmanr
    ro = pd.read_csv(rollout_csv)
    merged = shap_agg_df[["feature", "label", "diff_mean"]].rename(
        columns={"diff_mean": "shap_diff"}
    ).merge(
        ro[["feature", "diff_mean"]].rename(columns={"diff_mean": "rollout_diff"}),
        on="feature",
    )
    # Rank by descending diff (most AD-favoring at top)
    merged["shap_rank"] = merged["shap_diff"].rank(ascending=False).astype(int)
    merged["rollout_rank"] = merged["rollout_diff"].rank(ascending=False).astype(int)
    rho_signed, p_signed = spearmanr(merged["shap_diff"], merged["rollout_diff"])
    rho_abs, p_abs = spearmanr(merged["shap_diff"].abs(), merged["rollout_diff"].abs())
    merged = merged.sort_values("shap_rank").reset_index(drop=True)
    with open(out_path, "w") as f:
        f.write(f"# Spearman(signed diff):  rho={rho_signed:.3f}  p={p_signed:.2e}\n")
        f.write(f"# Spearman(|diff|):        rho={rho_abs:.3f}  p={p_abs:.2e}\n")
        merged.to_csv(f, index=False)
    print(f"    Spearman(signed): rho={rho_signed:+.3f}  p={p_signed:.2e}")
    print(f"    Spearman(|diff|): rho={rho_abs:+.3f}  p={p_abs:.2e}")
    return rho_signed, rho_abs


def run_one_fold(seed, fold, device, out_dir):
    ckpt_path = EXP / "cv_results" / f"seed_{seed}" / f"fold_{fold}" / "model.pth"
    scaler_path = EXP / "cv_results" / f"seed_{seed}" / f"fold_{fold}" / "scaler.pkl"

    train_df, val_df, test_df = regenerate_fold_splits(seed, fold)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    # cn_ad subset (stable CN/AD only)
    cn_ad_dir = EXP / "data" / "combined_cn_ad"
    stable_ids = set()
    for n in ["train.csv", "val.csv", "test.csv"]:
        p = cn_ad_dir / n
        if p.exists():
            stable_ids |= set(pd.read_csv(p)["subject_id"].values)
    cn_ad_df = test_df[test_df["subject_id"].isin(stable_ids)].reset_index(drop=True)

    def _X(df):
        return scaler.transform(
            df[TABULAR_FEATURES].apply(lambda c: c.fillna(c.median())).values.astype(np.float32)
        )

    X_train = _X(train_df)
    X_traj = _X(test_df)
    X_cn_ad = _X(cn_ad_df)

    # Model
    model = build_paper_model()
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict, strict=False)
    head = TabularHead(model).to(device)

    # Background: 100 train samples (enough for 16-feature model)
    rng = np.random.default_rng(seed * 1000 + fold)
    bg_idx = rng.choice(len(X_train), size=min(100, len(X_train)), replace=False)
    X_bg = X_train[bg_idx]

    results = {}
    for test_name, X, y in [("traj", X_traj, test_df["label"].values.astype(int)),
                            ("cn_ad", X_cn_ad, cn_ad_df["label"].values.astype(int))]:
        print(f"  [fold {fold}] {test_name}: N={len(y)}  computing SHAP...")
        sv = compute_shap(head, X_bg, X, device)  # (N, 16, 2)
        np.save(out_dir / f"shap_values_seed{seed}_fold{fold}_{test_name}.npy", sv)

        # Signed contribution to AD-class logit minus CN-class logit, averaged over samples
        # This is directly comparable to attention rollout's AD-CN diff.
        diff_per_sample = sv[:, :, 1] - sv[:, :, 0]  # (N, 16) favoring AD class
        # Split by true class for ranking tables
        cn_mask, ad_mask = (y == 0), (y == 1)
        cn_mean_abs = np.abs(sv[cn_mask, :, 0]).mean(axis=0)  # (16,)
        ad_mean_abs = np.abs(sv[ad_mask, :, 1]).mean(axis=0)
        # Our primary metric: mean signed (SHAP_AD - SHAP_CN) over ALL test samples
        diff_mean = diff_per_sample.mean(axis=0)

        df = pd.DataFrame({
            "feature": TABULAR_FEATURES,
            "label": [FEATURE_LABELS[f] for f in TABULAR_FEATURES],
            "shap_ad_abs_on_ad": ad_mean_abs,
            "shap_cn_abs_on_cn": cn_mean_abs,
            "diff": diff_mean,
        }).sort_values("diff", ascending=False)
        df.to_csv(out_dir / f"shap_seed{seed}_fold{fold}_{test_name}.csv", index=False)
        results[test_name] = df
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu")
    print(f"[*] Device: {device}")

    out_per = SCRIPT_DIR / "results_new" / "04_shap" / "per_fold"
    out_agg = SCRIPT_DIR / "results_new" / "04_shap" / "aggregated"
    out_per.mkdir(exist_ok=True, parents=True)
    out_agg.mkdir(exist_ok=True, parents=True)

    per_fold = {"traj": [], "cn_ad": []}
    for fold in args.folds:
        res = run_one_fold(args.seed, fold, device, out_per)
        for k, df in res.items():
            per_fold[k].append(df)

    rollout_dir = SCRIPT_DIR / "results_new" / "01_ft_attention" / "aggregated"
    for test_name, dfs in per_fold.items():
        if not dfs:
            continue
        agg = aggregate_folds(dfs, TABULAR_FEATURES)
        agg_csv = out_agg / f"shap_AGG_seed{args.seed}_{test_name}.csv"
        agg.to_csv(agg_csv, index=False)
        plot_agg(
            agg,
            out_agg / f"shap_AGG_seed{args.seed}_{test_name}.png",
            f"SHAP feature contribution (AD - CN), seed={args.seed}, {test_name}",
        )
        print(f"[*] Aggregated {test_name}: {agg_csv}")
        rollout_csv = rollout_dir / f"ft_attn_rollout_AGG_seed{args.seed}_{test_name}.csv"
        if rollout_csv.exists():
            print(f"[*] Comparing to rollout ({test_name}):")
            compare_with_rollout(
                agg, rollout_csv,
                out_agg / f"compare_shap_vs_rollout_seed{args.seed}_{test_name}.csv",
            )
        else:
            print(f"[!] Rollout CSV not found: {rollout_csv}")

    print("[*] Done.")


if __name__ == "__main__":
    main()

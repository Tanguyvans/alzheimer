#!/usr/bin/env python3
"""
Single-panel SHAP feature-importance figure for the paper.

Plots mean |SHAP| (AD-class logit) across 5 folds, ranked from most to least
important. Bars are color-coded by clinical category (cognitive tests vs.
demographics/risk factors) to make the neuropsychological battery story
visually obvious.

We do NOT plot signed SHAP differences in the paper figure: sign consistency
across folds was only ~3/5 for most features, so only magnitude claims are
well-supported.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))
from interpretability.ft_attention import TABULAR_FEATURES, FEATURE_LABELS  # noqa: E402

SHAP_DIR = SCRIPT_DIR / "results_new" / "04_shap" / "per_fold"
OUT_DIR = SCRIPT_DIR / "results_new" / "04_shap" / "aggregated"

COGNITIVE_FEATURES = {
    "CATANIMSC", "TRAASCOR", "TRABSCOR",
    "DSPANFOR", "DSPANBAC", "BNTTOTAL",
}


def aggregate_shap_magnitude(seed, folds, test):
    """Return per-feature mean |SHAP_AD| and std across folds.

    Each fold contributes one value: the mean across its test samples.
    Cross-fold std quantifies stability of the importance ranking.
    """
    per_fold = []
    for f in folds:
        sv = np.load(SHAP_DIR / f"shap_values_seed{seed}_fold{f}_{test}.npy")  # (N, 16, 2)
        per_fold.append(np.abs(sv[:, :, 1]).mean(axis=0))  # (16,)
    stack = np.stack(per_fold, axis=0)  # (K, 16)
    return stack.mean(axis=0), stack.std(axis=0), stack


def plot_single_panel(means, stds, out_path, title):
    order = np.argsort(-means)
    labels = [FEATURE_LABELS[TABULAR_FEATURES[i]] for i in order]
    is_cog = [TABULAR_FEATURES[i] in COGNITIVE_FEATURES for i in order]
    vals = means[order]
    errs = stds[order]

    colors = ["#c62828" if c else "#90a4ae" for c in is_cog]

    fig, ax = plt.subplots(figsize=(7, 5))
    y = np.arange(len(TABULAR_FEATURES))
    ax.barh(y, vals, xerr=errs, color=colors, alpha=0.88, capsize=3,
            edgecolor="black", linewidth=0.4)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel(r"Mean $|\mathrm{SHAP}|$ on AD logit (across 5 folds)", fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.grid(axis="x", alpha=0.3, linestyle=":")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    red_patch = plt.Rectangle((0, 0), 1, 1, color="#c62828", alpha=0.88)
    gray_patch = plt.Rectangle((0, 0), 1, 1, color="#90a4ae", alpha=0.88)
    ax.legend([red_patch, gray_patch],
              ["Cognitive tests", "Demographics / risk factors"],
              loc="lower right", fontsize=9, frameon=False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[saved] {out_path}")


def main():
    for test, label in [("traj", "AD-trajectory"), ("cn_ad", "CN vs AD")]:
        means, stds, _ = aggregate_shap_magnitude(42, (0, 1, 2, 3, 4), test)
        out = OUT_DIR / f"shap_importance_AGG_seed42_{test}_paper.png"
        plot_single_panel(
            means, stds, out,
            f"Tabular feature importance via SHAP (5 folds, {label})",
        )


if __name__ == "__main__":
    main()

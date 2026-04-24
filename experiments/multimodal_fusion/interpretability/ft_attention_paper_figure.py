#!/usr/bin/env python3
"""
Single-panel FT-T attention figure for the paper.

Keeps only the discrimination panel (AD-CN diff with consistency annotations)
which carries the main story. Aggregated over 5 folds (seed 42).
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))
from interpretability.ft_attention import TABULAR_FEATURES, FEATURE_LABELS  # noqa: E402
from interpretability.aggregate_ft_attention import aggregate  # noqa: E402

RESULTS = SCRIPT_DIR / "results_new" / "01_ft_attention" / "aggregated"


def plot_single_panel(cn_stack, ad_stack, out_path, variant, title):
    cn_mean = cn_stack.mean(axis=0)
    ad_mean = ad_stack.mean(axis=0)
    diff_mean = (ad_stack - cn_stack).mean(axis=0)
    diff_std = (ad_stack - cn_stack).std(axis=0)
    diff_signs = np.sign(ad_stack - cn_stack)
    agree_pos = (diff_signs > 0).sum(axis=0)
    agree_neg = (diff_signs < 0).sum(axis=0)
    K = cn_stack.shape[0]

    # Order by |diff_mean| for discriminative features first
    diff_order = np.argsort(-np.abs(diff_mean))

    fig, ax = plt.subplots(figsize=(7, 5))
    y = np.arange(len(TABULAR_FEATURES))
    lbls = [FEATURE_LABELS[TABULAR_FEATURES[i]] for i in diff_order]
    vals = diff_mean[diff_order]
    errs = diff_std[diff_order]
    colors = ["#c62828" if v > 0 else "#1976d2" for v in vals]
    ax.barh(y, vals, xerr=errs, color=colors, alpha=0.88, capsize=3,
            edgecolor="black", linewidth=0.4)

    ax.set_yticks(y)
    ax.set_yticklabels(lbls, fontsize=10)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel(r"$\Delta$ attention (AD $-$ CN), mean $\pm$ std across 5 folds",
                  fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.grid(axis="x", alpha=0.3, linestyle=":")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Annotate fold consistency (e.g., 5/5) next to each bar
    x_pad = max(abs(vals)) * 0.04
    for i, idx in enumerate(diff_order):
        k = max(agree_pos[idx], agree_neg[idx])
        v = vals[i]
        e = errs[i]
        ax.text(
            (v + e + x_pad) if v >= 0 else (v - e - x_pad),
            i,
            f"{k}/{K}",
            ha="left" if v >= 0 else "right",
            va="center",
            fontsize=8,
            color="#555555",
        )

    # Legend color key
    red_patch = plt.Rectangle((0, 0), 1, 1, color="#c62828", alpha=0.88)
    blue_patch = plt.Rectangle((0, 0), 1, 1, color="#1976d2", alpha=0.88)
    ax.legend([red_patch, blue_patch],
              ["Attended more for AD", "Attended more for CN"],
              loc="lower right", fontsize=9, frameon=False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[saved] {out_path}")


def main():
    for variant in ("rollout", "last_layer"):
        cn_stack, ad_stack = aggregate(
            seed=42, folds=(0, 1, 2, 3, 4), variant=variant, test="traj"
        )
        title = (
            f"FT-Transformer feature discrimination (5 folds, AD-trajectory)"
            if variant == "rollout" else
            f"FT-Transformer CLS-to-feature attention (5 folds, AD-trajectory)"
        )
        out = RESULTS / f"ft_attn_{variant}_AGG_seed42_traj_paper.png"
        plot_single_panel(cn_stack, ad_stack, out, variant, title)


if __name__ == "__main__":
    main()

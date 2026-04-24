#!/usr/bin/env python3
"""
Single compact cross-modal figure for the paper.

One figure sized for a 2-column paper (~7 inches wide, 3 inches tall).
Shows per-class accuracy of the 3 internal classifiers side by side,
with annotations highlighting the "complementary bias" story.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
CSV = SCRIPT_DIR / "results_new" / "03_cross_modal" / "cross_modal_predictions.csv"
OUT_DIR = SCRIPT_DIR / "results_new" / "03_cross_modal"


def main():
    df = pd.read_csv(CSV)
    cn = df[df["label"] == 0]
    ad = df[df["label"] == 1]

    heads = ["MRI-only", "Tab-only", "Fused"]
    cn_acc = [100 * (cn["pred_mri"] == 0).mean(),
              100 * (cn["pred_tab"] == 0).mean(),
              100 * (cn["pred_fused"] == 0).mean()]
    ad_acc = [100 * (ad["pred_mri"] == 1).mean(),
              100 * (ad["pred_tab"] == 1).mean(),
              100 * (ad["pred_fused"] == 1).mean()]

    # Disagreement resolution: when modalities disagree, fused correct rate
    dis = df[df["pred_mri"] != df["pred_tab"]]
    fused_dis = 100 * (dis["pred_fused"] == dis["label"]).mean()
    n_dis = len(dis)
    frac_dis = 100 * n_dis / len(df)

    fig, ax = plt.subplots(figsize=(7, 3.2))

    x = np.arange(len(heads))
    width = 0.35
    bars1 = ax.bar(x - width / 2, cn_acc, width, label=f"CN (N={len(cn)})",
                    color="#1976d2", alpha=0.9)
    bars2 = ax.bar(x + width / 2, ad_acc, width, label=f"AD (N={len(ad)})",
                    color="#c62828", alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(heads, fontsize=10)
    ax.set_ylabel("Accuracy (%)", fontsize=10)
    ax.set_ylim(0, 110)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.grid(axis="y", alpha=0.3, linestyle=":")
    ax.legend(loc="lower center", fontsize=9, ncol=2, frameon=False,
              bbox_to_anchor=(0.5, 1.0))

    for b in list(bars1) + list(bars2):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 2,
                f"{b.get_height():.0f}", ha="center", fontsize=9,
                fontweight="bold")

    # Inline annotation summarizing the key finding
    fig.text(
        0.5, -0.02,
        f"When MRI and Tab disagree ({n_dis}/{len(df)} = {frac_dis:.0f}% of cases),"
        f" Fused picks correctly {fused_dis:.0f}% of the time.",
        ha="center", fontsize=9, style="italic",
    )

    plt.tight_layout()
    out = OUT_DIR / "fig_cross_modal_compact.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()

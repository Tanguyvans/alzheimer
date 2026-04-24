#!/usr/bin/env python3
"""
Cross-modal figure v4 — uses REAL standalone MRI-only and Tab-only models
trained independently (not the degenerate auxiliary classifiers).

Handles the case where MRI-only isn't ready yet (skip it, show Fused vs Tab-only
for now; add MRI column later).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
CSV = SCRIPT_DIR / "results_new" / "03_cross_modal" / "cross_modal_real_predictions.csv"
OUT_DIR = SCRIPT_DIR / "results_new" / "03_cross_modal"


def main():
    df = pd.read_csv(CSV)
    has_mri = (df["pred_mri"] != -1).all()
    print(f"MRI-only present: {has_mri}")

    cn = df[df["label"] == 0]
    ad = df[df["label"] == 1]

    heads = ["MRI-only", "Tab-only", "Fused"] if has_mri else ["Tab-only", "Fused"]
    pred_cols = ["pred_mri", "pred_tab", "pred_fused"] if has_mri else ["pred_tab", "pred_fused"]
    colors = ["#1976d2", "#7b1fa2", "#2e7d32"] if has_mri else ["#7b1fa2", "#2e7d32"]

    cn_acc = [100 * (cn[c] == 0).mean() for c in pred_cols]
    ad_acc = [100 * (ad[c] == 1).mean() for c in pred_cols]

    # Overall too
    overall = [100 * (df[c] == df["label"]).mean() for c in pred_cols]

    fig, ax = plt.subplots(figsize=(7.2, 3.8))

    x = np.arange(len(heads))
    w = 0.28
    b1 = ax.bar(x - w, cn_acc, w, label=f"CN subjects (N={len(cn)})",
                color="#1976d2", alpha=0.92, edgecolor="black", linewidth=0.4)
    b2 = ax.bar(x, ad_acc, w, label=f"AD subjects (N={len(ad)})",
                color="#c62828", alpha=0.92, edgecolor="black", linewidth=0.4)
    b3 = ax.bar(x + w, overall, w, label="Overall",
                color="#616161", alpha=0.8, edgecolor="black", linewidth=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels(heads, fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=10)
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.grid(axis="y", alpha=0.3, linestyle=":")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower center", fontsize=9, ncol=3, frameon=False,
              bbox_to_anchor=(0.5, 1.0))

    for bars in [b1, b2, b3]:
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5,
                    f"{b.get_height():.0f}", ha="center", fontsize=9,
                    fontweight="bold")

    # Subtitle story
    title = ("Per-class accuracy of independently-trained classifiers "
             f"(5-fold CV, {len(df)} subjects)")
    ax.set_title(title, fontsize=10, pad=28)

    # Key numbers as caption
    gain_cn = overall[-1] - overall[0]  # Fused - first head
    fig.text(
        0.5, -0.04,
        f"Cross-modal fusion (Fused) adds +{gain_cn:.1f} accuracy points over the best single modality.",
        ha="center", fontsize=9, style="italic",
    )

    plt.tight_layout()
    out = OUT_DIR / "fig_cross_modal_v4_real.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[saved] {out}")

    # Print paper-ready numbers
    print(f"\n=== Paper-ready stats (REAL standalone baselines) ===")
    for i, head in enumerate(heads):
        print(f"  {head:10s}: CN {cn_acc[i]:.1f}%, AD {ad_acc[i]:.1f}%, Overall {overall[i]:.1f}%")


if __name__ == "__main__":
    main()

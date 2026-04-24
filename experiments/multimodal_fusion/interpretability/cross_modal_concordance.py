#!/usr/bin/env python3
"""
Cross-modal concordance figure — shows NEW information not in Table III.

Uses the 3-way predictions (MRI-only, Tab-only, Fused) on fold 0 (1213 subjects)
and builds a compact figure that answers: "when MRI and Tab disagree, how well
does the fused classifier resolve it?"

This is the interpretability complement to the per-class accuracy in Table III.
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
    df["agree"] = df["pred_mri"] == df["pred_tab"]
    df["correct_fused"] = df["pred_fused"] == df["label"]
    df["correct_mri"] = df["pred_mri"] == df["label"]
    df["correct_tab"] = df["pred_tab"] == df["label"]
    df["correct_oracle"] = df["correct_mri"] | df["correct_tab"]

    total = len(df)
    n_agree = df["agree"].sum()
    n_disagree = total - n_agree

    # Break down in disagreement
    disagree = df[~df["agree"]]
    correct_in_disagree = disagree["correct_fused"].sum()
    oracle_in_disagree = disagree["correct_oracle"].sum()

    # Break down in agreement
    agree = df[df["agree"]]
    correct_in_agree_both = ((agree["correct_mri"]) & agree["correct_fused"]).sum()
    wrong_in_agree = ((~agree["correct_mri"]) & (~agree["correct_fused"])).sum()
    rescued_in_agree = ((~agree["correct_mri"]) & agree["correct_fused"]).sum()

    print(f"Total: {total}")
    print(f"  Agreement: {n_agree} ({100 * n_agree / total:.1f}%)")
    print(f"    Both correct + fused correct: {correct_in_agree_both}")
    print(f"    Both wrong (fused follows): {wrong_in_agree}")
    print(f"    Fused rescues consensus: {rescued_in_agree}")
    print(f"  Disagreement: {n_disagree} ({100 * n_disagree / total:.1f}%)")
    print(f"    Fused correct: {correct_in_disagree} ({100 * correct_in_disagree / n_disagree:.1f}%)")
    print(f"    Oracle upper bound: {oracle_in_disagree} ({100 * oracle_in_disagree / n_disagree:.1f}%)")

    # --- Figure: 2 panels side-by-side ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))

    # Panel A: categorical stacked bar
    # Category 1: Agreement (both modalities vote same)
    # Category 2: Disagreement (MRI != Tab)
    # Within each, show: fused correct (green) vs fused wrong (red)
    ax = axes[0]
    cats = ["MRI ↔ Tab agree", "MRI ↔ Tab disagree"]
    correct_counts = [
        int(agree["correct_fused"].sum()),
        int(disagree["correct_fused"].sum()),
    ]
    wrong_counts = [
        int((~agree["correct_fused"]).sum()),
        int((~disagree["correct_fused"]).sum()),
    ]
    y = np.arange(len(cats))
    ax.barh(y, correct_counts, color="#2e7d32", alpha=0.9,
            label="Fused correct", edgecolor="black", linewidth=0.4)
    ax.barh(y, wrong_counts, left=correct_counts, color="#c62828", alpha=0.85,
            label="Fused wrong", edgecolor="black", linewidth=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels(cats, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("N subjects (fold 0)", fontsize=10)
    ax.set_xlim(0, total + 30)
    ax.grid(axis="x", alpha=0.3, linestyle=":")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    for i, (c, w) in enumerate(zip(correct_counts, wrong_counts)):
        n = c + w
        acc = 100 * c / n if n else 0
        ax.text(n + 10, i, f"{c}/{n} ({acc:.0f}%)", va="center", fontsize=9)
    ax.set_title("A. Fused accuracy by agreement regime",
                 fontsize=10, loc="left", pad=6)

    # Panel B: disagreement resolution — per class
    # For CN and AD subjects in disagreement, how often does Fused pick correctly?
    ax = axes[1]
    cn_dis = disagree[disagree["label"] == 0]
    ad_dis = disagree[disagree["label"] == 1]
    groups = [
        ("CN", 100 * cn_dis["correct_fused"].mean() if len(cn_dis) else 0,
         100 * cn_dis["correct_oracle"].mean() if len(cn_dis) else 0,
         len(cn_dis)),
        ("AD", 100 * ad_dis["correct_fused"].mean() if len(ad_dis) else 0,
         100 * ad_dis["correct_oracle"].mean() if len(ad_dis) else 0,
         len(ad_dis)),
    ]
    x = np.arange(2)
    width = 0.35
    fused_vals = [g[1] for g in groups]
    oracle_vals = [g[2] for g in groups]
    ns = [g[3] for g in groups]
    b1 = ax.bar(x - width / 2, fused_vals, width,
                color="#2e7d32", alpha=0.9, label="Fused classifier",
                edgecolor="black", linewidth=0.4)
    b2 = ax.bar(x + width / 2, oracle_vals, width,
                color="#9e9e9e", alpha=0.7, label="Oracle upper bound",
                edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{g[0]} subjects\n(N_dis={g[3]})" for g in groups],
        fontsize=9,
    )
    ax.set_ylabel("Accuracy on disagreement (%)", fontsize=10)
    ax.set_ylim(0, 110)
    ax.grid(axis="y", alpha=0.3, linestyle=":")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower center", fontsize=8, frameon=False)
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5,
                f"{b.get_height():.0f}", ha="center", fontsize=9)
    ax.set_title("B. Disagreement resolution per class",
                 fontsize=10, loc="left", pad=6)

    plt.tight_layout()
    out = OUT_DIR / "fig_cross_modal_concordance.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Cross-modal figure v3 : version horizontale ultra-lisible.

Shows per-class accuracy as horizontal bars, grouped by true class.
The visual message is immediate: each single modality has one big
failure class; only the fused classifier handles both.
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

    # Accuracies
    rows = [
        # (category, head, accuracy, color, n)
        ("CN", "Fused",   100 * (cn["pred_fused"] == 0).mean(), "#2e7d32", len(cn)),
        ("CN", "MRI-only",100 * (cn["pred_mri"]   == 0).mean(), "#1976d2", len(cn)),
        ("CN", "Tab-only",100 * (cn["pred_tab"]   == 0).mean(), "#90caf9", len(cn)),
        ("AD", "Fused",   100 * (ad["pred_fused"] == 1).mean(), "#2e7d32", len(ad)),
        ("AD", "Tab-only",100 * (ad["pred_tab"]   == 1).mean(), "#d32f2f", len(ad)),
        ("AD", "MRI-only",100 * (ad["pred_mri"]   == 1).mean(), "#ef9a9a", len(ad)),
    ]

    fig, ax = plt.subplots(figsize=(7, 3.6))

    y_positions = []
    labels = []
    for i, (cat, head, acc, color, n) in enumerate(rows):
        # Group with a gap between CN and AD blocks
        y = -(i if i < 3 else i + 0.8)
        y_positions.append(y)
        labels.append(f"{head}")
        bar = ax.barh(y, acc, height=0.7, color=color, alpha=0.92,
                      edgecolor="black", linewidth=0.5)
        ax.text(acc + 1.5, y, f"{acc:.0f}%", va="center", fontsize=10,
                fontweight="bold")

    # Group headers
    ax.text(-3, -1, "CN subjects\n(N=4747)", ha="right", va="center",
            fontsize=11, fontweight="bold", color="#1976d2")
    ax.text(-3, -4.8, "AD subjects\n(N=1318)", ha="right", va="center",
            fontsize=11, fontweight="bold", color="#c62828")

    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlim(0, 115)
    ax.set_xlabel("Accuracy (%)", fontsize=10)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.grid(axis="x", alpha=0.3, linestyle=":")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Disagreement story in caption-like text
    dis = df[df["pred_mri"] != df["pred_tab"]]
    fused_dis = 100 * (dis["pred_fused"] == dis["label"]).mean()
    n_dis = len(dis)
    ax.set_title(
        "Internal classifier accuracy — MRI and Tabular auxiliary heads fail\n"
        "on opposite classes, Fused combines both (5-fold CV, 6065 subjects)",
        fontsize=10, loc="left", pad=10,
    )

    fig.text(
        0.5, -0.06,
        f"When MRI and Tab disagree ({n_dis}/{len(df)} cases), "
        f"Fused picks the correct class {fused_dis:.0f}% of the time.",
        ha="center", fontsize=9, style="italic",
    )

    plt.tight_layout()
    out = OUT_DIR / "fig_cross_modal_v3.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()

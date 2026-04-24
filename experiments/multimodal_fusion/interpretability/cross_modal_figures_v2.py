#!/usr/bin/env python3
"""
Cross-modal analysis — clearer paper-ready figures.

Reads the cached predictions CSV produced by cross_modal_analysis.py and
generates two compact, self-explanatory figures:

  fig_cross_modal_headline.png:
    Panel A: per-class accuracy of the 3 internal classifiers (grouped bars)
    Panel B: disagreement resolution rate (how often fused picks the correct
             modality when MRI and Tab disagree) — split by true class

  fig_cross_modal_casebreak.png:
    2x2 matrix broken down by TRUE LABEL x (MRI_pred, Tab_pred) showing how
    the fused classifier resolves each combination (acc inside each cell).
    Makes the "adaptive modality selector" argument visually concrete.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS = SCRIPT_DIR / "results_new" / "03_cross_modal"
CSV = RESULTS / "cross_modal_predictions.csv"


def load_df():
    df = pd.read_csv(CSV)
    df["agree"] = df["pred_mri"] == df["pred_tab"]
    df["correct_fused"] = df["pred_fused"] == df["label"]
    df["correct_mri"] = df["pred_mri"] == df["label"]
    df["correct_tab"] = df["pred_tab"] == df["label"]
    return df


def plot_headline(df, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # ---- Panel A : per-class accuracy of the 3 heads ----
    ax = axes[0]
    heads = ["MRI-only", "Tab-only", "Fused"]
    pred_cols = ["pred_mri", "pred_tab", "pred_fused"]

    cn_acc = [100 * ((df[df["label"] == 0][c] == 0).mean()) for c in pred_cols]
    ad_acc = [100 * ((df[df["label"] == 1][c] == 1).mean()) for c in pred_cols]

    x = np.arange(len(heads))
    width = 0.35
    bars1 = ax.bar(x - width / 2, cn_acc, width, label="CN subjects (N=4747)",
                   color="#1976d2", alpha=0.85)
    bars2 = ax.bar(x + width / 2, ad_acc, width, label="AD subjects (N=1318)",
                   color="#c62828", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(heads, fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_ylim(0, 105)
    ax.set_title("A. Per-class accuracy of internal classifiers\n"
                 "MRI and Tab are biased opposite ways — only Fused balances both",
                 fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower right", fontsize=10)
    ax.axhline(100, color="gray", linewidth=0.5, linestyle=":")

    for b in list(bars1) + list(bars2):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5,
                f"{b.get_height():.0f}%", ha="center", fontsize=10,
                fontweight="bold" if b in bars1[-1:] + bars2[-1:] else "normal")

    # ---- Panel B : disagreement resolution ----
    ax = axes[1]
    # For CN and AD separately, compute fused accuracy when MRI and Tab disagree.
    def stats(label):
        sub = df[df["label"] == label]
        agree = sub["agree"].mean() * 100
        disag = sub[~sub["agree"]]
        fused_acc_disag = 100 * disag["correct_fused"].mean() if len(disag) else 0
        # Oracle = pick correct modality on each disagreement = (mri correct) or (tab correct)
        oracle = ((disag["correct_mri"]) | (disag["correct_tab"])).mean() * 100
        return agree, fused_acc_disag, oracle, len(disag)

    agree_cn, fused_dis_cn, oracle_cn, n_dis_cn = stats(0)
    agree_ad, fused_dis_ad, oracle_ad, n_dis_ad = stats(1)

    categories = ["CN subjects", "AD subjects"]
    fused_accs = [fused_dis_cn, fused_dis_ad]
    oracles = [oracle_cn, oracle_ad]
    ns = [n_dis_cn, n_dis_ad]

    x = np.arange(len(categories))
    width = 0.35
    b1 = ax.bar(x - width / 2, fused_accs, width, label="Fused classifier",
                color="#2e7d32", alpha=0.85)
    b2 = ax.bar(x + width / 2, oracles, width, label="Oracle upper bound",
                color="#bdbdbd", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{c}\n(N_disagree={n})" for c, n in zip(categories, ns)],
        fontsize=10,
    )
    ax.set_ylim(0, 105)
    ax.set_ylabel("Accuracy (%) on disagreement cases", fontsize=11)
    ax.set_title("B. When MRI and Tab disagree, how often does Fused pick correctly?\n"
                 "Fused approaches the oracle upper bound → acts as adaptive selector",
                 fontsize=11)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5,
                f"{b.get_height():.0f}%", ha="center", fontsize=10)

    plt.suptitle(
        "Cross-modal agreement (5 folds, 6065 subjects with cross-validated predictions)",
        fontsize=12, y=1.02,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close()
    print(f"[saved] {out_path.name}")


def plot_casebreak(df, out_path):
    """
    For each TRUE label (CN / AD), show the fused-classifier accuracy in each
    (MRI_pred, Tab_pred) combination. Cell text: N subjects + fused accuracy.

    This makes the "adaptive modality selector" visual:
      - Cells on the anti-diagonal (MRI and Tab disagree) show how fusion resolves
      - Cells on the diagonal (both agree) show how fusion handles consensus
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    for ax_i, (label, name) in enumerate([(0, "CN subjects (true label = CN)"),
                                           (1, "AD subjects (true label = AD)")]):
        sub = df[df["label"] == label]
        cell_n = np.zeros((2, 2), dtype=int)
        cell_acc = np.zeros((2, 2))  # fused accuracy within each cell
        for mri_pred in (0, 1):
            for tab_pred in (0, 1):
                mask = (sub["pred_mri"] == mri_pred) & (sub["pred_tab"] == tab_pred)
                cell_n[mri_pred, tab_pred] = mask.sum()
                if mask.sum() > 0:
                    cell_acc[mri_pred, tab_pred] = 100 * sub[mask]["correct_fused"].mean()

        ax = axes[ax_i]
        # Use green intensity for fused accuracy (darker = better)
        im = ax.imshow(cell_acc, cmap="Greens", vmin=0, vmax=100, alpha=0.85)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Tab: CN", "Tab: AD"], fontsize=11)
        ax.set_yticklabels(["MRI: CN", "MRI: AD"], fontsize=11)
        ax.set_title(name + f"\nN={len(sub)} subjects", fontsize=11)

        for i in range(2):
            for j in range(2):
                n = cell_n[i, j]
                if n > 0:
                    acc = cell_acc[i, j]
                    # Add "agree" / "disagree" badge on bottom-right quadrant
                    ax.text(j, i - 0.15, f"N={n}", ha="center", va="center",
                            fontsize=13, fontweight="bold")
                    ax.text(j, i + 0.15,
                            f"fused-acc\n{acc:.0f}%",
                            ha="center", va="center", fontsize=10,
                            color="white" if acc > 50 else "black")
                else:
                    ax.text(j, i, "—", ha="center", va="center", fontsize=13, color="gray")

        # Mark the "correct" quadrant for this true label with a border
        correct_i = label  # for CN label=0, MRI:CN is correct row 0
        correct_j = label
        ax.add_patch(plt.Rectangle(
            (correct_j - 0.5, correct_i - 0.5), 1, 1,
            fill=False, edgecolor="black", linewidth=2.5, linestyle="--",
        ))

    plt.suptitle(
        "How the fused classifier resolves each (MRI, Tab) prediction combination\n"
        "Dashed box = the cell where both modalities are correct. Other cells show "
        "disagreement / consensus-wrong handling.",
        fontsize=11, y=1.02,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close()
    print(f"[saved] {out_path.name}")


def main():
    df = load_df()
    print(f"[*] Loaded {len(df)} predictions")

    plot_headline(df, RESULTS / "fig_cross_modal_headline.png")
    plot_casebreak(df, RESULTS / "fig_cross_modal_casebreak.png")

    # Print headline stats for the paper
    print("\n=== Paper-ready stats ===")
    for cls, lab in [(0, "CN"), (1, "AD")]:
        sub = df[df["label"] == cls]
        for head in ["pred_mri", "pred_tab", "pred_fused"]:
            acc = 100 * (sub[head] == cls).mean()
            print(f"  {lab} subjects, {head:11s}: {acc:.1f}% acc (N={len(sub)})")
        print()


if __name__ == "__main__":
    main()

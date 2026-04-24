#!/usr/bin/env python3
"""
Cross-modal agreement analysis.

For each subject in each fold's test set, extracts three predictions from the
same multimodal model (seed 42 paper-grade checkpoints):
  1. MRI-only prediction (auxiliary MRI classifier)
  2. Tab-only prediction (auxiliary tabular classifier)
  3. Fused prediction (final cross-modal classifier)

Analyzes agreement / disagreement patterns and produces figures.

Usage:
    python cross_modal_analysis.py [--seed 42]
"""

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))
from dataset import MultiModalDataset  # noqa: E402
from interpretability.ft_attention import (  # noqa: E402
    TABULAR_FEATURES, build_paper_model,
)

REPO = Path("/home/tanguy/medical/alzheimer")
EXP = REPO / "experiments" / "multimodal_fusion"
RESULTS = SCRIPT_DIR / "results_new"


def regenerate_fold_test(seed, fold, n_folds=5):
    """Get fold's test set deterministically."""
    train_df = pd.read_csv(EXP / "data/combined_trajectory/train.csv")
    val_df = pd.read_csv(EXP / "data/combined_trajectory/val.csv")
    test_df = pd.read_csv(EXP / "data/combined_trajectory/test.csv")
    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    labels = all_df["label"].values
    indices = np.arange(len(all_df))
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fi, (_, tst_idx) in enumerate(skf.split(indices, labels)):
        if fi == fold:
            return all_df.iloc[tst_idx].reset_index(drop=True)
    raise ValueError(f"fold {fold} not found")


def run_fold(fold, seed, device):
    ckpt_path = EXP / "cv_results" / f"seed_{seed}" / f"fold_{fold}" / "model.pth"
    scaler_path = EXP / "cv_results" / f"seed_{seed}" / f"fold_{fold}" / "scaler.pkl"
    print(f"\n===== FOLD {fold} =====")

    test_df = regenerate_fold_test(seed, fold)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    tmp_csv = RESULTS / f"_tmp_cmtest_fold{fold}.csv"
    test_df.to_csv(tmp_csv, index=False)
    ds = MultiModalDataset(
        str(tmp_csv),
        tabular_features=TABULAR_FEATURES,
        target_shape=(128, 128, 128),
        augment=False,
        normalize_tabular=True,
        scaler=scaler,
        use_paper_preprocessing=True,
        target_spacing=1.75,
    )
    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)

    model = build_paper_model()
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(sd, strict=False)
    model = model.to(device).eval()

    all_labels = []
    all_logits_fused = []
    all_logits_mri = []
    all_logits_tab = []
    with torch.no_grad():
        for mri, tab, y in loader:
            mri = mri.to(device)
            tab = tab.to(device)
            out = model(mri, tab, return_auxiliary=True)
            all_logits_fused.append(out["logits"].cpu())
            all_logits_mri.append(out["mri_logits"].cpu())
            all_logits_tab.append(out["tab_logits"].cpu())
            all_labels.append(y)
    tmp_csv.unlink()

    labels = torch.cat(all_labels).numpy()
    def logits_to_preds(x):
        probs = torch.softmax(torch.cat(x), dim=-1).numpy()
        preds = probs.argmax(-1)
        return preds, probs
    pred_fused, prob_fused = logits_to_preds(all_logits_fused)
    pred_mri, prob_mri = logits_to_preds(all_logits_mri)
    pred_tab, prob_tab = logits_to_preds(all_logits_tab)

    subject_ids = test_df["subject_id"].values

    df = pd.DataFrame({
        "subject_id": subject_ids,
        "label": labels,
        "pred_fused": pred_fused,
        "pred_mri": pred_mri,
        "pred_tab": pred_tab,
        "prob_fused_ad": prob_fused[:, 1],
        "prob_mri_ad": prob_mri[:, 1],
        "prob_tab_ad": prob_tab[:, 1],
        "fold": fold,
    })

    # Print quick per-modality accuracy
    print(f"  MRI-only   acc: {100 * accuracy_score(labels, pred_mri):.2f}%")
    print(f"  Tab-only   acc: {100 * accuracy_score(labels, pred_tab):.2f}%")
    print(f"  Fused      acc: {100 * accuracy_score(labels, pred_fused):.2f}%")

    return df


def analyze_agreement(df, title="All folds"):
    """
    Build agreement categories and print / return statistics.

    Categories:
      - both_agree_correct: mri == tab == fused == label
      - both_agree_wrong: mri == tab == fused != label
      - disagree_fused_mri: mri != tab, fused == mri
      - disagree_fused_tab: mri != tab, fused == tab
      - disagree_fused_neither: mri != tab, fused != mri and != tab (rare for binary)
    """
    mri = df["pred_mri"].values
    tab = df["pred_tab"].values
    fused = df["pred_fused"].values
    lab = df["label"].values

    agree = mri == tab
    disagree = ~agree

    # Correct for each head
    correct_mri = mri == lab
    correct_tab = tab == lab
    correct_fused = fused == lab

    # For agreement cases, fused typically follows both
    agree_correct = agree & correct_mri  # since mri==tab, all three agree
    agree_wrong = agree & ~correct_mri

    # Disagreement handling
    fused_follows_mri = disagree & (fused == mri)
    fused_follows_tab = disagree & (fused == tab)

    stats = {
        "N": len(df),
        "Modalities agree": agree.sum(),
        "  correct": agree_correct.sum(),
        "  wrong": agree_wrong.sum(),
        "Modalities disagree": disagree.sum(),
        "  fused follows MRI": fused_follows_mri.sum(),
        "    MRI correct": (fused_follows_mri & correct_fused).sum(),
        "    MRI wrong": (fused_follows_mri & ~correct_fused).sum(),
        "  fused follows Tab": fused_follows_tab.sum(),
        "    Tab correct": (fused_follows_tab & correct_fused).sum(),
        "    Tab wrong": (fused_follows_tab & ~correct_fused).sum(),
    }

    print(f"\n--- Agreement analysis ({title}) ---")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # Accuracy in each regime
    if agree.sum() > 0:
        acc_agree = 100 * correct_fused[agree].mean()
        print(f"  Fused acc when modalities AGREE:    {acc_agree:.1f}%")
    if disagree.sum() > 0:
        acc_disagree = 100 * correct_fused[disagree].mean()
        print(f"  Fused acc when modalities DISAGREE: {acc_disagree:.1f}%")
        # Oracle: if fused always picked the correct one
        oracle = correct_mri | correct_tab
        acc_oracle = 100 * oracle[disagree].mean()
        print(f"  Oracle (pick correct modality) on disagreement: {acc_oracle:.1f}%")

    return stats


def plot_agreement_figure(df, out_path):
    """Main figure: accuracy per regime + confusion matrix of modality agreements."""
    mri = df["pred_mri"].values
    tab = df["pred_tab"].values
    fused = df["pred_fused"].values
    lab = df["label"].values

    # Classify each sample into a regime
    agree = mri == tab
    cats = np.empty(len(df), dtype=object)
    cats[agree & (mri == lab)] = "Agree-correct"
    cats[agree & (mri != lab)] = "Agree-wrong"
    cats[~agree & (fused == mri) & (fused == lab)] = "Disagree-fused→MRI-correct"
    cats[~agree & (fused == mri) & (fused != lab)] = "Disagree-fused→MRI-wrong"
    cats[~agree & (fused == tab) & (fused == lab)] = "Disagree-fused→Tab-correct"
    cats[~agree & (fused == tab) & (fused != lab)] = "Disagree-fused→Tab-wrong"

    categories = [
        "Agree-correct", "Agree-wrong",
        "Disagree-fused→MRI-correct", "Disagree-fused→MRI-wrong",
        "Disagree-fused→Tab-correct", "Disagree-fused→Tab-wrong",
    ]
    counts = [int((cats == c).sum()) for c in categories]
    colors = ["#4caf50", "#c62828",
              "#1976d2", "#ffa726",
              "#7b1fa2", "#f06292"]

    fig, axes = plt.subplots(1, 3, figsize=(22, 6))

    # Panel 1: regime counts
    y = np.arange(len(categories))
    axes[0].barh(y, counts, color=colors, alpha=0.85)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(categories)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("N subjects")
    axes[0].set_title(f"Modality agreement regimes (N={len(df)})")
    for i, c in enumerate(counts):
        axes[0].text(c + 15, i, f"{c} ({100 * c / len(df):.1f}%)", va="center", fontsize=9)

    # Panel 2: modality agreement confusion matrix (MRI pred × Tab pred, for each label)
    mri_tab = confusion_matrix(mri, tab, labels=[0, 1])
    im = axes[1].imshow(mri_tab, cmap="Blues")
    axes[1].set_xticks([0, 1])
    axes[1].set_yticks([0, 1])
    axes[1].set_xticklabels(["Tab: CN", "Tab: AD"])
    axes[1].set_yticklabels(["MRI: CN", "MRI: AD"])
    axes[1].set_title("MRI-pred × Tab-pred (N subjects per cell)")
    for i in range(2):
        for j in range(2):
            axes[1].text(j, i, mri_tab[i, j], ha="center", va="center",
                         color="black" if mri_tab[i, j] < mri_tab.max() / 2 else "white",
                         fontsize=14, weight="bold")
    plt.colorbar(im, ax=axes[1], fraction=0.04)

    # Panel 3: per-class acc of each head
    head_names = ["MRI-only", "Tab-only", "Fused"]
    preds_list = [mri, tab, fused]
    cn_accs = [100 * (p[lab == 0] == 0).mean() for p in preds_list]
    ad_accs = [100 * (p[lab == 1] == 1).mean() for p in preds_list]
    overall = [100 * (p == lab).mean() for p in preds_list]
    x = np.arange(3)
    axes[2].bar(x - 0.25, cn_accs, 0.25, label="CN", color="steelblue")
    axes[2].bar(x, ad_accs, 0.25, label="AD", color="coral")
    axes[2].bar(x + 0.25, overall, 0.25, label="Overall", color="gray", alpha=0.7)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(head_names)
    axes[2].set_ylim(50, 100)
    axes[2].set_ylabel("Accuracy (%)")
    axes[2].set_title("Per-class accuracy by modality head")
    axes[2].legend()
    axes[2].grid(axis="y", alpha=0.3)
    for i, (cn, ad, ov) in enumerate(zip(cn_accs, ad_accs, overall)):
        axes[2].text(i - 0.25, cn + 0.5, f"{cn:.1f}", ha="center", fontsize=8)
        axes[2].text(i, ad + 0.5, f"{ad:.1f}", ha="center", fontsize=8)
        axes[2].text(i + 0.25, ov + 0.5, f"{ov:.1f}", ha="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {out_path.name}")


def plot_disagreement_scatter(df, out_path):
    """
    Scatter plot: MRI prob_AD × Tab prob_AD, colored by Fused prediction,
    with label shape.
    """
    import matplotlib.patches as mpatches

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax_i, true_label in enumerate([0, 1]):
        ax = axes[ax_i]
        sub = df[df["label"] == true_label]
        # Color by fused prediction correctness
        correct = sub["pred_fused"] == true_label
        mri_prob = sub["prob_mri_ad"].values
        tab_prob = sub["prob_tab_ad"].values

        ax.scatter(mri_prob[correct], tab_prob[correct],
                   c="green", s=15, alpha=0.5, label="Fused correct")
        ax.scatter(mri_prob[~correct], tab_prob[~correct],
                   c="red", s=25, alpha=0.8, label="Fused wrong", marker="x")

        # Threshold lines at 0.5
        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
        ax.axvline(0.5, color="gray", linestyle="--", alpha=0.5)

        # Highlight disagreement quadrants (MRI and Tab disagree)
        ax.fill_between([0, 0.5], 0.5, 1.0, color="orange", alpha=0.07)  # MRI=CN, Tab=AD
        ax.fill_between([0.5, 1.0], 0, 0.5, color="purple", alpha=0.07)  # MRI=AD, Tab=CN

        ax.set_xlabel("MRI-only P(AD)")
        ax.set_ylabel("Tab-only P(AD)")
        ax.set_title(f"True label: {'CN' if true_label == 0 else 'AD'} (N={len(sub)})")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(loc="lower left")

        # Annotate quadrants
        ax.text(0.25, 0.95, "MRI:CN Tab:AD", ha="center", fontsize=8, alpha=0.6)
        ax.text(0.75, 0.95, "Both AD", ha="center", fontsize=8, alpha=0.6)
        ax.text(0.25, 0.03, "Both CN", ha="center", fontsize=8, alpha=0.6)
        ax.text(0.75, 0.03, "MRI:AD Tab:CN", ha="center", fontsize=8, alpha=0.6)

    plt.suptitle("Per-modality confidence (5 folds merged) — disagreement = orange/purple quadrants",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {out_path.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Device: {device}")

    dfs = []
    for fold in range(5):
        dfs.append(run_fold(fold, args.seed, device))
    full = pd.concat(dfs, ignore_index=True)
    full.to_csv(RESULTS / "cross_modal_predictions.csv", index=False)
    print(f"\n[*] Saved {len(full)} per-subject predictions to cross_modal_predictions.csv")

    # Global analysis
    stats = analyze_agreement(full, title=f"5 folds, seed {args.seed}")

    # Per-class break down
    for lab, name in [(0, "CN only"), (1, "AD only")]:
        _ = analyze_agreement(full[full["label"] == lab], title=name)

    # Figures
    plot_agreement_figure(full, RESULTS / "cross_modal_agreement_fig.png")
    plot_disagreement_scatter(full, RESULTS / "cross_modal_scatter_fig.png")

    print("\n[*] Done.")


if __name__ == "__main__":
    main()

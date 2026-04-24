#!/usr/bin/env python3
"""
Cross-modal fusion vs late-fusion baselines (Table III of the paper).

Compares the cross-attention fusion model against four alternatives on fold 0:
  - MRI-only ViT (standalone)
  - Tabular-only FT-Transformer (standalone)
  - Probability averaging of the two unimodal outputs
  - Logistic-regression stacker on (p_MRI, p_Tab), 5-fold CV'd to avoid
    training and evaluating on the same subjects

Reads per-subject predictions from cross_modal_real_predictions.csv (produced
by cross_modal_real_baselines.py) and emits a regime-by-regime accuracy table.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

SCRIPT_DIR = Path(__file__).resolve().parent
PRED_CSV = SCRIPT_DIR / "results_new" / "03_cross_modal" / "cross_modal_real_predictions.csv"
OUT_CSV = SCRIPT_DIR / "results_new" / "03_cross_modal" / "cross_modal_baselines_table.csv"


def max_confidence_pred(row):
    conf_mri = abs(row["prob_mri_ad"] - 0.5)
    conf_tab = abs(row["prob_tab_ad"] - 0.5)
    return row["pred_mri"] if conf_mri > conf_tab else row["pred_tab"]


def build_regimes(df):
    df = df.copy()
    df["agree"] = df["pred_mri"] == df["pred_tab"]
    df["both_right"] = (df["pred_mri"] == df["label"]) & (df["pred_tab"] == df["label"])
    return {
        "Agree, both right": df.agree & df.both_right,
        "Agree, both wrong": df.agree & ~df.both_right,
        "Disagree":          ~df.agree,
        "Overall":            pd.Series([True] * len(df)),
    }


def main():
    df = pd.read_csv(PRED_CSV)
    print(f"[*] N={len(df)} subjects from fold {df['fold'].unique()}")

    # Late-fusion baselines
    df["prob_avg"] = (df["prob_mri_ad"] + df["prob_tab_ad"]) / 2
    df["pred_avg"] = (df["prob_avg"] >= 0.5).astype(int)
    df["pred_maxconf"] = df.apply(max_confidence_pred, axis=1)

    # Stacking via LR with 5-fold CV to avoid train-on-test
    X = df[["prob_mri_ad", "prob_tab_ad"]].values
    y = df["label"].values
    stack_preds = np.zeros(len(df), dtype=int)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for tr, te in skf.split(X, y):
        clf = LogisticRegression()
        clf.fit(X[tr], y[tr])
        stack_preds[te] = clf.predict(X[te])
    df["pred_stack"] = stack_preds

    regimes = build_regimes(df)
    pred_cols = ["pred_mri", "pred_tab", "pred_avg", "pred_stack", "pred_maxconf", "pred_fused"]
    rows = []
    for name, mask in regimes.items():
        n = int(mask.sum())
        pct = 100.0 * n / len(df)
        row = {"Regime": name, "N": n, "%": round(pct, 1)}
        for c in pred_cols:
            acc = (df.loc[mask, c] == df.loc[mask, "label"]).mean() * 100
            row[c.replace("pred_", "")] = round(acc, 1)
        rows.append(row)

    out = pd.DataFrame(rows)
    print("\n=== Cross-modal accuracies by regime ===")
    print(out.to_string(index=False))
    out.to_csv(OUT_CSV, index=False)
    print(f"\n[*] saved {OUT_CSV}")


if __name__ == "__main__":
    main()

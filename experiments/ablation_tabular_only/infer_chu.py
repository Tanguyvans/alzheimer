#!/usr/bin/env python3
"""
Run tabular-only inference on the full CHU cohort (with or without MRI).

Loads fold checkpoint (model.pth + scaler.pkl) from ablation_tabular_only CV
and predicts AD trajectory for every CHU subject that has clinical data.

Usage:
    python experiments/ablation_tabular_only/infer_chu.py \
        --fold-dir experiments/ablation_tabular_only/cv_results/seed_42/fold_4 \
        --csv      experiments/multimodal_fusion/data/chu_tabular_all.csv
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from train_cv import FTTransformerClassifier  # noqa: E402

TABULAR_FEATURES = [
    "AGE", "PTGENDER", "PTEDUCAT", "PTMARRY",
    "CATANIMSC", "TRAASCOR", "TRABSCOR",
    "DSPANFOR", "DSPANBAC", "BNTTOTAL",
    "VSWEIGHT", "BMI",
    "MH14ALCH", "MH16SMOK", "MH4CARD", "MH2NEURL",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold-dir", type=Path,
                    default=HERE / "cv_results" / "seed_42" / "fold_4")
    ap.add_argument("--csv", type=Path,
                    default=HERE.parent / "multimodal_fusion" / "data" / "chu_tabular_all.csv")
    ap.add_argument("--config", type=Path, default=HERE / "config.yaml")
    ap.add_argument("--out", type=Path,
                    default=HERE / "results" / "chu_predictions_tabular.csv")
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Load model + scaler
    with open(args.fold_dir / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    df = pd.read_csv(args.csv)
    X = df[TABULAR_FEATURES].values.astype(np.float32)
    # Missing values in this CSV are encoded as -4 (ADNI convention — same
    # as the training data). We pass them through unchanged.
    Xs = scaler.transform(X)

    model = FTTransformerClassifier(
        num_features=len(TABULAR_FEATURES),
        config=config["model"],
        num_classes=config["model"]["num_classes"],
    )
    state = torch.load(args.fold_dir / "model.pth",
                       map_location=args.device, weights_only=True)
    model.load_state_dict(state)
    model.to(args.device).eval()
    print(f"Loaded {args.fold_dir}/model.pth")

    with torch.no_grad():
        logits = model(torch.tensor(Xs, device=args.device))
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    preds = probs.argmax(axis=1)

    out = df[["subject_id", "DX", "label"]].copy()
    out["P_AD"] = probs[:, 1]
    out["prediction"] = preds
    out["pred_DX"] = out["prediction"].map({0: "CN", 1: "AD"})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    # Pretty-print
    print()
    for _, r in out.iterrows():
        print(f"  {r['subject_id']:<10}  DX={r['DX']:<8}  "
              f"P(AD)={r['P_AD']:.3f}  pred={r['pred_DX']}")

    print("\n=== Summary ===")
    for dx in sorted(out["DX"].unique()):
        g = out[out["DX"] == dx]
        n_ad = (g["prediction"] == 1).sum()
        print(f"  {dx:<10} n={len(g):<3}  predicted AD: {n_ad}/{len(g)}  "
              f"mean P(AD)={g['P_AD'].mean():.3f}")

    known = out[out["label"].isin([0, 1])]
    if len(known):
        acc = (known["prediction"] == known["label"]).mean()
        print(f"\nAccuracy on labeled subjects ({len(known)}): {acc:.3f}")
    print(f"\nWrote: {args.out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Run CHU test inference using a fold checkpoint from cross-validation.

Loads:
  - model.pth (fold state_dict)
  - scaler.pkl (fitted StandardScaler for tabular features)
  - chu_test.csv (prepared by prepare_chu_test.py)

Outputs a predictions CSV + a summary.

Usage:
    python experiments/multimodal_fusion/infer_chu.py \
        --fold-dir experiments/multimodal_fusion/cv_results/seed_42/fold_4 \
        --csv      experiments/multimodal_fusion/data/chu_test.csv
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from dataset import MultiModalDataset
from model import build_model

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
                    default=HERE / "data" / "chu_test.csv")
    ap.add_argument("--config", type=Path, default=HERE / "config.yaml")
    ap.add_argument("--out", type=Path,
                    default=HERE / "results" / "chu_predictions.csv")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=1)
    args = ap.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    model_pth = args.fold_dir / "model.pth"
    scaler_pkl = args.fold_dir / "scaler.pkl"
    assert model_pth.exists(), f"missing {model_pth}"
    assert scaler_pkl.exists(), f"missing {scaler_pkl}"
    assert args.csv.exists(), f"missing {args.csv}"

    print(f"Device: {args.device}")
    print(f"Fold:   {args.fold_dir}")
    print(f"CSV:    {args.csv}")

    # Load scaler
    with open(scaler_pkl, "rb") as f:
        scaler = pickle.load(f)
    print(f"Loaded scaler ({scaler.n_features_in_} features, "
          f"mean[0]={scaler.mean_[0]:.2f}, scale[0]={scaler.scale_[0]:.2f})")

    # Build dataset using the fitted scaler
    ds = MultiModalDataset(
        csv_path=str(args.csv),
        tabular_features=TABULAR_FEATURES,
        target_shape=(128, 128, 128),
        augment=False,
        normalize_tabular=True,
        scaler=scaler,
        use_paper_preprocessing=config["preprocessing"].get("use_paper_preprocessing", True),
        target_spacing=config["preprocessing"].get("target_spacing", 1.75),
        handle_missing=config["preprocessing"].get("handle_missing", "median"),
    )

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, pin_memory=False)

    # Build model and load weights
    model = build_model(config, num_tabular_features=len(TABULAR_FEATURES))
    state = torch.load(model_pth, map_location=args.device, weights_only=True)
    model.load_state_dict(state)
    model.to(args.device).eval()
    print(f"Loaded weights from {model_pth}")

    # Inference
    df = pd.read_csv(args.csv)
    preds, probs_ad = [], []
    with torch.no_grad():
        for idx, (mri, tabular, label) in enumerate(loader):
            mri = mri.to(args.device)
            tabular = tabular.to(args.device)
            out = model(mri, tabular)
            # Model may return logits directly or a dict with 'logits'
            if isinstance(out, dict):
                logits = out.get("logits", out.get("fusion_logits"))
            else:
                logits = out
            p = torch.softmax(logits, dim=1).cpu().numpy()
            pred = int(p.argmax(axis=1)[0])
            preds.append(pred)
            probs_ad.append(float(p[0, 1]))
            subj = df.iloc[idx]["subject_id"]
            dx = df.iloc[idx]["DX"]
            print(f"  [{idx+1}/{len(df)}] {subj:<10}  DX={dx:<8}  "
                  f"P(AD)={p[0,1]:.3f}  pred={['CN','AD'][pred]}")

    df_out = df[["subject_id", "DX", "label"]].copy()
    df_out["prediction"] = preds
    df_out["P_AD"] = probs_ad
    df_out["pred_DX"] = df_out["prediction"].map({0: "CN", 1: "AD"})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(args.out, index=False)
    print(f"\nWrote {args.out}")

    # Summary
    print("\n=== Summary ===")
    for dx in sorted(df_out["DX"].unique()):
        g = df_out[df_out["DX"] == dx]
        n_ad = (g["prediction"] == 1).sum()
        mean_p = g["P_AD"].mean()
        print(f"  {dx:<10} n={len(g):<3}  predicted AD: {n_ad}/{len(g)}  "
              f"mean P(AD)={mean_p:.3f}")

    # Accuracy where label is known (excludes Unknown with label=-1)
    known = df_out[df_out["label"].isin([0, 1])]
    if len(known):
        acc = (known["prediction"] == known["label"]).mean()
        print(f"\nAccuracy on labeled subjects ({len(known)}): {acc:.3f}")


if __name__ == "__main__":
    main()

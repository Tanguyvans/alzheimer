#!/usr/bin/env python3
"""
Cross-check tabular model predictions against radiologist MRI readings.

For each CHU subject we combine:
  - DX (clinical diagnosis) from bilan.xlsx
  - MTAS (medial temporal atrophy) + Fazekas + IRM_patho from final.xlsx
  - model prediction from chu_predictions_tabular.csv

Produces chu_radio_compare.csv with everything side-by-side and prints a
concordance table.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def parse_mtas(v):
    try:
        return float(str(v).strip().replace(",", "."))
    except (ValueError, TypeError):
        return np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bilan", type=Path, default=Path("/home/tanguy/Desktop/bilan.xlsx"))
    ap.add_argument("--final", type=Path, default=Path("/home/tanguy/Desktop/final.xlsx"))
    ap.add_argument("--preds", type=Path,
                    default=HERE / "results" / "chu_predictions_tabular.csv")
    ap.add_argument("--out", type=Path,
                    default=HERE / "results" / "chu_radio_compare.csv")
    args = ap.parse_args()

    bilan = pd.read_excel(args.bilan, sheet_name="Feuil1")
    final = pd.read_excel(args.final, sheet_name="Feuil1")
    preds = pd.read_csv(args.preds)

    col_mtas = next(c for c in final.columns if "mtas" in str(c).lower() and "atrophy score" in str(c).lower())
    col_mtas_interp = next(c for c in final.columns if "mtas interpretation" in str(c).lower())
    col_faz_pv = next(c for c in final.columns
                      if "periventriculaire" in str(c).lower())
    col_faz_wm = next(c for c in final.columns
                      if "substance blanche" in str(c).lower())
    col_irm = next(c for c in final.columns
                   if "irm cerebrale" in str(c).lower() and "patho" in str(c).lower())

    rad = final[["Patient", col_mtas, col_mtas_interp, col_faz_pv, col_faz_wm, col_irm]].copy()
    rad.columns = ["subject_id", "MTAS", "MTAS_patho", "Fazekas_PV", "Fazekas_WM", "IRM_patho"]
    rad["MTAS"] = rad["MTAS"].apply(parse_mtas)
    for c in ["MTAS_patho", "Fazekas_PV", "Fazekas_WM", "IRM_patho"]:
        rad[c] = pd.to_numeric(rad[c], errors="coerce")

    rad = rad[rad["subject_id"].astype(str).str.match(r"^COGN\d{4}$", na=False)]

    df = preds.merge(rad, on="subject_id", how="left")
    df = df[["subject_id", "DX", "label", "P_AD", "pred_DX",
             "MTAS", "MTAS_patho", "Fazekas_PV", "Fazekas_WM", "IRM_patho"]]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    # Concordance table: model prediction (AD/CN) × radiologist MTAS_patho (1/0)
    sub = df.dropna(subset=["MTAS_patho"])
    tab = pd.crosstab(sub["pred_DX"], sub["MTAS_patho"].astype(int),
                      rownames=["model"], colnames=["MTAS_patho"],
                      margins=True, margins_name="total")
    print("\n=== Model prediction vs MTAS (radiologist atrophy) ===")
    print(tab.to_string())

    # Agreement rate
    agree = ((sub["pred_DX"] == "AD") & (sub["MTAS_patho"] == 1)) | \
            ((sub["pred_DX"] == "CN") & (sub["MTAS_patho"] == 0))
    print(f"\nAgreement model <-> radiologist: {agree.sum()}/{len(sub)}  "
          f"({100 * agree.mean():.1f}%)")

    # Disagreements
    disagree = sub[~agree]
    if len(disagree):
        print(f"\nDisagreements ({len(disagree)}):")
        print(disagree[["subject_id", "DX", "P_AD", "pred_DX", "MTAS",
                        "MTAS_patho", "IRM_patho"]].to_string(index=False))

    print(f"\nWrote: {args.out}")
    print(f"\nFull table:")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()

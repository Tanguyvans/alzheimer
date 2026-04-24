#!/usr/bin/env python3
"""
Build a CHU test CSV compatible with MultiModalDataset.

Inputs:
  - /home/tanguy/Desktop/bilan.xlsx  (CDR, MMSE, Classe ADNI)
  - /home/tanguy/Desktop/final.xlsx  (clinical features)
  - /home/tanguy/Desktop/irm_chu_work/skull/{subj}/{subj}_tp1_skull_stripped.nii.gz

Output:
  - experiments/multimodal_fusion/data/chu_test.csv
    columns: subject_id, scan_path, DX, label, source, AGE, PTGENDER,
             PTEDUCAT, PTMARRY, CATANIMSC, TRAASCOR, TRABSCOR, DSPANFOR,
             DSPANBAC, BNTTOTAL, VSWEIGHT, BMI, MH14ALCH, MH16SMOK,
             MH4CARD, MH2NEURL

Missing values are coded as -4.0 (ADNI convention, matches NACC training data).

Usage:
    python experiments/multimodal_fusion/prepare_chu_test.py
"""

import argparse
import re
from pathlib import Path

import pandas as pd

MRI_ROOT = Path("/home/tanguy/Desktop/irm_chu_work/skull")
BILAN_XLSX = Path("/home/tanguy/Desktop/bilan.xlsx")
FINAL_XLSX = Path("/home/tanguy/Desktop/final.xlsx")

MISSING = -4.0

# CHU column -> (ADNI name, transform)
# final.xlsx columns have very long labels; match by substring
def get_col(df, substr):
    for c in df.columns:
        if substr.lower() in str(c).lower():
            return c
    return None


def parse_num(v):
    """Coerce to float, treating 'manquant', 'stop', '<7', '' as missing."""
    if v is None:
        return MISSING
    if isinstance(v, (int, float)):
        if pd.isna(v):
            return MISSING
        return float(v)
    s = str(v).strip().lower()
    if s in ("", "manquant", "stop", "nan", "na", "n/a"):
        return MISSING
    # "<7" → 0 (low consumption → no abuse)
    if s.startswith("<"):
        try:
            return float(s[1:].strip())
        except ValueError:
            return MISSING
    # "7-14", "14-21", ">21" → midpoint-ish
    m = re.match(r"(\d+)\s*-\s*(\d+)", s)
    if m:
        return (float(m.group(1)) + float(m.group(2))) / 2
    if s.startswith(">"):
        try:
            return float(s[1:].strip()) * 1.5
        except ValueError:
            return MISSING
    try:
        return float(s.replace(",", "."))
    except ValueError:
        return MISSING


def educ_to_years(level):
    """Poitrenaud level (1-4) → ADNI years of education."""
    v = parse_num(level)
    if v == MISSING:
        return MISSING
    return {1: 6, 2: 9, 3: 12, 4: 16}.get(int(v), MISSING)


def alcohol_to_binary(v):
    """CHU alcohol (UA/week, categorical) → MH14ALCH (0=no/low, 1=abuse)."""
    n = parse_num(v)
    if n == MISSING:
        return MISSING
    # Cutoff at 14 UA/week = abuse
    return 1.0 if n >= 14 else 0.0


def class_to_label(cls):
    """Classe ADNI string → (DX, label).
    Model was trained on cn_ad_trajectory: CN=0, AD+MCI_to_AD=1.
    CHU has no CN; MCI treated as AD_trajectory=1.
    """
    if pd.isna(cls):
        return ("Unknown", -1)
    s = str(cls).strip().upper()
    if s == "AD":
        return ("AD", 1)
    if s == "MCI":
        return ("MCI", 1)
    if s in ("NORMAL", "CN"):
        return ("CN", 0)
    return (s, -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).parent / "data" / "chu_test.csv")
    args = ap.parse_args()

    bilan = pd.read_excel(BILAN_XLSX, sheet_name="Feuil1")
    final = pd.read_excel(FINAL_XLSX, sheet_name="Feuil1")

    # Identify final.xlsx columns by substring
    col_age = "Age"
    col_sex = get_col(final, "sexe")
    col_bmi = "BMI"
    col_educ = get_col(final, "socio-culturel")
    col_smok = get_col(final, "tabagisme")
    col_card = get_col(final, "cardiopathie")
    col_neur = get_col(final, "maladie neurologique")
    col_alc = get_col(final, "consommation alcool")
    col_catanim = get_col(final, "fluences sémantiques")
    col_traa = get_col(final, "trail making test (partie a)")
    col_trab = get_col(final, "trail making test temps (partie b)")
    col_dsf = get_col(final, "empans endroit")
    col_dsb = get_col(final, "empans envers")
    col_lexis = get_col(final, "dénomination lexis")

    # Two sets of subjects
    #  - those with preprocessed MRI (11)   -> multimodal CSV
    #  - all in bilan+final (union)         -> tabular-only CSV
    subjects_with_mri = sorted([d.name for d in MRI_ROOT.iterdir()
                                if d.is_dir() and (d / f"{d.name}_tp1_skull_stripped.nii.gz").exists()])
    all_subjects = sorted(s for s in (set(bilan["Patient"].dropna().astype(str)) |
                                      set(final["Patient"].dropna().astype(str)))
                          if re.match(r"^COGN\d{4}$", s))
    print(f"{len(subjects_with_mri)} subjects have MRI, "
          f"{len(all_subjects)} total across bilan+final.")

    rows = []
    for subj in all_subjects:
        b = bilan[bilan["Patient"] == subj]
        f = final[final["Patient"] == subj]

        dx_str = b["Classe ADNI (0=normal, 1=MCI, 2=AD)"].iloc[0] if not b.empty else None
        dx, label = class_to_label(dx_str)

        if f.empty:
            age = sexe = bmi = educ_y = catanim = traa = trab = MISSING
            dsf = dsb = lexis = smok = card = neur = alc = MISSING
        else:
            r = f.iloc[0]
            age = parse_num(r[col_age])
            # CHU: 0=M, 1=F → ADNI: 1=M, 2=F
            s = parse_num(r[col_sex])
            sexe = s + 1 if s != MISSING else MISSING
            bmi = parse_num(r[col_bmi])
            educ_y = educ_to_years(r[col_educ])
            smok = parse_num(r[col_smok])
            card = parse_num(r[col_card])
            neur = parse_num(r[col_neur])
            alc = alcohol_to_binary(r[col_alc])
            catanim = parse_num(r[col_catanim])
            traa = parse_num(r[col_traa])
            trab = parse_num(r[col_trab])
            dsf = parse_num(r[col_dsf])
            dsb = parse_num(r[col_dsb])
            lexis = parse_num(r[col_lexis])

        has_mri = subj in subjects_with_mri
        scan_path = (str(MRI_ROOT / subj / f"{subj}_tp1_skull_stripped.nii.gz")
                     if has_mri else "")

        rows.append({
            "subject_id": subj,
            "scan_path": scan_path,
            "has_mri": has_mri,
            "DX": dx,
            "label": label,
            "source": "CHU",
            "AGE": age,
            "PTGENDER": sexe,
            "PTEDUCAT": educ_y,
            "PTMARRY": MISSING,     # not collected at CHU
            "CATANIMSC": catanim,
            "TRAASCOR": traa,
            "TRABSCOR": trab,
            "DSPANFOR": dsf,
            "DSPANBAC": dsb,
            "BNTTOTAL": lexis,       # Lexis naming, closest substitute
            "VSWEIGHT": MISSING,     # not collected
            "BMI": bmi,
            "MH14ALCH": alc,
            "MH16SMOK": smok,
            "MH4CARD": card,
            "MH2NEURL": neur,
        })

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    # 1) multimodal CSV: only subjects with MRI
    df_mri = df[df["has_mri"]].drop(columns=["has_mri"]).reset_index(drop=True)
    df_mri.to_csv(args.out, index=False)

    # 2) tabular-only CSV: all subjects
    tab_out = args.out.parent / "chu_tabular_all.csv"
    df_all = df.drop(columns=["has_mri", "scan_path"]).reset_index(drop=True)
    df_all.to_csv(tab_out, index=False)

    print(f"\nWrote multimodal CSV: {args.out} (shape={df_mri.shape})")
    print(f"Wrote tabular CSV:   {tab_out}  (shape={df_all.shape})")
    df = df_all
    print(f"Shape (tabular full): {df.shape}")
    print(f"\nLabel distribution (DX):")
    print(df["DX"].value_counts().to_string())
    print(f"\nMissing count per feature (-4 = missing):")
    feat_cols = ["AGE","PTGENDER","PTEDUCAT","PTMARRY","CATANIMSC","TRAASCOR",
                 "TRABSCOR","DSPANFOR","DSPANBAC","BNTTOTAL","VSWEIGHT","BMI",
                 "MH14ALCH","MH16SMOK","MH4CARD","MH2NEURL"]
    for c in feat_cols:
        n_miss = (df[c] == MISSING).sum()
        if n_miss:
            print(f"  {c:<12} {n_miss}/{len(df)}")

    print(f"\nPreview:")
    print(df[["subject_id", "DX", "label", "AGE", "PTGENDER",
              "MMSE" if "MMSE" in df.columns else "CATANIMSC"]].to_string(index=False))


if __name__ == "__main__":
    main()

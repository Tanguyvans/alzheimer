#!/usr/bin/env python3
"""
CHU MRI diagnostic pass.

Walks each subject under --input, extracts zip archives once into
--work/staged/{subject}/, then enumerates every DICOM series present,
reads a handful of headers, scores each series for "T1-3D-MPRAGE-ness",
and writes a CSV so the operator can validate the auto-pick before
launching the full preprocessing pipeline.

Usage:
    python preprocessing/imaging/chu_diagnose_series.py \
        --input  /home/tanguy/Desktop/irm_chu \
        --work   /home/tanguy/Desktop/irm_chu_work
"""

import argparse
import csv
import logging
import shutil
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import pydicom
from pydicom.errors import InvalidDicomError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("chu_diag")


REJECT_KEYWORDS = (
    "T2", "FLAIR", "DWI", "DIFF", "PERF", "ASL", "SWI", "TOF",
    "LOC", "SCOUT", "SURVEY", "CALIB", "B0", "B1", "FIELDMAP",
    "MRA", "MRV", "VENOGRAM", "ANGIO", "BOLD", "FMRI", "SPECTRO",
    "PD", "DENSITY",
)
POSITIVE_KEYWORDS = ("MPRAGE", "BRAVO", "FSPGR", "TFE", "3DT1", "3D_T1", "SPGR")


def score_series(desc, mr_acq, scan_seq, seq_var, image_type,
                 tr, te, ti, slice_thickness, n_slices):
    desc_u = (desc or "").upper()
    img_u = " ".join(image_type or []).upper()

    # Hard reject on obvious non-T1 sequences
    for kw in REJECT_KEYWORDS:
        if kw in desc_u:
            return -999, f"reject:{kw}"
    if "DERIVED" in img_u and "PRIMARY" not in img_u:
        return -999, "reject:DERIVED-only"
    if "PROJECTION IMAGE" in img_u or "MPR" in img_u and "ORIGINAL" not in img_u:
        return -999, "reject:projection/MPR-derived"

    score = 0
    reasons = []

    for kw in POSITIVE_KEYWORDS:
        if kw in desc_u:
            score += 3
            reasons.append(f"+3 desc~{kw}")
            break
    if "T1" in desc_u and "FLAIR" not in desc_u:
        score += 2
        reasons.append("+2 desc~T1")

    if (mr_acq or "").upper() == "3D":
        score += 2
        reasons.append("+2 3D")

    if tr is not None and 1500 <= tr <= 2800 and te is not None and te <= 10:
        score += 2
        reasons.append(f"+2 TR/TE={tr:.0f}/{te:.1f}")
    if ti is not None and 800 <= ti <= 1200:
        score += 2
        reasons.append(f"+2 TI={ti:.0f}")

    if slice_thickness is not None and slice_thickness <= 1.5:
        score += 1
        reasons.append(f"+1 thk={slice_thickness:.2f}")

    if n_slices and 160 <= n_slices <= 260:
        score += 2
        reasons.append(f"+2 n={n_slices}")
    elif n_slices and n_slices >= 120:
        score += 1
        reasons.append(f"+1 n={n_slices}")
    elif n_slices and n_slices < 40:
        score -= 2
        reasons.append(f"-2 n={n_slices}")

    # Prefer original acquisition
    if "ORIGINAL" in img_u and "PRIMARY" in img_u:
        score += 1
        reasons.append("+1 ORIGINAL/PRIMARY")

    # Penalize if scanning sequence is pure SE (spin echo -> not MPRAGE)
    if (scan_seq or "").upper() == "SE":
        score -= 2
        reasons.append("-2 SE")

    return score, " ".join(reasons) or "(none)"


def ensure_subject_staged(subject_dir: Path, staged_root: Path) -> Path:
    """
    Return the directory to walk for DICOMs for a given subject.

    If the subject contains zip(s), extract once into staged_root/subject/.
    Otherwise use the subject dir itself.
    """
    zips = list(subject_dir.glob("*.zip"))
    if not zips:
        return subject_dir

    target = staged_root / subject_dir.name
    if target.exists() and any(target.rglob("*")):
        log.info(f"  staged already present: {target}")
        return target

    target.mkdir(parents=True, exist_ok=True)
    for z in zips:
        log.info(f"  extracting {z.name} -> {target}")
        try:
            with zipfile.ZipFile(z) as zf:
                zf.extractall(target)
        except zipfile.BadZipFile as e:
            log.error(f"    bad zip {z}: {e}")
    return target


def read_dicom_safely(path: Path):
    try:
        return pydicom.dcmread(str(path), stop_before_pixels=True, force=False)
    except (InvalidDicomError, Exception):
        return None


def walk_series(root: Path):
    """
    Walk every file under root, group by SeriesInstanceUID.

    Returns dict[series_uid] = {
        'files': [Path, ...],
        'sample_ds': pydicom.Dataset,
        'study_uid': str, 'study_date': str, 'study_time': str,
    }
    """
    series = defaultdict(lambda: {"files": [], "sample_ds": None,
                                  "study_uid": "", "study_date": "", "study_time": ""})
    skipped = 0
    n_seen = 0
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.name.startswith(".") or p.name.lower() == "dicomdir":
            continue
        ds = read_dicom_safely(p)
        n_seen += 1
        if ds is None:
            skipped += 1
            continue
        uid = getattr(ds, "SeriesInstanceUID", None)
        if not uid:
            skipped += 1
            continue
        entry = series[uid]
        entry["files"].append(p)
        if entry["sample_ds"] is None:
            entry["sample_ds"] = ds
            entry["study_uid"] = str(getattr(ds, "StudyInstanceUID", "") or "")
            entry["study_date"] = str(getattr(ds, "StudyDate", "") or "")
            entry["study_time"] = str(getattr(ds, "StudyTime", "") or "")
    log.info(f"  scanned {n_seen} files, kept {n_seen - skipped} DICOMs, "
             f"{len(series)} series")
    return series


def get(ds, name, default=None, cast=None):
    v = getattr(ds, name, None)
    if v in (None, ""):
        return default
    if cast is not None:
        try:
            return cast(v)
        except Exception:
            return default
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path,
                    help="Root dir with subject folders (e.g. /home/tanguy/Desktop/irm_chu)")
    ap.add_argument("--work", required=True, type=Path,
                    help="Working dir for staging + outputs")
    args = ap.parse_args()

    if not args.input.is_dir():
        log.error(f"Input not found: {args.input}")
        sys.exit(1)

    staged_root = args.work / "staged"
    staged_root.mkdir(parents=True, exist_ok=True)
    report_path = args.work / "series_report.csv"

    subjects = sorted(d for d in args.input.iterdir() if d.is_dir() and not d.name.startswith("."))
    log.info(f"{len(subjects)} subjects under {args.input}")

    fieldnames = [
        "subject", "timepoint", "study_date", "study_uid",
        "series_uid", "series_number", "series_description", "modality",
        "mr_acquisition_type", "scanning_sequence", "sequence_variant",
        "image_type", "tr", "te", "ti", "slice_thickness",
        "n_slices", "manufacturer", "model",
        "score", "picked", "reasons",
    ]

    rows = []
    for subj in subjects:
        log.info(f"=== {subj.name} ===")
        try:
            walk_root = ensure_subject_staged(subj, staged_root)
        except Exception as e:
            log.error(f"  staging failed: {e}")
            continue

        series = walk_series(walk_root)
        if not series:
            log.warning(f"  no DICOMs found")
            continue

        # Group series by (study_uid, study_date) to define timepoints
        studies = defaultdict(list)
        for uid, entry in series.items():
            key = (entry["study_uid"], entry["study_date"])
            studies[key].append((uid, entry))

        # Sort studies by date so timepoints get stable tp1, tp2...
        sorted_studies = sorted(studies.items(),
                                key=lambda kv: (kv[0][1] or "", kv[0][0]))
        tp_label = {key: (f"tp{i+1}" if len(sorted_studies) > 1 else "tp1")
                    for i, (key, _) in enumerate(sorted_studies)}

        # Score all series per timepoint, flag best as "picked"
        subj_rows = []
        for key, series_list in sorted_studies:
            tp = tp_label[key]
            scored = []
            for uid, entry in series_list:
                ds = entry["sample_ds"]
                image_type = list(getattr(ds, "ImageType", []) or [])
                desc = get(ds, "SeriesDescription", "")
                mr_acq = get(ds, "MRAcquisitionType", "")
                scan_seq = str(get(ds, "ScanningSequence", "") or "")
                seq_var = str(get(ds, "SequenceVariant", "") or "")
                tr = get(ds, "RepetitionTime", None, float)
                te = get(ds, "EchoTime", None, float)
                ti = get(ds, "InversionTime", None, float)
                thk = get(ds, "SliceThickness", None, float)
                n = len(entry["files"])
                sc, reasons = score_series(desc, mr_acq, scan_seq, seq_var,
                                           image_type, tr, te, ti, thk, n)
                scored.append((sc, uid, entry, {
                    "series_description": desc,
                    "modality": get(ds, "Modality", ""),
                    "mr_acquisition_type": mr_acq,
                    "scanning_sequence": scan_seq,
                    "sequence_variant": seq_var,
                    "image_type": "\\".join(image_type),
                    "tr": tr, "te": te, "ti": ti,
                    "slice_thickness": thk,
                    "n_slices": n,
                    "series_number": get(ds, "SeriesNumber", ""),
                    "manufacturer": get(ds, "Manufacturer", ""),
                    "model": get(ds, "ManufacturerModelName", ""),
                    "reasons": reasons,
                }))
            # best is the highest score, tiebreak by more slices
            scored.sort(key=lambda t: (t[0], t[3]["n_slices"]), reverse=True)
            for i, (sc, uid, entry, meta) in enumerate(scored):
                subj_rows.append({
                    "subject": subj.name,
                    "timepoint": tp,
                    "study_date": key[1],
                    "study_uid": key[0],
                    "series_uid": uid,
                    "score": sc,
                    "picked": "YES" if i == 0 and sc > 0 else "",
                    **meta,
                })

        log.info(f"  {len(subj_rows)} series rows, "
                 f"{sum(1 for r in subj_rows if r['picked'] == 'YES')} picked")
        rows.extend(subj_rows)

    # Write CSV
    with open(report_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    log.info(f"\nReport: {report_path}")
    log.info(f"Rows: {len(rows)}")
    picked = [r for r in rows if r["picked"] == "YES"]
    log.info(f"Auto-picked series: {len(picked)} "
             f"(over {len({(r['subject'], r['timepoint']) for r in rows})} subject/timepoints)")
    # Subjects missing a pick
    all_sts = {(r["subject"], r["timepoint"]) for r in rows}
    picked_sts = {(r["subject"], r["timepoint"]) for r in picked}
    missing = sorted(all_sts - picked_sts)
    if missing:
        log.warning(f"No pick for: {missing}")


if __name__ == "__main__":
    main()

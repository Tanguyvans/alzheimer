#!/usr/bin/env python3
"""
CHU MRI preprocessing pipeline (DICOM -> NIfTI -> N4+MNI -> SynthStrip).

Reads the picks from chu_diagnose_series.py (series_report.csv) and
converts only the selected T1 3D series per subject. Output layout
matches NACC-skull so downstream datasets can treat the two identically.

Usage:
    # Step 1: convert picked series to NIfTI
    python preprocessing/imaging/chu_pipeline.py --step convert \
        --input /home/tanguy/Desktop/irm_chu \
        --work  /home/tanguy/Desktop/irm_chu_work

    # Step 2: N4 + MNI registration (delegates to nacc_pipeline)
    python preprocessing/imaging/chu_pipeline.py --step register \
        --work /home/tanguy/Desktop/irm_chu_work

    # Step 3: SynthStrip skull stripping
    python preprocessing/imaging/chu_pipeline.py --step skull \
        --work /home/tanguy/Desktop/irm_chu_work

    # All steps in sequence
    python preprocessing/imaging/chu_pipeline.py --step all \
        --input /home/tanguy/Desktop/irm_chu \
        --work  /home/tanguy/Desktop/irm_chu_work
"""

import argparse
import csv
import logging
import shutil
import sys
import tempfile
from pathlib import Path

import SimpleITK as sitk

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Reuse NACC's registration + skull stripping machinery
from preprocessing.imaging.nacc_pipeline import (  # noqa: E402
    process_nifti_file,
    find_registered_files,
    run_skull_stripping as nacc_run_skull,
    load_progress,
    save_progress,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("chu")

DEFAULT_TEMPLATE = PROJECT_ROOT / (
    "mni_template/mni_icbm152_nlin_sym_09a_nifti/"
    "mni_icbm152_nlin_sym_09a/mni_icbm152_t1_tal_nlin_sym_09a.nii"
)


def read_picks(report_csv: Path):
    """
    Return picked rows that are usable 3D T1 volumes.

    We drop 2D clinical T1 (TSE 5mm etc.) because they cannot be coregistered
    to a 1mm MNI template without severe interpolation artifacts.
    """
    with open(report_csv) as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        if r["picked"] != "YES":
            continue
        if r["mr_acquisition_type"] != "3D":
            continue
        try:
            if int(r["n_slices"]) < 100:
                continue
        except ValueError:
            continue
        out.append(r)
    return out


def locate_series_files(subject: str, series_uid: str,
                        raw_input: Path, staged_root: Path):
    """
    Find all DICOM files under raw_input/subject or staged_root/subject
    that belong to `series_uid`.

    Returns a list of Path, properly ordered for SimpleITK.
    """
    import pydicom
    from pydicom.errors import InvalidDicomError

    candidates = []
    roots = [staged_root / subject, raw_input / subject]
    walk_root = next((r for r in roots if r.exists()), None)
    if walk_root is None:
        raise FileNotFoundError(f"No source dir for {subject}")

    for p in walk_root.rglob("*"):
        if not p.is_file():
            continue
        if p.name.startswith(".") or p.name.lower() == "dicomdir":
            continue
        try:
            ds = pydicom.dcmread(str(p), stop_before_pixels=True, force=False)
        except (InvalidDicomError, Exception):
            continue
        if str(getattr(ds, "SeriesInstanceUID", "")) != series_uid:
            continue
        candidates.append((p, ds))

    if not candidates:
        return []

    # Order by slice position along the normal of ImageOrientationPatient.
    def slice_pos(ds):
        try:
            ipp = [float(x) for x in ds.ImagePositionPatient]
            iop = [float(x) for x in ds.ImageOrientationPatient]
            # normal = iop[0:3] x iop[3:6]
            nx = iop[1] * iop[5] - iop[2] * iop[4]
            ny = iop[2] * iop[3] - iop[0] * iop[5]
            nz = iop[0] * iop[4] - iop[1] * iop[3]
            return ipp[0] * nx + ipp[1] * ny + ipp[2] * nz
        except Exception:
            return float(getattr(ds, "InstanceNumber", 0) or 0)

    candidates.sort(key=lambda pd: slice_pos(pd[1]))
    return [p for p, _ in candidates]


def convert_series_to_nifti(files, output_path: Path) -> bool:
    """
    Read a list of DICOM files in order and write a single NIfTI volume.

    Uses SimpleITK's GDCM series reader on a tempdir of symlinks so it can
    resolve spacing/orientation fields the same way it does for normal
    series, without caring about the original filename layout.
    """
    if not files:
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)

    tmp = tempfile.mkdtemp(prefix="chu_sr_")
    try:
        for i, src in enumerate(files):
            dst = Path(tmp) / f"{i:06d}.dcm"
            try:
                dst.symlink_to(src.resolve())
            except OSError:
                shutil.copy2(src, dst)

        reader = sitk.ImageSeriesReader()
        series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(tmp)
        if not series_ids:
            log.error(f"    GDCM found no series in tmp stage ({len(files)} files)")
            return False
        if len(series_ids) > 1:
            log.warning(f"    GDCM found {len(series_ids)} series in tmp stage, "
                        f"using first")
        names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(tmp, series_ids[0])
        reader.SetFileNames(names)
        img = reader.Execute()
        sitk.WriteImage(img, str(output_path))
        return True
    except Exception as e:
        log.error(f"    sitk read/write failed: {e}")
        return False
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def step_convert(args):
    raw_input = args.input
    work = args.work
    report_csv = work / "series_report.csv"
    staged_root = work / "staged"
    nifti_root = work / "nifti"

    if not report_csv.exists():
        log.error(f"Missing {report_csv}. Run chu_diagnose_series.py first.")
        sys.exit(1)

    picks = read_picks(report_csv)
    log.info(f"{len(picks)} picked series in report")

    nifti_root.mkdir(parents=True, exist_ok=True)
    ok, fail = [], []

    for r in picks:
        subject = r["subject"]
        tp = r["timepoint"]
        uid = r["series_uid"]
        out_dir = nifti_root / subject
        out_file = out_dir / f"{subject}_{tp}.nii.gz"

        if out_file.exists():
            log.info(f"[{subject}/{tp}] already exists, skipping")
            ok.append(subject)
            continue

        log.info(f"[{subject}/{tp}] locating DICOMs for series {uid[-20:]}")
        try:
            files = locate_series_files(subject, uid, raw_input, staged_root)
        except FileNotFoundError as e:
            log.error(f"  {e}")
            fail.append(subject)
            continue

        log.info(f"  {len(files)} files; desc='{r['series_description']}' "
                 f"n_slices_expected={r['n_slices']}")
        if not files:
            log.error(f"  no DICOMs matched series UID")
            fail.append(subject)
            continue

        success = convert_series_to_nifti(files, out_file)
        if success:
            img = sitk.ReadImage(str(out_file))
            log.info(f"  -> {out_file.name}  size={img.GetSize()} "
                     f"spacing={tuple(round(s, 3) for s in img.GetSpacing())}")
            ok.append(subject)
        else:
            fail.append(subject)

    log.info("")
    log.info(f"CONVERT summary: ok={len(ok)} fail={len(fail)}")
    if fail:
        log.error(f"Failed: {fail}")


def step_register(args):
    """Delegate to NACC's register logic against work/nifti -> work/registered."""
    work = args.work
    nifti_dir = work / "nifti"
    registered_dir = work / "registered"
    registered_dir.mkdir(parents=True, exist_ok=True)

    progress_file = registered_dir / "registration_progress.json"
    progress = load_progress(progress_file)

    # Build list of (subject_id, nifti_path)
    all_files = []
    for subj_dir in sorted(nifti_dir.iterdir()):
        if not subj_dir.is_dir():
            continue
        for f in subj_dir.glob("*.nii.gz"):
            if f.name.startswith("."):
                continue
            all_files.append((subj_dir.name, f))

    progress["total_files"] = len(all_files)
    log.info(f"Found {len(all_files)} NIfTI files to register")

    todo = [(s, p) for s, p in all_files
            if (registered_dir / s / f"{p.stem.replace('.nii', '')}_registered.nii.gz").exists() is False]

    for subject_id, nifti_path in all_files:
        subj_out = registered_dir / subject_id
        subj_out.mkdir(parents=True, exist_ok=True)
        stem = nifti_path.name.replace(".nii.gz", "")
        out_file = subj_out / f"{stem}_registered.nii.gz"

        if out_file.exists():
            log.info(f"[{subject_id}] already registered, skipping")
            if subject_id not in progress["processed"]:
                progress["processed"].append(subject_id)
            save_progress(progress_file, progress)
            continue

        log.info(f"[{subject_id}] registering {nifti_path.name}")
        success = process_nifti_file(
            str(nifti_path), str(out_file), str(args.template), subject_id
        )
        if success:
            progress["processed"].append(subject_id)
            log.info(f"  OK -> {out_file.name}")
        else:
            progress["failed"].append(subject_id)
            log.error(f"  FAIL")
        save_progress(progress_file, progress)

    log.info("")
    log.info(f"REGISTER summary: ok={len(progress['processed'])} "
             f"fail={len(progress['failed'])}")


def step_skull(args):
    """Delegate to NACC's skull-strip logic."""
    class NS:
        pass
    ns = NS()
    ns.input = str(args.work / "registered")
    ns.output = str(args.work / "skull")
    ns.resume = True
    nacc_run_skull(ns)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True,
                    choices=["convert", "register", "skull", "all"])
    ap.add_argument("--input", type=Path,
                    help="Raw CHU dir (needed for convert/all)")
    ap.add_argument("--work", required=True, type=Path,
                    help="Working dir (contains series_report.csv, staged/)")
    ap.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE,
                    help="MNI template NIfTI")
    args = ap.parse_args()

    if args.step in ("convert", "all") and args.input is None:
        ap.error("--input is required for convert/all")

    if args.step in ("convert", "all"):
        step_convert(args)
    if args.step in ("register", "all"):
        step_register(args)
    if args.step in ("skull", "all"):
        step_skull(args)


if __name__ == "__main__":
    main()

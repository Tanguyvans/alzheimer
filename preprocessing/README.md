# Medical Image Preprocessing Pipelines

Two preprocessing pipelines for MRI brain scans with **different outputs**:

| | **Pipeline 1: SynthStrip+ANTs** | **Pipeline 2: NPPY** |
|---|---|---|
| **Method** | Traditional (rule-based) | Learned (neural network) |
| **Steps** | DICOM→NIfTI → N4 Bias → MNI Register → Skull Strip | NIfTI → End-to-end learned preprocessing |
| **Intensity** | [0, 339], mean=165 | [-1, 121], mean=60 |
| **Use for** | General preprocessing, custom models | **3D HCCT models only** |
| **Time** | ~2.5 min/scan | ~5-10 sec/scan |
| **Result** | Clean traditional preprocessing | Optimized for ADNI dataset |

## ⚠️ Critical: Which Pipeline to Use?

- **Training 3D HCCT model?** → Use **Pipeline 2 (NPPY)** ✅
- **Reproducing 3D HCCT paper?** → Use **Pipeline 2 (NPPY)** ✅
- **General preprocessing?** → Use Pipeline 1 (SynthStrip+ANTs)
- **Custom models?** → Use Pipeline 1 (SynthStrip+ANTs)

**Why?** Using Pipeline 1 with 3D HCCT causes severe overfitting (Train 100%, Val 65%). The original paper uses NPPY preprocessing optimized for ADNI data.

---

## Directory Structure

```
preprocessing/
├── dicom_to_nifti.py             # Shared module: DICOM→NIfTI conversion
├── skull_stripping.py            # Shared module: SynthStrip functions
├── squeeze_4d_scans.py           # Utility: Convert 4D→3D scans
│
├── pipeline_1_synthstrip/        # Traditional preprocessing (standalone scripts)
│   ├── dicom_to_nifti.py         # Step 1: DICOM→NIfTI
│   ├── register_to_mni.py        # Step 2: N4 bias + MNI registration
│   └── skull_strip.py            # Step 3: Skull stripping
│
└── pipeline_2_nppy/              # Learned preprocessing (for 3D HCCT)
    └── run_nppy_preprocessing.py # End-to-end NPPY processing
```

---

## Pipeline 2: NPPY (For 3D HCCT)

**Requirement**: NIfTI files already converted (use Pipeline 1 Step 1 if needed)

### Step 0: Select Quality Scans

**IMPORTANT**: NPPY preprocessing fails on high-resolution "Accelerated_Sagittal_MPRAGE__MSV22" scans (>20MB, from 2024-2025). The patient selection script automatically filters these out and selects alternative scans.

```bash
cd experiments/cn_mci_ad_3dhcct

# Select patients with quality baseline scans (filters dimensions + file size)
python3 00_get_required_patients.py --config config.yaml
```

**What it does**:
- Filters scans with bad dimensions (any axis <100)
- **Filters large files** (>20MB - problematic high-res scans)
- **Automatically tries alternative scans** for each patient if first scan fails
- Sorts by file size (prefers smaller, more compatible scans)
- Outputs: `required_patients.txt` + `required_scans.txt`

**Quality checks**:
1. ✅ All dimensions ≥ 100 (removes localizer scans)
2. ✅ File size ≤ 20MB (removes high-res scans incompatible with NPPY)
3. ✅ 3D shape (no 4D scans)

**Options**:
- `--min-dim 100` - Minimum dimension threshold (default: 100)
- `--max-size 20.0` - Maximum file size in MB (default: 20.0)
- `--blacklist file.txt` - Optional additional blacklist

### Step 0b: Handle 4D Scans (If Any Remain)

Some ADNI scans have 4D shape (256, 256, 170, 1) instead of 3D (256, 256, 170). These are filtered by Step 0, but if you encounter any:

```bash
# Convert 4D scans to 3D (creates .4d_backup files)
python3 preprocessing/squeeze_4d_scans.py \
  --scan-list experiments/cn_mci_ad_3dhcct/required_scans.txt
```

**What it does**: Converts (256, 256, 170, 1) → (256, 256, 170) by squeezing singleton dimensions

**Time**: ~1-2 seconds per scan

### ⚠️ Important: N3 Bias Correction Required

According to ADNI documentation, **N3 bias field correction should be applied to raw MPRAGE scans** before NPPY preprocessing. This removes intensity artifacts and improves NPPY results.

**Step 1: Apply N3 Correction (RECOMMENDED)**

```bash
# Apply N3 correction to required baseline scans
python3 preprocessing/apply_n3_correction.py \
  --input /Volumes/KINGSTON/ADNI_nifti \
  --output /Volumes/KINGSTON/ADNI_nifti_n3 \
  --scan-list experiments/cn_mci_ad_3dhcct/required_scans.txt
```

**Time**: ~30-60 seconds per scan (uses N4, the improved version of N3)

**Step 2: Run NPPY Preprocessing**

```bash
python3 preprocessing/pipeline_2_nppy/run_nppy_preprocessing.py \
  --input /Volumes/KINGSTON/ADNI_nifti_n3 \
  --output /Volumes/KINGSTON/ADNI_nppy \
  --scan-list experiments/cn_mci_ad_3dhcct/required_scans.txt
```

**Options**:
- `--resume` - Resume interrupted processing
- `--nppy-script ~/nppy_docker.py` - Custom NPPY script path
- `--scan-list` - Process only specific scans (recommended for baseline-only)

**Time**: ~5-10 seconds per scan after N3 correction

**What it does**:
- **N3 correction**: Removes intensity non-uniformity (ADNI standard preprocessing)
- **NPPY**: End-to-end learned preprocessing (skull stripping + intensity + spatial normalization)
- Optimized specifically for ADNI dataset
- Produces intensity range [-1, 121] that 3D HCCT expects

**See**: [experiments/NPPY_SOLUTION_SUMMARY.md](../experiments/NPPY_SOLUTION_SUMMARY.md) for full details

### Quality Inspection Tools

After NPPY preprocessing, you can visually inspect the results to verify quality:

**Compare preprocessing stages side-by-side:**

```bash
# Compare NIFTI → SKULL → NPPY for specific patients
python3 utils/compare_preprocessing.py \
  --patient-list experiments/cn_mci_ad_3dhcct/required_patients.txt

# Or start at specific patient
python3 utils/compare_preprocessing.py --patient 128_S_0200
```

**Navigation**: Use arrow keys (←→ to change patients, ↑↓ to change slices)

**Identify problematic scans:**

```bash
# Extract patient IDs with large scans (>20MB) for inspection
python3 utils/extract_large_scan_patients.py \
  --scan-list experiments/cn_mci_ad_3dhcct/required_scans.txt \
  --max-size 20 \
  --output experiments/cn_mci_ad_3dhcct/large_scan_patients.txt

# Then inspect their NPPY quality
python3 utils/compare_preprocessing.py \
  --patient-list experiments/cn_mci_ad_3dhcct/large_scan_patients.txt
```

**Analyze file sizes:**

```bash
# Get detailed statistics on scan file sizes
python3 utils/filter_scans_by_size.py \
  --scan-list experiments/cn_mci_ad_3dhcct/required_scans.txt \
  --max-size 20
```

---

## Pipeline 1: SynthStrip + ANTs (Traditional)

Three standalone scripts for traditional preprocessing:

### Step 1: DICOM to NIfTI

```bash
python3 preprocessing/pipeline_1_synthstrip/dicom_to_nifti.py
```

Converts raw DICOM scans to NIfTI format using SimpleITK.

### Step 2: N4 Bias Correction + MNI Registration

```bash
python3 preprocessing/pipeline_1_synthstrip/register_to_mni.py
```

Uses ANTsPy to correct intensity artifacts and register to MNI152 template space.

### Step 3: Skull Stripping

```bash
python3 preprocessing/pipeline_1_synthstrip/skull_strip.py \
  --input /Volumes/KINGSTON/ADNI-registered \
  --output /Volumes/KINGSTON/ADNI-skull
```

Uses SynthStrip (Docker) to extract brain-only images. Automatically handles external drive mount issues on macOS.

**Requirements**:
- Docker Desktop installed and running
- SynthStrip image pulled automatically on first run

---

## Shared Modules

Both pipelines use shared utility modules:

- **[dicom_to_nifti.py](dicom_to_nifti.py)** - DICOM conversion functions
- **[skull_stripping.py](skull_stripping.py)** - SynthStrip Docker wrapper functions

These modules are imported by the pipeline scripts and can be used independently:

```python
from preprocessing.skull_stripping import synthstrip_skull_strip, setup_synthstrip_docker
from preprocessing.dicom_to_nifti import convert_dicom_to_nifti
```

---

## Requirements

```bash
# Activate virtual environment
source env/bin/activate

# Install dependencies
pip install SimpleITK antspyx numpy pandas tqdm python-dateutil>=2.7

# For Pipeline 1 Step 3: Install Docker Desktop
# https://www.docker.com/products/docker-desktop
```

---

## Key Differences Summary

| Aspect | Pipeline 1 | Pipeline 2 |
|--------|-----------|-----------|
| **Preprocessing approach** | Rule-based (N4, ANTs, SynthStrip) | Learned (NPPY neural network) |
| **Intensity output** | [0, 339], mean=165 | [-1, 121], mean=60 |
| **3D HCCT compatibility** | ❌ Causes overfitting | ✅ Original paper method |
| **Processing speed** | ~2.5 min/scan | ~5-10 sec/scan |
| **Use case** | General preprocessing | 3D HCCT only |

**For more details**: See [experiments/NPPY_SOLUTION_SUMMARY.md](../experiments/NPPY_SOLUTION_SUMMARY.md)

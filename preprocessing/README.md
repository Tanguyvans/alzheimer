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

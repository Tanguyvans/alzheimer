# Data Cleaning Pipeline for ADNI MRI Scans

Complete pipeline to transform raw DICOM brain scans into training-ready, skull-stripped NIfTI files for deep learning models.

## 📋 Pipeline Overview

The data cleaning process follows **3 sequential steps**:

```
Raw DICOM Files
    ↓
Step 1: DICOM → NIfTI Conversion
    ↓
Step 2: N4 Bias Correction + MNI Registration
    ↓
Step 3: Skull Stripping (SynthStrip)
    ↓
Training-Ready NIfTI Files
```

## 📁 Folder Structure

```text
data_cleaning/
├── README.md                           # This file
│
├── 01_dicom_conversion/                # Step 1: DICOM to NIfTI
│   └── convert_adni.py                 # MPRAGE sequence converter
│
├── 02_registration/                    # Step 2: N4 + MNI registration
│   └── run_registration_batch.py       # Batch processing script
│
├── 03_skull_stripping/                 # Step 3: Brain extraction
│   └── run_skull_strip_registered.py   # SynthStrip batch processing
│
└── scripts/                            # Optional utility scripts
    └── (helper scripts if needed)
```

## 🚀 Complete Workflow

### Prerequisites

```bash
# Python dependencies
pip install nibabel scipy SimpleITK numpy antspyx

# Docker (for SynthStrip skull stripping)
# Install Docker Desktop from: https://www.docker.com/products/docker-desktop

# ANTs (for registration)
# macOS: brew install ants
# Linux: Follow ANTs installation guide
```

**Note:** These batch scripts use the core preprocessing modules from [`preprocessing/`](../preprocessing/). The preprocessing folder contains reusable library functions for DICOM conversion, registration, and skull stripping.

### Step 1: DICOM to NIfTI Conversion

Convert raw DICOM scans to NIfTI format (only MPRAGE sequences).

```bash
cd data_cleaning/01_dicom_conversion

# Edit paths in convert_adni.py:
# - input_dir: Path to raw DICOM folders
# - output_dir: Where to save NIfTI files

python convert_adni.py
```

**Input structure:**
```
/Volumes/KINGSTON/ADNI/
├── patient1/
│   └── MPRAGE_sequence/
│       └── date/
│           └── *.dcm
└── patient2/
    └── ...
```

**Output:**
```
/Volumes/KINGSTON/ADNI_nifti/
├── patient1/
│   └── sequence_date_series.nii.gz
└── patient2/
    └── ...
```

**Progress tracking:**
- Shows real-time conversion progress
- Filters MPRAGE sequences automatically
- Reports total files converted

---

### Step 2: N4 Bias Correction + MNI Registration

Apply N4 bias field correction and register to MNI standard space.

```bash
cd data_cleaning/02_registration

# Run with default paths
python run_registration_batch.py

# Or specify custom paths
python run_registration_batch.py \
  --input /Volumes/KINGSTON/ADNI_nifti \
  --output /Volumes/KINGSTON/ADNI-registered \
  --template mni_template/mni_icbm152_t1_tal_nlin_sym_09a.nii

# Resume interrupted processing
python run_registration_batch.py --resume
```

**What it does:**
1. **N4 Bias Field Correction** - Removes intensity non-uniformity artifacts
2. **MNI Registration** - Aligns brains to standard MNI152 template space
3. **Resampling** - Ensures consistent voxel spacing

**Output:**
```
/Volumes/KINGSTON/ADNI-registered/
├── patient1/
│   └── sequence_date_series_registered.nii.gz
└── patient2/
    └── ...
```

**Progress tracking:**
- JSON file: `ADNI-registered/processing_progress.json`
- Saves progress after each file
- Can resume from interruptions (Ctrl+C)
- Shows ETA and average processing time

---

### Step 3: Skull Stripping

Remove skull and non-brain tissue using SynthStrip (FreeSurfer).

```bash
cd data_cleaning/03_skull_stripping

# Setup Docker (first time only)
docker pull freesurfer/synthstrip:latest

# Run skull stripping
python run_skull_strip_registered.py

# Or with custom paths
python run_skull_strip_registered.py \
  --input /Volumes/KINGSTON/ADNI-registered \
  --output /Volumes/KINGSTON/ADNI-skull

# Resume interrupted processing
python run_skull_strip_registered.py --resume
```

**What it does:**
- Removes skull, eyes, and non-brain tissue
- Uses SynthStrip (state-of-the-art deep learning method)
- Runs in Docker for consistency across systems

**Output (Training-Ready):**
```
/Volumes/KINGSTON/ADNI-skull/
├── patient1/
│   └── sequence_date_series_registered_skull_stripped.nii.gz
└── patient2/
    └── ...
```

**Progress tracking:**
- JSON file: `ADNI-skull/skull_stripping_progress.json`
- Saves progress after each file
- Can resume from interruptions
- Shows completion percentage and ETA

---

## 🎯 Quick Start (Complete Pipeline)

Process everything from raw DICOM to training-ready files:

```bash
# 1. Convert DICOM to NIfTI
cd data_cleaning/01_dicom_conversion
python convert_adni.py

# 2. Register to MNI space
cd ../02_registration
python run_registration_batch.py

# 3. Skull strip
cd ../03_skull_stripping
python run_skull_strip_registered.py

# 4. Done! Files ready in /Volumes/KINGSTON/ADNI-skull/
```

Total processing time: ~2-5 minutes per scan (depends on hardware)

---

## 📊 Data Organization by Diagnosis

After cleaning, organize files by diagnosis group for training:

```bash
# Create diagnosis folders
mkdir -p /Volumes/KINGSTON/ADNI-skull-organized/{AD,CN,MCI}

# Move files based on clinical diagnosis
# (Use your clinical CSV to map patients to diagnoses)

# Example structure:
/Volumes/KINGSTON/ADNI-skull-organized/
├── AD/
│   ├── patient1_scan.nii.gz
│   └── patient2_scan.nii.gz
├── CN/
│   └── ...
└── MCI/
    └── ...
```

Then use this organized data for model training:

```bash
# Prepare training CSV
python model_3d/utils/prepare_training_data.py \
  --input /Volumes/KINGSTON/ADNI-skull-organized \
  --output model_3d/data/training.csv
```

---

## 🔧 Configuration & Customization

### Modify Input/Output Paths

Edit the default paths in each script:

**`01_dicom_conversion/convert_adni.py`:**
```python
input_dir = "/Volumes/KINGSTON/ADNI"           # Raw DICOM
output_dir = "/Volumes/KINGSTON/ADNI_nifti"    # NIfTI output
```

**`02_registration/run_registration_batch.py`:**
```python
--input /Volumes/KINGSTON/ADNI_nifti           # NIfTI input
--output /Volumes/KINGSTON/ADNI-registered     # Registered output
--template mni_template/mni_icbm152_t1_tal_nlin_sym_09a.nii
```

**`03_skull_stripping/run_skull_strip_registered.py`:**
```python
--input /Volumes/KINGSTON/ADNI-registered      # Registered input
--output /Volumes/KINGSTON/ADNI-skull          # Skull-stripped output
```

### Resume Interrupted Processing

All scripts support graceful interruption and resumption:

```bash
# Press Ctrl+C to stop
# Progress is saved automatically

# Resume later with --resume flag
python run_registration_batch.py --resume
python run_skull_strip_registered.py --resume
```

---

## 📈 Expected Processing Times

Per scan (approximate):

| Step | Time | Notes |
|------|------|-------|
| **DICOM → NIfTI** | ~5-10s | Fast conversion |
| **Registration** | ~2-3 min | Most time-consuming |
| **Skull Stripping** | ~30-60s | Docker overhead included |
| **Total** | ~3-4 min | Full pipeline per scan |

For 1,000 scans: ~50-70 hours of processing time

---

## 🐛 Troubleshooting

### DICOM Conversion Issues

**Problem:** No files converted
```bash
# Check if MPRAGE sequences exist
ls -R /Volumes/KINGSTON/ADNI | grep -i mprage

# Verify DICOM files
ls /path/to/patient/sequence/date/ | grep .dcm
```

**Solution:** Ensure folder names contain "MPRAGE", "MP-RAGE", or "MP_RAGE"

---

### Registration Failures

**Problem:** "ANTs registration failed"
```bash
# Check if ANTs is installed
antsRegistration --version

# macOS installation:
brew install ants
```

**Problem:** Out of memory
```bash
# Reduce number of threads or use smaller batch sizes
# Edit preprocessing/image_enhancement.py to adjust thread count
```

---

### Skull Stripping Issues

**Problem:** "Docker not found"
```bash
# Install Docker Desktop
# https://www.docker.com/products/docker-desktop

# Pull SynthStrip image
docker pull freesurfer/synthstrip:latest

# Verify Docker is running
docker ps
```

**Problem:** "Permission denied"
```bash
# macOS: Grant Docker access to external drives
# Docker Desktop → Settings → Resources → File Sharing
# Add /Volumes/KINGSTON
```

---

## 📁 Output File Naming

Files are named systematically through the pipeline:

| Stage | Example Filename |
|-------|-----------------|
| **Raw DICOM** | `patient1/MPRAGE_seq/date/I00001.dcm` |
| **NIfTI** | `patient1/MPRAGE_seq_date_series.nii.gz` |
| **Registered** | `patient1/MPRAGE_seq_date_series_registered.nii.gz` |
| **Skull-stripped** | `patient1/MPRAGE_seq_date_series_registered_skull_stripped.nii.gz` |

For training, you can rename to simpler format:
```bash
# Example: 002_S_0729_bl.nii.gz
```

---

## 🔍 Quality Control

After processing, verify data quality:

```bash
# Check file integrity
python -c "import nibabel as nib; nii = nib.load('scan.nii.gz'); print(nii.shape)"

# Visualize a sample
# Use FSLeyes, ITK-SNAP, or 3D Slicer to inspect:
fsleyes /Volumes/KINGSTON/ADNI-skull/patient1/scan.nii.gz
```

**What to check:**
- ✅ Brain is centered and aligned
- ✅ Skull is completely removed
- ✅ No artifacts or distortions
- ✅ Consistent dimensions across scans

---

## 🔗 Integration with Model Training

After data cleaning, proceed to model training:

```bash
# 1. Organize by diagnosis (AD/CN/MCI folders)

# 2. Prepare training CSV
python model_3d/utils/prepare_training_data.py \
  --input /Volumes/KINGSTON/ADNI-skull-organized \
  --output model_3d/data/training.csv

# 3. Train model
python model_3d/training/train_densenet3d_clip.py \
  --data model_3d/data/training.csv \
  --gpus 1
```

See [`model_3d/README.md`](../model_3d/README.md) for complete training guide.

---

## 📚 References

- **ADNI Dataset**: http://adni.loni.usc.edu/
- **SynthStrip**: Hoopes et al., "SynthStrip: skull-stripping for any brain image", 2022
- **ANTs**: Avants et al., "Advanced Normalization Tools (ANTs)", 2011
- **N4 Bias Correction**: Tustison et al., "N4ITK: improved N3 bias correction", 2010
- **MNI Template**: ICBM 152 Nonlinear atlases (2009a)

---

## 💡 Best Practices

1. **Process in batches** - Monitor first few files before running full pipeline
2. **Use external storage** - Processing generates large temporary files
3. **Enable resume** - Always use `--resume` for long-running jobs
4. **Verify quality** - Check random samples at each stage
5. **Keep progress logs** - JSON files track what's been processed
6. **Backup raw data** - Keep original DICOM files separate

---

## 🆘 Support

For issues with:
- **DICOM conversion**: Check `preprocessing/dicom_to_nifti.py`
- **Registration**: Check `preprocessing/image_enhancement.py`
- **Skull stripping**: Check `preprocessing/skull_stripping.py`

See main project README for general support.

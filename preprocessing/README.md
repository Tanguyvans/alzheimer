# Medical Image Preprocessing Pipeline

Two preprocessing pipelines for MRI brain scans:
- **Pipeline 1 (SynthStrip+ANTs)**: Traditional preprocessing
- **Pipeline 2 (NPPY)**: Learned preprocessing for 3D HCCT model

## Directory Structure

```
preprocessing/
├── pipeline.py                    # Unified Pipeline 1 (uses modules below)
├── dicom_to_nifti.py             # Module: DICOM conversion
├── image_enhancement.py           # Module: N4 + MNI registration
├── skull_stripping.py             # Module: SynthStrip
├── pipeline_1_synthstrip/         # Standalone Pipeline 1 scripts
│   ├── dicom_to_nifti.py
│   ├── register_to_mni.py
│   └── skull_strip.py
├── pipeline_2_nppy/               # NPPY preprocessing (for 3D HCCT)
│   └── run_nppy_preprocessing.py
└── compare_preprocessing_interactive.py  # Compare both pipelines
```

---

## Pipeline 2: NPPY (Recommended for 3D HCCT)

**Use this to fix overfitting in 3D HCCT experiments!**

The original 3D HCCT paper uses NPPY preprocessing. Using Pipeline 1 causes severe overfitting (Train 100%, Val 65%).

### Quick Start - NPPY

```bash
cd pipeline_2_nppy
python3 run_nppy_preprocessing.py \
  --input /Volumes/KINGSTON/ADNI_nifti \
  --output /Volumes/KINGSTON/ADNI_nppy \
  --nppy-script ~/nppy_docker.py \
  --resume
```

**Time**: ~5-10 seconds per scan, ~1-2 hours for 636 scans

**What it does**:
- Skull stripping (learned)
- Intensity normalization (optimized for ADNI)
- Spatial normalization (standard space)

**See**: `experiments/NPPY_SOLUTION_SUMMARY.md` for full details

---

## Pipeline 1: SynthStrip + ANTs (Traditional)

Traditional preprocessing pipeline: **DICOM → NIfTI → N4 Bias Correction → MNI Registration → Skull Stripping**

### Pipeline 1 Steps

1. **DICOM to NIfTI Conversion** - Convert DICOM medical scans to NIfTI format
2. **N4 Bias Correction** - Remove intensity non-uniformity artifacts
3. **MNI Registration** - Register brain images to standard MNI template space
4. **Skull Stripping** - Brain extraction using SynthStrip (Docker-based)

## Requirements

- **Python 3.8+** with virtual environment
- **Docker Desktop** (for skull stripping step)
- **Required Python packages:**
  - SimpleITK
  - ANTs (antspyx) 
  - numpy, pandas, tqdm
  - python-dateutil>=2.7

## Quick Start

### 1. Setup Environment
```bash
# Activate your virtual environment
source env/bin/activate

# Ensure dependencies are up to date
pip install --upgrade python-dateutil
```

### 2. Run Full Pipeline
```bash
# Process ADNI data example
python3 preprocessing/pipeline.py \
  --input ADNI \
  --output ADNI_processed \
  --template mni_template/mni_icbm152_nlin_sym_09a_nifti/mni_icbm152_nlin_sym_09a/mni_icbm152_t1_tal_nlin_sym_09a.nii
```

### 3. Run Individual Steps
```bash
# Step 1: DICOM to NIfTI conversion only
python3 preprocessing/pipeline.py --input ADNI --output ADNI_processed --template [template] --step dicom

# Step 2: N4 bias correction + MNI registration only
python3 preprocessing/pipeline.py --nifti-input ADNI_processed/01_nifti --output ADNI_processed --template [template] --step process

# Step 3: Skull stripping only  
python3 preprocessing/pipeline.py --nifti-input ADNI_processed/01_nifti --output ADNI_processed --template [template] --step skull
```

## Output Structure

```
ADNI_processed/
├── 01_nifti/          # DICOM → NIfTI conversion results
├── 02_processed/      # N4 bias corrected + MNI registered images
└── 03_skull_stripped/ # Final brain-only images (ready for analysis)
```

## File Naming Convention

```
Original DICOM: ADNI_137_S_1414_MR_MP-RAGE__br_raw_20080226130530486_1_S46193_I92752.dcm
↓
NIfTI:         137_S_1414_MP-RAGE_2008-02-26_11_57_53.0_I92752_92752.nii.gz
↓  
Processed:     137_S_1414_MP-RAGE_2008-02-26_11_57_53.0_I92752_92752_processed.nii.gz
↓
Final:         137_S_1414_MP-RAGE_2008-02-26_11_57_53.0_I92752_92752_processed_skull_stripped.nii.gz
```

## Processing Time

- **DICOM → NIfTI**: ~0.5 seconds per scan
- **N4 + MNI Registration**: ~2 minutes per scan  
- **Skull Stripping**: ~40 seconds per scan
- **Total**: ~2.5 minutes per brain scan

## Docker Setup for Skull Stripping

The pipeline automatically:
1. Checks if Docker is running
2. Pulls `freesurfer/synthstrip:latest` image
3. Runs skull stripping in containers

**Install Docker Desktop**: https://www.docker.com/products/docker-desktop

## Supported Data Formats

- **Input**: DICOM files in ADNI folder structure
- **Template**: MNI152 NIfTI template  
- **Output**: Compressed NIfTI files (`.nii.gz`)

## Python API Usage

```python
from preprocessing import PreprocessingPipeline

# Initialize
pipeline = PreprocessingPipeline(
    output_root="ADNI_processed",
    mni_template="path/to/mni_template.nii"
)

# Run complete pipeline
results = pipeline.run_full_pipeline(dicom_root="ADNI")

# Or run steps individually  
nifti_files = pipeline.run_dicom_conversion("ADNI")
process_results = pipeline.run_processing(nifti_files)
skull_results = pipeline.run_skull_stripping()
```

## Troubleshooting

### Common Issues:
1. **ImportError: Matplotlib requires dateutil>=2.7**
   ```bash
   pip install --upgrade python-dateutil
   ```

2. **Docker not available**
   - Install Docker Desktop and ensure it's running
   - Check with: `docker --version`

3. **MNI template not found**
   - Verify the template path exists
   - Download MNI152 template if needed

### Logs:
- Pipeline creates `preprocessing_pipeline.log`
- Use `--log-level DEBUG` for detailed output

## Performance Notes

- Processing is sequential (one file at a time)
- Each step has progress bars and ETA estimates
- Intermediate files are preserved for resuming interrupted runs
- Already processed files are automatically skipped

---

## Comparing Pipelines

Visualize differences between SynthStrip+ANTs vs NPPY:

```bash
python3 compare_preprocessing_interactive.py --patient-id 035_S_6948
```

This opens an interactive viewer with sliders to explore preprocessing differences.

**Key differences**:
- **SynthStrip+ANTs**: Intensity range [0, 339], mean=165
- **NPPY**: Intensity range [-1, 121], mean=60
- **Impact**: Different intensity distributions cause 3D HCCT to overfit

---

## Which Pipeline Should I Use?

| Task | Use |
|------|-----|
| **3D HCCT model training** | **Pipeline 2 (NPPY)** |
| Reproducing 3D HCCT paper | **Pipeline 2 (NPPY)** |
| General preprocessing | Pipeline 1 (SynthStrip) |
| Custom model training | Pipeline 1 (SynthStrip) |

**Important**: If using 3D HCCT with Pipeline 1, expect severe overfitting. Use Pipeline 2 for Train/Val gap < 15%.
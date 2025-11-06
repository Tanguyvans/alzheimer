# Alzheimer's Disease Research Project

Comprehensive neuroimaging and machine learning pipeline for Alzheimer's disease classification and MCI progression prediction using ADNI dataset.

## Project Overview

This project combines medical image processing with machine learning to:

1. **Preprocess Brain MRI Scans** - DICOM to NIfTI conversion, bias correction, registration, and skull stripping
2. **Extract Clinical Features** - Neuropsychological assessments and cognitive test scores
3. **Classify CN vs AD** - Binary classification using XGBoost (99.6% accuracy)
4. **Predict MCI Direction** - Classify MCI patients as CN-like (stable) or AD-like (at risk)

## Dataset

### ADNI (Alzheimer's Disease Neuroimaging Initiative)

- **Total**: 1,980 MRI scans from 1,472 unique patients
- **CN (Cognitively Normal)**: 859 scans (564 patients)
- **MCI (Mild Cognitive Impairment)**: 845 scans (681 patients)
- **AD (Alzheimer's Disease)**: 276 scans (234 patients)

### NPPY Preprocessed Dataset (CN/MCI/AD - Stable Patients)

A quality-filtered dataset of NPPY (Neural Preprocessing Python) preprocessed MRI scans for stable patients with baseline-only diagnosis:

- **Archive**: `adni_stable_cn481_mci394_ad204_nppy.tar.gz` (1.79 GB)
- **Total**: 1,079 scans from stable patients
- **CN (Cognitively Normal)**: 481 scans
- **MCI (Mild Cognitive Impairment)**: 394 scans
- **AD (Alzheimer's Disease)**: 204 scans

**Quality Filtering Applied**:
- Dimension whitelist: 176×240×256, 160×192×192, 240×256×208
- File size filter: ≤20MB
- Manual blacklist: 1 patient (073_S_4230)
- Stable diagnosis: baseline visit only (diagnosis never changes)

**Preprocessing Pipeline**:
- NPPY (Neural Preprocessing Python) - End-to-end learned preprocessing
- Spatial normalization to MNI template
- Intensity normalization
- Output format: `*_mni_norm.nii.gz`

**Archive Structure**:
```
patient_id/scan_name
├── 018_S_0043/MPRAGE_Repeat_2009-01-21_10_42_57_mni_norm.nii.gz
├── 023_S_0058/MPRAGE_2008-10-10_10_35_48_mni_norm.nii.gz
└── ...
```

**Dataset Creation**:
```bash
python3 utils/create_nppy_dataset.py \
  --patient-list experiments/cn_mci_ad_3dhcct/required_patients.txt \
  --nppy-dir /Volumes/KINGSTON/ADNI_nppy \
  --dxsum /Volumes/KINGSTON/dxsum.csv \
  --blacklist experiments/cn_mci_ad_3dhcct/blacklist.txt
```

**Diagnosis Labels**: Obtained from `dxsum.csv` using `groupby('PTID').first()` - since patients are stable, the first visit diagnosis applies to all timepoints.

## Project Structure

```text
alzheimer/
├── docs/                              # Documentation
│   ├── README.md                      # Documentation index
│   ├── tabular/                       # Tabular ML documentation
│   │   ├── quickstart.md              # 5-minute getting started
│   │   └── features.md                # Clinical features guide
│   └── datasets/                      # Dataset documentation
│       ├── adni-dataset.md            # ADNI dataset overview
│       ├── diagnosis-progression-analysis.md  # Longitudinal analysis
│       └── data-analysis-guide.md     # Analysis guide
│
├── data/                              # Clinical and tabular data
│   ├── AD_CN_clinical_data.csv       # Clean training data (CN vs AD)
│   └── clinical_data_all_groups.csv  # Full dataset (CN, MCI, AD)
│
├── data_analysis/                     # Data analysis scripts
│   ├── adni_analysis.py               # ADNI directory analysis
│   ├── analyze_dxsum.py               # Diagnosis progression analysis
│   └── create_docs_visualization.py  # Generate documentation visualizations
│
├── preprocessing/                     # MRI preprocessing pipeline
│   ├── dicom_to_nifti.py             # DICOM conversion
│   ├── image_enhancement.py          # N4 bias correction
│   ├── registration.py               # MNI template registration
│   └── skull_stripping.py            # Brain extraction (SynthStrip)
│
├── tabular/                           # Tabular ML analysis
│   └── xgboost/                      # XGBoost implementation
│       ├── train.py                  # Train CN vs AD classifier
│       └── run.py                    # Predict MCI direction
│
└── outputs/                           # Organized outputs
    └── tabular/
        ├── models/                    # Trained models
        ├── predictions/               # CSV predictions
        ├── visualizations/            # Charts and plots
        └── reports/                   # Analysis reports
```

## Quick Start

### Prerequisites

```bash
# Activate virtual environment
source env/bin/activate

# Ensure dependencies are installed
pip install xgboost scikit-learn pandas numpy matplotlib seaborn
```

### Train XGBoost Model

```bash
cd tabular/xgboost
python3 train.py
```

**Output**: Model achieves 99.6% accuracy on CN vs AD classification

### Predict MCI Direction

```bash
cd tabular/xgboost
python3 run.py
```

**Output**: Classifies 845 MCI patients as CN-like (60%) or AD-like (40%)

## Key Results

### Binary Classification (CN vs AD)

- **Accuracy**: 99.6%
- **Precision**: 98.9%
- **Recall**: 100% (perfect AD detection)
- **F1 Score**: 99.4%

### MCI Progression Prediction

- **CN-like (Stable)**: 509/845 patients (60.2%)
- **AD-like (At Risk)**: 336/845 patients (39.8%)

### Top Predictive Features

1. **TRABSCOR** (37.3%) - Trail Making Test B (executive function)
2. **CATANIMSC** (10.1%) - Category fluency (semantic memory)
3. **DSPANBAC** (7.6%) - Digit span backward (working memory)

## Documentation

Comprehensive documentation in [`docs/`](docs/):

### Tabular Machine Learning

- **[Quick Start Guide](docs/tabular/quickstart.md)** - Get started with XGBoost in 5 minutes
- **[Clinical Features Reference](docs/tabular/features.md)** - Complete guide to all 13 cognitive test features

### Dataset Documentation

- **[ADNI Dataset Overview](docs/datasets/adni-dataset.md)** - Complete dataset reference with statistics
- **[Diagnosis Progression Analysis](docs/datasets/diagnosis-progression-analysis.md)** - Longitudinal progression patterns
- **[Data Analysis Guide](docs/datasets/data-analysis-guide.md)** - How to analyze the data with code examples

## Preprocessing Pipeline

Located in `preprocessing/`:

1. **DICOM to NIfTI** - Convert medical scans to standard format
2. **N4 Bias Correction** - Remove intensity artifacts
3. **MNI Registration** - Register to standard brain template
4. **Skull Stripping** - Extract brain-only images using SynthStrip

See [CLAUDE.md](CLAUDE.md) for detailed pipeline commands.

## Technology Stack

- **Python 3.12** with virtual environment
- **Medical Imaging**: SimpleITK, nibabel, MONAI, ANTs, nilearn, dicom2nifti
- **Machine Learning**: XGBoost, scikit-learn
- **Data Processing**: NumPy, pandas, matplotlib, seaborn

## Clinical Significance

### For Research

- Early detection of cognitive decline
- MCI progression risk assessment
- Treatment response prediction

### For Clinical Practice

- Risk stratification for MCI patients
- Personalized monitoring strategies
- Early intervention planning

## Research Context

- **Primary Goal**: Early Alzheimer's detection through hippocampus morphometry
- **Secondary Goal**: MCI progression prediction using cognitive assessments
- **Classification**: CN (Cognitively Normal) vs MCI (Mild Cognitive Impairment) vs AD (Alzheimer's Disease)

## References

- **ADNI**: [Alzheimer's Disease Neuroimaging Initiative](http://adni.loni.usc.edu/)
- **Dataset**: 1,472 unique patients, 1,980 MRI scans
- **Model**: XGBoost with 300 estimators, early stopping
- **Training**: 70/10/20 split (train/val/test)

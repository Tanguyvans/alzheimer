# Alzheimer's Disease Research Project

Comprehensive neuroimaging and machine learning pipeline for Alzheimer's disease classification and MCI progression prediction using ADNI dataset.

## Project Overview

This project combines medical image processing with machine learning to:

1. **Preprocess Brain MRI Scans** - DICOM to NIfTI conversion, bias correction, registration, and skull stripping
2. **Extract Clinical Features** - Neuropsychological assessments and cognitive test scores
3. **Classify CN vs AD** - Binary classification using XGBoost (99.6% accuracy)
4. **Predict MCI Direction** - Classify MCI patients as CN-like (stable) or AD-like (at risk)

## Dataset

**ADNI (Alzheimer's Disease Neuroimaging Initiative)**

- **Total**: 1,980 MRI scans from 1,472 unique patients
- **CN (Cognitively Normal)**: 859 scans (564 patients)
- **MCI (Mild Cognitive Impairment)**: 845 scans (681 patients)
- **AD (Alzheimer's Disease)**: 276 scans (234 patients)

## Project Structure

```
alzheimer/
├── docs/                              # Documentation
│   ├── ALZHEIMER_RESEARCH_GROUPS.md  # Group classifications and results
│   ├── TABULAR_README.md             # XGBoost implementation guide
│   └── TABULAR_METRICS_GUIDE.md      # Metrics and features explained
│
├── data/                              # Clinical and tabular data
│   ├── AD_CN_clinical_data.csv       # Clean training data (CN vs AD)
│   └── clinical_data_all_groups.csv  # Full dataset (CN, MCI, AD)
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

Detailed documentation in `docs/`:

- **[ALZHEIMER_RESEARCH_GROUPS.md](docs/ALZHEIMER_RESEARCH_GROUPS.md)** - Research group classifications, dataset statistics, and implementation results
- **[TABULAR_README.md](docs/TABULAR_README.md)** - XGBoost implementation guide, dataset analysis, and feature importance
- **[TABULAR_METRICS_GUIDE.md](docs/TABULAR_METRICS_GUIDE.md)** - Comprehensive metrics guide and clinical features

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

## License

Research use only. ADNI data usage subject to ADNI Data Use Agreement.

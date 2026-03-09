# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A medical imaging research project for Alzheimer's disease classification using MRI brain scans from multiple datasets (ADNI, OASIS, NACC). The project implements multimodal deep learning that fuses 3D brain MRI with clinical/tabular features for cognitive state classification.

**Primary Goal**: CN (Cognitively Normal) vs AD (Alzheimer's Disease) classification
**Secondary Goal**: MCI (Mild Cognitive Impairment) progression prediction
**Approach**: Multi-modal fusion combining 3D Vision Transformer (MRI) + FT-Transformer (tabular)

## Technology Stack

- **Python 3.12** with virtual environment in `env/`
- **Deep Learning**: PyTorch, PyTorch Lightning, MONAI
- **Medical Imaging**: SimpleITK, nibabel, ANTsPy, nilearn
- **ML Models**: 3D ViT, FT-Transformer, XGBoost
- **Experiment Tracking**: Weights & Biases (W&B)
- **Data Processing**: NumPy, pandas

## Project Structure

```
alzheimer/
├── preprocessing/           # MRI preprocessing pipelines
│   └── imaging/
│       ├── dicom_to_nifti.py
│       ├── skull_stripping.py (SynthStrip)
│       ├── apply_n3_correction.py
│       ├── pipeline_1_synthstrip/   # Traditional ANTs pipeline
│       └── pipeline_2_nppy/         # Neural network preprocessing
├── experiments/             # ML experiments
│   ├── multimodal_fusion/   # Main: ViT + FT-Transformer fusion
│   ├── mri_vit_ad/          # 3D Vision Transformer (MRI-only)
│   ├── xgboost_tabular/     # XGBoost on clinical features
│   ├── xgboost_longitudinal/# Longitudinal MCI prediction
│   ├── ablation_mri_only/   # Ablation: MRI baseline
│   └── ablation_tabular_only/# Ablation: Tabular baseline
├── data/                    # Clinical/tabular CSV data
│   ├── adni/, oasis/, nacc/ # Per-dataset clinical data
│   └── combined/            # Multi-cohort combined datasets
├── checkpoints/             # Saved model weights
├── mni_template/            # MNI brain template for registration
└── paper/                   # Research paper (LaTeX)
```

## Preprocessing Pipelines

### Pipeline 1: SynthStrip + ANTs (Traditional)
Rule-based preprocessing for general use:
1. **DICOM to NIfTI** - Convert medical scans
2. **N4 Bias Correction** - Remove intensity artifacts via ANTs
3. **MNI Registration** - Register to standard template
4. **Skull Stripping** - Extract brain via SynthStrip (Docker)

Output: intensity range [0, 339], ~2.5 min/scan

### Pipeline 2: NPPY (Neural Network-based)
Learned preprocessing optimized for model training:
- End-to-end neural preprocessing
- Includes N3 bias correction
- Output: intensity range [-1, 121], ~5-10 sec/scan
- **Required for 3D ViT/HCCT models**

## Model Architectures

### Multimodal Fusion (`experiments/multimodal_fusion/`)
Main architecture combining MRI and clinical features:

- **MRI Branch**: 3D Vision Transformer (ViT-base, 128x128x128 input, 768-dim features)
- **Tabular Branch**: FT-Transformer (16 clinical features, 64-dim output)
- **Fusion**: Cross-modal bidirectional attention with gated mechanism
- **Training**: Focal loss (alpha=0.75), auxiliary losses, TTA (8 augmentations)

Best result: 87.98% test accuracy, 85.29% balanced accuracy

### 3D Vision Transformer (`experiments/mri_vit_ad/`)
Custom ViT implementation for volumetric MRI:
- Patch size: 16x16x16 (512 patches from 128^3 volume)
- Variants: vit_tiny, vit_small, vit_base
- Supports MAE pre-training
- Layer-wise learning rate decay

### XGBoost Tabular (`experiments/xgboost_tabular/`)
Gradient boosting on clinical features:
- 16 features: demographics, cognitive tests, medical history
- Tasks: CN/AD binary, CN/MCI/AD 3-class, trajectory prediction
- Result: 84.7% accuracy on ADNI CN vs AD trajectory

## Key Development Commands

```bash
# Activate virtual environment
source env/bin/activate

# --- Preprocessing ---
# Run traditional preprocessing pipeline
python3 preprocessing/imaging/pipeline_1_synthstrip/run_pipeline.py \
  --input ADNI --output ADNI_processed \
  --template mni_template/mni_icbm152_nlin_sym_09a_nifti/mni_icbm152_nlin_sym_09a/mni_icbm152_t1_tal_nlin_sym_09a.nii

# Run NPPY preprocessing (for ViT training)
python3 preprocessing/imaging/pipeline_2_nppy/run_nppy_preprocessing.py

# --- Training ---
# Train multimodal fusion model
python3 experiments/multimodal_fusion/train.py --config experiments/multimodal_fusion/config.yaml

# Train multimodal with cross-validation
python3 experiments/multimodal_fusion/train_cv.py --config experiments/multimodal_fusion/config.yaml

# Train XGBoost on tabular data
python3 experiments/xgboost_tabular/train.py --config experiments/xgboost_tabular/configs/adni_cn_ad.yaml

# Train MRI-only ViT baseline
python3 experiments/mri_vit_ad/train.py --config experiments/mri_vit_ad/config.yaml

# --- Ablation Studies ---
python3 experiments/ablation_mri_only/train_cv.py
python3 experiments/ablation_tabular_only/train_cv.py
```

## Configuration System

All experiments use YAML configuration files:
- `experiments/*/config.yaml` - Main experiment configs
- `experiments/xgboost_tabular/configs/` - 14+ task/dataset configs

Key config sections: `experiment`, `model`, `data`, `training`, `wandb`

## Datasets

| Dataset | Patients | MRI Scans | CN | MCI | AD |
|---------|----------|-----------|-----|-----|-----|
| ADNI | 2,311 | 17,827 | 39% | 42% | 19% |
| OASIS | 1,340 | 7,794 | 68% | 3% | 16% |
| NACC | 55,004 | 8,163 | 49% | 18% | 29% |

Combined training set: 2,771 samples (71.3% CN, 28.7% AD)

## Tabular Features (16 total)

- **Demographics**: AGE, PTGENDER, PTEDUCAT, PTMARRY
- **Cognitive Tests**: CATANIMSC, TRAASCOR, TRABSCOR, DSPANFOR, DSPANBAC, BNTTOTAL
- **Medical History**: MH14ALCH, MH16SMOK, MH4CARD, MH2NEURL
- **Physical**: VSWEIGHT, BMI

## Data Flow

**MRI Path**: DICOM → NIfTI (.nii.gz) → Preprocessed volumes → NumPy arrays (.npy)
**Tabular Path**: CSV clinical data → Standardized features → FT-Transformer embeddings
**Fusion**: MRI features (768-d) + Tabular features (64-d) → Cross-modal attention → Classification

## File Naming Conventions

- `train.py` / `train_cv.py` - Training scripts (single run / cross-validation)
- `model.py` - Model architecture definitions
- `dataset.py` - Data loading and preprocessing
- `config.yaml` - Experiment configuration
- `prepare_dataset.py` - Dataset creation scripts

## Important Notes

1. **Preprocessing Choice**: Use NPPY pipeline (Pipeline 2) for ViT model training
2. **Class Imbalance**: Handled via focal loss (alpha=0.75) and class weighting
3. **Experiment Tracking**: All runs logged to W&B
4. **Checkpoints**: Best models saved to `checkpoints/` by validation metric
5. **Large Files**: MRI data, model weights, and outputs excluded from git (see .gitignore)

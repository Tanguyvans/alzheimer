# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a medical image analysis project for Alzheimer's disease and Multiple Sclerosis research using MRI brain scans. The project implements a comprehensive neuroimaging pipeline that processes DICOM brain scans, performs preprocessing, and uses deep learning for cognitive state classification (CN = Cognitively Normal vs MCI = Mild Cognitive Impairment).

## Technology Stack

- **Python 3.12** with virtual environment in `/env/`
- **Medical Imaging**: SimpleITK, nibabel, MONAI, ANTs, nilearn, dicom2nifti
- **Machine Learning**: PyTorch (3D ResNet), scikit-learn
- **Data Processing**: NumPy, pandas, matplotlib

## Core Pipeline Architecture

### Clean Preprocessing Pipeline (`preprocessing/`)
The main preprocessing pipeline provides a clean 4-step workflow:

1. **DICOM to NIfTI Conversion** - Convert medical scans to standard format
2. **N4 Bias Correction** - Remove intensity artifacts  
3. **MNI Registration** - Register to standard brain template
4. **Skull Stripping** - Extract brain-only images using SynthStrip

### Legacy Analysis Scripts
- `train_resnet3d.py` - 3D ResNet for Alzheimer's classification using hippocampus volumes (192x20x192 resolution)
- `ms-2-flames-segmentation.py` - FLAMeS (nnUNet) lesion segmentation
- `ms-volume.py` - Lesion volume quantification

## Key Development Commands

**Note**: This project uses a Python virtual environment in `/env/`.

```bash
# Activate virtual environment
source env/bin/activate

# Ensure dependencies are up to date
pip install --upgrade python-dateutil

# Run complete preprocessing pipeline (ADNI data example)
python3 preprocessing/pipeline.py \
  --input ADNI \
  --output ADNI_processed \
  --template mni_template/mni_icbm152_nlin_sym_09a_nifti/mni_icbm152_nlin_sym_09a/mni_icbm152_t1_tal_nlin_sym_09a.nii

# Run individual preprocessing steps
python3 preprocessing/pipeline.py --input ADNI --output ADNI_processed --template [template] --step dicom
python3 preprocessing/pipeline.py --nifti-input ADNI_processed/01_nifti --output ADNI_processed --template [template] --step process
python3 preprocessing/pipeline.py --nifti-input ADNI_processed/01_nifti --output ADNI_processed --template [template] --step skull

# Train Alzheimer's classification model
python train_resnet3d.py

# Visualize results
python visualize_processing_results.py
```

## Data Flow

**Input**: DICOM medical scans → **Intermediate**: NIfTI files (.nii.gz) → **Output**: NumPy arrays (.npy) for ML training

**Key Directories**:
- `output/npy/` - Processed brain volumes for training
- `output/register/` - Registered and preprocessed images
- `mni_template/` - MNI standard brain template for registration

## Important Technical Details

1. **Brain Registration**: All images are registered to MNI template space for standardized comparison
2. **Hippocampus Focus**: Primary region of interest for Alzheimer's classification
3. **Skull Stripping**: Multiple methods available (HD-BET, SynthStrip) - compare with `skull_comparison_plot.py`
4. **Image Enhancement**: N4 bias correction and denoising applied during preprocessing
5. **Volume Extraction**: Final volumes resized to 192x20x192 for consistent ML input

## Research Context

- **Primary Goal**: Early Alzheimer's detection through hippocampus morphometry
- **Secondary Goal**: Multiple Sclerosis lesion analysis using FLAIR sequences
- **Classification Task**: CN (Cognitively Normal) vs MCI (Mild Cognitive Impairment)
- **Model Architecture**: 3D ResNet adapted for medical imaging

## File Naming Conventions

- `ms-*` prefix: Multiple Sclerosis related processing
- `*_seg*`: Segmentation-related scripts
- `visualize_*`: Visualization and analysis tools
- Output files follow medical imaging standards (NIfTI .nii.gz format)
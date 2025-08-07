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

### Stage 1: DICOM to NIfTI Conversion
- `ms-1-dicom-nifti.py` - Standard T1 sequence conversion
- `dicom_flair_nifti_d2.py` - FLAIR sequence conversion

### Stage 2: Preprocessing & Registration
- `ms-1-register-python-skull.py` - MNI template registration
- `ms-1-nifti-enhanced.py` - N4 bias correction and enhancement
- `to_npy_seg_resize.py` - Complete preprocessing pipeline with skull stripping, hippocampus segmentation, and volume extraction

### Stage 3: Machine Learning Classification
- `train_resnet3d.py` - 3D ResNet for Alzheimer's classification using hippocampus volumes (192x20x192 resolution)

### Stage 4: Lesion Analysis (MS Research)
- `ms-2-flames-segmentation.py` - FLAMeS (nnUNet) lesion segmentation
- `ms-volume.py` - Lesion volume quantification

## Key Development Commands

**Note**: This project uses a Python virtual environment in `/env/`. No standard requirements.txt exists.

```bash
# Activate virtual environment
source env/bin/activate

# Run complete preprocessing pipeline
python to_npy_seg_resize.py

# Train Alzheimer's classification model
python train_resnet3d.py

# Process DICOM to NIfTI conversion
python ms-1-dicom-nifti.py

# Visualize processing results
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
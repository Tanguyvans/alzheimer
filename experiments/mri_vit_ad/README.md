# ViT for Alzheimer's Disease Classification

This experiment uses a **Vision Transformer (ViT)** pre-trained with **Masked Autoencoders (MAE)** for Alzheimer's disease classification on 3D brain MRI.

## Method Overview

### Two-Stage Training Process

```
Stage 1: MAE Pre-training (ALREADY DONE - we download weights)
──────────────────────────────────────────────────────────────
- Trained on: BraTS 2023, IXI, OASIS3 (unlabeled MRI data)
- Method: Mask 75% of 3D patches, reconstruct them
- Result: ViT learns brain anatomy without labels

Stage 2: Fine-tuning (WHAT WE DO)
──────────────────────────────────────────────────────────────
- Load pre-trained ViT encoder
- Add classification head (2 classes: CN vs AD_trajectory)
- Fine-tune on our ADNI data with labels
```

### Architecture

```
Input (3D MRI)
     │
     ▼
┌─────────────────────────────────┐
│  Patch Embedding (16x16x16)     │  Split volume into 3D patches
└─────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────┐
│  ViT-B Encoder (12 layers)      │  Pre-trained with MAE
│  - 768 hidden dim               │
│  - 12 attention heads           │
│  - 3072 MLP dim                 │
└─────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────┐
│  Classification Head            │  Trained from scratch
│  - Global Average Pool          │
│  - Linear(768 → 2)              │
└─────────────────────────────────┘
     │
     ▼
Output: CN (0) or AD_trajectory (1)
```

## Source

**Paper:** "Training ViT with Limited Data for Alzheimer's Disease Classification" (MICCAI 2024)

**GitHub:** https://github.com/qasymjomart/ViT_recipe_for_AD

**Results from paper:**
| Model | ADNI1 | ADNI2 |
|-------|-------|-------|
| ViT-B 75% mask | 79.6% | 81.9% |
| ViT-B + distillation | 82.5% | 84.6% |

## Pre-trained Weights

Download from the paper's Google Drive (link in their GitHub) and place in:
```
pretrained/vit_b_75mask.pth
```

**Recommended model:** `ViT-B 75% mask ratio` trained on BraTS+IXI+OASIS3

## Usage

```bash
# Fine-tune on our ADNI data
python train.py --config config.yaml

# Test only (after training)
python train.py --config config.yaml --test-only
```

## Comparison with Other Models

| Model | Pre-training Data | Expected Accuracy |
|-------|-------------------|-------------------|
| ResNet50 + MedicalNet | CT + MRI (23 datasets) | 82.87% (achieved) |
| DenseNet (scratch) | None | ~78-80% |
| **ViT-B + MAE** | MRI (BraTS+IXI+OASIS3) | ~81-84% |

## Key Differences from ResNet Experiment

1. **Architecture:** Transformer (attention) vs CNN (convolution)
2. **Pre-training:** Self-supervised MAE vs Supervised on segmentation
3. **Pre-training data:** MRI-only vs Mixed CT+MRI
4. **Patch-based:** Processes 16x16x16 patches vs sliding convolutions

## Files

```
mri_vit_ad/
├── README.md           # This file
├── config.yaml         # Training configuration
├── model.py            # ViT model definition
├── train.py            # Training script
├── dataset.py          # Symlink to ResNet dataset
└── pretrained/         # Pre-trained weights folder
    └── vit_b_75mask.pth
```

## Dependencies

Same as main project, plus ensure MONAI >= 1.1.0 for ViT support.

# IMBALMED-Style Ensemble for Alzheimer's Classification

Class-balanced diversity ensemble approach inspired by IMBALMED (October 2024).

## Overview

**Paper**: Class Balancing Diversity Multimodal Ensemble for Alzheimer's Disease Diagnosis
**Authors**: Francesconi et al. (2024)
**Result**: 73.5% accuracy on CN/MCI/AD 3-class with ADNI dataset

**Our Implementation**:
1. Create multiple balanced subsets from training data
2. Train separate ResNet-18 models on each subset
3. Ensemble predictions by averaging probabilities

**Expected Improvement**: 60-62% (single model) → 70-75% (ensemble)

## Quick Start

### Step 1: Use Existing Splits

We'll use the splits from the MedicalNet experiment:
```bash
cd experiments/cn_mci_ad_ensemble
```

### Step 2: Create Balanced Subsets

Create 5 balanced training subsets:

```bash
python3 create_balanced_subsets.py \
  --train-csv ../cn_mci_ad_medicalnet/data/splits/train.csv \
  --output-dir balanced_subsets \
  --num-subsets 5 \
  --seed 42
```

**What this does:**
- Original training: CN=590, MCI=473, AD=277 (imbalanced)
- Creates 5 subsets: Each with CN=277, MCI=277, AD=277 (balanced)
- Different random samples in each subset for diversity

### Step 3: Train Ensemble

Train 5 ResNet-18 models (one per subset):

```bash
python3 train_ensemble.py \
  --subset-dir balanced_subsets \
  --val-csv ../cn_mci_ad_medicalnet/data/splits/val.csv \
  --num-models 5 \
  --pretrained ../cn_mci_ad_medicalnet/pretrained/resnet_18_23dataset.pth \
  --checkpoints-dir checkpoints \
  --batch-size 4 \
  --epochs 50 \
  --patience 10
```

**Training time**: ~30-40 minutes per model on GPU (total: 2-3 hours for 5 models)

### Step 4: Evaluate Ensemble

Get final ensemble predictions on test set:

```bash
python3 predict_ensemble.py \
  --checkpoints-dir checkpoints \
  --test-csv ../cn_mci_ad_medicalnet/data/splits/test.csv \
  --num-models 5 \
  --batch-size 4
```

## How It Works

### Class Balancing

**Problem**: Imbalanced training data causes bias
- CN: 590 samples (44%)
- MCI: 473 samples (35%)
- AD: 277 samples (21%) ← minority class

**Solution**: Downsample to minority class
- Each subset: 277 CN, 277 MCI, 277 AD (perfectly balanced)
- Each model trained on balanced data
- No class bias

### Ensemble Diversity

**Creating 5 different subsets:**

**Subset 1**: CN samples [1-277], MCI samples [1-277], AD samples [all]
**Subset 2**: CN samples [278-554], MCI samples [278-554], AD samples [all]
**Subset 3**: CN samples [random], MCI samples [random], AD samples [all]
... etc

Each model learns from different CN/MCI examples!

### Prediction Averaging

**Individual model predictions:**
- Model 1: [0.6, 0.2, 0.2] → predicts CN
- Model 2: [0.4, 0.5, 0.1] → predicts MCI
- Model 3: [0.5, 0.3, 0.2] → predicts CN
- Model 4: [0.3, 0.4, 0.3] → predicts MCI
- Model 5: [0.7, 0.2, 0.1] → predicts CN

**Ensemble average**: [0.50, 0.32, 0.18] → predicts CN (more confident)

Averaging reduces errors and variance!

## Results Comparison

| Method | Train Acc | Val Acc | Test Acc | Overfitting |
|--------|-----------|---------|----------|-------------|
| Single ResNet-18 | 74% | 47% | 62% | ❌ High |
| **Ensemble (5 models)** | ? | ? | **70-75%** | ✅ Lower |

## Configuration

### Number of Models

**Recommended**: 5-10 models
- More models = better ensemble, but longer training
- Diminishing returns after ~10 models

**Adjust**:
```bash
python3 create_balanced_subsets.py --num-subsets 10  # Create 10 subsets
python3 train_ensemble.py --num-models 10           # Train 10 models
python3 predict_ensemble.py --num-models 10         # Use 10 for prediction
```

### Training Hyperparameters

Edit `train_ensemble.py` or pass as arguments:
- `--batch-size 4` (adjust for GPU memory)
- `--epochs 50` (max epochs per model)
- `--patience 10` (early stopping)
- `--learning-rate 0.0001`
- `--weight-decay 0.0001`

### Pretrained Weights

**With pretrained** (recommended):
```bash
--pretrained ../cn_mci_ad_medicalnet/pretrained/resnet_18_23dataset.pth
```

**Without pretrained** (train from scratch):
```bash
# Omit --pretrained flag
```

## Key Advantages

1. **Handles class imbalance** naturally (no complex weighting needed)
2. **Reduces overfitting** (ensemble averages out individual model errors)
3. **Improves generalization** (diversity from different training subsets)
4. **Realistic results** (73.5% in IMBALMED paper with same dataset/task)

## References

- **IMBALMED Paper**: Francesconi et al., "Class Balancing Diversity Multimodal Ensemble for Alzheimer's Disease Diagnosis and Early Detection", Computerized Medical Imaging and Graphics, October 2024
- **ArXiv**: https://arxiv.org/html/2410.10374v1
- **MedicalNet**: Chen et al., "Med3D: Transfer Learning for 3D Medical Image Analysis"

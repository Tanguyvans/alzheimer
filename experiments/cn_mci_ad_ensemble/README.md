# IMBALMED-Style Ensemble for Alzheimer's Classification

Class-balanced diversity ensemble approach inspired by IMBALMED (October 2024).

## Overview

**Paper**: Class Balancing Diversity Multimodal Ensemble for Alzheimer's Disease Diagnosis
**Authors**: Francesconi et al. (2024)
**Result**: 73.5% accuracy on CN/MCI/AD 3-class with ADNI dataset

**Our Implementation**:
1. Create multiple balanced subsets from training data
2. Train separate ResNet-18/SEResNet-18 models on each subset
3. Ensemble predictions by averaging probabilities

**Architectures Available**:
- ResNet-18 (vanilla): 33.2M parameters
- SEResNet-18 (with SE blocks): 33.2M parameters (+87K for SE attention, +0.3%)

**Results**:
- Vanilla ResNet-18 ensemble: 63% test accuracy
- SEResNet-18 ensemble: Testing in progress (expected 70-75%)

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

**Option A: SEResNet-18 (Recommended - with SE attention blocks)**

Train 5 SEResNet-18 models with Squeeze-and-Excitation blocks:

```bash
python3 train_ensemble.py \
  --subset-dir balanced_subsets \
  --val-csv ../cn_mci_ad_medicalnet/data/splits/val.csv \
  --num-models 5 \
  --pretrained ../cn_mci_ad_medicalnet/pretrained/resnet_18_23dataset.pth \
  --checkpoints-dir checkpoints_seresnet \
  --batch-size 4 \
  --epochs 50 \
  --patience 10 \
  --use-se
```

**Option B: Vanilla ResNet-18 (baseline)**

Train 5 vanilla ResNet-18 models without SE blocks:

```bash
python3 train_ensemble.py \
  --subset-dir balanced_subsets \
  --val-csv ../cn_mci_ad_medicalnet/data/splits/val.csv \
  --num-models 5 \
  --pretrained ../cn_mci_ad_medicalnet/pretrained/resnet_18_23dataset.pth \
  --checkpoints-dir checkpoints \
  --batch-size 4 \
  --epochs 50 \
  --patience 10 \
  --no-se
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

## Architecture: SEResNet-18 with Squeeze-and-Excitation Blocks

### What are SE Blocks?

Squeeze-and-Excitation (SE) blocks add **channel attention** to ResNet with minimal overhead:

1. **Squeeze**: Global average pooling aggregates spatial information per channel
2. **Excitation**: Two FC layers learn channel-wise attention weights (with bottleneck)
3. **Scale**: Multiply original features by attention weights to recalibrate channels

**Benefits**:
- Emphasizes important feature channels, suppresses irrelevant ones
- Only +87K parameters (+0.3% overhead)
- Paper achieved 93.26% accuracy on ADNI with SEResNet-18

**Architecture**:
```
Input (B, C, D, H, W)
  ↓
ResNet Block: conv → BN → ReLU → conv → BN
  ↓
SE Block:
  - Squeeze: GlobalAvgPool3D → (B, C)
  - Excitation: FC(C→C/16) → ReLU → FC(C/16→C) → Sigmoid
  - Scale: multiply features by attention weights
  ↓
Add residual → ReLU → Output
```

### Implementation

- `model_seresnet3d.py`: SEResNet-18/34/50 with configurable SE blocks
- `SEBlock`: Channel attention module
- `SEBasicBlock`: ResNet basic block + SE attention
- Pretrained weight loading: Loads conv/BN layers from MedicalNet, randomly initializes SE blocks

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

| Method | Test Acc | Balanced Acc | CN Precision | MCI Precision | AD Precision |
|--------|----------|--------------|--------------|---------------|--------------|
| Single ResNet-18 (MedicalNet) | 61.9% | 61.0% | 0.66 | 0.50 | 0.66 |
| **Vanilla ResNet-18 Ensemble** | **63.0%** | **62.0%** | **0.70** | **0.52** | **0.60** |
| SEResNet-18 Ensemble (expected) | 70-75% | - | - | - | - |

**Vanilla ResNet-18 Ensemble Confusion Matrix** (Test Set, n=168):
```
           Predicted
           CN  MCI  AD
Actual CN  59   12   3   (80% recall)
      MCI  23   23  13   (39% recall)
       AD   2    9  24   (69% recall)
```

**Analysis**:
- Ensemble improves over single model (63.0% vs 61.9%)
- CN classification is strong (70% precision, 80% recall)
- MCI remains challenging (52% precision, 39% recall) - typical for this class
- AD classification is decent (60% precision, 69% recall)
- Main confusion: MCI samples misclassified as CN or AD (transitional stage is hard)

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
  - ArXiv: <https://arxiv.org/html/2410.10374v1>

- **SEResNet Paper**: Gupta et al., "Deep CNN ResNet-18 based model with attention and transfer learning for Alzheimer's disease detection", Neural Computing and Applications, 2024
  - 93.26% accuracy on ADNI dataset with SEResNet-18
  - Uses Squeeze-and-Excitation blocks for channel attention

- **MedicalNet**: Chen et al., "Med3D: Transfer Learning for 3D Medical Image Analysis"
  - Pretrained 3D ResNet models on 23 medical imaging datasets

# MedicalNet + XGBoost: Feature Extraction + Classical ML

Hybrid approach combining deep learning feature extraction with classical machine learning.

## Overview

**Pipeline:**
1. **MedicalNet ResNet-18** (pretrained) → Extract 512-dimensional features from 3D MRI
2. **XGBoost** → Train on extracted features for CN/MCI/AD classification

**Advantages:**
- ✅ Pretrained CNN extracts meaningful medical imaging features
- ✅ XGBoost handles small datasets well (strong regularization)
- ✅ No overfitting (compared to end-to-end deep learning)
- ✅ Fast training (feature extraction is one-pass, XGBoost trains in seconds)
- ✅ Interpretable (feature importance analysis)

## Quick Start

### Step 1: Extract Features

Use the trained MedicalNet model from the previous experiment:

```bash
cd experiments/cn_mci_ad_medicalnet_xgboost

# Extract features from train set
python3 extract_features.py \
  --csv ../cn_mci_ad_medicalnet/data/splits/train.csv \
  --pretrained ../cn_mci_ad_medicalnet/pretrained/resnet_18_23dataset.pth \
  --output features/train_features.csv \
  --device cuda

# Extract features from val set
python3 extract_features.py \
  --csv ../cn_mci_ad_medicalnet/data/splits/val.csv \
  --pretrained ../cn_mci_ad_medicalnet/pretrained/resnet_18_23dataset.pth \
  --output features/val_features.csv \
  --device cuda

# Extract features from test set
python3 extract_features.py \
  --csv ../cn_mci_ad_medicalnet/data/splits/test.csv \
  --pretrained ../cn_mci_ad_medicalnet/pretrained/resnet_18_23dataset.pth \
  --output features/test_features.csv \
  --device cuda
```

**Note**: Feature extraction takes ~1-2 minutes per 100 scans on GPU.

### Step 2: Train XGBoost

```bash
python3 train_xgboost.py \
  --train features/train_features.csv \
  --val features/val_features.csv \
  --test features/test_features.csv \
  --output results
```

**Note**: XGBoost training takes seconds!

## What Gets Saved

After training, you'll have:

```
results/
├── xgboost_model.json           # XGBoost model (JSON format)
├── xgboost_model.pkl            # XGBoost model (pickle format)
├── feature_importance.csv       # Feature importance scores
├── train_predictions.csv        # Training set predictions
├── val_predictions.csv          # Validation set predictions
└── test_predictions.csv         # Test set predictions
```

## Expected Results

Based on similar approaches in the literature:

- **End-to-end deep learning** (ResNet-18 trained from scratch): 47-62% accuracy (overfitting)
- **MedicalNet + XGBoost**: 70-85% accuracy (better generalization)

The hybrid approach should significantly outperform end-to-end training because:
- Pretrained features are already meaningful
- XGBoost handles small datasets better than deep learning
- Strong regularization prevents overfitting

## Feature Details

**Extracted features:**
- 512-dimensional vector from ResNet-18 (before final classification layer)
- Represents high-level semantic information about brain structure
- Each dimension captures different aspects (atrophy patterns, tissue density, etc.)

**Feature importance:**
- XGBoost provides interpretable feature importance scores
- See `results/feature_importance.csv` for top features
- Can help identify which brain regions/patterns are most discriminative

## Comparison with Other Approaches

| Approach | Train Acc | Val Acc | Test Acc | Overfitting |
|----------|-----------|---------|----------|-------------|
| 3D HCCT (end-to-end) | 82% | 56% | 59% | ❌ High |
| VECNN (end-to-end) | 92% | 47% | 62% | ❌ Very High |
| MedicalNet ResNet-18 (end-to-end) | 74% | 47% | 62% | ❌ High |
| **MedicalNet + XGBoost** | ? | ? | ? | ✅ Expected Low |

## Hyperparameter Tuning

Edit `train_xgboost.py` to adjust XGBoost parameters:

```python
params = {
    'max_depth': 4,              # Tree depth (3-6 recommended)
    'learning_rate': 0.1,        # Learning rate (0.01-0.3)
    'n_estimators': 300,         # Number of trees
    'subsample': 0.8,            # Row sampling
    'colsample_bytree': 0.8,     # Column sampling
    'min_child_weight': 3,       # Minimum samples per leaf
    'gamma': 0.1,                # Regularization
    'reg_alpha': 0.1,            # L1 regularization
    'reg_lambda': 1.0,           # L2 regularization
}
```

## Using Different Pretrained Models

Try different ResNet architectures:

**ResNet-10** (lighter, faster):
```bash
python3 extract_features.py \
  --pretrained ../cn_mci_ad_medicalnet/pretrained/resnet_10_23dataset.pth \
  ...
```

**ResNet-34** (more capacity):
```bash
python3 extract_features.py \
  --pretrained ../cn_mci_ad_medicalnet/pretrained/resnet_34_23dataset.pth \
  ...
```

## Next Steps

If this approach works well, you can:
1. **Multi-modal fusion**: Combine imaging features with clinical/cognitive data
2. **Ensemble**: Combine predictions from multiple models
3. **Cross-validation**: Run 5-fold CV for more robust evaluation
4. **Fine-tuning**: Fine-tune ResNet on ADNI before feature extraction

## References

- **MedicalNet**: Chen et al., "Med3D: Transfer Learning for 3D Medical Image Analysis"
- **XGBoost**: Chen & Guestrin, "XGBoost: A Scalable Tree Boosting System"

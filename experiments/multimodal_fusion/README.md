# Multi-Modal Fusion: MRI + Tabular Data

This experiment combines 3D brain MRI with clinical tabular features for Alzheimer's disease classification (CN vs AD trajectory).

## Architecture

```
MRI Input (128x128x128)              Tabular Input (19 features)
         |                                      |
   [MRI Backbone]                      [Tabular Encoder]
    ResNet3D-18                         MLP or FT-Transformer
         |                                      |
    512-dim features                      64-dim features
         |                                      |
         +------------- [Fusion] ---------------+
                           |
                    Attention/Gated/Concat
                           |
                      512-dim fused
                           |
                     [Classifier]
                           |
                    2 classes (CN/AD)
```

## Components

### MRI Backbone Options
| Backbone | Feature Dim | Notes |
|----------|-------------|-------|
| ResNet3D-18 | 512 | Default, good performance |
| ResNet3D-50 | 2048 | Larger, may need more memory |
| ViT (MAE pretrained) | 768 | Vision Transformer |

### Tabular Encoder Options

#### 1. MLP (default)
Simple multi-layer perceptron:
```
Input (19) -> Linear(128) -> ReLU -> Linear(64) -> ReLU -> Output (64)
```

#### 2. FT-Transformer (Feature Tokenizer Transformer)
Each feature is embedded independently, then processed through transformer layers:
```
19 features -> [Feature Tokenizers] -> 19 tokens (64-dim each)
                        |
                [CLS] + 19 tokens
                        |
              3x Transformer Layers (4 heads, GELU)
                        |
                  [CLS] output -> 64-dim
```

Reference: "Revisiting Deep Learning Models for Tabular Data" (Gorishniy et al., 2021)

### Fusion Methods
| Method | Description |
|--------|-------------|
| `concat` | Concatenate features + MLP |
| `gated` | Learnable gates for each modality |
| `attention` | Cross-attention between modalities |

## Tabular Features (19 total)

**Demographics:**
- AGE, PTGENDER, PTEDUCAT, PTMARRY

**Vitals:**
- VSWEIGHT, BMI

**Medical History:**
- MH14ALCH (alcohol), MH16SMOK (smoking), MH4CARD (cardiovascular)
- MHPSYCH (psychiatric), MH2NEURL (neurological)

**Neuropsychological Tests:**
- TRAASCOR (Trail Making A), TRABSCOR (Trail Making B), TRABERRCOM (errors)
- CATANIMSC (category fluency), CLOCKSCOR (clock drawing)
- BNTTOTAL (Boston Naming), DSPANFOR/DSPANBAC (digit span)

**Excluded (diagnostic criteria):**
- MMSCORE (MMSE) - used for diagnosis
- CDGLOBAL (CDR) - used for diagnosis

## Configuration Files

| Config | MRI Backbone | Tabular Encoder | Fusion |
|--------|--------------|-----------------|--------|
| `config.yaml` | ViT | MLP | gated |
| `config_resnet.yaml` | ResNet3D-18 | MLP | attention |
| `config_resnet_fttransformer.yaml` | ResNet3D-18 | FT-Transformer | attention |

## Usage

### Single Train/Test Split
```bash
# Prepare dataset (creates train/val/test CSVs)
python prepare_dataset.py

# Train with ResNet + MLP
python train.py --config config_resnet.yaml

# Train with ResNet + FT-Transformer
python train.py --config config_resnet_fttransformer.yaml
```

### Cross-Validation
```bash
# 5-fold CV with 3 seeds
python train_cv.py --config config_resnet.yaml --n-folds 5 --seeds 42 123 456

# FT-Transformer CV
python train_cv.py --config config_resnet_fttransformer.yaml --n-folds 5
```

## Results

### CN vs AD Trajectory (ADNI)

#### Experiment Log

| Date | Config | MRI Backbone | Tabular Encoder | Fusion | Test Acc | Balanced Acc | Notes |
|------|--------|--------------|-----------------|--------|----------|--------------|-------|
| 2024-12-15 | config_resnet.yaml | ResNet3D-18 | MLP [128,64] | attention | 84.56% | 84.20% | Single split |
| 2024-12-16 | config_resnet.yaml | ResNet3D-18 | MLP [128,64] | attention | 83.57% ± 2.65% | 83.38% ± 2.82% | 5-fold CV, 3 seeds |
| 2024-12-16 | config_resnet_fttransformer.yaml | ResNet3D-18 | FT-Transformer (d=64, h=4, L=3) | attention | 82.35% | 82.20% | Single split, early stop @21 |

#### Config Details

**config_resnet.yaml:**
- Backbone: ResNet3D-18 (512-dim), unfrozen
- Tabular: MLP [128 → 64], dropout=0.3
- Fusion: attention, hidden=512
- Training: lr=0.0001, epochs=50, batch=4

**config_resnet_fttransformer.yaml:**
- Backbone: ResNet3D-18 (512-dim), unfrozen
- Tabular: FT-Transformer (embed=64, heads=4, layers=3)
- Fusion: attention, hidden=512
- Training: lr=0.0001, epochs=50, batch=4

#### Baselines

| Model | Data | Accuracy | Notes |
|-------|------|----------|-------|
| XGBoost | Tabular only (19 features) | ~89.8% | CN vs AD (not trajectory) |
| ViT + Gated | MRI + Tabular (MLP) | 81.62% | Single split |

## File Structure

```
multimodal_fusion/
├── config.yaml                      # ViT + MLP config
├── config_resnet.yaml               # ResNet + MLP config
├── config_resnet_fttransformer.yaml # ResNet + FT-Transformer config
├── model.py                         # Model definitions
├── dataset.py                       # MultiModal dataset
├── train.py                         # Single split training
├── train_cv.py                      # Cross-validation training
├── prepare_dataset.py               # Data preparation
├── data/                            # Train/val/test CSVs
├── checkpoints*/                    # Saved models
└── results*/                        # Experiment results
```

## Requirements

- PyTorch >= 1.10
- MONAI (for ResNet3D)
- scikit-learn
- pandas, numpy
- nibabel (for MRI loading)

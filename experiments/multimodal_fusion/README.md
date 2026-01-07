# Multi-Modal Fusion: MRI + Tabular Data

This experiment combines 3D brain MRI with clinical tabular features for Alzheimer's disease classification (CN vs AD).

## Best Results

| Dataset | Samples | Model | Test Accuracy | Balanced Accuracy |
|---------|---------|-------|---------------|-------------------|
| **ADNI+OASIS+NACC** | **6,065** | ViT + FT-Transformer + Gated | **90.77%** | **84.07%** |
| ADNI+OASIS+NACC | 2,771 | ViT + FT-Transformer + Gated | 87.98% | 85.29% |

### Methodology Notes
- **Subject-level split**: One sample per patient (first visit only) - prevents data leakage
- **3D subject-level MRI**: Full 3D volumes, not 2D slices
- **Multi-cohort validation**: Trained on 3 independent datasets for generalizability
- **No augmentation before split**: Proper train/val/test separation

## Architecture

```
MRI Input (128x128x128)              Tabular Input (7 features)
         |                                      |
   [MRI Backbone]                      [Tabular Encoder]
   ViT (MAE pretrained)                  FT-Transformer
         |                                      |
    768-dim features                      32-dim features
         |                                      |
         +------------- [Gated Fusion] ---------+
                           |
                      512-dim fused
                           |
                     [Classifier]
                           |
                    2 classes (CN/AD)
```

## Dataset

### Combined Dataset (ADNI + OASIS + NACC)

| Source | Samples | CN | AD | % of Total |
|--------|---------|-----|-----|------------|
| ADNI | 903 | 424 | 479 | 14.9% |
| OASIS | 1,030 | 742 | 288 | 17.0% |
| NACC | 4,132 | 3,581 | 551 | 68.1% |
| **Total** | **6,065** | **4,747 (78.3%)** | **1,318 (21.7%)** | 100% |

### Data Splits
| Split | Samples |
|-------|---------|
| Train | 4,245 (70%) |
| Val | 910 (15%) |
| Test | 910 (15%) |

### Tabular Features (7 common across all datasets)

**Demographics (4):**
- AGE, PTGENDER, PTEDUCAT, PTMARRY

**Neuropsychological Tests (3):**
- CATANIMSC (category fluency - animals)
- TRAASCOR (Trail Making A)
- TRABSCOR (Trail Making B)

## Configuration

### Best Model: `configs/config_combined.yaml`

```yaml
model:
  backbone:
    type: vit
    architecture: vit_base
    pretrained_path: "../mri_vit_ad/pretrained/vit_mae75_pretrained.pth"
    feature_dim: 768

  tabular:
    type: fttransformer
    n_blocks: 2
    d_token: 32
    attention_dropout: 0.2

  fusion:
    method: gated
    hidden_dim: 512
    dropout: 0.3

training:
  epochs: 100
  batch_size: 4
  learning_rate: 0.00002
  weight_decay: 0.01
  optimizer: adamw
  scheduler: cosine
  warmup_epochs: 5
  use_weighted_loss: true

callbacks:
  early_stopping:
    patience: 25
    monitor: val_accuracy
```

## Usage

```bash
# Prepare combined dataset
python prepare_dataset.py \
  --dataset combined \
  --output-dir data/combined \
  --oasis-mri-dir /path/to/OASIS-skull \
  --nacc-mri-dir /path/to/NACC-skull

# Train
python train.py --config configs/config_combined.yaml
```

## Experiment History

| Date | Dataset | Samples | Backbone | Tabular | Fusion | Test Acc | Bal Acc | Notes |
|------|---------|---------|----------|---------|--------|----------|---------|-------|
| 2026-01-05 | Combined | 6,065 | ViT | FT-Trans | gated | **90.77%** | 84.07% | +NACC expanded |
| 2026-01-05 | Combined | 2,771 | ViT | FT-Trans | gated | 87.98% | 85.29% | 100 epochs |
| 2024-12-16 | ADNI | 903 | ResNet3D | MLP | attention | 83.57% | 83.38% | 5-fold CV |

## Comparison with Literature

Studies with **proper subject-level splits** (no data leakage):

| Study | Dataset | Samples | Accuracy | Notes |
|-------|---------|---------|----------|-------|
| **Ours** | ADNI+OASIS+NACC | 6,065 | **90.77%** | Multi-cohort, multimodal |
| 3D-CNN-VSwinFormer (2025) | ADNI | ~1,000 | 92.92% | Single cohort |
| Wen et al. framework | ADNI | ~1,000 | 83-88% | Open-source benchmark |

Note: Studies reporting 95-99% accuracy often have data leakage issues (2D slice splits, augmentation before split).

## Potential Improvements

1. **Class balancing**: Increase AD weight or use oversampling (current: 78% CN, 22% AD)
2. **Cross-validation**: Run 5-fold CV for more robust estimates
3. **Ensemble**: Combine multiple models or seeds
4. **Additional features**: Add more neuropsych tests if available in all datasets
5. **Attention visualization**: Add Grad-CAM for interpretability

## File Structure

```
multimodal_fusion/
├── configs/
│   └── config_combined.yaml     # Best config (ViT + FT-Trans + Gated)
├── data/
│   └── combined/                # Train/val/test CSVs
├── model.py                     # Model definitions
├── dataset.py                   # MultiModal dataset
├── train.py                     # Training script
├── prepare_dataset.py           # Data preparation
├── checkpoints/                 # Saved models
└── results/                     # Experiment results
```

## Requirements

- PyTorch >= 1.10
- MONAI (for 3D transforms)
- scikit-learn
- pandas, numpy
- nibabel (for MRI loading)

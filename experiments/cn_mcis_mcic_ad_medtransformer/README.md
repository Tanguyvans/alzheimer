# MedTransformer 4-Class Classification

**Task**: CN | MCI_stable | MCI_to_AD | AD classification using multi-view Vision Transformers

## Overview

| Property | Value |
|----------|-------|
| **Model** | MedTransformer (ViT + Cross-Attention) |
| **Input** | 3 orthogonal views × 10 slices × 224×224×3 |
| **Classes** | 4 (CN, MCI_stable, MCI_to_AD, AD) |
| **Backbone** | vit_small_patch16_224 (pretrained) |
| **Dataset** | ADNI longitudinal trajectories |

## Reference

**Paper**: "MedTransformer: Accurate AD Diagnosis for 3D MRI Images through 2D Vision Transformers"
- **arXiv**: https://arxiv.org/abs/2402.xxxxx (2024)
- **Key Innovation**: Multi-view 2D slices processed by ViT, fused with cross-attention

## Architecture

```
3D MRI Volume (182×218×182)
        │
        ├── Axial slices (10) ─────┐
        │   [horizontal cuts]      │
        │                          │
        ├── Coronal slices (10) ───┼── ViT Encoder (shared)
        │   [frontal cuts]         │      │
        │                          │      ▼
        └── Sagittal slices (10) ──┘   Slice embeddings
                                           │
                                           ▼
                              ┌─────────────────────────────┐
                              │  Dimension-specific Encoder │
                              │  (Self-Attention + CLS)     │
                              └─────────────────────────────┘
                                           │
                     ┌─────────────────────┼─────────────────────┐
                     ▼                     ▼                     ▼
                  Axial CLS           Coronal CLS          Sagittal CLS
                     │                     │                     │
                     └──────── Cross-Attention Fusion ──────────┘
                                           │
                                           ▼
                                    FC → 4 classes
```

### Key Components

1. **Slice Encoder**: Pretrained ViT extracts features from each 2D slice
2. **Dimension Encoder**: Transformer with learnable CLS token aggregates slices per view
3. **Cross-Attention**: Views attend to each other (axial↔coronal, coronal↔sagittal, etc.)
4. **Fusion**: Final classification from concatenated view representations

**Parameters**: ~25M (ViT-Small backbone)

## Multi-View Slice Extraction

```
Axial View (axis=2)      Coronal View (axis=1)    Sagittal View (axis=0)
    ┌───────┐                ┌───────┐                ┌───────┐
    │   ─   │                │   │   │                │   ─   │
    │   ─   │                │   │   │                │   │   │
    │   ─   │                │   │   │                │   ─   │
    └───────┘                └───────┘                └───────┘
  [horizontal]              [frontal]                 [side]
```

- **Slice Range**: 25%-75% of brain depth (middle 50%)
- **Slices per View**: 10 evenly spaced
- **Total Input**: 30 slices → 30×768 embeddings

## Usage

### Step 1: Prepare Dataset (symlink from 3D experiment)

```bash
mkdir -p data/splits
ln -s ../../cn_mcis_mcic_ad_3d/data/splits/train.csv data/splits/
ln -s ../../cn_mcis_mcic_ad_3d/data/splits/val.csv data/splits/
ln -s ../../cn_mcis_mcic_ad_3d/data/splits/test.csv data/splits/
```

Or copy the prepare script:
```bash
cp ../cn_mcis_mcic_ad_3d/01_prepare_dataset.py .
python 01_prepare_dataset.py --config config.yaml
```

### Step 2: Train

```bash
python train.py --config config.yaml
```

### Step 3: Monitor Training

```bash
# Watch logs
tail -f data/logs/training.log

# Check GPU usage (if on server)
nvidia-smi -l 1
```

## Key Configuration

```yaml
model:
  name: "MedTransformer"
  backbone: "vit_small_patch16_224"  # Pretrained ViT
  num_slices: 10                      # Per view
  embed_dim: 384                      # ViT-Small dimension
  use_cross_attention: true           # Enable view fusion

training:
  batch_size: 4                       # Memory-limited
  accumulation_steps: 4               # Effective batch = 16
  learning_rate: 0.0001
  warmup_epochs: 5
  epochs: 50
```

## Hardware Requirements

| Setting | Memory | Speed |
|---------|--------|-------|
| batch_size=4, num_slices=10 | ~12GB | ~5 min/epoch |
| batch_size=2, num_slices=20 | ~16GB | ~8 min/epoch |

**Recommended**: 16GB+ GPU (RTX 3090, A100, etc.)

## Expected Results

| Metric | Expected Range | Notes |
|--------|----------------|-------|
| Balanced Accuracy | 45-55% | 4-class is challenging |
| CN Recall | 60-70% | Easiest class |
| AD Recall | 50-60% | Clear atrophy patterns |
| MCI_stable Recall | 30-40% | Hard to distinguish |
| MCI_to_AD Recall | 25-35% | Hardest (subtle changes) |

**Comparison with Other Methods**:

| Method | Expected 4-class BalAcc |
|--------|-------------------------|
| 3D ResNet | 35-45% |
| 2D Dual CNN | 50-55% |
| **MedTransformer** | **45-55%** |
| XGBoost (tabular) | 70-75% |
| Multimodal (vkola) | 80-88% |

## Files

```
cn_mcis_mcic_ad_medtransformer/
├── config.yaml           # Configuration
├── model.py              # MedTransformer architecture
├── dataset.py            # Multi-view DataLoader
├── train.py              # Training script
├── data/
│   ├── splits/           # Train/val/test CSVs
│   ├── checkpoints/      # Model weights
│   ├── logs/             # Training logs
│   └── results/          # Metrics & predictions
└── README.md             # This file
```

## Advantages of MedTransformer

1. **Multi-view fusion**: Captures information from all 3 anatomical planes
2. **Pretrained ViT**: Leverages ImageNet features for transfer learning
3. **Cross-attention**: Views can attend to each other for better fusion
4. **Memory efficient**: 2D slices vs full 3D volume
5. **Interpretable**: Can visualize attention maps per view

## Troubleshooting

### Out of Memory
```yaml
# Reduce in config.yaml:
training:
  batch_size: 2
model:
  num_slices: 5
```

### Slow Training
```yaml
# Use lighter backbone:
model:
  backbone: "vit_tiny_patch16_224"
  embed_dim: 192
  num_heads: 3
```

### Using MedTransformerLite
```yaml
model:
  name: "MedTransformerLite"  # CNN encoder instead of ViT
```

## Citation

```bibtex
@article{medtransformer2024,
  title={MedTransformer: Accurate AD Diagnosis for 3D MRI Images
         through 2D Vision Transformers},
  author={...},
  journal={arXiv preprint},
  year={2024}
}
```

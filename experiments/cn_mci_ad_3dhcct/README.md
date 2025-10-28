# 3D HCCT for 3-Class Alzheimer's Classification (CN vs MCI vs AD)

This experiment implements the **3D HCCT** architecture for **3-class classification** of stable CN vs MCI vs AD patients on ADNI data.

**Reproduced from**: [arindammajee/Alzheimer-Detection-with-3D-HCCT](https://github.com/arindammajee/Alzheimer-Detection-with-3D-HCCT)

## Classification Task

This experiment classifies patients into three categories:
- **CN (Cognitively Normal)**: Label = 0
- **MCI (Mild Cognitive Impairment)**: Label = 1
- **AD (Alzheimer's Disease)**: Label = 2

**Important**: Only patients with **stable diagnoses** throughout their visits are included.

## Original Paper & Repository

**Title**: "Enhancing MRI-Based Classification of Alzheimer's Disease with Explainable 3D Hybrid Compact Convolutional Transformers"

**Authors**: Arindam Majee, Avisek Gupta, Sourav Raha, Swagatam Das

**Paper**: [ArXiv:2403.16175](https://arxiv.org/abs/2403.16175)

**Original Repository**: https://github.com/arindammajee/Alzheimer-Detection-with-3D-HCCT

**Original Performance**: 96.06% accuracy on ADNI dataset (CN/MCI/AD 3-class classification)

## Quick Start

```bash
# 1. Navigate to experiment directory
cd experiments/cn_mci_ad_3dhcct

# 2. Activate virtual environment
source ../../env/bin/activate

# 3. Configure your data paths in config.yaml
# Edit: dxsum_csv and skull_dir

# 4. Prepare dataset splits
python3 01_prepare_dataset.py --config config.yaml

# 5. Train the model
python3 train.py --config config.yaml
```

## Project Structure

```
cn_mci_ad_3dhcct/
├── README.md              # This file
├── config.yaml            # Configuration file
├── 01_prepare_dataset.py  # Data preparation script for 3-class
├── dataset.py             # Dataset loader
├── model_hcct.py          # 3D HCCT model architecture
├── train.py               # Training script
├── data/
│   └── splits/            # Train/val/test CSV files
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
├── data/checkpoints/      # Saved model checkpoints
├── data/logs/             # Training logs
└── data/results/          # Evaluation results
```

## Model Architecture

### Input Shape
- **Volume size**: 192 × 192 × 192 voxels
- **Channels**: 1 (grayscale MRI)

### 3D Convolutional Encoder
```
Conv3D(1, 32) → BatchNorm → ReLU → MaxPool
Conv3D(32, 64) → BatchNorm → ReLU → MaxPool
Conv3D(64, 128) → BatchNorm → ReLU → MaxPool
Conv3D(128, 256) → BatchNorm → ReLU → MaxPool
Conv3D(256, 512) → BatchNorm → ReLU → MaxPool
```

### Vision Transformer Encoder

**Important Architecture Note**: This model uses an unconventional design:
- After 5 max-pooling operations: 192 → 96 → 48 → 24 → 12 → 6
- Spatial dimensions: 6×6×6 = **216 positions**
- Conv5 output: **512 channels**
- The rearrange treats **512 channels as sequence tokens** and **216 as feature dimension**

Components:
- **Embedding**: 512 sequence tokens with 216-dim features each
- **Position embeddings**: Learnable (513 positions including CLS)
- **CLS token**: Learnable classification token (216-dim)
- **Transformer blocks**: 3 layers
  - Multi-head self-attention (8 heads, 27-dim per head)
  - Layer normalization
  - Feed-forward MLP (216 → 648 → 216)
  - Residual connections

### Classification Head
- **Hybrid pooling**: Combines CLS token + attention-weighted pooling
- **Final classifier**: Linear(2 × 216, 3) for 3-class output

### Parameter Count
- **Total parameters**: ~6.2M

## Training Configuration

### Default Settings (Matching Original Paper)

- **Batch size**: 3 (same as original)
- **Optimizer**: AdamW (lr=8e-5, weight_decay=0.01)
- **Scheduler**: ReduceLROnPlateau (patience=5, factor=0.5)
- **Loss**: Weighted Cross-Entropy (accounts for class imbalance)
- **Epochs**: 100 (with early stopping patience=20)
- **Dropout**: 0.25 (same as original for 3-class task)

### Memory Requirements

For 192³ volumes with batch size 3:
- **Batch size 3**: ~12-15 GB GPU memory
- **Batch size 2**: ~8-10 GB GPU memory
- **Batch size 1**: ~5-6 GB GPU memory

## Expected Results

Based on the original paper (3-class task):
- **Accuracy**: ~96%
- **Training time**: ~2-3 hours on RTX 4090 GPU

## Usage

### Step 1: Data Preparation

Configure your data paths in `config.yaml`:
```yaml
data:
  dxsum_csv: "/path/to/your/dxsum.csv"
  skull_dir: "/path/to/your/ADNI-skull"
```

Create train/val/test splits:
```bash
python3 01_prepare_dataset.py --config config.yaml
```

### Step 2: Training

Train the model:
```bash
python3 train.py --config config.yaml
```

### Step 3: Testing Only

Evaluate a trained model:
```bash
python3 train.py --config config.yaml --test-only
```

## Monitoring Training

### Weights & Biases
If wandb is enabled in `config.yaml`, visit https://wandb.ai to view real-time metrics.

### Local Logs
- Training progress printed to console
- Checkpoints saved to `data/checkpoints/`
- Best model: `data/checkpoints/best.pth`

## Differences from Binary Classification

| Aspect | Binary (CN vs AD) | 3-Class (CN vs MCI vs AD) |
|--------|------------------|--------------------------|
| Classes | 2 (CN, AD) | 3 (CN, MCI, AD) |
| Output neurons | 2 | 3 |
| Dropout | 0.1 | 0.25 (higher for harder task) |
| Learning rate | 1e-4 | 8e-5 (lower for stability) |
| Expected accuracy | >95% | ~96% |

## Troubleshooting

### Out of Memory
Reduce batch size in `config.yaml`:
```yaml
training:
  batch_size: 2  # or 1
```

### Class Imbalance
The model uses weighted cross-entropy by default. Check class distribution in dataset preparation output.

## Citation

If you use this code, please cite the original paper:

```bibtex
@article{majee2024hcct,
  title={Enhancing MRI-Based Classification of Alzheimer's Disease with
         Explainable 3D Hybrid Compact Convolutional Transformers},
  author={Arindam Majee and Avisek Gupta and Sourav Raha and Swagatam Das},
  year={2024}
}
```

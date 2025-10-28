# 3D Hybrid Compact Convolutional Transformers (3D HCCT) for Alzheimer's Classification

This experiment implements the **3D HCCT** architecture for binary CN vs AD classification on ADNI data.

**Reproduced from**: [arindammajee/Alzheimer-Detection-with-3D-HCCT](https://github.com/arindammajee/Alzheimer-Detection-with-3D-HCCT)

## Architecture Overview

The 3D HCCT combines:
- **3D Convolutional Encoder**: 5 Conv3D blocks (32→64→128→256→512 channels) for local feature extraction
- **Vision Transformer Encoder**: Multi-head self-attention for capturing long-range dependencies
- **Hybrid Pooling**: Combines CLS token and attention pooling for classification

**Key Advantages:**
- Captures both local (CNN) and global (Transformer) features
- State-of-the-art performance on ADNI (96.06% accuracy in original paper)
- Explainable via attention visualization and Grad-CAM

## Original Paper & Repository

**Title**: "Enhancing MRI-Based Classification of Alzheimer's Disease with Explainable 3D Hybrid Compact Convolutional Transformers"

**Authors**: Arindam Majee, Avisek Gupta, Sourav Raha, Swagatam Das

**Paper**: [ArXiv:2403.16175](https://arxiv.org/abs/2403.16175)

**Original Repository**: https://github.com/arindammajee/Alzheimer-Detection-with-3D-HCCT

**Original Performance**: 96.06% accuracy on ADNI dataset (CN/MCI/AD 3-class classification)

**Note**: This implementation adapts their code for binary classification (CN vs AD) using our preprocessed ADNI data.

## Quick Start

```bash
# 1. Navigate to experiment directory
cd experiments/cn_vs_ad_3dhcct

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
cn_vs_ad_3dhcct/
├── README.md              # This file
├── config.yaml            # Configuration file
├── 01_prepare_dataset.py  # Data preparation script
├── dataset.py             # Dataset loader (adapted for our data)
├── model_hcct.py          # 3D HCCT model architecture
├── train.py               # Training script
├── data/
│   └── splits/            # Train/val/test CSV files (created by step 1)
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
├── data/checkpoints/      # Saved model checkpoints
├── data/logs/             # Training logs
└── data/results/          # Evaluation results
```

## Setup

### 1. Prerequisites

All dependencies are already installed in the main environment:
- PyTorch >= 2.0
- nibabel
- scipy
- numpy
- pandas
- einops
- pyyaml
- tqdm
- scikit-learn
- wandb (optional, for logging)

### 2. Configure Data Paths

Edit `config.yaml` and set your data paths:

```yaml
data:
  dxsum_csv: "/path/to/your/dxsum.csv"
  skull_dir: "/path/to/your/ADNI-skull"
```

### 3. Prepare Dataset Splits

Create train/val/test splits (70/15/15):

```bash
python3 01_prepare_dataset.py --config config.yaml
```

This will create:

- `data/splits/train.csv` - Training set (stable CN/AD patients)
- `data/splits/val.csv` - Validation set
- `data/splits/test.csv` - Test set
- `data/splits/dataset_metadata.json` - Dataset statistics

## Usage

### Training

Train the 3D HCCT model from scratch:

```bash
python3 train.py --config config.yaml
```

### Testing Only

Evaluate a trained model on the test set:

```bash
python3 train.py --config config.yaml --test-only
```

### Configuration

Edit `config.yaml` to customize:
- **Model architecture**: Hidden size, number of layers, attention heads
- **Training**: Batch size, learning rate, epochs, optimizer
- **Data augmentation**: Enable/disable augmentation
- **Hardware**: GPU settings, mixed precision
- **Logging**: Weights & Biases integration

## Key Differences from Original Implementation

### Adaptations Made:

1. **Input Data Format**:
   - Original: Uses their preprocessing pipeline with `_mni_norm.nii.gz` files
   - Ours: Uses existing skull-stripped, registered NIfTI files from ADNI

2. **Classification Task**:
   - Original: 3-class (CN/MCI/AD)
   - Ours: 2-class (CN/AD) - binary classification

3. **Dataset Loading**:
   - Original: Custom file structure with train/val/test folders
   - Ours: CSV-based splits with `scan_path` and `label` columns

4. **Data Preprocessing**:
   - Original: Uses `nppy` for skull-stripping
   - Ours: Already preprocessed data (skull-stripping done separately)

5. **Training Loop**:
   - Added: Balanced accuracy metric
   - Added: Weights & Biases integration
   - Added: More flexible configuration via YAML
   - Added: Better logging and checkpointing

## Model Architecture Details

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
After 5 max-pooling operations, the spatial dimensions are reduced significantly.

### Vision Transformer Encoder
- **Embedding**: Flattened conv features (512 channels)
- **Position embeddings**: Learnable
- **CLS token**: Learnable classification token
- **Transformer blocks**: 3 layers (configurable)
  - Multi-head self-attention (8 heads)
  - Layer normalization
  - Feed-forward MLP (hidden_size → 2048 → hidden_size)
  - Residual connections

### Classification Head
- **Hybrid pooling**: Combines CLS token + attention-weighted pooling of all patches
- **Final classifier**: Linear(2 × hidden_size, num_classes)

### Parameter Count
- **Total parameters**: ~15-20M (depends on config)
- **Trainable**: All layers

## Training Configuration

### Default Settings

- **Batch size**: 4 (192³ volumes are memory intensive)
- **Optimizer**: AdamW (lr=1e-4, weight_decay=0.01)
- **Scheduler**: ReduceLROnPlateau (patience=5, factor=0.5)
- **Loss**: Weighted Cross-Entropy (accounts for class imbalance)
- **Epochs**: 100 (with early stopping patience=20)

### Memory Requirements

For 192³ volumes:
- **Batch size 4**: ~16-20 GB GPU memory
- **Batch size 2**: ~10-12 GB GPU memory
- **Batch size 1**: ~6-8 GB GPU memory

If you run out of memory, reduce `batch_size` in `config.yaml`.

## Expected Results

Based on the original paper (3-class task):
- **Accuracy**: 96.06%
- **Precision**: High per-class precision
- **Training time**: ~2-3 hours on RTX 4090 GPU

For our **binary task (CN vs AD)**, we expect:
- **Higher accuracy** than the 3-class task (simpler problem)
- **Target**: >95% balanced accuracy on test set

## Monitoring Training

### Weights & Biases

If wandb is enabled in `config.yaml`:
1. Visit https://wandb.ai
2. Find your project: "alzheimer-research"
3. View metrics in real-time

### Local Logs

Training progress is printed to console with tqdm progress bars.

Checkpoints are saved to `checkpoints/`:
- `best.pth`: Best model based on validation accuracy
- `checkpoint_epoch_X.pth`: Periodic checkpoints every 10 epochs

## Troubleshooting

### Out of Memory (OOM)

Reduce batch size in `config.yaml`:
```yaml
training:
  batch_size: 2  # or even 1
```

Or enable mixed precision:
```yaml
hardware:
  mixed_precision: true
```

### No improvement in training

- Check learning rate (try 1e-5 or 5e-5)
- Enable data augmentation
- Increase dropout (try 0.2 or 0.3)
- Check class balance (use weighted loss)

### CUDA errors

Make sure PyTorch is installed with CUDA support:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## Comparison with Baseline

| Model | Architecture | Params | Val Acc | Test Acc |
|-------|-------------|---------|---------|----------|
| Baseline (3D ResNet) | CNN only | ~5M | TBD | TBD |
| **3D HCCT** | **CNN + Transformer** | **~15M** | **TBD** | **TBD** |

*(To be filled after training)*

## Next Steps

1. **Train the model**: Run `python train.py`
2. **Evaluate**: Check test set performance
3. **Visualize attention**: (Optional) Implement attention map visualization
4. **Compare with baseline**: Analyze performance differences
5. **Hyperparameter tuning**: Try different configurations

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

## Contact

For questions about this implementation, please refer to the main project documentation or the original repository.

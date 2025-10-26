# DenseNet3D + CLIP Training Guide for Alzheimer's Detection

Complete guide for training a DenseNet3D model with CLIP pretraining on ADNI skull-stripped MRI data.

## Overview

This training pipeline uses:
- **DenseNet3D** backbone for 3D MRI feature extraction
- **CLIP**-based pretraining for improved representation learning
- **PyTorch Lightning** for training orchestration
- **Patient-aware splits** to prevent data leakage
- **Class-balanced sampling** for handling class imbalance

## Quick Start

### 1. Prepare Your Data

First, create a CSV file with paths to your skull-stripped NIfTI files:

```bash
# For standard AD/CN/MCI organization:
python prepare_training_data.py \
  --input /Volumes/KINGSTON/ADNI-skull \
  --output data/adni_training.csv

# For organized MCI progression folders:
python prepare_training_data.py \
  --input /Volumes/KINGSTON/ADNI-MCI-organized \
  --output data/adni_mci_training.csv \
  --organized
```

This will create a CSV with columns:
- `nii_path`: Path to skull-stripped NIfTI file
- `Subject`: Patient ID (e.g., 002_S_0729)
- `RID`: Patient numeric ID
- `Group`: Diagnosis (AD, CN, MCI)
- `visit`: Baseline or follow-up visit

### 2. Install Dependencies

```bash
# Install PyTorch Lightning and other requirements
pip install pytorch-lightning tensorboard nibabel scipy pyyaml

# Make sure you have the ADNI_unimodal_models repo
git clone https://github.com/othmane42/ADNI_unimodal_models.git
```

### 3. Train the Model

```bash
# Basic training (CPU)
python train_densenet3d_clip.py \
  --data data/adni_training.csv \
  --config training_config.yaml \
  --batch-size 8 \
  --epochs 50 \
  --lr 0.0001

# GPU training
python train_densenet3d_clip.py \
  --data data/adni_training.csv \
  --config training_config.yaml \
  --batch-size 16 \
  --epochs 50 \
  --lr 0.0001 \
  --gpus 1

# Resume from checkpoint
python train_densenet3d_clip.py \
  --data data/adni_training.csv \
  --config training_config.yaml \
  --resume outputs/training/checkpoints/last.ckpt \
  --gpus 1
```

### 4. Monitor Training

View training progress with TensorBoard:

```bash
tensorboard --logdir outputs/training/tensorboard_logs
```

Open http://localhost:6006 in your browser to see:
- Training/validation loss curves
- Accuracy metrics
- Learning rate schedule
- Confusion matrices

## Configuration

### Training Parameters

Edit `training_config.yaml` to customize:

```yaml
training:
  batch_size: 8          # Increase with more GPU memory
  learning_rate: 0.0001  # Initial learning rate
  weight_decay: 0.00001  # L2 regularization
  max_epochs: 50         # Maximum training epochs
  num_workers: 4         # Data loading workers
  seed: 42               # Random seed for reproducibility
```

### Model Architecture

```yaml
model:
  freeze_encoder: false  # Set to true for fine-tuning only classifier
  hidden_dim: 0          # 0 = direct classification, >0 = add hidden layer
  dropout_rate: 0.0      # Dropout for regularization
```

### Early Stopping

```yaml
early_stopping:
  patience: 10           # Stop if no improvement for N epochs
  monitor: "val/loss"    # Metric to monitor
  mode: "min"            # "min" for loss, "max" for accuracy
```

## Command-Line Arguments

### Required Arguments

- `--data`: Path to CSV file with NIfTI paths and labels

### Optional Arguments

- `--config`: Config YAML file (default: `ADNI_unimodal_models/ADNI_dataset.yaml`)
- `--output`: Output directory (default: `outputs/training`)
- `--batch-size`: Batch size (default: 8)
- `--epochs`: Maximum epochs (default: 50)
- `--lr`: Learning rate (default: 1e-4)
- `--weight-decay`: Weight decay (default: 1e-5)
- `--num-workers`: DataLoader workers (default: 4)
- `--gpus`: Number of GPUs (default: 0 = CPU)
- `--seed`: Random seed (default: 42)
- `--freeze-encoder`: Freeze DenseNet3D weights
- `--resume`: Resume from checkpoint

## Output Files

Training creates the following outputs in `outputs/training/`:

```
outputs/training/
├── checkpoints/
│   ├── densenet3d_clip-epoch=XX-val_acc=0.XXXX.ckpt  # Best checkpoints
│   └── last.ckpt                                      # Latest checkpoint
└── tensorboard_logs/
    └── version_0/
        └── events.out.tfevents...                     # TensorBoard logs
```

## Using Trained Models

### For Inference

Use the existing prediction scripts with your trained checkpoint:

```bash
# Single file prediction
python predict.py \
  --model outputs/training/checkpoints/best.ckpt \
  --data /path/to/scan.nii.gz

# Batch prediction
python predict_batch.py \
  --input /path/to/AD_CN_folders \
  --model outputs/training/checkpoints/best.ckpt \
  --output predictions.csv

# MCI progression prediction
python predict_organized.py \
  --input /Volumes/KINGSTON/ADNI-MCI-organized \
  --model outputs/training/checkpoints/best.ckpt \
  --output mci_predictions.csv
```

### Loading Checkpoint in Python

```python
from train_densenet3d_clip import DenseNet3DCLIP

# Load model from checkpoint
model = DenseNet3DCLIP.load_from_checkpoint(
    'outputs/training/checkpoints/best.ckpt'
)
model.eval()

# Use for inference
with torch.no_grad():
    batch = {"image": input_tensor}
    logits = model(batch)
    probabilities = torch.softmax(logits, dim=1)
```

## Tips for Better Performance

### 1. Data Augmentation

Add augmentation to improve generalization:

```python
# In training_config.yaml, add custom transform
import torchio as tio

transforms = tio.Compose([
    tio.RandomAffine(scales=(0.9, 1.1), degrees=5),
    tio.RandomFlip(axes=(0,)),
    tio.RandomNoise(std=0.01)
])
```

### 2. Learning Rate Scheduling

The training script uses `ReduceLROnPlateau` by default:
- Reduces LR by 50% if validation loss doesn't improve for 5 epochs
- You can modify this in `DenseNet3DCLIP.configure_optimizers()`

### 3. Class Imbalance

The dataloader automatically handles class imbalance with `WeightedRandomSampler`:
- Oversamples minority classes during training
- Ensures balanced batches

### 4. GPU Memory Optimization

If you run out of GPU memory:
- Reduce `--batch-size` (try 4 or 2)
- Use gradient accumulation
- Use mixed precision training (FP16)

### 5. Prevent Overfitting

If validation accuracy is much lower than training:
- Add dropout: `dropout_rate: 0.2` in config
- Increase weight decay: `weight_decay: 0.0001`
- Use data augmentation
- Early stopping will help automatically

## Data Requirements

### Input Format

Each row in your CSV should have:
- **nii_path**: Absolute path to skull-stripped .nii.gz file
- **Subject**: Patient ID for patient-aware splitting
- **Group**: Diagnosis label (AD, CN, or MCI)

### Data Quality

Ensure your NIfTI files are:
- ✅ Skull-stripped (brain-only)
- ✅ MNI-registered (standardized space)
- ✅ N4 bias-corrected
- ✅ 3D volumes (not 4D time series)
- ✅ Reasonable dimensions (typically 128x128x128 or similar)

The `prepare_training_data.py` script validates files automatically.

## Troubleshooting

### Issue: "CUDA out of memory"

**Solution**: Reduce batch size
```bash
python train_densenet3d_clip.py --batch-size 2 --gpus 1 ...
```

### Issue: "No module named 'encoders'"

**Solution**: Ensure ADNI_unimodal_models is cloned
```bash
git clone https://github.com/othmane42/ADNI_unimodal_models.git
```

### Issue: "Invalid NIfTI file"

**Solution**: Validate your preprocessing pipeline
```bash
# Check a file manually
python -c "import nibabel as nib; nii = nib.load('scan.nii.gz'); print(nii.shape)"
```

### Issue: Low validation accuracy

**Solutions**:
1. Check data quality and labels
2. Increase training time (more epochs)
3. Try different learning rates
4. Add data augmentation
5. Check for data leakage (same patient in train/val)

## Advanced Usage

### Multi-GPU Training

```bash
# Use all available GPUs
python train_densenet3d_clip.py --gpus -1 --batch-size 32 ...

# Use specific GPUs
CUDA_VISIBLE_DEVICES=0,1 python train_densenet3d_clip.py --gpus 2 ...
```

### Mixed Precision Training

Add to trainer in `train_densenet3d_clip.py`:

```python
trainer = pl.Trainer(
    precision='16-mixed',  # FP16 training
    ...
)
```

### Custom Metrics

Add custom metrics in `DenseNet3DCLIP.validation_step()`:

```python
# Calculate AUC
from sklearn.metrics import roc_auc_score

probabilities = torch.softmax(all_logits, dim=1)
auc = roc_auc_score(all_labels.cpu(), probabilities.cpu(), multi_class='ovr')
self.log('val/auc', auc)
```

## References

- **ADNI Dataset**: [http://adni.loni.usc.edu/](http://adni.loni.usc.edu/)
- **DenseNet3D**: Dense Convolutional Networks for 3D Medical Imaging
- **CLIP**: Contrastive Language-Image Pre-training
- **PyTorch Lightning**: [https://www.pytorchlightning.ai/](https://www.pytorchlightning.ai/)

## Citation

If you use this training pipeline, please cite the ADNI dataset and relevant papers.

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review TensorBoard logs for training diagnostics
3. Verify data preparation with `prepare_training_data.py`

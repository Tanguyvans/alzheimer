# MedicalNet Transfer Learning for Alzheimer's Classification

Transfer learning experiment using pretrained 3D ResNet from MedicalNet for CN/MCI/AD classification.

## Overview

- **Model**: 3D ResNet-18 (pretrained on 23 medical imaging datasets)
- **Task**: 3-class classification (CN vs MCI vs AD)
- **Dataset**: 1,079 stable ADNI patients (patient-level splitting)
- **Advantage**: Reduces overfitting on small datasets vs training from scratch

## Setup

### 1. Download Pretrained Weights

Download MedicalNet pretrained ResNet-18 weights:

**Option A: From GitHub**
```bash
# Clone MedicalNet repo
git clone https://github.com/Tencent/MedicalNet.git
cp MedicalNet/pretrain/resnet_18.pth experiments/cn_mci_ad_medicalnet/pretrained/
```

**Option B: From Hugging Face**
```bash
mkdir -p pretrained
cd pretrained
wget https://huggingface.co/TencentMedicalNet/MedicalNet-Resnet18/resolve/main/resnet_18.pth
```

**Option C: Manual Download**
- Visit: https://github.com/Tencent/MedicalNet
- Download `resnet_18.pth` from pretrain folder
- Place in `experiments/cn_mci_ad_medicalnet/pretrained/`

### 2. Update Config

Edit [config.yaml](config.yaml):
- Set `pretrained_path` to your downloaded weights
- Adjust `skull_dir` and `dxsum_csv` paths for your setup

### 3. Generate Dataset Splits

```bash
cd experiments/cn_mci_ad_medicalnet
python3 01_prepare_dataset.py --config config.yaml
```

This creates train/val/test CSVs with **patient-level splitting** (80/10/10).

### 4. Train Model

```bash
python3 train.py --config config.yaml
```

## Key Features

### Transfer Learning
- Pretrained on 23 medical imaging datasets (Kinetics, CT scans, MRI scans)
- Only fine-tune on ADNI data
- Significantly reduces overfitting vs training from scratch

### Patient-Level Splitting
- All scans from one patient stay in same split (train/val/test)
- Prevents data leakage
- More realistic evaluation

### Architecture Options

Edit `config.yaml` to try different architectures:
```yaml
model:
  architecture: "resnet18"  # Options: resnet10, resnet18, resnet34, resnet50
```

**Recommended**: ResNet-18 (good balance of capacity and speed)

### Training Strategies

**Full Fine-Tuning** (default):
```yaml
model:
  freeze_backbone: false  # Train all layers
```

**Feature Extraction** (faster, less overfitting):
```yaml
model:
  freeze_backbone: true  # Only train FC layer
```

## Expected Results

With proper transfer learning, expect:
- **Without transfer learning**: 50-60% val accuracy (severe overfitting)
- **With transfer learning**: 70-85% val accuracy
- **VECNN paper (from scratch)**: 92% accuracy (but with 2,248 scans)

## Hyperparameters

Key settings in [config.yaml](config.yaml):
- Batch size: 4 (adjust based on GPU memory)
- Learning rate: 1e-4
- Epochs: 100 (with early stopping patience=15)
- Data augmentation: enabled
- Mixed precision: enabled (FP16)

## Monitoring

Training logs to Weights & Biases if enabled:
```yaml
wandb:
  enabled: true
  project: "alzheimer-research"
```

## References

- **MedicalNet Paper**: Med3D: Transfer Learning for 3D Medical Image Analysis
- **GitHub**: https://github.com/Tencent/MedicalNet
- **Hugging Face**: https://huggingface.co/TencentMedicalNet/

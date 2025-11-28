# 4-Class 3D MRI Classification

**Task**: CN | MCI_stable | MCI_to_AD | AD classification using 3D ResNet

## Overview

| Property | Value |
|----------|-------|
| **Model** | 3D ResNet-18 (MONAI) |
| **Input** | 192×192×192 skull-stripped MRI |
| **Classes** | 4 (CN, MCI_stable, MCI_to_AD, AD) |
| **Dataset** | ADNI longitudinal trajectories |

## Inspiration & References

### Primary Reference: MedicalNet (Tencent)
- **Paper**: [Med3D: Transfer Learning for 3D Medical Image Analysis](https://arxiv.org/abs/1904.00625)
- **GitHub**: https://github.com/Tencent/MedicalNet
- **Weights**: Pretrained on 23 medical imaging datasets

### Alternative: vkola-lab Multimodal
- **Paper**: [Nature Communications 2022 - Multimodal deep learning for Alzheimer's](https://www.nature.com/articles/s41467-022-31037-5)
- **GitHub**: https://github.com/vkola-lab/ncomms2022
- **Approach**: MRI + Clinical data fusion (achieves 82-88% accuracy)

### MONAI Framework
- **Docs**: https://monai.io/
- **ResNet3D**: Built-in 3D ResNet variants optimized for medical imaging

## 4-Class Labeling Strategy

Uses **longitudinal DXSUM data** to define disease trajectories:

```
Baseline DX → Last DX → Class
─────────────────────────────
CN         → CN       → 0: CN (stayed normal)
MCI        → MCI      → 1: MCI_stable (no conversion)
MCI        → AD       → 2: MCI_to_AD (converter)
AD         → *        → 3: AD (already diagnosed)
```

**Excluded**: CN→MCI, CN→AD (progression from normal)

## Usage

### Step 1: Prepare Dataset
```bash
python 01_prepare_dataset.py --config config.yaml
```

Creates patient-level stratified splits (70/15/15):
- `data/splits/train.csv`
- `data/splits/val.csv`
- `data/splits/test.csv`

### Step 2: Download Pretrained Weights (Optional)
```bash
mkdir -p pretrained
# Download from MedicalNet
wget -O pretrained/resnet_18.pth \
  https://huggingface.co/TencentMedicalNet/MedicalNet-Resnet18/resolve/main/resnet_18.pth
```

### Step 3: Train
```bash
python train.py --config config.yaml
```

## Key Configuration

```yaml
model:
  image_size: 192          # Volume size (192³)
  num_classes: 4

training:
  batch_size: 2            # Limited by GPU memory
  learning_rate: 0.0001
  epochs: 100
  use_weighted_loss: true  # Handle class imbalance
```

## Expected Results

| Metric | Expected Range |
|--------|----------------|
| Balanced Accuracy | 35-45% |
| CN Recall | 60-70% |
| AD Recall | 50-60% |
| MCI_to_AD Recall | 15-25% (hardest class) |

**Note**: 4-class is significantly harder than binary (CN vs AD ~85-90%).
MCI_stable vs MCI_to_AD distinction is subtle in neuroimaging.

## Hardware Requirements

- **GPU Memory**: 16GB+ recommended
- **Training Time**: ~5-10 min/epoch on RTX 3090
- **Disk**: ~50GB for skull-stripped MRI data

## Comparison with Other Approaches

| Approach | Classes | Expected Acc |
|----------|---------|--------------|
| 3D ResNet (this) | 4 | 35-45% |
| 2D Dual CNN | 4 | 50-65% |
| XGBoost Tabular | 4 | 70-75% |
| Multimodal (vkola) | 4 | 80-88% |

## Files

```
cn_mcis_mcic_ad_3d/
├── config.yaml           # Configuration
├── 01_prepare_dataset.py # Dataset preparation
├── dataset.py            # MRI DataLoader
├── train.py              # Training script
├── data/
│   ├── splits/           # Train/val/test CSVs
│   ├── checkpoints/      # Model weights
│   └── results/          # Metrics & plots
└── README.md             # This file
```

## Citation

If using MedicalNet weights:
```bibtex
@article{chen2019med3d,
  title={Med3D: Transfer Learning for 3D Medical Image Analysis},
  author={Chen, Sihong and Ma, Kai and Zheng, Yefeng},
  journal={arXiv preprint arXiv:1904.00625},
  year={2019}
}
```

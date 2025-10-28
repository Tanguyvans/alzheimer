# Experiment: CN vs AD Baseline Classification

**Goal**: Classify stable Cognitively Normal (CN) vs stable Alzheimer's Disease (AD) patients using single baseline MRI scans.

**Model**: ResNet3D-50 with MedicalNet pretrained weights

**Task**: Binary classification
- **CN (Cognitively Normal)**: label = 0
- **AD (Alzheimer's Disease)**: label = 1

---

## Quick Start

### 1. Setup Environment

Make sure you have a `.env` file in the project root with your wandb credentials:

```bash
# From project root (alzheimer/)
cp .env.example .env
# Edit .env and add your wandb credentials
```

### 2. Edit Configuration

Open [config.yaml](config.yaml) and verify your paths:

```yaml
data:
  dxsum_csv: "/Volumes/KINGSTON/dxsum.csv"
  skull_dir: "/Volumes/KINGSTON/ADNI-skull"
```

### 3. Run Pipeline

```bash
cd experiments/cn_vs_ad_baseline

# Step 1: Prepare dataset (creates train/val/test splits)
python 01_prepare_dataset.py --config config.yaml

# Step 2: Train model
python 02_train_model.py --config config.yaml
```

---

## Dataset Criteria

**Stable CN Patients**:
- All diagnosis records are CN (diagnosis code = 1)
- Never progressed to MCI or AD

**Stable AD Patients**:
- All diagnosis records are AD (diagnosis code = 3)
- Remained AD throughout all visits

**Baseline MRI**:
- Uses skull-stripped MRI from baseline visit
- Standard space (MNI-registered)
- Resolution: 96×96×96 voxels

---

## Model Architecture

- **Encoder**: ResNet3D-50 (MedicalNet pretrained on 23 medical datasets)
- **Parameters**: 46.2M
- **Input**: 96×96×96 MRI volume
- **Output**: 2 classes (CN=0, AD=1)
- **Loss**: Cross-entropy with class weighting
- **Optimizer**: Adam (lr=1e-4, weight_decay=1e-5)

---

## Expected Results

This is a **simpler task** than pMCI/sMCI prediction because:
- Clear diagnostic boundaries (healthy vs disease)
- Significant brain atrophy in AD vs CN
- Expected accuracy: **>85%** (literature baseline)

---

## Files

```
experiments/cn_vs_ad_baseline/
├── config.yaml                  # Configuration file
├── 01_prepare_dataset.py        # Dataset preparation
├── 02_train_model.py            # Training script
├── dataset.py                   # Dataset loader
├── models/                      # Model architecture (self-contained)
│   ├── resnet3d.py             # ResNet3D encoder
│   ├── classifier.py           # Binary classifier
│   └── unimodal.py             # Model wrapper
└── data/                        # Generated data (gitignored)
    ├── splits/                  # Train/val/test CSVs
    ├── checkpoints/             # Model checkpoints
    └── logs/                    # TensorBoard & wandb logs
```

---

## Monitoring

View training progress:
- **Wandb**: https://wandb.ai/tanguyvans/alzheimer-research
- **TensorBoard**: `tensorboard --logdir data/logs/tensorboard_logs`

---

## Next Steps After Training

1. Analyze confusion matrix (CN vs AD classification)
2. Visualize misclassified cases
3. Compare with pMCI/sMCI results
4. Use as baseline for more complex experiments

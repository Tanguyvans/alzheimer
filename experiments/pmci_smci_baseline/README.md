# Experiment: pMCI vs sMCI Baseline Classification

**Goal**: Predict MCI-to-AD conversion using single baseline MRI scans.

**Model**: DenseNet3D with binary classification (pMCI vs sMCI)

---

## Quick Start

### 1. Edit Configuration

Open [config.yaml](config.yaml) and set your paths:

```yaml
data:
  dxsum_csv: "/Volumes/KINGSTON/dxsum.csv"        # Path to diagnosis CSV
  skull_dir: "/Volumes/KINGSTON/ADNI-skull"      # Path to MRI scans
```

### 2. Run Pipeline

```bash
cd experiments/pmci_smci_baseline

# Step 1: Prepare dataset (creates train/val/test splits)
python 01_prepare_dataset.py

# Step 2: Train model
python 02_train_model.py

# Step 3: Monitor training
tensorboard --logdir data/logs
```

---

## Pipeline Overview

```
Step 1: Prepare Dataset
├─ Input:  dxsum.csv + ADNI-skull/ directory
├─ Output: data/splits/ (train.csv, val.csv, test.csv)
└─ Script: 01_prepare_dataset.py

Step 2: Train Model
├─ Input:  data/splits/*.csv
├─ Output: data/checkpoints/ + data/logs/
└─ Script: 02_train_model.py

Step 3: Evaluate (future)
├─ Input:  Best checkpoint from data/checkpoints/
├─ Output: Evaluation metrics and visualizations
└─ Script: 03_evaluate.py (to be implemented)
```

---

## Directory Structure

```
experiments/pmci_smci_baseline/
├── config.yaml                 # Configuration file (EDIT THIS!)
│
├── 01_prepare_dataset.py       # Step 1: Dataset preparation
├── 02_train_model.py           # Step 2: Training
├── dataset.py                  # PyTorch dataset class
│
├── data/                       # All experiment data (self-contained)
│   ├── splits/                 # Train/val/test CSV files
│   ├── checkpoints/            # Model checkpoints
│   └── logs/                   # TensorBoard logs
│
└── README.md                   # This file
```

**Everything is self-contained in this folder!**

---

## Configuration

All settings are in [config.yaml](config.yaml):

- **Data paths**: Where to find MRI scans
- **Model architecture**: Volume size, dropout, etc.
- **Training**: Batch size, epochs, learning rate
- **Hardware**: CPU/GPU, number of workers

---

## Expected Results

Based on ADNI literature:
- **Baseline accuracy**: 70-75%
- **Target accuracy**: 75-85%
- **Good balanced accuracy**: >0.70
- **Good AUC-ROC**: >0.75

---

## Outputs

### After Step 1 (Dataset Preparation)
```
data/splits/
├── train.csv              # Training set (318 samples)
├── val.csv                # Validation set (68 samples)
├── test.csv               # Test set (69 samples)
├── dataset_complete.csv   # All data combined
└── dataset_metadata.json  # Statistics
```

### After Step 2 (Training)
```
data/checkpoints/
├── densenet3d_mci-epoch=XX-val_balanced_acc=0.XXXX.ckpt  # Best models
└── last.ckpt                                              # Latest checkpoint

data/logs/
└── version_0/
    └── events.out.tfevents...  # TensorBoard logs
```

---

## For Cluster Use

1. Copy this entire folder to your cluster
2. Edit [config.yaml](config.yaml) with cluster paths
3. Adjust hardware settings (GPUs, workers)
4. Run the same commands

---

## Notes

- **Single config file** controls everything
- **All data stays in `data/`** subdirectory
- **No absolute paths** needed (except for input MRI scans)
- **Reproducible**: Random seeds are set in config

---

## Next Steps

After training:
1. Check TensorBoard for training curves
2. Evaluate best checkpoint on test set (Step 3, to be implemented)
3. Analyze confusion matrix and per-class metrics
4. Try different configurations (edit config.yaml)

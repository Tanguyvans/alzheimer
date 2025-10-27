# Alzheimer's Disease Research - Experiments Guide

This repository contains neuroimaging research for Alzheimer's disease and MCI prediction.

---

## 🚀 Quick Start for MCI Classification

Want to predict MCI-to-AD conversion? Start here:

```bash
cd experiments/pmci_smci_baseline

# 1. Edit config.yaml with your data paths
nano config.yaml

# 2. Run the pipeline
python 01_prepare_dataset.py    # Prepare train/val/test splits
python 02_train_model.py         # Train DenseNet3D model
```

See [experiments/pmci_smci_baseline/README.md](experiments/pmci_smci_baseline/README.md) for details.

---

## 📂 Repository Structure

```
alzheimer/
├── experiments/              ⭐ START HERE - Self-contained experiments
│   ├── README.md            # Experiment overview
│   └── pmci_smci_baseline/  # MCI-to-AD prediction experiment
│       ├── config.yaml      # Configuration (edit this!)
│       ├── 01_prepare_dataset.py
│       ├── 02_train_model.py
│       └── data/            # All outputs here
│
├── ADNI_unimodal_models/    # DenseNet3D implementation
├── preprocessing/           # Image preprocessing pipeline
├── data/                    # Clinical/tabular data
├── mni_template/           # Brain templates
├── utils/                  # Shared utilities
│
└── [Other directories]     # Legacy/other experiments
```

---

## 🎯 Available Experiments

### 1. **pMCI vs sMCI Classification** ([experiments/pmci_smci_baseline/](experiments/pmci_smci_baseline/))

**Goal**: Predict which MCI patients will convert to Alzheimer's Disease

**Data**: 455 patients (215 pMCI, 240 sMCI)
**Model**: DenseNet3D binary classifier
**Input**: Single baseline MRI scan per patient
**Status**: ✅ Ready to run

**Quick start:**
```bash
cd experiments/pmci_smci_baseline
python 01_prepare_dataset.py
python 02_train_model.py
```

---

## 🔧 How Experiments Work

Each experiment is **self-contained** in its own folder:

```
experiments/experiment_name/
├── config.yaml              # Single config file (edit paths here!)
├── 01_prepare_dataset.py    # Step 1: Data preparation
├── 02_train_model.py        # Step 2: Training
├── 03_evaluate.py           # Step 3: Evaluation
├── dataset.py               # PyTorch dataset
├── data/                    # All experiment data
│   ├── splits/              # Train/val/test CSVs
│   ├── checkpoints/         # Model weights
│   └── logs/                # TensorBoard logs
└── README.md                # Documentation
```

**Benefits:**
- ✅ Everything in one place
- ✅ Clear numbered pipeline
- ✅ Single config file
- ✅ Easy to transfer to cluster
- ✅ Reproducible

---

## 📊 Data Requirements

### For MCI Classification Experiment

You need:
1. **dxsum.csv** - Diagnosis summary with patient IDs and labels
2. **ADNI-skull/** - Directory with skull-stripped MRI scans (.nii.gz)

Structure:
```
ADNI-skull/
├── 002_S_0295/
│   ├── MP-RAGE_2006-04-18_..._skull_stripped.nii.gz
│   └── ...
├── 002_S_0413/
│   └── ...
└── ...
```

Edit paths in [experiments/pmci_smci_baseline/config.yaml](experiments/pmci_smci_baseline/config.yaml)

---

## 🖥️ Running on Cluster

1. **Copy experiment folder** to cluster:
   ```bash
   scp -r experiments/pmci_smci_baseline/ user@cluster:/path/
   ```

2. **Edit config.yaml** with cluster paths:
   ```yaml
   data:
     dxsum_csv: "/cluster/data/dxsum.csv"
     skull_dir: "/cluster/data/ADNI-skull"

   hardware:
     gpus: 1
     num_workers: 8
   ```

3. **Run pipeline**:
   ```bash
   python 01_prepare_dataset.py
   python 02_train_model.py
   ```

---

## 📈 Monitoring Training

View training progress with TensorBoard:

```bash
cd experiments/pmci_smci_baseline
tensorboard --logdir data/logs
```

Open: http://localhost:6006

---

## 🔬 Other Components

### Preprocessing Pipeline ([preprocessing/](preprocessing/))
- DICOM to NIfTI conversion
- N4 bias correction
- MNI registration
- Skull stripping

### Model Architectures ([ADNI_unimodal_models/](ADNI_unimodal_models/))
- DenseNet3D with CLIP
- Unimodal and multimodal models

### Analysis Tools
- `data_analysis/` - Dataset analysis scripts
- `data_cleaning/` - Data cleaning utilities
- `tabular/` - Tabular data processing

---

## 📚 Documentation

- **[experiments/README.md](experiments/README.md)** - Experiment structure guide
- **[experiments/pmci_smci_baseline/README.md](experiments/pmci_smci_baseline/README.md)** - MCI experiment details
- **[CLAUDE.md](CLAUDE.md)** - Project overview and commands
- **[README.md](README.md)** - Main project README

---

## 🆘 Getting Help

1. Check experiment README: `experiments/*/README.md`
2. Check configuration: `experiments/*/config.yaml`
3. View logs: `tensorboard --logdir experiments/*/data/logs`
4. Open an issue on GitHub

---

## 🎓 Citation

If you use this code, please cite:
- ADNI dataset: [adni.loni.usc.edu](http://adni.loni.usc.edu)
- DenseNet3D architecture (if applicable)

---

## 📝 Notes

- All experiments use Python 3.12
- Virtual environment in `/env/`
- GPU training recommended but CPU works
- Expected training time: 2-4 hours (CPU), 30-60 min (GPU)

---

**Ready to start?** → Go to [experiments/pmci_smci_baseline/](experiments/pmci_smci_baseline/)

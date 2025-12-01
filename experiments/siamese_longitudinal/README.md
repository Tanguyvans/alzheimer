# Siamese Network for Longitudinal MRI Analysis

This experiment uses a Siamese network architecture to compare paired MRI scans (baseline + follow-up) and predict conversion to Alzheimer's disease.

## Architecture

The Siamese network uses a **shared 3D CNN encoder** to process both baseline and follow-up MRI scans:

```
Baseline MRI ──┐
               ├──> Shared 3D CNN Encoder ──> Embeddings ──> Classifier ──> Prediction
Follow-up MRI ─┘
```

**Key features:**
- Shared weights ensure consistent feature extraction
- Learns temporal changes between scans
- Time delta (years between scans) incorporated as feature
- Binary classification: Converter vs Non-Converter

## Pipeline

```bash
# Step 1: Prepare paired MRI data (requires external drive with skull-stripped NIfTI)
python 01_prepare_pairs.py --config config.yaml

# Step 2: Train Siamese network
python 02_train.py --config config.yaml
```

## Configuration

All settings are in `config.yaml`:

```yaml
data:
  skull_dir: "/Volumes/KINGSTON/ADNI-skull"  # Path to skull-stripped MRI
  dxsum_csv: "..."  # Diagnosis data for trajectories

model:
  target_shape: [64, 64, 64]  # Volume resize dimensions
  embedding_dim: 256
  dropout: 0.3

training:
  batch_size: 4
  epochs: 50
  learning_rate: 0.0001
```

## Data Requirements

The `01_prepare_pairs.py` script creates pairs from:
- Skull-stripped NIfTI files (`.nii.gz`)
- DXSUM diagnosis data for patient trajectories

Each pair contains:
- `baseline_path`: Path to baseline MRI volume
- `followup_path`: Path to follow-up MRI volume
- `is_converter`: 1 if patient converted (MCI→AD), 0 otherwise
- `days_between`: Days between scans
- `trajectory`: Patient trajectory (CN_stable, MCI_stable, MCI_to_AD, AD_stable)

## Model Architecture

```
Input: (B, 1, D, H, W) - 3D MRI volume

Encoder (shared):
├── Conv3D(1→32) + BN + ReLU + MaxPool
├── Conv3D(32→64) + BN + ReLU + MaxPool
├── Conv3D(64→128) + BN + ReLU + MaxPool
├── Conv3D(128→256) + BN + ReLU + AdaptiveAvgPool
└── Linear(256→embedding_dim)

Classifier:
├── Input: [baseline_emb, followup_emb, diff, time_delta]
├── Linear(embedding_dim*3 + 1 → 128) + ReLU + Dropout
├── Linear(128 → 64) + ReLU + Dropout
└── Linear(64 → num_classes)
```

## Output Files

After training, results are saved in `results/`:
- `best_model.pt` - Best model checkpoint
- `training_curves_*.png` - Loss and accuracy plots
- `confusion_matrix_*.png` - Test set confusion matrix
- `results_*.json` - Metrics and configuration

## Advantages Over Single-Scan Models

1. **Captures temporal dynamics** - Directly models brain changes over time
2. **Shared encoder** - Efficient parameter sharing, reduced overfitting
3. **Time-aware** - Incorporates follow-up duration in predictions
4. **Interpretable** - Can visualize which changes are most predictive

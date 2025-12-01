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
# Step 1: Prepare paired MRI data
python 01_prepare_pairs.py

# Step 2: Train Siamese network
python 02_train.py

# With custom parameters:
python 02_train.py --epochs 100 --batch-size 4 --lr 1e-4 --target-shape 64 64 64
```

## Data Requirements

The `01_prepare_pairs.py` script creates pairs from:
- Preprocessed MRI volumes (`.npy` files)
- Clinical data with visit information

Each pair contains:
- `baseline_path`: Path to baseline MRI volume
- `followup_path`: Path to follow-up MRI volume
- `is_converter`: 1 if patient converted (MCI→AD), 0 otherwise
- `days_between`: Days between scans
- `ptid`: Patient identifier
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
├── Linear(embedding_dim*3 + 1 → 256) + ReLU + Dropout
├── Linear(256 → 128) + ReLU + Dropout
└── Linear(128 → num_classes)
```

## Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 50 | Maximum training epochs |
| `--batch-size` | 4 | Batch size (adjust based on GPU memory) |
| `--lr` | 1e-4 | Learning rate |
| `--weight-decay` | 1e-4 | L2 regularization |
| `--patience` | 15 | Early stopping patience |
| `--target-shape` | 64 64 64 | Volume resizing dimensions |
| `--embedding-dim` | 256 | Embedding vector size |
| `--weighted` | False | Use time-weighted attention model |

## Output Files

After training, results are saved in `results/`:
- `best_model.pt` - Best model checkpoint
- `training_curves.png` - Loss and accuracy plots
- `confusion_matrix.png` - Test set confusion matrix
- `results.json` - Metrics and configuration

## Advantages Over Single-Scan Models

1. **Captures temporal dynamics** - Directly models brain changes over time
2. **Shared encoder** - Efficient parameter sharing, reduced overfitting
3. **Time-aware** - Incorporates follow-up duration in predictions
4. **Interpretable** - Can visualize which changes are most predictive

## References

- Koch et al. (2015) - Siamese Neural Networks for One-shot Image Recognition
- Bhagwat et al. (2018) - Modeling and prediction of clinical symptom trajectories in Alzheimer's disease

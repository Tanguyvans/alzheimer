# CN vs AD Binary Classification Ensemble

Simplified binary classification (CN vs AD) to establish a strong baseline before tackling the harder 3-class problem.

## Rationale

The 3-class problem (CN/MCI/AD) achieved only 63% accuracy with MCI having 39% recall. By focusing on CN vs AD:

1. **Remove the hardest class**: MCI is a transitional stage with subtle biomarkers
2. **Establish baseline**: Binary classification should achieve 85-95% based on literature
3. **Validate improvements**: Test SEResNet-18 and augmentation strategies on cleaner problem
4. **Build confidence**: Prove our pipeline works before tackling MCI

## Expected Performance

| Method | Expected Accuracy |
|--------|------------------|
| Single ResNet-18 | 85-90% |
| SEResNet-18 Ensemble (5 models) | 90-95% |
| SEResNet-18 Ensemble + Augmentation | 93-97% |

## Quick Start

### Step 1: Create CN vs AD Splits

Filter 3-class data to keep only CN and AD samples:

```bash
cd experiments/cn_ad_ensemble

python3 01_create_cn_ad_splits.py \
  --train-csv ../cn_mci_ad_medicalnet/data/splits/train.csv \
  --val-csv ../cn_mci_ad_medicalnet/data/splits/val.csv \
  --test-csv ../cn_mci_ad_medicalnet/data/splits/test.csv \
  --output-dir data/splits
```

**Output:**
- `data/splits/train.csv` - Training set (CN=0, AD=1)
- `data/splits/val.csv` - Validation set
- `data/splits/test.csv` - Test set

**Expected Distribution:**
- Train: ~867 samples (CN: 590, AD: 277)
- Val: ~108 samples
- Test: ~109 samples

### Step 2: Create Balanced Subsets

Create balanced training subsets for ensemble diversity:

```bash
python3 02_create_balanced_subsets.py \
  --train-csv data/splits/train.csv \
  --output-dir balanced_subsets \
  --num-subsets 5 \
  --seed 42
```

**What this does:**
- Original training: CN=590, AD=277 (imbalanced, 68% CN)
- Creates 5 subsets: Each with CN=277, AD=277 (perfectly balanced, 50/50)
- Different CN samples in each subset for diversity (all AD samples used in each)

### Step 3: Train SEResNet-18 Ensemble

Train 5 SEResNet-18 models with Squeeze-and-Excitation blocks:

```bash
python3 train_ensemble.py \
  --subset-dir balanced_subsets \
  --val-csv data/splits/val.csv \
  --num-models 5 \
  --pretrained ../cn_mci_ad_medicalnet/pretrained/resnet_18_23dataset.pth \
  --checkpoints-dir checkpoints_seresnet \
  --batch-size 4 \
  --epochs 50 \
  --patience 10 \
  --use-se
```

**Training details:**
- Uses MedicalNet pretrained weights (trained on 23 medical imaging datasets)
- SE blocks add channel attention (+87K params, +0.3% overhead)
- Early stopping with patience=10 (stops if no improvement)
- Cosine annealing LR scheduler

### Step 4: Evaluate Ensemble

Get ensemble predictions on test set:

```bash
python3 predict_ensemble.py \
  --checkpoints-dir checkpoints_seresnet \
  --test-csv data/splits/test.csv \
  --num-models 5 \
  --batch-size 4
```

**Output:**
- Confusion matrix
- Per-class precision, recall, F1-score
- Overall accuracy
- `ensemble_predictions.csv` with predictions and probabilities

## Architecture

### SEResNet-18 with SE Blocks

**Squeeze-and-Excitation (SE) Blocks:**
- **Squeeze**: Global average pooling → channel descriptor
- **Excitation**: FC(C→C/16) → ReLU → FC(C/16→C) → Sigmoid
- **Scale**: Recalibrate feature maps by learned channel weights

**Benefits:**
- Emphasizes important channels (e.g., hippocampus features)
- Suppresses irrelevant channels
- Only 87K extra parameters
- Literature: 93.26% on ADNI 3-class, expect >95% on binary

### Binary vs 3-Class Comparison

| Metric | 3-Class (CN/MCI/AD) | Binary (CN/AD) |
|--------|-------------------|----------------|
| Difficulty | Hard (MCI is transitional) | Easier (clear endpoints) |
| Baseline (single model) | 61.9% | 85-90% (expected) |
| Ensemble (5 models) | 63.0% | 90-95% (expected) |
| MCI/CN Confusion | High (23/59 errors) | N/A |
| MCI/AD Confusion | High (13/59 errors) | N/A |

## Class Distribution

**Original 3-Class Dataset:**
- Train: CN=590 (44%), MCI=473 (35%), AD=277 (21%)
- Val: CN=74, MCI=59, AD=35
- Test: CN=74, MCI=59, AD=35

**Binary (CN vs AD):**
- Train: CN=590 (68%), AD=277 (32%) → Balance to 277/277 per subset
- Val: CN=74, AD=35 (68% CN)
- Test: CN=74, AD=35 (68% CN)

**Note**: Class imbalance (68% CN) is handled by:
1. Balanced training subsets (50/50 CN/AD)
2. Ensemble diversity (different CN samples per model)

## Next Steps After CN/AD

Once we achieve 90-95% on CN vs AD:

1. **Add data augmentation**: Rotation, flipping, noise → +5-10%
2. **Test whole-brain input**: Currently using hippocampus (192x192x192), try full brain (176x208x176)
3. **Try focal loss**: Better handling of hard examples
4. **Return to 3-class**: Apply successful strategies to CN/MCI/AD
5. **Add MCI staging**: Separate EMCI (Early MCI) vs LMCI (Late MCI)

## Training Configuration

**Hyperparameters:**
- Batch size: 4 (adjust for GPU memory)
- Learning rate: 1e-4 (AdamW optimizer)
- Weight decay: 1e-4 (L2 regularization)
- Epochs: 50 (with early stopping)
- Patience: 10 (stop if no val improvement)
- LR scheduler: Cosine annealing (smooth decay)

**Data:**
- Input shape: (1, 192, 192, 192) - single channel 3D MRI
- Normalization: Percentile-based (p1-p99) for robustness
- Preprocessing: N4 bias correction + skull stripping + MNI registration

## References

- **SEResNet Paper**: Gupta et al., "Deep CNN ResNet-18 based model with attention and transfer learning for Alzheimer's disease detection", Neural Computing and Applications, 2024
  - 93.26% accuracy on ADNI 3-class with SEResNet-18

- **MedicalNet**: Chen et al., "Med3D: Transfer Learning for 3D Medical Image Analysis"
  - Pretrained 3D ResNet on 23 medical datasets

- **IMBALMED**: Francesconi et al., "Class Balancing Diversity Multimodal Ensemble", 2024
  - 73.5% on 3-class ADNI with balanced subsets ensemble

## File Structure

```
cn_ad_ensemble/
├── README.md                          # This file
├── 01_create_cn_ad_splits.py         # Filter 3-class to binary
├── 02_create_balanced_subsets.py     # Create balanced training subsets
├── train_ensemble.py                  # Train ensemble models
├── predict_ensemble.py                # Evaluate ensemble
├── dataset.py                         # PyTorch dataset (symlink)
├── model_resnet3d.py                  # Vanilla ResNet-18 (symlink)
├── model_seresnet3d.py                # SEResNet-18 with SE blocks (symlink)
├── data/
│   └── splits/                        # Train/val/test CSVs
├── balanced_subsets/                  # Balanced training subsets
│   ├── train_subset_1.csv
│   ├── train_subset_2.csv
│   └── ...
└── checkpoints_seresnet/              # Model checkpoints
    ├── model_1_best.pth
    ├── model_2_best.pth
    └── ...
```

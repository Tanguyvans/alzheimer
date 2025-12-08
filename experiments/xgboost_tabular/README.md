# XGBoost Tabular Classification

XGBoost models for Alzheimer's disease classification using clinical tabular data.

## Available Models

| Script | Classes | Description |
|--------|---------|-------------|
| `train_cn_ad_mci_ad.py` | 2 | CN vs (MCI→AD + AD) |
| `train_cn_mci_ad.py` | 3 | CN vs MCI vs AD |
| `train_cn_mcis_mcic_ad.py` | 4 | CN vs MCI_stable vs MCI_converter vs AD |

## Commands

### 1. Binary Classification: CN vs (MCI→AD + AD)

```bash
python train_cn_ad_mci_ad.py \
    --input-csv /path/to/data/ALL_4class_clinical.csv \
    --output-dir results/cn_ad_mci_ad
```

### 2. 3-Class Classification (CN vs MCI vs AD)

```bash
python train_cn_mci_ad.py \
    --input-csv /path/to/data/ALL_4class_clinical.csv \
    --output-dir results/cn_mci_ad
```

### 3. 4-Class Classification (CN vs MCI_stable vs MCI_converter vs AD)

```bash
python train_cn_mcis_mcic_ad.py \
    --input-csv /path/to/data/ALL_4class_clinical.csv \
    --output-dir results/cn_mcis_mcic_ad
```

### 4. Test on MCI Stable Patients

After training binary model, test on MCI stable patients:

```bash
python test_mci_stable.py \
    --model-dir results/cn_ad_mci_ad \
    --mci-csv /path/to/mci_stable_patients.csv \
    --output-dir results/mci_stable_analysis
```

### 5. Plot Confidence Comparison

```bash
python plot_confidence_comparison.py
```

### 6. Check Patient Similarity (Data Leakage Check)

```bash
python check_patient_similarity.py \
    --train results/cn_ad_mci_ad/train_data.csv \
    --test results/cn_ad_mci_ad/test_data.csv
```

## Input Data

Expected CSV with columns:

- `Subject`: Patient ID (for patient-level splitting)
- `Group`: Original diagnosis at scan time (CN, MCI, AD)
- `DX`: Final diagnosis after follow-up (CN, MCI, AD)
- Clinical features: MMSCORE, CDGLOBAL, PTGENDER, PTEDUCAT, AGE, etc.

Data location: `/path/to/data/ALL_4class_clinical.csv`

## Output

Each experiment saves to its output directory:

```text
results/<experiment>/
├── xgboost_model.json      # Trained model
├── scaler.pkl              # Feature scaler
├── metrics.json            # Test metrics
├── predictions.csv         # Test predictions with Subject IDs
├── train_data.csv          # Training split (for verification)
├── test_data.csv           # Test split (for verification)
├── feature_importance.csv  # Feature importance scores
├── confusion_matrix.png    # Confusion matrix plot
└── roc_curve.png           # ROC curve plot
```

## Results Summary

| Model | Accuracy | Balanced Acc | AUC-ROC |
|-------|----------|--------------|---------|
| CN vs AD | 92.2% | 92.2% | 0.969 |
| CN vs MCI vs AD | - | - | - |
| 4-class | - | - | - |

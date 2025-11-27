# XGBoost Tabular Classification

XGBoost models for Alzheimer's disease classification using clinical tabular data.

## Experiments

### 1. Binary Classification
**CN vs (AD + MCI→AD)**
- Cognitively Normal vs Alzheimer's Disease (including MCI converters)

### 2. 4-Class Classification
**CN | MCI stable | MCI→AD | AD**
- Class 0: CN (Cognitively Normal)
- Class 1: MCI stable (MCI patients who did NOT convert to AD)
- Class 2: MCI→AD (MCI patients who converted to AD)
- Class 3: AD (Alzheimer's Disease)

## Quick Start

```bash
# Run both experiments
./run_experiments.sh /path/to/ALL_classes_clinical.csv

# Or run individually
python train_cn_ad_mci_ad.py \
    --input-csv /path/to/ALL_classes_clinical.csv \
    --output-dir results/cn_ad_mci_ad

python train_cn_mcis_mcic_ad.py \
    --input-csv /path/to/ALL_classes_clinical.csv \
    --output-dir results/cn_mcis_mcic_ad
```

## Input Data

Expected CSV columns:
- `Subject`: Patient ID (for patient-level splitting)
- `Group`: Original diagnosis at scan time (CN, MCI, AD)
- `DX`: Final diagnosis after follow-up (CN, MCI, AD)
- Clinical features: MMSCORE, CDGLOBAL, PTGENDER, PTEDUCAT, etc.

## Output

Each experiment saves to its output directory:
- `xgboost_model.json` - Trained model
- `scaler.pkl` - Feature scaler
- `metrics.json` - Test metrics (accuracy, balanced accuracy, AUC-ROC)
- `predictions.csv` - Test set predictions with probabilities
- `feature_importance.csv` - Feature importance scores
- `confusion_matrix.png` - Visualization

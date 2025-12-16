# XGBoost Tabular Classification

XGBoost for Alzheimer's classification using clinical tabular data across multiple datasets.

## Datasets

| Dataset | Samples (CN vs AD) | Subjects | Features |
|---------|-------------------|----------|----------|
| ADNI | 972 (706 CN, 266 AD) | 655 | 24 |
| OASIS | 6,539 (5,792 CN, 747 AD) | 1,346 | 17 |
| NACC | 40,163 (22,760 CN, 17,403 AD) | 40,163 | 19 |
| NACC (MRI) | 4,743 (3,352 CN, 1,391 AD) | 4,743 | 19 |
| **ALL** | **47,674** | **42,164** | 16 (common) |

*Note: NACC datasets use first visit only per subject to avoid data leakage.*

## Results (CN vs AD)

| Dataset | Accuracy | Balanced Acc | AUC-ROC | Config |
|---------|----------|--------------|---------|--------|
| ADNI | 89.80% | 83.32% | 0.9538 | cn_ad_adni.yaml |
| OASIS | 84.89% | 75.12% | 0.8446 | cn_ad_oasis.yaml |
| **NACC** | **89.98%** | **89.70%** | **0.9634** | cn_ad_nacc.yaml |
| NACC (MRI) | 92.13% | 90.38% | 0.9650 | cn_ad_nacc_mri.yaml |
| ALL (combined) | 88.56% | 87.99% | 0.9474 | cn_ad_all.yaml |

## Usage

```bash
# Prepare NACC data (run once)
python build_nacc_dataset.py --output data/nacc/nacc_tabular.csv

# Train on single dataset
python train.py --config configs/cn_ad_adni.yaml
python train.py --config configs/cn_ad_oasis.yaml
python train.py --config configs/cn_ad_nacc.yaml

# Train on all datasets combined
python train.py --config configs/cn_ad_all.yaml
```

## Available Configs

| Config | Task | Dataset | Notes |
|--------|------|---------|-------|
| `cn_ad_adni.yaml` | CN vs AD | ADNI | 24 features |
| `cn_ad_oasis.yaml` | CN vs AD | OASIS | 17 features |
| `cn_ad_nacc.yaml` | CN vs AD | NACC | 19 features |
| `cn_ad_nacc_mri.yaml` | CN vs AD | NACC (MRI only) | 19 features |
| `cn_ad_all.yaml` | CN vs AD | ADNI+OASIS+NACC | 16 common features |
| `cn_ad_combined.yaml` | CN vs AD | ADNI+OASIS | Legacy |
| `cn_mci_ad_adni.yaml` | CN vs MCI vs AD | ADNI | 3-class |
| `cn_mci_ad_oasis.yaml` | CN vs MCI vs AD | OASIS | 3-class |
| `cn_mcis_mcic_ad_adni.yaml` | CN vs MCI-S vs MCI-C vs AD | ADNI | 4-class |

## Features (19 common across ADNI/NACC)

**Demographics:**

- AGE, PTGENDER, PTEDUCAT, PTMARRY

**Vitals:**

- VSWEIGHT, BMI

**Medical History:**

- MH14ALCH (alcohol), MH16SMOK (smoking), MH4CARD (cardiovascular)
- MHPSYCH (psychiatric), MH2NEURL (neurological)

**Neuropsych Tests:**

- TRAASCOR (Trail Making A), TRABSCOR (Trail Making B), TRABERRCOM (errors)
- CATANIMSC (category fluency), CLOCKSCOR (clock drawing)
- BNTTOTAL (Boston Naming), DSPANFOR/DSPANBAC (digit span)

## Output

Results saved to `results/<config_name>/`:

- `xgboost_model.json` - Trained model
- `scaler.pkl` - Feature scaler
- `metrics.json` - Test metrics
- `predictions.csv` - Per-sample predictions
- `feature_importance.csv` - Feature rankings
- `confusion_matrix.png` - Confusion matrix
- `roc_curve.png` - ROC curves
- `feature_importance.png` - Top features plot

## Data Preparation

### NACC

```bash
# Build NACC tabular dataset from UDS data
python build_nacc_dataset.py \
    --uds-csv data/nacc/investigator_ftldlbd_nacc71.csv \
    --mri-csv data/nacc/investigator_mri_nacc71.csv \
    --output data/nacc/nacc_tabular.csv \
    --task cn_ad

# For MRI-only subjects
python build_nacc_dataset.py --mri-only --output data/nacc/nacc_tabular_mri.csv
```

### Feature Mapping (NACC → ADNI names)

| NACC | ADNI | Description |
|------|------|-------------|
| NACCAGE | AGE | Age at visit |
| SEX | PTGENDER | Sex |
| EDUC | PTEDUCAT | Years of education |
| TRAILA | TRAASCOR | Trail Making A |
| TRAILB | TRABSCOR | Trail Making B |
| ANIMALS | CATANIMSC | Category fluency |
| BOSTON | BNTTOTAL | Boston Naming Test |
| DIGIF/DIGIB | DSPANFOR/DSPANBAC | Digit span |

# XGBoost Tabular Classification

XGBoost for Alzheimer's classification using clinical tabular data.

## Usage

```bash
python train.py --config configs/<config>.yaml
```

## Available Configs

| Config | Task | Dataset |
|--------|------|---------|
| `cn_ad_adni.yaml` | CN vs AD | ADNI |
| `cn_ad_oasis.yaml` | CN vs AD | OASIS |
| `cn_mci_ad_adni.yaml` | CN vs MCI vs AD | ADNI |
| `cn_mci_ad_oasis.yaml` | CN vs MCI vs AD | OASIS |
| `cn_mcis_mcic_ad_adni.yaml` | CN vs MCI-S vs MCI-C vs AD | ADNI |

## Output

Results saved to `results/<task>_<dataset>/`:

- `xgboost_model.json` - Model
- `metrics.json` - Test metrics
- `predictions.csv` - Predictions
- `confusion_matrix.png` - Plots
- `feature_importance.png`

## Features

Demographics, cognitive tests, medical history (see `train.py` for full list).

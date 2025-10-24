# XGBoost Quickstart

Get started with tabular analysis in 5 minutes.

## Prerequisites

```bash
# Activate virtual environment
source env/bin/activate

# Install dependencies (if needed)
pip install xgboost scikit-learn pandas numpy matplotlib seaborn
```

## 1. Train a Model (2 minutes)

```bash
cd tabular/xgboost
python3 train.py
```

**What it does:**
- Trains XGBoost on 1,179 subjects (CN vs AD)
- Saves model to `outputs/tabular/models/`
- Generates performance metrics and visualizations

**Expected output:**
```
Test Accuracy: 99.6%
F1 Score: 99.4%
AD Detection: 87/87 cases (100%)
```

## 2. Predict MCI Direction (1 minute)

```bash
python3 run.py
```

**What it does:**
- Loads trained model
- Predicts on 845 MCI patients
- Classifies as CN-like (stable) or AD-like (at risk)

**Expected output:**
```
MCI Classification:
  CN-like (stable):  509 (60.2%)
  AD-like (at risk): 336 (39.8%)
```

## 3. View Results

All outputs saved in `outputs/tabular/`:

```
outputs/tabular/
├── models/
│   ├── xgboost_model.pkl          # Trained model
│   └── xgboost_model.json         # XGBoost format
├── predictions/
│   ├── mci_predictions.csv        # MCI patient predictions
│   └── predictions_all_groups.csv # All predictions
└── visualizations/
    ├── confusion_matrix_xgboost.png
    ├── feature_importance_plot.png
    └── mci_predictions_visualization.png
```

## Key Files

**Predictions CSV columns:**
- `PTID`: Patient ID
- `Group`: True group (CN/MCI/AD)
- `Prediction`: Model prediction (CN/AD)
- `Probability_CN`: Probability of being CN
- `Probability_AD`: Probability of being AD
- `Confidence`: Prediction confidence

## Understanding Results

### For MCI Patients

**CN-like (stable):**
- Probability_AD < 0.5
- Cognitive patterns similar to healthy controls
- Lower risk of AD progression

**AD-like (at risk):**
- Probability_AD ≥ 0.5
- Cognitive patterns similar to AD patients
- Higher risk of AD progression

### Risk Levels

| Probability_AD | Risk Level | Action |
|----------------|------------|---------|
| 0.0 - 0.2 | Very low | Annual monitoring |
| 0.2 - 0.4 | Low | Regular monitoring |
| 0.4 - 0.6 | Moderate | Close monitoring |
| 0.6 - 0.8 | High | Intervention recommended |
| 0.8 - 1.0 | Very high | Immediate intervention |

## Next Steps

- **[Training Guide](training-guide.md)** - Customize training
- **[Features](features.md)** - Understand clinical features
- **[MCI Prediction](mci-prediction.md)** - Deep dive into MCI analysis

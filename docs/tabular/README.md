# Tabular Analysis Documentation

Machine learning classification using clinical and cognitive assessment data.

## Contents

- **[Quickstart](quickstart.md)** - Get started in 5 minutes
- **[Training Guide](training-guide.md)** - Train your own models
- **[MCI Prediction](mci-prediction.md)** - Predict MCI patient outcomes
- **[Features](features.md)** - Clinical features explained
- **[Metrics](metrics.md)** - Performance metrics guide

## Overview

The tabular analysis uses **XGBoost** to:

1. **Classify CN vs AD** - Binary classification with 99.6% accuracy
2. **Predict MCI Direction** - Identify patients at risk of AD progression

## Quick Example

```bash
# Train model
cd tabular/xgboost
python3 train.py

# Predict MCI direction
python3 run.py
```

## Results Summary

**CN vs AD Classification:**
- Accuracy: 99.6%
- Recall: 100% (perfect AD detection)
- Top feature: TRABSCOR (Trail Making Test B)

**MCI Predictions:**
- 60% classified as CN-like (stable)
- 40% classified as AD-like (at risk)

## File Structure

```
tabular/
└── xgboost/
    ├── train.py    # Train CN vs AD classifier
    └── run.py      # Predict MCI direction
```

## Next Steps

1. Read the [Quickstart](quickstart.md) to get started
2. Review [Features](features.md) to understand the data
3. Check [Metrics](metrics.md) for performance details

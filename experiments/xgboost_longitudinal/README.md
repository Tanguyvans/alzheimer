# XGBoost with Longitudinal Features

This experiment uses cognitive change rates over time to improve Alzheimer's classification.

## Pipeline

```bash
# Step 1: Compute longitudinal features (cognitive change rates)
python 01_compute_features.py

# Step 2: Train XGBoost (baseline vs baseline+longitudinal)
python 02_train.py

# Step 3: Plot converter detection results
python 03_plot_converter_detection.py
```

## Features

### Baseline Features (cross-sectional)
- Demographics: AGE, PTGENDER, PTEDUCAT, etc.
- Medical history: MH14ALCH, MH4CARD, etc.
- Cognitive scores: TRAASCOR, TRABSCOR, CATANIMSC, CLOCKSCOR, MMSCORE, etc.

### Longitudinal Features (change over time)
- `years_between`: Follow-up duration
- `n_visits`: Number of visits
- `trail_a_change_rate`: Trail Making A change per year
- `trail_b_change_rate`: Trail Making B change per year
- `category_change_rate`: Category fluency change per year
- `clock_change_rate`: Clock drawing change per year
- `bnt_change_rate`: Boston Naming Test change per year
- Percentage changes for each score

## Results

| Model | Accuracy | Balanced Accuracy |
|-------|----------|-------------------|
| Baseline Only | 67.3% | 62.1% |
| Baseline + Longitudinal | 78.5% | 72.2% |
| **Improvement** | +11.2% | **+10.1%** |

### Converter Detection

The model excels at identifying MCI patients who will convert to AD:

| Trajectory | Accuracy |
|------------|----------|
| CN stable | 77.8% |
| MCI stable | 44.2% |
| **MCI → AD** | **84.2%** |
| AD stable | 87.8% |

# Tabular Data Analysis

Exploratory data analysis for Alzheimer's tabular datasets.

## Usage

```bash
python analyze.py --config configs/adni.yaml
python analyze.py --config configs/oasis.yaml
python analyze.py --config configs/combined.yaml
```

## Output

Reports saved to `reports/<dataset>/`:

- `dataset_summary.md` - Overview statistics
- `class_distribution.png` - Sample/subject counts by class
- `feature_distributions.png` - Histograms by class
- `feature_boxplots.png` - Boxplots by class
- `missing_values.png` - Missing data visualization
- `correlation_matrix.png` - Feature correlations
- `feature_stats.csv` - Per-feature statistics with p-values
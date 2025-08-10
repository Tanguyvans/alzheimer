# Tabular Models for Alzheimer's Classification

This directory contains machine learning models for classifying Alzheimer's Disease using clinical and demographic tabular data from the ADNI dataset.

## Overview

The tabular approach uses clinical features and cognitive test scores to perform binary classification:
- **AD**: Alzheimer's Disease (433 subjects)
- **Non-AD**: Combined CN (Cognitively Normal) + MCI (Mild Cognitive Impairment) (2062 subjects)

## Dataset Features

**Clinical Features Used**:
- `AGE`: Patient age at baseline
- `PTGENDER_Male`: Gender (binary: male=1, female=0)
- `PTEDUCAT`: Years of education
- `MMSE`: Mini-Mental State Examination score (cognitive screening)
- `FAQ`: Functional Activities Questionnaire score
- `APOE4`: APOE4 genetic risk factor (0, 1, or 2 alleles)

**Data Statistics**:
- Total subjects: 2,495
- Train/test split: 70/30
- Class imbalance ratio: 4.86:1 (Non-AD:AD)

## Model Architecture

**XGBoost Gradient Boosting**:
- 100 estimators
- Default hyperparameters with class balancing
- Cross-entropy loss function
- Binary classification output

### Class Balancing Strategy

Due to severe class imbalance (4.86:1), we tested multiple weighting approaches:

1. **Default**: No class balancing (`scale_pos_weight=1.0`)
2. **Balanced**: Automatic balancing (`scale_pos_weight=4.86`)
3. **2x Weight**: Double the balanced weight (`scale_pos_weight=9.72`)
4. **3x Weight**: Triple the balanced weight (`scale_pos_weight=14.58`)

## Performance Results

### Model Comparison

| Model | Accuracy | Precision | Recall | F1 Score | AD Detection Rate |
|-------|----------|-----------|--------|----------|-------------------|
| **XGBoost 2x Weight** | **69.16%** | **25.51%** | **37.04%** | **30.21%** | **50/135 (37.0%)** |
| XGBoost 3x Weight | 66.36% | 24.00% | 40.00% | 30.00% | 54/135 (40.0%) |
| XGBoost Balanced | 71.30% | 24.36% | 28.15% | 26.12% | 38/135 (28.1%) |
| XGBoost Default | 79.71% | 34.55% | 14.07% | 20.00% | 19/135 (14.1%) |

### Best Model: XGBoost 2x Weight

**Confusion Matrix**:
```
             Predicted
             NonAD   AD
True NonAD    468   146
True   AD      85    50
```

**Key Metrics**:
- **Sensitivity (AD Detection)**: 37.04% - Successfully identifies 37% of AD cases
- **Specificity (Non-AD Detection)**: 76.22% - Correctly identifies 76% of non-AD cases
- **Precision**: 25.51% - When predicting AD, 26% are correct
- **F1 Score**: 30.21% - Balanced performance measure

### Clinical Interpretation

**Strengths**:
- **Improved AD Detection**: 2x weighting increases AD detection from 14% to 37%
- **Balanced Performance**: Reasonable trade-off between sensitivity and specificity
- **Clinical Relevance**: Uses standard cognitive assessments (MMSE, FAQ)
- **Fast Inference**: Lightweight model suitable for clinical deployment

**Limitations**:
- **Class Imbalance Impact**: High false positive rate due to imbalanced dataset
- **Limited Features**: Only 6 clinical features available
- **Precision Trade-off**: Lower precision (26%) means many false AD predictions

## Scripts

### `train_weighted_xgboost.py`

Main training script with class balancing:

```bash
# Train weighted XGBoost models
python3 train_weighted_xgboost.py
```

**Key Features**:
- Multiple class weighting strategies
- Comprehensive evaluation metrics
- Confusion matrix analysis
- Results comparison table
- CSV output for further analysis

### `data_preprocessing_tabular.py`

Data preprocessing pipeline:

```bash
# Preprocess ADNI tabular data
python3 data_preprocessing_tabular.py
```

**Processing Steps**:
1. Load ADNI clinical CSV data
2. Remove diagnostic leakage features
3. Handle missing values
4. Convert 3-class to binary classification
5. Feature scaling and normalization

## Clinical Context

### MMSE Score Integration

The Mini-Mental State Examination (MMSE) is retained as a feature because:
- **Screening Tool**: MMSE is a standard cognitive screening instrument
- **Non-Diagnostic**: Scores indicate cognitive impairment but don't diagnose AD
- **Clinical Relevance**: Routinely collected in memory clinics
- **Predictive Value**: Strong predictor of cognitive decline

### Feature Importance

Most predictive features for AD classification:
1. **MMSE**: Cognitive screening score (most important)
2. **FAQ**: Functional abilities assessment  
3. **AGE**: Older age increases AD risk
4. **APOE4**: Genetic risk factor
5. **Education**: Higher education may be protective

## Model Output

- **Trained Models**: `weighted_xgboost_results.csv`
- **Performance Metrics**: Displayed in console output
- **Best Model**: XGBoost with 2x class weighting

## Usage Recommendations

**When to Use Tabular Models**:
- Limited computational resources
- Need for model interpretability  
- Clinical decision support systems
- Screening applications with available clinical data

**Clinical Application**:
- **Primary Screening**: First-line cognitive assessment
- **Risk Stratification**: Identify high-risk patients for further testing
- **Monitoring**: Track cognitive decline over time
- **Resource Allocation**: Prioritize patients for imaging studies

## Comparison with Other Approaches

| Approach | Accuracy | AD Detection | Advantages | Limitations |
|----------|----------|--------------|------------|-------------|
| **Tabular XGBoost** | 69.16% | 37.0% | Fast, interpretable | Limited features |
| **3D CNN** | 95.19% | 90.5% | High accuracy | Requires MRI, GPU |
| **2D Hippocampus** | TBD | TBD | Focused ROI | Requires segmentation |

## Technical Details

**Dependencies**:
- XGBoost 2.0+
- scikit-learn
- pandas, numpy
- matplotlib, seaborn

**Runtime**:
- Training: <5 minutes on CPU
- Inference: <1ms per sample
- Memory: <100MB

**Data Requirements**:
- Clinical CSV with demographic/cognitive features
- No imaging data required
- Minimal preprocessing needed
# Alzheimer's Disease Classification Metrics Guide

This document explains all metrics and features used in the XGBoost-based Alzheimer's Disease classification system.

## Overview

This project uses machine learning to classify patients into:
- **CN (Cognitively Normal)** - Class 0
- **AD (Alzheimer's Disease)** - Class 1

The classification is based on clinical assessment scores and demographic data from the ADNI dataset.

---

## 📊 Performance Metrics

### 1. **Confusion Matrix**
A 2x2 table showing prediction accuracy:

```
                Predicted
                CN    AD
Actual  CN     TN    FP
        AD     FN    TP
```

- **TN (True Negative)**: CN correctly identified as CN
- **FP (False Positive)**: CN incorrectly labeled as AD
- **FN (False Negative)**: AD incorrectly labeled as CN  
- **TP (True Positive)**: AD correctly identified as AD

### 2. **Core Metrics**

#### **Accuracy**
- **Formula**: (TP + TN) / (TP + TN + FP + FN)
- **Meaning**: Overall percentage of correct predictions
- **Range**: 0.0 to 1.0 (higher is better)
- **Clinical**: How often the model is right overall

#### **Precision (Positive Predictive Value)**
- **Formula**: TP / (TP + FP)
- **Meaning**: When model predicts AD, how often is it correct?
- **Range**: 0.0 to 1.0 (higher is better)
- **Clinical**: Reduces false alarms - important for patient anxiety

#### **Recall (Sensitivity)**
- **Formula**: TP / (TP + FN)
- **Meaning**: Of all actual AD cases, how many did we catch?
- **Range**: 0.0 to 1.0 (higher is better)
- **Clinical**: Critical for early intervention - missing AD is dangerous

#### **Specificity**
- **Formula**: TN / (TN + FP)
- **Meaning**: Of all actual CN cases, how many did we correctly identify?
- **Range**: 0.0 to 1.0 (higher is better)
- **Clinical**: Avoids unnecessary worry in healthy individuals

#### **F1 Score**
- **Formula**: 2 × (Precision × Recall) / (Precision + Recall)
- **Meaning**: Harmonic mean of precision and recall
- **Range**: 0.0 to 1.0 (higher is better)
- **Clinical**: Balanced measure when both false positives and false negatives matter

### 3. **Clinical Interpretation**

| Metric | Good Value | Clinical Priority |
|--------|------------|-------------------|
| **Recall** | > 0.85 | Don't miss AD cases |
| **Precision** | > 0.80 | Avoid false alarms |
| **F1 Score** | > 0.80 | Balanced performance |
| **Specificity** | > 0.75 | Don't worry healthy people |

---

## 🧠 Clinical Features Explained

### **Cognitive Assessment Features**

#### **1. TRABSCOR - Trail Making Test B** 
- **Type**: Executive Function
- **Task**: Connect alternating numbers and letters (1-A-2-B-3-C...)
- **Score**: Time in seconds (lower is better)
- **Impairment**: Slower completion indicates executive dysfunction
- **AD Relevance**: Executive dysfunction is early AD sign

#### **2. CATANIMSC - Category Fluency (Animals)**
- **Type**: Semantic Memory
- **Task**: Name as many animals as possible in 60 seconds
- **Score**: Total number of animals (higher is better)
- **Impairment**: Fewer animals suggests semantic network breakdown
- **AD Relevance**: Temporal lobe/hippocampal dysfunction

#### **3. BNTTOTAL - Boston Naming Test**
- **Type**: Language/Word Retrieval
- **Task**: Name pictures of objects (60 items)
- **Score**: Total correct (higher is better)
- **Impairment**: Lower scores indicate anomia (word-finding difficulty)
- **AD Relevance**: Language network deterioration

#### **4. CLOCKSCOR - Clock Drawing Test**
- **Type**: Visuospatial/Executive
- **Task**: Draw clock showing specific time
- **Score**: 0-5 scale (higher is better)
- **Impairment**: Poor clock indicates visuospatial/executive problems
- **AD Relevance**: Parietal lobe dysfunction

#### **5. DSPANBAC - Digit Span Backward**
- **Type**: Working Memory
- **Task**: Repeat number sequences in reverse order
- **Score**: Longest sequence recalled (higher is better)
- **Impairment**: Lower spans indicate working memory deficits
- **AD Relevance**: Prefrontal cortex dysfunction

#### **6. TRAASCOR - Trail Making Test A**
- **Type**: Processing Speed/Attention
- **Task**: Connect numbered circles in sequence (1-2-3-4...)
- **Score**: Time in seconds (lower is better)
- **Impairment**: Slower completion indicates cognitive slowing
- **AD Relevance**: General cognitive efficiency

#### **7. DSPANFOR - Digit Span Forward**
- **Type**: Attention/Immediate Memory
- **Task**: Repeat number sequences forward
- **Score**: Longest sequence recalled (higher is better)
- **Impairment**: Lower spans indicate attention deficits
- **AD Relevance**: Basic attention capacity

### **Demographic Features**

#### **8. PTGENDER - Gender**
- **Values**: 0 = Female, 1 = Male
- **AD Relevance**: Women have ~2x higher AD risk
- **Factors**: Hormonal, genetic, longevity differences

#### **9. PTDOBYY - Birth Year (Age)**
- **Type**: Continuous
- **AD Relevance**: Primary risk factor - older age = higher risk
- **Mechanism**: Brain aging and pathology accumulation

#### **10. Additional Features**
- **VSHEIGHT**: Height - may reflect developmental factors
- **VSWEIGHT**: Weight - weight loss common in AD
- **PTEDUCAT**: Education level - cognitive reserve factor
- **PTRACCAT**: Race category - population-specific risk factors

---

## ⚖️ Class Imbalance Handling

### **Scale Pos Weight**
- **Purpose**: Address unequal class distribution
- **Formula**: Number of CN cases / Number of AD cases
- **Effect**: Gives more importance to minority class (usually AD)
- **Values Tested**:
  - Default (1.0): No adjustment
  - Balanced: Automatic ratio calculation
  - 2x Weight: Double the calculated ratio
  - 3x Weight: Triple the calculated ratio

### **Why Important?**
- AD cases are typically fewer than CN cases
- Without adjustment, model may ignore AD class
- Proper weighting improves AD detection sensitivity

---

## 📈 Model Variants Compared

| Model | Description | Best For |
|-------|-------------|----------|
| **XGBoost Default** | No class weighting | Balanced datasets |
| **XGBoost Balanced** | Automatic weight calculation | Most imbalanced datasets |
| **XGBoost 2x Weight** | Double emphasis on AD | High recall needed |
| **XGBoost 3x Weight** | Triple emphasis on AD | Maximum AD detection |

---

## 🎯 Clinical Decision Making

### **High Recall Strategy** (Catch all AD)
- Accept more false positives
- Better safe than sorry approach
- Use higher class weights

### **High Precision Strategy** (Avoid false alarms)
- Accept missing some AD cases
- Minimize patient anxiety
- Use lower class weights

### **Balanced Strategy** (F1 optimization)
- Balance both concerns
- Most practical for screening
- Use moderate class weights

---

## 📋 Output Files Generated

1. **weighted_xgboost_results.csv** - Model comparison metrics
2. **feature_importance.csv** - Feature rankings and scores
3. **feature_importance_plot.png** - Visual feature importance
4. **confusion_matrix_best_model.png** - Visual confusion matrix
5. **xgboost_weight_comparison.png** - Model performance comparison

---

## 🔬 Clinical Validation

### **Cross-Validation**
- **Method**: 5-fold stratified cross-validation
- **Purpose**: Ensure model generalizes to new patients
- **Report**: Mean ± standard deviation of performance

### **Best Practices**
1. Always report both sensitivity and specificity
2. Consider clinical cost of false negatives vs false positives
3. Validate on independent test set
4. Report confidence intervals
5. Compare to clinical gold standards

---

## 📚 References

- **ADNI**: Alzheimer's Disease Neuroimaging Initiative
- **Trail Making Test**: Neuropsychological assessment battery
- **Boston Naming Test**: Language assessment standard
- **XGBoost**: Extreme Gradient Boosting algorithm
- **ROC Analysis**: Receiver Operating Characteristic curves

---

## ⚠️ Important Notes

- **Not for clinical diagnosis**: This is a research tool
- **Requires clinical interpretation**: Always combine with expert assessment  
- **Population specific**: Trained on ADNI cohort characteristics
- **Regular validation needed**: Model performance may drift over time
- **Ethical considerations**: Discuss implications with patients carefully
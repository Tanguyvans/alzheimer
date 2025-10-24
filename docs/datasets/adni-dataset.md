# ADNI Dataset Reference

## Alzheimer's Disease Neuroimaging Initiative (ADNI)

This document describes the ADNI datasets used in this project, including clinical tabular data and preprocessed MRI scans.

## 📊 Clinical Tabular Data

### Source: clinical_data_all_groups.csv

**Origin:** ADNI database - clinical assessments and cognitive test scores

**Dataset Size:**

- **Total**: 1,980 scan sessions from 1,472 unique patients
- **CN (Cognitively Normal)**: 859 scans (564 patients, 43.4%)
- **MCI (Mild Cognitive Impairment)**: 845 scans (681 patients, 42.7%)
- **AD (Alzheimer's Disease)**: 276 scans (234 patients, 13.9%)

**Longitudinal Data:**

- CN: 1.52 scans/patient (most follow-ups)
- MCI: 1.24 scans/patient
- AD: 1.18 scans/patient

**Data Quality:** 30-70% missing data in most features (see below)

---

## 🧠 MRI Imaging Data

### Source: ADNI-skull (Preprocessed)

**Location:** `/Volumes/KINGSTON/ADNI-skull/`

**Dataset Size:**

- **Unique patients**: 2,333 patient directories
- **Successfully processed**: 10,123 skull-stripped MRI files
- **Failed**: 1 file (99.99% success rate)

**Preprocessing Pipeline Applied:**

1. ✅ DICOM to NIfTI conversion
2. ✅ N4 bias field correction
3. ✅ MNI template registration (mni_icbm152)
4. ✅ Skull stripping (SynthStrip)

**Output Format:** NIfTI (.nii.gz) brain-only images in MNI space

**Directory Structure:**

```text
ADNI-skull/
├── patient_id_1/
│   └── scan_registered_skull_stripped.nii.gz
├── patient_id_2/
│   └── scan_registered_skull_stripped.nii.gz
└── ...
```

---

## Group Definitions

### CN - Cognitively Normal

**Criteria:**
- Normal memory function
- No functional impairment
- MMSE scores: 24-30
- CDR (Clinical Dementia Rating) = 0

**Use in ML:**
- Control group (Class 0)
- Baseline for comparison

---

### MCI - Mild Cognitive Impairment

**Criteria:**
- Memory complaints
- Objective memory impairment
- Preserved general cognition
- Minimal functional impairment
- Does not meet dementia criteria

**Subtypes (in some datasets):**
- **EMCI**: Early MCI (milder)
- **LMCI**: Late MCI (more advanced)

**Use in ML:**
- Progression prediction target
- Classified as CN-like or AD-like
- 60% CN-like, 40% AD-like in current model

---

### AD - Alzheimer's Disease

**Criteria:**
- Meets NINCDS-ADRDA criteria
- MMSE scores: 20-26
- CDR ≥ 0.5 (usually 1.0 or 2.0)
- Significant functional impairment
- Progressive cognitive decline

**Use in ML:**
- Positive class (Class 1)
- Training target for classification

## Progression Pathway

```
CN → MCI → AD
     ↓
   CN-like (60%) - Stable/revert
   AD-like (40%) - Progress
```

## Dataset Statistics

### Training Data (AD_CN_clinical_data.csv)

**1,179 subjects:**
- CN: 746 (63.3%)
- AD: 433 (36.7%)
- Class imbalance: 1.72:1

**Data quality:** Clean, minimal missing values

---

### Full Data (clinical_data_all_groups.csv)

**1,980 scans, 1,472 patients:**
- CN: 859 scans
- MCI: 845 scans
- AD: 276 scans

**Data quality:** 30-70% missing in most features

## XGBoost Implementation Results

### Binary Classification (CN vs AD)

**Performance (clean data):**
- Accuracy: 99.6%
- Precision: 98.9%
- Recall: 100%
- F1: 99.4%

**Performance (missing data):**
- CN: 87.3% correct
- AD: 71.4% correct
- Degraded due to imputation

### MCI Progression Prediction

**845 MCI patients:**
- CN-like: 509 (60.2%)
- AD-like: 336 (39.8%)

**Risk stratification:**
- Very low: 0.0-0.2 AD probability
- Low: 0.2-0.4
- Moderate: 0.4-0.6
- High: 0.6-0.8
- Very high: 0.8-1.0

## Clinical Significance

### Research Applications

- **CN**: Control group, baseline studies
- **MCI**: Intervention targets, progression modeling
- **AD**: Disease mechanism studies

### Clinical Practice

- **CN**: Annual screening
- **MCI CN-like**: Regular monitoring
- **MCI AD-like**: Close monitoring, early intervention
- **AD**: Treatment and care planning

## References

- ADNI: [adni.loni.usc.edu](http://adni.loni.usc.edu/)
- NINCDS-ADRDA: Diagnostic criteria for AD
- CDR: Clinical Dementia Rating scale
- MMSE: Mini-Mental State Examination

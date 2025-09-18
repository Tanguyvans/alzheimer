# ADNI Alzheimer's Research Group Classifications

## Overview
This document defines the research group classifications used in the Alzheimer's Disease Neuroimaging Initiative (ADNI) study as found in `study_entry.csv`.

## Classifications by Prevalence
Based on the dataset analysis:
- **CN**: 844 subjects (36.3%)
- **MCI**: 693 subjects (29.8%) 
- **AD**: 359 subjects (15.4%)
- **EMCI**: 246 subjects (10.6%)
- **LMCI**: 133 subjects (5.7%)
- **SMC**: 62 subjects (2.7%)

## Detailed Definitions

### CN - Cognitively Normal
- **Full Name**: Cognitively Normal Controls
- **Description**: Healthy elderly individuals with no cognitive impairment
- **Criteria**: 
  - Normal memory function
  - No functional impairment
  - Mini-Mental State Exam (MMSE) scores typically 24-30
  - Clinical Dementia Rating (CDR) = 0

### MCI - Mild Cognitive Impairment
- **Full Name**: Mild Cognitive Impairment (Classic/Original Definition)
- **Description**: General MCI category used in early ADNI phases (ADNI-1) before EMCI/LMCI subdivision
- **Criteria**:
  - Memory complaints
  - Objective memory impairment
  - Preserved general cognition
  - Minimal functional impairment
  - Does not meet dementia criteria
- **Note**: In later ADNI phases (ADNI-GO, ADNI-2), this was subdivided into EMCI and LMCI for more precise staging

### EMCI - Early Mild Cognitive Impairment
- **Full Name**: Early Mild Cognitive Impairment
- **Description**: Earlier/milder form of MCI with subtle cognitive changes
- **Criteria**:
  - Less severe than classic MCI
  - MMSE scores typically 24-30
  - CDR = 0.5
  - Minimal memory complaints
  - Very subtle objective memory impairment

### LMCI - Late Mild Cognitive Impairment
- **Full Name**: Late Mild Cognitive Impairment
- **Description**: More advanced MCI, closer to dementia conversion
- **Criteria**:
  - More pronounced cognitive impairment than EMCI
  - MMSE scores typically 20-26
  - CDR = 0.5
  - Greater functional impact
  - Higher risk of conversion to AD

### AD - Alzheimer's Disease
- **Full Name**: Alzheimer's Disease Dementia
- **Description**: Clinical diagnosis of Alzheimer's disease with dementia
- **Criteria**:
  - Meets NINCDS-ADRDA criteria for probable AD
  - MMSE scores typically 20-26
  - CDR ≥ 0.5 (usually 1.0 or 2.0)
  - Significant functional impairment
  - Progressive cognitive decline

### SMC - Subjective Memory Complaint
- **Full Name**: Subjective Memory Complaint
- **Description**: Individuals with memory concerns but normal cognitive testing
- **Criteria**:
  - Self-reported memory decline
  - Normal performance on cognitive tests
  - No objective cognitive impairment
  - May represent preclinical stage

## Progression Pathway

### Modern ADNI Classification (ADNI-GO, ADNI-2, ADNI-3)
```
CN → SMC → EMCI → LMCI → AD
     ↓      ↓      ↓
   Normal  Early  Late
   Aging   MCI    MCI
```

### Original ADNI Classification (ADNI-1)
```
CN → MCI → AD
     ↓
  General MCI
  (now split into EMCI/LMCI)
```

## Historical Context
- **ADNI-1** (2004-2009): Used broad MCI category
- **ADNI-GO** (2009-2011): Introduced EMCI/LMCI subdivision
- **ADNI-2/3** (2011+): Continued refined classification

## Clinical Significance

### For Research
- **CN/SMC**: Control groups and preclinical studies
- **EMCI**: Early intervention targets
- **MCI/LMCI**: Treatment efficacy studies
- **AD**: Established disease studies

### For Machine Learning
- **Binary Classification**: 
  - CN vs. (MCI+EMCI+LMCI+AD) - normal vs. impaired
  - CN vs. AD - normal vs. dementia
- **Multi-class Options**:
  - **3-class**: CN vs. MCI vs. AD (original/simplified)
  - **5-class**: CN vs. EMCI vs. LMCI vs. AD vs. SMC (full modern)
  - **4-class**: CN vs. (EMCI+LMCI) vs. AD vs. SMC (grouped MCI)
- **Progression**: 
  - MCI → AD conversion prediction
  - EMCI → LMCI → AD progression modeling

## Dataset Context
- **Total Subjects**: 2,338
- **Study Period**: 2006-2018+ (longitudinal)
- **Primary Use**: Alzheimer's disease progression modeling
- **Imaging Focus**: T1-weighted MPRAGE sequences for hippocampus morphometry

## Notes
- Classifications can change over time in longitudinal studies
- Some subjects may convert between categories during follow-up
- EMCI/LMCI subdivisions were introduced in later ADNI phases
- SMC is a newer category representing the earliest detectable stage
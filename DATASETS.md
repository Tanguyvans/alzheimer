# Datasets Documentation

This document provides a comprehensive analysis of the three datasets used in this Alzheimer's disease classification project: ADNI, OASIS, and NACC.

## Summary Table

| Dataset | Patients | Total Visits | Visits/Patient | T1 MRI Scans | MRI/Patient |
|---------|----------|--------------|----------------|--------------|-------------|
| ADNI | 2,311 | 12,227 | 5.3 | 17,827 | 7.7 |
| OASIS | 1,340 | 8,500 | 6.3 | 7,794 | 5.8 |
| NACC | 55,004 | 205,908 | 3.7 | 8,164 | - |

## Diagnosis Distribution (All Visits)

| Dataset | CN | MCI | AD/Dementia | Other |
|---------|-----|-----|-------------|-------|
| ADNI | 4,760 (39%) | 5,088 (42%) | 2,355 (19%) | - |
| OASIS | 5,792 (68%) | 220 (3%) | 1,378 (16%)* | 1,110 (13%) |
| NACC | 100,992 (49%) | 36,254 (18%) | 59,572 (29%) | 9,090 (4%)** |

*OASIS AD includes 747 AD + 631 Other Dementia
**NACC "Other" = Impaired-not-MCI category

---

## ADNI (Alzheimer's Disease Neuroimaging Initiative)

### Overview
ADNI is a longitudinal multicenter study designed to develop clinical, imaging, genetic, and biochemical biomarkers for the early detection and tracking of Alzheimer's disease.

### Data Statistics
- **Total patients**: 2,311
- **Total visits**: 12,227
- **Visits per patient**: 5.3 average
- **T1 MRI scans**: 17,827
- **MRI per patient**: 7.6 average
- **Phases**: ADNI1 (3,711), ADNI2 (4,501), ADNI3 (2,662), ADNI4 (939), ADNIGO (414)

### Diagnosis Distribution (All Visits)
| Diagnosis | Count | Percentage |
|-----------|-------|------------|
| CN (Cognitively Normal) | 4,760 | 39% |
| MCI (Mild Cognitive Impairment) | 5,088 | 42% |
| AD (Alzheimer's Disease) | 2,355 | 19% |

### First Visit Distribution
| Diagnosis | Count |
|-----------|-------|
| CN | 909 |
| MCI | 1,049 |
| AD | 353 |

### Patient Trajectory Categories

| Category | Count | % |
|----------|-------|---|
| Stable CN | 751 | 32.5% |
| Stable MCI | 603 | 26.1% |
| Stable AD | 347 | 15.0% |
| CN → MCI | 117 | 5.1% |
| MCI → AD | 350 | 15.1% |
| CN → MCI → AD | 37 | 1.6% |
| Other | 106 | 4.6% |

### ADNI Transition Matrix (Visit-to-Visit)

| From ↓ / To → | CN | MCI | AD |
|---------------|-----|-----|-----|
| **CN** | 94.7% | 5.1% | 0.3% |
| **MCI** | 3.2% | 87.6% | 9.2% |
| **AD** | 0.1% | 2.1% | 97.8% |

### CN Definition (ADNI)
- **Criteria**: Normal cognition based on clinical assessment
- **MMSE**: 24-30 (inclusive)
- **CDR**: 0
- **Memory**: No significant impairment on Wechsler Memory Scale Logical Memory II

### MCI Definition (ADNI) - Petersen Criteria
- **Subjective memory complaint**: Reported by patient or informant
- **Objective memory impairment**: Below education-adjusted cutoff on Wechsler Memory Scale
- **Preserved general cognition**: MMSE 24-30
- **Intact activities of daily living**: No significant functional impairment
- **CDR**: 0.5
- **Not demented**: Does not meet criteria for dementia

### AD Definition (ADNI)
- **NINCDS-ADRDA criteria**: Probable Alzheimer's Disease
- **MMSE**: 20-26
- **CDR**: 0.5 or 1.0

---

## OASIS (Open Access Series of Imaging Studies)

### Overview
OASIS-3 is a longitudinal neuroimaging, clinical, and cognitive dataset for normal aging and Alzheimer's disease, released by Washington University in St. Louis.

### Data Statistics
- **Total patients**: 1,340
- **Total visits**: 8,500
- **Visits per patient**: 6.3 average
- **Patients with T1 MRI**: 1,339
- **T1 MRI scans**: 3,913
- **MRI per patient**: 2.9 average

### Diagnosis Distribution (All Visits)
| Diagnosis | Count | Percentage |
|-----------|-------|------------|
| CN (Cognitively Normal) | 5,792 | 68.1% |
| Other (non-AD impairment) | 1,110 | 13.1% |
| AD (Alzheimer's Disease) | 747 | 8.8% |
| Other Dementia | 631 | 7.4% |
| MCI | 220 | 2.6% |

### First Visit Distribution
| Diagnosis | Count |
|-----------|-------|
| CN | 798 |
| Other | 279 |
| AD | 155 |
| Other Dementia | 71 |
| MCI | 37 |

### Patient Trajectory Categories (CN/MCI/AD only)

| Category | Count | % |
|----------|-------|---|
| Stable CN | 860 | 77.6% |
| Stable AD | 184 | 16.6% |
| CN → MCI | 18 | 1.6% |
| Stable MCI | 17 | 1.5% |
| MCI → AD | 9 | 0.8% |
| Other | 20 | 1.8% |

### OASIS Transition Matrix (Visit-to-Visit)

| From ↓ / To → | CN | MCI | AD |
|---------------|-----|-----|-----|
| **CN** | 97.7% | 1.7% | 0.6% |
| **MCI** | 11.9% | 67.9% | 20.2% |
| **AD** | 0.7% | 1.3% | 98.0% |

### CN Definition (OASIS)
- **CDR**: 0 (no dementia)
- **NORMCOG**: 1 (normal cognition flag)
- Based on UDS (Uniform Data Set) clinical assessment

### MCI Definition (OASIS) - NIA-AA Criteria
OASIS uses the NIA-AA framework with subtypes:
- **MCIAMEM**: Amnestic MCI (memory impairment only)
- **MCIAPLUS**: Amnestic MCI Plus (memory + other domains)
- **MCINON1**: Non-amnestic MCI single domain
- **MCINON2**: Non-amnestic MCI multiple domains
- **CDR**: 0.5 (questionable dementia)

### AD Definition (OASIS)
- **PROBAD**: Probable AD (NIA-AA criteria)
- **POSSAD**: Possible AD
- **CDR**: 0.5 or higher
- **DEMENTED**: 1 (dementia flag)

---

## NACC (National Alzheimer's Coordinating Center)

### Overview
NACC maintains the largest Alzheimer's disease database in the United States, aggregating data from ~40 NIH-funded Alzheimer's Disease Research Centers (ADRCs).

### Data Statistics
- **Total patients**: 55,004
- **Total visits**: 205,908
- **Visits per patient**: 3.7 average
- **T1 MRI scans**: 8,164
- **MRI subset (CN vs AD)**: 4,743 samples (CN: 3,352, AD: 1,391)

### NACCUDSD Diagnosis Codes (Original)
| Code | Meaning | All Visits | First Visit |
|------|---------|------------|-------------|
| 1 | Cognitively Normal | 100,992 | 22,755 |
| 2 | Impaired-not-MCI | 9,090 | 2,482 |
| 3 | MCI | 36,254 | 12,353 |
| 4 | Dementia | 59,572 | 17,414 |

### MRI Subset (CN vs AD only)
| Diagnosis | Count |
|-----------|-------|
| CN | 3,352 |
| AD | 1,391 |
| **Total** | **4,743** |

### CN Definition (NACC)
- **NACCUDSD**: 1
- **Criteria**: Normal cognition per UDS clinician judgment
- **CDR Global**: 0
- No cognitive complaints beyond normal aging
- Intact daily functioning

### MCI Definition (NACC) - NIA-AA Criteria
- **NACCUDSD**: 3
- **Criteria**: Cognitive concern + objective impairment + preserved independence
- **CDR Global**: 0.5
- Does not meet dementia criteria
- Subtypes tracked via MCISUB variable

### Dementia/AD Definition (NACC)
- **NACCUDSD**: 4 (Dementia)
- **NACCETPR**: Etiology of dementia
  - AD Primary: ~76% of dementia cases
  - Lewy Body: ~6%
  - Vascular: ~5%
  - Other/Mixed: ~13%

### "Impaired-not-MCI" Category (NACC-specific)
- **NACCUDSD**: 2
- Cognitive impairment that doesn't meet formal MCI criteria
- May have:
  - Isolated non-memory impairment
  - Subjective complaints without objective evidence
  - Impairment explained by other factors (depression, medications)

### Patient Trajectory Categories (CN/MCI/Dementia only)

| Category | Count | % |
|----------|-------|---|
| Stable CN | 19,229 | 35.6% |
| Stable Dementia | 17,071 | 31.6% |
| Stable MCI | 7,690 | 14.2% |
| MCI → Dementia | 3,566 | 6.6% |
| CN → MCI | 2,811 | 5.2% |
| CN → MCI → Dementia | 1,094 | 2.0% |
| Other | 1,893 | 3.5% |
| CN → Dementia | 498 | 0.9% |

### NACC Transition Matrix (Visit-to-Visit)

| From ↓ / To → | CN | MCI | Dementia |
|---------------|-----|-----|----------|
| **CN** | 93.2% | 5.9% | 0.9% |
| **MCI** | 11.1% | 70.2% | 18.8% |
| **Dementia** | 0.5% | 1.9% | 97.7% |

---

## Diagnostic Criteria Comparison

### MCI Criteria Across Datasets

| Aspect | ADNI (Petersen) | OASIS (NIA-AA) | NACC (NIA-AA) |
|--------|-----------------|----------------|---------------|
| Memory complaint | Required (subjective) | Not required | Not required |
| Objective impairment | Memory-focused | Any domain | Any domain |
| CDR requirement | 0.5 | 0.5 | 0.5 |
| Subtypes | Single definition | 4 subtypes | Multiple subtypes |
| Year introduced | 1999 | 2011 | 2011 |

### Key Differences

1. **ADNI MCI** is more restrictive:
   - Requires subjective memory complaint
   - Focuses on amnestic presentation
   - Based on older Petersen criteria (1999)

2. **OASIS/NACC MCI** use NIA-AA criteria:
   - Broader definition including non-amnestic MCI
   - Recognizes multiple subtypes
   - Does not require subjective complaint
   - More inclusive of early cognitive decline

3. **Practical Impact**:
   - Same patient might be classified as MCI in OASIS/NACC but not in ADNI
   - NACC has additional "Impaired-not-MCI" category for edge cases
   - Cross-dataset comparisons should account for these differences

---

## Data Processing Notes

### First Visit Only (Avoiding Data Leakage)
For machine learning experiments, we use **first visit only** per subject to prevent:
- Training/test leakage from longitudinal data
- Temporal bias in predictions
- Overrepresentation of frequently-visited subjects

### Class Mapping for Binary Classification
| Dataset | CN Mapping | AD Mapping |
|---------|------------|------------|
| ADNI | DX = "CN" | DX = "AD" |
| OASIS | CDR = 0, NORMCOG = 1 | PROBAD = 1 or POSSAD = 1 |
| NACC | NACCUDSD = 1 | NACCUDSD = 4 + AD etiology |

### MRI Availability
- **ADNI**: Well-curated MRI-clinical pairs
- **OASIS**: Multiple MRI sessions per subject
- **NACC**: MRI subset is smaller than full cohort (~2,225 subjects with converted scans)

---

## References

1. **ADNI**: [adni.loni.usc.edu](http://adni.loni.usc.edu)
   - Petersen RC, et al. (1999). Mild cognitive impairment. Arch Neurol.

2. **OASIS**: [oasis-brains.org](https://www.oasis-brains.org)
   - LaMontagne PJ, et al. (2019). OASIS-3: Longitudinal Neuroimaging, Clinical, and Cognitive Dataset.

3. **NACC**: [naccdata.org](https://naccdata.org)
   - Beekly DL, et al. (2007). The National Alzheimer's Coordinating Center (NACC) Database.

4. **NIA-AA Criteria**: Albert MS, et al. (2011). The diagnosis of mild cognitive impairment due to Alzheimer's disease. Alzheimers Dement.

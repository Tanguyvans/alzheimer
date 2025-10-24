# ADNI Data Analysis Guide

Complete guide to analyzing the ADNI datasets used in this project.

## Available Analysis Scripts

### 1. ADNI Dataset Structure Analysis

**Script:** `data_analysis/adni_analysis.py`

**Purpose:** Analyzes the ADNIDenoise directory structure and MRI distributions

**What it does:**

- Counts unique patients by diagnosis (CN, MCI, AD)
- Calculates MRIs per patient statistics
- Analyzes CSV metadata files
- Generates comprehensive visualizations

**Usage:**

```bash
cd data_analysis
python3 adni_analysis.py
```

**Outputs:**

- `adni_analysis.png` - 4-panel visualization
- `adni_report.txt` - Detailed statistics

![ADNI Analysis](../data_analysis/adni_analysis.png)

---

### 2. Diagnosis Progression Analysis

**Script:** `data_analysis/analyze_dxsum.py`

**Purpose:** Longitudinal analysis of diagnosis changes over time

**What it does:**

- Tracks diagnosis changes across visits
- Identifies stable vs. progressive patients
- Analyzes transition patterns (CN→MCI→AD)
- Links MRI scan counts to diagnosis stability

**Usage:**

```bash
cd data_analysis
python3 analyze_dxsum.py
```

**Key outputs:**

- `dxsum_analysis_report.md` - Comprehensive progression analysis
- `dxsum_visualizations.png` - Diagnosis transition charts
- `patient_diagnosis_summary.csv` - Per-patient trajectories

See: [Diagnosis Progression Analysis](diagnosis-progression-analysis.md)

---

## Key Datasets

### 1. Clinical Data

| File | Records | Patients | Purpose |
|------|---------|----------|---------|
| `AD_CN_clinical_data.csv` | 1,179 | 1,179 | Clean training data (CN vs AD) |
| `clinical_data_all_groups.csv` | 1,980 | 1,472 | Full dataset with MCI |
| `dxsum.csv` | 12,227 | 2,311 | Longitudinal diagnosis tracking |

### 2. Patient Trajectories

**File:** `data_analysis/patient_diagnosis_summary.csv`

**Columns:**

- `PTID`: Patient ID
- `num_visits`: Total visits
- `first_diagnosis`: Baseline diagnosis
- `last_diagnosis`: Final diagnosis
- `diagnosis_changed`: Yes/No
- `num_unique_diagnoses`: How many different diagnoses
- `all_diagnoses`: Complete trajectory (e.g., "CN → CN → MCI → AD")

**Example insights:**

```csv
PTID,num_visits,first_diagnosis,last_diagnosis,diagnosis_changed
011_S_0002,10,CN,MCI,Yes  # Progressed from CN to MCI
023_S_0030,6,MCI,AD,Yes   # Progressed from MCI to AD
011_S_0023,10,CN,CN,No    # Remained stable CN
```

---

## Interesting Analyses to Run

### 1. Stability Rates by Baseline

**Question:** What percentage of patients maintain their diagnosis?

**Data:** `patient_diagnosis_summary.csv`

**Analysis:**

```python
import pandas as pd

df = pd.read_csv('patient_diagnosis_summary.csv')

for baseline in ['CN', 'MCI', 'AD']:
    baseline_patients = df[df['first_diagnosis'] == baseline]
    stable = baseline_patients[baseline_patients['diagnosis_changed'] == 'No']

    print(f"{baseline}: {len(stable)}/{len(baseline_patients)} "
          f"stable ({len(stable)/len(baseline_patients)*100:.1f}%)")
```

**Current results:**

- CN: 83.4% stable
- MCI: 57.2% stable
- AD: 98.7% stable

---

### 2. Conversion Time Analysis

**Question:** How long does it take for MCI to convert to AD?

**Data:** `dxsum.csv`

**Key variables:**

- `EXAMDATE`: Visit date
- `DIAGNOSIS`: Diagnosis code (1=CN, 2=MCI, 3=AD)
- `RID`: Patient ID

**Analysis approach:**

1. Filter patients with MCI → AD transition
2. Calculate time between first MCI and first AD diagnosis
3. Analyze distribution of conversion times

---

### 3. Scanner Impact on Diagnosis

**Question:** Does scanner type affect diagnosis accuracy?

**Data:** `dxsum.csv` (has scanner info)

**Analysis:**

- Group by `MANUFACTURER` and `FIELD_STRENGTH`
- Compare diagnosis distributions
- Check for scanner-specific biases

**Current scanner distribution:**

- SIEMENS: 65.4%
- GE: 21.0%
- PHILIPS: 13.6%

---

### 4. Visit Frequency and Detection

**Question:** Do more frequent visits lead to earlier detection?

**Data:** `dxsum.csv` + `patient_diagnosis_summary.csv`

**Hypothesis:** Patients with more scans are more likely to have diagnosis changes detected

**Current finding:**

- Stable patients: 5.69 MRI scans average
- Changed patients: 9.41 MRI scans average (+3.72 more)

**Interpretation:** More monitoring = better detection of changes

---

### 5. Fluctuation Patterns

**Question:** Why do some patients fluctuate between diagnoses?

**Data:** `patient_diagnosis_summary.csv`

**Patterns found:**

- 56 patients returned to baseline after changing
- Most common: MCI → CN → MCI (23 patients)
- Could indicate:
  - Diagnostic uncertainty
  - Treatment effects
  - Natural variation

**Example fluctuations:**

```
MCI → CN → MCI                  (23 patients)
CN → MCI → CN                   (18 patients)
MCI → AD → MCI                  (10 patients)
MCI → CN → MCI → CN → MCI       (1 patient - very unstable)
```

---

## Data Quality Considerations

### Missing Data by Dataset

**AD_CN_clinical_data.csv:**

- ✅ Minimal missing (<5% most features)
- ✅ Best for ML training
- ⚠️ Only CN and AD (no MCI)

**clinical_data_all_groups.csv:**

- ⚠️ 30-70% missing in most features
- ⚠️ Requires imputation
- ✅ Includes all groups (CN, MCI, AD)

**Impact on model:**

- Clean data: 99.6% accuracy
- Missing data: 71.4% accuracy

### Recommendations

1. **For training models:** Use `AD_CN_clinical_data.csv`
2. **For MCI prediction:** Use `clinical_data_all_groups.csv` (accept lower accuracy)
3. **For progression analysis:** Use `dxsum.csv` (longitudinal)
4. **For patient trajectories:** Use `patient_diagnosis_summary.csv`

---

## Running Custom Analyses

### Template Script

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
dxsum = pd.read_csv('dxsum.csv')
summary = pd.read_csv('data_analysis/patient_diagnosis_summary.csv')

# Your analysis here
# ...

# Save results
plt.savefig('my_analysis.png', dpi=300, bbox_inches='tight')
results_df.to_csv('my_results.csv', index=False)
```

### Best Practices

1. **Always check data types:** `df.dtypes`
2. **Handle missing values:** `df.isnull().sum()`
3. **Verify counts:** Cross-check with known totals
4. **Document findings:** Save markdown reports
5. **Version outputs:** Date-stamp analysis files

---

## Useful Queries

### Find all patients who progressed from CN to AD

```python
df = pd.read_csv('data_analysis/patient_diagnosis_summary.csv')
cn_to_ad = df[(df['first_diagnosis'] == 'CN') &
              (df['last_diagnosis'] == 'AD')]
print(f"Found {len(cn_to_ad)} patients: CN → AD")
```

### Get average visits by diagnosis outcome

```python
df = pd.read_csv('data_analysis/patient_diagnosis_summary.csv')
avg_visits = df.groupby(['first_diagnosis', 'last_diagnosis'])['num_visits'].mean()
print(avg_visits)
```

### Analyze transition timing

```python
dxsum = pd.read_csv('dxsum.csv')
dxsum['EXAMDATE'] = pd.to_datetime(dxsum['EXAMDATE'])

# Group by patient
for patient in dxsum['RID'].unique()[:10]:  # First 10 patients
    patient_data = dxsum[dxsum['RID'] == patient].sort_values('EXAMDATE')
    print(f"\nPatient {patient}:")
    print(patient_data[['EXAMDATE', 'DIAGNOSIS']].to_string(index=False))
```

---

## References

- **Analysis scripts:** `data_analysis/`
- **Visualizations:** `data_analysis/*.png`
- **Reports:** See [Diagnosis Progression Analysis](diagnosis-progression-analysis.md)
- **Dataset overview:** See [ADNI Dataset](adni-dataset.md)

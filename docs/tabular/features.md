# Clinical Features Reference

Comprehensive guide to all 13 clinical features used in the XGBoost model.

## Overview

The model uses neuropsychological assessments and demographic data to classify patients.

## Feature Importance

| Rank | Feature | Importance | Domain |
|------|---------|------------|--------|
| 1 | TRABSCOR | 37.3% | Executive Function |
| 2 | CATANIMSC | 10.1% | Semantic Memory |
| 3 | DSPANBAC | 7.6% | Working Memory |
| 4 | CLOCKSCOR | 7.6% | Visuospatial |
| 5 | DSPANFOR | 7.0% | Attention |
| 6 | BNTTOTAL | 5.9% | Language |
| 7 | TRAASCOR | 4.9% | Processing Speed |
| 8 | VSHEIGHT | 3.8% | Demographics |
| 9 | VSWEIGHT | 3.6% | Demographics |
| 10 | PTDOBYY | 3.4% | Demographics |

## Cognitive Assessments

### 1. TRABSCOR - Trail Making Test B (37.3%)

**What it measures:** Executive function, cognitive flexibility, task switching

**Test description:**
- Connect numbered and lettered circles in alternating sequence (1-A-2-B-3-C...)
- Time to complete the task (seconds)
- Lower scores = better performance (faster completion)

**Clinical significance:**
- **Most important feature** for AD detection
- Impaired in early AD due to executive dysfunction
- Requires mental flexibility and attention

**Typical scores:**
- Normal: < 78 seconds
- MCI: 78-273 seconds
- AD: > 273 seconds

---

### 2. CATANIMSC - Category Fluency (Animals) (10.1%)

**What it measures:** Semantic memory, language retrieval

**Test description:**
- Name as many animals as possible in 60 seconds
- Score = total number of unique animals named

**Clinical significance:**
- Tests semantic memory (stored knowledge)
- Sensitive to early AD changes
- Requires language and memory systems

**Typical scores:**
- Normal: > 17 animals
- MCI: 10-17 animals
- AD: < 10 animals

---

### 3. DSPANBAC - Digit Span Backward (7.6%)

**What it measures:** Working memory, mental manipulation

**Test description:**
- Repeat digits in reverse order
- Start with 2 digits, increase difficulty
- Score = longest sequence correctly reversed

**Clinical significance:**
- Tests working memory capacity
- Requires mental manipulation
- More difficult than forward span

**Typical scores:**
- Normal: 6-8 digits
- MCI: 4-6 digits
- AD: < 4 digits

---

### 4. CLOCKSCOR - Clock Drawing Test (7.6%)

**What it measures:** Visuospatial ability, executive function

**Test description:**
- Draw clock showing specific time (e.g., 11:10)
- Score based on accuracy (0-5 scale)

**Clinical significance:**
- Tests visuospatial construction
- Sensitive to parietal lobe dysfunction
- Simple screening tool

**Scoring:**
- 5 = Perfect
- 4 = Minor spacing errors
- 3 = Some numbers missing
- 2 = Significant errors
- 1 = Vague representation
- 0 = No reasonable attempt

---

### 5. DSPANFOR - Digit Span Forward (7.0%)

**What it measures:** Attention, short-term memory

**Test description:**
- Repeat digits in same order as heard
- Start with 2 digits, increase difficulty
- Score = longest sequence correctly repeated

**Clinical significance:**
- Tests basic attention span
- Less sensitive than backward span
- Baseline for working memory

**Typical scores:**
- Normal: 7-9 digits
- MCI: 5-7 digits
- AD: < 5 digits

---

### 6. BNTTOTAL - Boston Naming Test (5.9%)

**What it measures:** Confrontation naming, language

**Test description:**
- Name pictures of objects (60 items)
- Score = number correctly named
- May give semantic or phonemic cues

**Clinical significance:**
- Tests word retrieval
- Sensitive to temporal lobe function
- Language assessment

**Typical scores:**
- Normal: > 27/30
- MCI: 20-27/30
- AD: < 20/30

---

### 7. TRAASCOR - Trail Making Test A (4.9%)

**What it measures:** Processing speed, visual scanning

**Test description:**
- Connect numbered circles in sequence (1-2-3-4...)
- Time to complete (seconds)
- Lower scores = better performance

**Clinical significance:**
- Baseline for Trail B
- Tests processing speed
- Less sensitive than Trail B

**Typical scores:**
- Normal: < 29 seconds
- MCI: 29-78 seconds
- AD: > 78 seconds

---

## Demographics

### 8. VSHEIGHT - Height (3.8%)

**What it measures:** Physical characteristic (cm or inches)

**Clinical significance:**
- Weak predictor
- May correlate with early life nutrition
- Used in combination with other features

---

### 9. VSWEIGHT - Weight (3.6%)

**What it measures:** Physical characteristic (kg or pounds)

**Clinical significance:**
- Weak predictor
- BMI-related risk factors
- Used in combination with other features

---

### 10. PTDOBYY - Birth Year (3.4%)

**What it measures:** Age proxy

**Clinical significance:**
- Age is AD risk factor
- Cohort effects
- Education era differences

---

### 11-13. PTEDUCAT, PTGENDER, PTRACCAT

**PTEDUCAT - Education Years**
- Years of formal education
- Cognitive reserve factor
- Higher education = protective

**PTGENDER - Gender**
- Male = 1, Female = 2
- Women higher AD prevalence
- Gender-specific risk factors

**PTRACCAT - Race/Ethnicity**
- Categorical encoding
- Population-specific risk
- Socioeconomic factors

## Cognitive Domains

### Executive Function (37.3%)
- TRABSCOR - Primary discriminator
- Critical for AD detection

### Memory (17.7%)
- CATANIMSC (10.1%) - Semantic memory
- DSPANBAC (7.6%) - Working memory

### Attention & Processing (11.9%)
- DSPANFOR (7.0%)
- TRAASCOR (4.9%)

### Visuospatial (7.6%)
- CLOCKSCOR

### Language (5.9%)
- BNTTOTAL

## Data Quality Notes

### Missing Data (clinical_data_all_groups.csv)

**High missing (70%):**
- PTDOBYY, PTEDUCAT, PTRACCAT, VSHEIGHT

**Moderate missing (30-60%):**
- TRAASCOR, TRABSCOR, CATANIMSC, CLOCKSCOR, BNTTOTAL, DSPANFOR, DSPANBAC

**Low missing (<2%):**
- VSWEIGHT, PTGENDER

**Impact:**
- Model trained on clean data (AD_CN_clinical_data.csv)
- Full dataset uses median/mode imputation
- Performance degrades with missing data

## Clinical Interpretation

**Executive Function Dominance:**

Trail Making Test B (TRABSCOR) accounts for 37% of model decisions, confirming that executive dysfunction is the most reliable AD indicator.

**Multi-Domain Assessment:**

Best discrimination comes from combining multiple cognitive domains rather than relying on single tests.

**Cognitive Profile:**

AD patients show characteristic pattern:
- Severe executive dysfunction (low TRABSCOR)
- Impaired semantic memory (low CATANIMSC)
- Preserved attention (relatively normal DSPANFOR)

## References

- Trail Making Test: Reitan (1958)
- Category Fluency: Benton & Hamsher (1976)
- Digit Span: Wechsler Memory Scale
- Clock Drawing: Shulman et al. (1993)
- Boston Naming: Kaplan et al. (1983)

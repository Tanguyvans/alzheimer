# Alzheimer's Disease Classification — Multi-Cohort Multimodal Fusion

CN (Cognitively Normal) vs AD (Alzheimer's Disease) classification using 3D brain MRI and clinical tabular features from 3 cohorts (ADNI, OASIS, NACC).

## Key Results

Best model: **MLP Late Weighted Fusion** — AUC 0.950 +/- 0.003 (5 seeds)

| Method | Acc % | Bal Acc % | AUC | Seeds |
|--------|-------|-----------|-----|-------|
| MLP Late Wt | 90.9 +/- 1.5 | 89.1 +/- 0.5 | 0.950 +/- 0.003 | 4 |
| XGB Late Wt | 92.0 +/- 0.8 | 88.6 +/- 0.7 | 0.949 +/- 0.002 | 5 |
| MLP Early | 91.9 +/- 0.8 | 89.1 +/- 0.7 | 0.946 +/- 0.006 | 5 |
| XGB Early | 89.6 +/- 1.3 | 85.3 +/- 2.3 | 0.931 +/- 0.018 | 5 |
| Tab only (XGB) | 88.8 +/- 0.2 | 86.0 +/- 0.3 | 0.937 +/- 0.002 | 5 |
| MRI only | 86.7 +/- 0.7 | 82.7 +/- 0.6 | 0.901 +/- 0.010 | 5 |

Full results: [`experiments/report_multi_seed/`](experiments/report_multi_seed/)

## Datasets

| Cohort | Patients | MRI Scans | CN | MCI | AD |
|--------|----------|-----------|-----|-----|-----|
| ADNI | 2,311 | 17,827 | 39% | 42% | 19% |
| OASIS | 1,340 | 7,794 | 68% | 3% | 16% |
| NACC | 55,004 | 8,163 | 49% | 18% | 29% |

Combined test set: 910 samples (78.2% CN, 21.8% AD)

## Architecture

4 fusion strategies combining ResNet50 3D (MedicalNet pretrained) with 16 clinical features:

- **MLP Early Fusion**: ResNet3D (2048-d) + Tabular MLP (32-d) → concat → MLP classifier
- **MLP Late Fusion**: ResNet3D classifier + Tabular MLP → probability fusion
- **XGBoost Early Fusion**: ResNet3D features + tabular → XGBoost
- **XGBoost Late Fusion**: ResNet3D classifier + Tabular XGBoost → probability fusion

### 16 Tabular Features

| Category | Features |
|----------|----------|
| Demographics | AGE, PTGENDER, PTEDUCAT, PTMARRY |
| Cognitive tests | CATANIMSC, TRAASCOR, TRABSCOR, DSPANFOR, DSPANBAC, BNTTOTAL |
| Medical history | MH14ALCH, MH16SMOK, MH4CARD, MH2NEURL |
| Physical | VSWEIGHT, BMI |

## Project Structure

```
alzheimer/
├── experiments/
│   ├── resnet3d_mlp/                  # MLP Early + Late Fusion
│   │   ├── model.py                   # ResNet3DBackbone + EarlyFusionModel
│   │   ├── train.py                   # Early fusion training
│   │   ├── train_late_fusion.py       # Late fusion training
│   │   └── config.yaml
│   ├── resnet3d_xgboost/             # XGBoost Early + Late Fusion
│   │   ├── train_finetuned.py         # Early fusion (finetune + XGB)
│   │   ├── train_late_fusion.py       # Late fusion
│   │   └── config.yaml
│   ├── multimodal_fusion/             # Dataset + preprocessing
│   │   ├── dataset.py                 # MultiModalDataset (MRI + tabular)
│   │   └── data/combined_trajectory/  # Train/val/test CSV splits
│   ├── analyze_multi_seed.py          # Multi-seed analysis + Integrated Gradients
│   ├── generate_ig_all_models.py      # IG for all 4 models
│   ├── generate_report_docx.py        # Word report generator
│   ├── run_all_seeds.sh               # Run all experiments (15 seeds)
│   └── report_multi_seed/             # Results and reports
│       ├── summary_table.csv
│       ├── per_seed_metrics.csv
│       ├── delong_pvalues.csv
│       ├── boxplots.png
│       ├── roc_curves.png
│       ├── confusion_matrices.png
│       ├── delong_test.png
│       ├── resnet3d_fusion_report.docx
│       └── interpretability/          # Integrated Gradients maps
│           ├── mlp_early_fusion/      # 5 AD + 5 CN individual maps
│           ├── mlp_late_fusion/
│           ├── xgb_early_fusion/
│           ├── xgb_late_fusion/
│           ├── cross_model_comparison.png
│           ├── group_average_AD.png
│           ├── group_average_CN.png
│           ├── group_difference_AD_minus_CN.png
│           └── summary_figure.png
├── preprocessing/                     # MRI preprocessing pipelines
├── data/                              # Clinical/tabular CSV data
│   ├── adni/, oasis/, nacc/
│   └── combined/
└── paper/                             # Research paper (LaTeX)
```

## Reports

- **Word report**: [`experiments/report_multi_seed/resnet3d_fusion_report.docx`](experiments/report_multi_seed/resnet3d_fusion_report.docx) — Performance summary, DeLong tests, interpretability examples
- **Interpretability**: [`experiments/report_multi_seed/interpretability/`](experiments/report_multi_seed/interpretability/) — Integrated Gradients for all 4 models (same 5 AD + 5 CN patients), group averages, differential map (AD - CN)

## Quick Start

```bash
source env/bin/activate

# Train MLP Early Fusion (single seed)
cd experiments/resnet3d_mlp
python train.py --config config.yaml --output-dir results_early/seed_0 --seed 0

# Train MLP Late Fusion
python train_late_fusion.py --config config.yaml --output-dir results_late_fusion/seed_0 --seed 0

# Run multi-seed analysis
cd experiments
python analyze_multi_seed.py --gradcam

# Generate Integrated Gradients for all 4 models
cd experiments/resnet3d_mlp
python ../generate_ig_all_models.py --seed 2 --n-individual 5 --n-steps 100

# Generate Word report
cd experiments
python generate_report_docx.py
```

## Technology Stack

- **Python 3.12** with virtual environment
- **Deep Learning**: PyTorch, MONAI (ResNet50 3D, MedicalNet pretrained)
- **ML**: XGBoost, scikit-learn
- **Medical Imaging**: SimpleITK, nibabel, ANTsPy, nilearn
- **Interpretability**: Integrated Gradients (custom implementation)

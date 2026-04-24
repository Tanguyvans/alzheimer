# Interpretability results — organized layout

All results produced from paper-grade checkpoints (`cv_results/seed_42/fold_{0..4}/`)
which reproduce Table II at ±0.3% accuracy.

## Structure

```
01_ft_attention/                       § 2 — Tabular attention (FT-Transformer)
├── aggregated/                             Main paper figures (5-fold aggregated)
│   ├── ft_attn_rollout_AGG_seed42_traj.png     ← recommended main figure
│   ├── ft_attn_rollout_AGG_seed42_cn_ad.png
│   ├── ft_attn_last_layer_AGG_seed42_{traj,cn_ad}.png
│   └── *.csv                                    Per-feature stats with consistency
├── per_fold/                              Per-fold individual results
│   └── ft_attn_{last_layer,rollout}_seed42_fold{0..4}_{traj,cn_ad}.{png,csv}
└── raw_data/                              Raw attention tensors
    └── attn_seed42_fold{0..4}_{traj,cn_ad}.npy  (N, layers, heads, 17, 17)

02_vit_attention/                      § 3 — MRI spatial attention (ViT rollout)
├── paper_figures/                         Compact figures for paper
│   ├── vit_paper_main.png                      ← recommended main figure (2x3)
│   └── vit_paper_diff.png                      Discrimination map (AD - CN)
├── individual_patients/                   Per-patient overlays (no averaging)
│   └── vit_individual_patients_fold{0,2,4}.png  6 subjects on 3 folds
├── per_fold/                              Per-fold 2x3 figures (consistency check)
│   └── vit_per_fold_{0..4}.png
├── extended_slices/                       10-slice versions (multi-slice exploration)
│   ├── vit_roll_AGG_CN_slices.png
│   ├── vit_roll_AGG_AD_slices.png
│   └── vit_roll_AGG_comparison.png
└── raw_data/                              Raw attention grids
    ├── vit_roll_fold{0..4}_{CN,AD}_cases.npy    (4, 8, 8, 8) per class per fold
    ├── vit_roll_fold{0..4}_{CN,AD}_mean.npy     (8, 8, 8) fold mean
    └── vit_roll_AGG_{CN,AD,diff}.npy            (128, 128, 128) 5-fold aggregate

03_cross_modal/                        § 4 — Cross-modal agreement analysis
├── cross_modal_agreement_fig.png               Main figure (regimes + confusion + per-class)
├── cross_modal_scatter_fig.png                 MRI prob × Tab prob scatter
└── cross_modal_predictions.csv                 6065 rows of per-subject predictions
```

## Recommended figures for the camera-ready

| Section | Figure | Path |
|---------|--------|------|
| Tabular attention | `ft_attn_rollout_AGG_seed42_traj.png` | `01_ft_attention/aggregated/` |
| MRI spatial attention | `vit_paper_main.png` | `02_vit_attention/paper_figures/` |
| Cross-modal contribution | `cross_modal_agreement_fig.png` | `03_cross_modal/` |

See `REPORT.md` (one level up) for the full write-up and narrative.

## Scripts

Produced by (all in `../`):
- `ft_attention.py` — §2 per-fold extraction
- `aggregate_ft_attention.py` — §2 aggregation
- `vit_attention.py` — §3 per-fold extraction
- `vit_visualize.py` — §3 extended-slice views
- `vit_paper_figure.py` — §3 compact paper figures
- `vit_per_fold_figure.py` — §3 per-fold consistency check
- `vit_individual_patients.py` — §3 per-patient visualizations
- `cross_modal_analysis.py` — §4 agreement analysis

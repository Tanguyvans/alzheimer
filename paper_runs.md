# Paper camera-ready — interpretability work

## Contexte

Papier "End-to-End Multimodal Transformers for Multi-Cohort Alzheimer's Classification" accepté. Deadline camera-ready : **2026-04-27**. Priorité = ajouter une section **Model Interpretability** pour répondre aux deux reviewers (point commun) et renforcer la réponse au Reviewer 2.

## Où se trouve le vrai modèle du papier

**Checkpoints du modèle rapporté** (cross-modal ViT + FT-Transformer, 16 features, équations 1-5 du papier) :

```
experiments/multimodal_fusion/cross_task_cv_results/
  model_seed42_fold0.pth   (368 MB, state_dict brut)
  model_seed42_fold1.pth
  model_seed42_fold2.pth
  model_seed42_fold3.pth
  model_seed42_fold4.pth
  combined_train_val.csv
```

- Ce sont des `state_dict` bruts (pas de wrapper `{model_state_dict, config}` → charger avec `model.load_state_dict(torch.load(...))`)
- Seed 42 uniquement pour les checkpoints. Seeds 123/456 des metrics existent mais les poids ne sont pas sauvegardés
- Architecture attendue (à construire via `model.py:build_model`) :
  - ViT-Base (768-dim, 128³, patch 16³)
  - FT-Transformer (16 features, embed 64, 4 heads, 3 layers)
  - Cross-modal fusion (hidden 512, 8 heads, dropout 0.3, auxiliary_losses=True)

## Splits CV officiels

Les CSVs utilisés pour produire les chiffres du papier sont préservés dans :

```
experiments/ablation_mri_only/cv_results/seed_{42,123,456}/fold_{0-4}/
  train.csv       (4367 samples)
  val.csv
  test.csv        (1213 samples, incl. MCI converters)
  cn_ad_test.csv  (1130 samples, CN vs established AD)
```

16 features tabulaires conformes au papier :
`AGE, PTGENDER, PTEDUCAT, PTMARRY, CATANIMSC, TRAASCOR, TRABSCOR, DSPANFOR, DSPANBAC, BNTTOTAL, VSWEIGHT, BMI, MH14ALCH, MH16SMOK, MH4CARD, MH2NEURL`

## Correspondance chiffres papier ↔ repo

| Papier (Table II/III) | Source dans le repo | Valeur |
|-----------------------|---------------------|--------|
| Multimodal traj acc 92.4±1.0 | `multimodal_fusion/cv_results/cv_summary.json` | 92.43±0.98 |
| Multimodal cn_ad acc 93.3±0.9 | idem | 93.32±0.90 |
| MRI-only 83.3±0.7 | `ablation_mri_only/cv_results/cv_results.json` | 83.28 |
| Tabular-only 84.9±0.4 | `ablation_tabular_only/cv_results/cv_results.json` | 84.87 |

## Meilleur fold pour les figures

Seed 42, fold 1 — traj_acc **93.57%**, cn_ad_acc **94.08%**, AUC **0.968** (le plus haut du CV).

## Autres checkpoints utiles

- `experiments/mri_vit_ad/checkpoints/combined/best.pth` : ViT-only multimodal cohorts (val_acc 86.7%). Baseline pour la figure "attention ViT seul vs ViT+fusion".
- `experiments/mri_vit_ad/pretrained/vit_mae75_pretrained.pth` : poids MAE de Kunanbayev et al.

## Checkpoints à IGNORER (pas le modèle du papier)

- `multimodal_fusion/checkpoints/cn_ad_trajectory_combined_vit_fttransformer_gated_*` : 11 runs avec fusion `gated` et 7 features. Expérimentations, **pas le modèle du papier**.
- `multimodal_fusion/checkpoints/cn_ad_trajectory_multi_cohort_resnet_ft_transformer_cross_modal_20260107_160829` : run expérimental `multimodal_fusion_improved` avec 19 features. **Pas le modèle du papier** malgré l'archi cross-modal.

## Plan

### Jour 1 — sans GPU
1. Sanity check : charger `model_seed42_fold1.pth`, évaluer sur `ablation_mri_only/cv_results/seed_42/fold_1/test.csv`, doit retrouver ~93.57% de traj_acc (nécessite GPU ou CPU ~4h). **Reporter si GPU occupé.**
2. FT-Transformer attention : extraire attention `[CLS] → features` dans `tabular_encoder.transformer`, moyenner sur test set séparément pour CN et AD → bar chart "importance par feature × classe". 100% CPU, ~30 min.
3. SHAP sur la branche tabulaire : `shap.KernelExplainer` sur un wrapper qui n'évalue que la branche FT-Transformer + classifier auxiliaire, pour produire les contributions par feature. Complémentaire à l'attention.

### Jour 2 — avec GPU
4. ViT attention rollout (Abnar & Zuidema 2020) sur 4-8 cas CN/AD bien classés → overlay axial/sagittal/coronal. Zones attendues : hippocampe, lobe temporal médian, ventricules.
5. Cross-modal attention maps : visualiser les attention weights de `fusion.mri_cross_attn` et `fusion.tab_cross_attn` → figure "quelle feature tabulaire influence quelle région MRI" (figure "signature" correspondant à la fusion cross-modale décrite dans le papier).

### Jour 3 — rédaction
6. Section V.D "Model Interpretability" (~1/2 colonne) + 2-3 figures.
7. Paragraphe "Clinical workflow integration" dans V.B (réponse Reviewer 2).
8. Reformulation V.C "Limitations" pour expliciter le cross-cohort training comme forme de validation externe (réponse Reviewer 1).

## Points d'attention pour camera-ready

- Vérifier orthographe "Saïd" (encodage PDF)
- Figure 2 légende : expliciter 4 heads (FT-T) vs 8 heads (fusion)
- Table II : clarifier "Estab. AD" = AD établi sans MCI converters
- Acknowledgment NACC/ADNI très long (~1.5 pages) : vérifier si IEEE permet de raccourcir via lien
- Résultats multimodal ablation dans le papier indiquent AUC = 0.959 pour le multimodal, 0.826 pour MRI-only, 0.925 pour tabular-only (Table III)

## État d'avancement

- [x] Identification du checkpoint correspondant au papier
- [x] Validation des splits CV dans `ablation_*/cv_results/`
- [x] Plan d'interprétabilité validé
- [ ] Sanity check eval (en attente de GPU libre)
- [x] FT-T attention extraction (fold 1, seed 42)
- [ ] SHAP tabular
- [ ] ViT attention rollout
- [ ] Cross-modal attention maps
- [ ] Rédaction section Interpretability
- [ ] Paragraphe clinical workflow
- [ ] Reformulation Limitations

## Notes techniques

### PyTorch fast-path (gotcha)
`nn.TransformerEncoderLayer` en mode `eval()` + batch_first + norm_first déclenche la fast-path C++ `torch._transformer_encoder_layer_fwd` qui bypass silencieusement l'override de `_sa_block`. Pour capturer les attention weights, il faut override `forward` entièrement (voir `AttentionCapturingLayer` dans `interpretability/ft_attention.py`). Premier run avait produit des attention weights fantômes (std ~1e-7 entre samples) parce que la capture ne se déclenchait jamais.

### Aux `tab_classifier` inutilisable pour SHAP
Le classifier auxiliaire tabulaire du checkpoint papier donne 27% accuracy standalone sur le test set (biaisé vers AD, conséquence de `aux_loss_weight=0.3` + weighted CE + label smoothing, jamais optimisé pour classifier seul). Les gradients SHAP à travers ce head seront dégénérés. Options pour SHAP tabulaire :
1. Entraîner un MLP surrogate (~5 min CPU) sur les mêmes 16 features et faire SHAP dessus
2. Utiliser le modèle complet avec MRI mean-feature précalculée → nécessite une session GPU pour précalculer les MRI features
3. Utiliser `attention × |input value|` comme proxy d'importance (pas vraiment SHAP mais interprétable)

## Résultats FT-T attention (fold 1, seed 42)

Test set AD-trajectory (1213 samples, 950 CN / 263 AD). Attention CLS→feature, dernière couche, moyennée sur les heads, moyenne par classe.

**Features où AD > CN** (le modèle regarde davantage ces features en AD) :
- TMT-A : +0.023 (ralentissement fonction exécutive — marqueur AD classique)
- Cardiac Hx : +0.011 (facteur de risque CV)
- Weight : +0.006
- TMT-B : +0.005
- Neurological Hx : +0.005

**Features où AD < CN** (le modèle diminue son attention en AD) :
- BMI : -0.019 (cohérent avec perte de poids en AD)
- Digit Span Bwd : -0.016 (paradoxal : feature discriminative mais moins attendue)
- Smoking Hx : -0.007

Artefacts : `interpretability/results/ft_attn_{last_layer,rollout}_seed42_fold1_{traj,cn_ad}.{csv,png}` + raw attention matrices en `.npy`.

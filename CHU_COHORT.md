# Cohorte CHU — Prétraitement IRM et inférence

Adaptation du pipeline ADNI/NACC à la cohorte externe **CHU** (27 sujets, centres La Louvière / Jolimont / CHR Mons, protocole bilan mémoire clinique).

## Résumé en chiffres

| | Valeur |
|---|---:|
| Sujets CHU total | **27** (29 acquisitions, COGN0613 a 2 timepoints) |
| T1 3D MPRAGE utilisable | **11** |
| T1 2D 5mm (clinique routine) | **15** |
| T1 2D Spin-Echo isolé | 1 (COGN0086) |
| Zip sans DICOM valide | 2 (COGN0485, 0487 = PNG) |
| Sujets avec labels cliniques | 26 (13 AD, 13 MCI, 0 CN) |
| Sujets avec features cliniques complètes | 26 |

**Décision** : préprocessing complet uniquement pour les 11 sujets 3D T1. Les autres sont utilisables uniquement en tabulaire.

Inventaire détaillé (par sujet, par série DICOM) : `/home/tanguy/Desktop/irm_chu_work/COHORT_INVENTORY.md`.

## Pipeline de prétraitement

```
DICOM (zip ou flat)
  ↓   preprocessing/imaging/chu_diagnose_series.py
      → series_report.csv (score chaque série pour T1-3D-MPRAGE)
  ↓   preprocessing/imaging/chu_pipeline.py --step convert
      → nifti/ (11 volumes 3D T1)
  ↓   chu_pipeline.py --step register
      → registered/ (N4 bias + MNI SyN via ANTs)
  ↓   chu_pipeline.py --step skull
      → skull/ (SynthStrip Docker, MNI 197×233×189 @ 1mm iso)
  ↓   preprocessing/imaging/chu_qc_full.py
      → qc/ (Dice vs masque MNI, NCC vs template, mosaïques PNG)
```

Le format de sortie est **identique à NACC-skull / ADNI-skull** → compatible avec les dataloaders existants.

**QC global** : 10/11 sujets avec Dice ≥ 0.92 vs masque MNI. COGN0613 (scan gadolinium) à 0.84 — utilisable mais flaggué.

## Scripts ajoutés

```
preprocessing/imaging/
├── chu_diagnose_series.py   # Scorer + CSV rapport de séries T1
├── chu_pipeline.py          # convert / register / skull (3 étapes)
├── chu_qc.py                # QC rapide (mosaïque 3 vues)
└── chu_qc_full.py           # QC complet (Dice, NCC, PNG par sujet)

experiments/multimodal_fusion/
├── prepare_chu_test.py      # Excel CHU → chu_test.csv (11) + chu_tabular_all.csv (26)
└── infer_chu.py             # Inférence multimodale fold_4/model.pth

experiments/ablation_tabular_only/
├── infer_chu.py             # Inférence tabulaire-seul (26 sujets)
└── chu_radio_compare.py     # Croisement avec le score MTAS du radiologue
```

## Inférence

Deux modèles évalués, même fold : `seed_42 / fold_4`.

| Modèle | Sujets utilisables | Accuracy |
|---|---:|:---:|
| Multimodal (IRM 3D + tabulaire) — `multimodal_fusion` | 9 labellisés | **100%** (9/9) |
| Tabulaire seul — `ablation_tabular_only` | 26 labellisés | **92.3%** (24/26) |

**Détails :**
- Cohorte CHU entièrement symptomatique (0 CN, 13 AD, 13 MCI) → on ne mesure que la **sensibilité**, pas la spécificité.
- Modèle multimodal : détection parfaite des 4 AD et 5 MCI avec IRM 3D. COGN0613 (gadolinium) prédit CN (OOD).
- Modèle tabulaire : 13/13 AD + 11/13 MCI détectés. Les 2 MCI manqués (COGN0078 P=0.25 ; COGN0427 P=0.47) ont tous les deux **MTAS patho** (atrophie temporale visible à l'IRM) — cas où l'IRM aurait aidé, mais ils n'ont que du T1 2D inutilisable.
- Concordance modèle tabulaire ↔ score MTAS radiologue : **15/25 (60%)**. Biais pro-AD attendu sur une cohorte 100% symptomatique.

Fichiers de sortie (non versionnés, générés localement) :
```
experiments/multimodal_fusion/results/chu_predictions.csv            (11 lignes)
experiments/ablation_tabular_only/results/chu_predictions_tabular.csv (26 lignes)
experiments/ablation_tabular_only/results/chu_radio_compare.csv      (26 + MTAS/Fazekas)
```

### Prédictions multimodales (11 sujets avec IRM 3D)

Modèle : `multimodal_fusion` fold_4 / seed_42. `P(AD)` = probabilité de la classe AD trajectory (softmax).

| Sujet | DX réel | P(AD) | Prédiction |
|---|---|---:|:-:|
| COGN0006 | MCI | 1.000 | AD ✓ |
| COGN0043 | MCI | 0.981 | AD ✓ |
| COGN0066 | AD  | 0.997 | AD ✓ |
| COGN0169 | MCI | 1.000 | AD ✓ |
| COGN0403 | ?   | 1.000 | AD |
| COGN0405 | MCI | 1.000 | AD ✓ |
| COGN0425 | AD  | 1.000 | AD ✓ |
| COGN0428 | AD  | 1.000 | AD ✓ |
| COGN0464 | AD  | 1.000 | AD ✓ |
| COGN0494 | MCI | 1.000 | AD ✓ |
| COGN0613 | ?   | 0.014 | CN (OOD, scan gadolinium) |

**Accuracy sur 9 sujets labellisés : 100%** (9/9).

### Prédictions tabulaires (26 sujets avec clinique, IRM absente ou 2D)

Modèle : `ablation_tabular_only` fold_4 / seed_42.

| Sujet | DX | P(AD) | Prédiction |
|---|---|---:|:-:|
| COGN0006 | MCI | 0.975 | AD ✓ |
| COGN0043 | MCI | 0.729 | AD ✓ |
| COGN0065 | MCI | 0.980 | AD ✓ |
| COGN0066 | AD  | 0.913 | AD ✓ |
| **COGN0078** | **MCI** | **0.246** | **CN** ✗ |
| COGN0086 | AD  | 0.986 | AD ✓ |
| COGN0112 | AD  | 0.980 | AD ✓ |
| COGN0133 | MCI | 0.878 | AD ✓ |
| COGN0155 | MCI | 0.519 | AD ✓ |
| COGN0169 | MCI | 0.986 | AD ✓ |
| COGN0194 | AD  | 0.970 | AD ✓ |
| COGN0199 | AD  | 0.971 | AD ✓ |
| COGN0255 | MCI | 0.978 | AD ✓ |
| COGN0294 | AD  | 0.890 | AD ✓ |
| COGN0405 | MCI | 0.798 | AD ✓ |
| COGN0425 | AD  | 0.966 | AD ✓ |
| **COGN0427** | **MCI** | **0.474** | **CN** ✗ |
| COGN0428 | AD  | 0.934 | AD ✓ |
| COGN0464 | AD  | 0.984 | AD ✓ |
| COGN0471 | MCI | 0.972 | AD ✓ |
| COGN0480 | AD  | 0.973 | AD ✓ |
| COGN0485 | AD  | 0.973 | AD ✓ |
| COGN0487 | AD  | 0.985 | AD ✓ |
| COGN0494 | MCI | 0.979 | AD ✓ |
| COGN0644 | AD  | 0.966 | AD ✓ |
| COGN0680 | MCI | 0.974 | AD ✓ |

**Accuracy : 24/26 = 92.3%** — les 2 manqués (COGN0078, COGN0427) sont des MCI avec MMSE 25 et tests limites, mais tous deux ont un **MTAS patho** (atrophie hippocampique visible au radiologue), signal que l'IRM aurait aidé.

### Concordance modèle tabulaire ↔ MTAS radiologue (25 sujets)

|  | MTAS normal (rad dit CN) | MTAS patho (rad dit AD) |
|---|:-:|:-:|
| Modèle → CN | 0 | 2 |
| Modèle → AD | 8 | 15 |

**Agreement : 15/25 = 60%**. Biais pro-AD du modèle attendu sur une cohorte 100% symptomatique.

## Mapping Excel CHU → features ADNI

Les Excel CHU (`bilan.xlsx`, `final.xlsx`) ont été mappées sur les 16 features d'entraînement ADNI/NACC. Résumé :

| Feature ADNI | Source CHU | Transform |
|---|---|---|
| AGE | `Age` | direct |
| PTGENDER | `Sexe (M=0, F=1)` | +1 → ADNI (1=M, 2=F) |
| PTEDUCAT | `Niveau socio-culturel Poitrenaud (1-4)` | 1→6 / 2→9 / 3→12 / 4→16 ans |
| PTMARRY | — non collecté | -4 (missing) |
| CATANIMSC | `Fluences sémantiques` | direct |
| TRAASCOR | `Trail Making A` | parse, "stop" → -4 |
| TRABSCOR | `Trail Making B temps` | parse, "stop" → -4 |
| DSPANFOR | `Empans endroit` | direct |
| DSPANBAC | `Empans envers` | direct |
| BNTTOTAL | `Dénomination Lexis` | substitut (pas le vrai BNT) |
| VSWEIGHT | — non collecté | -4 |
| BMI | `BMI` | direct |
| MH14ALCH | `Consommation alcool (UA/sem)` | ≥14 → 1, sinon 0 |
| MH16SMOK | `Tabagisme actif` | direct |
| MH4CARD | `Cardiopathie` | direct |
| MH2NEURL | `Antécédants maladie neurologique` | direct |

Valeurs manquantes codées `-4.0` comme dans NACC. Les labels sont binaires CN (0) vs AD_trajectory (1) ; MCI mappé sur 1 (tâche `cn_ad_trajectory`).

## Limitations identifiées

1. **Aucun sujet CN dans la cohorte CHU** → pas de mesure de spécificité, les modèles sur-détectent l'AD.
2. **Protocole clinique sans MPRAGE** sur 15/27 sujets → IRM inutilisable pour le ViT 3D (résolution Z 5mm vs 1mm entraînement). C'est une limite d'acquisition non-réparable.
3. **Features cliniques incomplètes** : PTMARRY (100%), VSWEIGHT (100%), PTEDUCAT (38%), TRABSCOR (54%) manquants → compensé par imputation médiane côté loader, mais décalage inévitable avec la distribution d'entraînement.
4. **2 sujets exportés en PNG au lieu de DICOM** (COGN0485, 0487) → à ré-exporter depuis le PACS.

## Recommandations pour les vagues CHU futures

- **Demander à la radio d'ajouter une MPRAGE** au protocole bilan mémoire (~5 min/patient) pour maximiser l'exploitabilité IRM.
- **Collecter au moins quelques sujets CN** (contrôles appariés en âge) pour mesurer la spécificité.
- **Vérifier l'export PACS** des zips avant livraison (`find ... -iname "*.dcm"` doit retourner > 0).

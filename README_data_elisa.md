# Données Alzheimer — Guide pour Elisa

## Objectif du projet

Classification CN (Cognitively Normal) vs AD (Alzheimer's Disease) à partir d'IRM cérébrales 3D et de données cliniques tabulaires. Les données proviennent de 3 cohortes : **ADNI**, **OASIS** et **NACC**.

---

## Structure des dossiers

```
~/data/
├── data/                    # Données nettoyées, prêtes à l'emploi
│   ├── combined_trajectory/ # Splits finaux multimodaux (MRI + tabulaire)
│   ├── combined/            # Splits MRI-only multi-cohorte
│   ├── adni/                # Données ADNI nettoyées
│   ├── oasis/               # Données OASIS nettoyées
│   └── nacc/                # Données NACC nettoyées
│
└── raw_data/                # Données brutes originales (avant nettoyage)
    ├── adni/                # CSV cliniques bruts ADNI
    │   └── tabular_raw/     # 18 fichiers CSV ADNI directement téléchargés
    ├── oasis/               # 20+ fichiers OASIS3_UDS*.csv
    └── nacc/                # Fichiers NACC bruts + archives zip
```

---

## data/ — Données nettoyées

### combined_trajectory/ (le plus important)

C'est le dataset **principal utilisé par les expériences**. Chaque ligne = 1 scan IRM + 16 features cliniques.

| Fichier | Lignes | Description |
|---------|--------|-------------|
| train.csv | 4 245 | Set d'entraînement |
| val.csv | 910 | Set de validation |
| test.csv | 910 | Set de test |
| all.csv | 6 065 | Tout combiné |
| metadata.json | — | Métadonnées (stats, paramètres) |

**Colonnes :**
- `subject_id` — identifiant patient
- `scan_path` — chemin vers le fichier IRM .npy
- `DX` — diagnostic (CN ou AD)
- `label` — 0 = CN, 1 = AD
- `source` — cohorte d'origine (adni, oasis, nacc)
- **16 features tabulaires** (voir section ci-dessous)

### combined/

Splits MRI-only multi-cohorte (sans features tabulaires).

| Fichier | Lignes |
|---------|--------|
| mri_cn_ad_train.csv | 9 302 |
| mri_cn_ad_val.csv | 1 843 |
| mri_cn_ad_test.csv | 1 821 |

Colonnes : `scan_path, subject_id, group, label, source`

### adni/

| Fichier | Lignes | Description |
|---------|--------|-------------|
| adni_cn_ad.csv | 644 | Patients ADNI CN/AD avec 32 features cliniques |
| adni_cn_ad_trajectory.csv | 903 | Idem avec trajectoire longitudinale |
| mri_cn_ad_train/val/test.csv | 8 249 | Splits MRI-only ADNI |

### oasis/

| Fichier | Lignes |
|---------|--------|
| mri_cn_ad_train.csv | 1 656 |
| mri_cn_ad_val.csv | 331 |
| mri_cn_ad_test.csv | 372 |

### nacc/

| Fichier | Lignes | Description |
|---------|--------|-------------|
| nacc_tabular_mri.csv | 4 743 | Patients NACC avec features tabulaires + chemin IRM |

---

## Les 16 features tabulaires

Ce sont les variables cliniques utilisées par les modèles multimodaux (présentes dans `combined_trajectory/`).

| Catégorie | Feature | Description |
|-----------|---------|-------------|
| **Démographie** | AGE | Age du patient |
| | PTGENDER | Genre (1=Homme, 2=Femme) |
| | PTEDUCAT | Années d'éducation |
| | PTMARRY | Statut marital |
| **Tests cognitifs** | CATANIMSC | Fluence catégorielle (animaux) |
| | TRAASCOR | Trail Making Test A (vitesse) |
| | TRABSCOR | Trail Making Test B (flexibilité cognitive) |
| | DSPANFOR | Digit Span Forward (mémoire de travail) |
| | DSPANBAC | Digit Span Backward (mémoire de travail) |
| | BNTTOTAL | Boston Naming Test (langage) |
| **Antécédents** | MH14ALCH | Consommation d'alcool |
| | MH16SMOK | Tabagisme |
| | MH4CARD | Antécédents cardiovasculaires |
| | MH2NEURL | Antécédents neurologiques |
| **Physique** | VSWEIGHT | Poids |
| | BMI | Indice de masse corporelle |

Les valeurs manquantes sont imputées par la **médiane**.

---

## raw_data/ — Données brutes

### adni/

- `clinical_data_all_groups.csv` — Données cliniques tous groupes (CN, MCI, AD)
- `clinical_tabular_data.csv` — Données tabulaires extraites
- `ALL_4class_clinical.csv` — Classification 4 classes
- `dxsum.csv` — Résumé des diagnostics longitudinaux
- `mci_stable_patients.csv` — Patients MCI stables
- `tabular_raw/` — 18 fichiers CSV bruts téléchargés depuis ADNI (NEUROBAT, MEDHIST, VITALS, DXSUM, etc.)

### oasis/

20+ fichiers `OASIS3_UDS*.csv` correspondant aux formulaires UDS (Uniform Data Set) :
- `UDSa1` — Démographie
- `UDSa2` — Informant
- `UDSa3` — Historique familial
- `UDSa5` — Historique de santé
- `UDSb1` — Examen physique
- `UDSb4` — CDR (Clinical Dementia Rating)
- `UDSc1` — Évaluations cognitives
- `UDSd1` — Diagnostics
- etc.

### nacc/

- `investigator_*.csv` — Données investigateur NACC
- `nacc_tabular.csv`, `nacc_tabular_t1.csv` — Données tabulaires NACC
- `idaSearch_*.csv`, `nacc-t1_*.csv` — Recherches et métadonnées
- `*.zip` — Archives de scans MRI et PET

---

## Cohortes — Résumé

| Cohorte | Patients | Scans IRM | CN | MCI | AD |
|---------|----------|-----------|-----|-----|-----|
| ADNI | 2 311 | 17 827 | 39% | 42% | 19% |
| OASIS | 1 340 | 7 794 | 68% | 3% | 16% |
| NACC | 55 004 | 8 163 | 49% | 18% | 29% |

**Répartition du dataset combiné** : 71.3% CN, 28.7% AD (déséquilibre géré par focal loss).

# XGBoost 4-Class Classification: CN | MCI-stable | MCI→AD | AD

Classification multi-classe pour prédire le spectre cognitif complet de la maladie d'Alzheimer.

## Vue d'ensemble

**Objectif** : Distinguer les 4 stades cognitifs à partir de données cliniques tabulaires :
- **CN (0)** : Cognitively Normal (cognitivement normal)
- **MCI-stable (1)** : Mild Cognitive Impairment qui reste stable
- **MCI→AD (2)** : MCI qui va convertir vers Alzheimer
- **AD (3)** : Alzheimer's Disease confirmée

**Challenge principal** : Prédire au baseline (première visite) si un patient MCI va rester stable ou convertir vers AD.

## Pourquoi 4 classes au lieu de 2 ?

### ❌ Problème avec 2 classes (CN | AD+MCI→AD) :

Si on utilise seulement 2 classes :
```
Classes : CN (0) | AD+MCI→AD (1)
```

**Les patients MCI-stable seraient forcément mal classés** car :
- Ils ne sont **pas CN** (ils ont des troubles cognitifs)
- Ils ne sont **pas AD** (ils n'ont pas Alzheimer)
- Ils ne sont **pas MCI→AD** (ils ne vont pas convertir)

→ Le modèle est **obligé** de les classer incorrectement dans l'une des deux classes.

### ✅ Avantages de 4 classes :

```
Classes : CN (0) | MCI-stable (1) | MCI→AD (2) | AD (3)
```

1. **Cliniquement cohérent** : Reflète le spectre réel de la maladie
2. **Chaque patient a sa classe** : Pas de classification forcée incorrecte
3. **Plus utile** : Distinguer MCI-stable vs MCI→AD est l'objectif clé pour la recherche clinique
4. **Meilleure stratification** : Permet d'identifier les patients à risque de conversion

## Données longitudinales ?

### Structure des données :

Les données ADNI contiennent **2 visites maximum** par patient :
- **`bl` (baseline)** : 1387 scans - Première visite
- **`sc` (screening)** : 593 scans - Visite de suivi
- **508 patients** ont les 2 visites

### Limitation importante :

Les données tabulaires **ne capturent PAS la progression temporelle complète** !

**Exemple d'un patient MCI→AD** :
```
Patient X (MCI converter)
├── Visite 1 (bl) : État = MCI
│   └── Données : MMSE=26, CDR=0.5
│
└── Visite 2 (sc) : État = AD (déjà converti !)
    └── Données : MMSE=22, CDR=1.0
```

**Problème** : Entre les 2 visites, on ne sait pas :
- Quand exactement le patient a converti
- Les étapes intermédiaires de progression

### Solution : Baseline Only

**Approche adoptée** : Utiliser **uniquement les scans baseline** (`VISCODE == 'bl'`)

```
Input : Données au PREMIER scan (baseline) uniquement
Output : 4 classes
  - CN : Cognitively Normal
  - MCI-stable : MCI qui ne va PAS convertir
  - MCI→AD : MCI qui VA convertir (prédit au baseline !)
  - AD : Déjà Alzheimer
```

**Objectif du modèle** : Apprendre à distinguer au baseline (avant conversion) si un patient MCI va rester stable ou convertir vers AD, en utilisant uniquement les features de la première visite.

## Dataset

**Sources** :
- `clinical_data_all_groups.csv` : Toutes les données cliniques ADNI
- `AD_CN_MCI_to_AD.csv` : Liste des patients MCI qui ont converti vers AD

**Distribution attendue** (baseline only) :
- **CN** : ~700 patients
- **MCI-stable** : ~570 patients (MCI non-converters)
- **MCI→AD** : ~110 patients (MCI converters)
- **AD** : ~220 patients

**Total** : ~1600 patients (1 scan par patient)

**Déséquilibre de classes** :
- MCI→AD est la **classe minoritaire** (~7% du dataset)
- Solution : **Inverse Frequency Weighting** (Option 1)

## Pondération des classes (Inverse Frequency Weighting)

Pour gérer le déséquilibre, on utilise la **pondération par fréquence inverse** :

```python
# Formule : poids_i = N_total / (N_classes * N_échantillons_classe_i)

Poids calculés (approximatifs) :
- CN :         poids ≈ 0.57  (classe majoritaire → poids faible)
- MCI-stable : poids ≈ 0.70
- MCI→AD :     poids ≈ 3.64  (classe minoritaire → poids fort !)
- AD :         poids ≈ 1.82
```

**Effet** : Les erreurs sur MCI→AD sont **pénalisées ~6× plus** que sur CN, forçant le modèle à mieux apprendre cette classe difficile.

## Pipeline d'exécution

### Step 1 : Préparation des données

```bash
python3 01_prepare_data_4class.py \
  --all-groups-csv ../../data/clinical_data_all_groups.csv \
  --converters-csv ../../data/AD_CN_MCI_to_AD.csv \
  --output-dir data/splits \
  --seed 42
```

**Fonctionnalités** :
- Charge les 4 classes
- **Filtre baseline only** (`VISCODE == 'bl'`)
- Calcule AGE à partir de `EXAMDATE`
- Calcule BMI
- Remplit les valeurs manquantes (médiane)
- Split patient-level (70/15/15)

### Step 2 : Entraînement XGBoost

```bash
python3 02_train_xgboost_4class.py \
  --train-csv data/splits/train.csv \
  --val-csv data/splits/val.csv \
  --output-dir models
```

**Fonctionnalités** :
- Normalisation StandardScaler
- **Inverse Frequency Weighting** automatique
- XGBoost multiclasse (`multi:softprob`)
- Early stopping (20 itérations)
- Feature importance

### Step 3 : Évaluation

```bash
python3 03_evaluate_xgboost_4class.py \
  --model-dir models \
  --test-csv data/splits/test.csv \
  --output-csv predictions_4class.csv
```

**Génère** :
- Matrice de confusion 4×4 avec counts + pourcentages
- Barplot des métriques par classe (Precision, Recall, F1)
- Predictions CSV avec probabilités

## Résultats attendus

### Performance globale :

- **Accuracy attendue** : 70-80% (beaucoup plus difficile que binaire)
- **Balanced Accuracy** : 65-75%

### Performance par classe :

| Classe | Recall attendu | Difficulté |
|--------|----------------|------------|
| **CN** | 80-90% | Facile (bien distinct) |
| **MCI-stable** | 60-70% | Moyen (proche de MCI→AD) |
| **MCI→AD** | 40-60% | **Très difficile** (classe minoritaire + similaire à MCI-stable) |
| **AD** | 70-80% | Moyen (peut être confondu avec MCI→AD avancé) |

### Challenge principal : MCI-stable vs MCI→AD

La **distinction la plus difficile** est entre :
- **MCI-stable** : Patients avec troubles légers qui restent stables
- **MCI→AD** : Patients avec troubles légers qui vont progresser

**Au baseline, ces patients se ressemblent beaucoup !** Le modèle doit apprendre des patterns subtils pour les distinguer.

## Features utilisées (30)

### Démographie (6)
- AGE, PTGENDER, PTEDUCAT, PTRACCAT, PTHAND, PTMARRY

### Mesures physiques (3)
- VSWEIGHT, VSHEIGHT, BMI

### Antécédents médicaux (9)
- MH14ALCH, MH17MALI, MH16SMOK, MH15DRUG, MH4CARD, MHPSYCH, MH2NEURL, MH6HEPAT, MH12RENA

### Scores cognitifs (9)
- **MMSCORE** (Mini-Mental State Exam)
- TRAASCOR, TRABSCOR, TRABERRCOM (Trail Making Test)
- CATANIMSC (Fluence verbale)
- CLOCKSCOR (Clock Drawing)
- BNTTOTAL (Boston Naming Test)
- DSPANFOR, DSPANBAC (Digit Span)

### Évaluations cliniques (3)
- **CDGLOBAL** (Clinical Dementia Rating) ← Très prédictif
- BCFAQ (Functional Activities)
- BCDEPRES (Depression)

## Outputs

```
models/
├── xgboost_model_4class.json          # Modèle XGBoost entraîné
├── scaler.pkl                         # StandardScaler pour normalisation
├── feature_names.json                 # Liste des features
├── feature_importance.csv             # Importance des features
├── confusion_matrix_4class.png        # Matrice 4×4 avec %
└── per_class_performance.png          # Barplot Precision/Recall/F1

predictions_4class.csv                 # Prédictions avec probabilities
```

## Interprétation clinique

### Top features attendues :

1. **CDGLOBAL** (CDR) : Sévérité de la démence
2. **MMSCORE** (MMSE) : Fonction cognitive globale
3. **TRABSCOR** : Flexibilité cognitive (Trail Making B)
4. **AGE** : Âge du patient
5. **BCFAQ** : Activités fonctionnelles quotidiennes

### Patterns clés :

**MCI-stable vs MCI→AD** (au baseline) :
- MCI→AD a probablement :
  - MMSE légèrement plus bas
  - CDR plus élevé (0.5 vs 0)
  - Scores Trail Making plus lents
  - Plus de symptômes dépressifs

**CN vs MCI** :
- MMSE : CN ≈ 29-30, MCI ≈ 24-28
- CDR : CN = 0, MCI = 0.5

**MCI vs AD** :
- MMSE : MCI ≈ 24-28, AD ≈ 18-23
- CDR : MCI = 0.5, AD = 1-2

## Comparaison avec classification binaire

| Approche | Classes | Accuracy | Problème |
|----------|---------|----------|----------|
| **Binaire** | CN \| AD+MCI→AD | 98.69% | MCI-stable mal classés |
| **4-classes** | CN \| MCI-stable \| MCI→AD \| AD | ~70-80% | Plus réaliste cliniquement |

**Conclusion** : La classification 4-classes a une accuracy plus basse **mais est beaucoup plus utile cliniquement** car elle ne force pas les MCI-stable dans des classes incorrectes.

## Commandes complètes

```bash
# Tout en une commande
cd experiments/xgboost_4class && \
python3 01_prepare_data_4class.py \
  --all-groups-csv ../../data/clinical_data_all_groups.csv \
  --converters-csv ../../data/AD_CN_MCI_to_AD.csv \
  --output-dir data/splits \
  --seed 42 && \
python3 02_train_xgboost_4class.py \
  --train-csv data/splits/train.csv \
  --val-csv data/splits/val.csv \
  --output-dir models && \
python3 03_evaluate_xgboost_4class.py \
  --model-dir models \
  --test-csv data/splits/test.csv \
  --output-csv predictions_4class.csv
```

## Limitations et améliorations futures

### Limitations :

1. **Données non-longitudinales** : Pas de suivi temporel complet
2. **Classe MCI→AD minoritaire** : Difficile à apprendre (110 patients)
3. **Baseline only** : Perd ~25% des données (scans de suivi)
4. **Features statiques** : Pas de taux de déclin (MMSE/an, etc.)

### Améliorations possibles :

1. **Features longitudinales** :
   - Pour patients avec 2 visites : calculer ΔMM SE, ΔCDR
   - Ajouter "vitesse de déclin" comme feature

2. **Data augmentation** :
   - SMOTE pour MCI→AD (classe minoritaire)
   - Augmente artificiellement le nombre d'exemples

3. **Fusion multimodale** :
   - Combiner XGBoost (tabular) + 3D-ViT (imaging)
   - Accuracy attendue : +5-10% boost

4. **Hyperparameter tuning** :
   - Grid search sur max_depth, learning_rate, etc.
   - Optimiser spécifiquement pour MCI→AD recall

## Seed de reproductibilité

**Seed utilisé partout : `42`**

- Préparation des données : `--seed 42`
- XGBoost : `random_state=42`
- Splits : patient-level avec seed fixe

Tous les résultats sont **100% reproductibles**.

---

**Auteur** : Expériences ADNI - Classification Alzheimer
**Date** : Novembre 2025
**Dataset** : ADNI (Alzheimer's Disease Neuroimaging Initiative)

# Quick Start : vkola-lab Fusion (Machine avec KINGSTON)

**Gain attendu** : 66% → 82-88% accuracy
**Temps total** : 2-3 jours GPU

---

## ✅ Déjà Fait (sur cette machine)

1. ✅ Recherche SOTA architectures
2. ✅ Sélection vkola-lab/ncomms2022
3. ✅ Script de préparation données créé
4. ✅ Config 4 classes créée
5. ✅ Documentation complète

**Fichiers prêts à transférer** :
- `01_prepare_4class_mri_dataset.py`
- `task_config_4class.json`
- `README.md` (guide complet)
- `QUICK_START.md` (ce fichier)

---

## 🚀 À Faire (sur machine avec KINGSTON)

### Étape 1 : Setup (30 min)

```bash
# 1. Transférer fichiers vkola_adapter/
cd experiments
mkdir vkola_adapter
# Copier les fichiers depuis cette machine via clé USB ou rsync

# 2. Clone vkola-lab
git clone --depth 1 https://github.com/vkola-lab/ncomms2022.git
cd ncomms2022

# 3. Install dependencies
pip install torch torchvision nibabel shap xgboost catboost
```

---

### Étape 2 : Préparer Données (1h)

```bash
cd ../vkola_adapter

# Générer CSV avec MRI paths réels
python3 01_prepare_4class_mri_dataset.py \
  --clinical-csv ../../data/clinical_data_all_groups.csv \
  --converters-csv ../../data/AD_CN_MCI_to_AD.csv \
  --skull-dir /Volumes/KINGSTON/ADNI-skull \
  --output-dir lookupcsv/CrossValid \
  --seed 42
```

**Output attendu** :
```
✓ train.csv: ~1377 samples
✓ valid.csv: ~305 samples
✓ test.csv: ~298 samples

Class distribution:
  CN         : 600 (43.6%)
  MCI-stable : 510 (37.0%)
  MCI→AD     : 80 (5.8%)
  AD         : 187 (13.6%)
```

---

### Étape 3 : Copier vers vkola-lab (2 min)

```bash
# Copier CSV
cp lookupcsv/CrossValid/*.csv ../ncomms2022/lookupcsv/CrossValid/

# Copier config
cp task_config_4class.json ../ncomms2022/

# Créer train script
cd ../ncomms2022
```

---

### Étape 4 : Créer Script d'Entraînement (5 min)

**Fichier** : `ncomms2022/train_4class.py`

```python
#!/usr/bin/env python3
from model_wrappers import Multask_Wrapper
from nonImg_model_wrappers import NonImg_Model_Wrapper, Fusion_Model_Wrapper
from utils import read_json

def train_fusion():
    print("=" * 80)
    print("TRAINING FUSION MODEL (MRI + TABULAR)")
    print("=" * 80)

    model = Fusion_Model_Wrapper(
        tasks=['COG_4class'],
        csv_dir='lookupcsv/CrossValid/',
        seed=42
    )

    model.train()
    model.gen_score(['test'])
    model.shap()

    print("\n✓ Fusion training completed!")

if __name__ == '__main__':
    train_fusion()
```

---

### Étape 5 : Entraîner (2-3 jours)

#### A. CNN seul (MRI) - 1 jour

```python
python3 -c "
from model_wrappers import Multask_Wrapper
from utils import read_json

model = Multask_Wrapper(
    tasks=['COG_4class'],
    device=0,
    main_config=read_json('config.json'),
    task_config=read_json('task_config_4class.json'),
    seed=42
)
model.train()
thres = model.get_optimal_thres()
model.gen_score(['test'], thres)
"
```

**Accuracy attendue** : 75-80%

---

#### B. Tabular seul (XGBoost) - 30 min

```python
python3 -c "
from nonImg_model_wrappers import NonImg_Model_Wrapper

model = NonImg_Model_Wrapper(
    tasks=['COG_4class'],
    model_name='XGBoost',
    csv_dir='lookupcsv/CrossValid/',
    seed=42
)
model.train()
model.gen_score(['test'])
"
```

**Accuracy attendue** : 66-70%

---

#### C. Fusion (MRI + Tabular) - 1 jour ⭐

```bash
python3 train_4class.py
```

**Accuracy attendue** : **82-88%** 🎯

---

### Étape 6 : Évaluer Résultats (30 min)

```python
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

# Load predictions
df = pd.read_csv('tb_log/COG_4class_test_predictions.csv')

y_true = df['true_label'].values
y_pred = df['pred_label'].values

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)

# Classification report
class_names = ['CN', 'MCI-stable', 'MCI→AD', 'AD']
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# MCI→AD specific
mci_ad_mask = y_true == 2
mci_ad_recall = (y_pred[mci_ad_mask] == 2).mean()
print(f"\nMCI→AD Recall: {mci_ad_recall*100:.2f}%")
```

---

## 📊 Résultats Attendus

| Modèle | Accuracy | MCI→AD Recall | Temps |
|--------|----------|---------------|-------|
| XGBoost original | 66.23% | 17.7% | - |
| vkola CNN seul | 75-80% | 50-60% | 1 jour |
| vkola Tabular seul | 66-70% | 35-40% | 30 min |
| **vkola Fusion** ✅ | **82-88%** | **60-70%** | 1 jour |

**Gain total** : **+16-22% accuracy**, **+42-52% MCI→AD recall**

---

## ⚠️ Troubleshooting

### Problème 1 : MRI path not found

**Solution** :
```bash
# Vérifier KINGSTON monté
ls /Volumes/KINGSTON/ADNI-skull/ | head -5

# Tester un path
import nibabel as nib
mri = nib.load('/Volumes/KINGSTON/ADNI-skull/002_S_0295/..._registered_skull_stripped.nii.gz')
print(mri.shape)
```

---

### Problème 2 : CUDA out of memory

**Solution** :
```json
// Dans task_config_4class.json
{
  "COG_4class": {
    "batch_size": 2  // Réduire de 4 à 2
  }
}
```

---

### Problème 3 : MCI→AD recall faible

**Normal** : Class très minoritaire (5.8%)

**Solutions** :
- Class weights déjà dans config ✅
- SMOTE sur tabular
- Focal Loss pour CNN
- Data augmentation MRI

---

## 📝 Checklist Rapide

**Setup** :
- [ ] Clone vkola-lab
- [ ] Install dependencies

**Données** :
- [ ] Préparer CSV avec script
- [ ] Vérifier MRI paths accessibles
- [ ] Copier vers ncomms2022/

**Entraînement** :
- [ ] CNN seul (optionnel)
- [ ] Tabular seul (optionnel)
- [ ] Fusion (obligatoire)

**Évaluation** :
- [ ] Confusion matrix
- [ ] Classification report
- [ ] Comparer vs XGBoost original

---

## 🎯 Commande Minimale (Si Pressé)

**Si tu veux juste le meilleur résultat rapidement** :

```bash
# 1. Setup
git clone --depth 1 https://github.com/vkola-lab/ncomms2022.git
pip install torch nibabel shap xgboost

# 2. Préparer données
python3 01_prepare_4class_mri_dataset.py --skull-dir /Volumes/KINGSTON/ADNI-skull
cp lookupcsv/CrossValid/*.csv ncomms2022/lookupcsv/CrossValid/
cp task_config_4class.json ncomms2022/

# 3. Entraîner fusion directement
cd ncomms2022
python3 -c "
from nonImg_model_wrappers import Fusion_Model_Wrapper
model = Fusion_Model_Wrapper(tasks=['COG_4class'], csv_dir='lookupcsv/CrossValid/', seed=42)
model.train()
model.gen_score(['test'])
"
```

**Temps** : 1-2 jours (uniquement fusion, sans CNN/Tabular séparés)

---

## 📞 Support

Si problème, référence ces documents :
- `VKOLA_LAB_EXPLAINED.md` : Explication détaillée architecture
- `README.md` : Guide complet 5 jours
- `BEST_ARCHITECTURE_RECOMMENDATION.md` : Comparaison architectures

---

**Auteur** : Quick Start vkola-lab
**Date** : Novembre 2025
**Objectif** : 82-88% accuracy avec fusion multimodale

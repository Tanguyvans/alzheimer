# ResNet3D Experiments — CN vs AD Classification

**Dataset**: Combined trajectory (ADNI + OASIS + NACC) — Train: 4245 / Val: 910 / Test: 910 (78% CN / 22% AD)

**Backbone**: MONAI ResNet50 3D, pretrained on 23 medical datasets (MedicalNet)

---

## ResNet3D + XGBoost

| Méthode | Fusion | Acc | Bal Acc | Sens | Spec | AUC |
|---------|--------|-----|---------|------|------|-----|
| MRI only (ResNet3D) | — | 88.0% | 82.1% | 71.7% | 92.6% | 0.907 |
| Tabular only (XGBoost) | — | 88.4% | 85.8% | 81.3% | 90.3% | 0.937 |
| ResNet3D emb + Tab → XGBoost | Early | 89.6% | 85.3% | 77.8% | 92.8% | 0.928 |
| Average (probas) | Late | 90.4% | 84.6% | 74.2% | 94.9% | 0.946 |
| Weighted (probas) | Late | **92.5%** | **88.1%** | 80.3% | **95.9%** | **0.948** |
| Stacking (LogReg) | Late | 91.4% | 84.1% | 71.2% | 97.1% | 0.949 |

## ResNet3D + MLP

| Méthode | Fusion | Acc | Bal Acc | Sens | Spec | AUC |
|---------|--------|-----|---------|------|------|-----|
| MRI only (ResNet3D) | — | 85.3% | 81.5% | 74.7% | 88.2% | 0.898 |
| Tabular only (MLP) | — | 83.2% | 85.2% | 88.9% | 81.6% | 0.932 |
| ResNet3D + MLP concat | Early | 90.7% | 88.2% | 83.8% | 92.6% | 0.952 |
| Average (probas) | Late | 88.0% | 86.3% | 83.3% | 89.3% | 0.943 |
| Weighted (probas) | Late | **88.9%** | **88.3%** | **87.4%** | 89.3% | **0.947** |
| Stacking (LogReg) | Late | 88.5% | 78.4% | 60.6% | 96.2% | 0.877 |

---

## Notes

- **Early fusion MLP** : LayerNorm (remplace BatchNorm incompatible avec batch_size=2), end-to-end training
- **Early fusion XGBoost** : fine-tune ResNet3D (30 epochs, AMP), extraction embeddings 2048-d + 16 features tabulaires → un seul XGBoost
- **Late fusion** : chaque branche prédit indépendamment, puis combinaison des probabilités
  - *Average* : moyenne simple des probas
  - *Weighted* : moyenne pondérée (poids optimisés sur val)
  - *Stacking* : LogisticRegression sur les probas des deux branches
- **ResNet3D** : 30 epochs, backbone frozen 3 premiers epochs, unfrozen ensuite avec differential LR (backbone 10x plus faible), mixed precision (AMP)
- **MLP tabulaire** : LayerNorm, hidden dims [128, 64, 32], dropout=0.3, 100 epochs
- **XGBoost tabulaire** : max_depth=6, lr=0.1, subsample=0.8, 300 rounds, early stopping=30

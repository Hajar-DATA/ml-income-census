# Income Prediction — End-to-End ML Pipeline with Drift Analysis

**Dataset :** UCI Adult Census (32 561 observations, classification binaire)  
**Objectif :** Prédire si un individu gagne >50K$/an, avec analyse complète de robustesse au data drift

---

## Pipeline — 8 phases

| Phase | Contenu |
|-------|---------|
| 0 | EDA & preprocessing — valeurs manquantes, encodage, distributions, corrélations |
| 1 | Learning curves — Logistic Regression (baseline) vs Gradient Boosting |
| 2 | Cross-validation stratifiée 5-fold + Bootstrap OOB (100 itérations) |
| 3 | Hyperparameter tuning — RandomizedSearchCV (30 itérations) → accuracy **0.8717** |
| 4 | Interprétabilité — Permutation Importance + SHAP (summary, dependence, force plot) |
| 5 | Simulation de data drift — covariate shift sur âge, heures travaillées, statut marital, occupation |
| 6 | Détection du drift — KS, Wasserstein, PSI (numériques) · chi², JSD (catégorielles) |
| 7 | Stratégies de mitigation — feature suppression vs retraining · balanced accuracy 0.789 → 0.778 |

---

## Résultats clés

- Meilleur modèle : GradientBoostingClassifier (lr=0.065, max_depth=4, n_estimators=267)
- Accuracy : **0.867** · Balanced accuracy : **0.789** · F1 : **0.697**
- Top features (SHAP) : `capital_gain`, `marital_status_Married-civ-spouse`, `education_num`
- Sous drift : accuracy 0.867 → 0.827, balanced accuracy 0.789 → 0.655
- Après retraining sur domaine mixé : balanced accuracy remonte à **0.778**

---

## Stack

**Modélisation :** scikit-learn (GradientBoosting, Pipeline, ColumnTransformer, RandomizedSearchCV)  
**Interprétabilité :** SHAP (TreeExplainer, summary plot, dependence plot, force plot)  
**Data drift :** KS test · Wasserstein · PSI · Jensen-Shannon divergence · chi²  
**Data & viz :** pandas · matplotlib · seaborn · scipy

---

## Structure du repo

    ml-income-prediction/
    ├── README.md
    ├── notebook/
    │   └── ML_with_Python.ipynb
    ├── data/
    │   └── adult.csv
    └── report/
        └── ML_with_Python_final.pdf

---

## Données

UCI Adult Census Income dataset — [source](https://archive.ics.uci.edu/dataset/2/adult)  
Cible : `income` (<=50K / >50K) · Déséquilibre 76/24

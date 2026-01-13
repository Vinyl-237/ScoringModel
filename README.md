
PLAN FINAL DU CODE 

PARTIE 1 — Infrastructure de la pipeline

- Timer
- Chargement des datasets
- Encodage one-hot (VERSION KAGGLE)
- Fonctions d’agrégations
- Feature engineering
- Fusion des tables

PARTIE 2 — Modélisation

- Séparation train/test
- Création DU SCORE MÉTIER (pondération FP/FN)
- Cross-validation avec ce score
- Modèle LightGBM paramétrable
- Modèles alternatifs + GridSearchCV
- Gestion du déséquilibre (class_weight / SMOTE)

PARTIE 3 — Analyse & Visualisation

- Head() du dataframe final
- Importances LightGBM (bar plot)
- Radar plot des top features
- SHAP global + local

Comparaison des modèles :

Quelle pondération pour le score métier ?

Option 1 — Le coût du faux négatif est 5× plus grave (classique crédit)
Un client risqué accepté = très grave (FN)
Un bon client refusé = moins grave (FP)

Score métier :
Score = 1− ((1/5⋅FN + 1⋅FP)/Total)
​	
Option 2 — Pondération équilibrée
FN = 2 × FP (typique assurance)

Option 3 — Prédéfinir des poids
Exemple :
FN = 10
FP = 1

En gros : Quelle pondération utiliser pour le score métier ?
Option 1 (FN ×5)
Option 2 (FN ×2)
Option 3 (donne des valeurs)

Comment on sélectionnera le meilleur modèle ?

Le meilleur modèle sera celui qui :

1. Maximise le score métier
2. Réduit fortement FN (priorité absolue métier)
3. Garde un FP raisonnable
4. A un bon AUC (entre 0.75 et 0.82, sans overfit !)
5. Tenir compte du déséquilibre des classes
6. Est stable en cross-validation


Tester l’API localement
Dans un terminal : uvicorn src.api.app:app --reload

Pour lancer le test unitaire : pytest tests/

Comment visualiser data_drift_report.html (son rapport) :

- Depuis la racine du projet :
open reports/data_drift_report.html
Ça l’ouvrira directement dans ton navigateur par défaut (Safari / Chrome).
- Option pro (serveur local)
Utile pour partager ou éviter des soucis de chargement :
cd reports
python -m http.server 8000

Ok donne moi un plan de présentation en te basant sur ce référentiel en incluant une introduction, une conclusion et perspectives(Ce que tu peux encore améliorer (OPTIONNEL)), biensur il faut expliquer ce qu'est une API, MLOps
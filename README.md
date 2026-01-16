
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

PARTIE 4 — API et dasboard

1 - Tester l’API localement
Dans un terminal : uvicorn src.api.app:app --reload

2 - Pour lancer le test unitaire : pytest tests/

3 - Comment visualiser data_drift_report.html (son rapport) :

Depuis la racine du projet :
- Lancer le calcul du drift : python -m src.monitoring.data_drift
- OuvriR directement le raportage dans le navigateur : open src/report/data_drift_report.html
- Depuis le serveur local
Utile pour partager ou éviter des soucis de chargement :
cd reports
python -m http.server 8000

4 - Lancement du dashboard en local pour tester

Étape 1 : 
pip install -r requirements.txt

Étape 2 : Lancer l'API (Terminal 1)
Ouvrez un premier terminal (à la racine du projet) et lancez le serveur FastAPI :  uvicorn src.api.app:app --reload
Vous devriez voir un message indiquant : Uvicorn running on http://127.0.0.1:8000.
Gardez ce terminal ouvert.

Étape 3 : Lancer le Dashboard (Terminal 2)
Ouvrez un deuxième terminal et lancez Streamlit : streamlit run src/dashboard.py

Ok donne moi un plan de présentation en te basant sur ce référentiel en incluant une introduction, une conclusion et perspectives(Ce que tu peux encore améliorer (OPTIONNEL)), biensur il faut expliquer ce qu'est une API, MLOps

5 - Hebergement :

- Formulaire streamlit cloud
Name : Scoring Model.
Region : Frankfurt (plus proche) ou Ohio.
Branch : evolution (ou main si vous avez fusionné).
Runtime : Python 3.
Build Command : pip install -r requirements.txt
Start Command : uvicorn src.api.app:app --host 0.0.0.0 --port $PORT

- Mise en ligne de l'interface sur : share.streamlit.io.
Connection de GitHub.
cliquer sur "New app".
Remplissage du formulaire :
Repository : dépôt ScoringModel.
Branch : nom-de-la-branche.
Main file path : (src/dashboard.py) chemin vers le dashboard
Cliquer sur "Deploy".

-  Lancement
Ouvrir l'URL de l'API Render 5 minutes avant la démo (pour réveiller le serveur). qui affichera {"message": "API is running"}.
Ouvrir le Dashboard Streamlit.
Collez l'URL Render dans la sidebar.  
Faites votre démo !


https://scoringmodel-mt9w7d7psalzchsqbwza69.streamlit.app

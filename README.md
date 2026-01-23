# Projet de Scoring Crédit - "Prêt à dépenser"

Ce projet vise à développer un outil de scoring crédit pour calculer la probabilité qu'un client rembourse son crédit, puis à classifier la demande en crédit accordé ou refusé.

Il inclut :
- Un modèle de Machine Learning (LightGBM) entraîné sur des données historiques.
- Une API REST (FastAPI) pour servir les prédictions.
- Un Dashboard interactif (Streamlit) pour les chargés de relation client.
- Une chaîne CI/CD pour l'intégration continue.

## Structure du projet

Le projet est organisé comme suit :

*   `src/` : Code source principal.
    *   `api/` : Code de l'API (FastAPI).
    *   `dashboard.py` : Interface utilisateur (Streamlit).
    *   `preprocessing/` : Scripts de nettoyage et feature engineering.
    *   `training/` : Scripts d'entraînement et de scoring.
    *   `monitoring/` : Détection du Data Drift.
*   `tests/` : Tests unitaires (pytest).
*   `.github/workflows/` : Configuration de l'intégration continue (CI).
*   `data/` : Dossier pour les datasets (non versionné).
*   `models/` : Dossier pour les modèles sérialisés (.pkl).

## Installation

1.  Cloner le dépôt :
    ```bash
    git clone <url_du_repo>
    cd ScoringModel
    ```

2.  Installer les dépendances :
    ```bash
    pip install -r requirements.txt
    ```

## Utilisation

### 1. Lancer l'API (Backend)

L'API expose le modèle de prédiction.

```bash
uvicorn src.api.app:app --reload
```
L'API sera accessible sur `http://127.0.0.1:8000`.

### 2. Lancer le Dashboard (Frontend)

Le dashboard permet de visualiser les scores et l'interprétabilité (SHAP).

```bash
streamlit run src/dashboard.py
```

### 3. Tests et Qualité

Pour lancer les tests unitaires :
```bash
pytest tests/
```

Pour générer le rapport de Data Drift :
```bash
python -m src.monitoring.data_drift
```

## Déploiement

L'application est déployée sur le cloud :
*   **API :** Render / Azure App Service
*   **Dashboard :** Streamlit Cloud
*   **Lien public :** https://scoringmodel-mt9w7d7psalzchsqbwza69.streamlit.app

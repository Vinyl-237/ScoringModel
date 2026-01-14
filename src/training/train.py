"""
Pipeline d'entraînement du modèle LightGBM
- Cross-validation stratifiée
- Optimisation des hyperparamètres (GridSearchCV)
- Score métier
- Tracking des expériences avec MLflow
- Enregistrement du modèle dans le Model Registry
"""

import os
import joblib
import json
import mlflow
import mlflow.lightgbm
import pandas as pd
import datetime 

from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from src.training.scoring import optimize_decision_threshold
from src.config.config import config
from src.training.scoring import business_scorer
#from src.utils.timer import timer

def train_model():
    """
    Pipeline d'entraînement du modèle LightGBM
    - Cross-validation stratifiée
    - Optimisation des hyperparamètres (GridSearchCV)
    - Score métier
    - Tracking des expériences avec MLflow
    - Enregistrement du modèle dans le Model Registry
    """
    print("Démarrage de l'entraînement du modèle...")

    # 1. Chargement des données
    X_train = pd.read_pickle(
        os.path.join(config.DATA_DIR, "X_train.pkl")
    )
    y_train = pd.read_pickle(
        os.path.join(config.DATA_DIR, "y_train.pkl")
    )
    X_test = pd.read_pickle(
        os.path.join(config.DATA_DIR, "X_test.pkl")
    )
    y_test = pd.read_pickle(
        os.path.join(config.DATA_DIR, "y_test.pkl")
    )

    # 2. Initialisation MLflow
    mlflow.set_experiment("credit_scoring_lgbm")
    
    # --- MODÈLES DE COMPARAISON (Baseline) ---
    print("Évaluation des modèles de référence...")
    
    # A. Dummy Classifier (Naïf)
    dummy = DummyClassifier(strategy="stratified", random_state=42)
    dummy.fit(X_train, y_train)
    dummy_score = business_scorer(dummy, X_test, y_test)
    print(f"Dummy Business Score: {dummy_score:.4f}")
    
    # B. Régression Logistique (Linéaire)
    # Nécessite une mise à l'échelle (StandardScaler)
    logreg = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000)
    )
    logreg.fit(X_train, y_train)
    logreg_score = business_scorer(logreg, X_test, y_test)
    print(f"Logistic Regression Business Score: {logreg_score:.4f}")

    # --- MODÈLE PRINCIPAL (LightGBM) ---
    
    # 3. Définition du modèle
    lgbm_model = LGBMClassifier(
        objective="binary",
        class_weight="balanced",
        n_jobs=-1,
        random_state=42
    )

    # 4. Grille d’hyperparamètres
    param_grid = {
        "num_leaves": [63],
        "learning_rate": [0.1],
        "n_estimators": [400],
        "max_depth": [6, 8, 10],
        "min_child_samples": [20, 50, 100],
        "subsample": [0.8, 0.9, 1.0]
    }

    # 5. Cross-validation
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    # 6. GridSearchCV
    grid_search = GridSearchCV(
        estimator=lgbm_model,
        param_grid=param_grid,
        scoring=business_scorer,
        cv=cv,
        n_jobs=-1,
        verbose=1,
        refit=True
    )

    # 7. Entraînement + Tracking MLflow
    run_name = f"LGBM_GridSearch_CV_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    with mlflow.start_run(run_name=run_name):
        
        # Log des scores baseline
        mlflow.log_metric("dummy_business_score", dummy_score)
        mlflow.log_metric("logreg_business_score", logreg_score)

        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_

        # Log des hyperparamètres
        mlflow.log_params(grid_search.best_params_)

        # Log métrique métier
        mlflow.log_metric(
            "business_score",
            grid_search.best_score_
        )

        # Log modèle dans le registry
        mlflow.lightgbm.log_model(
            best_model,
            artifact_path="model",
            registered_model_name="credit_scoring_lgbm"
        )
        

        # Sauvegarde locale (backup)
        os.makedirs(config.MODELS_DIR, exist_ok=True)
        joblib.dump(
            best_model,
            os.path.join(config.MODELS_DIR, "final_model_LightGBM.pkl")
        )

        # 7. Recherche du meilleur seuil sur le jeu de test (après entraînement)
        y_proba = best_model.predict_proba(X_test)[:, 1]
        best_threshold_info = optimize_decision_threshold(
            y_true=y_test,
            y_proba=y_proba
        )
        # Sauvegarde du seuil
        with open(os.path.join(config.DATA_DIR, "best_threshold.json"), "w") as f:
            json.dump(best_threshold_info, f)

        print("Entraînement terminé – modèle enregistré dans MLflow")

if __name__ == "__main__":
    train_model()

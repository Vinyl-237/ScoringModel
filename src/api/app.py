# Import du framework FastAPI pour créer l’API
from fastapi import FastAPI

# Pour gérer les requêtes JSON entrantes
from pydantic import BaseModel

# Pour charger le modèle sauvegardé
import joblib

# Pour manipuler les données sous forme de DataFrame
import pandas as pd

# Importation du schéma de données client
from src.api.schemas import ClientData

# Importation de la fonction de décision
from src.training.scoring import make_decision

# Importation de la configuration
import os
import json
from src.config.config import config

# Création de l'application FastAPI
app = FastAPI(
    title="Home Credit Scoring API",
    description="API de prédiction du risque de défaut client",
    version="1.0"
)

# Chargement du modèle
model = joblib.load(os.path.join(config.MODELS_DIR, "final_model_LightGBM.pkl"))
# Chargement du seuil optimal
THRESHOLD_PATH = os.path.join(config.DATA_DIR, "best_threshold.json")
with open(THRESHOLD_PATH, "r") as f:
    THRESHOLD = json.load(f)["best_threshold"]

@app.post("/predict")
def predict(client: ClientData):
    """
    Endpoint de prédiction du risque client
    """

    # Conversion du JSON en DataFrame
    df = pd.DataFrame([client.model_dump()])

    # Prédiction de la classe (0 = remboursé, 1 = défaut)
    prediction = model.predict(df)[0]

    # Probabilité associée à la classe 1 (défaut)
    probability = model.predict_proba(df)[0][1]

    # Décision finale selon le seuil optimal
    decision = make_decision(probability, THRESHOLD)

    # Retour JSON
    return {
    "prediction": int(prediction),
    "probability_default": float(probability),
    "threshold_used": THRESHOLD,
    "prediction": decision,
    "decision": "REFUSÉ" if decision == 1 else "ACCEPTÉ"
    }

@app.get("/")
def root():
    """
    Endpoint racine pour vérifier que l'API fonctionne
    """
    return {
        "message": "API is running"
    }

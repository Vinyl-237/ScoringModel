# src/config.py
import os

class Config:

    # Chemin de base du projet
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    # Dossier pour le monitoring
    MONITORING_DIR = os.path.join(BASE_DIR, "monitoring")
    # Dossier pour le reporting
    REPORTING_DIR = os.path.join(BASE_DIR, "report")
    # Dossier contenant les CSV bruts du Home Credit Dataset
    DATA_DIR = os.path.join(BASE_DIR, "data")
    # Dossier où seront enregistrés les modèles
    MODELS_DIR = os.path.join(BASE_DIR, "models")

    TARGET = "TARGET"

    # Vérifie que les répertoires existent
    os.makedirs(BASE_DIR, exist_ok=True)
    os.makedirs(MONITORING_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

config = Config()

if __name__ == "__main__":
    print("BASE_DIR =", config.BASE_DIR)
    print("MONITORING_DIR =", config.MONITORING_DIR)
    print("REPORTING_DIR =", config.REPORTING_DIR)
    print("DATA_DIR =", config.DATA_DIR)
    print("MODELS_DIR =", config.MODELS_DIR)
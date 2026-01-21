# src/data_loader.py
import os
import pandas as pd
import sys

from src.config import config
from src.utils.timer import timer

def load_csv(filename: str) -> pd.DataFrame:
    """
    Charge un fichier CSV dans le dossier /data.
    """
    path = os.path.join(config.DATA_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Erreur: Le fichier {path} est introuvable.")
    return pd.read_csv(path)


def load_all_data():
    """
    Charge toutes les tables du dataset Home Credit.
    Equivalent de la partie chargement du notebook Kaggle,
    mais sous forme de module réutilisable dans tout le projet.
    """
    with timer("Chargement de tous les fichiers CSV"):
        application_train = load_csv("application_train.csv")
        application_test  = load_csv("application_test.csv")
        bureau            = load_csv("bureau.csv")
        bureau_balance    = load_csv("bureau_balance.csv")
        previous_app      = load_csv("previous_application.csv")
        pos_cash          = load_csv("POS_CASH_balance.csv")
        installments      = load_csv("installments_payments.csv")
        credit_card       = load_csv("credit_card_balance.csv")

    return {
        "train": application_train,
        "test": application_test,
        "bureau": bureau,
        "bureau_balance": bureau_balance,
        "previous": previous_app,
        "pos": pos_cash,
        "installments": installments,
        "credit": credit_card
    }

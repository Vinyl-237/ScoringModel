import os
import pandas as pd

from evidently.report import Report
from evidently.metrics import DataDriftTable
from src.config.config import config



def run_data_drift():
    print("Lancement de l'analyse de data drift...")

    # Chargement des données
    # X_train = pd.read_pickle(os.path.join(config.DATA_DIR, "X_train.pkl"))
    # X_test = pd.read_pickle(os.path.join(config.DATA_DIR, "X_test.pkl"))

    X_train = pd.read_pickle(os.path.join(config.DATA_DIR, "application_train.csv"))
    X_test = pd.read_pickle(os.path.join(config.DATA_DIR, "application_test.csv"))

    print(f"Train shape : {X_train.shape}")
    print(f"Test shape  : {X_test.shape}")

    # Création du rapport Evidently
    report = Report(
        metrics=[DataDriftTable()]
    )
    report.run(
        reference_data=X_train,
        current_data=X_test
    )

    # Dossier de reporting 
    os.makedirs(config.REPORTING_DIR, exist_ok=True)
    # Chemin de sauvegarde du rapport
    output_path = os.path.join(
        config.REPORTING_DIR,
        "data_drift_report.html"
    )

    # Sauvegarde du rapport
    report.save_html(output_path)
    print(f"Rapport sauvegardé : {output_path}")


if __name__ == "__main__":
    run_data_drift()

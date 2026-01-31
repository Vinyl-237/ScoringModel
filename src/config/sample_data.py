import pandas as pd
import os
from src.config.config import config

def generate_dashboard_sample():
    """
    Génère un fichier CSV allégé pour le dashboard à partir de application_train.csv.
    Ce fichier 'sample_dashboard.csv' sera utilisé pour les comparaisons de population.
    """
    source_path = os.path.join(config.DATA_DIR, "application_train.csv")
    target_path = os.path.join(config.DATA_DIR, "sample_dashboard.csv")
    
    if os.path.exists(source_path):
        print(f"Chargement des données source : {source_path}")
        df_full = pd.read_csv(source_path)
        
        print("Création de l'échantillon (1000 lignes)...")
        # Nom parlant pour la dataframe destinée au dashboard
        df_dashboard_sample = df_full.sample(1000, random_state=42)
        
        df_dashboard_sample.to_csv(target_path, index=False)
        print(f"Fichier échantillon créé avec succès : {target_path}")
    else:
        print(f"Erreur : Le fichier source {source_path} est introuvable.")

if __name__ == "__main__":
    generate_dashboard_sample()
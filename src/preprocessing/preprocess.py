import pandas as pd
import numpy as np

from src.utils.timer import timer

class Preprocessor:
    """
    Nettoyage et préprocessing des données.
    Mélange entre :
    - bonnes pratiques Kaggle
    - exigences de ton référentiel (OHE, imputations, normalisation, pas de fuite)
    """

    def __init__(self):
        self.numeric_features = None
        self.categorical_features = None
        self.imputation_values = {}
        self.final_columns = []

    # 1. Nettoyage général
    def basic_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        """Nettoyage de base :"""
        df = df.copy()

        # Remplacements de valeurs
        df.replace({365243: np.nan}, inplace=True)

        # Suppression des colonnes 100% NaN
        df.dropna(axis=1, how="all", inplace=True)

        return df

    def fit(self, df: pd.DataFrame):
        """
        Apprend les transformations à partir du jeu de données d'entraînement.
        - Calcule les moyennes pour l'imputation.
        - Détermine la liste finale des colonnes après one-hot encoding.
        """
        with timer("Fitting Preprocessor"):
            df_clean = self.basic_cleaning(df)
            self.numeric_features = df_clean.select_dtypes(include=[np.number]).columns.tolist()
            self.categorical_features = df_clean.select_dtypes(exclude=[np.number]).columns.tolist()

            # Apprendre les valeurs d'imputation
            for col in self.numeric_features:
                self.imputation_values[col] = df_clean[col].mean()
            for col in self.categorical_features:
                self.imputation_values[col] = "Unknown"

            # Apprendre les colonnes du one-hot encoding en simulant une transformation
            df_imputed = self._impute(df_clean)
            df_ohe = pd.get_dummies(df_imputed, drop_first=True)
            self.final_columns = df_ohe.columns.tolist()

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applique les transformations apprises au jeu de données."""
        with timer("Transforming data"):
            df_clean = self.basic_cleaning(df)
            df_imputed = self._impute(df_clean)
            df_ohe = pd.get_dummies(df_imputed, drop_first=True)

            # Aligner les colonnes avec celles du jeu d'entraînement
            missing_cols = set(self.final_columns) - set(df_ohe.columns)
            for c in missing_cols:
                df_ohe[c] = 0

            # S'assurer que l'ordre et le nombre de colonnes sont identiques
            return df_ohe[self.final_columns]

    def _impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Imputation interne utilisant les valeurs stockées."""
        df = df.copy()
        for col in df.columns:
            if col in self.imputation_values:
                df[col] = df[col].fillna(self.imputation_values[col])
        return df

import pandas as pd
import numpy as np

from src.utils.timer import timer

class Preprocessor:
    """
    Nettoyage et préprocessing des données Home Credit.
    Mélange entre :
    - bonnes pratiques Kaggle
    - exigences de ton référentiel (OHE, imputations, normalisation, pas de fuite)
    """

    def __init__(self):
        self.numeric_features = None
        self.categorical_features = None

    # 1. Nettoyage général
    def basic_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        """Nettoyage de base :"""
        df = df.copy()

        # Remplacements de valeurs
        df.replace({365243: np.nan}, inplace=True)

        # Suppression des colonnes 100% NaN
        df.dropna(axis=1, how="all", inplace=True)

        return df

    # 2. Séparation num / cat
    def detect_feature_types(self, df: pd.DataFrame):
        """Sauvegarde les noms des colonnes catégorielles / numériques"""
        self.numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_features = df.select_dtypes(exclude=[np.number]).columns.tolist()

    # 3. Imputation
    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Imputation (moyenne + 'Unknown')"""
        df = df.copy()

        # numériques -> moyenne
        df[self.numeric_features] = df[self.numeric_features].fillna(df[self.numeric_features].mean())

        # catégorielles -> Unknown
        df[self.categorical_features] = df[self.categorical_features].fillna("Unknown")

        return df

    # 4. Encodage One-Hot
    def one_hot_encode(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encodage OneHotEncoder style Kaggle mais sans fuite (fit séparé train/test)"""
        return pd.get_dummies(df, drop_first=True)

    # 5. MASTER METHOD
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Pipeline complète preprocessing"""
        with timer("Preprocessing complet"):
            df = self.basic_cleaning(df)
            self.detect_feature_types(df)
            df = self.impute(df)
            df = self.one_hot_encode(df)
        return df


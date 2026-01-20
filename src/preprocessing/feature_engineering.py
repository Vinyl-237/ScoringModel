# src/preprocessing/feature_engineering.py
import pandas as pd
import numpy as np
from src.utils.timer import timer

class FeatureEngineer:
    """
    Feature Engineering
    - Partiellement basé sur Kaggle (agrégations bureau, previous_app, etc.)
    - Optimisée pour :
        * vitesse
        * lisibilité
        * performance
    - Aucun data leakage
    """

    # -------------------------------------------
    # 1. AGRÉGATIONS (inspiré Kaggle, version optimisée)
    # -------------------------------------------
    def aggregate_numeric(self, df, group_var, df_name):
        """
        Fonction d'agrégation numérique :
        - https://www.kaggle.com/c/home-credit-default-risk
        simplifie les agrégations à moyenne/max/min/sum/count
        """
        df = df.copy()

        numeric_df = df.select_dtypes(include=[np.number])
        numeric_df[group_var] = df[group_var]

        agg = numeric_df.groupby(group_var).agg(['mean', 'min', 'max', 'sum'])
        agg.columns = ['{}_{}_{}'.format(df_name, col, stat) for col, stat in agg.columns]

        return agg

    # -------------------------------------------
    # 2. Transformations sur les tables secondaires
    # -------------------------------------------
    def process_bureau(self, bureau, bureau_balance):
        """Agrégations bureau + bureau_balance (Kaggle, simplifié)"""
        with timer("FE bureau + bureau_balance"):
            bureau_balance_agg = self.aggregate_numeric(
                bureau_balance, group_var="SK_ID_BUREAU", df_name="bb"
            )

            bureau = bureau.merge(bureau_balance_agg, on="SK_ID_BUREAU", how="left")
            return self.aggregate_numeric(bureau, group_var="SK_ID_CURR", df_name="bureau")

    def process_previous(self, previous_app):
        """Agrégations previous_application"""
        with timer("FE previous_application"):
            return self.aggregate_numeric(previous_app, "SK_ID_CURR", "prev")

    def process_installments(self, inst):
        """Agrégations installments_payments (version simplifiée)"""
        with timer("FE installments"):
            return self.aggregate_numeric(inst, "SK_ID_CURR", "ins")

    def process_pos(self, pos):
        """Agrégations POS_CASH_balance"""
        with timer("FE POS"):
            return self.aggregate_numeric(pos, "SK_ID_CURR", "pos")

    def process_credit(self, credit):
        """Agrégations credit_card_balance"""
        with timer("FE credit"):
            return self.aggregate_numeric(credit, "SK_ID_CURR", "cc")

    # -------------------------------------------
    # 3. MASTER METHOD
    # -------------------------------------------
    def merge_all(self, data_dict):
        """Fusionne toutes les tables secondaires dans train/test"""
        train = data_dict["train"]
        test  = data_dict["test"]

        with timer("Feature Engineering global"):
            bureau_agg = self.process_bureau(data_dict["bureau"], data_dict["bureau_balance"])
            prev_agg   = self.process_previous(data_dict["previous"])
            ins_agg    = self.process_installments(data_dict["installments"])
            pos_agg    = self.process_pos(data_dict["pos"])
            cc_agg     = self.process_credit(data_dict["credit"])

            for agg in [bureau_agg, prev_agg, ins_agg, pos_agg, cc_agg]:
                train = train.merge(agg, on="SK_ID_CURR", how="left")
                test  = test.merge(agg,  on="SK_ID_CURR", how="left")

        return train, test

"""
Fonctions de scoring métier et optimisation du seuil de décision.
"""

import numpy as np
import os
import json
from sklearn.metrics import confusion_matrix
from sklearn.metrics import make_scorer
from src.config.config import config

# 1. SCORE MÉTIER

def business_score(
    y_true,
    y_pred,
    weight_FN: int = 5,
    weight_FP: int = 1
):
    """
    Score métier pénalisant plus fortement les faux négatifs
    (clients à risque acceptés).

    Le score est borné entre 0 et 1.

    Parameters
    ----------
    y_true : array-like
        Vérités terrain
    y_pred : array-like
        Prédictions binaires
    weight_FN : int
        Poids des faux négatifs
    weight_FP : int
        Poids des faux positifs

    Returns
    -------
    float
        Score métier
    """

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Score métier (plus grand = meilleur)
    score = 1 / (1 + weight_FN * fn + weight_FP * fp)

    return score


# Scorer business
business_scorer = make_scorer(
    business_score,
    greater_is_better=True
)

# 2. OPTIMISATION DU SEUIL MÉTIER

def optimize_decision_threshold(
    y_true,
    y_proba,
    thresholds: np.ndarray = None,
    weight_FN: int = 5,
    weight_FP: int = 1
):
    """
    Recherche du seuil de décision optimal selon le score métier.

    Parameters
    ----------
    y_true : array-like
        Vérités terrain
    y_proba : array-like
        Probabilités prédites (classe 1)
    thresholds : np.ndarray
        Liste des seuils testés
    weight_FN : int
        Poids des faux négatifs
    weight_FP : int
        Poids des faux positifs

    Returns
    -------
    dict
        {
            "best_threshold": float,
            "best_score": float,
            "scores": list,
            "thresholds": list
        }
    """

    # if y_true is None or y_proba is None:
    #     try:
    #         with open(os.path.join(config.DATA_DIR, "best_threshold.json")) as f:
    #             return json.load(f)
    #     except FileNotFoundError:
    #         raise ValueError("Aucun seuil sauvegardé trouvé et aucune donnée fournie pour l'optimisation.")
    
    # Seuils par défaut
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.5, 50)

    # Calcul des scores pour chaque seuil
    scores = []

    # Boucle sur les seuils
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        score = business_score(
            y_true,
            y_pred,
            weight_FN=weight_FN,
            weight_FP=weight_FP
        )
        scores.append(score)

    # Recherche du meilleur seuil
    best_idx = int(np.argmax(scores))

    best_threshold = thresholds[best_idx]
    best_score = scores[best_idx]
    
    # Sauvegarde du meilleur seuil et score dans un fichier JSON
    try:
        with open(os.path.join(config.DATA_DIR, "best_threshold.json"), "w") as f:
            json.dump({
                "best_threshold": float(best_threshold),
                "best_score": float(best_score)
            }, f)
    except Exception as e:
        print(f"Attention: Impossible de sauvegarder le seuil optimal: {e}")

    return {
        "best_threshold": float(thresholds[best_idx]),
        "best_score": float(scores[best_idx]),
        "scores": scores,
        "thresholds": thresholds.tolist()
    }

# 3. DÉCISION FINALE
def make_decision(probability: float, threshold: float) -> int:
    """
    0 = accepté
    1 = refusé (défaut)
    """
    return int(probability >= threshold)

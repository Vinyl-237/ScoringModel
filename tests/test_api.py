# On importe TestClient pour tester l'API sans lancer de serveur réel
from fastapi.testclient import TestClient

# On importe l'application FastAPI
from src.api.app import app

# Création d'un client de test
client = TestClient(app)


def test_root_endpoint():
    """
    Test simple pour vérifier que l'API répond
    """
    response = client.get("/")  # Appel du endpoint racine
    assert response.status_code == 200  # Le serveur répond
    assert "message" in response.json()  # Le message est présent


def test_predict_valid_client():
    """
    Test du endpoint /predict avec un client valide
    """

    # Données d'entrée conformes au schéma ClientData
    payload = {
        "DAYS_BIRTH": -12000,
        "DAYS_EMPLOYED": -2000,
        "bureau_DAYS_CREDIT_UPDATE_mean": -30.0,
        "REGION_RATING_CLIENT": 2,
        "NAME_INCOME_TYPE_Working": 1,
        "DAYS_LAST_PHONE_CHANGE": -1000,
        "CODE_GENDER_M": 1,
        "DAYS_ID_PUBLISH": -3000,
        "REG_CITY_NOT_WORK_CITY": 1.0,
        "EXT_SOURCE_1": 0.45,
        "EXT_SOURCE_2": 0.62,
        "EXT_SOURCE_3": 0.58
    }

    # Appel POST vers /predict
    response = client.post("/predict", json=payload)

    # Vérification du statut HTTP
    assert response.status_code == 200

    # Vérification de la structure de la réponse
    json_response = response.json()
    assert "prediction" in json_response
    assert "probability_default" in json_response


def test_predict_invalid_payload():
    """
    Test avec une donnée invalide (champ manquant)
    """

    # Payload volontairement incomplet
    payload = {
        "DAYS_BIRTH": -12000
    }

    # Appel POST
    response = client.post("/predict", json=payload)

    # FastAPI doit refuser la requête
    assert response.status_code == 422  # Unprocessable Entity

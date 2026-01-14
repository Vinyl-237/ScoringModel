import streamlit as st
import requests
import json

# Configuration de la page
st.set_page_config(
    page_title="Scoring Crédit - Dashboard",
    page_icon="🏦",
    layout="centered"
)

# Titre et description
st.title("🏦 Prédiction de Solvabilité Client")
st.markdown("Ce dashboard permet d'interroger l'API de scoring pour évaluer le risque de défaut d'un client.")

# --- Configuration de l'API ---
# En local, utilisez http://127.0.0.1:8000/predict
# En production, remplacez par l'URL de votre API déployée (ex: https://mon-api.herokuapp.com/predict)
API_URL = "http://127.0.0.1:8000/predict"
# Détection automatique : Si on est sur Streamlit Cloud, on utilise l'URL de prod, sinon local
# Vous devez remplacer l'URL ci-dessous par VOTRE URL Render une fois déployée
PROD_API_URL = "https://VOTRE-NOM-APP.onrender.com/predict"
LOCAL_API_URL = "http://127.0.0.1:8000/predict"

# Si l'URL de prod est vide ou par défaut, on laisse le choix
API_URL = PROD_API_URL if "streamlit.app" in str(st.query_params) else LOCAL_API_URL

st.sidebar.header("Configuration")
api_url_input = st.sidebar.text_input("URL de l'API", API_URL)
api_url_input = st.sidebar.text_input("URL de l'API", value=API_URL)

# --- Formulaire de saisie ---
st.header("Informations du Client")

with st.form("client_form"):
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Données Personnelles")
        days_birth = st.number_input(
            "Âge (en jours négatifs, ex: -12000)", 
            value=-12000, 
            help="Exemple : -12000 correspond à environ 33 ans"
        )
        days_employed = st.number_input(
            "Ancienneté emploi (jours négatifs)", 
            value=-2000
        )
        code_gender_m = st.selectbox(
            "Genre", 
            options=[0, 1], 
            format_func=lambda x: "Homme (1)" if x == 1 else "Femme (0)"
        )
        name_income_type_working = st.selectbox(
            "Type de revenu : Travaillant", 
            options=[0, 1],
            index=1
        )

    with col2:
        st.subheader("Données Financières & Externes")
        ext_source_1 = st.slider("Source Externe 1 (Normalisée)", 0.0, 1.0, 0.5)
        ext_source_2 = st.slider("Source Externe 2 (Normalisée)", 0.0, 1.0, 0.5)
        ext_source_3 = st.slider("Source Externe 3 (Normalisée)", 0.0, 1.0, 0.5)
        
        bureau_days_credit_mean = st.number_input(
            "Moyenne jours crédit Bureau", 
            value=-300.5
        )
        pos_months_balance_min = st.number_input(
            "Solde mensuel min (POS)", 
            value=-5.0
        )

    # Autres champs requis par le schéma
    st.subheader("Autres indicateurs")
    c3, c4 = st.columns(2)
    with c3:
        region_rating = st.slider("Note région client", 1, 3, 2)
        days_last_phone = st.number_input("Dernier changement téléphone (jours)", value=-1000)
    with c4:
        days_id_publish = st.number_input("Publication ID (jours)", value=-3000)

    # Bouton de soumission
    submit_button = st.form_submit_button(label="🔍 Analyser le dossier")

# --- Logique de prédiction ---
if submit_button:
    # Construction du payload JSON respectant le schéma ClientData
    client_data = {
        "bureau_DAYS_CREDIT_mean": bureau_days_credit_mean,
        "DAYS_BIRTH": int(days_birth),
        "DAYS_EMPLOYED": int(days_employed),
        "REGION_RATING_CLIENT": int(region_rating),
        "NAME_INCOME_TYPE_Working": int(name_income_type_working),
        "DAYS_LAST_PHONE_CHANGE": int(days_last_phone),
        "CODE_GENDER_M": int(code_gender_m),
        "DAYS_ID_PUBLISH": int(days_id_publish),
        "pos_MONTHS_BALANCE_min": pos_months_balance_min,
        "EXT_SOURCE_1": ext_source_1,
        "EXT_SOURCE_2": ext_source_2,
        "EXT_SOURCE_3": ext_source_3
    }

    st.info("Envoi des données à l'API...")
    
    try:
        response = requests.post(api_url_input, json=client_data)
        
        if response.status_code == 200:
            result = response.json()
            
            st.success("Analyse terminée !")
            
            # Affichage du résultat
            col_res1, col_res2 = st.columns(2)
            col_res1.metric("Décision", result["decision"], delta_color="normal" if result["decision"]=="ACCEPTÉ" else "inverse")
            col_res2.metric("Probabilité de défaut", f"{result['probability_default']:.2%}")
            
            st.progress(result['probability_default'], text="Niveau de risque")
            st.caption(f"Seuil de décision utilisé : {result['threshold_used']:.3f}")
            
        else:
            st.error(f"Erreur API : {response.status_code}")
            st.text(response.text)
            
    except requests.exceptions.ConnectionError:
        st.error("Impossible de se connecter à l'API. Vérifiez qu'elle est bien lancée.")
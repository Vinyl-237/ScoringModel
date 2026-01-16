import streamlit as st
import requests
import json
import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Configuration de la page
st.set_page_config(
    page_title="Scoring Crédit - Dashboard",
    page_icon=":bank:",
    layout="wide"
)

# Titre et description
st.markdown("<h1 style='text-align: center;'> Prédiction de Solvabilité Client</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Dashboard de scoring pour évaluer le risque de défaut d'un client.</p>", unsafe_allow_html=True)

# --- Configuration de l'API ---
API_URL = "http://127.0.0.1:8000/predict"
# Détection automatique : Si on est sur Streamlit Cloud, on utilise l'URL de prod, sinon local
# URL config
PROD_API_URL = "https://scoring-model-0gz7.onrender.com/predict"
LOCAL_API_URL = "http://127.0.0.1:8000/predict"

# Si l'URL de prod est vide ou par défaut, on laisse le choix
API_URL = PROD_API_URL if "streamlit.app" in str(st.query_params) else LOCAL_API_URL

st.sidebar.header("Configuration")
api_url_input = st.sidebar.text_input("URL de l'API", value=API_URL)

# --- Mode de saisie ---
input_mode = st.sidebar.radio("Mode de saisie", ["Formulaire manuel", "Upload CSV"])

client_data = {}
use_file = False
analyze_trigger = False

if input_mode == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Charger un fichier client (CSV)", type="csv")
    if uploaded_file:
        df_upload = pd.read_csv(uploaded_file)
        # On prend la première ligne pour l'exemple
        client_data = df_upload.iloc[0].to_dict()
        st.info("Fichier chargé. Analyse du premier client du fichier.")
        use_file = True
        if st.button(":mag: Analyser ce fichier"):
            analyze_trigger = True

# --- Formulaire de saisie ---
left_col, right_col = st.columns([0.6, 0.4])

with left_col:
    st.header("Informations du Client")

    if not use_file:
        with st.form("client_form"):
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
            days_id_publish = st.number_input("Publication ID (jours)", value=-3000)
            region_rating = st.slider("Note région client", 1, 3, 2)

            st.subheader("Données Financières & Externes")
            ext_source_1 = st.slider("Source Externe 1", 0.0, 1.0, 0.5)
            ext_source_2 = st.slider("Source Externe 2", 0.0, 1.0, 0.5)
            ext_source_3 = st.slider("Source Externe 3", 0.0, 1.0, 0.5)
            
            bureau_days_credit_update_mean = st.number_input(
                "Moyenne jours update crédit Bureau", 
                value=-30.0
            )
            reg_city_not_work_city = st.selectbox(
                "La ville résidence est-elle différente de la ville de travail",
                options=[0.0, 1.0],
                format_func=lambda x: "Oui" if x == 1.0 else "Non"
            )
            days_last_phone = st.number_input("Dernier changement téléphone (jours)", value=-1000)

            # Bouton de soumission
            submit_button = st.form_submit_button(label=":mag: Analyser le dossier")
            
            if submit_button:
                # Construction du payload JSON manuel
                client_data = {
                    "DAYS_BIRTH": int(days_birth),
                    "DAYS_EMPLOYED": int(days_employed),
                    "bureau_DAYS_CREDIT_UPDATE_mean": bureau_days_credit_update_mean,
                    "REGION_RATING_CLIENT": int(region_rating),
                    "NAME_INCOME_TYPE_Working": int(name_income_type_working),
                    "DAYS_LAST_PHONE_CHANGE": int(days_last_phone),
                    "CODE_GENDER_M": int(code_gender_m),
                    "DAYS_ID_PUBLISH": int(days_id_publish),
                    "REG_CITY_NOT_WORK_CITY": float(reg_city_not_work_city),
                    "EXT_SOURCE_1": ext_source_1,
                    "EXT_SOURCE_2": ext_source_2,
                    "EXT_SOURCE_3": ext_source_3
                }
                analyze_trigger = True

# --- Logique de prédiction ---
with right_col:
    if analyze_trigger and client_data:
        st.header("Résultats de l'analyse")
        #st.info("Envoi des données à l'API...")
        
        try:
            response = requests.post(api_url_input, json=client_data)
            
            if response.status_code == 200:
                result = response.json()
                
                st.success("Analyse terminée !")
                
                # Affichage du résultat
                col_res1, col_res2, col_res3 = st.columns(3)
                col_res1.metric("Décision", result["decision"], delta_color="normal" if result["decision"]=="ACCEPTÉ" else "REFUSÉ")
                col_res2.metric("Probabilité de défaut", f"{result['probability_default']:.2%}")
                col_res3.metric("Probabilité de non défaut", f"{1 - result['probability_default']:.2%}")
                
                st.progress(result['probability_default'], text=f"Niveau de risque : {result['probability_default']:.2%} / 100%")
                st.caption(f"Seuil de décision utilisé : {result['threshold_used']:.3f}")
                
                # --- Interprétabilité (SHAP) ---
                st.markdown("---")
                st.subheader(":mag: Explication de la décision (SHAP)")
                
                if "shap_values" in result:
                    # Reconstruction de l'objet Explanation pour SHAP
                    shap_exp = shap.Explanation(
                        values=np.array(result["shap_values"]),
                        base_values=result["base_value"],
                        data=np.array([client_data[col] for col in result["feature_names"]]),
                        feature_names=result["feature_names"]
                    )
                    
                    # Affichage du Waterfall Plot
                    fig, ax = plt.subplots(figsize=(8, 6))
                    shap.plots.waterfall(shap_exp, max_display=10, show=False)
                    st.pyplot(fig)
                    
                    st.info("Caractéristiques de contribution à la décision : Les barres rouges poussent vers le défaut, les bleues vers l'acceptation.")
                
            else:
                st.error(f"Erreur API : {response.status_code}")
                st.text(response.text)
                
        except requests.exceptions.ConnectionError:
            st.error("Impossible de se connecter à l'API. Vérifiez qu'elle est bien lancée.")
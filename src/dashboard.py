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
    layout="wide",
    initial_sidebar_state="expanded"
)

# Titre et description
st.markdown("<h1 style='text-align: center;'> Prédiction de Solvabilité Client</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Dashboard de scoring pour évaluer le risque de défaut d'un client.</p>", unsafe_allow_html=True)

# --- Configuration de l'API ---
PROD_API_URL = "https://scoring-model-0gz7.onrender.com/predict"
LOCAL_API_URL = "http://127.0.0.1:8000/predict"

# Par défaut on utilise la PROD, sauf si on veut tester en local
API_URL = PROD_API_URL 

st.sidebar.header("Configuration")
api_url_input = st.sidebar.text_input("URL de l'API", value=API_URL)

client_data = {}
analyze_trigger = False

# --- Formulaire de saisie ---
left_col, right_col = st.columns([0.6, 0.4])

with left_col:
    st.header("Informations du Client")

    with st.form("client_form"):
        st.subheader("Données Personnelles")
        days_birth = st.number_input(
            "Âge (en jours négatifs, ex: -12000)", 
            value=-12000, 
            help="Entrez l'âge en jours. Une valeur négative est attendue (ex: -12000 jours ≈ 33 ans)."
        )
        days_employed = st.number_input(
            "Ancienneté emploi (jours négatifs)", 
            value=-2000,
            help="Nombre de jours depuis le début de l'emploi actuel (valeur négative)."
        )
        name_income_type_working = st.selectbox(
            "Type de revenu (salarié(1) ou autre(0))", 
            options=[0, 1],
            index=1,
            help="Sélectionnez 1 si le client travaille, 0 sinon."
        )
        days_id_publish = st.number_input(
            "Publication ID (jours)", 
            value=-3000,
            help="Nombre de jours depuis la publication de la pièce d'identité."
        )
        region_rating = st.slider("Note région client", 1, 3, 2, help="Note de 1 à 3 évaluant la région du client.")

        st.subheader("Données Financières & Externes")
        ext_source_1 = st.slider("Source Externe 1", 0.0, 1.0, 0.5, help="Score normalisé provenant d'une source de données externe 1.")
        ext_source_2 = st.slider("Source Externe 2", 0.0, 1.0, 0.5, help="Score normalisé provenant d'une source de données externe 2.")
        ext_source_3 = st.slider("Source Externe 3", 0.0, 1.0, 0.5, help="Score normalisé provenant d'une source de données externe 3.")
        
        bureau_days_credit_update_mean = st.number_input(
            "Jours moyens dernière maj crédit", 
            value=-30.0,
            help="Moyenne des jours depuis la dernière mise à jour des crédits au bureau (valeur négative)."
        )
        days_last_phone = st.number_input("Dernier changement de téléphone (jours)", value=-1000, help="Jours depuis le dernier changement de téléphone.")

        # Bouton de soumission
        submit_button = st.form_submit_button(label=":mag: Analyser")
        
        if submit_button:
            # Construction du payload JSON manuel
            client_data = {
                "DAYS_BIRTH": int(days_birth),
                "DAYS_EMPLOYED": int(days_employed),
                "bureau_DAYS_CREDIT_UPDATE_mean": bureau_days_credit_update_mean,
                "REGION_RATING_CLIENT_W_CITY": int(region_rating),
                "NAME_INCOME_TYPE_Working": int(name_income_type_working),
                "DAYS_LAST_PHONE_CHANGE": int(days_last_phone),
                "DAYS_ID_PUBLISH": int(days_id_publish),
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
                
                # Gestion de la couleur et de l'icône pour l'accessibilité (ne pas se baser que sur la couleur)
                decision_label = result["decision"]
                if decision_label == "REFUSÉ":
                    decision_icon = "⛔"
                    decision_color = "inverse" # Streamlit gère le rouge par défaut pour les deltas négatifs, mais ici on affiche du texte
                else:
                    decision_icon = "✅"
                
                col_res1.metric("Décision Recommandée", f"{decision_icon} {decision_label}")
                col_res2.metric("Probabilité de défaut", f"{result['probability_default']:.2%}")
                col_res3.metric("Seuil de décision", f"{result['threshold_used']:.3f}")
                
                st.progress(result['probability_default'], text=f"Niveau de risque : {result['probability_default']:.2%} / 100%")
                st.caption("Note : Si le niveau de risque dépasse le seuil, le crédit est refusé.")

                # --- Interprétabilité (SHAP) ---
                st.markdown("---")
                st.subheader("Détails de la décision (SHAP)")
                
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
                    # show=False permet de récupérer la figure pour Streamlit
                    shap.plots.waterfall(shap_exp, max_display=10, show=False) 
                    st.pyplot(fig)
                    
                    # Calcul de f(x) pour l'explication
                    f_x = shap_exp.base_values + np.sum(shap_exp.values)
                    
                    # WCAG 1.1.1 : Alternative textuelle au graphique
                    st.markdown("### 📝 Explication textuelle du graphique")
                    st.markdown(f"""
                    Le graphique ci-dessus (Waterfall Plot) décompose le score de risque :
                    *   **Valeur de base (moyenne)** : {shap_exp.base_values:.3f}
                    *   Les barres **rouges** indiquent les facteurs qui **augmentent** le risque.
                    *   Les barres **bleues** indiquent les facteurs qui **diminuent** le risque.
                    - Valeur du client : {f_x:.3f}
                    """)
                    
                    # Tableau de simulation des seuils
                    st.markdown("### Simulation de la décision selon le seuil")
                    thresholds_sim = [0.35, 0.45, 0.50]
                    sim_data = []
                    
                    for t in thresholds_sim:
                        dec_sim = "REFUSÉ" if result['probability_default'] >= t else "ACCEPTÉ"
                        sim_data.append({"Seuil": f"{t:.2f}", "Décision": dec_sim})
                        
                    st.table(pd.DataFrame(sim_data).set_index("Seuil"))
                
            else:
                st.error(f"Erreur API : {response.status_code}")
                st.text(response.text)
                
        except requests.exceptions.ConnectionError:
            st.error("Impossible de se connecter à l'API. Vérifiez qu'elle est bien lancée.")
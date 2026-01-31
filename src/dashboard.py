import streamlit as st
import os
import sys
import requests
import json
import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Ajout du chemin racine du projet pour résoudre les ModuleNotFoundError
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config.config import config

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

# Initialisation de l'état de session pour persister les résultats
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "client_data" not in st.session_state:
    st.session_state.client_data = None

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
            current_client_data = {
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
            st.session_state.client_data = current_client_data
            
            # Appel API immédiat et stockage en session
            try:
                with st.spinner("Analyse en cours..."):
                    response = requests.post(api_url_input, json=current_client_data)
                    
                    if response.status_code == 200:
                        st.session_state.analysis_result = response.json()
                    else:
                        st.error(f"Erreur API : {response.status_code}")
                        st.text(response.text)
                        st.session_state.analysis_result = None
            except requests.exceptions.ConnectionError:
                st.error("Impossible de se connecter à l'API. Vérifiez qu'elle est bien lancée.")
                st.session_state.analysis_result = None

# --- Logique de prédiction ---
with right_col:
    # On vérifie si un résultat est stocké en session (persistance)
    if st.session_state.analysis_result is not None:
        result = st.session_state.analysis_result
        client_data = st.session_state.client_data
        
        st.header("Résultats de l'analyse")
        st.success("Analyse disponible")
        
        # Affichage du résultat
        col_res1, col_res2, col_res3 = st.columns(3)
        
        decision_label = result["decision"]
        with col_res1:
            st.markdown("**Décision Recommandée**")
            if decision_label == "REFUSÉ":
                st.error(decision_label)
            else:
                st.success(decision_label)

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
            st.markdown("### Explication textuelle du graphique")
            st.markdown(f"""
            Le graphique ci-dessus (Waterfall Plot) décompose le score de risque :
            *   **Valeur de base (moyenne)** : {shap_exp.base_values:.3f}
            *   Les barres **rouges** indiquent les facteurs qui **augmentent** le risque.
            *   Les barres **bleues** indiquent les facteurs qui **diminuent** le risque.
            - Valeur du client : {f_x:.3f}
            """)
            
        # --- Comparaison avec la population (Critère CE2) ---
        @st.dialog("Informations & Analyse")
        def show_details_dialog(client_data):
            st.subheader("Position du client par rapport à la population")
            
            @st.cache_data
            def load_reference_data():
                """Charge un échantillon des données pour la comparaison"""
                # 1. Priorité au fichier allégé dédié au dashboard
                path_sample = os.path.join(config.DATA_DIR, "sample_dashboard.csv")
                if os.path.exists(path_sample):
                    return pd.read_csv(path_sample)
                
                # 2. Sinon, on essaie le fichier complet (plus lent)
                path_full = os.path.join(config.DATA_DIR, "application_train.csv")
                if os.path.exists(path_full):
                    return pd.read_csv(path_full).sample(1000, random_state=42)
                
                return None

            df_ref = load_reference_data()

            if df_ref is not None:
                # Création de la variable dérivée pour la comparaison si elle n'existe pas
                if "NAME_INCOME_TYPE" in df_ref.columns and "NAME_INCOME_TYPE_Working" not in df_ref.columns:
                     df_ref["NAME_INCOME_TYPE_Working"] = (df_ref["NAME_INCOME_TYPE"] == "Working").astype(int)

                # Dictionnaire pour mapper les labels lisibles aux colonnes du DataFrame
                features_dict = {
                    "Âge (jours)": "DAYS_BIRTH",
                    "Ancienneté emploi (jours)": "DAYS_EMPLOYED",
                    "Source Externe 1": "EXT_SOURCE_1",
                    "Source Externe 2": "EXT_SOURCE_2",
                    "Source Externe 3": "EXT_SOURCE_3",
                    "Note Région": "REGION_RATING_CLIENT_W_CITY",
                    "Publication ID (jours)": "DAYS_ID_PUBLISH",
                    "Dernier changement téléphone (jours)": "DAYS_LAST_PHONE_CHANGE",
                    "Type de revenu (Working)": "NAME_INCOME_TYPE_Working"
                }
                
                # Sélecteur de variable
                feature_label = st.selectbox("Choisir une variable à comparer :", list(features_dict.keys()))
                feature_col = features_dict[feature_label]
                
                if feature_col in df_ref.columns:
                    client_val = client_data[feature_col]
                    
                    fig, ax = plt.subplots(figsize=(8, 4))
                    
                    # Histogramme de la population (en gris)
                    vals = df_ref[feature_col].dropna()
                    ax.hist(vals, bins=30, color="#e0e0e0", edgecolor="white", density=True, label="Population")
                    
                    # Ligne verticale pour le client (en bleu)
                    ax.axvline(client_val, color="#0056b3", linestyle="--", linewidth=2, label="Client actuel")
                    
                    ax.set_title(f"Positionnement du client : {feature_label}")
                    ax.legend()
                    st.pyplot(fig)
                    st.caption(f"Le trait bleu indique la valeur du client ({client_val}) par rapport à la distribution des autres clients.")
                
                # --- Distribution par classe (Boxplot) ---
                st.markdown("---")
                st.subheader("Position du client dans la distribution par classe")
                
                fig_box, ax_box = plt.subplots(figsize=(8, 3))
                
                data_0 = df_ref[df_ref["TARGET"] == 0][feature_col].dropna()
                data_1 = df_ref[df_ref["TARGET"] == 1][feature_col].dropna()
                
                # Création du boxplot horizontal
                bplot = ax_box.boxplot([data_0, data_1], vert=False, patch_artist=True, labels=["Remboursé (0)", "Défaut (1)"])
                
                # Couleurs (Bleu pour 0, Rouge pour 1)
                colors = ['#5bc0de', '#d9534f']
                for patch, color in zip(bplot['boxes'], colors):
                    patch.set_facecolor(color)
                
                # Ligne du client
                ax_box.axvline(client_val, color="black", linestyle="--", linewidth=2, label="Client actuel")
                
                ax_box.set_title(f"Distribution : {feature_label} (Sains vs Défaut)")
                ax_box.legend()
                st.pyplot(fig_box)
                st.caption("Ce graphique compare la distribution de la variable pour les clients ayant remboursé (bleu) et ceux en défaut (rouge).")

                # --- Analyse Bi-variée (Critère demandé) ---
                st.markdown("---")
                st.subheader("Analyse Bi-variée")
                
                if "TARGET" in df_ref.columns:
                    # Filtre pour ne garder que les variables continues pertinentes pour un scatter plot
                    # On exclut les variables binaires (ex: Type de revenu) qui n'ont pas de sens ici
                    bivariate_features = {k: v for k, v in features_dict.items() if v != "NAME_INCOME_TYPE_Working"}

                    col_x, col_y = st.columns(2)
                    with col_x:
                        feat_x = st.selectbox("Feature X", options=list(bivariate_features.keys()), index=0)
                    with col_y:
                        feat_y = st.selectbox("Feature Y", options=list(bivariate_features.keys()), index=1)
                    
                    fx = bivariate_features[feat_x]
                    fy = bivariate_features[feat_y]
                    
                    if fx in df_ref.columns and fy in df_ref.columns:
                        fig_bi, ax_bi = plt.subplots(figsize=(8, 5))
                        
                        # Scatter plot avec couleur selon la classe (0=Bleu, 1=Rouge)
                        # On gère les NaN pour l'affichage
                        df_plot = df_ref.dropna(subset=[fx, fy, "TARGET"])
                        colors = df_plot["TARGET"].map({0: "#0056b3", 1: "#d9534f"})
                        
                        ax_bi.scatter(df_plot[fx], df_plot[fy], c=colors, alpha=0.5, s=20, label="Population")
                        
                        # Point du client actuel
                        ax_bi.scatter(client_data[fx], client_data[fy], c="red", s=150, marker="X", label="Client actuel")
                        
                        ax_bi.set_xlabel(feat_x)
                        ax_bi.set_ylabel(feat_y)
                        ax_bi.set_title(f"Croisement : {feat_x} vs {feat_y}")
                        st.pyplot(fig_bi)
                        st.caption("Les points rouges représentent les clients en défaut, les bleus ceux qui ont remboursé.")

            else:
                st.info("Les données de référence (application_train.csv) ne sont pas disponibles pour la comparaison.")

        # Injection de CSS pour le bouton vert clignotant
        st.markdown(
            """
            <style>
            div.stButton > button {
                background-color: #28a745 !important;
                color: white !important;
                animation: blinking 1.5s infinite;
            }
            @keyframes blinking {
                0% { opacity: 1; }
                50% { opacity: 0.5; }
                100% { opacity: 1; }
            }
            </style>
            """, unsafe_allow_html=True
        )
        if st.button("Statistiques & Analyse Client"):
            show_details_dialog(client_data)
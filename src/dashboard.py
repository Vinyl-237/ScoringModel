import streamlit as st
import os
import sys
import requests
import json
import shap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib

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
    st.markdown("<h2 style='text-align: center;'>Informations du Client</h2>", unsafe_allow_html=True)

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
        region_rating = st.selectbox(
            "Note région client", 
            options=[1, 2, 3],
            index=1, # L'index 1 correspond à la valeur 2 (le défaut)
            help="Note de 1 à 3 évaluant la région du client."
        )

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
        
        st.markdown("<h2 style='text-align: center;'>Résultats de l'analyse</h2>", unsafe_allow_html=True)
        #st.success("Analyse disponible")
        
        # Affichage du résultat
        decision_label = result["decision"]
        prob_default = result['probability_default']
        threshold = result['threshold_used']

        # Configuration des ticks de la jauge pour afficher le seuil
        tick_vals = [0, 0.5, 1]
        if threshold not in tick_vals:
            tick_vals.append(threshold)
        tick_vals.sort()
        
        tick_text = [f"{t:.1f}" for t in tick_vals]
        # Remplacement du texte pour le seuil
        tick_text[tick_vals.index(threshold)] = f"Seuil<br>{threshold:.3f}"

        # Jauge de Score (Plotly)
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = prob_default,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Probabilité de Défaut", 'font': {'size': 18}},
            delta = {'reference': threshold, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
            gauge = {
                'axis': {'range': [None, 1], 'tickwidth': 1, 'tickmode': 'array', 'tickvals': tick_vals, 'ticktext': tick_text},
                'bar': {'color': "darkblue"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, threshold], 'color': "rgba(0, 255, 0, 0.3)"},
                    {'range': [threshold, 1], 'color': "rgba(255, 0, 0, 0.3)"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': threshold
                }
            }
        ))
        # Réduction de la taille (hauteur) et des marges
        fig_gauge.update_layout(height=215, margin=dict(l=20, r=20, t=80, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Indicateurs Clés en dessous
        #st.markdown("### Indicateurs Clés")
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Risque de Défaut", f"{prob_default:.1%}", delta_color="inverse")
        col_m2.metric("Chance de Remboursement", f"{1-prob_default:.1%}")
        col_m3.metric("Seuil de Décision", f"{threshold:.3f}", help="Au-dessus de ce seuil, le crédit est refusé.")

        if decision_label == "REFUSÉ":
            st.error(f"DÉCISION RECOMMANDÉE : {decision_label}")
        else:
            st.success(f"DÉCISION RECOMMANDÉE : {decision_label}")

        # --- Interprétabilité (SHAP) ---
        st.markdown("---")
        st.subheader("Interprétabilité du Modèle")
        
        # Mapping des noms de variables pour l'affichage (Cohérence avec le formulaire)
        feature_mapping = {
            "DAYS_BIRTH": "Âge",
            "DAYS_EMPLOYED": "Ancienneté Emploi",
            "bureau_DAYS_CREDIT_UPDATE_mean": "Dernière MAJ Crédit",
            "REGION_RATING_CLIENT_W_CITY": "Note Région",
            "NAME_INCOME_TYPE_Working": "Type Revenu (Salarié)",
            "DAYS_LAST_PHONE_CHANGE": "Changement Téléphone",
            "DAYS_ID_PUBLISH": "Publication ID",
            "EXT_SOURCE_1": "Source Externe 1",
            "EXT_SOURCE_2": "Source Externe 2",
            "EXT_SOURCE_3": "Source Externe 3"
        }

        tab_local, tab_global = st.tabs(["Importance Locale (Client)", "Importance Globale (Modèle)"])
        
        with tab_local:
            st.markdown("##### Analyse des facteurs d'influence (Client)")
            if "shap_values" in result:
                # Application du mapping sur les noms de features
                feature_names_display = [feature_mapping.get(col, col) for col in result["feature_names"]]
                
                # Reconstruction de l'objet Explanation pour SHAP
                shap_exp = shap.Explanation(
                    values=np.array(result["shap_values"]),
                    base_values=result["base_value"],
                    data=np.array([client_data[col] for col in result["feature_names"]]),
                    feature_names=feature_names_display
                )
                
                # Affichage du Waterfall Plot
                fig, ax = plt.subplots(figsize=(8, 6))
                shap.plots.waterfall(shap_exp, max_display=10, show=False) 
                
                # Ajout de la légende personnalisée
                red_patch = mpatches.Patch(color='#ff0051', label='Augmente le risque')
                blue_patch = mpatches.Patch(color='#008bfb', label='Diminue le risque')
                ax.legend(handles=[red_patch, blue_patch], loc='lower right')
                
                st.pyplot(fig)
                st.caption("Graphique en cascade (Waterfall) montrant comment chaque variable contribue positivement (bleu) ou négativement (rouge) au score du client.")
                
                # Explication textuelle
                f_x = shap_exp.base_values + np.sum(shap_exp.values)
                st.markdown(f"**Score final (Log-odds) :** {f_x:.3f}")
        
        with tab_global:
            #st.markdown("##### Comparaison : Client vs Modèle Global")
            model_path = os.path.join(config.MODELS_DIR, "final_model_LightGBM.pkl")
            if os.path.exists(model_path):
                try:
                    model = joblib.load(model_path)
                    # Récupération de l'importance des features
                    importances = model.feature_importances_
                    feature_names = model.feature_name_
                    
                    # Mapping des noms pour le global
                    feature_names_mapped = [feature_mapping.get(col, col) for col in feature_names]
                    
                    # DataFrame Global
                    df_imp = pd.DataFrame({"feature": feature_names_mapped, "importance": importances})
                    
                    # DataFrame Local (Client) - Valeur absolue pour comparer l'impact
                    if "shap_values" in result:
                        local_imp = np.abs(result["shap_values"])
                        # Mapping des noms pour le local (doit correspondre au global pour la fusion)
                        feature_names_local_mapped = [feature_mapping.get(col, col) for col in result["feature_names"]]
                        df_local = pd.DataFrame({"feature": feature_names_local_mapped, "importance_locale": local_imp})
                        
                        # Fusion des deux
                        df_merge = pd.merge(df_local, df_imp, on="feature", how="inner")
                        
                        # Normalisation Min-Max pour rendre les échelles comparables visuellement
                        df_merge["importance_globale_norm"] = df_merge["importance"] / df_merge["importance"].max()
                        df_merge["importance_locale_norm"] = df_merge["importance_locale"] / df_merge["importance_locale"].max()
                        
                        # On garde les 10 features les plus impactantes pour CE client
                        df_merge = df_merge.sort_values(by="importance_locale", ascending=False).head(10)
                        
                        # Graphique comparatif INTERACTIF (Plotly) - Répond au critère CE2
                        fig_comp = go.Figure()
                        
                        fig_comp.add_trace(go.Bar(
                            x=df_merge["feature"],
                            y=df_merge["importance_locale_norm"],
                            name='Impact Client (Local)',
                            marker_color='#1f77b4'
                        ))
                        
                        fig_comp.add_trace(go.Bar(
                            x=df_merge["feature"],
                            y=df_merge["importance_globale_norm"],
                            name='Importance Globale (Modèle)',
                            marker_color='#ff7f0e',
                            opacity=0.7
                        ))

                        # Ajout des annotations de différence (fonctionnalité perdue lors du passage à Plotly)
                        for i in range(len(df_merge)):
                            feature = df_merge["feature"].iloc[i]
                            loc_val = df_merge["importance_locale_norm"].iloc[i]
                            glob_val = df_merge["importance_globale_norm"].iloc[i]
                            diff = loc_val - glob_val
                            
                            # Couleur selon le signe (Bleu si Client > Global, Orange sinon)
                            color = '#1f77b4' if diff > 0 else '#ff7f0e'
                            y_pos = max(loc_val, glob_val)
                            
                            fig_comp.add_annotation(
                                x=feature,
                                y=y_pos,
                                text=f"{diff:+.2f}",
                                showarrow=False,
                                font=dict(color=color, size=11, family="Arial, sans-serif"),
                                yshift=10 # Décalage pour placer le texte au-dessus de la barre
                            )

                        fig_comp.update_layout(
                            title="Variables du Client vs Importance Globale",
                            xaxis_title="Variables",
                            yaxis_title="Importance Normalisée (0-1)",
                            barmode='group',
                            xaxis_tickangle=-45,
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                        )
                        
                        st.plotly_chart(fig_comp, use_container_width=True)
                        st.caption("Ce graphique compare l'importance des variables pour ce client spécifique (bleu) par rapport à leur importance globale dans le modèle (orange).")
                    else:
                        st.warning("Pas de données SHAP disponibles pour la comparaison.")

                except Exception as e:
                    st.warning(f"Erreur lors du chargement du modèle : {e}")
            else:
                st.info("Le modèle n'est pas disponible localement pour afficher l'importance globale.")
            
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
                    
                    # Amélioration CE3 : Taille de police
                    ax.set_title(f"Positionnement du client : {feature_label}", fontsize=12)
                    ax.legend(fontsize=10)
                    st.pyplot(fig)
                    st.caption(f"Histogramme de distribution. Le trait bleu indique la valeur du client ({client_val}) par rapport à la population.")
                
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
                
                ax_box.set_title(f"Distribution : {feature_label} (Sains vs Défaut)", fontsize=12)
                ax_box.legend(fontsize=10)
                st.pyplot(fig_box)
                st.caption("Boîte à moustaches comparant la distribution pour les clients ayant remboursé (bleu) et ceux en défaut (rouge). Le trait noir représente le client actuel.")

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
                        
                        ax_bi.set_xlabel(feat_x, fontsize=10)
                        ax_bi.set_ylabel(feat_y, fontsize=10)
                        ax_bi.set_title(f"Croisement : {feat_x} vs {feat_y}", fontsize=12)
                        st.pyplot(fig_bi)
                        st.caption("Nuage de points croisant deux variables. Les points rouges représentent les clients en défaut, les bleus ceux qui ont remboursé et la croix rouge le client actuel.")

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
        col_b1, col_b2, col_b3 = st.columns([1, 2, 1])
        with col_b2:
            if st.button("Statistiques & Analyse Client", use_container_width=True):
                show_details_dialog(client_data)
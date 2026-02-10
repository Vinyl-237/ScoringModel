# Projet de Scoring Crédit - "Prêt à dépenser"

Ce projet vise à développer un outil de scoring crédit pour calculer la probabilité qu'un client rembourse son crédit, puis à classifier la demande en crédit accordé ou refusé.

Il inclut :
- Un modèle de Machine Learning (LightGBM) entraîné sur des données historiques.
- Une API REST (FastAPI) pour servir les prédictions.
- Un Dashboard interactif (Streamlit) pour les chargés de relation client.
- Une chaîne CI/CD pour l'intégration continue.
- **Nouveau :** Un module de veille technique comparant une approche multimodale récente (CLIP, 2021) à une baseline solide (BERT + EfficientNet).

## Structure du projet

Le projet est organisé comme suit :

*   `src/` : Code source principal.
    *   `api/` : Code de l'API (FastAPI).
    *   `dashboard.py` : Interface utilisateur (Streamlit).
    *   `preprocessing/` : Scripts de nettoyage et feature engineering.
    *   `training/` : Scripts d'entraînement et de scoring.
    *   `veille/` : POC de veille technique (Multimodal).
    *   `monitoring/` : Détection du Data Drift.
*   `tests/` : Tests unitaires (pytest).
*   `.github/workflows/` : Configuration de l'intégration continue (CI).
*   `data/` : Dossier pour les datasets (non versionné).
*   `models/` : Dossier pour les modèles sérialisés (.pkl).

## Installation

1.  Cloner le dépôt :
    ```bash
    git clone <url_du_repo>
    cd ScoringModel
    ```

2.  Installer les dépendances :
    ```bash
    pip install -r requirements.txt
    ```

## Utilisation

### 1. Lancer l'API (Backend)

L'API expose le modèle de prédiction.

```bash
uvicorn src.api.app:app --reload
```
L'API sera accessible sur `http://127.0.0.1:8000`.

### 2. Lancer le Dashboard (Frontend)

Le dashboard permet de visualiser les scores et l'interprétabilité (SHAP).

```bash
streamlit run src/dashboard.py
```

### 3. Tests et Qualité

Pour lancer les tests unitaires :
```bash
pytest tests/
```

Pour générer le rapport de Data Drift :
```bash
python -m src.monitoring.data_drift
```

## Déploiement

L'application est déployée sur le cloud :
*   **API :** Render / Azure App Service
*   **Dashboard :** Streamlit Cloud
*   **Lien public :** https://scoringmodel-mt9w7d7psalzchsqbwza69.streamlit.app


## Veille Technologique : Classification Multimodale

Ce module présente une preuve de concept (POC) sur une tâche de classification de produits (E-commerce), comparant une approche classique (Fusion Tardive) à une approche multimodale native (CLIP).

### 1. La Baseline : Fusion Tardive (BERT + EfficientNet)

Cette approche consiste à utiliser deux modèles "experts" séparés pour extraire les caractéristiques de chaque modalité, puis à fusionner ces informations.

#### A. Traitement d'Image : EfficientNet (2019)
*   **Origine :** Google Research (Mingxing Tan, Quoc V. Le).
*   **Principe Clé :** Contrairement aux CNN précédents qui augmentaient arbitrairement la profondeur (ResNet) ou la largeur, EfficientNet utilise une méthode de **Compound Scaling**. Il augmente uniformément la profondeur, la largeur et la résolution de l'image à l'aide d'un coefficient composé $\phi$.
*   **Fonctionnement :** Le modèle cherche à optimiser la précision sous contrainte de FLOPs (opérations flottantes). Il utilise un bloc de base "MBConv" (Mobile Inverted Bottleneck Convolution).

*   **Formule (Scaling) :**
     α = alpha, φ = phi, β = beta, γ = gamma $$

    $$ α × β² × γ² ≈ 2 : Les facteurs d’augmentation de la profondeur (α), de la largeur (β) et de la résolution (γ) sont choisis de façon à ce que le coût de calcul double quand on augmente le niveau du modèle. $$

    - profondeur = α^φ
    - largeur = β^φ, 
    - résolution = γ^φ $$
    
    > **Explication :** α, β et γ sont des constantes qui déterminent comment redimensionner respectivement la profondeur (nombre de couches), la largeur (nombre de canaux) et la résolution. φ est le coefficient global choisi par l'utilisateur. La contrainte $\approx 2$ garantit que les ressources de calcul (FLOPs) augmentent de manière prévisible (elles doublent à chaque incrément entier de $\phi$).
*   **Références :**
    *   [Papier Arxiv (1905.11946)](https://arxiv.org/abs/1905.11946)
    *   [Machine Learning Mastery : EfficientNet Guide](https://machinelearningmastery.com/image-recognition-with-efficientnet/)

#### B. Traitement de Texte : BERT (2018)
*   **Origine :** Google AI Language (Jacob Devlin et al.).
*   **Principe Clé :** BERT (Bidirectional Encoder Representations from Transformers) a introduit la **bidirectionnalité** profonde. Contrairement aux modèles précédents qui lisaient le texte de gauche à droite, BERT lit toute la phrase simultanément grâce au mécanisme d'Attention.
*   **Fonctionnement :** Il est pré-entraîné sur deux tâches :
    1.  **Masked Language Model (MLM) :** Masquer 15% des mots et essayer de les deviner grâce au contexte.
    2.  **Next Sentence Prediction (NSP) :** Prédire si la phrase B suit logiquement la phrase A.
*   **Formule (Attention) :**
    Attention(Q, K, V) = softmax( (Q × Kᵀ) / √dₖ ) × V
    > **Explication :** $Q$ (Query), $K$ (Key) et $V$ (Value) sont des représentations vectorielles des mots. Le produit $QK^T$ mesure la similarité entre les mots (qui regarde qui ?). La division par $\sqrt{d_k}$ stabilise les gradients lors de l'entraînement. Le softmax transforme ces scores en probabilités, permettant au modèle de pondérer l'importance (l'attention) de chaque mot $V$ par rapport aux autres pour construire le sens de la phrase.
*   **Références :**
    *   [Papier Arxiv (1810.04805)](https://arxiv.org/abs/1810.04805)
    *   [Machine Learning Mastery : A Gentle Introduction to BERT](https://machinelearningmastery.com/a-brief-introduction-to-bert/)

### 2. Le Challenger : CLIP (2021)

#### CLIP (Contrastive Language-Image Pre-training)
*   **Origine :** OpenAI (Alec Radford et al.).
*   **Principe Clé :** CLIP ne cherche pas à classifier des images directement (comme EfficientNet sur ImageNet). Il apprend à **associer** une image à sa description textuelle dans un espace vectoriel commun.
*   **Fonctionnement :**
    *   Il utilise deux encodeurs (un pour l'image, un pour le texte).
    *   Il est entraîné sur 400 millions de paires (image, texte) collectées sur internet.
    *   L'objectif est l'**apprentissage contrastif** : pour un lot de $N$ paires, le modèle doit maximiser la similarité cosinus des $N$ bonnes paires (diagonale) et minimiser celle des $N^2 - N$ mauvaises paires.
*   **Formule (Loss Contrastive simplifiée) :**
    Pour une image $I$ et un texte $T$, on cherche à maximiser : sim(I, T) = ( I · T ) / ( ||I|| × ||T|| )
    > **Explication :** C'est la formule de la **similarité cosinus**. Elle mesure l'angle entre le vecteur image $I$ et le vecteur texte $T$. Si les vecteurs pointent dans la même direction (angle nul), la valeur est 1 (forte similarité sémantique). S'ils sont orthogonaux, c'est 0. La division par les normes $\|I\|$ et $\|T\|$ rend la mesure indépendante de la "longueur" ou magnitude des vecteurs, se concentrant uniquement sur leur orientation (le sens).
*   **Références :**
    *   [Papier Arxiv (2103.00020)](https://arxiv.org/abs/2103.00020)
    *   [Papers With Code : CLIP Explained](https://paperswithcode.com/method/clip)

### 3. Conclusion et Comparaison

| Caractéristique | Baseline (BERT + EfficientNet) | Challenger (CLIP) |
| :--- | :--- | :--- |
| **Architecture** | Deux modèles disjoints + classifieur final | Deux encodeurs entraînés conjointement |
| **Espace Latent** | Espaces séparés (texte vs image) | Espace multimodal unique |
| **Entraînement** | Nécessite des labels précis (Supervisé) | Apprend sur des paires brutes (Auto-supervisé) |
| **Flexibilité** | Rigide (ré-entraînement pour nouvelles classes) | **Zero-Shot** (peut classifier sans entraînement spécifique) |

**Pourquoi CLIP est une rupture ?**
Alors que l'approche Baseline nécessite d'entraîner un classifieur spécifique pour dire "Ceci est une chaussure" (en lui montrant des milliers de chaussures labellisées), CLIP comprend *sémantiquement* le concept de chaussure. Il peut ainsi classifier des objets qu'il n'a jamais vus explicitement durant un fine-tuning, simplement en comparant l'image aux descriptions textuelles des catégories ("Une photo de chaussure", "Une photo de chemise").

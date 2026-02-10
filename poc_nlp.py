import os
import sys
import pandas as pd
import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, adjusted_rand_score

# Bibliothèques Deep Learning
import timm
from transformers import AutoTokenizer, AutoModel, CLIPProcessor, CLIPModel
from torchvision import transforms

# Ajout du chemin racine pour pouvoir importer `config`
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from src.config.config import config

def run_poc_multimodal():
    print("Démarrage du POC : Classification Multimodale (CLIP vs Baseline)")
    print("="*60)

    # Configuration Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Utilisation du device : {device}")

    # --- CONFIGURATION ---
    CSV_PATH = os.path.join(config.DATA_DIR, "produits_clean.csv")
    IMG_DIR = os.path.join(config.DATA_DIR, "données_veille_tech")

    # Chargement des données
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        # Filtrage des colonnes utiles
        colonnes_gardees = ['uniq_id', 'image', 'description', 'category_main']
        if set(colonnes_gardees).issubset(df.columns):
            df = df[colonnes_gardees]
        print(f"Dataset chargé : {df.shape[0]} produits")
    else:
        print(f"ERREUR : Fichier CSV '{CSV_PATH}' introuvable.")
        return

    # 1. EfficientNet (Image) 
    print("Chargement EfficientNet (Baseline Image)...")
    effnet = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)
    effnet.to(device)
    effnet.eval()

    effnet_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 2. BERT (Texte) 
    print("Chargement BERT (Baseline Texte)...")
    bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    bert_model = AutoModel.from_pretrained('bert-base-uncased')
    bert_model.to(device)
    bert_model.eval()

    # 3. CLIP (Image + Texte)
    print("Chargement CLIP (Challenger)...")
    clip_model_name = "openai/clip-vit-base-patch32"
    clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)
    clip_processor = CLIPProcessor.from_pretrained(clip_model_name)

    def extract_features(df, img_dir, max_samples=None):
        features_baseline = []
        features_clip = []
        labels = []

        if max_samples:
            df = df.head(max_samples)

        print("Extraction des features en cours...")
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            # Adaptation aux colonnes spécifiques
            img_name = row['image']
            text = row['description']
            label = row['category_main']

            if pd.isna(img_name):
                continue
                
            img_path = os.path.join(img_dir, img_name)
            if not os.path.exists(img_path):
                continue

            try:
                image = Image.open(img_path).convert("RGB")

                # --- BASELINE --- #
                # 1. Image (EfficientNet)
                img_tensor = effnet_transforms(image).unsqueeze(0).to(device)
                with torch.no_grad():
                    eff_emb = effnet(img_tensor).cpu().numpy().flatten()

                # 2. Texte (BERT)
                inputs = bert_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
                with torch.no_grad():
                    bert_out = bert_model(**inputs)
                    bert_emb = bert_out.last_hidden_state[:, 0, :].cpu().numpy().flatten()

                baseline_vec = np.concatenate([eff_emb, bert_emb])
                features_baseline.append(baseline_vec)

                # --- CHALLENGER (CLIP) --- #
                inputs_clip = clip_processor(text=[text], images=image, return_tensors="pt", padding=True, truncation=True).to(device)
                with torch.no_grad():
                    outputs_clip = clip_model(**inputs_clip)
                    clip_img_emb = outputs_clip.image_embeds.cpu().numpy().flatten()
                    clip_txt_emb = outputs_clip.text_embeds.cpu().numpy().flatten()
                
                clip_vec = np.concatenate([clip_img_emb, clip_txt_emb])
                features_clip.append(clip_vec)

                labels.append(label)

            except Exception as e:
                print(f"Erreur sur {img_name}: {e}")
                continue

        return np.array(features_baseline), np.array(features_clip), np.array(labels)

    # Extraction
    X_baseline, X_clip, y = extract_features(df, IMG_DIR, max_samples=None)
    
    if len(y) == 0:
        print("Aucune donnée extraite. Vérifiez les chemins d'images.")
        return

    print(f"Features Baseline shape: {X_baseline.shape}")
    print(f"Features CLIP shape: {X_clip.shape}")

    # Split Train/Test
    X_base_train, X_base_test, y_train, y_test = train_test_split(X_baseline, y, test_size=0.2, random_state=42)
    X_clip_train, X_clip_test, _, _ = train_test_split(X_clip, y, test_size=0.2, random_state=42)

    # Entraînement Baseline
    print("Entraînement Baseline (Logistic Regression)...")
    clf_baseline = LogisticRegression(max_iter=1000)
    clf_baseline.fit(X_base_train, y_train)
    y_pred_base = clf_baseline.predict(X_base_test)

    # Entraînement CLIP
    print("Entraînement CLIP (Logistic Regression)...")
    clf_clip = LogisticRegression(max_iter=1000)
    clf_clip.fit(X_clip_train, y_train)
    y_pred_clip = clf_clip.predict(X_clip_test)

    # Résultats
    print("\n --- RÉSULTATS BASELINE (EfficientNet + BERT) ---")
    print(classification_report(y_test, y_pred_base))
    acc_base = accuracy_score(y_test, y_pred_base)
    ari_base = adjusted_rand_score(y_test, y_pred_base)
    print(f"ARI Baseline : {ari_base:.4f}")

    print("\n --- RÉSULTATS CHALLENGER (CLIP) ---")
    print(classification_report(y_test, y_pred_clip))
    acc_clip = accuracy_score(y_test, y_pred_clip)
    ari_clip = adjusted_rand_score(y_test, y_pred_clip)
    print(f"ARI CLIP : {ari_clip:.4f}")

    print(f"\nGain de performance (Accuracy) : {(acc_clip - acc_base) * 100:.2f} points")
    print(f"Gain de performance (ARI)      : {(ari_clip - ari_base):.4f}")

if __name__ == "__main__":
    run_poc_multimodal()
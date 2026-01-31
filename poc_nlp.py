import time
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# Essai d'importation des librairies modernes (à installer via pip install setfit datasets)
try:
    from setfit import SetFitModel, SetFitTrainer
    from datasets import load_dataset
    SETFIT_AVAILABLE = True
except ImportError:
    SETFIT_AVAILABLE = False
    print("⚠️ La librairie 'setfit' n'est pas installée. La partie 'Moderne' ne sera pas exécutée.")
    print("👉 Installez-la avec : pip install setfit datasets")

def run_poc_nlp():
    print("🔬 Démarrage du POC : Classification d'intentions bancaires (NLP)")
    print("="*60)

    # 1. Préparation des données (Dataset Banking77 - subset)
    # Si datasets n'est pas dispo, on utilise un petit jeu de données dummy pour la démo
    if SETFIT_AVAILABLE:
        print("📥 Chargement du dataset 'Banking77' via Hugging Face...")
        dataset = load_dataset("banking77")
        # On prend un sous-échantillon pour aller vite (Few-Shot simulation)
        # 10 exemples par classe pour le train, 50 pour le test
        train_ds = dataset["train"].shuffle(seed=42).select(range(200)) 
        test_ds = dataset["test"].shuffle(seed=42).select(range(100))
        
        X_train = train_ds["text"]
        y_train = train_ds["label"]
        X_test = test_ds["text"]
        y_test = test_ds["label"]
        
        # Récupération des noms de labels pour l'affichage
        label_names = dataset["train"].features["label"].names
    else:
        print("⚠️ Utilisation de données synthétiques (Mode dégradé)...")
        data = [
            ("I lost my card", 0), ("My card is stolen", 0), ("Where is my card?", 0),
            ("I want to open an account", 1), ("New account creation", 1), ("How to join?", 1),
            ("What is the interest rate?", 2), ("Tell me about rates", 2), ("Loan rates", 2)
        ] * 10
        df = pd.DataFrame(data, columns=["text", "label"])
        X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.3, random_state=42)
        label_names = ["Card", "Account", "Rate"]

    print(f"📊 Données : {len(X_train)} exemples d'entraînement, {len(X_test)} de test.")
    
    results = {}

    # --- APPROCHE 1 : CLASSIQUE (Baseline) ---
    print("\n🏛️  Approche Classique : TF-IDF + Régression Logistique")
    start_time = time.time()
    
    model_classic = make_pipeline(
        TfidfVectorizer(),
        LogisticRegression(max_iter=1000)
    )
    model_classic.fit(X_train, y_train)
    preds_classic = model_classic.predict(X_test)
    
    time_classic = time.time() - start_time
    acc_classic = accuracy_score(y_test, preds_classic)
    
    results["Classic"] = {"Accuracy": acc_classic, "Time": time_classic}
    print(f"✅ Accuracy Classique : {acc_classic:.4f} (Temps: {time_classic:.2f}s)")

    # --- APPROCHE 2 : MODERNE (SetFit - 2022) ---
    if SETFIT_AVAILABLE:
        print("\n🚀 Approche Moderne : SetFit (Transformer Fine-tuning)")
        print("ℹ️  Papier : 'Efficient Few-Shot Learning Without Prompts' (2022)")
        
        start_time = time.time()
        
        # Chargement d'un petit modèle Sentence Transformer (rapide)
        model_setfit = SetFitModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        
        # Entraînement
        trainer = SetFitTrainer(
            model=model_setfit,
            train_dataset=train_ds,
            eval_dataset=test_ds,
            loss_class=None, # Utilise la CosineSimilarityLoss par défaut
            metric="accuracy",
            batch_size=16,
            num_iterations=20, # Nombre de paires générées pour le contraste
            num_epochs=1
        )
        
        trainer.train()
        metrics = trainer.evaluate()
        
        time_setfit = time.time() - start_time
        acc_setfit = metrics["accuracy"]
        
        results["SetFit"] = {"Accuracy": acc_setfit, "Time": time_setfit}
        print(f"✅ Accuracy SetFit : {acc_setfit:.4f} (Temps: {time_setfit:.2f}s)")
    
    # --- SYNTHÈSE ---
    print("\n🏆 RÉSULTATS COMPARATIFS")
    print("-" * 30)
    print(f"{'Modèle':<15} | {'Accuracy':<10} | {'Temps (s)':<10}")
    print("-" * 30)
    for name, res in results.items():
        print(f"{name:<15} | {res['Accuracy']:.4f}     | {res['Time']:.2f}")
    print("-" * 30)
    
    if SETFIT_AVAILABLE and results["SetFit"]["Accuracy"] > results["Classic"]["Accuracy"]:
        print("🎉 Conclusion : L'approche moderne (SetFit) surpasse l'approche classique !")

if __name__ == "__main__":
    run_poc_nlp()
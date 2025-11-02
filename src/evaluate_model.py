import os
import pickle
import json
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from pathlib import Path
import numpy as np

def evaluate_model(data_dir, processed_dir, models_dir, metrics_dir):

    # ================================
    # 1️⃣ Chargement du modèle final
    # ================================
    model_path = os.path.join(models_dir, "final_model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Modèle final introuvable : {model_path}")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    print(f"✅ Modèle chargé depuis {model_path}")

    # ================================
    # 2️⃣ Chargement des données de test
    # ================================
    X_test_path = os.path.join(processed_dir, "X_test_normalized.csv")
    y_test_path = os.path.join(processed_dir, "y_test.csv")

    if not os.path.exists(X_test_path) or not os.path.exists(y_test_path):
        raise FileNotFoundError("❌ X_test_normalized.csv ou y_test.csv introuvable dans data/processed")

    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path).squeeze()  # vecteur 1D

    # Garder uniquement les colonnes numériques
    X_test = X_test.select_dtypes(include=['number'])

    # ================================
    # 3️⃣ Prédictions
    # ================================
    y_pred = model.predict(X_test)

    # ================================
    # 4️⃣ Calcul des métriques
    # ================================
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    scores = {
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    }

    print(f"✅ Évaluation terminée : {scores}")

    # ================================
    # 5️⃣ Sauvegarde des prédictions
    # ================================
    os.makedirs(data_dir, exist_ok=True)
    predictions_df = pd.DataFrame({
        "y_true": y_test,
        "y_pred": y_pred
    })
    predictions_path = os.path.join(data_dir, "predictions.csv")
    predictions_df.to_csv(predictions_path, index=False)
    print(f"✅ Prédictions sauvegardées dans {predictions_path}")

    # ================================
    # 6️⃣ Sauvegarde des métriques
    # ================================
    os.makedirs(metrics_dir, exist_ok=True)
    scores_path = os.path.join(metrics_dir, "scores.json")
    with open(scores_path, "w") as f:
        json.dump(scores, f, indent=4)

    print(f"✅ Métriques sauvegardées dans {scores_path}")


# ================================
# 7️⃣ Exécution directe
# ================================
if __name__ == "__main__":
    BASE_DIR = Path(__file__).parent.resolve()
    data_dir = str(BASE_DIR / "../models/data")
    processed_dir = str(BASE_DIR / "../data/processed")
    models_dir = str(BASE_DIR / "../models")
    metrics_dir = str(BASE_DIR / "../metrics")

    evaluate_model(data_dir, processed_dir, models_dir, metrics_dir)

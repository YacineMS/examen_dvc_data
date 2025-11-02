import os
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from pathlib import Path

def train_final_model(data_dir, models_dir, best_params=None, save_name="final_model.pkl"):
   
    x_train_path = os.path.join(data_dir, "X_train_normalized.csv")
    y_train_path = os.path.join(data_dir, "y_train.csv")

    if not os.path.exists(x_train_path) or not os.path.exists(y_train_path):
        raise FileNotFoundError(f"Les fichiers {x_train_path} ou {y_train_path} sont introuvables.")

    X_train = pd.read_csv(x_train_path)
    y_train = pd.read_csv(y_train_path).squeeze()  # vecteur 1D
    X_train = X_train.select_dtypes(include=['number'])

    # ================================
    # 3️⃣ Définition du modèle avec les meilleurs paramètres
    # ================================
    if best_params:
        model = RandomForestRegressor(random_state=42, **best_params)
        print(f"✅ Modèle RandomForest avec les paramètres : {best_params}")
    else:
        model = RandomForestRegressor(random_state=42)
        print("⚠️ Aucun paramètre fourni. Modèle entraîné avec les valeurs par défaut.")

    # ================================
    # 4️⃣ Entraînement du modèle
    # ================================
    model.fit(X_train, y_train)
    print("✅ Modèle entraîné avec succès.")

    # ================================
    # 5️⃣ Sauvegarde du modèle entraîné
    # ================================
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, save_name)

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    print(f"✅ Modèle final sauvegardé dans : {model_path}")


# ================================
# 6️⃣ Exécution directe
# ================================
if __name__ == "__main__":
    BASE_DIR = Path(__file__).parent.resolve()
    data_dir = str(BASE_DIR / "../data/processed")
    models_dir = str(BASE_DIR / "../models")

    # Exemple : charger les meilleurs paramètres depuis un fichier pickle
    best_model_path = os.path.join(models_dir, "best_model.pkl")
    if os.path.exists(best_model_path):
        with open(best_model_path, "rb") as f:
            best_model = pickle.load(f)
        best_params = best_model.get_params()
        best_params.pop("random_state", None)
    else:
        best_params = None

    train_final_model(data_dir, models_dir, best_params=best_params, save_name="final_model.pkl")

import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pathlib import Path

def normalize_data(processed_dir):
    """
    Normalise uniquement les colonnes numériques de X_train et X_test,
    après vérification de l'existence des fichiers.
    """

    x_train_path = os.path.join(processed_dir, "X_train.csv")
    x_test_path = os.path.join(processed_dir, "X_test.csv")

    # ================================
    # 1️⃣ Vérification de l'existence des fichiers
    # ================================
    if not os.path.exists(x_train_path) or not os.path.exists(x_test_path):
        raise FileNotFoundError(
            f"❌ Impossible de normaliser : les fichiers X_train.csv ou X_test.csv sont introuvables dans {processed_dir}"
        )

    print(f"✅ Fichiers trouvés :\n - {x_train_path}\n - {x_test_path}")

    # ================================
    # 2️⃣ Chargement des datasets
    # ================================
    X_train = pd.read_csv(x_train_path)
    X_test = pd.read_csv(x_test_path)

    # ================================
    # 3️⃣ Sélection uniquement des colonnes numériques
    # ================================
    numeric_cols = X_train.select_dtypes(include=["number"]).columns
    print(f"🔍 Colonnes numériques détectées : {list(numeric_cols)}")

    # ================================
    # 4️⃣ Normalisation avec StandardScaler
    # ================================
    scaler = StandardScaler()

    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])

    # ================================
    # 5️⃣ Sauvegarde des données normalisées
    # ================================
    X_train_scaled.to_csv(os.path.join(processed_dir, "X_train_normalized.csv"), index=False)
    X_test_scaled.to_csv(os.path.join(processed_dir, "X_test_normalized.csv"), index=False)

    print(f"✅ Données normalisées sauvegardées dans {processed_dir}")
    print("→ X_train_normalized.csv et X_test_normalized.csv")

# ================================
# 6️⃣ Exécution directe
# ================================
if __name__ == "__main__":
    BASE_DIR = Path(__file__).parent.resolve()
    processed_dir = str(BASE_DIR / "../data/processed")
    normalize_data(processed_dir)

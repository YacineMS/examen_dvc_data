import pandas as pd
from pathlib import Path
import os
from sklearn.model_selection import train_test_split

def preprocess_data(input_file__path, output_path):
    # Charger les données
    df = pd.read_csv(input_file__path)
    df = df.drop(columns=['date'])
    df = df.dropna()

    # Séparation features / target
    target_col = 'silica_concentrate'
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # ================================
    # 3️⃣ Split en train / test
    # ================================

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ================================
    # 4️⃣ Sauvegarde des datasets
    # ================================

    # Sauvegarde en CSV
    X_train.to_csv(os.path.join(output_path,"X_train.csv"), index=False)
    X_test.to_csv(os.path.join(output_path,"X_test.csv"), index=False)
    y_train.to_csv(os.path.join(output_path,"y_train.csv"), index=False)
    y_test.to_csv(os.path.join(output_path,"y_test.csv"), index=False)

    print("Les fichiers suivants ont été créés dans data/processed :")
    print("- X_train.csv")
    print("- X_test.csv")
    print("- y_train.csv")
    print("- y_test.csv")

BASE_DIR = Path(__file__).parent.resolve()
raw_data_dir = BASE_DIR / "../data/raw_data/raw_data.csv"
processed_dir = BASE_DIR / "../data/processed/"

# Exécuter le preprocessing
preprocess_data(raw_data_dir, processed_dir)

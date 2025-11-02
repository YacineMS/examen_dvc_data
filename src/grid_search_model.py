import os
import pickle
import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer, r2_score
from pathlib import Path

def grid_search_best_model(data_dir, models_dir):
    # ================================
    # 1️⃣ Chargement des données
    # ================================
    x_train_path = os.path.join(data_dir, "X_train_normalized.csv")
    y_train_path = os.path.join(data_dir, "y_train.csv")

    if not os.path.exists(x_train_path) or not os.path.exists(y_train_path):
        raise FileNotFoundError(
            f"❌ Les fichiers normalisés X_train_normalized.csv et y_train.csv sont introuvables dans {data_dir}"
        )

    print(f"✅ Données trouvées :\n - {x_train_path}\n - {y_train_path}")

    X_train = pd.read_csv(x_train_path)
    y_train = pd.read_csv(y_train_path).squeeze()  # pour obtenir un vecteur

    # ================================
    # 2️⃣ Définition du modèle et de la grille d’hyperparamètres
    # ================================
    model = RandomForestRegressor(random_state=42)

    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    # ================================
    # 3️⃣ Configuration du GridSearch
    # ================================
    scoring = make_scorer(r2_score)  # métrique de régression
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        verbose=2
    )

    # ================================
    # 4️⃣ Entraînement et recherche
    # ================================
    print("🚀 Démarrage du GridSearchCV...")
    grid_search.fit(X_train, y_train)
    print("✅ Recherche terminée.")

    print("\nMeilleurs paramètres trouvés :")
    print(grid_search.best_params_)
    print(f"Score R² : {grid_search.best_score_:.4f}")

    # ================================
    # 5️⃣ Sauvegarde du meilleur modèle
    # ================================
    os.makedirs(models_dir, exist_ok=True)
    best_model_path = os.path.join(models_dir, "best_model.pkl")

    with open(best_model_path, "wb") as f:
        pickle.dump(grid_search.best_estimator_, f)

    print(f"✅ Modèle enregistré dans : {best_model_path}")

# ================================
# 6️⃣ Exécution directe
# ================================
if __name__ == "__main__":
    BASE_DIR = Path(__file__).parent.resolve()
    data_dir = str(BASE_DIR / "../data/processed")
    models_dir = str(BASE_DIR / "../models")

    grid_search_best_model(data_dir, models_dir)
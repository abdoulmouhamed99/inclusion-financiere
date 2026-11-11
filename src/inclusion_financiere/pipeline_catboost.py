# src/inclusion_financiere/pipeline_catboost.py

from catboost import CatBoostClassifier

from .data import CAT_COLS

catboost = CatBoostClassifier(
    cat_features=tuple(CAT_COLS),
    verbose=0,
    random_seed=42,
    iterations=500,
    learning_rate=0.05,
    depth=6
)


if __name__ == "__main__":
    print("Modèle CatBoost créé avec succès !")
    print(catboost)
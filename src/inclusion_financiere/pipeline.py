
# src/inclusion_financiere/pipeline.py

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from .data import CAT_COLS, NUM_COLS
from .features import AddFeatures, onehot_preprocessor

# Variables catégorielles
CAT_FE = CAT_COLS + ["age_group"]

# Variables numériques
NUM_FE = NUM_COLS + [
    "education_rank",
    "is_head",
    "formal_job",
    "urban_phone",
    "age_squared"
]


# Construction du pipeline
def pipe(model):
    return make_pipeline(
        AddFeatures(),
        onehot_preprocessor(CAT_FE, NUM_FE),
        model
    )


# Modèle de régression logistique
logreg = pipe(
    LogisticRegression(max_iter=2000)
)


if __name__ == "__main__":
    print("Pipeline de régression logistique créé avec succès !")
    print(logreg)

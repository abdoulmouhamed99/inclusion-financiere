
# src/inclusion_financiere/data.py

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import RANDOM_STATE, RAW, TARGET

CAT_COLS = [
    "country",
    "location_type",
    "cellphone_access",
    "gender_of_respondent",
    "relationship_with_head",
    "marital_status",
    "education_level",
    "job_type"
]

NUM_COLS = [
    "household_size",
    "age_of_respondent"
]


def load_raw(path=RAW) -> pd.DataFrame:
    """Charge les données brutes."""
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Sépare X et y ; retire les colonnes qui ne doivent pas servir de variables."""

    # Définition de la variable cible
    y = (df[TARGET] == "Yes").astype(int)

    # Suppression de la cible et des colonnes interdites
    X = df.drop(columns=[TARGET, "uniqueid", "year"])

    return X, y


def get_train_test(test_size=0.2):
    '''Découpage train/test unique pour tout le projet (étape 4.1).'''
    X, y = clean(load_raw())
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE)


# Test direct du module
if __name__ == "__main__":

    print("Chargement des données...")
    df = load_raw()

    print("\nDimensions des données brutes :", df.shape)

    print("\nSéparation des variables X et y...")
    X, y = clean(df)

    print("\nDimensions de X :", X.shape)
    print("Dimensions de y :", y.shape)

    print("\nRépartition de la variable cible :")
    print(y.value_counts())

    print("\nVérification des colonnes supprimées :")
    print("uniqueid dans X :", "uniqueid" in X.columns)
    print("year dans X :", "year" in X.columns)
    print(TARGET, "dans X :", TARGET in X.columns)

    print("\nTest terminé avec succès !")

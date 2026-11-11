
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

EDU_ORDER = {
    "No formal education": 0,
    "Primary education": 1,
    "Secondary education": 2,
    "Vocational/Specialised training": 3,
    "Tertiary education": 4
}

FORMAL_JOBS = {
    "Formally employed Private",
    "Formally employed Government"
}


class AddFeatures(BaseEstimator, TransformerMixin):

    """Feature engineering sans apprentissage : aucune statistique tirée des données."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        X["education_rank"] = (
            X["education_level"].map(EDU_ORDER).fillna(-1)
        )

        X["is_head"] = (
            X["relationship_with_head"] == "Head of Household"
        ).astype(int)

        X["formal_job"] = (
            X["job_type"].isin(FORMAL_JOBS)
        ).astype(int)

        X["urban_phone"] = (
            (X["location_type"] == "Urban")
            & (X["cellphone_access"] == "Yes")
        ).astype(int)

        X["age_group"] = pd.cut(
            X["age_of_respondent"],
            [15, 25, 35, 50, 65, 120],
            labels=["16-25", "26-35", "36-50", "51-65", "66+"]
        ).astype(str)

        X["age_squared"] = X["age_of_respondent"] ** 2

        return X


def onehot_preprocessor(cat_cols, num_cols):

    return ColumnTransformer([
        (
            "cat",
            OneHotEncoder(
                handle_unknown="infrequent_if_exist",
                min_frequency=20
            ),
            cat_cols
        ),
        (
            "num",
            StandardScaler(),
            num_cols
        ),
    ])

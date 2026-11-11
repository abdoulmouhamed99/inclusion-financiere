from inclusion_financiere.data import clean, load_raw
from inclusion_financiere.features import AddFeatures


def test_clean_retire_les_colonnes_interdites():
    X, y = clean(load_raw())
    assert {'uniqueid', 'year', 'bank_account'}.isdisjoint(X.columns)
    assert set(y.unique()) == {0, 1}

def test_add_features_garde_toutes_les_lignes():
    X, _ = clean(load_raw())
    Xf = AddFeatures().fit_transform(X)
    assert len(Xf) == len(X)
    assert Xf['education_rank'].between(-1, 4).all()

#Bonus utile, tests/test_evaluate.py , qui n’a pas besoin des données :
import numpy as np

from inclusion_financiere.evaluate import cout_total, expected_calibration_error


def test_cout_total():
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([0, 1, 1, 0]) # 1 faux négatif, 1 faux positif
    assert cout_total(y_true, y_pred, c_fp=0.5, c_fn=5.0) == 5.5

def test_ece_probabilites_parfaites():
    y = np.array([0, 0, 1, 1])
    assert expected_calibration_error(y, y.astype(float)) == 0.0
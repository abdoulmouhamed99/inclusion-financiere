# src/inclusion_financiere/evaluate.py
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import (
    StratifiedGroupKFold,
    StratifiedKFold,
    cross_validate,
    learning_curve,
    validation_curve,
)
from sklearn.pipeline import make_pipeline

from .config import COST_FN, COST_FP, RANDOM_STATE
from .data import CAT_COLS, NUM_COLS
from .features import onehot_preprocessor
from .pipeline import logreg, pipe

# ---- 4.2 Validation croisée : UN SEUL objet cv pour tout le projet ----
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
scoring = {
    'pr_auc': 'average_precision',
    'roc_auc': 'roc_auc',
    'brier': 'neg_brier_score',
    'logloss': 'neg_log_loss',
    'acc': 'accuracy',
}

def summarize_cv(res, scoring):
    '''Résume un résultat de cross_validate en UNE ligne de tableau.'''
    summary = {}
    for metric, scorer in scoring.items():
        train_scores = res[f'train_{metric}']
        val_scores = res[f'test_{metric}']
        if scorer.startswith('neg_'): # scikit-learn renvoie les pertes en négatif
              train_scores = -train_scores
              val_scores = -val_scores
        summary[f'train_{metric}_mean'] = train_scores.mean()
        summary[f'train_{metric}_std'] = train_scores.std()
        summary[f'validation_{metric}_mean'] = val_scores.mean()
        summary[f'validation_{metric}_std'] = val_scores.std()
    summary['fit_time_mean'] = res['fit_time'].mean()
    summary['fit_time_std'] = res['fit_time'].std()
    return pd.DataFrame([summary])



def evaluate_stratified(model, X_train, y_train):
    '''4.2 : évalue un modèle avec la CV stratifiée.'''
    res = cross_validate(
        model, X_train, y_train, cv=cv, scoring=scoring, 
    
return_train_score=True
    )
    return summarize_cv(res, scoring)


# ---- 4.3 Généralisation à un pays jamais vu ----
def evaluate_country(model, X_train, y_train):
    sgkf = StratifiedGroupKFold(n_splits=4, shuffle=True,
random_state=RANDOM_STATE)
    groups = X_train['country']
    print('Répartition des pays par fold :')
    for i, (tr, va) in enumerate(sgkf.split(X_train, y_train, groups)):
        print(
            f'Fold {i + 1} :',
            groups.iloc[va].unique(),
            '| proportion de positifs :',
            y_train.iloc[va].mean().round(3),
        )
    res = cross_validate(
        model, X_train, y_train, groups=groups, cv=sgkf,
        scoring={'pr_auc': 'average_precision', 'roc_auc': 'roc_auc'},
    )
    return pd.DataFrame({
        'metrique': ['PR-AUC', 'ROC-AUC'],
        'moyenne': [res['test_pr_auc'].mean(), res['test_roc_auc'].mean()],
        'ecart_type': [res['test_pr_auc'].std(), res['test_roc_auc'].std()],
    })



# ---- 5.1 Trois modèles de référence ----
def evaluate_baselines(X_train, y_train):
    '''Compare les références sur les MÊMES plis et les mêmes métriques.'''
    models = {
        'Dummy': make_pipeline(
            onehot_preprocessor(CAT_COLS, NUM_COLS),
            DummyClassifier(strategy='most_frequent'),
        ),
        'Logistic Regression sans nouvelles variables': make_pipeline(
            onehot_preprocessor(CAT_COLS, NUM_COLS),
            LogisticRegression(max_iter=2000),
        ),
        'Logistic Regression avec nouvelles variables': logreg,
        'Random Forest': pipe(
           RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)
        ),
    }
    results = []
    for name, model in models.items():
        print(f'Évaluation : {name}')
        res = cross_validate(
            model, X_train, y_train, cv=cv, scoring=scoring, return_train_score=True
        )
        
        summary = summarize_cv(res, scoring)
        summary.insert(0, 'model', name)
        results.append(summary)
    return pd.concat(results, ignore_index=True)



# ---- 5.2 Courbes d’apprentissage et de validation ----
def compute_learning_curve(model, X_train, y_train):
    sizes, tr, va = learning_curve(
        model, X_train, y_train, cv=cv, scoring='average_precision',
        train_sizes=np.linspace(0.1, 1.0, 5),
    )
    return pd.DataFrame({
        'train_size': sizes,
        'train_mean': tr.mean(axis=1), 'train_std': tr.std(axis=1),
        'validation_mean': va.mean(axis=1), 'validation_std': va.std(axis=1),
    })



def compute_validation_curve(rf, X_train, y_train):
    leaf = [1, 2, 5, 10, 20, 50]
    tr, va = validation_curve(
        rf, X_train, y_train, cv=cv, scoring='average_precision',
        param_name='randomforestclassifier__min_samples_leaf', param_range=leaf,
    )
    results = pd.DataFrame({
        'min_samples_leaf': leaf,
        'train_mean': tr.mean(axis=1), 'train_std': tr.std(axis=1),
        'validation_mean': va.mean(axis=1), 'validation_std': va.std(axis=1),
    })
    # Règle d’une erreur-type : le plus simple (plus grande feuille) à moins    d’un SE du meilleur
    best = results['validation_mean'].idxmax()
    best_se = results.loc[best, 'validation_std'] / np.sqrt(cv.get_n_splits())
    threshold = results.loc[best, 'validation_mean'] - best_se
    eligible = results[results['validation_mean'] >= threshold]

    return results, int(eligible['min_samples_leaf'].max())




# ---- 8. Calibration ----
def expected_calibration_error(y_true, p, n_bins=10):
    y_true, p = np.asarray(y_true), np.asarray(p)

    bins = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    ece = 0.0

    for b in range(n_bins):
        m = bins == b
        if m.any():
            ece += m.mean() * abs(y_true[m].mean() - p[m].mean())

    return ece



# ---- 9. Coût métier ----
def cout_total(y_true, y_pred, c_fp=COST_FP, c_fn=COST_FN):
    _tn, fp, fn, _tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return fp * c_fp + fn * c_fn


# ---- 10. Intervalle de confiance bootstrap ----
def bootstrap_ci(y_true, p, metric, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    y_true, p = np.asarray(y_true), np.asarray(p)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        if y_true[idx].min() == y_true[idx].max():
            continue
        vals.append(metric(y_true[idx], p[idx]))
    return np.percentile(vals, [2.5, 97.5])



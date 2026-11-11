
# src/inclusion_financiere/config.py



from pathlib import Path

# Racine du projet
ROOT = Path(__file__).resolve().parents[2]

# Chemin des données brutes
RAW = ROOT / "data" / "raw" / "inclusion-financiere-en-afrique.csv"

# Variable cible à prédire
TARGET = "bank_account"

# Graine aléatoire
RANDOM_STATE = 42

# Coûts métier (USD), hypothèses pédagogiques fixées une fois pour toutes
COST_FP = 0.5 # contacter une personne sans compte
COST_FN = 5.0 # oublier une personne bancarisée

"""Shared settings for the COGS 118A classifier comparison.

Everything the scripts need to agree on lives here: datasets, sizes,
partitions, trials, seeds and per-model caps. Change values here, not
inside the scripts.
"""
from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"

# ---------------------------------------------------------------- datasets
# Locked 2026-11-16: Adult, Letter (A-M vs N-Z), Covertype (largest class vs
# rest), HIGGS. All binary.
DATASETS = ["adult", "letter", "covertype", "higgs"]

# Rows kept per dataset after cleaning. None = keep everything.
# Covertype and HIGGS capped at 100k (decision #13, 2026-11-24): one
# stratified subsample, shared by every classifier. Re-run prepare_data.py
# after changing these.
N_ROWS = {
    "adult": None,          # ~45k after dropping rows with missing values
    "letter": None,         # 20k
    "covertype": 100_000,   # stratified subsample of the full 581k
    "higgs": 100_000,       # stratified subsample of the HIGGS rows read in
}

# Upper bound on how many HIGGS rows are read from the raw file before
# subsampling (the full file is 11M rows, about 8 GB in memory).
HIGGS_RAW_SAMPLE = 2_000_000

# ---------------------------------------------------------------- protocol
# Train fraction for each partition: 20/80, 50/50, 80/20 (train/test).
PARTITIONS = {"20_80": 0.2, "50_50": 0.5, "80_20": 0.8}
N_TRIALS = 3
CV_FOLDS = 5

# One seed controls the dataset subsample; split seeds are derived from it
# so every classifier sees exactly the same train/test rows.
BASE_SEED = 118


def split_seed(partition: str, trial: int) -> int:
    """Seed for one (partition, trial) split. Same for every classifier."""
    return BASE_SEED * 1000 + list(PARTITIONS).index(partition) * 10 + trial


# ---------------------------------------------------------------- model caps
# Max training rows per model (None = no cap). Rows are a stratified random
# subset of the training split. Test set is never capped.
TRAIN_CAP = {
    "rf": None,
    "xgboost": None,
    "svm": 10_000,      # decision #14 (2026-11-24); RBF SVM scales ~n^2
    "ann": None,
    "knn": None,
    "tabpfn": 10_000,   # TabPFN v2 open weights, decided 2026-11-16
}

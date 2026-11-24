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
# Covertype and HIGGS capped at 100k on 2026-11-24 (decision #14): one
# stratified, fixed-seed subsample before any splits, so every classifier
# sees the same rows. Re-run prepare_data.py after changing these.
N_ROWS = {
    "adult": None,          # ~45k after dropping rows with missing values
    "letter": None,         # 20k
    "covertype": 100_000,   # stratified subsample of 581,012
    "higgs": 100_000,       # stratified subsample of the HIGGS_RAW_SAMPLE read
}

# Balance classes to exactly 50/50 (decision #18, 2026-11-24): randomly drop
# rows of the larger class until both classes have the same count. Done after
# cleaning and BEFORE the N_ROWS cap, so capped datasets come out 50/50 too.
# Adult shrinks most (~25% positive -> about half its rows are dropped).
BALANCE_CLASSES = True

# Upper bound on how many HIGGS rows are read from the raw file before
# subsampling (the full file is 11M rows, about 8 GB in memory).
HIGGS_RAW_SAMPLE = 2_000_000

# ---------------------------------------------------------------- protocol
# Train fraction for each partition: 20/80, 50/50, 80/20 (train/test).
PARTITIONS = {"20_80": 0.2, "50_50": 0.5, "80_20": 0.8}
N_TRIALS = 3
CV_FOLDS = 5            # fallback: 3 if the timing test says 5 is too slow

# Tuning happens once per dataset x partition x model, on this trial only
# (decision #16). Other trials refit the best settings on their own split.
# Trials are numbered 0-2 in code; 0 is "trial 1" in the docs.
TUNE_TRIAL = 0

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
    "svm": 10_000,      # decision #15 (was provisional 50k); for runtime
    "ann": None,
    "knn": None,
    "tabpfn": 10_000,   # TabPFN v2 open weights, decided 2026-11-16
}

# ---------------------------------------------------------------- speed
# Added 2026-11-24 (decision #12). Prediction speed is timed on a fixed-size
# random sample of each test split, not the whole test set, so per-row
# numbers are comparable across partitions and repeats stay cheap (TabPFN on
# an 800k-row HIGGS test set would otherwise be predicted 6 times).
TIMING_N_ROWS = 10_000   # test rows used for the prediction-speed measurement
TIMING_REPEATS = 5       # timed repeats; report the median
TIMING_WARMUP = 1        # untimed warm-up calls first (GPU start-up cost)

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

# Rows kept per dataset after cleaning (and balancing, where on).
# 2026-11-26 (decision #19): every dataset capped at 12,500 rows, one
# stratified fixed-seed subsample before any splits. The 80/20 partition then
# trains on exactly 10,000 rows, which fits TabPFN's limit, so no model needs
# its own training cap and every model sees identical rows in every cell (as
# in the paper, which trained every model on the same 4k rows). Training sizes
# are 2,500 / 6,250 / 10,000 across the three partitions.
# Re-run prepare_data.py after changing these.
N_ROWS = {
    "adult": 12_500,       # from 45,222 after dropping rows with missing values
    "letter": 12_500,      # from ~19.9k after balancing
    "covertype": 12_500,   # from 581,012
    "higgs": 12_500,       # from the HIGGS_RAW_SAMPLE read
}

# Balance classes to exactly 50/50 by randomly dropping rows of the larger
# class, after cleaning and BEFORE the N_ROWS cap (decision #18).
# 2026-11-26 (decision #21): Adult stays unbalanced (~24.8% positive, the
# paper's Adult was 25%) so one dataset keeps its natural class mix, as in
# the paper. Report % positive per dataset and the majority-class baseline.
BALANCE_CLASSES = {
    "adult": False,
    "letter": True,
    "covertype": True,
    "higgs": True,
}

# Upper bound on how many HIGGS rows are read from the raw file before
# subsampling (the full file is 11M rows, about 8 GB in memory).
HIGGS_RAW_SAMPLE = 200_000   # plenty for a 12.5k subsample

# ---------------------------------------------------------------- protocol
# Train fraction for each partition: 20/80, 50/50, 80/20 (train/test).
PARTITIONS = {"20_80": 0.2, "50_50": 0.5, "80_20": 0.8}
N_TRIALS = 3
CV_FOLDS = 5            # fallback: 3 if the timing test says 5 is too slow

# Tuning runs in EVERY trial (decision #20, 2026-11-26; replaces #16 "tune on
# trial 1 only"): each dataset x partition x trial x model gets its own
# GridSearchCV on its own training split, as in the paper (which picked
# parameters on each trial's own validation set) and the rubric ("each time,
# report the best accuracy under the chosen hyper-parameter").
# Trials are numbered 0-2 in code; 0 is "trial 1" in the docs.

# One seed controls the dataset subsample; split seeds are derived from it
# so every classifier sees exactly the same train/test rows.
BASE_SEED = 118


def split_seed(partition: str, trial: int) -> int:
    """Seed for one (partition, trial) split. Same for every classifier."""
    return BASE_SEED * 1000 + list(PARTITIONS).index(partition) * 10 + trial


# ---------------------------------------------------------------- model caps
# Max training rows per model (None = no cap). Since decision #19 the largest
# training split is 10,000 rows, so no caps are needed. Kept as a guard: the
# experiment script should fail loudly if a training split ever exceeds a cap,
# instead of silently subsampling.
TRAIN_CAP = {
    "rf": None,
    "xgboost": None,
    "svm": None,        # was 10k (decision #15), not needed at 12.5k rows
    "ann": None,
    "knn": None,
    "tabpfn": 10_000,   # TabPFN v2 hard limit; the 80/20 split is exactly 10k
}

# ---------------------------------------------------------------- speed
# Added 2026-11-24 (decision #12). Prediction speed is timed on a fixed-size
# random sample of each test split, not the whole test set, so per-row
# numbers are comparable across partitions.
TIMING_N_ROWS = 2_500    # test rows used for the prediction-speed measurement
                         # (= smallest test split, 80/20 of 12.5k)
TIMING_REPEATS = 5       # timed repeats; report the median
TIMING_WARMUP = 1        # untimed warm-up calls first (GPU start-up cost)

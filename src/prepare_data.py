"""Step 1 of the pipeline: download, clean and binarize the four datasets,
then pre-compute every train/test split.

Output, one file per dataset:  data/processed/<name>.npz
    X           float32 (n_rows, n_features)
    y           int8    (n_rows,)   0/1 labels
    feature_names
    train_idx__<partition>__t<trial> / test_idx__<partition>__t<trial>
                indices into X/y for each of the 3 partitions x 3 trials

Every classifier loads the same file, so all models are trained and
tested on exactly the same rows.

Usage:
    python src/prepare_data.py                 # all datasets
    python src/prepare_data.py --only adult letter
    python src/prepare_data.py --no-download   # use files already in data/raw
"""
from __future__ import annotations

import argparse
import gzip
import json
import shutil
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402

UCI = "https://archive.ics.uci.edu/ml/machine-learning-databases"
RAW_FILES = {
    "adult": [("adult.data", f"{UCI}/adult/adult.data"),
              ("adult.test", f"{UCI}/adult/adult.test")],
    "letter": [("letter-recognition.data",
                f"{UCI}/letter-recognition/letter-recognition.data")],
    "covertype": [("covtype.data.gz", f"{UCI}/covtype/covtype.data.gz")],
    "higgs": [("HIGGS.csv.gz", f"{UCI}/00280/HIGGS.csv.gz")],  # ~2.6 GB
}
# Dataset landing pages, for manual download if the URLs above move.
UCI_PAGES = {
    "adult": "https://archive.ics.uci.edu/dataset/2/adult",
    "letter": "https://archive.ics.uci.edu/dataset/59/letter+recognition",
    "covertype": "https://archive.ics.uci.edu/dataset/31/covertype",
    "higgs": "https://archive.ics.uci.edu/dataset/280/higgs",
}


# ---------------------------------------------------------------- download
def fetch_raw(name: str, download: bool) -> None:
    C.RAW_DIR.mkdir(parents=True, exist_ok=True)
    for fname, url in RAW_FILES[name]:
        dest = C.RAW_DIR / fname
        if dest.exists():
            continue
        if not download:
            sys.exit(f"[{name}] missing {dest}. Download it from "
                     f"{UCI_PAGES[name]} or run without --no-download.")
        print(f"[{name}] downloading {url}")
        tmp = dest.with_suffix(dest.suffix + ".part")
        try:
            with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
                shutil.copyfileobj(r, f, length=1 << 20)
        except Exception as e:  # noqa: BLE001
            tmp.unlink(missing_ok=True)
            sys.exit(f"[{name}] download failed ({e}). Get {fname} from "
                     f"{UCI_PAGES[name]} and put it in {C.RAW_DIR}.")
        tmp.rename(dest)


# ---------------------------------------------------------------- loaders
ADULT_COLS = ["age", "workclass", "fnlwgt", "education", "education-num",
              "marital-status", "occupation", "relationship", "race", "sex",
              "capital-gain", "capital-loss", "hours-per-week",
              "native-country", "income"]


def load_adult() -> tuple[pd.DataFrame, np.ndarray]:
    """Train + test files combined (48,842 rows). Rows with any '?' dropped
    (about 45,222 left). Label: income >50K = 1. Categoricals one-hot."""
    parts = []
    for fname in ("adult.data", "adult.test"):
        df = pd.read_csv(C.RAW_DIR / fname, header=None, names=ADULT_COLS,
                         skipinitialspace=True, na_values="?",
                         comment="|")  # adult.test starts with a '|1x3' line
        parts.append(df)
    df = pd.concat(parts, ignore_index=True).dropna(how="all")
    df = df.dropna()
    y = df.pop("income").str.rstrip(".").map({">50K": 1, "<=50K": 0})
    if y.isna().any():
        raise ValueError("Unexpected Adult label values")
    cat_cols = df.select_dtypes(exclude="number").columns
    X = pd.get_dummies(df, columns=list(cat_cols), dtype=np.float32)
    return X, y.to_numpy()


def load_letter() -> tuple[pd.DataFrame, np.ndarray]:
    """20,000 rows, 16 numeric features. Label: letters A-M = 0, N-Z = 1
    (the paper's balanced LETTER.p2 version)."""
    df = pd.read_csv(C.RAW_DIR / "letter-recognition.data", header=None)
    letters = df.pop(0)
    df.columns = [f"f{i}" for i in range(1, 17)]
    y = (letters >= "N").astype(int).to_numpy()
    return df, y


def load_covertype() -> tuple[pd.DataFrame, np.ndarray]:
    """581,012 rows, 54 features (10 numeric + 44 binary). Label: the most
    common cover type (class 2, Lodgepole Pine, ~49%) = 1, all others = 0.
    The largest class is found from the data, not hard-coded."""
    df = pd.read_csv(C.RAW_DIR / "covtype.data.gz", header=None)
    cover = df.pop(54)
    largest = cover.value_counts().idxmax()
    print(f"[covertype] largest class = {largest}")
    df.columns = [f"f{i}" for i in range(54)]
    return df, (cover == largest).astype(int).to_numpy()


HIGGS_COLS = (
    ["lepton_pT", "lepton_eta", "lepton_phi", "missing_energy_magnitude",
     "missing_energy_phi"]
    + [f"jet{j}_{v}" for j in range(1, 5) for v in ("pt", "eta", "phi", "btag")]
    + ["m_jj", "m_jjj", "m_lv", "m_jlv", "m_bb", "m_wbb", "m_wwbb"]
)


def load_higgs(seed: int) -> tuple[pd.DataFrame, np.ndarray]:
    """11M rows, 28 features, label in column 0 (1 = signal). Reads the
    gzip in chunks and keeps a random fraction of each chunk so the whole
    file never has to fit in memory."""
    total = 11_000_000
    keep_frac = min(1.0, C.HIGGS_RAW_SAMPLE / total)
    rng = np.random.default_rng(seed)
    kept = []
    with gzip.open(C.RAW_DIR / "HIGGS.csv.gz", "rt") as fh:
        for chunk in pd.read_csv(fh, header=None, chunksize=1_000_000,
                                 dtype=np.float32):
            mask = rng.random(len(chunk)) < keep_frac
            kept.append(chunk[mask])
            print(f"[higgs] read chunk, kept {mask.sum():,}")
    df = pd.concat(kept, ignore_index=True)
    y = df.pop(0).astype(int).to_numpy()
    df.columns = HIGGS_COLS
    return df, y


LOADERS = {
    "adult": lambda seed: load_adult(),
    "letter": lambda seed: load_letter(),
    "covertype": lambda seed: load_covertype(),
    "higgs": load_higgs,
}


# ---------------------------------------------------------------- main step
def balance(X: pd.DataFrame, y: np.ndarray, seed: int):
    """Undersample the larger class so both classes have the same count.
    Rows are drawn at random with a fixed seed; original order is kept."""
    rng = np.random.default_rng(seed)
    idx0, idx1 = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
    n = min(len(idx0), len(idx1))
    keep = np.sort(np.concatenate([rng.choice(idx0, n, replace=False),
                                   rng.choice(idx1, n, replace=False)]))
    return X.iloc[keep].reset_index(drop=True), y[keep]


def subsample(X: pd.DataFrame, y: np.ndarray, n: int | None, seed: int):
    if n is None or n >= len(y):
        return X.reset_index(drop=True), y
    idx, _ = train_test_split(np.arange(len(y)), train_size=n,
                              stratify=y, random_state=seed)
    idx = np.sort(idx)
    return X.iloc[idx].reset_index(drop=True), y[idx]


def make_splits(y: np.ndarray) -> dict[str, np.ndarray]:
    out = {}
    all_idx = np.arange(len(y))
    for part, frac in C.PARTITIONS.items():
        for t in range(C.N_TRIALS):
            tr, te = train_test_split(all_idx, train_size=frac, stratify=y,
                                      random_state=C.split_seed(part, t))
            out[f"train_idx__{part}__t{t}"] = np.sort(tr).astype(np.int32)
            out[f"test_idx__{part}__t{t}"] = np.sort(te).astype(np.int32)
    return out


def prepare(name: str, download: bool) -> dict:
    fetch_raw(name, download)
    X, y = LOADERS[name](C.BASE_SEED)
    rows_raw, pos_raw = int(len(y)), round(float(y.mean()), 4)
    if C.BALANCE_CLASSES:
        X, y = balance(X, y, C.BASE_SEED)
        print(f"[{name}] balanced: {rows_raw:,} rows ({pos_raw:.1%} positive)"
              f" -> {len(y):,} rows (50/50)")
    X, y = subsample(X, y, C.N_ROWS[name], C.BASE_SEED)
    if X.isna().any().any():
        raise ValueError(f"[{name}] NaNs left after cleaning")

    C.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        C.PROCESSED_DIR / f"{name}.npz",
        X=X.to_numpy(dtype=np.float32),
        y=y.astype(np.int8),
        feature_names=np.array(X.columns, dtype=str),
        **make_splits(y),
    )
    summary = {"dataset": name, "rows": int(len(y)),
               "features": int(X.shape[1]),
               "positive_rate": round(float(y.mean()), 4),
               "rows_before_balancing": rows_raw,
               "positive_rate_before_balancing": pos_raw}
    print(f"[{name}] saved: {summary}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--only", nargs="+", choices=C.DATASETS)
    ap.add_argument("--no-download", action="store_true")
    args = ap.parse_args()

    summaries = [prepare(n, not args.no_download)
                 for n in (args.only or C.DATASETS)]
    # Summary for docs 03 (dataset sizes) — merged with any earlier runs.
    path = C.PROCESSED_DIR / "dataset_summary.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    old.update({s["dataset"]: s for s in summaries})
    path.write_text(json.dumps(old, indent=2))
    print(f"Summary written to {path}")


if __name__ == "__main__":
    main()

"""Step 2 of the pipeline: run ONE job = dataset x model x partition x trial.

What it does (docs/01 Step 2):
  1. Load the stored train/test split for this partition and trial.
  2. 5-fold GridSearchCV on the training split (every trial, decision #20).
     Same folds for every model (fixed seed). TabPFN has one fixed setting
     but still goes through CV, so it gets a validation accuracy too.
  3. Refit the best setting on the whole training split; score train and
     test accuracy; test confusion matrix.
  4. Speed: tuning time, final fit time, prediction time on a fixed
     2,500-row sample of the test split (warm-up, then median of repeats).
  5. Write results/<dataset>_<model>_<partition>_trial<k>.json (k = 1..3).
     Written to a temp file first and renamed, so a cut-off job never
     leaves a half-written results file behind.

Usage:
    python src/run_job.py --dataset higgs --model xgboost --partition 80_20 --trial 1
    python src/run_job.py ... --smoke        # tiny grid, 2-fold CV -> results/smoke/
    python src/run_job.py ... --device cpu   # no GPU, testing only -> results/cpu_test/
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import socket
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
import models  # noqa: E402


# ---------------------------------------------------------------- helpers
def result_path(dataset: str, model: str, partition: str, trial: int,
                results_dir: Path = C.RESULTS_DIR) -> Path:
    """trial is 1-based (1..3), matching the docs."""
    return results_dir / f"{dataset}_{model}_{partition}_trial{trial}.json"


def gpu_status() -> dict:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,index,memory.free,memory.total,"
             "utilization.gpu,driver_version", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20, check=True).stdout
        name, idx, free, total, util, driver = [x.strip() for x in
                                                out.splitlines()[0].split(",")]
        return {"name": name, "index": idx, "memory_free_mib": free,
                "memory_total_mib": total, "utilization_pct": util,
                "driver": driver}
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}


def versions() -> dict:
    out = {"python": platform.python_version()}
    for mod in ["numpy", "sklearn", "cuml", "xgboost", "torch", "skorch",
                "tabpfn"]:
        try:
            out[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001
            out[mod] = None
    return out


def to_numpy(a) -> np.ndarray:
    """Make sure predictions are a NumPy array on the CPU."""
    if hasattr(a, "to_numpy"):
        a = a.to_numpy()
    if hasattr(a, "get"):   # cupy
        a = a.get()
    return np.asarray(a).ravel()


def jsonable(x):
    """Turn numpy scalars/arrays and tuples into plain JSON types."""
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def output_dir(smoke: bool, device: str) -> Path:
    """Real runs go to results/. Smoke and CPU test runs go to their own
    subfolders so they can never mix into the report's numbers."""
    if smoke:
        return C.RESULTS_DIR / "smoke"
    if device == "cpu":
        return C.RESULTS_DIR / "cpu_test"
    return C.RESULTS_DIR


def smoke_grid(grid: dict) -> dict:
    """One value per parameter — for quick pipeline checks only."""
    return {k: v[:1] for k, v in grid.items()}


def time_predictions(est, X_sample: np.ndarray) -> dict:
    for _ in range(C.TIMING_WARMUP):
        to_numpy(est.predict(X_sample))
    times = []
    for _ in range(C.TIMING_REPEATS):
        t0 = time.perf_counter()
        to_numpy(est.predict(X_sample))   # timer stops once back as NumPy
        times.append(time.perf_counter() - t0)
    med = float(np.median(times))
    n = len(X_sample)
    return {"predict_rows": n, "predict_times_s": times,
            "predict_median_s": med,
            "predict_ms_per_1k": med / n * 1000 * 1000,
            "predict_rows_per_s": n / med if med > 0 else None}


# ---------------------------------------------------------------- main job
def run(dataset: str, model: str, partition: str, trial: int, *,
        device: str = "gpu", smoke: bool = False, cv_folds: int | None = None,
        results_dir: Path = C.RESULTS_DIR, force: bool = False) -> Path:
    out_path = result_path(dataset, model, partition, trial, results_dir)
    if out_path.exists() and not force:
        print(f"[skip] {out_path.name} exists")
        return out_path

    t_job = time.perf_counter()
    started = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    gpu_before = gpu_status() if device == "gpu" else {"device": "cpu"}

    # 1. data and stored split (trial is 1-based; stored keys are 0-based)
    data = np.load(C.PROCESSED_DIR / f"{dataset}.npz")
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)
    tr = data[f"train_idx__{partition}__t{trial - 1}"]
    te = data[f"test_idx__{partition}__t{trial - 1}"]
    X_tr, y_tr, X_te, y_te = X[tr], y[tr], X[te], y[te]

    cap = C.TRAIN_CAP.get(model)
    if cap is not None and len(tr) > cap:
        raise SystemExit(f"{model}: training split has {len(tr):,} rows, over "
                         f"its cap of {cap:,}. Check N_ROWS / partitions.")

    seed = C.split_seed(partition, trial - 1)
    est, grid = models.build(model, X.shape[1], device=device, seed=seed)
    if smoke:
        grid = smoke_grid(grid)
    folds = cv_folds or (2 if smoke else C.CV_FOLDS)
    # Same folds for every model in this (dataset, partition, trial).
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

    # skorch's classifier wants int64 labels; cuML wants int32. Others: either.
    y_fit = y_tr.astype(np.int32) if (device == "gpu" and model in ("rf", "svm")) else y_tr

    # 2. grid search (every trial)
    search = GridSearchCV(est, [grid] if grid else [{}], scoring="accuracy",
                          cv=cv, refit=True, return_train_score=True,
                          n_jobs=1, error_score="raise")
    n_settings = int(np.prod([len(v) for v in grid.values()])) if grid else 1
    print(f"[{dataset}/{model}/{partition}/trial{trial}] {n_settings} settings"
          f" x {folds} folds on {len(tr):,} training rows ...", flush=True)
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        search.fit(X_tr, y_fit)
    tune_time = time.perf_counter() - t0
    best = search.best_estimator_

    # 3. accuracy (full train and test sets)
    yhat_tr = to_numpy(best.predict(X_tr))
    yhat_te = to_numpy(best.predict(X_te))
    train_acc = accuracy_score(y_tr, yhat_tr)
    test_acc = accuracy_score(y_te, yhat_te)
    cm = confusion_matrix(y_te, yhat_te, labels=[0, 1])

    # 4. prediction speed on a fixed sample of the test split
    rng = np.random.default_rng(seed)
    n_time = min(C.TIMING_N_ROWS, len(te))
    X_time = X_te[np.sort(rng.choice(len(te), n_time, replace=False))]
    speed = time_predictions(best, X_time)

    cvr = search.cv_results_
    keep = ["params", "mean_train_score", "std_train_score",
            "mean_test_score", "std_test_score", "rank_test_score",
            "mean_fit_time", "mean_score_time"]
    cv_results = {k: cvr[k] for k in keep if k in cvr}
    cv_results.update({k: cvr[k] for k in cvr if k.startswith("split")
                       and k.endswith("_test_score")})

    result = {
        "dataset": dataset, "model": model, "partition": partition,
        "trial": trial, "train_frac": C.PARTITIONS[partition],
        "n_train": int(len(tr)), "n_test": int(len(te)),
        "n_features": int(X.shape[1]),
        "train_positive_rate": float(y_tr.mean()),
        "test_positive_rate": float(y_te.mean()),
        "test_majority_baseline": float(max(y_te.mean(), 1 - y_te.mean())),
        "cv_folds": folds, "n_settings": len(cvr["params"]),
        "smoke": smoke, "device": device, "seed": seed,
        "best_params": search.best_params_,
        "train_acc": train_acc,
        "val_acc": float(search.best_score_),
        "val_acc_std": float(cvr["std_test_score"][search.best_index_]),
        "cv_train_acc": float(cvr["mean_train_score"][search.best_index_]),
        "test_acc": test_acc,
        "confusion_matrix": {"labels": [0, 1], "rows_true_cols_pred": cm},
        "tune_time_s": tune_time,
        "fit_time_s": float(search.refit_time_),
        **speed,
        "cv_results": cv_results,
        "gpu_at_start": gpu_before,
        "versions": versions(),
        "host": socket.gethostname(),
        "started": started,
        "job_wall_time_s": time.perf_counter() - t_job,
    }

    # 5. write atomically
    results_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(jsonable(result), indent=1))
    os.replace(tmp, out_path)
    print(f"[{dataset}/{model}/{partition}/trial{trial}] train {train_acc:.4f} "
          f"val {result['val_acc']:.4f} test {test_acc:.4f} | best "
          f"{search.best_params_} | tune {tune_time:.1f}s fit "
          f"{result['fit_time_s']:.2f}s predict {speed['predict_ms_per_1k']:.2f}"
          f" ms/1k -> {out_path.name}", flush=True)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=C.DATASETS)
    ap.add_argument("--model", required=True, choices=C.MODELS)
    ap.add_argument("--partition", required=True, choices=list(C.PARTITIONS))
    ap.add_argument("--trial", required=True, type=int,
                    choices=range(1, C.N_TRIALS + 1))
    ap.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
    ap.add_argument("--smoke", action="store_true",
                    help="tiny grid + 2-fold CV; writes to results/smoke/")
    ap.add_argument("--cv-folds", type=int, default=None)
    ap.add_argument("--force", action="store_true",
                    help="re-run even if the results file exists")
    args = ap.parse_args()
    results_dir = output_dir(args.smoke, args.device)
    run(args.dataset, args.model, args.partition, args.trial,
        device=args.device, smoke=args.smoke, cv_folds=args.cv_folds,
        results_dir=results_dir, force=args.force)


if __name__ == "__main__":
    main()

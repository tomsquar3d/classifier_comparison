"""Rescore every finished job with the paper's probability-based metrics.

The sweep saved hard predictions only (accuracy + confusion matrix). The
paper's other metrics need each model's predicted probability (or score) per
test row. This script, for each results/<job>.json:
  1. rebuilds the model with that job's best hyperparameters (no tuning),
  2. refits it on the same stored training split, with the same seed,
  3. scores the same test split, and saves
       results/scores/<job>.npz    y_true, score (probability of class 1)
       results/metrics/<job>.json  the metrics below.

Metrics (as in Caruana & Niculescu-Mizil 2006, section 2.2):
  roc_auc  area under the ROC curve
  apr      average precision
  bep      precision/recall break-even point (precision = recall when the
           top-k scored rows are called positive, k = number of positives)
  lift     lift at 25%: positive rate in the top 25% of scores / overall rate
  rms      root mean squared error of the probability
  mxe      mean cross-entropy (log loss, natural log, probabilities clipped)
  acc_refit  accuracy of the refit model's own predict() — a check against
             the sweep's accuracy (GPU models are not bit-for-bit
             reproducible, so small differences are expected)

SVM gives no probability without calibration. As in the paper's uncalibrated
results, its decision values are min-max scaled to [0, 1] on each test set
(rank metrics are unaffected; rms and mxe depend on that scaling — note it).

Usage (DSMLP container, cogs118a env):
    python src/rescore.py                  # all jobs, skips ones already done
    python src/rescore.py --models svm     # a subset
    python src/rescore.py --force          # redo everything
Takes roughly 20-40 minutes for all 216 jobs (no grid search).
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
import models  # noqa: E402
from run_job import jsonable, to_numpy  # noqa: E402

LIFT_FRACTION = 0.25          # paper: lift at 25%
EPS = 1e-15                   # probability clip for cross-entropy


# ---------------------------------------------------------------- metrics
def bep(y: np.ndarray, s: np.ndarray) -> float:
    k = int(y.sum())
    if k == 0:
        return float("nan")
    top = np.argsort(-s, kind="stable")[:k]
    return float(y[top].mean())     # precision at k = recall at k


def lift(y: np.ndarray, s: np.ndarray, frac: float = LIFT_FRACTION) -> float:
    n_top = max(1, int(round(frac * len(y))))
    top = np.argsort(-s, kind="stable")[:n_top]
    return float(y[top].mean() / y.mean())


def prob_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    pc = np.clip(p, EPS, 1 - EPS)
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "apr": float(average_precision_score(y, p)),
        "bep": bep(y, p),
        "lift": lift(y, p),
        "rms": float(np.sqrt(np.mean((p - y) ** 2))),
        "mxe": float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc))),
    }


def positive_score(model_name: str, est, X: np.ndarray) -> tuple[np.ndarray, str]:
    """Probability of class 1 (or a [0,1]-scaled score for SVM)."""
    if model_name == "svm":
        d = to_numpy(est.decision_function(X)).astype(float)
        lo, hi = d.min(), d.max()
        return ((d - lo) / (hi - lo) if hi > lo else np.full_like(d, 0.5)), \
            "svm decision values, min-max scaled to [0,1] on this test set"
    proba = est.predict_proba(X)
    if hasattr(proba, "to_numpy"):
        proba = proba.to_numpy()
    if hasattr(proba, "get"):
        proba = proba.get()
    proba = np.asarray(proba)
    classes = getattr(est, "classes_", None)
    col = 1
    if classes is not None:
        classes = to_numpy(classes)
        col = int(np.flatnonzero(classes == 1)[0])
    return proba[:, col].astype(float), "predict_proba"


# ---------------------------------------------------------------- one job
def rescore(path: Path, device: str, out_root: Path, force: bool) -> str:
    r = json.loads(path.read_text())
    if r.get("smoke") or r.get("device") != device:
        return "skip (smoke or other device)"
    name = path.stem
    m_path = out_root / "metrics" / f"{name}.json"
    if m_path.exists() and not force:
        return "skip (done)"

    d, m, p, t = r["dataset"], r["model"], r["partition"], r["trial"]
    data = np.load(C.PROCESSED_DIR / f"{d}.npz")
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)
    tr = data[f"train_idx__{p}__t{t - 1}"]
    te = data[f"test_idx__{p}__t{t - 1}"]

    seed = r["seed"]
    est, _ = models.build(m, X.shape[1], device=device, seed=seed)
    params = dict(r["best_params"])
    if m == "ann" and "net__module__hidden" in params:
        params["net__module__hidden"] = tuple(params["net__module__hidden"])
    if params:
        est.set_params(**params)

    y_fit = y[tr].astype(np.int32) if (device == "gpu" and m in ("rf", "svm")) else y[tr]
    t0 = time.perf_counter()
    est.fit(X[tr], y_fit)
    fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    score, how = positive_score(m, est, X[te])
    score_s = time.perf_counter() - t0

    y_te = y[te]
    met = prob_metrics(y_te, score)
    met["acc_refit"] = float(np.mean(to_numpy(est.predict(X[te])).astype(int) == y_te))
    out = {"dataset": d, "model": m, "partition": p, "trial": t,
           "best_params": r["best_params"], "score_source": how,
           "acc_sweep": r["test_acc"], "acc_refit_diff": met["acc_refit"] - r["test_acc"],
           "refit_time_s": fit_s, "score_time_s": score_s, **met}

    (out_root / "metrics").mkdir(parents=True, exist_ok=True)
    (out_root / "scores").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_root / "scores" / f"{name}.npz",
                        y_true=y_te.astype(np.int8), score=score.astype(np.float32))
    tmp = m_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(jsonable(out), indent=1))
    tmp.replace(m_path)
    return (f"auc {met['roc_auc']:.4f} apr {met['apr']:.4f} rms {met['rms']:.4f} "
            f"acc refit {met['acc_refit']:.4f} (sweep {r['test_acc']:.4f}) "
            f"[{fit_s + score_s:.1f}s]")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--datasets", nargs="+", choices=C.DATASETS, default=C.DATASETS)
    ap.add_argument("--models", nargs="+", choices=C.MODELS, default=C.MODELS)
    ap.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
    ap.add_argument("--results-dir", type=Path, default=C.RESULTS_DIR,
                    help="where the sweep's results JSONs are (default results/)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    jobs = sorted(p for p in args.results_dir.glob("*_trial*.json")
                  if any(p.name.startswith(f"{d}_{m}_") for d in args.datasets
                         for m in args.models))
    print(f"{len(jobs)} result files -> {args.results_dir}/metrics, /scores", flush=True)
    failed, t_start = [], time.perf_counter()
    for i, path in enumerate(jobs, 1):
        try:
            msg = rescore(path, args.device, args.results_dir, args.force)
        except Exception as e:  # noqa: BLE001
            failed.append(path.stem)
            msg = f"FAILED: {type(e).__name__}: {e}"
            traceback.print_exc(limit=3)
        print(f"[{i}/{len(jobs)}] {path.stem}: {msg}", flush=True)
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
    print(f"\nDone in {time.perf_counter() - t_start:.0f}s; {len(failed)} failed.")
    for f in failed:
        print("  ", f)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

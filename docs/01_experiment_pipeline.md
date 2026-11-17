# Experiment Pipeline
**Project:** COGS 118A Classifier Comparison
**Environment:** Python 3.11 (conda env `cogs118a`) on yggdrasil — 8× NVIDIA L40S (48 GB each, driver 580.82.07, CUDA 13.0), 64 CPU cores, 503 GB RAM. **Shared:** other users' jobs hold ~29–39 GB per GPU and several GPUs run near 100% (checked 2026-11-16)
**Version:** v1 — data prep built and run; experiment script and launcher not written
**Last Updated:** 2026-11-16

---

## 🔵 STATUS — Step 1 done; Steps 2–5 designed, not built

Step 1 ran on the server 2026-11-16. Steps 2–5 are the agreed plan. Everything is `[DRAFTED — not yet executed]` until it runs; then relabel 🟢 and record outputs in `02_results.md`.

---

## Document conventions

🟢 [ORIGINAL] — run/preserved verbatim
🟡 [RECONSTRUCTED] — rebuilt from descriptions; verify before re-running
🔵/🟣/🟠 [NEW — <date>] — added on that date
[DRAFTED — not yet executed] — written, never run

---

## Overview

```
data prep (per dataset) → processed files in data/
        ↓
timing test (1 dataset, all 6 models, 80/20, 1 trial)
        ↓
launcher → 216 independent jobs, one per GPU at a time
        ↓
each job: split → GridSearchCV (5-fold) → refit best → score → write results/<job>.json
        ↓
aggregate → mean over 3 trials → tables + figures
```

Job count: 6 classifiers × 4 datasets × 3 partitions × 3 trials = 216.

| Classifier | Library | Device |
|------------|---------|--------|
| Random forest | RAPIDS cuML | GPU |
| XGBoost | xgboost (`device="cuda"`, `tree_method="hist"`) | GPU |
| SVM, RBF kernel | cuML `SVC` | GPU |
| ANN (MLP) | PyTorch via skorch | GPU |
| KNN | cuML | GPU |
| TabPFN | tabpfn 10.0.0, **v2 weights** via `TabPFNClassifier.create_default_for_version(ModelVersion.V2)` (package default is 3.5, which needs a license token) | GPU |

---

## Step 1: Data prep — 🟢 [NEW — 2026-11-04, run 2026-11-16]

For each dataset: download from UCI → handle missing values → one-hot encode categoricals → binarize labels (rules in `03`) → subsample if large → save processed file + a summary row (rows, features, % positive) for the report's dataset table.

Also pre-computes all 9 train/test splits (3 partitions × 3 trials, stratified, seeds from `config.split_seed`) and stores them in the same file, so every classifier sees identical rows. Step 2's "split with the trial seed" now means "load the stored split".

Code: `src/prepare_data.py` (settings in `src/config.py`). Run: `python src/prepare_data.py`.
Output: `data/processed/<dataset>.npz` + `data/processed/dataset_summary.json`. Row counts in `03`.

## Step 2: Per-job experiment script — 🔵 [NEW — 2026-11-04] [DRAFTED — not yet executed]

Inputs: dataset, classifier, partition (0.2 / 0.5 / 0.8 train share), trial seed.

1. Load the stored train/test indices for this partition and trial (from Step 1).
2. Apply any per-model training cap (SVM, TabPFN — see `03`) by random subsample of the training split only. Test set stays the same for every model.
3. Pipeline: scaler + model for SVM, KNN, ANN. No scaling for RF, XGBoost. TabPFN handles its own preprocessing.
4. `GridSearchCV(cv=5, scoring="accuracy", return_train_score=True, n_jobs=1)` on the training split. TabPFN: no search.
5. Refit best config on the full training split; record train accuracy, mean CV validation accuracy, test accuracy, best params, fit time.
6. Write `results/<dataset>_<clf>_<partition>_trial<k>.json` plus the full `cv_results_` (for heatmaps).

⚠️ `n_jobs=1` inside each job — parallel folds on one GPU fight for memory. Parallelize across jobs/GPUs instead.

Code: ⛔ NOT YET RECORDED (not written).

## Step 3: Launcher — 🔵 [NEW — 2026-11-04] [DRAFTED — not yet executed]

Builds the 216-job list, skips jobs whose results file already exists (so restarts resume), assigns each job a GPU via `CUDA_VISIBLE_DEVICES`, runs several at once.

⚠️ GPUs are shared with other users (see header). The launcher must check free memory (`nvidia-smi`) before placing a job instead of assuming an empty GPU, and compute times will be noisier than on a dedicated machine.

Code: ⛔ NOT YET RECORDED (not written).

## Step 4: Timing test — 🔵 [NEW — 2026-11-04] [DRAFTED — not yet executed]

Run one dataset, all 6 models, 80/20, 1 trial at target size. Multiply by 36 for a total-time estimate. Use the result to finalize dataset sizes and per-model caps.

Result: ⛔ NOT YET RECORDED.

## Step 5: Aggregation + figures — 🔵 [NEW — 2026-11-04] [DRAFTED — not yet executed]

Average over 3 trials; build the main results table and the figures listed in `04_analysis_plan.md`.

Code: ⛔ NOT YET RECORDED (not written).

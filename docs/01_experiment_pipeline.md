# Experiment Pipeline
**Project:** COGS 118A Classifier Comparison
**Environment:** Python 3.11 (conda env `cogs118a`, CUDA 13.0). **Platform changed 2026-11-24:** the lab server (yggdrasil, 8× L40S) is no longer available. Now: UCSD DSMLP (first choice, access pending), Kaggle 2× T4 (backup), Colab T4 (overflow). GPUs must be Turing / T4 or newer; the personal GTX 1080 Ti (Pascal) is ruled out
**Version:** v2 — revised for cloud GPUs; data prep must be re-run on the new platform; experiment script and runner not written
**Last Updated:** 2026-11-24

---

## 🔵 STATUS — Step 1 built (needs re-run on new platform); Steps 2–5 designed, not built

Step 1 ran on yggdrasil 2026-11-16, but that server and its `data/` folder are gone. Re-run it on the new platform with the 12.5k caps and Adult unbalanced (action item #29). Steps 2–5 are the agreed plan, revised 2026-11-24 (`05` #13–17) and 2026-11-26 (`05` #19–21). Everything is `[DRAFTED — not yet executed]` until it runs; then relabel 🟢 and record outputs in `02_results.md`.

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
timing test (HIGGS 12.5k, all 6 models, 80/20, 1 trial)
        ↓
resumable runner → 216 jobs; skips any job whose results file exists
        ↓
every job: GridSearchCV (5-fold) on its own training split → refit best → score → write results/<job>.json
(jobs are independent; any order)
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
| KNN | cuML (proposed 2026-11-26: scikit-learn CPU, for distance weighting — `05` #24) | GPU |
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
2. No per-model caps (`05` #19). Check the training split is within `TRAIN_CAP` (TabPFN 10k) and stop with an error if not.
3. Pipeline: scaler + model for SVM, KNN, ANN. No scaling for RF, XGBoost. TabPFN handles its own preprocessing.
4. **Every trial** (`05` #20, 2026-11-26): `GridSearchCV(cv=5, scoring="accuracy", return_train_score=True, n_jobs=1)` on this trial's training split, then save the best params. TabPFN: no search.
5. Jobs don't depend on each other (trial-1-only tuning was dropped, `05` #20).
6. Record train accuracy, validation accuracy (best CV score), test accuracy, best params, and the test confusion matrix (rubric welcomes confusion matrices).
7. Measure speed (🟣 [NEW — 2026-11-24], see below).
8. Write `results/<dataset>_<clf>_<partition>_trial<k>.json` plus the full `cv_results_` (for heatmaps).

⚠️ `n_jobs=1` inside each job — parallel folds on one GPU fight for memory.

MLP: early stopping on, grid of 6–8 combos (`03`). Fallback if the timing test says the sweep is too slow: `CV_FOLDS = 3`, documented in the report.

### Speed measurement — 🟣 [NEW — 2026-11-24] [DRAFTED — not yet executed]

Four numbers per job, all wall-clock seconds from `time.perf_counter()`. Settings live in `src/config.py` (`TIMING_*`).

| Field | What it times | How |
|-------|---------------|-----|
| `tune_time_s` | Whole grid search: every config × 5 folds, plus the refit | Timer around `GridSearchCV.fit`. Every job. TabPFN = 0 (no search) |
| `fit_time_s` | Training the final model once on the full training split | `GridSearchCV.refit_time_`. TabPFN: timer around `fit` |
| `predict_ms_per_1k` | Prediction speed, normalized | Median of `TIMING_REPEATS` timed `predict` calls on a fixed `TIMING_N_ROWS`-row sample of the test split, ÷ rows × 1000 |
| `predict_rows_per_s` | Same measurement as throughput | rows ÷ median time |

Rules:
- **Warm-up first.** `TIMING_WARMUP` untimed `predict` call(s) before timing, so GPU start-up cost isn't counted.
- **Time until predictions are back on the CPU as a NumPy array.** That forces the GPU to finish its work (GPU calls return before the work is done), and it counts the same data copies for every model. For cuML, set `output_type="numpy"`.
- **Same timing sample for every model:** drawn with a fixed seed from the stored test indices, so all six models are timed on the same rows.
- **Only `fit`/`predict` inside the timers.** No data loading, scaling setup or scoring.
- **Record GPU conditions at job start:** GPU model name, index, free memory and utilization (`nvidia-smi --query-gpu=name,index,memory.free,utilization.gpu --format=csv`). This lets noisy timings be flagged, and shows which GPU type each job ran on.
- ⚠️ **Only compare timings from the same GPU type.** If jobs end up spread across different machines or GPU models, speed numbers from different GPUs can't go in the same comparison; re-time on one GPU type instead (action item #26).

Accuracy still comes from one full `predict` on the whole test set. The timing sample is only for speed.

Code: ⛔ NOT YET RECORDED (not written).

## Step 3: Resumable runner — 🟠 [REVISED — 2026-11-24] [DRAFTED — not yet executed]

Builds the 216-job list (4 datasets × 3 partitions × 3 trials × 6 models), runs them in any order (jobs are independent since `05` #20), and skips any job whose results file already exists. Cloud sessions get cut off (Kaggle: 12-h sessions, ~30 GPU-h/week; Colab limits not guaranteed), so a cut-off loses one job, not the sweep.

Kaggle: run as a batch "Save version" job; with 2× T4, two jobs can run at once, one per GPU (`CUDA_VISIBLE_DEVICES`). Results files must be copied out of each session (e.g. pushed to the repo or saved as a Kaggle dataset) before it ends.

Replaces the 2026-11-04 multi-GPU launcher design (8 shared L40S, free-memory checks), which no longer applies.

Code: ⛔ NOT YET RECORDED (not written).

## Step 4: Timing test — 🔵 [NEW — 2026-11-04] [DRAFTED — not yet executed]

Run the worst case: HIGGS 12.5k, all 6 models, 80/20, 1 trial. Multiply each model's time by 36 for a sweep estimate (rough expectation from planning: 5–10 T4 GPU-hours — an estimate to check, not a measurement). Decide from it whether 5-fold CV fits or the 3-fold fallback is needed. Also a first check that the speed fields (Step 2) come out sensible.

Result: ⛔ NOT YET RECORDED.

## Step 5: Aggregation + figures — 🔵 [NEW — 2026-11-04] [DRAFTED — not yet executed]

Average over 3 trials; build the main results table, the speed table and the figures listed in `04_analysis_plan.md`.

Code: ⛔ NOT YET RECORDED (not written).

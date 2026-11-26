# Results
**Project:** COGS 118A Classifier Comparison
**Results source:** `results/*.json` (none yet)
**Version:** v0.2
**Last Updated:** 2026-11-24

---

## 🔵 STATUS — Data prepped on DSMLP (2026-11-26); no model results yet

Data prep ran on yggdrasil 2026-11-16 (§1), but every dataset is now capped at 12,500 rows with Adult left unbalanced (`05` #19, #21) and that server is gone, so `prepare_data.py` must be re-run on the new platform. No classifier has run. Every number in this doc must come from a results file or a script run — no estimates. Superseded numbers stay visible (version table or a note) when anything is re-run. Speed results (§6) added to the plan 2026-11-24.

---

## Run history

| Version | Jobs complete | Change | Date |
|---------|---------------|--------|------|
| — | 0 / 216 | — | — |

---

## 1. Dataset summary (after prep) — 🟢 run on DSMLP 2026-11-26

| Dataset | Rows used | Features (after encoding) | % positive | Majority-class baseline | Source rows |
|---------|-----------|---------------------------|------------|-------------------------|-------------|
| Adult | 12,500 | 104 | 24.78% | 75.22% | 45,222 (after dropping missing) |
| Letter | 12,500 | 16 | 50.30% | 50.30% | 20,000 |
| Covertype | 12,500 | 54 | 48.76% | 51.24% | 581,012 |
| HIGGS | 12,500 | 28 | 52.96% | 52.96% | 199,190 (random read of the 11M file) |

> Source: `data/processed/dataset_summary.json` from `src/prepare_data.py`, run on DSMLP 2026-11-26 (12.5k stratified cap, no rebalancing — `05` #19, #23). Training rows per partition: 2,500 / 6,250 / 10,000.
>
> Superseded: the 2026-11-16 yggdrasil run (Adult 45,222 / Letter 20,000 / Covertype 581,012 / HIGGS 1,000,000 used in full or at 100k), and the planned 100k caps and 50/50 balancing (`05` #14, #18).

## 2. Timing test — 🟢 run on DSMLP 2026-11-26

HIGGS, 80/20 (10,000 training rows), trial 1, full grids, 5-fold CV. RTX PRO 6000 Blackwell MIG slice (24 GB). Whole run 15m17s. Source: `results/logs/higgs_*_80_20_trial1.log` (results JSONs on DSMLP, not yet in git).

| Model | Train | Val | Test | Best params | Tune (s) | Fit (s) | Predict (ms / 1k rows) | Job wall time |
|---|---|---|---|---|---|---|---|---|
| TabPFN | 0.8104 | 0.7208 | 0.7316 | — | 92.1 | 3.80 | 3175.08 | 2m43s |
| XGBoost | 0.9102 | 0.7040 | 0.7124 | max_depth 6, n_estimators 128 | 61.0 | 0.22 | 0.72 | 1m06s |
| RF | 1.0000 | 0.7055 | 0.7080 | max_features 8 | 100.2 | 1.71 | 0.63 | 1m46s |
| ANN | 0.7033 | 0.6742 | 0.6872 | hidden (32,), momentum 0.9 | 421.9 | 3.06 | 319.90 | 7m17s |
| SVM | 0.7516 | 0.6659 | 0.6732 | C 100, gamma 0.005 | 105.7 | 0.57 | 1.44 | 1m50s |
| KNN | 1.0000 | 0.6179 | 0.6280 | K 25, distance | 26.5 | 0.00 | 16.44 | 0m32s |

Majority-class baseline (HIGGS): 0.5296.

Findings:
- **Sweep estimate:** ≤ 15.3 min × 36 ≈ 9.2 h upper bound (every job at the 10k size); smaller partitions are faster, so ~5–6 h expected. Fits a 12-h container → **keep 5-fold CV** (no 3-fold fallback).
- **ANN undertrained:** train accuracy only 0.70 and the smallest net won — points to early stopping / learning rate, not capacity. Diagnostic `src/ann_check.py` written; ANN settings to be fixed before the sweep. ANN numbers above are superseded once fixed.
- **ANN predict time** inflated by 128-row prediction batches; prediction batch raised to 4,096 (`ANN_PREDICT_BATCH_SIZE`), accuracy unaffected.
- **KNN train accuracy 1.000** is expected with distance weighting (each training point is its own nearest neighbour) — say so in the report.
- **TabPFN predict** is genuinely slow (in-context model re-reads the training set at prediction time) — use in the speed-vs-accuracy section.

## 3. Main results — mean train / val / test accuracy over 3 trials

One table per dataset, rows = classifier, column groups = partition (20/80, 50/50, 80/20), each with Train / Val / Test. To be filled from aggregated results.

## 4. Best hyperparameters

Per classifier × dataset × partition, the selected config in each trial.

## 5. Rankings

Per dataset × partition rank order; overall mean rank.

## 6. Speed — 🟣 [NEW — 2026-11-24]

Per classifier × dataset × partition, median over 3 trials. Definitions in `01` Step 2 (speed measurement).

| Column | Meaning |
|--------|---------|
| Tune (s) | Whole grid search incl. refit |
| Fit (s) | Final model, one training run |
| Predict (ms / 1k rows) | Median of timed repeats on the fixed timing sample |
| Throughput (rows/s) | Same measurement, inverted |

Plus the GPU conditions logged at each job's start, so slow outliers caused by other users' load can be identified. All numbers ⛔ NOT YET RECORDED until the sweep runs.

## 7. Bonus analyses

Rank significance (bootstrap or Friedman / critical-difference), accuracy-vs-speed plot (§6), any optional extras.

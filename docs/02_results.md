# Results
**Project:** COGS 118A Classifier Comparison
**Results source:** `results/*.json` (none yet)
**Version:** v0.2
**Last Updated:** 2026-11-24

---

## 🟢 STATUS — Full sweep complete (216/216 jobs, 2026-11-26)

Data prep ran on yggdrasil 2026-11-16 (§1), but every dataset is now capped at 12,500 rows with Adult left unbalanced (`05` #19, #21) and that server is gone, so `prepare_data.py` must be re-run on the new platform. No classifier has run. Every number in this doc must come from a results file or a script run — no estimates. Superseded numbers stay visible (version table or a note) when anything is re-run. Speed results (§6) added to the plan 2026-11-24.

---

## Run history

| Version | Jobs complete | Change | Date |
|---------|---------------|--------|------|
| v1.0 | 216 / 216 (3 trials) | Trials 2–3 added; same GPU type throughout; trial-1 files unchanged | 2026-11-26 |
| v0.3 | 72 / 216 (trial 1, all cells) | First real sweep on DSMLP; all jobs on one GPU type (RTX PRO 6000 Blackwell Max-Q, MIG slice) | 2026-11-26 |

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
- **ANN not undertrained (checked 2026-11-26):** `src/ann_check.py` on this split, 2×128 net, 8,000 fit rows / 2,000 held out:

  | Setting | Epochs run | Best epoch | Train | Held-out |
  |---|---|---|---|---|
  | SGD lr 0.01, patience 10 (current) | 31 | 21 | 0.725 | 0.663 |
  | SGD lr 0.01, patience 30 | 51 | 21 | 0.725 | 0.663 |
  | SGD lr 0.05, patience 10 | 20 | 10 | 0.751 | 0.659 |
  | SGD lr 0.1, patience 10 | 14 | 4 | 0.703 | 0.655 |
  | SGD lr 0.1, patience 30 | 34 | 4 | 0.703 | 0.655 |
  | Adam lr 0.001, patience 10 | 17 | 7 | 0.713 | 0.656 |

  Every setting ends at ~0.66 held-out; longer patience finds the same best epoch; higher lr / Adam don't help. Low training accuracy is early stopping keeping pre-overfitting weights, not slow learning. HIGGS at ~8k rows is data-limited for neural nets (smallest net won the timing test). **Settings unchanged.**
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

## 3. Main results — 3 trials (216 jobs, 2026-11-26)

Source: `results/*_trial{1,2,3}.json` → `python src/aggregate.py` → `results/summary/`. Supersedes the trial-1-only interim tables (v0.3).

Mean over 3 trial(s). Cells: train / val / test accuracy (test ± std when ≥2 trials). Generated by `src/aggregate.py`.

**Adult** (majority-class baseline 0.752)

| Model | 20/80 | 50/50 | 80/20 |
|---|---|---|---|
| Random forest | 0.968 / 0.855 / **0.849** ± 0.003 | 0.904 / 0.857 / **0.855** ± 0.004 | 0.925 / 0.858 / **0.860** ± 0.008 |
| XGBoost | 0.896 / 0.860 / **0.855** ± 0.001 | 0.884 / 0.865 / **0.864** ± 0.003 | 0.884 / 0.867 / **0.869** ± 0.005 |
| SVM (RBF) | 0.881 / 0.840 / **0.842** ± 0.001 | 0.868 / 0.849 / **0.845** ± 0.003 | 0.862 / 0.850 / **0.852** ± 0.004 |
| MLP | 0.857 / 0.840 / **0.836** ± 0.002 | 0.865 / 0.847 / **0.843** ± 0.004 | 0.861 / 0.847 / **0.849** ± 0.009 |
| KNN | 0.893 / 0.833 / **0.830** ± 0.001 | 0.946 / 0.835 / **0.828** ± 0.003 | 0.946 / 0.834 / **0.830** ± 0.010 |
| TabPFN | 0.879 / 0.855 / **0.854** ± 0.002 | 0.873 / 0.860 / **0.856** ± 0.004 | 0.870 / 0.859 / **0.863** ± 0.006 |

**Letter (A–M vs N–Z)** (majority-class baseline 0.503)

| Model | 20/80 | 50/50 | 80/20 |
|---|---|---|---|
| Random forest | 1.000 / 0.919 / **0.924** ± 0.003 | 1.000 / 0.948 / **0.957** ± 0.003 | 1.000 / 0.961 / **0.965** ± 0.001 |
| XGBoost | 1.000 / 0.919 / **0.925** ± 0.004 | 1.000 / 0.952 / **0.957** ± 0.002 | 1.000 / 0.964 / **0.969** ± 0.002 |
| SVM (RBF) | 0.999 / 0.937 / **0.937** ± 0.001 | 1.000 / 0.964 / **0.971** ± 0.001 | 1.000 / 0.974 / **0.977** ± 0.002 |
| MLP | 0.970 / 0.901 / **0.903** ± 0.005 | 0.983 / 0.938 / **0.940** ± 0.007 | 0.989 / 0.951 / **0.956** ± 0.001 |
| KNN | 1.000 / 0.926 / **0.933** ± 0.002 | 1.000 / 0.956 / **0.962** ± 0.002 | 1.000 / 0.965 / **0.968** ± 0.004 |
| TabPFN | 1.000 / 0.956 / **0.959** ± 0.005 | 1.000 / 0.976 / **0.981** ± 0.002 | 1.000 / 0.984 / **0.986** ± 0.001 |

**Covertype** (majority-class baseline 0.512)

| Model | 20/80 | 50/50 | 80/20 |
|---|---|---|---|
| Random forest | 0.992 / 0.797 / **0.792** ± 0.007 | 1.000 / 0.821 / **0.830** ± 0.003 | 1.000 / 0.838 / **0.845** ± 0.008 |
| XGBoost | 0.952 / 0.787 / **0.787** ± 0.002 | 1.000 / 0.817 / **0.825** ± 0.005 | 1.000 / 0.838 / **0.845** ± 0.007 |
| SVM (RBF) | 0.824 / 0.777 / **0.764** ± 0.007 | 0.880 / 0.796 / **0.804** ± 0.004 | 0.899 / 0.821 / **0.821** ± 0.005 |
| MLP | 0.800 / 0.771 / **0.763** ± 0.009 | 0.814 / 0.788 / **0.783** ± 0.004 | 0.843 / 0.800 / **0.798** ± 0.028 |
| KNN | 1.000 / 0.747 / **0.744** ± 0.005 | 1.000 / 0.783 / **0.789** ± 0.005 | 1.000 / 0.804 / **0.806** ± 0.006 |
| TabPFN | 0.934 / 0.807 / **0.819** ± 0.005 | 0.953 / 0.850 / **0.858** ± 0.003 | 0.938 / 0.868 / **0.868** ± 0.014 |

**HIGGS** (majority-class baseline 0.530)

| Model | 20/80 | 50/50 | 80/20 |
|---|---|---|---|
| Random forest | 1.000 / 0.684 / **0.689** ± 0.003 | 1.000 / 0.705 / **0.702** ± 0.006 | 1.000 / 0.709 / **0.713** ± 0.004 |
| XGBoost | 0.896 / 0.685 / **0.687** ± 0.003 | 0.803 / 0.705 / **0.701** ± 0.002 | 0.835 / 0.707 / **0.712** ± 0.006 |
| SVM (RBF) | 0.701 / 0.640 / **0.644** ± 0.005 | 0.765 / 0.663 / **0.663** ± 0.005 | 0.749 / 0.668 / **0.680** ± 0.006 |
| MLP | 0.675 / 0.636 / **0.619** ± 0.012 | 0.696 / 0.657 / **0.662** ± 0.009 | 0.707 / 0.674 / **0.686** ± 0.001 |
| KNN | 0.758 / 0.595 / **0.596** ± 0.003 | 1.000 / 0.613 / **0.615** ± 0.005 | 1.000 / 0.618 / **0.626** ± 0.006 |
| TabPFN | 0.876 / 0.699 / **0.709** ± 0.003 | 0.865 / 0.721 / **0.717** ± 0.007 | 0.828 / 0.721 / **0.732** ± 0.003 |

**Mean rank by test accuracy** (1 = best; over all dataset × partition cells)

| Random forest | XGBoost | SVM (RBF) | MLP | KNN | TabPFN |
|---|---|---|---|---|---|
| 3.00 | 2.67 | 3.58 | 5.33 | 5.17 | 1.25 |

Observations:
- **TabPFN best in 9 of 12 cells**; **XGBoost best in the other 3 (all of Adult)**. Paper ordering for the five paper models holds: boosted trees (XGBoost) ≈ RF > SVM > ANN / KNN.
- **More training data → higher test accuracy in 23 of 24 dataset × model series** (rubric comparison (b)). Only exception: KNN on Adult, flat (0.830 / 0.828 / 0.830). The trial-1 dips (MLP Covertype, TabPFN HIGGS) disappeared with averaging.
- Trial-to-trial spread is small: mean test-accuracy std 0.003–0.008 per model.
- Adult margins over the majority-class baseline (0.752) are the smallest; HIGGS is the hardest problem (best 0.732, baseline 0.530).
- Largest mean train–test gaps: KNN 0.168 (distance weighting → train acc 1.0), RF 0.151 (deep trees); smallest: ANN 0.035, SVM 0.053.

## 4. Ranking significance (bonus)

Blocks = the 12 dataset × partition cells; value = mean test accuracy over 3 trials (Demšar 2006).

| | TabPFN | XGBoost | RF | SVM | KNN | MLP |
|---|---|---|---|---|---|---|
| Mean rank | 1.25 | 2.67 | 3.00 | 3.58 | 5.17 | 5.33 |

- **Friedman test:** χ² = 41.67, p = 6.9e-08 → the models are not all equal.
- **Nemenyi critical difference (α = 0.05):** 2.18 ranks. Significantly different pairs (rank gap > CD): TabPFN vs SVM, KNN, MLP; XGBoost vs KNN, MLP; RF vs MLP. **Not** significant: TabPFN vs XGBoost / RF; XGBoost vs RF vs SVM; RF vs KNN (gap 2.17, just under the CD).
- **Bootstrap rank probabilities** (paper Table 4 format; 1,000 resamples of cells and trials, ranking by mean accuracy): TabPFN 1st 99.9%; XGBoost 2nd 82.6%; RF 3rd 82.3%; SVM 4th 99.6%; MLP 5th 88.4%; KNN 6th 88.4%. Note: by *mean accuracy* MLP edges KNN; by *mean rank* KNN edges MLP — they trade places depending on the summary, consistent with the CD test calling them tied.

Figures: `fig7_bootstrap_ranks`, `fig8_critical_difference`.

## 5. Speed (bonus) — 80/20 split, medians over datasets and trials, one GPU type

| Model | Tune (s) | Final fit (s) | Predict (ms / 1k rows) | Mean test acc (80/20) |
|---|---|---|---|---|
| TabPFN | 104.9 | 3.52 | 3480 | 0.865 |
| XGBoost | 53.8 | 0.52 | 1.73 | 0.857 |
| RF | 84.6 | 1.78 | 0.58 | 0.852 |
| SVM | 52.5 | 0.14 | 1.17 | 0.836 |
| MLP | 307.5 | 5.35 | 80.0 | 0.823 |
| KNN | 29.4 | 0.004 | 15.3 | 0.818 |

TabPFN is the most accurate but ~2,000× slower to predict than RF/XGBoost (it re-reads the training set at prediction time). RF/XGBoost give near-top accuracy at the lowest prediction cost. Source `results/summary/speed_table.csv`; figure `fig5_accuracy_vs_speed`.

Figures (all PNG + PDF in `figures/`): fig1 accuracy vs training share (± 1 std bands), fig2 rank table, fig3 SVM C × gamma heatmaps, fig4 KNN curves, fig5 accuracy vs speed, fig6 Adult confusion matrices, fig7 bootstrap ranks, fig8 critical difference.

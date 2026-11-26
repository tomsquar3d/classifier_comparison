# Results
**Project:** COGS 118A Classifier Comparison
**Results source:** `results/*.json` (none yet)
**Version:** v0.2
**Last Updated:** 2026-11-24

---

## 🔵 STATUS — Trial 1 complete (72/216 jobs, 2026-11-26); trials 2–3 pending

Data prep ran on yggdrasil 2026-11-16 (§1), but every dataset is now capped at 12,500 rows with Adult left unbalanced (`05` #19, #21) and that server is gone, so `prepare_data.py` must be re-run on the new platform. No classifier has run. Every number in this doc must come from a results file or a script run — no estimates. Superseded numbers stay visible (version table or a note) when anything is re-run. Speed results (§6) added to the plan 2026-11-24.

---

## Run history

| Version | Jobs complete | Change | Date |
|---------|---------------|--------|------|
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

## 3. Trial 1 results — interim (72 jobs, 2026-11-26)

⚠️ One trial only: no error bars yet; numbers change once trials 2–3 are averaged in. Regenerate with `python src/aggregate.py` (writes `results/summary/` and `figures/`).

Mean over 1 trial(s). Cells: train / val / test accuracy (test ± std when ≥2 trials). Generated by `src/aggregate.py`.

**Adult** (majority-class baseline 0.752)

| Model | 20/80 | 50/50 | 80/20 |
|---|---|---|---|
| Random forest | 0.959 / 0.850 / **0.852** | 0.910 / 0.853 / **0.857** | 0.901 / 0.855 / **0.869** |
| XGBoost | 0.877 / 0.854 / **0.855** | 0.883 / 0.864 / **0.865** | 0.878 / 0.864 / **0.876** |
| SVM (RBF) | 0.888 / 0.843 / **0.841** | 0.862 / 0.848 / **0.849** | 0.859 / 0.846 / **0.856** |
| MLP | 0.850 / 0.839 / **0.834** | 0.869 / 0.848 / **0.847** | 0.863 / 0.843 / **0.858** |
| KNN | 1.000 / 0.830 / **0.829** | 1.000 / 0.832 / **0.828** | 1.000 / 0.831 / **0.840** |
| TabPFN | 0.871 / 0.853 / **0.856** | 0.871 / 0.859 / **0.860** | 0.868 / 0.857 / **0.869** |

**Letter (A–M vs N–Z)** (majority-class baseline 0.503)

| Model | 20/80 | 50/50 | 80/20 |
|---|---|---|---|
| Random forest | 1.000 / 0.919 / **0.921** | 1.000 / 0.948 / **0.956** | 1.000 / 0.962 / **0.965** |
| XGBoost | 1.000 / 0.918 / **0.922** | 1.000 / 0.953 / **0.956** | 1.000 / 0.963 / **0.968** |
| SVM (RBF) | 0.999 / 0.937 / **0.936** | 1.000 / 0.964 / **0.972** | 1.000 / 0.975 / **0.979** |
| MLP | 0.964 / 0.908 / **0.907** | 0.983 / 0.936 / **0.943** | 0.991 / 0.952 / **0.956** |
| KNN | 1.000 / 0.925 / **0.934** | 1.000 / 0.957 / **0.963** | 1.000 / 0.967 / **0.964** |
| TabPFN | 1.000 / 0.955 / **0.954** | 1.000 / 0.977 / **0.980** | 1.000 / 0.985 / **0.986** |

**Covertype** (majority-class baseline 0.512)

| Model | 20/80 | 50/50 | 80/20 |
|---|---|---|---|
| Random forest | 1.000 / 0.792 / **0.799** | 1.000 / 0.817 / **0.833** | 1.000 / 0.839 / **0.844** |
| XGBoost | 0.947 / 0.782 / **0.789** | 1.000 / 0.815 / **0.831** | 1.000 / 0.837 / **0.844** |
| SVM (RBF) | 0.794 / 0.781 / **0.756** | 0.868 / 0.795 / **0.801** | 0.901 / 0.823 / **0.816** |
| MLP | 0.797 / 0.780 / **0.753** | 0.843 / 0.785 / **0.787** | 0.804 / 0.803 / **0.766** |
| KNN | 1.000 / 0.745 / **0.738** | 1.000 / 0.783 / **0.784** | 1.000 / 0.809 / **0.800** |
| TabPFN | 0.930 / 0.808 / **0.812** | 0.949 / 0.849 / **0.859** | 0.933 / 0.865 / **0.862** |

**HIGGS** (majority-class baseline 0.530)

| Model | 20/80 | 50/50 | 80/20 |
|---|---|---|---|
| Random forest | 1.000 / 0.674 / **0.688** | 1.000 / 0.709 / **0.696** | 1.000 / 0.706 / **0.708** |
| XGBoost | 0.768 / 0.680 / **0.689** | 0.795 / 0.711 / **0.699** | 0.910 / 0.704 / **0.712** |
| SVM (RBF) | 0.692 / 0.636 / **0.640** | 0.775 / 0.667 / **0.659** | 0.752 / 0.666 / **0.673** |
| MLP | 0.654 / 0.635 / **0.610** | 0.705 / 0.665 / **0.666** | 0.703 / 0.674 / **0.687** |
| KNN | 0.649 / 0.594 / **0.595** | 1.000 / 0.621 / **0.610** | 1.000 / 0.618 / **0.628** |
| TabPFN | 0.843 / 0.694 / **0.709** | 0.857 / 0.729 / **0.709** | 0.810 / 0.721 / **0.732** |

**Mean rank by test accuracy** (1 = best; over all dataset × partition cells)

| Random forest | XGBoost | SVM (RBF) | MLP | KNN | TabPFN |
|---|---|---|---|---|---|
| 3.12 | 2.50 | 3.75 | 5.08 | 5.33 | 1.21 |

Interim observations (trial 1 only — re-check after 3 trials):
- **TabPFN ranks first** in 10 of 12 cells (mean rank 1.21); XGBoost wins Adult at 50/50 and 80/20.
- **XGBoost and random forest are close** (mean ranks 2.50 vs 3.12) — consistent with the paper putting boosted trees and RF at the top.
- **SVM is strong on Letter** (2nd in all 3 Letter cells) but weaker on HIGGS; MLP and KNN rank last overall. Paper ordering (boosted trees ≈ RF > SVM > ANN > KNN) mostly holds.
- **More training data helps** almost everywhere (rubric comparison (b)). Exceptions: MLP on Covertype drops at 80/20 (0.787 → 0.766); KNN on Adult flat at 20/80 → 50/50; TabPFN on HIGGS flat at 20/80 → 50/50. Check whether these survive averaging.
- Every model beats its majority-class baseline; Adult's margin is smallest (baseline 0.752, best 0.876).
- **Speed** (80/20, same GPU): RF and XGBoost predict fastest (<2 ms / 1k rows); TabPFN slowest (~3 s / 1k rows) but most accurate; MLP slowest to tune (~5 min). See `figures/fig5_accuracy_vs_speed.png`.

Figures (trial 1): `fig1_test_acc_vs_train_size`, `fig2_rank_table`, `fig3_svm_heatmaps`, `fig4_knn_curves`, `fig5_accuracy_vs_speed`, `fig6_confusion_adult` (PNG + PDF in `figures/`).

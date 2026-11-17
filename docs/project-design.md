# COGS 118A Final Project — Design Decisions

Replication + modernization of Caruana & Niculescu-Mizil (ICML 2006), "An Empirical Comparison of Supervised Learning Algorithms."

## Classifiers (6, all GPU-capable)
| # | Classifier | From paper? | Library | Notes |
|---|---|---|---|---|
| 1 | Random Forest | Yes (#2 in paper) | RAPIDS cuML | Anchor for grading point (a) |
| 2 | XGBoost | Modern version of paper's boosted trees | xgboost, `device="cuda", tree_method="hist"` | Replaced LightGBM (easier GPU) |
| 3 | SVM, RBF kernel | Yes (#4 calibrated) | cuML SVC | Replaced AdaBoost; train cap provisional 50k |
| 4 | ANN (MLP) | Yes | PyTorch via skorch | 1–3 hidden layers |
| 5 | KNN | Yes | cuML | |
| 6 | TabPFN | New (tabular foundation model) | tabpfn | No tuning; **v2 open weights, 10k train-row cap** (decided 2026-11-16) |

Dropped: AdaBoost (no GPU, overlaps XGBoost), LightGBM (GPU build fiddly), RealMLP/TabM (overkill).

## Datasets (4, binary) — locked 2026-11-16
| Dataset | Rows used | Features | % positive | Rule |
|---|---|---|---|---|
| Adult | 45,222 | 104 | 24.8% | >50K = 1; missing rows dropped |
| Letter | 20,000 | 16 | 50.3% | A–M = 0, N–Z = 1 |
| Covertype | 581,012 (may shrink) | 54 | 48.8% | Largest class vs rest |
| HIGGS | 1,000,000 (may shrink) | 28 | 53.0% | Signal vs background |

## Protocol
- Partitions 20/80, 50/50, 80/20 × 3 random stratified trials; splits pre-computed so every classifier sees identical rows
- Tuning: plain GridSearchCV (5-fold, `return_train_score=True`) on training split only. Bayesian HPO dropped as overkill.
- Report mean train / validation / test accuracy + best hyperparameters per cell
- Scale features for SVM, KNN, ANN (inside Pipeline); not for trees
- Document library differences (cuML vs sklearn) and any per-model subsampling (SVM, TabPFN)

## Infrastructure
- yggdrasil: 8× L40S 48 GB, CUDA 13.0, 64 cores, 503 GB RAM — **shared with other users**
- Conda env `cogs118a`: cuML 26.06, xgboost 3.2.0, PyTorch 2.14.1, skorch 1.4.0, tabpfn 10.0.0, sklearn 1.9.1
- 216 independent jobs; launcher checks free GPU memory before placing each one; `n_jobs=1` inside each job
- Timing test (1 dataset, all models, 80/20, 1 trial) before the full sweep

## Bonus angles
- ≥5 classifiers, ≥4 datasets (named bonus)
- TabPFN as a modern twist; XGBoost vs paper's boosted trees
- Rank analysis (bootstrap ranks or Friedman/critical-difference diagram)
- Optional: calibration (AUC, Brier, reliability diagrams), accuracy vs compute time

# COGS 118A Final Project — Design Decisions

Replication + modernization of Caruana & Niculescu-Mizil (ICML 2006), "An Empirical Comparison of Supervised Learning Algorithms."

## Classifiers (6, all GPU-capable)
| # | Classifier | From paper? | Library | Notes |
|---|---|---|---|---|
| 1 | Random Forest | Yes (#2 in paper) | RAPIDS cuML | Anchor for grading point (a) |
| 2 | XGBoost | Modern version of paper's boosted trees | xgboost, `device="cuda", tree_method="hist"` | Replaced LightGBM (easier GPU) |
| 3 | SVM, RBF kernel | Yes (#4 calibrated) | cuML SVC | Replaced AdaBoost; cap train set ~20–50k on big datasets |
| 4 | ANN (MLP) | Yes | PyTorch via skorch | 1–3 hidden layers |
| 5 | KNN | Yes | cuML | |
| 6 | TabPFN | New (tabular foundation model) | tabpfn | No tuning; v2 ~10k train rows, newer versions need license token |

Dropped: AdaBoost (no GPU, overlaps XGBoost), LightGBM (GPU build fiddly), RealMLP/TabM (overkill).

## Datasets (4, binary)
- Adult (~48k, full)
- Letter (20k, full; binarize O-vs-rest or A–M vs N–Z)
- Covertype (largest class vs rest; 100k–full 581k pending timing test)
- HIGGS (subsample ~500k–1M) — the "large and challenging" dataset

## Protocol
- Partitions 20/80, 50/50, 80/20 × 3 random stratified trials
- Tuning: plain GridSearchCV (5-fold, `return_train_score=True`) on training split only. Bayesian HPO dropped as overkill.
- Report mean train / validation / test accuracy + best hyperparameters per cell
- Scale features for SVM, KNN, ANN (inside Pipeline); not for trees
- Document library differences (cuML vs sklearn) and any per-model subsampling (SVM, TabPFN)

## Infrastructure
- Private multi-GPU server, no queue
- 6 × 4 × 3 × 3 = 216 independent jobs; one job per GPU, `n_jobs=1` inside each; each job writes its own results file
- Run a timing test (1 dataset, all models, 80/20, 1 trial) before the full sweep

## Bonus angles
- ≥5 classifiers, ≥4 datasets (named bonus)
- TabPFN as a modern twist; XGBoost vs paper's boosted trees
- Rank analysis (bootstrap ranks or Friedman/critical-difference diagram)
- Optional: calibration (AUC, Brier, reliability diagrams), accuracy vs compute time

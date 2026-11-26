# COGS 118A Final Project — Design Decisions

Replication + modernization of Caruana & Niculescu-Mizil (ICML 2006), "An Empirical Comparison of Supervised Learning Algorithms."

**Due:** 11:59pm PST, Friday Dec 11, 2026.

**Last updated:** 2026-11-26 — every dataset capped at 12,500 rows (no per-model caps); tune in every trial; no rebalancing (natural class rates); grids proposed; NeurIPS LaTeX template; no TA check-in (see Change log).

## Classifiers (6, all GPU-capable)
| # | Classifier | From paper? | Library | Notes |
|---|---|---|---|---|
| 1 | Random Forest | Yes (#2 in paper) | RAPIDS cuML | Anchor for grading point (a); always set `max_depth` explicitly |
| 2 | XGBoost | Modern version of paper's boosted trees | xgboost, `device="cuda", tree_method="hist"` | Replaced LightGBM (easier GPU) |
| 3 | SVM, RBF kernel | Yes (#4 calibrated) | cuML SVC | Replaced AdaBoost; no training cap (largest split is 10k) |
| 4 | ANN (MLP) | Yes | PyTorch via skorch | 1–3 hidden layers; **early stopping; grid of 6–8 combos** |
| 5 | KNN | Yes | cuML (proposed: scikit-learn CPU, for the paper's distance weighting) | |
| 6 | TabPFN | New (tabular foundation model) | tabpfn | No tuning; v2 open weights; 10k training-row limit, which the 80/20 split meets exactly. The project's novelty |

Dropped: AdaBoost (no GPU, overlaps XGBoost), LightGBM (GPU build fiddly), RealMLP/TabM (overkill).

## Datasets (4, binary) — sizes updated 2026-11-26
Every dataset is a **12,500-row** stratified, fixed-seed subsample taken once, before any splits, so every classifier sees identical rows in every cell (as in the paper, where every model trained on the same 4k rows). Training sizes: 2,500 / 6,250 / 10,000 for 20/80, 50/50, 80/20.

**No dataset is rebalanced**: each keeps its natural class rate, as in the paper (decision #23). Adult (~24.8%) matches the paper's Adult (25%).

| Dataset | Source rows | Features | % positive (source) | Used |
|---|---|---|---|---|
| Adult | 45,222 (missing rows dropped) | 104 | 24.8% | 12,500, ~24.8% |
| Letter | 20,000 | 16 | 50.3% | 12,500, ~50.3% |
| Covertype | 581,012 | 54 | 48.8% | 12,500, ~48.8% |
| HIGGS | 11M (1M sample read 2026-11-16) | 28 | 53.0% | 12,500, ~53.0% |

- The paper didn't rebalance anything; it reported each dataset's % positive (%POZ) in its dataset table. Do the same, and show each dataset's majority-class baseline next to its accuracies (Adult ~75.2%, Letter ~50.3%, Covertype ~51.2%, HIGGS ~53.0%).
- Paper's Table 1 lists COV_TYPE at 36% positive while saying "largest class"; ours is 48.8% (class 2). Note it in the report.
- Test sets are 2,500–10,000 rows; the paper used large final test sets. State this.
- Not yet re-run at 12.5k (repo action item #29).

## Protocol
- Partitions 20/80, 50/50, 80/20 × 3 random stratified trials; splits pre-computed so every classifier sees identical rows.
- Tuning: plain `GridSearchCV` (5-fold, `return_train_score=True`) on the training split only, **in every trial** (as the paper did and the rubric asks: "each time, report the best accuracy under the chosen hyper-parameter"). Bayesian HPO dropped as overkill.
- Report mean train / validation / test accuracy over the 3 trials + best hyperparameters per trial. Save full `cv_results_` for accuracy-vs-hyperparameter curves/heatmaps (graded under experimental design), and the test confusion matrix per job.
- **Speed per job (added 2026-11-24):** tuning time, final fit time, prediction ms per 1k rows and throughput. Prediction timed on a fixed 2,500-row test sample (the smallest test split), 1 warm-up then median of 5 repeats, timer stops when predictions are back as NumPy. Log GPU name, free memory and utilization per job.
- Scale features for SVM, KNN, ANN (inside Pipeline); not for trees.
- No per-model data differences. Document library differences (cuML vs sklearn).
- Fallback if too slow after the timing test: 3-fold CV (document it).
- **Grids (proposed 2026-11-26, from paper §2.1; full table in repo `docs/03`):** RF 1024 trees × max_features {1,2,4,6,8,12,16,20}; XGBoost n_estimators {32,128,512,2048} × max_depth {2,4,6,8}, lr 0.1; SVM gamma {0.001,0.005,0.01,0.05,0.1,0.5,1,2} × C {10⁻²…10³}; MLP {(32),(128),(128,128),(128,128,128)} × momentum {0.5,0.9}, early stopping; KNN K {1,3,5,9,15,25,45,75,125,201,351,501} × {uniform, distance}.
- **Report:** NeurIPS LaTeX template.

## Infrastructure — updated 2026-11-24
- **yggdrasil (lab server) is not available.**
- **Personal GTX 1080 Ti ruled out:** Pascal GPU; CUDA 13 and RAPIDS 24.02+ require newer cards.
- Platforms (any GPU must be Turing / T4 or newer so the env works unchanged):
  1. **UCSD DataHub / DSMLP**: first choice, access pending (email datahub@ucsd.edu). Request a Turing+ GPU type with `-v`; the pool also has 1080 Tis.
  2. **Kaggle**: backup; 2× T4 16 GB, ~30 GPU-h/week, 12-h sessions; run as batch "Save version" jobs.
  3. **Google Colab (free T4)**: overflow only; limits not guaranteed.
- Conda env `cogs118a`: cuML 26.06, xgboost 3.2.0, PyTorch 2.14.1, skorch 1.4.0, tabpfn 10.0.0, sklearn 1.9.1 (CUDA 13.0). ⚠️ Built on yggdrasil's driver 580; check the new platform's driver supports CUDA 13, else use CUDA 12 builds.
- 216 result cells (4 datasets × 3 partitions × 3 trials × 6 models). Jobs are independent. **Runner must be resumable**: each job writes its own results file and skips itself if that file exists.
- Timing test first on the worst case: HIGGS 12.5k, all models, 80/20, 1 trial. Multiply each model's time by 36 for the sweep estimate. Expect far less than the earlier 5–10 T4 GPU-hours estimate (that was for 100k rows); confirm with the timing test.

## Bonus angles
- ≥5 classifiers, ≥4 datasets (named bonus)
- TabPFN as a modern twist; XGBoost vs paper's boosted trees
- Rank analysis (bootstrap ranks or Friedman/critical-difference diagram)
- Accuracy vs. speed (paper had no speed metric) — **only compare timings from jobs that ran on the same GPU type**
- Optional: calibration (AUC, Brier, reliability diagrams)
- **Open (repo #34):** rubric item 1 gives 10 pts for "how challenging and large are the datasets". A separate, clearly labelled scaling run on larger Covertype/HIGGS samples could cover it without changing the main study.

## Change log
- **2026-11-26 (rates, grids):** Balancing switched off; all datasets at natural class rates (decision #23). Grids proposed from the paper (#24). NeurIPS LaTeX template chosen.
- **2026-11-26:** Every dataset capped at 12,500 rows; SVM cap dropped; TabPFN fits at 80/20 (decision #19). Tune in every trial, replacing trial-1-only tuning (#20). Adult left unbalanced, like the paper (#21). TA check-in dropped; TabPFN kept as the novelty (#22). Due date recorded.
- **2026-11-24 (balancing):** All datasets balanced 50/50 by undersampling (decision #18). Adult shrinks to ~22.4k and no longer matches the paper's unbalanced Adult.
- **2026-11-24 (sync):** Repo `docs/` and `src/config.py` brought in line with the replan below (decisions #13–17 in `05_decisions.md`; config: Covertype/HIGGS 100k, SVM cap 10k, `TUNE_TRIAL = 0`).
- **2026-11-24 (speed):** Speed metrics added: tune / fit / predict time + throughput per job, fixed 10k-row timing sample, GPU name logged (decision #12).
- **2026-11-24:** Lab server lost; moved to DSMLP / Kaggle / Colab (Turing+ GPUs). Covertype and HIGGS capped at 100k rows. Large-data novelty pitch dropped so every classifier is compared on the same data. SVM cap 50k → 10k. Tune on trial 1 only. MLP early stopping + smaller grid. Resumable runner.
- **2026-11-16:** Datasets locked; TabPFN v2 with 10k cap; GPU smoke test passed on yggdrasil.

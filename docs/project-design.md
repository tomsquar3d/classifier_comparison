# COGS 118A Final Project — Design Decisions

Replication + modernization of Caruana & Niculescu-Mizil (ICML 2006), "An Empirical Comparison of Supervised Learning Algorithms."

**Last updated:** 2026-11-24 — lab server lost; datasets capped at 100k rows; tuning and compute cuts; speed metrics added; repo docs synced (see Change log).

## Classifiers (6, all GPU-capable)
| # | Classifier | From paper? | Library | Notes |
|---|---|---|---|---|
| 1 | Random Forest | Yes (#2 in paper) | RAPIDS cuML | Anchor for grading point (a); always set `max_depth` explicitly |
| 2 | XGBoost | Modern version of paper's boosted trees | xgboost, `device="cuda", tree_method="hist"` | Replaced LightGBM (easier GPU) |
| 3 | SVM, RBF kernel | Yes (#4 calibrated) | cuML SVC | Replaced AdaBoost; **10k train-row cap** (was provisional 50k) |
| 4 | ANN (MLP) | Yes | PyTorch via skorch | 1–3 hidden layers; **early stopping; grid of 6–8 combos** |
| 5 | KNN | Yes | cuML | |
| 6 | TabPFN | New (tabular foundation model) | tabpfn | No tuning; v2 open weights, 10k train-row cap (decided 2026-11-16) |

Dropped: AdaBoost (no GPU, overlaps XGBoost), LightGBM (GPU build fiddly), RealMLP/TabM (overkill).

## Datasets (4, binary) — sizes updated 2026-11-24
All datasets balanced to **50/50** (added 2026-11-24): the larger class is randomly undersampled (fixed seed) after cleaning and before the 100k cap. % positive below is before balancing.

| Dataset | Rows used | Features | % positive (pre-balancing) | Rule |
|---|---|---|---|---|
| Adult | ~22.4k after balancing (45,222 before) | 104 | 24.8% | >50K = 1; missing rows dropped |
| Letter | ~19.9k after balancing (20,000 before) | 16 | 50.3% | A–M = 0, N–Z = 1 |
| Covertype | **100,000** (stratified subsample of 581,012) | 54 | 48.8% (full data) | Largest class vs rest |
| HIGGS | **100,000** (stratified subsample) | 28 | 53.0% (1M sample) | Signal vs background |

- Covertype and HIGGS are subsampled **once**, stratified, with a fixed seed, before any splits. Every classifier uses the same 100k rows. Not yet re-run at 100k (repo action item #29).
- Largest training set is 80k rows (80/20), still 20× the paper's 4k.
- **No large-data novelty pitch.** All classifiers are compared on the same data sizes; no full-data side runs.

## Protocol
- Partitions 20/80, 50/50, 80/20 × 3 random stratified trials; splits pre-computed so every classifier sees identical rows.
- Tuning: plain `GridSearchCV` (5-fold, `return_train_score=True`) on the training split only. Bayesian HPO dropped as overkill.
- **Tune once per dataset × partition × model, on trial 1.** Trials 2–3 refit those best settings on their own training split and score on their own test split. Trials measure split-to-split variation, not tuning variation. State this in the report.
- Report mean train / validation / test accuracy + best hyperparameters per cell (validation accuracy comes from the trial-1 CV).
- **Speed per job (added 2026-11-24):** tuning time (trial 1 only, since that's where tuning happens), final fit time, prediction ms per 1k rows and throughput. Prediction timed on a fixed 10k-row test sample, 1 warm-up then median of 5 repeats, timer stops when predictions are back as NumPy. Log GPU name, free memory and utilization per job.
- Scale features for SVM, KNN, ANN (inside Pipeline); not for trees.
- **Per-model exceptions (the only places classifiers see different data):** SVM and TabPFN train on a stratified 10k-row subsample of the training split when it is larger than 10k. Same test split as everyone else. TabPFN's cap is a hard model limit; SVM's is for runtime. Document both, plus library differences (cuML vs sklearn).
- Fallback if still too slow after the timing test: 3-fold CV (document it).

## Infrastructure — updated 2026-11-24
- **yggdrasil (lab server) is not available.**
- **Personal GTX 1080 Ti ruled out:** Pascal GPU; CUDA 13 and RAPIDS 24.02+ require newer cards.
- Platforms (any GPU must be Turing / T4 or newer so the env works unchanged):
  1. **UCSD DataHub / DSMLP**: first choice, access pending (email datahub@ucsd.edu). Request a Turing+ GPU type with `-v`; the pool also has 1080 Tis.
  2. **Kaggle**: backup; 2× T4 16 GB, ~30 GPU-h/week, 12-h sessions; run as batch "Save version" jobs.
  3. **Google Colab (free T4)**: overflow only; limits not guaranteed.
- Conda env `cogs118a`: cuML 26.06, xgboost 3.2.0, PyTorch 2.14.1, skorch 1.4.0, tabpfn 10.0.0, sklearn 1.9.1 (CUDA 13.0). ⚠️ Built on yggdrasil's driver 580; check the new platform's driver supports CUDA 13, else use CUDA 12 builds.
- 216 result cells (4 datasets × 3 partitions × 3 trials × 6 models). **Runner must be resumable**: each job writes its own results file and skips itself if that file exists, so a cut-off session loses one job, not the sweep.
- Timing test first on the worst case: HIGGS 100k, all models, 80/20, 1 trial. Multiply each model's time by 36 for the sweep estimate.
- Expected total: roughly **5–10 T4 GPU-hours** (estimate; confirm with the timing test).

## Bonus angles
- ≥5 classifiers, ≥4 datasets (named bonus)
- TabPFN as a modern twist; XGBoost vs paper's boosted trees
- Rank analysis (bootstrap ranks or Friedman/critical-difference diagram)
- Accuracy vs. speed (paper had no speed metric) — **only compare timings from jobs that ran on the same GPU type**; if the sweep is split across platforms, re-time on one GPU type in a short timing-only pass
- Optional: calibration (AUC, Brier, reliability diagrams)

## Change log
- **2026-11-24 (balancing):** All datasets balanced 50/50 by undersampling (decision #18). Adult shrinks to ~22.4k and no longer matches the paper's unbalanced Adult.
- **2026-11-24 (sync):** Repo `docs/` and `src/config.py` brought in line with the replan below (decisions #13–17 in `05_decisions.md`; config: Covertype/HIGGS 100k, SVM cap 10k, `TUNE_TRIAL = 0`).
- **2026-11-24 (speed):** Speed metrics added: tune / fit / predict time + throughput per job, fixed 10k-row timing sample, GPU name logged (decision #12).
- **2026-11-24:** Lab server lost; moved to DSMLP / Kaggle / Colab (Turing+ GPUs). Covertype and HIGGS capped at 100k rows. Large-data novelty pitch dropped so every classifier is compared on the same data. SVM cap 50k → 10k. Tune on trial 1 only. MLP early stopping + smaller grid. Resumable runner.
- **2026-11-16:** Datasets locked; TabPFN v2 with 10k cap; GPU smoke test passed on yggdrasil.

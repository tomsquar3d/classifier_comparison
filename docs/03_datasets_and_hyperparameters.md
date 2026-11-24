# Datasets & Hyperparameter Reference
**Project:** COGS 118A Classifier Comparison
**Version:** v2 — datasets locked; Covertype/HIGGS capped at 100k; SVM cap 10k; grids not final
**Last Updated:** 2026-11-24

---

## 🔵 STATUS — Lineup and sizes set; grids not final

Dataset lineup and Letter rule locked 2026-11-16 (`05` #7–8). Covertype and HIGGS capped at 100k rows and SVM cap set to 10k on 2026-11-24 (`05` #14–15); data prep not yet re-run at those sizes. Grids below are starting points, not final.

---

## Datasets (locked 2026-11-16)

| Dataset | Source | Positive class rule | Rows used | Features | % positive | Planned size |
|---------|--------|---------------------|-----------|----------|------------|--------------|
| Adult | UCI (`adult.data` + `adult.test` combined, 48,842 raw) | Income >50K | 45,222 (rows with missing values dropped) | 104 (after one-hot) | 24.8% | Whole dataset |
| Letter | UCI | A–M = 0, N–Z = 1 (paper's LETTER.p2, balanced) | 20,000 | 16 | 50.3% | Whole dataset |
| Covertype | UCI | Largest class (class 2, Lodgepole Pine) = 1, rest = 0 (paper's rule) | 581,012 full (2026-11-16) | 54 | 48.8% (full) | **100,000**, stratified subsample (2026-11-24) |
| HIGGS | UCI (11M raw) | Signal = 1, background = 0 | 1,000,000 subsample (2026-11-16) | 28 | 53.0% (1M) | **100,000**, stratified subsample (2026-11-24) |

> **Note:** Row counts are from `data/processed/dataset_summary.json`, produced by `src/prepare_data.py` on yggdrasil 2026-11-16. Change sizes in `src/config.py` (`N_ROWS`) and re-run the script.
>
> **100k subsample (2026-11-24):** taken once per dataset, stratified on the label, fixed seed (`BASE_SEED`), before any splits — so every classifier uses the same 100k rows. HIGGS: the script first reads a random ~2M rows of the 11M file (`HIGGS_RAW_SAMPLE`), then takes the stratified 100k from those. The largest training set is now 80k rows (80/20), still 20× the paper's 4k.

The Adult, Covertype and Letter binarization rules come from the paper (§2.5).

### ⚠️ Per-model training caps

Applied to the training split only, by random subsample. Test set is never capped, so all models are scored on the same test data. Report every cap in the methods section.

| Model | Cap | Why |
|-------|-----|-----|
| SVM (RBF) | **10k rows** (2026-11-24; was provisional 50k) | Runtime — training cost grows much faster than linear with rows |
| TabPFN | 10k rows (v2 weights, decided 2026-11-16) | Hard model limit |

SVM and TabPFN are the only places classifiers see different training data (a stratified 10k subsample of the training split when it's larger than 10k). Same test split as everyone else.

### Speed-timing sample — 🟣 [NEW — 2026-11-24]

Prediction speed is timed on a random 10,000-row sample of each test split (`TIMING_N_ROWS`), drawn with a fixed seed so every model is timed on the same rows. 5 timed repeats, median reported, 1 untimed warm-up first. Settings in `src/config.py`; method in `01` Step 2.

### Preprocessing

- Adult: drop rows with any missing value (`?`), then one-hot encode categorical features.
- Standardize features (mean 0, std 1) for SVM, KNN, ANN — fit on training folds only, inside the Pipeline. No scaling for RF, XGBoost (matches paper §2.1).

---

## Hyperparameter grids (starting points, not final)

Tuning runs on trial 1 only; trials 2–3 reuse the trial-1 best settings (`05` #16). 5-fold CV, with 3-fold as the fallback if the timing test says it's too slow.

| Classifier | Hyperparameters | Values |
|------------|-----------------|--------|
| Random forest | `max_features` | 1, 2, 4, 6, 8, 12, 16, 20 (paper) |
| | `n_estimators` | ~500 trees, fixed |
| XGBoost | learning rate, max depth (or leaves), n_estimators | 2–3 knobs only, values ⛔ NOT YET RECORDED; rest at defaults |
| SVM (RBF) | C | 10⁻², 10⁻¹, …, 10³ |
| | gamma | 10⁻³, 10⁻², 10⁻¹, 1 |
| ANN (MLP) | hidden layers / units, learning rate | Up to 3 hidden layers; **early stopping on; 6–8 combos total** (2026-11-24); exact values ⛔ NOT YET RECORDED |
| KNN | K | 1 to ~100; exact list ⛔ NOT YET RECORDED |
| | weighting | uniform, distance |
| TabPFN | — | No tuning |

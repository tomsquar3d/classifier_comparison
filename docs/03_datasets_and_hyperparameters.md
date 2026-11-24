# Datasets & Hyperparameter Reference
**Project:** COGS 118A Classifier Comparison
**Version:** v1 — datasets locked; Covertype/HIGGS sizes and grids not final
**Last Updated:** 2026-11-24

---

## 🔵 STATUS — Lineup locked; sizes and grids not final

Dataset lineup and Letter rule locked 2026-11-16 (`05` #7–8). Covertype and HIGGS sizes wait on the timing test. Grids below are starting points from the paper / discussion, not final.

---

## Datasets (locked 2026-11-16)

| Dataset | Source | Positive class rule | Rows used | Features | % positive | Planned size |
|---------|--------|---------------------|-----------|----------|------------|--------------|
| Adult | UCI (`adult.data` + `adult.test` combined, 48,842 raw) | Income >50K | 45,222 (rows with missing values dropped) | 104 (after one-hot) | 24.8% | Whole dataset |
| Letter | UCI | A–M = 0, N–Z = 1 (paper's LETTER.p2, balanced) | 20,000 | 16 | 50.3% | Whole dataset |
| Covertype | UCI | Largest class (class 2, Lodgepole Pine) = 1, rest = 0 (paper's rule) | 581,012 (full) | 54 | 48.8% | Full for now; may drop to ~100k–250k after timing test |
| HIGGS | UCI (11M raw) | Signal = 1, background = 0 | 1,000,000 (random subsample) | 28 | 53.0% | 1M for now; may drop to ~500k after timing test |

> **Note:** Row counts are from `data/processed/dataset_summary.json`, produced by `src/prepare_data.py` on the server 2026-11-16. Change sizes in `src/config.py` (`N_ROWS`) and re-run the script.

The Adult, Covertype and Letter binarization rules come from the paper (§2.5).

### ⚠️ Per-model training caps

Applied to the training split only, by random subsample. Test set is never capped, so all models are scored on the same test data. Report every cap in the methods section.

| Model | Cap | Why |
|-------|-----|-----|
| SVM (RBF) | 50k rows, **provisional** (`TRAIN_CAP` in `config.py`); final value after timing test | Training cost grows much faster than linear with rows |
| TabPFN | 10k rows (v2 weights, decided 2026-11-16) | Model's built-in size limit |

### Speed-timing sample — 🟣 [NEW — 2026-11-24]

Prediction speed is timed on a random 10,000-row sample of each test split (`TIMING_N_ROWS`), drawn with a fixed seed so every model is timed on the same rows. 5 timed repeats, median reported, 1 untimed warm-up first. Settings in `src/config.py`; method in `01` Step 2.

### Preprocessing

- Adult: drop rows with any missing value (`?`), then one-hot encode categorical features.
- Standardize features (mean 0, std 1) for SVM, KNN, ANN — fit on training folds only, inside the Pipeline. No scaling for RF, XGBoost (matches paper §2.1).

---

## Hyperparameter grids (starting points, not final)

| Classifier | Hyperparameters | Values |
|------------|-----------------|--------|
| Random forest | `max_features` | 1, 2, 4, 6, 8, 12, 16, 20 (paper) |
| | `n_estimators` | ~500 trees, fixed |
| XGBoost | learning rate, max depth (or leaves), n_estimators | 2–3 knobs only, values ⛔ NOT YET RECORDED; rest at defaults |
| SVM (RBF) | C | 10⁻², 10⁻¹, …, 10³ |
| | gamma | 10⁻³, 10⁻², 10⁻¹, 1 |
| ANN (MLP) | hidden layers / units, learning rate | Up to 3 hidden layers; values ⛔ NOT YET RECORDED |
| KNN | K | 1 to ~100; exact list ⛔ NOT YET RECORDED |
| | weighting | uniform, distance |
| TabPFN | — | No tuning |

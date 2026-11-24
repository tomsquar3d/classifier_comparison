# Datasets & Hyperparameter Reference
**Project:** COGS 118A Classifier Comparison
**Version:** v2 — sizes and caps locked; grids not final
**Last Updated:** 2026-11-24

---

## 🔵 STATUS — Lineup, sizes and caps locked; grids not final

Dataset lineup and Letter rule locked 2026-11-16 (`05` #7–8). Covertype and HIGGS capped at 100k and SVM cap set to 10k on 2026-11-24 (`05` #13–14). `prepare_data.py` must be re-run with the new sizes. Grids below are starting points, not final.

---

## Datasets (locked 2026-11-16)

| Dataset | Source | Positive class rule | Rows used | Features | % positive | Planned size |
|---------|--------|---------------------|-----------|----------|------------|--------------|
| Adult | UCI (`adult.data` + `adult.test` combined, 48,842 raw) | Income >50K | 45,222 (rows with missing values dropped) | 104 (after one-hot) | 24.8% | Whole dataset |
| Letter | UCI | A–M = 0, N–Z = 1 (paper's LETTER.p2, balanced) | 20,000 | 16 | 50.3% | Whole dataset |
| Covertype | UCI | Largest class (class 2, Lodgepole Pine) = 1, rest = 0 (paper's rule) | 100,000 (stratified subsample of 581,012; set 2026-11-24, not yet re-run) | 54 | 48.8% | 100k |
| HIGGS | UCI (11M raw) | Signal = 1, background = 0 | 100,000 (stratified subsample; set 2026-11-24, not yet re-run) | 28 | 53.0% | 100k |

> **Note:** Adult and Letter counts are from `data/processed/dataset_summary.json` (server run 2026-11-16). Covertype and HIGGS counts are the new `N_ROWS` targets in `src/config.py` (2026-11-24); confirm % positive after re-running `prepare_data.py` on the new platform. The 2026-11-16 run had Covertype 581,012 and HIGGS 1,000,000 rows.

The Adult, Covertype and Letter binarization rules come from the paper (§2.5).

### ⚠️ Per-model training caps

Applied to the training split only, by random subsample. Test set is never capped, so all models are scored on the same test data. Report every cap in the methods section.

| Model | Cap | Why |
|-------|-----|-----|
| SVM (RBF) | 10k rows (`TRAIN_CAP` in `config.py`; was provisional 50k, changed 2026-11-24, `05` #14) | Training cost grows much faster than linear with rows; 50k would eat most of the T4 budget |
| TabPFN | 10k rows (v2 weights, decided 2026-11-16) | Model's built-in size limit |

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
| ANN (MLP) | hidden layers / units, learning rate | Up to 3 hidden layers; grid of 6–8 combos with early stopping (`05` #16); values ⛔ NOT YET RECORDED |
| KNN | K | 1 to ~100; exact list ⛔ NOT YET RECORDED |
| | weighting | uniform, distance |
| TabPFN | — | No tuning |

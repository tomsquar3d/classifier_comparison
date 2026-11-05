# Datasets & Hyperparameter Reference
**Project:** COGS 118A Classifier Comparison
**Version:** v0 — proposed, not finalized
**Last Updated:** 2026-11-04

---

## 🔵 STATUS — Proposed lineup; sizes and grids not final

Dataset choice is proposed but not confirmed. Sizes wait on the timing test. Grids below are starting points from the paper / discussion, not final.

---

## Datasets (proposed)

| Dataset | Source | Positive class rule | Full size | Planned size |
|---------|--------|---------------------|-----------|--------------|
| Adult | UCI | Income >50K (binary already) | ~48k rows (to verify) | Whole dataset |
| Letter | UCI | Paper option p1: "O" vs. rest (very imbalanced) / p2: A–M vs. N–Z (balanced) — pick one | ~20k rows (to verify) | Whole dataset |
| Covertype | UCI | Largest class = positive, rest negative (paper's rule) | ~581k rows (to verify) | ~100k up to full — pending timing test |
| HIGGS | UCI | Signal vs. background (binary already) | ~11M rows (to verify) | ~500k–1M — pending timing test |

> **Note:** Full sizes are approximate figures from discussion, not yet checked against the UCI pages.

The Adult, Covertype and Letter binarization rules come from the paper (§2.5).

### ⚠️ Per-model training caps

Applied to the training split only, by random subsample. Test set is never capped, so all models are scored on the same test data. Report every cap in the methods section.

| Model | Cap | Why |
|-------|-----|-----|
| SVM (RBF) | ~20k–50k rows on large datasets (exact ⛔ NOT YET RECORDED) | Training cost grows much faster than linear with rows |
| TabPFN | Depends on version: ~10k rows for v2 (exact ⛔ NOT YET RECORDED) | Model's built-in size limit |

### Preprocessing

- One-hot encode categorical features (Adult has them).
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

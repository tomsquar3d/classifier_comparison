# Datasets & Hyperparameter Reference
**Project:** COGS 118A Classifier Comparison
**Version:** v4 — every dataset 12,500 rows; Adult unbalanced, others 50/50; no per-model caps; grids not final
**Last Updated:** 2026-11-26

---

## 🔵 STATUS — Lineup and sizes set; grids not final

Dataset lineup and Letter rule locked 2026-11-16 (`05` #7–8). On 2026-11-26 every dataset was capped at 12,500 rows, per-model caps dropped and Adult left unbalanced (`05` #19, #21). Data prep not yet re-run. Grids below are starting points, not final.

---

## Datasets (locked 2026-11-16)

| Dataset | Source | Positive class rule | Rows after cleaning (2026-11-16) | Features | % positive (source data) | Used: rows / % positive |
|---------|--------|---------------------|-----------|----------|------------|--------------|
| Adult | UCI (`adult.data` + `adult.test` combined, 48,842 raw) | Income >50K | 45,222 (rows with missing values dropped) | 104 (after one-hot) | 24.8% | **12,500 / ~24.8% (not balanced)** |
| Letter | UCI | A–M = 0, N–Z = 1 (paper's LETTER.p2) | 20,000 | 16 | 50.3% | **12,500 / 50%** |
| Covertype | UCI | Largest class (class 2, Lodgepole Pine) = 1, rest = 0 (paper's rule) | 581,012 | 54 | 48.8% | **12,500 / 50%** |
| HIGGS | UCI (11M raw) | Signal = 1, background = 0 | 1,000,000 subsample (2026-11-16) | 28 | 53.0% (1M) | **12,500 / 50%** |

> **Note:** Source-data counts are from `data/processed/dataset_summary.json`, produced by `src/prepare_data.py` on yggdrasil 2026-11-16. Final counts fill in after the re-run (action item #29). Change sizes in `src/config.py` (`N_ROWS`) and re-run the script.
>
> **12,500-row subsample (2026-11-26, `05` #19):** taken once per dataset, stratified on the label, fixed seed (`BASE_SEED`), after balancing and before any splits — so every classifier uses the same rows. Training sizes: 2,500 / 6,250 / 10,000 (20/80, 50/50, 80/20); the paper trained on 4,000. HIGGS: the script reads a random ~200k rows of the 11M file (`HIGGS_RAW_SAMPLE`) first.
>
> **Paper comparison:** the paper's Table 1 lists Adult at 25% positive and LETTER.p2 at 53%. It lists COV_TYPE at 36% positive even though the text says "largest class"; ours comes out 48.8% (class 2). Likely a different subset or class — note it in the report.

The Adult, Covertype and Letter binarization rules come from the paper (§2.5).

### ⚠️ Class balancing — 🟠 [UPDATED — 2026-11-26]

Letter, Covertype and HIGGS are made exactly 50/50 by randomly dropping rows of the larger class (fixed seed), after cleaning and **before** the 12.5k cap (`05` #18). **Adult is left unbalanced** at its natural ~24.8% positive (`05` #21), matching the paper's Adult (25%). All splits are stratified, so every split keeps its dataset's rate. Switch: `BALANCE_CLASSES` dict in `src/config.py`.

The paper did not rebalance any dataset; it reported each dataset's % positive (%POZ) in its dataset table. Consequences to state in the report:
- Include a "% positive" column in the dataset table, like the paper.
- Adult's accuracy baseline is ~75% (always predict ≤50K), not 50%. Show the majority-class baseline next to Adult accuracies.
- Letter, Covertype and HIGGS were already near 50/50, so balancing barely changes them.

### Per-model training caps — removed 2026-11-26

None. The largest training split is 10,000 rows, which fits TabPFN v2's limit, and SVM no longer needs a runtime cap (`05` #19). `TRAIN_CAP` in `src/config.py` keeps TabPFN's 10k as a guard; the experiment script should stop with an error if a training split ever exceeds it, not subsample silently.

### Speed-timing sample — 🟣 [NEW — 2026-11-24]

Prediction speed is timed on a random 2,500-row sample of each test split (= the smallest test split) (`TIMING_N_ROWS`), drawn with a fixed seed so every model is timed on the same rows. 5 timed repeats, median reported, 1 untimed warm-up first. Settings in `src/config.py`; method in `01` Step 2.

### Preprocessing

- Adult: drop rows with any missing value (`?`), then one-hot encode categorical features.
- Standardize features (mean 0, std 1) for SVM, KNN, ANN — fit on training folds only, inside the Pipeline. No scaling for RF, XGBoost (matches paper §2.1).

---

## Hyperparameter grids (starting points, not final)

Tuning runs in every trial, on that trial's own training split (`05` #20). 5-fold CV, with 3-fold as the fallback if the timing test says it's too slow.

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

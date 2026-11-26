# Datasets & Hyperparameter Reference
**Project:** COGS 118A Classifier Comparison
**Version:** v6 — every dataset 12,500 rows at natural class rates; no per-model caps; grids approved
**Last Updated:** 2026-11-26

---

## 🔵 STATUS — Lineup and sizes set; grids not final

Dataset lineup and Letter rule locked 2026-11-16 (`05` #7–8). On 2026-11-26 every dataset was capped at 12,500 rows, per-model caps dropped and balancing switched off for all four (`05` #19, #21, #23). Grids approved (`05` #24). Data prepped on DSMLP 2026-11-26 (`02` §1).

---

## Datasets (locked 2026-11-16)

| Dataset | Source | Positive class rule | Rows after cleaning (2026-11-16) | Features | % positive (source data) | Used: rows / % positive |
|---------|--------|---------------------|-----------|----------|------------|--------------|
| Adult | UCI (`adult.data` + `adult.test` combined, 48,842 raw) | Income >50K | 45,222 (rows with missing values dropped) | 104 (after one-hot) | 24.8% | **12,500 / ~24.8% (not balanced)** |
| Letter | UCI | A–M = 0, N–Z = 1 (paper's LETTER.p2) | 20,000 | 16 | 50.3% | **12,500 / ~50.3%** |
| Covertype | UCI | Largest class (class 2, Lodgepole Pine) = 1, rest = 0 (paper's rule) | 581,012 | 54 | 48.8% | **12,500 / ~48.8%** |
| HIGGS | UCI (11M raw) | Signal = 1, background = 0 | 1,000,000 subsample (2026-11-16) | 28 | 53.0% (1M) | **12,500 / ~53.0%** |

> **Note:** Source-data counts are from `data/processed/dataset_summary.json`, produced by `src/prepare_data.py` on yggdrasil 2026-11-16. Final counts fill in after the re-run (action item #29). Change sizes in `src/config.py` (`N_ROWS`) and re-run the script.
>
> **12,500-row subsample (2026-11-26, `05` #19):** taken once per dataset, stratified on the label, fixed seed (`BASE_SEED`), before any splits — so every classifier uses the same rows. Training sizes: 2,500 / 6,250 / 10,000 (20/80, 50/50, 80/20); the paper trained on 4,000. HIGGS: the script reads a random ~200k rows of the 11M file (`HIGGS_RAW_SAMPLE`) first.
>
> **Paper comparison:** the paper's Table 1 lists Adult at 25% positive and LETTER.p2 at 53%. It lists COV_TYPE at 36% positive even though the text says "largest class"; ours comes out 48.8% (class 2). Likely a different subset or class — note it in the report.

The Adult, Covertype and Letter binarization rules come from the paper (§2.5).

### Class rates — 🟠 [UPDATED — 2026-11-26]

**No dataset is rebalanced** (`05` #23, reversing #18). Every dataset keeps its natural class rate; the subsample and all splits are stratified, so every split keeps its dataset's rate. The `balance()` code is still in `prepare_data.py`, switched off via `BALANCE_CLASSES` in `src/config.py`.

The paper did the same: no rebalancing, with each dataset's % positive (%POZ) reported in its dataset table. For the report:
- Include a "% positive" column in the dataset table, like the paper.
- Show each dataset's majority-class baseline next to its accuracies: Adult ~75.2%, Letter ~50.3%, Covertype ~51.2%, HIGGS ~53.0% (`dataset_summary.json` records the exact value).

### Per-model training caps — removed 2026-11-26

None. The largest training split is 10,000 rows, which fits TabPFN v2's limit, and SVM no longer needs a runtime cap (`05` #19). `TRAIN_CAP` in `src/config.py` keeps TabPFN's 10k as a guard; the experiment script should stop with an error if a training split ever exceeds it, not subsample silently.

### Speed-timing sample — 🟣 [NEW — 2026-11-24]

Prediction speed is timed on a random 2,500-row sample of each test split (= the smallest test split) (`TIMING_N_ROWS`), drawn with a fixed seed so every model is timed on the same rows. 5 timed repeats, median reported, 1 untimed warm-up first. Settings in `src/config.py`; method in `01` Step 2.

### Preprocessing

- Adult: drop rows with any missing value (`?`), then one-hot encode categorical features.
- Standardize features (mean 0, std 1) for SVM, KNN, ANN — fit on training folds only, inside the Pipeline. No scaling for RF, XGBoost (matches paper §2.1).

---

## Hyperparameter grids — approved 2026-11-26 (`05` #24); in `src/config.py` `GRIDS`

Tuning runs in every trial, on that trial's own training split (`05` #20), 5-fold CV (3-fold fallback if the timing test says it's too slow). Each grid starts from the paper's §2.1 settings, trimmed where the paper's range is wasteful at 12.5k rows.

| Classifier | Paper (§2.1) | Proposed grid | Combos |
|------------|--------------|---------------|--------|
| Random forest | 1024 trees; features per split 1, 2, 4, 6, 8, 12, 16, 20 | `n_estimators` = 1024 (fixed); `max_features` ∈ {1, 2, 4, 6, 8, 12, 16, 20}, dropping values above the feature count (Letter has 16); `max_depth` fixed high (e.g. 32) so trees are near-unpruned like Breiman's | 8 (7 on Letter) |
| XGBoost | Boosted trees after 2, 4, …, 2048 steps | `n_estimators` ∈ {32, 128, 512, 2048} × `max_depth` ∈ {2, 4, 6, 8}; `learning_rate` = 0.1 fixed | 16 |
| SVM (RBF) | RBF width {0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2}; C from 10⁻⁷ to 10³ by ×10 (plus linear/poly kernels) | `gamma` ∈ the paper's 8 widths × `C` ∈ {10⁻², 10⁻¹, …, 10³}. RBF only (the assignment counts other kernels as the same classifier); C below 10⁻² dropped because it only underfits | 48 |
| ANN (MLP) | 1 hidden layer, {1, 2, 4, 8, 32, 128} units; momentum {0, 0.2, 0.5, 0.9}; early stopping on validation | Hidden layers ∈ {(32), (128), (128, 128), (128, 128, 128)} × SGD momentum ∈ {0.5, 0.9}; learning rate 0.01; early stopping on a 10% slice of the training fold (patience 10, max 200 epochs). Deeper nets follow the assignment's tip to try up to 3 layers | 8 |
| KNN | 26 values of K from 1 to the training-set size; Euclidean; also distance-weighted | `n_neighbors` ∈ {1, 3, 5, 9, 15, 25, 45, 75, 125, 201, 351, 501} × `weights` ∈ {uniform, distance}; Euclidean. **Proposed switch to scikit-learn (CPU)**: cuML's KNN classifier only supports uniform weights, and sklearn is fast at this size | 24 |
| TabPFN | — (not in paper) | No tuning; still run through the same 5-fold CV so it gets a validation accuracy | 1 |

Features scaled to mean 0, std 1 for SVM, ANN and KNN, as in the paper.

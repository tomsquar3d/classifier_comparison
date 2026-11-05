# Action Items — COGS 118A Classifier Comparison
**Last Updated:** 2026-11-04

## Open
| # | Added | Item | Blocking? | Notes |
|---|-------|------|-----------|-------|
| 1 | 2026-11-04 | Confirm dataset lineup (Adult, Letter, Covertype, HIGGS) | Yes — data prep | Proposed in planning, not confirmed |
| 2 | 2026-11-04 | Pick Letter binarization: "O" vs. rest (imbalanced) or A–M vs. N–Z (balanced) | Yes — data prep | 03 |
| 3 | 2026-11-04 | Verify full dataset sizes on UCI pages | No | 03 has ~ values from discussion |
| 4 | 2026-11-04 | Ask instructor/TA: is TabPFN OK as an added classifier? | No, but ask early | 04 open question 1 |
| 5 | 2026-11-04 | Ask instructor/TA: are documented per-model training caps OK? | No | 04 open question 2 |
| 6 | 2026-11-04 | Record GPU server specs (GPU model/count, CPU, RAM) | No | 01 header |
| 7 | 2026-11-04 | Set up Python env; record library versions (xgboost, cuML, PyTorch, skorch, tabpfn, scikit-learn) | Yes — all runs | 04 Software |
| 8 | 2026-11-04 | Decide TabPFN version (v2 open weights vs. newer licensed version + token) | Yes — TabPFN runs | Sets its row cap in 03 |
| 9 | 2026-11-04 | Write data prep script (01 Step 1) | Yes | |
| 10 | 2026-11-04 | Write per-job experiment script (01 Step 2) | Yes | |
| 11 | 2026-11-04 | Write GPU launcher with resume (01 Step 3) | Yes | |
| 12 | 2026-11-04 | Run timing test; record in 02 §2 | Yes — sets sizes/caps | 01 Step 4 |
| 13 | 2026-11-04 | Finalize Covertype and HIGGS sizes | Yes — full sweep | After #12 |
| 14 | 2026-11-04 | Finalize SVM training cap | Yes — full sweep | After #12 |
| 15 | 2026-11-04 | Finalize grids for XGBoost, ANN, KNN | Yes — full sweep | 03 |
| 16 | 2026-11-04 | Run full 216-job sweep | Yes — results | |
| 17 | 2026-11-04 | Aggregate results; fill 02 tables | Yes — report | |
| 18 | 2026-11-04 | Make figures: bar charts, learning curves, heatmaps, rank plot | Yes — report | 04 Planned Comparisons |
| 19 | 2026-11-04 | Choose report template | No | 04 Report |
| 20 | 2026-11-04 | Write report incl. Bonus Points section + AI-use disclosure | Yes — submission | Due Friday night of finals week (exact date ⛔ NOT YET RECORDED) |
| 21 | 2026-11-04 | Record exact due date | No | |

## Done
| # | Added | Done | Item | Resolution |
|---|-------|------|------|------------|

# Action Items — COGS 118A Classifier Comparison
**Last Updated:** 2026-11-16

## Open
| # | Added | Item | Blocking? | Notes |
|---|-------|------|-----------|-------|
| 3 | 2026-11-04 | Verify full dataset sizes on UCI pages | No | Covered by #22: `dataset_summary.json` gives the real counts |
| 4 | 2026-11-04 | Ask instructor/TA: is TabPFN OK as an added classifier? | No, but ask early | 04 open question 1 |
| 5 | 2026-11-04 | Ask instructor/TA: are documented per-model training caps OK? | No | 04 open question 2 |
| 6 | 2026-11-04 | Record GPU server specs (GPU model/count, CPU, RAM) | No | 01 header |
| 7 | 2026-11-04 | Set up Python env; record library versions (xgboost, cuML, PyTorch, skorch, tabpfn, scikit-learn) | Yes — all runs | 04 Software |
| 10 | 2026-11-04 | Write per-job experiment script (01 Step 2) | Yes | Reads `data/processed/<name>.npz` + caps from `src/config.py` |
| 11 | 2026-11-04 | Write GPU launcher with resume (01 Step 3) | Yes | |
| 12 | 2026-11-04 | Run timing test; record in 02 §2 | Yes — sets sizes/caps | 01 Step 4 |
| 13 | 2026-11-04 | Finalize Covertype and HIGGS sizes | Yes — full sweep | After #12. Set `N_ROWS` in `config.py`, re-run `prepare_data.py` |
| 14 | 2026-11-04 | Finalize SVM training cap | Yes — full sweep | After #12. Provisional 50k in `TRAIN_CAP` |
| 15 | 2026-11-04 | Finalize grids for XGBoost, ANN, KNN | Yes — full sweep | 03 |
| 16 | 2026-11-04 | Run full 216-job sweep | Yes — results | |
| 17 | 2026-11-04 | Aggregate results; fill 02 tables | Yes — report | |
| 18 | 2026-11-04 | Make figures: bar charts, learning curves, heatmaps, rank plot | Yes — report | 04 Planned Comparisons |
| 19 | 2026-11-04 | Choose report template | No | 04 Report |
| 20 | 2026-11-04 | Write report incl. Bonus Points section + AI-use disclosure | Yes — submission | Due Friday night of finals week (exact date ⛔ NOT YET RECORDED) |
| 21 | 2026-11-04 | Record exact due date | No | |
| 22 | 2026-11-16 | Run `python src/prepare_data.py` on the server; copy row counts from `data/processed/dataset_summary.json` into 03 | Yes — all runs | Tested only on fake data so far (UCI unreachable from Claude's workspace) |
| 23 | 2026-11-16 | Update 03 and 05 with the 2026-11-16 decisions (Done table below) | No | 03/05 weren't in this session |

## Done
| # | Added | Done | Item | Resolution |
|---|-------|------|------|------------|
| 1 | 2026-11-04 | 2026-11-16 | Confirm dataset lineup | Kept Adult, Letter, Covertype, HIGGS. Considered MAGIC Gamma, Bank Marketing, SUSY; kept current set because 3 of 4 match the paper |
| 2 | 2026-11-04 | 2026-11-16 | Pick Letter binarization | A–M = 0 vs N–Z = 1 (balanced, paper's LETTER.p2) |
| 8 | 2026-11-04 | 2026-11-16 | Decide TabPFN version | v2 open weights; 10k training-row cap |
| 9 | 2026-11-04 | 2026-11-16 | Write data prep script (01 Step 1) | `src/prepare_data.py` + `src/config.py`. Adult: train+test combined, rows with `?` dropped, categoricals one-hot. Covertype: largest class vs rest. HIGGS: chunked random subsample. All 9 splits per dataset pre-computed and stored in the `.npz`, so every model sees the same rows |

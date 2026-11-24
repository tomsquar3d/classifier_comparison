# Action Items — COGS 118A Classifier Comparison
**Last Updated:** 2026-11-24 (session 3)

## Open
| # | Added | Item | Blocking? | Notes |
|---|-------|------|-----------|-------|
| 4 | 2026-11-04 | Ask instructor/TA: is TabPFN OK as an added classifier? | No, but ask early | 04 open question 1 |
| 5 | 2026-11-04 | Ask instructor/TA: are documented per-model training caps OK? | No | 04 open question 2 |
| 10 | 2026-11-04 | Write per-job experiment script (01 Step 2) | Yes | Reads `data/processed/<name>.npz` + caps from `src/config.py`. Load TabPFN as v2 explicitly; set cuML RF `max_depth` explicitly |
| 11 | 2026-11-04 | Write GPU launcher with resume (01 Step 3) | Yes | Updated 2026-11-24: resumable runner for cloud sessions; trial-1 jobs first (05 #15); save results somewhere that survives the session |
| 12 | 2026-11-04 | Run timing test; record in 02 §2 | Yes — confirms the 5–10 GPU-h estimate | 01 Step 4. Updated 2026-11-24: HIGGS 100k, on the sweep's platform |
| 15 | 2026-11-04 | Finalize grids for XGBoost, ANN, KNN | Yes — full sweep | 03. ANN: 6–8 combos + early stopping (05 #16) |
| 16 | 2026-11-04 | Run full 216-job sweep | Yes — results | |
| 17 | 2026-11-04 | Aggregate results; fill 02 tables | Yes — report | |
| 18 | 2026-11-04 | Make figures: bar charts, learning curves, heatmaps, rank plot | Yes — report | 04 Planned Comparisons |
| 19 | 2026-11-04 | Choose report template | No | 04 Report |
| 20 | 2026-11-04 | Write report incl. Bonus Points section + AI-use disclosure | Yes — submission | Due Friday night of finals week (exact date ⛔ NOT YET RECORDED) |
| 21 | 2026-11-04 | Record exact due date | No | |
| 25 | 2026-11-24 | Request DSMLP access (email datahub@ucsd.edu) | Yes — first-choice platform | Ask for a Turing+ GPU type; pool also has 1080 Tis (05 #12) |
| 26 | 2026-11-24 | Set up Kaggle notebook with the `cogs118a` packages; re-run GPU smoke test there | Yes — backup platform | Batch "Save version" runs avoid idle timeout |
| 27 | 2026-11-24 | Re-run `prepare_data.py` with Covertype/HIGGS at 100k; record counts + % positive in 03 | Yes — full sweep | `config.py` already updated |

## Done
| # | Added | Done | Item | Resolution |
|---|-------|------|------|------------|
| 1 | 2026-11-04 | 2026-11-16 | Confirm dataset lineup | Kept Adult, Letter, Covertype, HIGGS. Considered MAGIC Gamma, Bank Marketing, SUSY; kept current set because 3 of 4 match the paper |
| 2 | 2026-11-04 | 2026-11-16 | Pick Letter binarization | A–M = 0 vs N–Z = 1 (balanced, paper's LETTER.p2) |
| 8 | 2026-11-04 | 2026-11-16 | Decide TabPFN version | v2 open weights; 10k training-row cap |
| 9 | 2026-11-04 | 2026-11-16 | Write data prep script (01 Step 1) | `src/prepare_data.py` + `src/config.py`. Adult: train+test combined, rows with `?` dropped, categoricals one-hot. Covertype: largest class vs rest. HIGGS: chunked random subsample. All 9 splits per dataset pre-computed and stored in the `.npz`, so every model sees the same rows |
| 3 | 2026-11-04 | 2026-11-16 | Verify full dataset sizes | Real counts from the server run, recorded in 03 |
| 6 | 2026-11-04 | 2026-11-16 | Record GPU server specs | 8× L40S 48 GB, CUDA 13.0, 64 cores, 503 GB RAM; shared with other users. In 01 header |
| 7 | 2026-11-04 | 2026-11-16 | Set up env; record versions | Conda env `cogs118a`; versions in 04 Software; GPU smoke test passed |
| 22 | 2026-11-16 | 2026-11-16 | Run prepare_data.py on server | Adult 45,222 / Letter 20,000 / Covertype 581,012 / HIGGS 1,000,000 |
| 23 | 2026-11-16 | 2026-11-16 | Update 03 and 05 with 2026-11-16 decisions | 05 #7–11 added; 01, 03, 04, 06 also updated |
| 13 | 2026-11-04 | 2026-11-24 | Finalize Covertype and HIGGS sizes | 100k each, stratified, shared by all models (05 #13). Re-prep is #27 |
| 14 | 2026-11-04 | 2026-11-24 | Finalize SVM training cap | 10k (05 #14) |
| 24 | 2026-11-16 | 2026-11-24 | Ask server owner about GPU etiquette | No longer needed — lab server unavailable (05 #12) |

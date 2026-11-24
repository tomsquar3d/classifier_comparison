# Action Items — COGS 118A Classifier Comparison
**Last Updated:** 2026-11-24 (session 4)

## Open
| # | Added | Item | Blocking? | Notes |
|---|-------|------|-----------|-------|
| 4 | 2026-11-04 | Ask instructor/TA: is TabPFN OK as an added classifier? | No, but ask early | 04 open question 1 |
| 5 | 2026-11-04 | Ask instructor/TA: are documented per-model training caps OK? | No | 04 open question 2 |
| 10 | 2026-11-04 | Write per-job experiment script (01 Step 2) | Yes | Reads `data/processed/<name>.npz` + caps from `src/config.py`. Load TabPFN as v2 explicitly; set cuML RF `max_depth` explicitly. Include speed timing per 01 Step 2 (`TIMING_*` in config) |
| 11 | 2026-11-04 | Write resumable runner (01 Step 3) | Yes | Skips jobs with existing results; trial 1 before trials 2–3. Free-memory check no longer needed (shared server gone) |
| 12 | 2026-11-04 | Run timing test (HIGGS 100k, all models, 80/20, 1 trial); record in 02 §2 | Yes — decides 5- vs 3-fold CV | 01 Step 4 |
| 15 | 2026-11-04 | Finalize grids for XGBoost, ANN, KNN | Yes — full sweep | 03. MLP: early stopping, 6–8 combos (05 #17) |
| 16 | 2026-11-04 | Run full 216-job sweep | Yes — results | |
| 17 | 2026-11-04 | Aggregate results; fill 02 tables | Yes — report | |
| 18 | 2026-11-04 | Make figures: bar charts, learning curves, heatmaps, rank plot | Yes — report | 04 Planned Comparisons |
| 19 | 2026-11-04 | Choose report template | No | 04 Report |
| 20 | 2026-11-04 | Write report incl. Bonus Points section + AI-use disclosure | Yes — submission | Due Friday night of finals week (exact date ⛔ NOT YET RECORDED) |
| 21 | 2026-11-04 | Record exact due date | No | |
| 28 | 2026-11-24 | Get DSMLP access (email datahub@ucsd.edu); request a Turing-or-newer GPU type | Yes — compute | 05 #13. Kaggle is the backup |
| 29 | 2026-11-24 | Re-run `prepare_data.py` on the new platform at 100k caps; record Covertype/HIGGS rows + % positive in 02/03 | Yes — all runs | Old `data/` was on yggdrasil |
| 30 | 2026-11-24 | Rebuild the env on the new platform; check its GPU driver supports CUDA 13 (else use CUDA 12 builds) and re-record versions in 04 | Yes — all runs | Env was built for CUDA 13.0 / driver 580 on yggdrasil; cloud drivers may be older |
| 31 | 2026-11-24 | Plan how results files get out of each Kaggle/Colab session (push to repo or save as dataset) | Yes — before sweep | 01 Step 3 |
| 32 | 2026-11-24 | Ask instructor/TA: is tuning on trial 1 and reusing settings for trials 2–3 OK? | No | 04 open question 3 |
| 25 | 2026-11-24 | Make speed table + accuracy-vs-speed figures | No — bonus | 02 §6, 04 Speed vs. accuracy |
| 26 | 2026-11-24 | If sweep timings are too noisy or came from mixed GPU types, run a short timing-only pass on one GPU type | No | Decide after the sweep |

## Done
| # | Added | Done | Item | Resolution |
|---|-------|------|------|------------|
| 13 | 2026-11-04 | 2026-11-24 | Finalize Covertype and HIGGS sizes | 100k each, stratified fixed-seed subsample (05 #14). Decided without the timing test, as part of the replan |
| 14 | 2026-11-04 | 2026-11-24 | Finalize SVM training cap | 10k (05 #15) |
| 24 | 2026-11-16 | 2026-11-24 | Ask server owner about GPU etiquette | Not needed — server no longer available (05 #13) |
| 27 | 2026-11-24 | 2026-11-24 | Sync repo docs + config with the 2026-11-24 project-doc changes | Done this session: 05 #13–17, config.py, 01–07, README |
| 1 | 2026-11-04 | 2026-11-16 | Confirm dataset lineup | Kept Adult, Letter, Covertype, HIGGS. Considered MAGIC Gamma, Bank Marketing, SUSY; kept current set because 3 of 4 match the paper |
| 2 | 2026-11-04 | 2026-11-16 | Pick Letter binarization | A–M = 0 vs N–Z = 1 (balanced, paper's LETTER.p2) |
| 8 | 2026-11-04 | 2026-11-16 | Decide TabPFN version | v2 open weights; 10k training-row cap |
| 9 | 2026-11-04 | 2026-11-16 | Write data prep script (01 Step 1) | `src/prepare_data.py` + `src/config.py`. Adult: train+test combined, rows with `?` dropped, categoricals one-hot. Covertype: largest class vs rest. HIGGS: chunked random subsample. All 9 splits per dataset pre-computed and stored in the `.npz`, so every model sees the same rows |
| 3 | 2026-11-04 | 2026-11-16 | Verify full dataset sizes | Real counts from the server run, recorded in 03 |
| 6 | 2026-11-04 | 2026-11-16 | Record GPU server specs | 8× L40S 48 GB, CUDA 13.0, 64 cores, 503 GB RAM; shared with other users. In 01 header |
| 7 | 2026-11-04 | 2026-11-16 | Set up env; record versions | Conda env `cogs118a`; versions in 04 Software; GPU smoke test passed |
| 22 | 2026-11-16 | 2026-11-16 | Run prepare_data.py on server | Adult 45,222 / Letter 20,000 / Covertype 581,012 / HIGGS 1,000,000 |
| 23 | 2026-11-16 | 2026-11-16 | Update 03 and 05 with 2026-11-16 decisions | 05 #7–11 added; 01, 03, 04, 06 also updated |

# Action Items — COGS 118A Classifier Comparison
**Last Updated:** 2026-11-26 (session 6)

## Open
| # | Added | Item | Blocking? | Notes |
|---|-------|------|-----------|-------|
| 10 | 2026-11-04 | Write per-job experiment script (01 Step 2) | Yes | Reads `data/processed/<name>.npz`; tunes every trial (05 #20); saves full `cv_results_` (rubric: accuracy curves vs. hyperparameters) and test confusion matrix; errors if a split exceeds `TRAIN_CAP`. Load TabPFN as v2 explicitly; set cuML RF `max_depth` explicitly. Include speed timing per 01 Step 2 (`TIMING_*` in config) |
| 11 | 2026-11-04 | Write resumable runner (01 Step 3) | Yes | Skips jobs with existing results; jobs independent, any order (05 #20). Free-memory check no longer needed (shared server gone) |
| 12 | 2026-11-04 | Run timing test (HIGGS 12.5k, all models, 80/20, 1 trial); record in 02 §2 | Yes — decides 5- vs 3-fold CV | 01 Step 4 |
| 15 | 2026-11-04 | Finalize grids for XGBoost, ANN, KNN | Yes — full sweep | Proposed 2026-11-26 in 03 (05 #24); waiting on Tom's OK, incl. KNN → sklearn. MLP: early stopping, 6–8 combos (05 #17) |
| 16 | 2026-11-04 | Run full 216-job sweep | Yes — results | |
| 17 | 2026-11-04 | Aggregate results; fill 02 tables | Yes — report | |
| 18 | 2026-11-04 | Make figures: bar charts, learning curves, heatmaps, rank plot | Yes — report | 04 Planned Comparisons |
| 20 | 2026-11-04 | Write report incl. Bonus Points section + AI-use disclosure | Yes — submission | Due 11:59pm PST, Fri Dec 11, 2026 |
| 29 | 2026-11-24 | Re-run `prepare_data.py` on the new platform (12.5k caps, natural class rates); record final rows for all 4 datasets in 02/03 | Yes — all runs | Old `data/` was on yggdrasil. Check all 12,500 rows with rates ≈ Adult 24.8%, Letter 50.3%, Covertype 48.8%, HIGGS 53.0% |
| 31 | 2026-11-24 | Plan how results files get out of each Kaggle/Colab session (push to repo or save as dataset) | Yes — before sweep | 01 Step 3 |
| 34 | 2026-11-26 | Decide on a scaling experiment for rubric item 1 (dataset size, 10 pts): larger Covertype/HIGGS samples for GPU models, reported separately from the main 12.5k study | No | 04 open question |
| 25 | 2026-11-24 | Make speed table + accuracy-vs-speed figures | No — bonus | 02 §6, 04 Speed vs. accuracy |
| 26 | 2026-11-24 | If sweep timings are too noisy or came from mixed GPU types, run a short timing-only pass on one GPU type | No | Decide after the sweep |

## Done
| # | Added | Done | Item | Resolution |
|---|-------|------|------|------------|
| 30 | 2026-11-24 | 2026-11-26 | Rebuild env on new platform | Built on DSMLP at `~/envs/cogs118a`, same versions as before; smoke test 6/6 (04 Software) |
| 28 | 2026-11-24 | 2026-11-26 | Get DSMLP access | Tom already had access. Launch: `launch-scipy-ml.sh -g 1 -c 4 -m 16 -l gpu-class=medium` → RTX PRO 6000 Blackwell **MIG slice** (24 GB, 46 SMs), driver 580.126.20, CUDA 13.0. Letter data prep ran OK in the container |
| 19 | 2026-11-04 | 2026-11-26 | Choose report template | NeurIPS LaTeX |
| 4 | 2026-11-04 | 2026-11-26 | Ask TA: is TabPFN OK? | Not asking — TabPFN is the project's novelty (05 #22) |
| 5 | 2026-11-04 | 2026-11-26 | Ask TA: are per-model caps OK? | Moot — caps dropped (05 #19) |
| 21 | 2026-11-04 | 2026-11-26 | Record exact due date | **11:59pm PST, Fri Dec 11, 2026** (from the assignment slides) |
| 32 | 2026-11-24 | 2026-11-26 | Ask TA: is trial-1 tuning OK? | Moot — tune every trial (05 #20) |
| 33 | 2026-11-24 | 2026-11-26 | Mention balancing to TA? | Moot — Adult left unbalanced like the paper (05 #21, #22) |
| 13 | 2026-11-04 | 2026-11-24 | Finalize Covertype and HIGGS sizes | 100k each, stratified fixed-seed subsample (05 #14). Decided without the timing test, as part of the replan. **Superseded 2026-11-26: 12.5k (05 #19)** |
| 14 | 2026-11-04 | 2026-11-24 | Finalize SVM training cap | 10k (05 #15). **Superseded 2026-11-26: no cap (05 #19)** |
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

# Action Items — COGS 118A Classifier Comparison
**Last Updated:** 2026-11-26 (session 6)

## Open
| # | Added | Item | Blocking? | Notes |
|---|-------|------|-----------|-------|
| 20 | 2026-11-04 | Write report incl. Bonus Points section + AI-use disclosure | Yes — submission | Due 11:59pm PST, Fri Dec 11, 2026 |
| 34 | 2026-11-26 | Decide on a scaling experiment for rubric item 1 (dataset size, 10 pts): larger Covertype/HIGGS samples for GPU models, reported separately from the main 12.5k study | No | 04 open question |
| 26 | 2026-11-24 | If sweep timings are too noisy or came from mixed GPU types, run a short timing-only pass on one GPU type | No | Decide after the sweep |

## Done
| # | Added | Done | Item | Resolution |
|---|-------|------|------|------------|
| 16 | 2026-11-04 | 2026-11-26 | Run full 216-job sweep | 216/216 on DSMLP, one GPU type; results in git |
| 17 | 2026-11-04 | 2026-11-26 | Aggregate results | `src/aggregate.py`; tables in 02 §3–5 |
| 18 | 2026-11-04 | 2026-11-26 | Make figures | 8 figures incl. Friedman / critical-difference and bootstrap ranks |
| 25 | 2026-11-24 | 2026-11-26 | Speed table + accuracy-vs-speed figure | 02 §5, fig5 |
| 35 | 2026-11-26 | 2026-11-26 | Check ANN undertraining | Not undertrained: all lr/patience/optimizer settings ~0.66 held-out; settings unchanged (02 §2). Delete the timing-test ANN JSON so it re-runs with the new prediction batch size |
| 12 | 2026-11-04 | 2026-11-26 | Run timing test | 15m17s for HIGGS 80/20, all 6 models; keep 5-fold CV; sweep ~5–6 h (≤9.2 h). ANN undertrained → #35 (02 §2) |
| 10 | 2026-11-04 | 2026-11-26 | Write per-job experiment script | `src/run_job.py` + `src/models.py`; tested on fake data, CPU mode (01 Step 2) |
| 11 | 2026-11-04 | 2026-11-26 | Write resumable runner | `src/run_sweep.py`; tested incl. skip-if-done (01 Step 3) |
| 15 | 2026-11-04 | 2026-11-26 | Finalize grids | Approved by Tom; in `config.GRIDS` (05 #24) |
| 31 | 2026-11-24 | 2026-11-26 | Plan how results leave each session | DSMLP home folder persists, so results stay put; real results JSONs now tracked in git |
| 29 | 2026-11-24 | 2026-11-26 | Re-run `prepare_data.py` on new platform | Done on DSMLP: all 12,500 rows; Adult 24.78%, Letter 50.30%, Covertype 48.76%, HIGGS 52.96% (02 §1). HIGGS raw file staged in `/tmp` to stay under the home quota |
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

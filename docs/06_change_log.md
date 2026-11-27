# Change Log
**Project:** COGS 118A Classifier Comparison

Append-only. Newest first. Never edit past entries — append corrections.

---

## 2026-11-26 — Session 6 (cont. 6)
**Files changed:** `src/aggregate.py`, `results/summary/all_metrics.csv`, `results/summary/rankings.csv`

- New combined tables: `all_metrics.csv` (one row per dataset × partition × model: train/val/test mean ± std, majority baseline, rank in cell, tune/fit/predict time, best params per trial) and `rankings.csv` (per model: mean rank overall / by dataset / by partition, cells won, mean test accuracy, bootstrap rank probabilities). Also put in a Claude Sheet for Tom.
- Branch `replan-12k-cap` fast-forwarded into `main`.

## 2026-11-26 — Session 6 (cont. 5)
**Files changed:** `results/*_trial{2,3}.json` (144), `results/summary/`, `figures/`, `src/aggregate.py`, `docs/02`, `07`

- Trials 2–3 finished (216/216; trial-1 files byte-identical to the committed ones). `aggregate.py` gains Friedman + Nemenyi critical difference and paper-style bootstrap rank probabilities (fig7, fig8). `02` §3–5 rewritten with 3-trial results; interim trial-1 section superseded.

## 2026-11-26 — Session 6 (cont. 4)
**Files changed:** new `src/aggregate.py`, `results/*_trial1.json` (72), `results/summary/`, `figures/`, `docs/02`, `07`

- Trial 1 sweep finished on DSMLP: 72/72 jobs, all on one GPU type. Results copied back via tar/scp and committed.
- New `src/aggregate.py`: main table (train/val/test mean ± std), ranks, speed table, and 6 figures (accuracy vs training share, rank table, SVM C×gamma heatmaps, KNN curves, accuracy vs speed, Adult confusion matrices). Palette validated with the dataviz validator (6 categorical slots, light mode).
- `02` §3: interim trial-1 tables and observations.

## 2026-11-26 — Session 6 (cont. 3)
**Files changed:** `src/config.py`, `src/models.py`, new `src/ann_check.py`, `docs/02`, `07`

- Smoke sweep passed 6/6 on DSMLP GPU (incl. TabPFN). Timing test run (closes `07` #12): 15m17s; keep 5-fold CV; results in `02` §2.
- ANN undertrained (train acc 0.70, smallest net best). New diagnostic `src/ann_check.py` (tested on fake data, CPU); added `07` #35.
- `ANN_PREDICT_BATCH_SIZE = 4096` for prediction (speed fairness; no accuracy effect).
- `ann_check.py` run on DSMLP: ANN not undertrained (every setting ~0.66 held-out; best epoch unchanged with longer patience). Settings unchanged; closes `07` #35. Earlier "undertrained" reading in this log was wrong.

## 2026-11-26 — Session 6 (cont. 2)
**Files changed:** `src/config.py`, new `src/models.py`, `src/run_job.py`, `src/run_sweep.py`, `.gitignore`, `docs/01`, `03`, `05`, `07`, `docs/project-design.md`

- Grids approved by Tom (`05` #24); added to `config.GRIDS` with fixed settings (`RF_N_TREES`, `RF_MAX_DEPTH`, `XGB_LEARNING_RATE`, `ANN_*`).
- New experiment script `run_job.py` and resumable runner `run_sweep.py` (closes `07` #10, #11, #15, #31). TabPFN goes through the same 5-fold CV (one setting) so it has a validation accuracy.
- Tested in Claude's workspace on fake data, CPU mode only (see `01` Steps 2–3). Not yet run on the GPU.
- `.gitignore`: real `results/*.json` now tracked; smoke/CPU-test runs and logs are not.

## 2026-11-26 — Session 6 (cont.)
**Files changed:** `src/config.py`, README.md, `docs/01`–`07`, `docs/project-design.md`

- Balancing switched off for all four datasets; every dataset keeps its natural class rate, as in the paper (`05` #23, reverses #18). (source: Tom's call this session)
- Hyperparameter grids proposed from the paper's §2.1 in `03`, incl. moving KNN to scikit-learn for distance weighting (`05` #24). Pending Tom's OK.
- Report template chosen: NeurIPS LaTeX (closes `07` #19).
- DSMLP working (closes `07` #28): RTX PRO 6000 Blackwell MIG slice, 24 GB, driver 580.126.20, CUDA 13.0. `prepare_data.py --only letter` ran in the container: 12,500 rows, 50.3% positive. New `src/smoke_test.py` fits all 6 models on random data on the GPU and prints versions.
- Env rebuilt on DSMLP (closes `07` #30): same versions as yggdrasil; smoke test 6/6. Hit the home-folder quota during pip install; fixed by clearing conda/pip caches. Build steps in `04` Software.
- Data prep run on DSMLP (closes `07` #29): all four at 12,500 rows, natural rates (02 §1). HIGGS save hit the home quota; fixed by moving the 2.6 GB raw file to `/tmp` (container-local) and symlinking it into `data/raw`.

## 2026-11-26 — Session 6
**Files changed:** `src/config.py`, `src/prepare_data.py`, README.md, `docs/01`–`07`, `docs/project-design.md`

- Replan after checking the assignment text and the paper's protocol (Tom agreed this session): every dataset capped at 12,500 rows, so the 80/20 split trains on exactly 10k and no model needs its own cap (`05` #19); tune in every trial, as the paper did (#20); Adult left unbalanced at ~24.8% positive, like the paper's 25% (#21); TA check-in dropped (#22). #14 sizes, #15 and #16 superseded; #18 amended.
- `src/config.py`: `N_ROWS` 12,500 for all four; `BALANCE_CLASSES` now a per-dataset dict (Adult False); `TUNE_TRIAL` removed; `TRAIN_CAP` all None except TabPFN 10k (guard only); `TIMING_N_ROWS` 10,000 → 2,500; `HIGGS_RAW_SAMPLE` 2M → 200k.
- `src/prepare_data.py`: reads the per-dataset balancing switch; summary JSON adds `balanced` and `majority_class_baseline`. Tested on fake data in Claude's workspace: Adult-like 24.8% stays ~24.9% at 12,500 rows; HIGGS-like 53% → 50%; train sizes 2,500 / 6,250 / 10,000. Not yet run on real data.
- `07`: closed #4, #5, #21 (due 11:59pm PST Dec 11, 2026), #32, #33; added #34 (scaling experiment for rubric item 1).

## 2026-11-24 — Session 5
**Files changed:** `src/config.py`, `src/prepare_data.py`, README.md, `docs/02`–`07`, `docs/project-design.md`

- All datasets balanced 50/50 by undersampling the larger class, before the 100k cap. Logged as `05` #18. (source: Tom's request this session; rationale not stated)
- `src/config.py`: new `BALANCE_CLASSES = True`. `src/prepare_data.py`: new `balance()`; summary JSON also records rows and % positive before balancing. Tested on fake data in Claude's workspace (Letter-like 30% → 50%, Covertype-like 40.5% → 50%, all 80/20 test splits 50%). Not yet run on real data.
- `02`, `03`: pre-balancing numbers relabelled; expected post-balancing sizes added as approximations (Adult ~22.4k, Letter ~19.9k, Covertype/HIGGS 100k). `04`: rubric mapping drops the imbalance angle.
- `07`: #29 notes updated; added #33 (TA: is balancing OK / note Adult departs from the paper).

## 2026-11-24 — Session 4
**Files changed:** `src/config.py`, README.md, `docs/01`–`07`, `docs/project-design.md`

- Synced the repo with the 2026-11-24 replan recorded in the claude.ai project doc `project-design.md` (made in a separate chat; Tom confirmed the sync this session). Logged as `05` #13–17.
- `src/config.py`: Covertype and HIGGS `N_ROWS` → 100,000; SVM `TRAIN_CAP` 50,000 → 10,000; new `TUNE_TRIAL = 0`; note on the 3-fold CV fallback.
- `01`: platform now DSMLP / Kaggle / Colab (Turing+); Step 2 tunes on trial 1 only, trials 2–3 refit; Step 3 is now a resumable runner (old multi-GPU launcher design retired); timing test on HIGGS 100k.
- `02`: Covertype/HIGGS sizes marked as set-in-config but not yet re-run (⛔ until #29); 2026-11-16 sizes kept as superseded.
- `03`, `04`: new sizes, caps, trial-1 tuning, MLP grid limits; rubric mapping no longer claims large datasets; new open question on trial-1 tuning.
- `07`: closed #13, #14, #24, #27; added #28–32 (DSMLP access, re-run data prep, rebuild env + CUDA driver check, getting results out of cloud sessions, TA question on trial-1 tuning).

## 2026-11-24 — Session 3
**Files changed:** `src/config.py`, README.md, `docs/01`–`07`, `docs/project-design.md`

- Added speed metrics to the design: tuning time, final fit time, prediction ms per 1k rows, throughput, plus GPU conditions logged per job. Logged as `05` #12. (source: Tom's request this session)
- `src/config.py`: new `TIMING_N_ROWS = 10_000`, `TIMING_REPEATS = 5`, `TIMING_WARMUP = 1`. Not used yet — the experiment script isn't written.
- `01` Step 2: speed-measurement procedure (warm-up, time until NumPy output, same timing sample for all models, GPU snapshot). `03`: timing-sample note. `04`: speed vs. accuracy added as a planned bonus comparison; LeCun 1995 and Lim 2000 noted as precedent; status/dataset lines brought up to date (were still "proposed"). `02`: new §6 Speed (old §6 → §7).
- `02` §1 dataset summary: replaced ⛔ markers with the counts already recorded in `03` from the 2026-11-16 run.
- `07`: added #25 (speed figures) and #26 (fallback clean timing pass); #10 notes updated.
- Follow-up: each job also logs the GPU model name, and speed comparisons are limited to one GPU type (`01`, `04`, `05` #12, `07` #26).
- ⚠️ Not yet in these docs: the 2026-11-24 infrastructure changes recorded in the claude.ai project doc `project-design.md` (lab server unavailable, Covertype/HIGGS capped at 100k, SVM cap 10k, tune on trial 1 only). Sync pending (action item #27).

## 2026-11-16 — Session 2
**Files changed:** `src/config.py`, `src/prepare_data.py`, `.gitignore`, README.md (moved to repo root), all docs

- Locked dataset lineup, Letter rule (A–M vs. N–Z) and TabPFN v2; logged as `05` #7–11. (source: Tom's answers this session)
- Wrote `src/config.py` (shared settings) and `src/prepare_data.py` (Step 1). Tested on fake data in Claude's workspace, then run on the server: Adult 45,222 / Letter 20,000 / Covertype 581,012 / HIGGS 1,000,000 rows. Sizes recorded in `03`.
- Set up GitHub repo `tomsquar3d/classifier_comparison`; fixed upload layout (`.gitignore` name, duplicate README and action items).
- Installed Miniforge + conda env `cogs118a`; GPU libraries installed and smoke-tested. Versions in `04` Software; server specs in `01` header.
- Found GPUs are shared with other users; launcher design updated in `01` Step 3.

## 2026-11-04 — Session 1
**Files changed:** all (folder created)

- Created project folder (`docs/`, `src/`, `data/`, `results/`, `figures/`, `report/`) using the research-project-docs skill. (source: Tom's request this session)
- Seeded docs from the assignment rubric, the Caruana & Niculescu-Mizil (ICML 2006) PDF, and decisions made in this session's planning discussion.
- Logged decisions #1–6 in `05_decisions.md`: 6-classifier lineup, AdaBoost → SVM, XGBoost over LightGBM, RealMLP/TabM dropped, grid search only, GPU + larger datasets.
- Datasets recorded as proposed, not confirmed. Sizes, grids, versions and server specs left as ⛔ markers with matching action items.
- Skill's xlsx table workbooks not created — that format is built for EHR model tables (OR/HR, demographic blocks) and doesn't fit accuracy results.

# Change Log
**Project:** COGS 118A Classifier Comparison

Append-only. Newest first. Never edit past entries — append corrections.

---

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

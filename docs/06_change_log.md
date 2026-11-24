# Change Log
**Project:** COGS 118A Classifier Comparison

Append-only. Newest first. Never edit past entries — append corrections.

---

## 2026-11-24 — Session 3
**Files changed:** `src/config.py`, README.md, `docs/01`–`07`, `docs/project-design.md`

- Added speed metrics to the design: tuning time, final fit time, prediction ms per 1k rows, throughput, plus GPU conditions logged per job. Logged as `05` #12. (source: Tom's request this session)
- `src/config.py`: new `TIMING_N_ROWS = 10_000`, `TIMING_REPEATS = 5`, `TIMING_WARMUP = 1`. Not used yet — the experiment script isn't written.
- `01` Step 2: speed-measurement procedure (warm-up, time until NumPy output, same timing sample for all models, GPU snapshot). `03`: timing-sample note. `04`: speed vs. accuracy added as a planned bonus comparison; LeCun 1995 and Lim 2000 noted as precedent; status/dataset lines brought up to date (were still "proposed"). `02`: new §6 Speed (old §6 → §7).
- `02` §1 dataset summary: replaced ⛔ markers with the counts already recorded in `03` from the 2026-11-16 run.
- `07`: added #25 (speed figures) and #26 (fallback clean timing pass); #10 notes updated.

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

# COGS 118A Classifier Comparison — Folder Contents
**Date created:** 2026-11-04
**Status:** Plan revised 2026-11-24 after losing the lab server and again 2026-11-26 (12.5k rows, tune every trial) — data prep must be re-run on the new platform; experiment script next, no results yet
**Scope:** 6 classifiers × 4 binary datasets (Adult, Letter, Covertype, HIGGS — locked 2026-11-16; 12,500 rows each, Letter/Covertype/HIGGS balanced 50/50 and Adult left unbalanced, 2026-11-26) × 3 partitions × 3 trials; metrics: accuracy (train/val/test) + speed (tune, fit, predict time — added 2026-11-24)

---

## What this folder is

Final project for COGS 118A (Fall 2026, Prof. Zhuowen Tu). Solo project. It replicates the core experiment from Caruana & Niculescu-Mizil (ICML 2006), "An Empirical Comparison of Supervised Learning Algorithms," and adds modern classifiers (XGBoost, TabPFN) to ask whether the 2006 rankings still hold.

Basic requirement from the assignment: 3 classifiers from the paper on 3 UCI datasets, binary labels, 20/80, 50/50, 80/20 train/test partitions, 3 trials each, cross-validated hyperparameters, report train/validation/test accuracy plus the chosen hyperparameters. This project goes beyond that for bonus (≥5 classifiers, ≥4 datasets, modern models).

Experiments run on cloud GPUs (UCSD DSMLP first choice, Kaggle backup, Colab overflow). The lab server used for setup is no longer available (2026-11-24).

---

## Key results — none yet

No results yet — see `07_action_items.md`.

---

## Repo layout

| Folder | Holds |
|--------|-------|
| `docs/` | This documentation set (numbered docs below) |
| `src/` | Data prep, experiment, launcher and plotting scripts |
| `data/` | Raw + processed datasets (keep out of git) |
| `results/` | One results file per job (classifier × dataset × partition × trial) |
| `figures/` | Plots for the report |
| `report/` | Final report source + PDF |

## Docs in this folder

| File | Description |
|------|-------------|
| `README.md` | This file |
| `01_experiment_pipeline.md` | Step-by-step pipeline: data prep → per-job experiment → launcher → aggregation → figures |
| `02_results.md` | All accuracy numbers, best hyperparameters, timings — numbers live here only |
| `03_datasets_and_hyperparameters.md` | Dataset definitions, label binarization rules, subsampling, hyperparameter grids |
| `04_analysis_plan.md` | Design, rubric mapping, planned comparisons and figures, open questions |
| `05_decisions.md` | Numbered decision log with rationale |
| `06_change_log.md` | Append-only session log |
| `07_action_items.md` | Running to-do tracker |

---

## Reading order

1. `04_analysis_plan.md` — what the project is testing and how it's graded
2. `05_decisions.md` — why the classifier/dataset lineup looks the way it does
3. `03_datasets_and_hyperparameters.md` — exact data and tuning setup
4. `01_experiment_pipeline.md` — how to run it
5. `02_results.md` — what came out
6. `07_action_items.md` — what's left

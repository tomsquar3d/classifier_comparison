# Results
**Project:** COGS 118A Classifier Comparison
**Results source:** `results/*.json` (none yet)
**Version:** v0.2
**Last Updated:** 2026-11-24

---

## 🔵 STATUS — No model results yet; data prep must be re-run

Data prep ran on yggdrasil 2026-11-16 (§1), but Covertype and HIGGS are now capped at 100k (`05` #14) and that server is gone, so `prepare_data.py` must be re-run on the new platform. No classifier has run. Every number in this doc must come from a results file or a script run — no estimates. Superseded numbers stay visible (version table or a note) when anything is re-run. Speed results (§6) added to the plan 2026-11-24.

---

## Run history

| Version | Jobs complete | Change | Date |
|---------|---------------|--------|------|
| — | 0 / 216 | — | — |

---

## 1. Dataset summary (after prep)

| Dataset | Rows used | Features (after encoding) | % positive |
|---------|-----------|---------------------------|------------|
| Adult | 45,222 | 104 | 24.8% |
| Letter | 20,000 | 16 | 50.3% |
| Covertype | 100,000 (set in config; ⛔ NOT YET RECORDED from a re-run) | 54 | 48.8% at 581,012 rows (2026-11-16); ⛔ NOT YET RECORDED for 100k |
| HIGGS | 100,000 (set in config; ⛔ NOT YET RECORDED from a re-run) | 28 | 53.0% at 1,000,000 rows (2026-11-16); ⛔ NOT YET RECORDED for 100k |

> Source: `data/processed/dataset_summary.json` from the 2026-11-16 yggdrasil run, as recorded in `03`. Adult and Letter are unchanged by the 2026-11-24 caps. Covertype and HIGGS rows fill in after the re-run (action item #29); the subsample is stratified, so % positive should barely move.
>
> Superseded: Covertype 581,012 rows, HIGGS 1,000,000 rows (2026-11-16 sizes).

## 2. Timing test

⛔ NOT YET RECORDED

## 3. Main results — mean train / val / test accuracy over 3 trials

One table per dataset, rows = classifier, column groups = partition (20/80, 50/50, 80/20), each with Train / Val / Test. To be filled from aggregated results.

## 4. Best hyperparameters

Per classifier × dataset × partition, the selected config in each trial.

## 5. Rankings

Per dataset × partition rank order; overall mean rank.

## 6. Speed — 🟣 [NEW — 2026-11-24]

Per classifier × dataset × partition, median over 3 trials. Definitions in `01` Step 2 (speed measurement).

| Column | Meaning |
|--------|---------|
| Tune (s) | Whole grid search incl. refit |
| Fit (s) | Final model, one training run |
| Predict (ms / 1k rows) | Median of timed repeats on the fixed timing sample |
| Throughput (rows/s) | Same measurement, inverted |

Plus the GPU conditions logged at each job's start, so slow outliers caused by other users' load can be identified. All numbers ⛔ NOT YET RECORDED until the sweep runs.

## 7. Bonus analyses

Rank significance (bootstrap or Friedman / critical-difference), accuracy-vs-speed plot (§6), any optional extras.

# Results
**Project:** COGS 118A Classifier Comparison
**Results source:** `results/*.json` (none yet)
**Version:** v0.1
**Last Updated:** 2026-11-24

---

## 🔵 STATUS — Dataset summary filled; no model results yet

Data prep ran 2026-11-16 (§1). No classifier has run. Every number in this doc must come from a results file or a script run — no estimates. Superseded numbers stay visible (version table or a note) when anything is re-run. Speed results (§6) added to the plan 2026-11-24.

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
| Covertype | 581,012 (provisional, may shrink) | 54 | 48.8% |
| HIGGS | 1,000,000 (provisional, may shrink) | 28 | 53.0% |

> Source: `data/processed/dataset_summary.json` from the 2026-11-16 server run, as recorded in `03`.

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

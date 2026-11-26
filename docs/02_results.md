# Results
**Project:** COGS 118A Classifier Comparison
**Results source:** `results/*.json` (none yet)
**Version:** v0.2
**Last Updated:** 2026-11-24

---

## 🔵 STATUS — Data prepped on DSMLP (2026-11-26); no model results yet

Data prep ran on yggdrasil 2026-11-16 (§1), but every dataset is now capped at 12,500 rows with Adult left unbalanced (`05` #19, #21) and that server is gone, so `prepare_data.py` must be re-run on the new platform. No classifier has run. Every number in this doc must come from a results file or a script run — no estimates. Superseded numbers stay visible (version table or a note) when anything is re-run. Speed results (§6) added to the plan 2026-11-24.

---

## Run history

| Version | Jobs complete | Change | Date |
|---------|---------------|--------|------|
| — | 0 / 216 | — | — |

---

## 1. Dataset summary (after prep) — 🟢 run on DSMLP 2026-11-26

| Dataset | Rows used | Features (after encoding) | % positive | Majority-class baseline | Source rows |
|---------|-----------|---------------------------|------------|-------------------------|-------------|
| Adult | 12,500 | 104 | 24.78% | 75.22% | 45,222 (after dropping missing) |
| Letter | 12,500 | 16 | 50.30% | 50.30% | 20,000 |
| Covertype | 12,500 | 54 | 48.76% | 51.24% | 581,012 |
| HIGGS | 12,500 | 28 | 52.96% | 52.96% | 199,190 (random read of the 11M file) |

> Source: `data/processed/dataset_summary.json` from `src/prepare_data.py`, run on DSMLP 2026-11-26 (12.5k stratified cap, no rebalancing — `05` #19, #23). Training rows per partition: 2,500 / 6,250 / 10,000.
>
> Superseded: the 2026-11-16 yggdrasil run (Adult 45,222 / Letter 20,000 / Covertype 581,012 / HIGGS 1,000,000 used in full or at 100k), and the planned 100k caps and 50/50 balancing (`05` #14, #18).

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

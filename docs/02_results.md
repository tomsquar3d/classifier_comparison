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

> ⚠️ All four datasets are balanced to 50/50 before use (`05` #18, 2026-11-24). The table below is the 2026-11-16 run, **before balancing**; final rows fill in after the re-run (action item #29).

| Dataset | Rows (pre-balancing) | Features (after encoding) | % positive (pre-balancing) |
|---------|-----------|---------------------------|------------|
| Adult | 45,222 | 104 | 24.8% |
| Letter | 20,000 | 16 | 50.3% |
| Covertype | 581,012 | 54 | 48.8% |
| HIGGS | 1,000,000 (2026-11-16 sample) | 28 | 53.0% |

> Source: `data/processed/dataset_summary.json` from the 2026-11-16 yggdrasil run, as recorded in `03`. Expected after balancing (computed from the rounded % positive above, so approximate): Adult ~22.4k rows, Letter ~19.9k, Covertype and HIGGS 100,000 each — all 50% positive. Final rows for all four: ⛔ NOT YET RECORDED until the re-run (action item #29).
>
> Superseded as the *used* sizes: Covertype 581,012 and HIGGS 1,000,000 (replaced by the 100k cap, `05` #14); Adult 45,222 and Letter 20,000 (replaced by balancing, `05` #18).

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

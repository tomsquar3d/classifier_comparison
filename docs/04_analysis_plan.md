# Analysis Plan
**Project:** COGS 118A Classifier Comparison
**Version:** v1
**Last Updated:** 2026-11-24

---

## 🔵 STATUS — Design locked; compute moved off the lab server

Classifiers, datasets, sizes and tuning are decided (see `05`, #1–16). The lab server is no longer available; runs move to T4-class GPUs (DSMLP / Kaggle, `05` #12). Nothing has run.

> **Updated 2026-11-24:** Covertype and HIGGS capped at 100k rows; large-data pitch dropped so every classifier is compared on the same data (`05` #13). Rubric mapping and tuning updated below.

---

## Study Design

Replicate Caruana & Niculescu-Mizil (ICML 2006) with accuracy as the metric, and add modern classifiers.

- **Classifiers (6):**
  - From the paper: random forest, SVM (RBF), ANN (MLP), KNN
  - Modern version of a paper model: XGBoost (the paper's boosted trees)
  - New: TabPFN (tabular foundation model)
- **Datasets (4, locked):** Adult (45k), Letter (20k), Covertype (100k subsample), HIGGS (100k subsample) — see `03`
- **Partitions:** 20/80, 50/50, 80/20 train/test
- **Trials:** 3 random stratified splits per partition, results averaged
- **Tuning:** 5-fold `GridSearchCV` on the training split only, run on trial 1 of each dataset × partition; trials 2–3 refit those settings (`05` #15)
- **Reported per cell:** mean train, validation and test accuracy, plus the best hyperparameters

## What the paper predicts (for comparison)

Paper ranking by mean normalized score:
1. Calibrated boosted trees
2. Random forest
3. Bagged trees
4. Calibrated SVM
5. ANN
6. KNN
7. Boosted stumps
8. Decision tree
9. Logistic regression
10. Naive Bayes

Expect RF and boosting near the top, SVM sensitive to its settings, KNN mediocre but not terrible, and accuracy rising with more training data. Our largest training sets (80k rows) are 20× the paper's 4k.

## Rubric mapping

| Rubric item | Points | How this project covers it |
|-------------|--------|----------------------------|
| Dataset size / challenge | 10 | Training sets up to 80k rows (20× the paper's 4k), 104-feature one-hot Adult, imbalanced Adult (25% positive), hard HIGGS physics task |
| Novelty | 10 | TabPFN (tabular foundation model), XGBoost vs. the 2006 boosted trees, does the 2006 ranking hold 20 years later |
| Experimental design | 50 | Full 3×3 partition/trial grid, CV tuning, train/val/test reporting, hyperparameter plots |
| Report | 30 | Abstract, intro, data, methods, experiments, conclusion, references |
| Bonus | — | ≥5 classifiers, ≥4 datasets; Bonus Points section in the report |

## Planned Comparisons

### 🔲 (a) Per dataset × partition: compare classifiers
Grouped bar chart of test accuracy with error bars across trials, plus a rank table.

### 🔲 (b) Per classifier: compare partitions
Learning curves of test accuracy vs. training share (20/50/80), one line per classifier.

### 🔲 Hyperparameter plots
Train vs. validation accuracy heatmaps — at least SVM (C × gamma) and one more model.

### 🔲 Ranking significance (bonus)
Bootstrap rank table like the paper's Table 4, or Friedman test + critical-difference diagram.

### 🔲 Optional extras (only if time)
- Calibration: AUC, Brier score, reliability diagrams
- Accuracy vs. compute time — only if every timed job ran on the same GPU type (no mixing DSMLP and Kaggle hardware)

## ⚠️ Open Questions for instructor / TA

1. Is adding TabPFN (not on the assignment's classifier list) fine as a bonus classifier? The assignment asks to discuss your own ideas beforehand.
2. Are per-model training caps (SVM and TabPFN train on 10k rows when the training split is bigger) acceptable if documented?

## Report

Template choice: ⛔ NOT YET RECORDED (NeurIPS / ICML / ICLR / Google Docs all allowed). Must include an AI-use disclosure and a Bonus Points section. Over 1,000 words excluding references.

## Software

Recorded 2026-11-16 (conda env `cogs118a`, CUDA 13.0):

| Package | Version |
|---|---|
| Python | 3.11 |
| RAPIDS cuML | 26.06 (conda, rapidsai channel) |
| xgboost | 3.2.0 |
| PyTorch | 2.14.1+cu130 |
| skorch | 1.4.0 |
| tabpfn | 10.0.0 (using v2 weights) |
| scikit-learn | 1.9.1 |

All six models passed a GPU smoke test on 2026-11-16 (on yggdrasil, no longer available). The env needs a Turing (T4) or newer GPU: CUDA 13 and RAPIDS 24.02+ don't support Pascal cards such as the GTX 1080 Ti (`05` #12). Re-run the smoke test on the new platform.

⚠️ cuML's random forest `max_depth` default changes from 16 to unlimited in cuML 26.08. Always set `max_depth` explicitly so a library update can't change results.

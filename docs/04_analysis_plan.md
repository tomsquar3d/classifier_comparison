# Analysis Plan
**Project:** COGS 118A Classifier Comparison
**Version:** v0
**Last Updated:** 2026-11-04

---

## 🔵 STATUS — Design locked for classifiers and tuning; datasets proposed

Classifier lineup and tuning method are decided (see `05`). Datasets proposed, not confirmed. Nothing has run.

---

## Study Design

Replicate Caruana & Niculescu-Mizil (ICML 2006) with accuracy as the metric, and add modern classifiers.

- **Classifiers (6):**
  - From the paper: random forest, SVM (RBF), ANN (MLP), KNN
  - Modern version of a paper model: XGBoost (the paper's boosted trees)
  - New: TabPFN (tabular foundation model)
- **Datasets (4, proposed):** Adult, Letter, Covertype, HIGGS — see `03`
- **Partitions:** 20/80, 50/50, 80/20 train/test
- **Trials:** 3 random stratified splits per partition, results averaged
- **Tuning:** 5-fold `GridSearchCV` on the training split only
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

Expect RF and boosting near the top, SVM sensitive to its settings, KNN mediocre but not terrible, and accuracy rising with more training data.

## Rubric mapping

| Rubric item | Points | How this project covers it |
|-------------|--------|----------------------------|
| Dataset size / challenge | 10 | Large datasets (Covertype, HIGGS), imbalanced Letter option |
| Novelty | 10 | TabPFN, XGBoost vs. the 2006 results, GPU-scale replication |
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
- Accuracy vs. compute time

## ⚠️ Open Questions for instructor / TA

1. Is adding TabPFN (not on the assignment's classifier list) fine as a bonus classifier? The assignment asks to discuss your own ideas beforehand.
2. Are per-model training caps (SVM, TabPFN) on the large datasets acceptable if documented?

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

All six models passed a GPU smoke test on 2026-11-16.

⚠️ cuML's random forest `max_depth` default changes from 16 to unlimited in cuML 26.08. Always set `max_depth` explicitly so a library update can't change results.

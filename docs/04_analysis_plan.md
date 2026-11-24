# Analysis Plan
**Project:** COGS 118A Classifier Comparison
**Version:** v2
**Last Updated:** 2026-11-24

---

## 🔵 STATUS — Design revised 2026-11-24; no models run

Classifiers, tuning method and datasets are decided (`05` #1–11). On 2026-11-24: speed metrics added (#12); lab server lost, Covertype/HIGGS capped at 100k, SVM cap 10k, tuning on trial 1 only, MLP grid trimmed (#13–17). Data prep must be re-run at the new sizes; no classifier has run yet.

---

## Study Design

Replicate Caruana & Niculescu-Mizil (ICML 2006) with accuracy as the metric, and add modern classifiers.

- **Classifiers (6):**
  - From the paper: random forest, SVM (RBF), ANN (MLP), KNN
  - Modern version of a paper model: XGBoost (the paper's boosted trees)
  - New: TabPFN (tabular foundation model)
- **Datasets (4, locked 2026-11-16):** Adult (45,222), Letter (20,000), Covertype and HIGGS (100k each, capped 2026-11-24) — see `03`
- **Partitions:** 20/80, 50/50, 80/20 train/test
- **Trials:** 3 random stratified splits per partition, results averaged
- **Tuning:** 5-fold `GridSearchCV` on the training split only, **once per dataset × partition × model, on trial 1**. Trials 2–3 refit the trial-1 best settings on their own split, so the 3 trials measure split-to-split variation, not tuning variation. Say this in the report.
- **Reported per cell:** mean train and test accuracy over 3 trials, validation accuracy from the trial-1 CV, plus the best hyperparameters
- **Speed per cell (added 2026-11-24):** tuning time, final fit time, prediction time per 1,000 rows, throughput — definitions in `01` Step 2

The paper measured no speed metric. Two studies it cites did include computational cost alongside accuracy: LeCun et al. (1995) and Lim et al. (2000). Cite them as precedent.

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
| Dataset size / challenge | 10 | Up to 100k rows per dataset (80k training rows at 80/20, 20× the paper's 4k); HIGGS is a hard problem; Adult moderately imbalanced. No large-data novelty pitch (`05` #14) |
| Novelty | 10 | TabPFN, XGBoost vs. the 2006 results, accuracy-vs-speed tradeoff (not in the paper) |
| Experimental design | 50 | Full 3×3 partition/trial grid, CV tuning (trial 1), train/val/test reporting, hyperparameter plots |
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

### 🔲 Speed vs. accuracy (bonus, added 2026-11-24)
- Speed table: tune / fit / predict time per classifier × dataset (`02` §6)
- Scatter of test accuracy vs. prediction time (ms per 1k rows, log scale), one point per classifier, one panel per dataset. Top-left = fast and accurate.
- Second scatter with total tuning + fit time on the x-axis — the "cost to get a model ready" view, where untuned TabPFN should stand out
- Talking point: speed doesn't split evenly. KNN and TabPFN barely train but do their work at prediction time; RF and XGBoost are the reverse; SVM prediction cost grows with its number of support vectors.
- ⚠️ Only timings from the same GPU type go in one comparison. Present them as relative speeds on that GPU, not absolute numbers.

### 🔲 Optional extras (only if time)
- Calibration: AUC, Brier score, reliability diagrams

## ⚠️ Open Questions for instructor / TA

1. Is adding TabPFN (not on the assignment's classifier list) fine as a bonus classifier? The assignment asks to discuss your own ideas beforehand.
2. Are per-model training caps (SVM and TabPFN at 10k training rows) acceptable if documented?
3. Is tuning once per cell (trial 1) and reusing those settings for trials 2–3 acceptable?

## Report

Template choice: ⛔ NOT YET RECORDED (NeurIPS / ICML / ICLR / Google Docs all allowed). Must include an AI-use disclosure and a Bonus Points section. Over 1,000 words excluding references.

## Software

Recorded 2026-11-16 on yggdrasil (conda env `cogs118a`, CUDA 13.0). Must be rebuilt on the new platform; versions may change if the platform's GPU driver doesn't support CUDA 13 (action item #30):

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

# Analysis Plan
**Project:** COGS 118A Classifier Comparison
**Version:** v2
**Last Updated:** 2026-11-24

---

## 🔵 STATUS — Design revised 2026-11-26 (12.5k rows, tune every trial, Adult unbalanced — `05` #19–22); no models run

Classifiers, tuning method and datasets are decided (`05` #1–11). On 2026-11-24: speed metrics added (#12); lab server lost, Covertype/HIGGS capped at 100k, SVM cap 10k, tuning on trial 1 only, MLP grid trimmed (#13–17). On 2026-11-26: every dataset 12,500 rows with no per-model caps, tuning in every trial, Adult left unbalanced, TA check-in dropped (#19–22). Data prep must be re-run at the new sizes; no classifier has run yet.

---

## Study Design

Replicate Caruana & Niculescu-Mizil (ICML 2006) with accuracy as the metric, and add modern classifiers.

- **Classifiers (6):**
  - From the paper: random forest, SVM (RBF), ANN (MLP), KNN
  - Modern version of a paper model: XGBoost (the paper's boosted trees)
  - New: TabPFN (tabular foundation model)
- **Datasets (4, locked 2026-11-16), 12,500 rows each (2026-11-26):** all at natural class rates, as in the paper (Adult ~24.8% positive; Letter, Covertype, HIGGS near 50%) — see `03`
- **Partitions:** 20/80, 50/50, 80/20 train/test
- **Trials:** 3 random stratified splits per partition, results averaged
- **Tuning:** 5-fold `GridSearchCV` on the training split only, **in every trial** (as the paper did, `05` #20).
- **Reported per cell:** mean train, validation and test accuracy over 3 trials, plus the best hyperparameters per trial
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
| Dataset size / challenge | 10 | 12,500 rows per dataset (`05` #19), drawn from large sources (Covertype 581k, HIGGS 11M); HIGGS is a hard problem; Adult imbalanced, natural rates throughout (`05` #23). ⚠️ Smaller than the earlier plan — see open question below |
| Novelty | 10 | TabPFN, XGBoost vs. the 2006 results, accuracy-vs-speed tradeoff (not in the paper) |
| Experimental design | 50 | Full 3×3 partition/trial grid, CV tuning in every trial, train/val/test reporting, hyperparameter plots, confusion matrices |
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

Not sending these (`05` #22). Earlier questions on TabPFN, per-model caps, trial-1 tuning and balancing are closed by `05` #19–22.

Still open (for Tom, not the TA): does the 12.5k cap cost points on rubric item 1 ("How challenging and large are the datasets", 10 pts)? Option under discussion 2026-11-26: a separate, clearly labelled scaling experiment on larger Covertype/HIGGS samples for the GPU-friendly models.

## Report

Template choice: ⛔ NOT YET RECORDED (NeurIPS / ICML / ICLR / Google Docs all allowed). Must include an AI-use disclosure and a Bonus Points section. Over 1,000 words excluding references.

## Software

Rebuilt 2026-11-26 on DSMLP (`launch-scipy-ml.sh -g 1 -c 4 -m 16 -l gpu-class=medium`; RTX PRO 6000 Blackwell MIG slice, 24 GB, 46 SMs; driver 580.126.20, CUDA 13.0). Env at `~/envs/cogs118a` (conda prefix env in the home folder, so it survives container restarts). Same versions as the 2026-11-16 yggdrasil env:

| Package | Version |
|---|---|
| Python | 3.11 |
| RAPIDS cuML | 26.06 (conda, rapidsai channel); cupy 14.2.0 |
| xgboost | 3.2.0 |
| PyTorch | 2.14.1+cu130 |
| skorch | 1.4.0 |
| tabpfn | 10.0.0 (using v2 weights) |
| scikit-learn | 1.9.1 |

Build steps (DSMLP): `conda create -y -p ~/envs/cogs118a -c rapidsai -c conda-forge python=3.11 cuml=26.06 "cuda-version=13.0" scikit-learn=1.9.1`; `pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130`; `pip install --no-cache-dir xgboost==3.2.0 skorch==1.4.0 tabpfn==10.0.0`. Notes: run `source ~/.bashrc` after `conda init`; pip warns that cuML needs `cupy-cuda13x`/`libcuml` — harmless, conda provides them (don't pip-install cupy). Home quota is small: run `conda clean -a -y` and `pip cache purge` after building.

`src/smoke_test.py` passed 6/6 on DSMLP 2026-11-26 (random data; RF, SVM on cuML; XGBoost and MLP on CUDA; KNN on scikit-learn; TabPFN v2 on CUDA). XGBoost warns that predict input is on the CPU — expected, since every model gets NumPy input and is timed the same way.

⚠️ cuML's random forest `max_depth` default changes from 16 to unlimited in cuML 26.08. Always set `max_depth` explicitly so a library update can't change results.

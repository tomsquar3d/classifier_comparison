# Data Notes, Decisions & Caveats
**Project:** COGS 118A Classifier Comparison
**Version:** v0
**Last Updated:** 2026-11-04

---

## 🔵 STATUS — 6 design decisions logged from the 2026-11-04 planning session

Entries are permanent. If a decision is reversed, add a new entry pointing back to the old one.

---

## 1. Classifier lineup: 6 classifiers — 2026-11-04

**Decision:** Random forest, XGBoost, SVM (RBF), ANN (MLP), KNN, TabPFN.
**Rationale (from planning discussion):** Covers 5+ clearly distinct classifier types, which meets the assignment's ≥5-classifier bonus even under strict counting. Four come straight from the paper. XGBoost is the modern version of the paper's boosted trees. TabPFN is the main novelty point. RF is kept because grading point (a) expects RF to come out near the top.
**Impact:** 216 jobs total (6 × 4 × 3 × 3).

## 2. AdaBoost dropped, replaced by SVM (RBF) — 2026-11-04

**Decision:** No AdaBoost / 2006-style boosted trees. Use an RBF SVM instead.
**Rationale (from planning discussion):** AdaBoost has no GPU version, and alongside XGBoost it's a second boosting model, which risks counting as one classifier. SVM is from the paper (ranked #4 after calibration), runs on GPU via cuML, and gives the classic C × gamma tuning heatmap.
**Impact:** No direct 2006-vs-2026 boosting comparison. Boosting is represented by XGBoost only.

## 3. XGBoost instead of LightGBM — 2026-11-04

**Decision:** Use XGBoost as the boosting classifier.
**Rationale (from planning discussion):** GPU support works from a standard install (`device="cuda"`), whereas LightGBM's GPU build is fiddly. It's also the boosting library the assignment links.
**Impact:** None on design.

## 4. RealMLP / TabM dropped — 2026-11-04

**Decision:** No modern tabular MLP.
**Rationale (from planning discussion):** Judged overkill. Slow and finicky to train, adds a dependency, and the plain ANN already covers neural nets.
**Impact:** Lineup goes from 7 to 6 classifiers.

## 5. Grid search only, no Bayesian hyperparameter optimization — 2026-11-04

**Decision:** Use plain 5-fold `GridSearchCV` for every tuned model.
**Rationale:** Tom judged Bayesian optimization overkill. The assignment only requires cross-validated hyperparameters, and grid search produces the train/val heatmaps it encourages.
**Impact:** XGBoost grid limited to 2–3 key hyperparameters.

## 6. Run on GPUs with larger datasets — 2026-11-04

**Decision:** Use the private multi-GPU server. Scale datasets up past the paper's 5k training rows, with per-model training caps where needed.
**Rationale (from planning discussion):** Dataset size is worth 10 rubric points, and GPUs make larger runs feasible. Caps keep the SVM (cost grows fast with rows) and TabPFN (built-in size limit) runnable.
**Impact:** Final sizes depend on the timing test. Library differences (cuML vs. scikit-learn) and caps must be documented in the report.

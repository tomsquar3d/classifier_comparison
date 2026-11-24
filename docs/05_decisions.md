# Data Notes, Decisions & Caveats
**Project:** COGS 118A Classifier Comparison
**Version:** v2
**Last Updated:** 2026-11-24

---

## 🔵 STATUS — 16 decisions logged (2026-11-04 planning, 2026-11-16 setup, 2026-11-24 replanning)

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

## 7. Dataset lineup locked: Adult, Letter, Covertype, HIGGS — 2026-11-16

**Decision:** Keep the proposed four datasets.
**Rationale:** Three of the four (Adult, Letter, Covertype) are from the paper, so results can be put side by side with Caruana's. HIGGS covers the large, hard dataset. Also considered: MAGIC Gamma Telescope (natively binary Letter substitute), Bank Marketing (second categorical, imbalanced set), SUSY (HIGGS's easier sibling). None beat the paper overlap.
**Impact:** Closes action item #1.

## 8. Letter binarized as A–M vs. N–Z — 2026-11-16

**Decision:** Letters A–M = 0, N–Z = 1 (the paper's LETTER.p2).
**Rationale:** Balanced (~50% positive), so accuracy differences between models mean something. The "O vs. rest" version is ~4% positive, where always guessing "not O" already gets ~96%.
**Impact:** Lose the imbalanced-Letter angle in the rubric's "dataset challenge" item. Adult (25% positive) is now the only moderately imbalanced dataset.

## 9. TabPFN v2 open weights, 10k training-row cap — 2026-11-16

**Decision:** Use TabPFN v2 weights, loaded explicitly with `TabPFNClassifier.create_default_for_version(ModelVersion.V2)`.
**Rationale:** No license token needed. The `tabpfn` package now defaults to TabPFN-3.5, whose weights (like 2.5 and later) are under non-commercial licenses that require accepting terms and a token; v2 weights stay under a modified Apache license.
**Impact:** TabPFN trains on at most 10k rows per job. Report this cap. The comparison is against an older TabPFN, not the newest one.

## 10. Adult: drop rows with missing values — 2026-11-16

**Decision:** Combine `adult.data` and `adult.test` (48,842 rows), drop any row with a missing value (`?`), one-hot encode categoricals.
**Rationale:** Simplest standard cleaning. Leaves 45,222 rows; the 104 one-hot features match the feature count the paper reported for Adult.
**Impact:** ~7% of rows removed. Mention in the data section of the report.

## 11. Splits pre-computed once per dataset; GPUs are shared — 2026-11-16

**Decision:** `prepare_data.py` creates all 9 train/test splits per dataset up front and stores them with the data. The launcher will check free GPU memory before placing each job.
**Rationale:** Pre-computed splits guarantee every classifier is trained and tested on exactly the same rows. The server turned out to be shared: other users' jobs hold ~29–39 GB on every GPU and several run near 100%, so "one job per empty GPU" doesn't hold.
**Impact:** Compute-time comparisons will be noisier than on a dedicated machine; caveat this if accuracy-vs-time is used as a bonus figure. Worth asking the server's owner about usage etiquette.

## 12. Lab server lost; run on T4-class GPUs (DSMLP / Kaggle / Colab) — 2026-11-24

**Decision:** yggdrasil is no longer available. Run on UCSD DataHub / DSMLP (first choice, access pending; request a Turing or newer GPU type with `-v`), with Kaggle (2× T4 16 GB, ~30 GPU-h/week, 12-h sessions) as backup and free Colab only for overflow.
**Rationale:** Tom's own GTX 1080 Ti is a Pascal card. CUDA 13 dropped Pascal, RAPIDS dropped it in 24.02, and PyTorch's Pascal-capable CUDA 12.6 wheels end after 2.14, so the existing env (cuML 26.06, CUDA 13.0) won't run on it. T4s run the env unchanged.
**Impact:** Partly replaces #6 (multi-GPU server) and makes #11's shared-server launcher checks moot. Runner must be resumable because cloud sessions get cut off. Timing comparisons only if every timed job ran on the same GPU type.

## 13. Covertype and HIGGS capped at 100k rows; large-data pitch dropped — 2026-11-24

**Decision:** One stratified 100k-row subsample each for Covertype and HIGGS, fixed seed, made before splitting and shared by every classifier. No full-data side runs (e.g. XGBoost-only on full HIGGS).
**Rationale:** Full sizes put the sweep at roughly 25–45+ T4 GPU-hours. Tom chose to drop the large-data novelty pitch so every classifier is compared on the same data. Largest training set is still 80k rows, 20× the paper's 4k.
**Impact:** Replaces the "scale datasets up" part of #6. Rubric "dataset size / challenge" now leans on 20× the paper's training size, Adult's 104 features and imbalance, and HIGGS's difficulty (`04`). Re-run `prepare_data.py`.

## 14. SVM training cap 10k (was provisional 50k) — 2026-11-24

**Decision:** SVM trains on a stratified 10k-row subsample of the training split when it is larger. Same test split as every other model.
**Rationale:** RBF SVM cost grows much faster than the data; at 50k it alone would take ~20–30 T4 GPU-hours.
**Impact:** SVM and TabPFN (#9) are the only models that see less training data. Report both caps.

## 15. Tune on trial 1 only; trials 2–3 reuse those settings — 2026-11-24

**Decision:** Run `GridSearchCV` once per dataset × partition × model, on trial 1. Trials 2–3 refit trial 1's best settings on their own training split and score on their own test split.
**Rationale:** Cuts tuning time by about two-thirds. Trials then measure split-to-split variation rather than tuning variation, which is easy to state in the report.
**Impact:** Validation accuracy and best hyperparameters come from trial 1 for each cell. Train/test accuracy is still averaged over 3 trials.

## 16. MLP: early stopping and a 6–8 combo grid — 2026-11-24

**Decision:** Train the MLP with early stopping and keep its grid to 6–8 combinations.
**Rationale:** The MLP was the largest remaining cost (~10–30 T4 GPU-hours at the old settings).
**Impact:** Grid values still to be set (`03`).

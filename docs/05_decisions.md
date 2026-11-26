# Data Notes, Decisions & Caveats
**Project:** COGS 118A Classifier Comparison
**Version:** v2
**Last Updated:** 2026-11-24

---

## 🔵 STATUS — 18 decisions logged (2026-11-04 planning, 2026-11-16 setup, 2026-11-24 speed metrics + replan after losing the lab server)

#13–17 were made in a separate 2026-11-24 chat and recorded in the claude.ai project doc `project-design.md`; Tom confirmed syncing them here the same day. #14 partly reverses #6; #13 replaces the launcher part of #11.

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

## 12. Add speed metrics alongside accuracy — 2026-11-24

**Decision:** Every job also records tuning time, final fit time, prediction time per 1,000 rows and throughput. Prediction speed is timed on a fixed 10,000-row test sample (median of 5 repeats after a warm-up), and GPU conditions are logged at the start of each job.
**Rationale (from 2026-11-24 discussion):** Tom asked for a metric of how fast each classifier is. The paper measured none, so it's a novelty point, and the classifiers differ in where their cost falls (KNN/TabPFN at prediction time, trees at training time). A fixed-size timing sample keeps per-row numbers comparable across partitions and avoids re-predicting huge test sets several times.
**Impact:** A few timer lines in the experiment script (action item #10) and new fields in each results file. Adds a speed table (`02` §6) and accuracy-vs-speed figures (`04`). Timings are only comparable within one GPU type (each job logs its GPU name), and are noisy on shared GPUs (#11), so present them as relative comparisons.

## 13. Lab server lost; move to cloud GPUs (Turing or newer) — 2026-11-24

**Decision:** yggdrasil is no longer available. Run on UCSD DataHub / DSMLP first (access pending — email datahub@ucsd.edu; request a Turing-or-newer GPU type, since the pool also has 1080 Tis), Kaggle as backup (2× T4 16 GB, ~30 GPU-h/week, 12-h sessions, batch "Save version" jobs), Colab free T4 for overflow only. The runner must be resumable: each job writes its own results file and skips itself if the file exists.
**Rationale (per project doc):** Personal GTX 1080 Ti ruled out — it's a Pascal GPU, and CUDA 13 / RAPIDS 24.02+ need newer cards. Turing+ keeps the existing environment usable. Resumable runner so a cut-off session loses one job, not the sweep. Why yggdrasil became unavailable: not stated.
**Impact:** Replaces the multi-GPU launcher part of #11. Data prep must be re-run on the new platform. Speed timings only comparable within one GPU type (#12).

## 14. Covertype and HIGGS capped at 100k rows; no large-data pitch — 2026-11-24 (sizes superseded by #19)

**Decision:** Covertype and HIGGS each use one stratified, fixed-seed 100k-row subsample taken before any splits. No full-data side runs.
**Rationale (per project doc):** Compute cut after losing the lab server. Dropping the large-data novelty pitch keeps every classifier compared on the same data sizes.
**Impact:** Partly reverses #6 (bigger datasets). Largest training set is 80k rows (80/20), still 20× the paper's 4k. The "dataset size" rubric item now rests on 100k-row datasets rather than 500k–1M.

## 15. SVM training cap 50k → 10k — 2026-11-24 (superseded by #19)

**Decision:** SVM trains on at most 10k rows (stratified subsample of the training split), same as TabPFN.
**Rationale (per project doc):** Runtime. (TabPFN's 10k cap is a hard model limit; SVM's is a compute choice.)
**Impact:** SVM and TabPFN are the only models that see different training data. Document both caps in the report.

## 16. Tune on trial 1 only — 2026-11-24 (superseded by #20)

**Decision:** Grid search runs once per dataset × partition × model, on trial 1. Trials 2–3 refit those best settings on their own training split and score on their own test split.
**Rationale (per project doc):** Part of the compute cuts. The 3 trials then measure split-to-split variation, not tuning variation — say so in the report.
**Impact:** Validation accuracy is one value per cell (from trial-1 CV). Tuning time (#12) is recorded on trial 1 only. The runner must finish trial 1 before trials 2–3 of the same cell.

## 17. MLP early stopping + small grid; 3-fold CV fallback — 2026-11-24

**Decision:** MLP uses early stopping and a grid of 6–8 combinations. If the timing test shows the sweep is still too slow, drop to 3-fold CV and document it.
**Rationale (per project doc):** Part of the compute cuts after losing the lab server.
**Impact:** Exact MLP grid values still to set (action item #15).

## 18. Balance every dataset to 50/50 — 2026-11-24 (reversed by #23: no dataset is balanced)

**Decision:** All four datasets are balanced to exactly 50% positive by randomly undersampling the larger class (fixed seed), after cleaning and before the 100k cap. Splits stay stratified, so all train/test splits are 50/50.
**Rationale:** Not stated (Tom's request, 2026-11-24).
**Impact:** Adult drops from 45,222 to about 22.4k rows and loses its natural 25% positive rate; the paper used Adult unbalanced, so Adult results are less comparable to the paper's. Letter, Covertype and HIGGS were already near 50/50 and barely change. Accuracy baseline is 50% everywhere. Drops the "imbalanced dataset" angle from the rubric's dataset-challenge item. `src/config.py` `BALANCE_CLASSES = True`; `balance()` in `src/prepare_data.py`.

## 19. Every dataset capped at 12,500 rows; no per-model caps — 2026-11-26

**Decision:** All four datasets use one stratified, fixed-seed 12,500-row subsample taken before any splits. Training sizes are 2,500 / 6,250 / 10,000 rows for 20/80, 50/50, 80/20. The 80/20 split fits TabPFN's 10k limit exactly, so the SVM and TabPFN training caps are dropped.
**Rationale:** Every classifier sees identical rows in every cell, as in the paper (every model trained on the same 4,000 rows per trial, §2.4 / Table 1). A flat 10k cap on *training* rows was considered and rejected: on the bigger datasets all three partitions would train on 10k rows, which would erase the 20/80 vs 80/20 comparison the rubric grades. Much smaller data also makes tuning every trial (#20) affordable.
**Impact:** Supersedes the sizes in #14 and the cap in #15. Largest training set is 10k rows (2.5× the paper's 4k), not 80k. Test sets shrink to 2,500 rows at 80/20 (the paper used large final test sets) — state this. Weakens the rubric's "how large are the datasets" item (10 pts); see open question in `04`. `TIMING_N_ROWS` → 2,500 (smallest test split).

## 20. Tune in every trial — 2026-11-26

**Decision:** `GridSearchCV` (5-fold) runs on each trial's own training split, for every dataset × partition × trial × model. Replaces #16.
**Rationale:** Matches the paper (best parameters chosen on each trial's own 1k validation set) and the rubric ("each time you always report the best accuracy under the chosen hyper-parameter"). Affordable after #19.
**Impact:** Validation accuracy is now a mean over 3 trials, like train and test. Tuning time recorded every job. Trials no longer depend on each other, so the runner can run jobs in any order. Closes TA question on trial-1 tuning.

## 21. Adult stays unbalanced — 2026-11-26

**Decision:** Amends #18. Adult keeps its natural class mix (~24.8% positive; the paper's Adult was 25%). Letter, Covertype and HIGGS stay balanced 50/50. Splits stay stratified.
**Rationale:** Tom's call: keep one dataset unbalanced and note it. Adult then matches the paper's version directly. The paper did not rebalance anything; it reported class balance as a %POZ column in its dataset table.
**Impact:** Report a "% positive" column in the dataset table (as the paper did) and show the majority-class baseline (~75%) next to Adult accuracies. `BALANCE_CLASSES` in `src/config.py` is now a per-dataset dict; `dataset_summary.json` records `balanced` and `majority_class_baseline`.

## 22. No TA check-in on design choices — 2026-11-26

**Decision:** Not sending the TA questions (#4, #5, #32, #33 in `07`).
**Rationale:** TabPFN is the project's novelty and fits the rubric's bonus item (b). #19–21 remove the other questions (no per-model caps, tuning every trial, Adult matches the paper).
**Impact:** Closes those action items.

## 23. No rebalancing; every dataset keeps its natural class rate — 2026-11-26

**Decision:** Reverses #18 and extends #21. Letter (~50.3% positive), Covertype (~48.8%) and HIGGS (~53.0%) are no longer undersampled to 50/50; Adult stays at ~24.8%. The 12,500-row subsample and all splits stay stratified, so each keeps its rate. The `balance()` code stays in `prepare_data.py`, switched off in `config.BALANCE_CLASSES`.
**Rationale:** Tom's call: leave natural rates. Matches the paper, which did not rebalance and reported %POZ per dataset. The three were already near 50/50, so balancing changed little and was one more thing to explain.
**Impact:** Dataset table gets a "% positive" column and a majority-class baseline per dataset (Adult ~75.2%, Letter ~50.3%, Covertype ~51.2%, HIGGS ~53.0%).

## 24. Hyperparameter grids proposed from the paper — 2026-11-26 (PROPOSED, pending Tom's OK)

**Decision:** Grids in `03` "Hyperparameter grids", adapted from the paper's §2.1 to 12.5k-row data and 5-fold CV. Also proposed: KNN on scikit-learn (CPU) instead of cuML, because the paper used distance-weighted KNN and cuML's KNN classifier supports uniform weights only.
**Rationale:** Tom asked for standard grids based on the paper.
**Impact:** Closes #15 once approved; values then go into `src/config.py` for the experiment script.


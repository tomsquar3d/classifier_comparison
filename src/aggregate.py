"""Step 5 of the pipeline: turn results/*.json into tables and figures.

Works on whatever trials exist so far (means and std over the trials found;
std is blank with one trial). Re-run after each trial finishes.

Outputs
  results/summary/main_table.csv     every dataset x partition x model:
                                      train / val / test mean + std, n trials,
                                      best params per trial
  results/summary/main_table.md      the same, compact, for docs/02
  results/summary/ranks.csv          rank of each model (1 = best test acc)
                                      per dataset x partition, + mean rank
  results/summary/speed_table.csv    tune / fit / predict time per model x
                                      dataset (80/20 partition)
  figures/fig1_test_acc_vs_train_size.{png,pdf}   comparisons (a) and (b)
  figures/fig2_rank_table.{png,pdf}                paper-style rank table
  figures/fig3_svm_heatmaps.{png,pdf}              CV train/val acc over C x gamma
  figures/fig4_knn_curves.{png,pdf}                CV train/val acc vs K
  figures/fig5_accuracy_vs_speed.{png,pdf}         bonus: accuracy vs predict time
  figures/fig6_confusion_adult.{png,pdf}           confusion matrices, Adult 80/20
  figures/fig7_bootstrap_ranks.{png,pdf}           paper-style P(rank) table
  figures/fig8_critical_difference.{png,pdf}       Friedman + Nemenyi CD diagram
  results/summary/significance.json   Friedman test, Nemenyi CD, mean ranks
  results/summary/bootstrap_ranks.csv P(model finishes 1st..6th), bootstrap

Usage:
    python src/aggregate.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402

SUMMARY_DIR = C.RESULTS_DIR / "summary"
FIG_DIR = C.ROOT / "figures"

# ---------------------------------------------------------------- style
# Reference categorical palette (dataviz skill, light mode), slots 1-6 in
# fixed order; validated for adjacent pairs. Colour follows the model.
MODEL_COLORS = {"rf": "#257de0", "xgboost": "#f47230", "svm": "#00b894",
                "ann": "#f2a600", "knn": "#ee7fac", "tabpfn": "#008a17"}
MODEL_MARKERS = {"rf": "o", "xgboost": "s", "svm": "^", "ann": "D",
                 "knn": "v", "tabpfn": "P"}
MODEL_LABELS = {"rf": "Random forest", "xgboost": "XGBoost", "svm": "SVM (RBF)",
                "ann": "MLP", "knn": "KNN", "tabpfn": "TabPFN"}
DATASET_LABELS = {"adult": "Adult", "letter": "Letter (A–M vs N–Z)",
                  "covertype": "Covertype", "higgs": "HIGGS"}
PART_LABELS = {"20_80": "20/80", "50_50": "50/50", "80_20": "80/20"}
TEXT, TEXT2, GRID = "#191919", "#666666", "#e6e6e6"
SEQ_CMAP = "Blues"   # sequential = one hue, light -> dark

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.edgecolor": TEXT2, "axes.labelcolor": TEXT, "xtick.color": TEXT2,
    "ytick.color": TEXT2, "text.color": TEXT, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "legend.frameon": False, "savefig.dpi": 200,
    "savefig.bbox": "tight", "figure.facecolor": "white",
})


def save(fig, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"{name}.{ext}")
    plt.close(fig)
    print(f"  figures/{name}.png/.pdf")


# ---------------------------------------------------------------- load
def load() -> tuple[pd.DataFrame, list[dict]]:
    raw = [json.loads(p.read_text()) for p in sorted(C.RESULTS_DIR.glob("*_trial*.json"))]
    raw = [r for r in raw if not r.get("smoke") and r.get("device") == "gpu"]
    if not raw:
        sys.exit("No GPU results in results/.")
    df = pd.DataFrame([{k: r[k] for k in
                        ["dataset", "model", "partition", "trial", "n_train",
                         "train_acc", "val_acc", "test_acc", "test_majority_baseline",
                         "tune_time_s", "fit_time_s", "predict_ms_per_1k"]}
                       | {"best_params": json.dumps(r["best_params"]),
                          "gpu": r["gpu_at_start"].get("name")} for r in raw])
    return df, raw


def order(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["dataset"] = pd.Categorical(df["dataset"], C.DATASETS, ordered=True)
    df["model"] = pd.Categorical(df["model"], C.MODELS, ordered=True)
    df["partition"] = pd.Categorical(df["partition"], list(C.PARTITIONS), ordered=True)
    return df


# ---------------------------------------------------------------- tables
def main_table(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["dataset", "partition", "model"], observed=True)
    t = g.agg(n_trials=("trial", "nunique"),
              train_mean=("train_acc", "mean"), train_std=("train_acc", "std"),
              val_mean=("val_acc", "mean"), val_std=("val_acc", "std"),
              test_mean=("test_acc", "mean"), test_std=("test_acc", "std"),
              baseline=("test_majority_baseline", "mean")).reset_index()
    params = (df.sort_values("trial").groupby(["dataset", "partition", "model"], observed=True)
              ["best_params"].apply(lambda s: " | ".join(s)).reset_index())
    return t.merge(params, on=["dataset", "partition", "model"])


def write_markdown(t: pd.DataFrame, ranks: pd.DataFrame, n_trials: int) -> None:
    lines = [f"Mean over {n_trials} trial(s). Cells: train / val / test accuracy "
             "(test ± std when ≥2 trials). Generated by `src/aggregate.py`.", ""]
    for d in C.DATASETS:
        sub = t[t.dataset == d]
        if sub.empty:
            continue
        lines += [f"**{DATASET_LABELS[d]}** (majority-class baseline "
                  f"{sub.baseline.iloc[0]:.3f})", "",
                  "| Model | " + " | ".join(PART_LABELS[p] for p in C.PARTITIONS) + " |",
                  "|---|" + "---|" * len(C.PARTITIONS)]
        for m in C.MODELS:
            cells = []
            for p in C.PARTITIONS:
                r = sub[(sub.model == m) & (sub.partition == p)]
                if r.empty:
                    cells.append("—")
                    continue
                r = r.iloc[0]
                sd = f" ± {r.test_std:.3f}" if r.n_trials > 1 else ""
                cells.append(f"{r.train_mean:.3f} / {r.val_mean:.3f} / **{r.test_mean:.3f}**{sd}")
            lines.append(f"| {MODEL_LABELS[m]} | " + " | ".join(cells) + " |")
        lines.append("")
    mr = ranks.groupby("model", observed=True)["rank"].mean().reindex(C.MODELS)
    lines += ["**Mean rank by test accuracy** (1 = best; over all dataset × partition cells)", "",
              "| " + " | ".join(MODEL_LABELS[m] for m in mr.index) + " |",
              "|" + "---|" * len(mr),
              "| " + " | ".join(f"{v:.2f}" for v in mr.values) + " |", ""]
    (SUMMARY_DIR / "main_table.md").write_text("\n".join(lines))


def rank_table(t: pd.DataFrame) -> pd.DataFrame:
    r = t[["dataset", "partition", "model", "test_mean"]].copy()
    r["rank"] = r.groupby(["dataset", "partition"], observed=True)["test_mean"].rank(
        ascending=False, method="average")
    return r


def speed_table(df: pd.DataFrame) -> pd.DataFrame:
    s = df[df.partition == "80_20"].groupby(["model", "dataset"], observed=True).agg(
        tune_s=("tune_time_s", "median"), fit_s=("fit_time_s", "median"),
        predict_ms_per_1k=("predict_ms_per_1k", "median"),
        test_acc=("test_acc", "mean")).reset_index()
    return s


# ---------------------------------------------------------------- figures
def fig1_curves(t: pd.DataFrame) -> None:
    """Test accuracy vs training share, one panel per dataset, one line per
    model. Shows (a) model spread at each partition and (b) each model's
    trend with more training data."""
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.1), sharey=False)
    x = np.array([C.PARTITIONS[p] for p in C.PARTITIONS]) * 100
    for ax, d in zip(axes, C.DATASETS):
        sub = t[t.dataset == d]
        for m in C.MODELS:
            s = sub[sub.model == m].sort_values("partition")
            if s.empty:
                continue
            ax.plot(x[:len(s)], s.test_mean, color=MODEL_COLORS[m], lw=2,
                    marker=MODEL_MARKERS[m], ms=5, label=MODEL_LABELS[m],
                    markeredgecolor="white", markeredgewidth=0.8)
            if s.n_trials.max() > 1:
                ax.fill_between(x[:len(s)], s.test_mean - s.test_std,
                                s.test_mean + s.test_std, color=MODEL_COLORS[m],
                                alpha=0.12, lw=0)
        base = sub.baseline.iloc[0]
        lo = sub.test_mean.min()
        if base > lo - 0.05:       # only draw when it's near the data
            ax.axhline(base, color=TEXT2, lw=1, ls="--")
            ax.text(x[-1], base, " majority class", color=TEXT2, fontsize=7,
                    va="bottom", ha="right")
        ax.set_title(DATASET_LABELS[d])
        ax.set_xticks(x, [f"{int(v)}%" for v in x])
        ax.set_xlabel("Training share")
    axes[0].set_ylabel("Test accuracy")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    save(fig, "fig1_test_acc_vs_train_size")


def fig2_ranks(t: pd.DataFrame, ranks: pd.DataFrame) -> None:
    """Paper-style table: test accuracy per cell, shaded by rank."""
    cols = [(d, p) for d in C.DATASETS for p in C.PARTITIONS]
    acc = np.full((len(C.MODELS), len(cols)), np.nan)
    rk = np.full_like(acc, np.nan)
    for j, (d, p) in enumerate(cols):
        for i, m in enumerate(C.MODELS):
            r = ranks[(ranks.dataset == d) & (ranks.partition == p) & (ranks.model == m)]
            if not r.empty:
                acc[i, j], rk[i, j] = r.test_mean.iloc[0], r["rank"].iloc[0]
    mean_rank = np.nanmean(rk, axis=1)
    idx = np.argsort(mean_rank)
    acc, rk, mean_rank = acc[idx], rk[idx], mean_rank[idx]
    models = [C.MODELS[i] for i in idx]

    fig, ax = plt.subplots(figsize=(11, 2.9))
    shade = (len(C.MODELS) + 1 - rk) / len(C.MODELS)   # rank 1 = darkest
    ax.imshow(shade, cmap=SEQ_CMAP, vmin=0, vmax=1.25, aspect="auto")
    for i in range(acc.shape[0]):
        for j in range(acc.shape[1]):
            if np.isnan(acc[i, j]):
                continue
            dark = shade[i, j] > 0.6
            ax.text(j, i, f"{acc[i, j]:.3f}", ha="center", va="center", fontsize=7.5,
                    color="white" if dark else TEXT,
                    fontweight="bold" if rk[i, j] == 1 else "normal")
    ax.set_xticks(range(len(cols)), [PART_LABELS[p] for _, p in cols], fontsize=7.5)
    for k, d in enumerate(C.DATASETS):
        ax.text(k * 3 + 1, -0.9, DATASET_LABELS[d], ha="center", va="bottom", fontsize=9)
        if k:
            ax.axvline(k * 3 - 0.5, color="white", lw=3)
    ax.set_yticks(range(len(models)),
                  [f"{MODEL_LABELS[m]}  (mean rank {r:.2f})" for m, r in zip(models, mean_rank)])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title("Test accuracy by dataset and train/test split — darker = higher rank "
                 "(bold = best in column)", pad=34, fontsize=9.5)
    save(fig, "fig2_rank_table")


def cv_frame(raw: list[dict], model: str, dataset: str, partition: str, trial: int):
    r = [x for x in raw if (x["model"], x["dataset"], x["partition"], x["trial"])
         == (model, dataset, partition, trial)]
    if not r:
        return None
    cv = r[0]["cv_results"]
    f = pd.DataFrame(cv["params"])
    f["train"], f["val"] = cv["mean_train_score"], cv["mean_test_score"]
    return f


def fig3_svm(raw: list[dict]) -> None:
    """CV train / validation accuracy over the SVM grid (trial 1, 80/20)."""
    fig, axes = plt.subplots(2, 4, figsize=(12, 5.0), gridspec_kw={"wspace": 0.12, "hspace": 0.45})
    for j, d in enumerate(C.DATASETS):
        f = cv_frame(raw, "svm", d, "80_20", 1)
        if f is None:
            continue
        for i, kind in enumerate(["train", "val"]):
            piv = f.pivot(index="svc__C", columns="svc__gamma", values=kind)
            ax = axes[i, j]
            im = ax.imshow(piv.values, cmap=SEQ_CMAP, vmin=0.5, vmax=1.0,
                           origin="lower", aspect="auto")
            best = np.unravel_index(np.nanargmax(f.pivot(index="svc__C", columns="svc__gamma",
                                                         values="val").values), piv.shape)
            ax.scatter(best[1], best[0], marker="o", s=60, facecolor="none",
                       edgecolor="#ec4647", lw=1.5)
            ax.set_xticks(range(len(piv.columns)), [f"{g:g}" for g in piv.columns],
                          rotation=60, fontsize=6.5)
            ax.set_yticks(range(len(piv.index)),
                          [f"{c:g}" for c in piv.index] if j == 0 else [], fontsize=6.5)
            ax.grid(False)
            if i == 0:
                ax.set_title(DATASET_LABELS[d])
            if i == 1:
                ax.set_xlabel("gamma (kernel width)")
            if j == 0:
                ax.set_ylabel(f"C\n{'CV train' if kind == 'train' else 'CV validation'} acc.")
    fig.colorbar(im, ax=axes, shrink=0.8, label="Accuracy")
    fig.suptitle("SVM: cross-validated accuracy over the C × gamma grid "
                 "(80/20, trial 1; circle = chosen setting)", fontsize=10)
    save(fig, "fig3_svm_heatmaps")


def fig4_knn(raw: list[dict]) -> None:
    """CV train / validation accuracy vs K (trial 1, 80/20)."""
    fig, axes = plt.subplots(1, 4, figsize=(11, 2.9), sharey=False)
    for ax, d in zip(axes, C.DATASETS):
        f = cv_frame(raw, "knn", d, "80_20", 1)
        if f is None:
            continue
        for w, ls in [("uniform", "-"), ("distance", "--")]:
            s = f[f.knn__weights == w].sort_values("knn__n_neighbors")
            ax.plot(s.knn__n_neighbors, s.val, color=TEXT, lw=2, ls=ls,
                    label=f"validation, {w}")
            ax.plot(s.knn__n_neighbors, s.train, color="#9c9c9c", lw=2,
                    ls=ls, label=f"train, {w}")
        ax.set_xscale("log")
        ax.set_title(DATASET_LABELS[d])
        ax.set_xlabel("K (log scale)")
    axes[0].set_ylabel("CV accuracy")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.1))
    fig.suptitle("KNN: cross-validated train vs validation accuracy as K grows "
                 "(80/20, trial 1)", fontsize=10, y=1.03)
    fig.tight_layout()
    save(fig, "fig4_knn_curves")


def fig5_speed(speed: pd.DataFrame) -> None:
    """Bonus: mean test accuracy vs prediction time, 80/20, averaged over
    datasets. Direct labels + marker shapes (6 series > 3-colour scatter cap)."""
    s = speed.groupby("model", observed=True).agg(
        acc=("test_acc", "mean"), pred=("predict_ms_per_1k", "median"),
        tune=("tune_s", "median")).reindex(C.MODELS).dropna()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.3))
    for ax, col, lab in [(axes[0], "pred", "Prediction time (ms per 1,000 rows, log)"),
                         (axes[1], "tune", "Tuning time per job (s, log)")]:
        for m, r in s.iterrows():
            ax.scatter(r[col], r.acc, s=70, color=MODEL_COLORS[m], marker=MODEL_MARKERS[m],
                       edgecolor="white", linewidth=1.5, zorder=3)
            dx, dy, ha = LABEL_OFFSETS.get((m, col), (6, 4, "left"))
            ax.annotate(MODEL_LABELS[m], (r[col], r.acc), xytext=(dx, dy), ha=ha,
                        textcoords="offset points", fontsize=8, color=TEXT)
        ax.set_xscale("log")
        ax.set_xlabel(lab)
    axes[0].set_ylabel("Mean test accuracy (4 datasets)")
    fig.suptitle("Accuracy vs speed, 80/20 split (same GPU type for every job)",
                 fontsize=10)
    fig.tight_layout()
    save(fig, "fig5_accuracy_vs_speed")


LABEL_OFFSETS = {("rf", "pred"): (0, -13, "center")}  # avoid overlapping XGBoost


def fig6_confusion(raw: list[dict], dataset: str = "adult") -> None:
    rs = {r["model"]: r for r in raw if r["dataset"] == dataset
          and r["partition"] == "80_20" and r["trial"] == 1}
    if not rs:
        return
    fig, axes = plt.subplots(1, 6, figsize=(12, 2.3), gridspec_kw={"wspace": 0.15})
    for ax, m in zip(axes, C.MODELS):
        if m not in rs:
            ax.axis("off")
            continue
        cm = np.array(rs[m]["confusion_matrix"]["rows_true_cols_pred"])
        frac = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(frac, cmap=SEQ_CMAP, vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}\n{frac[i, j]:.0%}", ha="center", va="center",
                        fontsize=7, color="white" if frac[i, j] > 0.6 else TEXT)
        ax.set_xticks([0, 1], ["≤50K", ">50K"], fontsize=7)
        ax.set_yticks([0, 1], ["≤50K", ">50K"] if m == C.MODELS[0] else [], fontsize=7)
        ax.grid(False)
        ax.set_title(f"{MODEL_LABELS[m]}\nacc {rs[m]['test_acc']:.3f}", fontsize=8)
        ax.set_xlabel("Predicted", fontsize=7)
    axes[0].set_ylabel("True", fontsize=7)
    fig.suptitle(f"{DATASET_LABELS[dataset]} test-set confusion matrices "
                 "(80/20, trial 1; % of each true class)", fontsize=10, y=1.12)
    save(fig, f"fig6_confusion_{dataset}")


# ---------------------------------------------------------------- significance
# Nemenyi critical values q_0.05 (Demsar 2006, Table 5), by number of models k
NEMENYI_Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949,
               8: 3.031, 9: 3.102, 10: 3.164}
N_BOOT = 1000


def significance(t: pd.DataFrame) -> dict:
    """Friedman test over the dataset x partition cells (blocks), using each
    model's mean test accuracy over the 3 trials, then the Nemenyi critical
    difference for mean ranks (Demsar 2006)."""
    from scipy.stats import friedmanchisquare
    wide = t.pivot_table(index=["dataset", "partition"], columns="model",
                         values="test_mean", observed=True)[C.MODELS].dropna()
    ranks = wide.rank(axis=1, ascending=False)
    n, k = wide.shape
    stat, pval = friedmanchisquare(*[wide[m].values for m in C.MODELS])
    cd = NEMENYI_Q05[k] * np.sqrt(k * (k + 1) / (6 * n))
    out = {"n_blocks": int(n), "n_models": int(k),
           "blocks": "dataset x partition cells, mean test accuracy over trials",
           "friedman_chi2": float(stat), "friedman_p": float(pval),
           "nemenyi_cd_alpha_0.05": float(cd),
           "mean_rank": ranks.mean().round(4).to_dict()}
    (SUMMARY_DIR / "significance.json").write_text(json.dumps(out, indent=1))
    return out


def bootstrap_ranks(df: pd.DataFrame, seed: int = C.BASE_SEED) -> pd.DataFrame:
    """Paper-style rank probabilities (its Table 4 bootstraps over problems
    and metrics). Each round: draw the 12 dataset x partition cells with
    replacement, and for each drawn cell one of its 3 trials at random;
    average each model's test accuracy over the drawn cells and rank the
    models. Repeat N_BOOT times; report how often each model lands 1st..kth."""
    rng = np.random.default_rng(seed)
    acc = df.pivot_table(index=["dataset", "partition", "trial"], columns="model",
                         values="test_acc", observed=True)[C.MODELS]
    cells = acc.index.droplevel("trial").unique()
    k = len(C.MODELS)
    counts = np.zeros((k, k))
    for _ in range(N_BOOT):
        draw = cells[rng.integers(len(cells), size=len(cells))]
        rows = [acc.loc[(d, p)].iloc[rng.integers(len(acc.loc[(d, p)]))] for d, p in draw]
        mean = np.mean(np.vstack([r.values for r in rows]), axis=0)
        order_ = np.argsort(-mean)
        for place, mi in enumerate(order_):
            counts[mi, place] += 1
    probs = pd.DataFrame(counts / N_BOOT, index=C.MODELS,
                         columns=[f"rank_{i + 1}" for i in range(k)])
    probs["expected_rank"] = (probs.values * np.arange(1, k + 1)).sum(axis=1)
    probs = probs.sort_values("expected_rank")
    probs.to_csv(SUMMARY_DIR / "bootstrap_ranks.csv", float_format="%.3f")
    return probs


def fig7_bootstrap(probs: pd.DataFrame) -> None:
    k = len(C.MODELS)
    vals = probs[[f"rank_{i + 1}" for i in range(k)]].values
    fig, ax = plt.subplots(figsize=(7.5, 2.6))
    ax.imshow(vals, cmap=SEQ_CMAP, vmin=0, vmax=1, aspect="auto")
    for i in range(k):
        for j in range(k):
            ax.text(j, i, f"{vals[i, j]:.3f}", ha="center", va="center", fontsize=8,
                    color="white" if vals[i, j] > 0.55 else TEXT)
    ords = ["1st", "2nd", "3rd"] + [f"{i}th" for i in range(4, k + 1)]
    ax.set_xticks(range(k), ords)
    ax.xaxis.tick_top()
    ax.set_yticks(range(k), [MODEL_LABELS[m] for m in probs.index])
    ax.grid(False)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(f"How often each model ranks 1st–{k}th by mean test accuracy "
                 f"({N_BOOT:,} bootstrap samples of cells and trials)", fontsize=9, pad=24)
    save(fig, "fig7_bootstrap_ranks")


def fig8_cd(sig: dict) -> None:
    """Critical-difference diagram: models on a mean-rank axis; models joined
    by a black bar are NOT significantly different (Nemenyi, alpha 0.05)."""
    mr = pd.Series(sig["mean_rank"]).sort_values()
    cd, k = sig["nemenyi_cd_alpha_0.05"], sig["n_models"]
    rs = mr.values
    cliques = []
    for i in range(len(rs)):
        j = i
        while j + 1 < len(rs) and rs[j + 1] - rs[i] < cd:
            j += 1
        if j > i and not any(a_ <= i and j <= b_ for a_, b_ in cliques):
            cliques.append((i, j))

    y_axis, bar_gap, lab_gap = 1.0, 0.07, 0.13
    y_bars0 = y_axis - 0.09
    y_lab0 = y_bars0 - bar_gap * len(cliques) - 0.06
    half = (len(mr) + 1) // 2
    y_bottom = y_lab0 - lab_gap * (half - 1) - 0.08
    fig, ax = plt.subplots(figsize=(7.5, 2.9))
    ax.set_xlim(k + 0.35, 0.65)         # best (rank 1) on the right
    ax.set_ylim(y_bottom, y_axis + 0.33)
    ax.axis("off")
    ax.plot([1, k], [y_axis, y_axis], color=TEXT2, lw=1)
    for r in range(1, k + 1):
        ax.plot([r, r], [y_axis, y_axis + 0.03], color=TEXT2, lw=1)
        ax.text(r, y_axis + 0.05, str(r), ha="center", va="bottom", fontsize=8, color=TEXT2)
    y_cd = y_axis + 0.2
    ax.plot([k, k - cd], [y_cd, y_cd], color=TEXT, lw=1.5)
    for x in (k, k - cd):
        ax.plot([x, x], [y_cd - 0.02, y_cd + 0.02], color=TEXT, lw=1.5)
    ax.text(k - cd - 0.08, y_cd, f"critical difference = {cd:.2f}", ha="left",
            va="center", fontsize=8)
    for n_, (a_, b_) in enumerate(cliques):
        yb = y_bars0 - bar_gap * n_
        ax.plot([rs[a_] - 0.04, rs[b_] + 0.04], [yb, yb], color=TEXT, lw=3,
                solid_capstyle="round", zorder=3)
    for i, (m, r) in enumerate(mr.items()):
        right = i < half
        y = y_lab0 - lab_gap * (i if right else len(mr) - 1 - i)
        x_lab = 0.8 if right else k + 0.2
        ax.plot([r, r, x_lab], [y_axis, y, y], color=MODEL_COLORS[m], lw=1.5, zorder=2)
        ax.text(x_lab - 0.04 if right else x_lab + 0.04, y, f"{MODEL_LABELS[m]} ({r:.2f})",
                ha="left" if right else "right", va="center", fontsize=8)
    ax.set_title(f"Mean rank over {sig['n_blocks']} dataset × split cells "
                 f"(Friedman p = {sig['friedman_p']:.1e}).\nModels joined by a "
                 "black bar are not significantly different (Nemenyi, α = 0.05).",
                 fontsize=9)
    save(fig, "fig8_critical_difference")


# ---------------------------------------------------------------- main
def main() -> None:
    df, raw = load()
    df = order(df)
    gpus = df.gpu.dropna().unique()
    n_trials = df.trial.nunique()
    print(f"{len(df)} results, trials {sorted(df.trial.unique())}, GPU(s): {list(gpus)}")
    if len(gpus) > 1:
        print("  ⚠️ more than one GPU type: speed numbers are not comparable (docs/01)")

    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    t = main_table(df)
    t.to_csv(SUMMARY_DIR / "main_table.csv", index=False, float_format="%.4f")
    ranks = rank_table(t)
    ranks.to_csv(SUMMARY_DIR / "ranks.csv", index=False, float_format="%.4f")
    speed = speed_table(df)
    speed.to_csv(SUMMARY_DIR / "speed_table.csv", index=False, float_format="%.4f")
    write_markdown(t, ranks, n_trials)
    print("  results/summary/main_table.csv/.md, ranks.csv, speed_table.csv")

    fig1_curves(t)
    fig2_ranks(t, ranks)
    fig3_svm(raw)
    fig4_knn(raw)
    fig5_speed(speed)
    fig6_confusion(raw, "adult")
    if n_trials >= 2:
        sig = significance(t)
        print(f"  Friedman chi2 {sig['friedman_chi2']:.2f}, p {sig['friedman_p']:.2e}; "
              f"Nemenyi CD {sig['nemenyi_cd_alpha_0.05']:.2f}")
        probs = bootstrap_ranks(df)
        print(probs.round(3).to_string())
        fig7_bootstrap(probs)
        fig8_cd(sig)


if __name__ == "__main__":
    main()

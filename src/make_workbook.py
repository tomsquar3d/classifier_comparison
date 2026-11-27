"""Build results/summary/classifier_metrics.xlsx from the summary CSVs.

Run after src/aggregate.py. Sheets:
  Rankings by metric  mean rank of each model on each of the paper's 8
                      metrics, their mean, extra metrics and time; bootstrap
                      P(rank) over cells and metrics
  Accuracy rankings   accuracy-only ranks: overall, by dataset, by split,
                      cells won, bootstrap P(rank)
  All metrics         one row per dataset x split x model: every metric
                      (mean over 3 trials), times, best params per trial
  Per job             one row per job (216): every metric, every trial
  Notes               what each measure means

Usage:
    python src/make_workbook.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402

S = C.RESULTS_DIR / "summary"
OUT = S / "classifier_metrics.xlsx"
MODEL = {"rf": "Random forest", "xgboost": "XGBoost", "svm": "SVM (RBF)",
         "ann": "MLP", "knn": "KNN", "tabpfn": "TabPFN"}
DATASET = {"adult": "Adult", "letter": "Letter", "covertype": "Covertype", "higgs": "HIGGS"}
PART = {"20_80": "20/80", "50_50": "50/50", "80_20": "80/20"}
HDR = PatternFill("solid", fgColor="F3F3F3")


def params_text(s: str) -> str:
    d = json.loads(s)
    if not d:
        return "—"
    return ", ".join(f"{k.split('__')[-1]}={tuple(v) if isinstance(v, list) else v}"
                     for k, v in d.items())


def add_sheet(wb, title, df, cols, fmts, widths, freeze="B2", first=False):
    ws = wb.active if first else wb.create_sheet(title)
    ws.title = title
    ws.append([c[1] for c in cols])
    for _, r in df.iterrows():
        ws.append([r[c[0]] for c in cols])
    for j, (key, _) in enumerate(cols, 1):
        ws.column_dimensions[get_column_letter(j)].width = widths.get(key, 12)
        h = ws.cell(1, j)
        h.fill, h.font = HDR, Font(bold=True)
        h.alignment = Alignment(wrap_text=True, vertical="top")
        if key in fmts:
            for i in range(2, len(df) + 2):
                ws.cell(i, j).number_format = fmts[key]
    ws.row_dimensions[1].height = 45
    ws.freeze_panes = freeze


def main() -> None:
    wb = Workbook()
    rk = pd.read_csv(S / "rankings.csv")
    ba = pd.read_csv(S / "bootstrap_ranks_all_metrics.csv", index_col=0)
    ba.index = ba.index.map(MODEL)
    rk = rk.merge(ba.add_prefix("ball_"), left_on="model", right_index=True, how="left")

    # 1. Rankings by metric
    m8 = ["ACC", "FSC", "LFT", "ROC", "APR", "BEP", "RMS", "MXE"]
    lab = {"ACC": "Accuracy", "FSC": "F-score", "LFT": "Lift (25%)", "ROC": "ROC area",
           "APR": "Avg. precision", "BEP": "Break-even point", "RMS": "RMS error",
           "MXE": "Cross-entropy", "PREC": "Precision", "REC": "Recall",
           "BAL_ACC": "Balanced accuracy", "TUNE_S": "Tune time",
           "FIT_S": "Fit time", "PREDICT_MS_1K": "Predict time"}
    cols = ([("model", "Model"), ("mean_rank_all8_metrics", "Mean rank, paper's 8 metrics")]
            + [(f"mean_rank_metric_{k}", f"Rank: {lab[k]}") for k in m8]
            + [(f"mean_rank_metric_{k}", f"Rank: {lab[k]}") for k in ["PREC", "REC", "BAL_ACC"]]
            + [(f"mean_rank_metric_{k}", f"Rank: {lab[k]} (fastest = 1)")
               for k in ["TUNE_S", "FIT_S", "PREDICT_MS_1K"]]
            + [(f"ball_rank_{i}", f"Bootstrap P(rank {i}), 8 metrics") for i in range(1, 7)]
            + [("ball_expected_rank", "Bootstrap expected rank, 8 metrics")])
    f = {c: "0.00" for c, _ in cols if c.startswith("mean_rank") or c == "ball_expected_rank"}
    f.update({c: "0.0%" for c, _ in cols if c.startswith("ball_rank_")})
    add_sheet(wb, "Rankings by metric", rk.sort_values("mean_rank_all8_metrics"),
              cols, f, {"model": 16}, first=True)

    # 2. Accuracy rankings
    cols = ([("model", "Model"), ("mean_rank", "Mean rank by accuracy (12 cells)"),
             ("cells_won", "Cells won (of 12)"), ("mean_test_acc", "Mean test accuracy")]
            + [(f"mean_rank_{d}", f"Rank: {d}") for d in DATASET.values()]
            + [(f"mean_rank_{p}", f"Rank: {p} split") for p in PART.values()]
            + [(f"bootstrap_P_rank_{i}", f"Bootstrap P(rank {i}), accuracy") for i in range(1, 7)]
            + [("bootstrap_expected_rank", "Bootstrap expected rank, accuracy")])
    f = {c: "0.00" for c, _ in cols if c.startswith("mean_rank") or c.endswith("expected_rank")}
    f.update({c: "0.0%" for c, _ in cols if c.startswith("bootstrap_P")})
    f["mean_test_acc"] = "0.0000"
    add_sheet(wb, "Accuracy rankings", rk.sort_values("mean_rank"), cols, f, {"model": 16})

    # 3. All metrics (cell means)
    am = pd.read_csv(S / "all_metrics.csv")
    for c in [c for c in am.columns if c.startswith("best_params")]:
        am[c] = am[c].map(params_text)
    acc4 = "0.0000"
    cols = ([("dataset", "Dataset"), ("partition", "Split (train/test)"), ("model", "Model"),
             ("rank", "Accuracy rank in cell"), ("test_mean", "Test accuracy (mean)"),
             ("test_std", "Test accuracy (std)"), ("val_mean", "Val accuracy (mean)"),
             ("train_mean", "Train accuracy (mean)"), ("baseline", "Majority baseline")]
            + [(k, lab[k]) for k in m8[1:]]
            + [(k, lab[k]) for k in ["PREC", "REC", "BAL_ACC"]]
            + [("tune_s", "Tune time (s, median)"), ("fit_s", "Final fit (s, median)"),
               ("predict_ms_per_1k", "Predict ms / 1k rows (median)"), ("n_train", "Training rows")]
            + [(f"best_params_trial{i}", f"Best params, trial {i}") for i in (1, 2, 3)])
    f = {k: acc4 for k in ["test_mean", "test_std", "val_mean", "train_mean", "baseline",
                           "FSC", "ROC", "APR", "BEP", "RMS", "MXE", "PREC", "REC", "BAL_ACC"]}
    f.update({"LFT": "0.000", "rank": "0.#", "n_train": "#,##0", "tune_s": "0.0",
              "fit_s": "0.00", "predict_ms_per_1k": "#,##0.00"})
    add_sheet(wb, "All metrics", am, cols, f,
              {"dataset": 11, "partition": 10, "model": 14, "best_params_trial1": 30,
               "best_params_trial2": 30, "best_params_trial3": 30}, freeze="D2")

    # 4. Per job
    pj = pd.read_csv(S / "per_job_metrics.csv")
    pj["dataset"] = pj.dataset.map(DATASET)
    pj["partition"] = pj.partition.map(PART)
    pj["model"] = pj.model.map(MODEL)
    cols = ([("dataset", "Dataset"), ("partition", "Split"), ("model", "Model"),
             ("trial", "Trial")]
            + [(k, lab[k]) for k in m8 + ["PREC", "REC", "BAL_ACC"]]
            + [("TUNE_S", "Tune time (s)"), ("FIT_S", "Fit time (s)"),
               ("PREDICT_MS_1K", "Predict ms / 1k rows"),
               ("acc_refit_diff", "Refit accuracy − sweep accuracy")])
    f = {k: acc4 for k in m8 + ["PREC", "REC", "BAL_ACC", "acc_refit_diff"]}
    f.update({"LFT": "0.000", "TUNE_S": "0.0", "FIT_S": "0.00", "PREDICT_MS_1K": "#,##0.00"})
    add_sheet(wb, "Per job", pj, cols, f, {"dataset": 11, "model": 14}, freeze="E2")

    # 5. Notes
    sig = json.loads((S / "significance.json").read_text())
    sa = json.loads((S / "significance_all_metrics.json").read_text())
    ws = wb.create_sheet("Notes")
    notes = [
        "About these tables",
        "Source: github.com/tomsquar3d/classifier_comparison, results/summary/ — made by "
        "src/aggregate.py and src/make_workbook.py from 216 jobs (4 datasets × 3 splits × "
        "3 trials × 6 models), all run on one GPU type on DSMLP, 2026-11-26.",
        "The paper's 8 metrics (Caruana & Niculescu-Mizil 2006, §2.2): accuracy, F-score, "
        "lift at 25%, ROC area, average precision, precision/recall break-even point, RMS "
        "error of the probability, cross-entropy. Lower is better for RMS and cross-entropy.",
        "Accuracy and F-score / precision / recall / balanced accuracy come from each "
        "sweep job's test predictions. The other six come from src/rescore.py: each job "
        "refit with its own best settings on the same split, scored on the same test set "
        "(refit accuracy matched the sweep exactly for 5 of 6 models; MLP within ±0.03).",
        "SVM has no probabilities without calibration: its decision values are min-max "
        "scaled to [0, 1] per test set (as for the paper's uncalibrated SVM). Its RMS and "
        "cross-entropy ranks reflect that, as in the paper.",
        "Rank: 1 = best of the 6 models in a dataset × split cell (mean over 3 trials), "
        "averaged over the 12 cells. Time ranks: fastest = 1.",
        f"Significance, accuracy only: Friedman p = {sig['friedman_p']:.1e}, Nemenyi "
        f"critical difference {sig['nemenyi_cd_alpha_0.05']:.2f} ranks (α = 0.05).",
        f"Significance, mean rank over the 8 metrics: Friedman p = {sa['friedman_p']:.1e}, "
        f"critical difference {sa['nemenyi_cd_alpha_0.05']:.2f} ranks.",
        "Bootstrap P(rank): share of 1,000 resamples in which the model came k-th. "
        "Accuracy version resamples cells and trials; 8-metric version resamples cells "
        "and metrics (the paper's Table 4 resampled problems and metrics).",
        "Times: medians over the 3 trials; prediction timed on a fixed 2,500-row test "
        "sample. Majority baseline = accuracy from always guessing the commoner class.",
    ]
    for n in notes:
        ws.append([n])
    ws["A1"].font = Font(bold=True)
    ws.column_dimensions["A"].width = 120
    for i in range(2, len(notes) + 1):
        ws.cell(i, 1).alignment = Alignment(wrap_text=True, vertical="top")
    wb.save(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

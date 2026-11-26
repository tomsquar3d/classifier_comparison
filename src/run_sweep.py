"""Step 3 of the pipeline: run many jobs, resumably.

Builds the job list (datasets x models x partitions x trials = 216 by
default), skips every job whose results file already exists, and runs the
rest one at a time. Each job runs in its own Python process, so a crash or
GPU memory leak in one job can't take down the sweep; the sweep keeps going
and lists failed jobs at the end. Re-run the same command to retry failures
and continue after a cut-off session. Per-job output goes to
results/logs/<job>.log.

Usage:
    python src/run_sweep.py --timing-test         # HIGGS, 80/20, trial 1, all 6 models
    python src/run_sweep.py                       # everything (216 jobs)
    python src/run_sweep.py --models rf svm --datasets adult
    python src/run_sweep.py --dry-run             # just list what would run
    python src/run_sweep.py --smoke               # tiny grids, quick end-to-end check

Run inside tmux/screen or with nohup on DSMLP so it survives a dropped SSH
connection, e.g.:
    nohup python src/run_sweep.py > results/logs/sweep.out 2>&1 &
"""
from __future__ import annotations

import argparse
import itertools
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
from run_job import output_dir, result_path  # noqa: E402

RUN_JOB = Path(__file__).resolve().parent / "run_job.py"


def job_list(datasets, models, partitions, trials):
    # Trial-major order: all of trial 1 first, so a partial sweep still
    # covers every dataset x model x partition cell at least once.
    return [(d, m, p, t) for t, d, p, m in
            itertools.product(trials, datasets, partitions, models)]


def fmt(sec: float) -> str:
    sec = int(sec)
    return f"{sec // 3600}h{sec % 3600 // 60:02d}m{sec % 60:02d}s"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--datasets", nargs="+", choices=C.DATASETS, default=C.DATASETS)
    ap.add_argument("--models", nargs="+", choices=C.MODELS, default=C.MODELS)
    ap.add_argument("--partitions", nargs="+", choices=list(C.PARTITIONS),
                    default=list(C.PARTITIONS))
    ap.add_argument("--trials", nargs="+", type=int,
                    choices=range(1, C.N_TRIALS + 1),
                    default=list(range(1, C.N_TRIALS + 1)))
    ap.add_argument("--timing-test", action="store_true",
                    help="worst case only: HIGGS, 80/20, trial 1, all models")
    ap.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--cv-folds", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.timing_test:
        args.datasets, args.partitions, args.trials = ["higgs"], ["80_20"], [1]

    out_dir = output_dir(args.smoke, args.device)
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    jobs = job_list(args.datasets, args.models, args.partitions, args.trials)
    todo = [j for j in jobs if not result_path(*j, results_dir=out_dir).exists()]
    print(f"{len(jobs)} jobs selected, {len(jobs) - len(todo)} already done, "
          f"{len(todo)} to run -> {out_dir}", flush=True)
    if args.dry_run:
        for j in todo:
            print("  ", result_path(*j, results_dir=out_dir).name)
        return

    failed, t_start = [], time.perf_counter()
    for i, (d, m, p, t) in enumerate(todo, 1):
        name = result_path(d, m, p, t, results_dir=out_dir).stem
        cmd = [sys.executable, str(RUN_JOB), "--dataset", d, "--model", m,
               "--partition", p, "--trial", str(t), "--device", args.device]
        if args.smoke:
            cmd.append("--smoke")
        if args.cv_folds:
            cmd += ["--cv-folds", str(args.cv_folds)]
        t0 = time.perf_counter()
        print(f"[{i}/{len(todo)}] {name} ...", end=" ", flush=True)
        with open(log_dir / f"{name}.log", "w") as log:
            rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
        took = time.perf_counter() - t0
        if rc == 0:
            print(f"done in {fmt(took)}", flush=True)
        else:
            failed.append(name)
            print(f"FAILED (exit {rc}) after {fmt(took)} — see "
                  f"{log_dir / (name + '.log')}", flush=True)
        elapsed = time.perf_counter() - t_start
        if i < len(todo):
            print(f"    elapsed {fmt(elapsed)}, rough time left "
                  f"{fmt(elapsed / i * (len(todo) - i))}", flush=True)

    print(f"\nFinished {len(todo) - len(failed)}/{len(todo)} jobs in "
          f"{fmt(time.perf_counter() - t_start)}.")
    if failed:
        print("Failed (re-run the same command to retry):")
        for f in failed:
            print("  ", f)
        sys.exit(1)


if __name__ == "__main__":
    main()

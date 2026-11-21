"""Time and memory of a sensitivity analysis for parallel trends
(`core.sensitivity.trend_sensitivity`) against R's HonestDiD 0.2.8 and the Python
package diff-diff 4.0.0, on simulated event studies with 4 to 64 pre-treatment and 4
post-treatment periods.

    python benchmarks/sensitivity_benchmark.py              # 4, 8, 16, 32, 64 pre-periods
    python benchmarks/sensitivity_benchmark.py 4 8          # pick your own
    python benchmarks/plot_benchmarks.py                    # redraw the docs' table

Four jobs, each R's default interval for its restriction or the one most often asked
for: relative magnitudes (C-LF), smoothness and relative magnitudes (C-LF),
smoothness (FLCI) and smoothness with the FLCI hybrid (C-F). Each runs for one value
of M (1 for the relative-magnitudes restrictions, half the largest default value for
smoothness) and for R's default grid of 10 values, which every library is given
explicitly, so all of them compute the same intervals.

Configurations: HonestDiD sequentially and with `parallel = TRUE` (doParallel, one
forked worker per logical CPU; it runs the values of M in parallel, so only the
10-value grid is measured with it); hdfe_stream on one thread and on all of them;
and diff-diff for the smoothness FLCI. diff-diff's relative-magnitudes interval is
not the method's: it widens the identified set by z standard errors (its conditional
test is disabled), so its time is not comparable for that restriction, and it is not
measured.

Each configuration runs in its own process. The time excludes loading the packages
(and, for hdfe_stream, a warm-up call that loads the compiled kernels from numba's
cache). Peak memory is the process's VmHWM, or, for R's parallel workers, the
largest sampled Pss of the process and its workers together, if larger; the
memory reported is the increase over the process's value after loading the
packages.

R is the Rscript of benchmarks/external/.r (benchmarks/README.md), or
HDFE_BENCH_RSCRIPT; diff-diff is the Python of benchmarks/external/.diff_diff, or
HDFE_BENCH_DIFF_DIFF_PYTHON. Either one missing is skipped.

Results go to benchmarks/results/sensitivity.csv, one row per (pre-periods, job,
grid, configuration), with the intervals; plot_benchmarks.py writes the docs' table
(docs/benchmarks/sensitivity_table.md) from it.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from _harness import (DATA_DIR, RESULTS_DIR, diff_diff_python, external_executable,
                      external_missing, is_done, measure, upsert_csv, write_machine_info)

SCRIPT = Path(__file__).resolve()
HERE = SCRIPT.parent
CSV = RESULTS_DIR / "sensitivity.csv"
KEY = ("n_pre", "job", "grid", "configuration")

DEFAULT_PRE_PERIODS = [4, 8, 16, 32, 64]
N_POST = 4

# job -> (restriction, ci_method, name in the table)
JOBS = {
    "relative_magnitudes": ("relative_magnitudes", None, "relative magnitudes, C-LF"),
    "smoothness_relative_magnitudes": ("smoothness_relative_magnitudes", None,
                                       "smoothness and relative magnitudes, C-LF"),
    "smoothness": ("smoothness", None, "smoothness, FLCI"),
    "smoothness_hybrid": ("smoothness", "hybrid_fixed_length", "smoothness, C-F"),
}
# one value of M, or R's default grid of 10
GRIDS = ("one", "default")

CONFIGS = {
    "HonestDiD (R)": ("honestdid", "sequential"),
    "HonestDiD (R, parallel)": ("honestdid", "parallel"),
    "hdfe_stream (1 thread)": ("hdfe", "1"),
    "hdfe_stream (all threads)": ("hdfe", "all"),
    "diff-diff": ("diff_diff", "default"),
}

# a configuration that runs past this is recorded as a timeout, not waited on
TIMEOUT_S = 2 * 3600


def applies(label, job, grid):
    """Whether a configuration is measured for this job and grid."""
    kind, variant = CONFIGS[label]
    if variant == "parallel":
        return grid == "default"            # HonestDiD runs the values of M in parallel
    if kind == "diff_diff":
        return job == "smoothness"          # its FLCI; its other intervals differ
    return True


def event_study(n_pre, n_post, seed):
    """A simulated event study: correlated coefficients with a small linear
    pre-trend and an effect of about three standard errors."""
    rng = np.random.default_rng(seed)
    n = n_pre + n_post
    F = rng.normal(size=(n, 2))
    C = F @ F.T + np.eye(n)
    d = np.sqrt(np.diag(C))
    se = 0.05 * np.exp(rng.normal(scale=0.3, size=n))
    sigma = (C / np.outer(d, d)) * np.outer(se, se)
    times = np.concatenate([-np.arange(n_pre, 0, -1), np.arange(1, n_post + 1)])
    beta = 0.005 * times + np.concatenate([np.zeros(n_pre),
                                           0.15 + 0.02 * np.arange(1, n_post + 1)])
    beta = beta + np.linalg.cholesky(sigma) @ rng.normal(size=n)
    return {"beta": beta.tolist(), "sigma": sigma.tolist(), "n_pre": n_pre}


def grid_values(es, job, grid):
    """The values of M: R's default grid for the restriction, or one value of it
    (1 for the relative-magnitudes restrictions, half the largest default value
    for smoothness)."""
    from hdfe_stream.core.sensitivity import default_M

    restriction = JOBS[job][0]
    values = default_M(np.array(es["beta"]), np.array(es["sigma"]), es["n_pre"],
                       restriction)
    if grid == "default":
        return [float(v) for v in values]
    return [1.0] if "relative" in restriction else [float(values[-1]) / 2]


# --------------------------------------------------------------------------
# the child process: hdfe_stream
# --------------------------------------------------------------------------

def child(spec):
    from _harness import peak_rss_mb
    from hdfe_stream.core.sensitivity import trend_sensitivity

    es = json.load(open(spec["path"]))
    b, V, n_pre = np.array(es["beta"]), np.array(es["sigma"]), es["n_pre"]
    restriction, ci_method, _name = JOBS[spec["job"]]
    options = dict(restriction=restriction, n_threads=spec["threads"])
    if ci_method:
        options["ci_method"] = ci_method
    # load the compiled kernels from numba's cache, untimed
    trend_sensitivity(b[n_pre - 2:n_pre + 2], V[n_pre - 2:n_pre + 2, n_pre - 2:n_pre + 2],
                      2, M=[spec["M"][0]], **options)
    baseline = peak_rss_mb()
    t0 = time.perf_counter()
    result = trend_sensitivity(b, V, n_pre, M=spec["M"], **options)
    seconds = time.perf_counter() - t0
    table = result.table
    print("RESULT " + json.dumps({
        "seconds": seconds, "peak_rss_mb": peak_rss_mb(), "baseline_rss_mb": baseline,
        "lower": table["lower"].to_list(), "upper": table["upper"].to_list(),
        "workers": spec["threads"]}))


# --------------------------------------------------------------------------
# the parent process
# --------------------------------------------------------------------------

def command(label, path, job, M):
    """The command that runs one configuration, or None if its language is not
    installed."""
    kind, variant = CONFIGS[label]
    if kind == "honestdid":
        rscript = external_executable("honestdid")
        if rscript is None:
            return None
        spec = {"path": str(path), "job": job, "M": M, "parallel": variant == "parallel"}
        return [rscript, str(HERE / "external" / "honestdid_benchmark.R"), json.dumps(spec)]
    if kind == "diff_diff":
        python = diff_diff_python()
        if python is None:
            return None
        return [python, str(HERE / "external" / "diff_diff_sensitivity.py"),
                json.dumps({"path": str(path), "M": M})]
    threads = 1 if variant == "1" else (os.cpu_count() or 1)
    spec = {"path": str(path), "job": job, "M": M, "threads": threads}
    return [sys.executable, str(SCRIPT), "--child", json.dumps(spec)]


def stored_rows():
    if not CSV.exists():
        return []
    with open(CSV, newline="") as handle:
        return list(csv.DictReader(handle))


def difference(row, reference):
    """The largest difference between two sets of intervals, as a share of the
    reference's widest interval."""
    try:
        a = np.array(json.loads(row["lower"]) + json.loads(row["upper"]), dtype=float)
        b = np.array(json.loads(reference["lower"]) + json.loads(reference["upper"]),
                     dtype=float)
    except (KeyError, TypeError, ValueError):
        return None
    if a.shape != b.shape:
        return None
    width = np.nanmax(np.array(json.loads(reference["upper"]), dtype=float)
                      - np.array(json.loads(reference["lower"]), dtype=float))
    return float(np.nanmax(np.abs(a - b)) / width) if width > 0 else None


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pre_periods", nargs="*", type=int, default=DEFAULT_PRE_PERIODS,
                        help="numbers of pre-treatment periods")
    parser.add_argument("--only", nargs="+", metavar="CONFIG",
                        help="run only these configurations (by label)")
    parser.add_argument("--child", metavar="SPEC", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child:
        child(json.loads(args.child))
        return

    labels = [label for label in CONFIGS if not args.only or label in args.only]
    write_machine_info(RESULTS_DIR / "sensitivity_machine.json")
    workdir = DATA_DIR / "work"
    folder = DATA_DIR / "sensitivity"
    folder.mkdir(parents=True, exist_ok=True)
    for n_pre in args.pre_periods:
        es = event_study(n_pre, N_POST, 1000 + n_pre)
        path = folder / f"event_study_{n_pre}x{N_POST}.json"
        path.write_text(json.dumps(es))
        print(f"\n=== {n_pre} pre-treatment periods, {N_POST} post-treatment ===",
              flush=True)
        for j, job in enumerate(JOBS):
            for g, grid in enumerate(GRIDS):
                M = grid_values(es, job, grid)
                rows = {}
                for c, label in enumerate(CONFIGS):
                    if label not in labels or not applies(label, job, grid):
                        continue
                    name = f"{JOBS[job][2]}, {len(M)} M"
                    values = {"n_pre": n_pre, "job": job, "grid": grid,
                              "configuration": label}
                    if is_done(CSV, values, KEY):
                        print(f"  {name:<52}{label:<28}measured before, kept", flush=True)
                        continue
                    cmd = command(label, path, job, M)
                    if cmd is None:
                        kind = "honestdid" if CONFIGS[label][0] == "honestdid" else None
                        print(f"  {name:<52}{label:<28}" + (
                            external_missing(kind) if kind else
                            "skipped: diff-diff is not installed (benchmarks/external/"
                            ".diff_diff, or set HDFE_BENCH_DIFF_DIFF_PYTHON)"), flush=True)
                        continue
                    report = measure(SCRIPT, "", "", path, workdir, timeout=TIMEOUT_S,
                                     command=cmd, tree=CONFIGS[label][1] == "parallel")
                    row = {**values, "n_post": N_POST, "n_m": len(M),
                           "order": j * 100 + g * 10 + c, "status": report["status"]}
                    if report["status"] == "ok":
                        peak = max(report["peak_rss_mb"], report.get("peak_tree_mb", 0.0))
                        row.update(seconds=round(report["seconds"], 5),
                                   peak_memory_mb=round(peak, 1),
                                   import_memory_mb=round(report["baseline_rss_mb"], 1),
                                   extra_memory_mb=round(peak - report["baseline_rss_mb"], 1),
                                   workers=report.get("workers", 1),
                                   M=json.dumps(M), lower=json.dumps(report["lower"]),
                                   upper=json.dumps(report["upper"]))
                        print(f"  {name:<52}{label:<28}{row['seconds']:10.3f} s   "
                              f"+{row['extra_memory_mb']:7,.0f} MB", flush=True)
                    else:
                        print(f"  {name:<52}{label:<28}{report['status']}", flush=True)
                    rows[label] = row
                    upsert_csv(CSV, [row], KEY)
                # agreement with R's sequential intervals, measured now or before
                stored = {r["configuration"]: r for r in stored_rows()
                          if int(r["n_pre"]) == n_pre and r["job"] == job
                          and r["grid"] == grid and r["status"] == "ok"}
                reference = stored.get("HonestDiD (R)")
                if reference is not None:
                    for label, row in stored.items():
                        if label != "HonestDiD (R)":
                            diff = difference(row, reference)
                            if diff is not None:
                                print(f"    {label}: intervals differ from R's by up to "
                                      f"{diff:.1%} of R's widest", flush=True)


if __name__ == "__main__":
    main()

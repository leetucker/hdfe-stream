"""An event study and its sensitivity analysis, end to end: how the time divides
between fitting the event study and the sensitivity analysis (Rambachan and Roth's,
R's HonestDiD), in hdfe_stream, in R (fixest, then HonestDiD) and in diff-diff.

    python benchmarks/event_study_benchmark.py              # 1,000,000 units
    python benchmarks/event_study_benchmark.py 100000       # pick your own
    python benchmarks/plot_benchmarks.py                    # redraw the figure

The panel is simulated: units observed for 13 years, about half of them treated
from year 9 on (one treatment date, no staggering), with 8 pre-treatment and 4
post-treatment event times, a small linear pre-trend and an effect that grows. The
event study is the two-way fixed effects regression

    y ~ i(rel_time, ref=-1) | unit + year

clustered by unit (rel_time is the event time for treated units and -1, the
reference, for the others), and the sensitivity analysis bounds the first
post-treatment effect for relative magnitudes (C-LF, M from 0 to 2) and for
smoothness (FLCI), each at 10 values of M, the same values in every library.

Configurations: hdfe_stream (feols_stream, then the fit's trend_sensitivity); R
(fixest's feols, then HonestDiD on its coefficients and covariance), with HonestDiD
run sequentially and with parallel = TRUE (doParallel, one forked worker per logical
CPU); and diff-diff (TwoWayFixedEffects with event_study=True, then its HonestDiD).
diff-diff's relative-magnitudes interval is not the method's (it widens the
estimated identified set by z standard errors), so diff-diff is timed for the
smoothness FLCI only.

Estimation includes reading the data: the in-memory libraries read the Parquet file
into a data frame first, and hdfe_stream reads it itself, in batches. The
sensitivity analysis does not depend on the number of units, only on the number of
event times (sensitivity_benchmark.py varies that). Peak memory is the process's
VmHWM, or, with R's parallel workers, the largest sampled Pss of the process and
its workers together, if larger.

Results go to benchmarks/results/event_study.csv, one row per (units,
configuration).
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
CSV = RESULTS_DIR / "event_study.csv"
KEY = ("n_units", "configuration")

DEFAULT_SIZES = [1_000_000]
N_PRE, N_POST = 8, 4
FIRST_TREATED = N_PRE + 1              # years 0 .. N_PRE + N_POST
FML = "y ~ i(rel_time, ref=-1) | unit + year"
# the values of M, the same in every library: R's default grid for relative
# magnitudes, and for smoothness 10 values from 0 to 0.015, about R's default upper
# bound on the 1,000,000-unit panel (0.0147: the 95% upper bound on the largest
# second difference of the pre-treatment coefficients)
MBAR = [float(v) for v in np.linspace(0, 2, 10)]
M_SMOOTHNESS = [float(v) for v in np.linspace(0, 0.015, 10)]

CONFIGS = {
    "hdfe_stream": ("hdfe", "default"),
    "fixest + HonestDiD (R)": ("event_study_r", "sequential"),
    "fixest + HonestDiD (R, parallel)": ("event_study_r", "parallel"),
    "diff-diff": ("diff_diff", "default"),
}

# a configuration that runs past this is recorded as a timeout, not waited on
TIMEOUT_S = 3 * 3600


# --------------------------------------------------------------------------
# the data
# --------------------------------------------------------------------------

def data_path(n_units, seed=2026):
    """The simulated panel, written once (in chunks of units) and reused."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = DATA_DIR / f"event_study_{n_units}_{N_PRE}x{N_POST}.parquet"
    if path.exists():
        return path
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    T = N_PRE + 1 + N_POST
    years = np.arange(T, dtype=np.int32)
    e = years - FIRST_TREATED
    # the treated units' departure from the controls: a linear pre-trend of 0.002
    # per year (zero in the reference year) and an effect of 0.05 + 0.01 e
    path_treated = 0.002 * (e + 1) + np.where(e >= 0, 0.05 + 0.01 * e, 0.0)
    year_effect = 0.05 * years
    tmp = path.with_suffix(".part")
    writer = None
    chunk = 200_000
    for c, start in enumerate(range(0, n_units, chunk)):
        rng = np.random.default_rng([seed, c])
        n = min(chunk, n_units - start)
        unit = np.repeat(np.arange(start, start + n, dtype=np.int64), T)
        year = np.tile(years, n)
        treated = np.repeat(rng.random(n) < 0.5, T)
        rel_time = np.where(treated, np.tile(e, n), -1).astype(np.int32)
        y = (np.repeat(rng.normal(size=n), T) + np.tile(year_effect, n)
             + treated * np.tile(path_treated, n) + rng.normal(size=n * T))
        table = pa.table({"unit": unit, "year": year, "treated": treated.astype(np.int8),
                          "rel_time": rel_time, "y": y})
        if writer is None:
            writer = pq.ParquetWriter(tmp, table.schema)
        writer.write_table(table)
    writer.close()
    os.replace(tmp, path)
    return path


# --------------------------------------------------------------------------
# the child process: hdfe_stream
# --------------------------------------------------------------------------

def child(spec):
    from _harness import peak_rss_mb
    from hdfe_stream import feols_stream
    from hdfe_stream.core.sensitivity import trend_sensitivity

    # load the compiled sensitivity kernels from numba's cache, untimed
    b = np.array([0.0, 0.01, 0.1, 0.12])
    trend_sensitivity(b, np.eye(4) * 1e-4, 2, M=[1.0])
    trend_sensitivity(b, np.eye(4) * 1e-4, 2, restriction="smoothness", M=[0.01])
    baseline = peak_rss_mb()
    t0 = time.perf_counter()
    fit = feols_stream(FML, spec["path"], workdir=spec["workdir"], vcov={"CRV1": "unit"},
                       verbose=False, save_resid=False)
    t_fit = time.perf_counter() - t0
    t0 = time.perf_counter()
    rm = fit.trend_sensitivity(M=spec["mbar"])
    t_rm = time.perf_counter() - t0
    t0 = time.perf_counter()
    sd = fit.trend_sensitivity(restriction="smoothness", M=spec["m"])
    t_sd = time.perf_counter() - t0
    beta = [float(fit.coef()[f"rel_time::{k}"]) for k in
            [*range(-N_PRE - 1, -1), *range(0, N_POST)]]
    out = {"t_read": 0.0, "t_fit": t_fit, "t_relative_magnitudes": t_rm,
           "t_smoothness": t_sd, "n_obs": int(fit.n_obs), "beta": beta,
           "peak_rss_mb": peak_rss_mb(), "baseline_rss_mb": baseline,
           "peak_disk_mb": fit.diagnostics["disk_peak_gb"] * 1000,
           "relative_magnitudes": {"lower": rm.table["lower"].to_list(),
                                   "upper": rm.table["upper"].to_list()},
           "smoothness": {"lower": sd.table["lower"].to_list(),
                          "upper": sd.table["upper"].to_list()}}
    fit.cleanup()
    print("RESULT " + json.dumps(out))


# --------------------------------------------------------------------------
# the parent process
# --------------------------------------------------------------------------

def command(label, path, workdir):
    kind, variant = CONFIGS[label]
    spec = {"path": str(path), "n_pre": N_PRE, "n_post": N_POST, "mbar": MBAR,
            "m": M_SMOOTHNESS}
    if kind == "event_study_r":
        rscript = external_executable("event_study_r")
        if rscript is None:
            return None
        spec.update(threads=os.cpu_count(), parallel=variant == "parallel")
        return [rscript, str(HERE / "external" / "event_study.R"), json.dumps(spec)]
    if kind == "diff_diff":
        python = diff_diff_python()
        if python is None:
            return None
        return [python, str(HERE / "external" / "diff_diff_event_study.py"), json.dumps(spec)]
    spec["workdir"] = str(workdir)
    return [sys.executable, str(SCRIPT), "--child", json.dumps(spec)]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sizes", nargs="*", type=int, default=DEFAULT_SIZES,
                        help="numbers of units")
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
    for n_units in args.sizes:
        path = data_path(n_units)
        n_rows = n_units * (N_PRE + 1 + N_POST)
        print(f"\n=== {n_units:,} units, {n_rows:,} rows "
              f"({path.stat().st_size / 1e6:.0f} MB on disk) ===", flush=True)
        if any(CONFIGS[label][0] == "hdfe" for label in labels):
            # numba's kernels for the fit, compiled or loaded once, untimed
            measure(SCRIPT, "", "", path, workdir, command=command(
                "hdfe_stream", data_path(10_000), workdir))
        betas = {}
        for order, label in enumerate(CONFIGS):
            if label not in labels:
                continue
            values = {"n_units": n_units, "configuration": label}
            if is_done(CSV, values, KEY):
                print(f"  {label:<36}measured before, kept", flush=True)
                continue
            cmd = command(label, path, workdir)
            if cmd is None:
                print(f"  {label:<36}" + (
                    external_missing("event_study_r") if CONFIGS[label][0] != "diff_diff"
                    else "skipped: diff-diff is not installed (benchmarks/external/"
                    ".diff_diff, or set HDFE_BENCH_DIFF_DIFF_PYTHON)"), flush=True)
                continue
            report = measure(SCRIPT, "", "", path, workdir, timeout=TIMEOUT_S, command=cmd,
                             tree=CONFIGS[label][1] == "parallel")
            row = {**values, "n_rows": n_rows, "n_pre": N_PRE, "n_post": N_POST,
                   "order": order, "status": report["status"]}
            if report["status"] == "ok":
                peak = max(report["peak_rss_mb"], report.get("peak_tree_mb", 0.0))
                estimation = report["t_read"] + report["t_fit"]
                rm = report.get("t_relative_magnitudes")
                sd = report["t_smoothness"]
                row.update(
                    read_s=round(report["t_read"], 3), fit_s=round(report["t_fit"], 3),
                    estimation_s=round(estimation, 3),
                    relative_magnitudes_s=round(rm, 4) if rm is not None else "",
                    smoothness_s=round(sd, 4),
                    peak_memory_mb=round(peak, 1),
                    import_memory_mb=round(report["baseline_rss_mb"], 1),
                    peak_disk_mb=round(report.get("peak_disk_mb", 0.0), 1),
                    beta=json.dumps(report.get("beta")) if report.get("beta") else "",
                    relative_magnitudes=json.dumps(report.get("relative_magnitudes", "")),
                    smoothness=json.dumps(report["smoothness"]))
                if report.get("beta"):
                    betas[label] = np.array(report["beta"])
                print(f"  {label:<36}estimation {estimation:8.1f} s   relative magnitudes "
                      + (f"{rm:7.2f} s" if rm is not None else "      – ")
                      + f"   smoothness {sd:6.2f} s   peak {peak:8,.0f} MB", flush=True)
            else:
                print(f"  {label:<36}{report['status']}", flush=True)
            upsert_csv(CSV, [row], KEY)
        if CSV.exists():
            with open(CSV, newline="") as handle:
                for r in csv.DictReader(handle):
                    if (int(r["n_units"]) == n_units and r["status"] == "ok" and r.get("beta")
                            and r["configuration"] not in betas):
                        betas[r["configuration"]] = np.array(json.loads(r["beta"]))
        if len(betas) > 1:
            ref = next(iter(betas.values()))
            worst = max(float(np.max(np.abs(b - ref))) for b in betas.values())
            print(f"  event-study coefficients agree to {worst:.1e}", flush=True)


if __name__ == "__main__":
    main()

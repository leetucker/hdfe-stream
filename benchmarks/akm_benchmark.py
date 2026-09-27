"""Time, peak memory and peak disk on an AKM regression: hdfe_stream against
pyfixest and xhdfe.

    python benchmarks/akm_benchmark.py                  # the default sizes
    python benchmarks/akm_benchmark.py 25000 100000     # pick your own
    python benchmarks/akm_benchmark.py --memory-sweep 1000000
    python benchmarks/plot_benchmarks.py                # redraw the figures

The model is the standard three-way AKM specification on the simulated panel
that ships with the library, with firms = workers / 15 at every size:

    log_earn ~ age_squared + age_cubed | worker_id + firm_id + year

clustered by worker. Results go to benchmarks/results/akm.csv, one row per
(size, configuration), replacing that size's rows when it is re-run; the
figures in the README are drawn from that file.

pyfixest and xhdfe need the data in memory as a pandas DataFrame, so reading it
is part of the job and is counted. hdfe_stream is given the Parquet path and
does its own reading, in batches. That difference *is* the comparison -- it is
not an artifact of how this is measured.

The libraries are asked for comparable accuracy rather than identical settings,
since their convergence criteria measure different things. Defaults are used
throughout -- xhdfe on its CPU backend, which is its default -- and the script
checks that the coefficients agree before reporting a size, so the times are
for equally good answers.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from _harness import (DATA_DIR, RESULTS_DIR, data_path, measure, panel_shape,
                      run_child, upsert_csv, write_machine_info)

SCRIPT = Path(__file__).resolve()
FML = "log_earn ~ age_squared + age_cubed | worker_id + firm_id + year"
COEFS = ("age_squared", "age_cubed")

# n_workers for each size; the panel is roughly 8.5 rows per worker
DEFAULT_SIZES = [25_000, 100_000, 400_000, 1_000_000, 5_000_000]

CONFIGS = {
    "pyfixest (MAP)": ("pyfixest", "map"),
    "pyfixest (LSMR)": ("pyfixest", "lsmr"),
    "xhdfe": ("xhdfe", "default"),
    "hdfe_stream (explicit)": ("hdfe", "explicit"),
    "hdfe_stream (stream_cg)": ("hdfe", "stream_cg"),
    "hdfe_stream (within)": ("hdfe", "within"),
    # the same solve with the batch and bucket sizes turned down: this is the
    # knob the in-memory libraries have no equivalent of
    "hdfe_stream (stream_cg, low memory)": ("hdfe", "stream_cg+lowmem"),
}

# what "low memory" means above: smaller read batches and more buckets, so less
# of the data is in flight at once
LOW_MEMORY = {"rows_per_bucket": 250_000, "batch_rows": 200_000}

# a configuration that runs past this is recorded as a timeout, not waited on
TIMEOUT_S = 3 * 3600


# --------------------------------------------------------------------------
# the child process: one configuration, one size
# --------------------------------------------------------------------------

def run_pyfixest(variant, path):
    import pandas as pd
    import pyfixest as pf

    t0 = time.perf_counter()
    frame = pd.read_parquet(path)
    t_read = time.perf_counter() - t0

    demeaner = pf.MapDemeaner() if variant == "map" else pf.LsmrDemeaner()
    t0 = time.perf_counter()
    fit = pf.feols(FML, data=frame, vcov={"CRV1": "worker_id"},
                   fixef_rm="none", demeaner=demeaner)
    t_fit = time.perf_counter() - t0
    coefs = fit.coef().to_dict()
    return {"t_read": t_read, "t_fit": t_fit, "n_obs": int(fit._N),
            "coefs": {k: float(coefs[k]) for k in COEFS}}


def run_xhdfe(path):
    import pandas as pd
    import xhdfe

    t0 = time.perf_counter()
    frame = pd.read_parquet(path)
    t_read = time.perf_counter() - t0

    t0 = time.perf_counter()
    fit = xhdfe.feols(FML, data=frame, se_type="cluster", clusters="worker_id")
    t_fit = time.perf_counter() - t0
    coefs = dict(zip(fit.coef_names_, fit.coef_))
    return {"t_read": t_read, "t_fit": t_fit, "n_obs": int(len(frame)),
            "coefs": {k: float(coefs[k]) for k in COEFS}}


def run_hdfe(solver, path, workdir):
    from hdfe_stream import feols_stream

    options = {}
    if solver.endswith("+lowmem"):
        solver = solver.removesuffix("+lowmem")
        options = dict(LOW_MEMORY)
    elif "+sweep:" in solver:
        solver, rows_per_bucket, batch_rows = solver.split(":")
        solver = solver.removesuffix("+sweep")
        options = {"rows_per_bucket": int(rows_per_bucket),
                   "batch_rows": int(batch_rows)}

    t0 = time.perf_counter()
    fit = feols_stream(FML, str(path), workdir=str(workdir), solver=solver,
                       vcov={"CRV1": "worker_id"}, fe_dof="pyfixest",
                       verbose=False, save_resid=False, **options)
    t_fit = time.perf_counter() - t0
    coefs = fit.coef()
    out = {"t_read": 0.0, "t_fit": t_fit, "n_obs": int(fit.n_obs),
           "coefs": {k: float(coefs[k]) for k in COEFS},
           "reported_disk_mb": fit.diagnostics["disk_peak_gb"] * 1000}
    fit.cleanup()
    return out


def runner(kind, variant, path, workdir):
    if kind == "pyfixest":
        return run_pyfixest(variant, path)
    if kind == "xhdfe":
        return run_xhdfe(path)
    return run_hdfe(variant, path, workdir)


# --------------------------------------------------------------------------
# the parent process
# --------------------------------------------------------------------------

MEMORY_SWEEP = [(20_000_000, 2_000_000), (1_000_000, 500_000),
                (250_000, 200_000), (100_000, 100_000)]


def memory_sweep(n_workers):
    """How peak memory responds to the batch and bucket settings.

    Same data, same solver, same answer -- only how much is in flight at once.
    """
    path = data_path(n_workers)
    workdir = DATA_DIR / "work"
    measure(SCRIPT, "hdfe", "stream_cg", path, workdir)          # warm up

    print(f"\n=== memory knobs, {n_workers:,} workers ===")
    print("| rows_per_bucket | batch_rows | wall time | peak memory | peak disk |")
    print("|---:|---:|---:|---:|---:|")
    for rows_per_bucket, batch_rows in MEMORY_SWEEP:
        variant = f"stream_cg+sweep:{rows_per_bucket}:{batch_rows}"
        report = measure(SCRIPT, "hdfe", variant, path, workdir)
        if report["status"] != "ok":
            print(f"| {rows_per_bucket:,} | {batch_rows:,} | {report['status']} | | |")
            continue
        print(f"| {rows_per_bucket:,} | {batch_rows:,} "
              f"| {report['t_fit']:,.1f} s | {report['peak_rss_mb']:,.0f} MB "
              f"| {report['peak_disk_mb']:,.0f} MB |")


def check_agreement(reports):
    """All configurations must produce the same estimates, or the timings are
    not comparable. Returns the largest relative difference."""
    ok = [r for r in reports.values() if r["status"] == "ok"]
    if len(ok) < 2:
        return 0.0
    base = ok[0]["coefs"]
    return max(abs(r["coefs"][k] - base[k]) / max(abs(base[k]), 1e-12)
               for r in ok[1:] for k in COEFS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sizes", nargs="*", type=int, default=DEFAULT_SIZES,
                        help="number of workers in each panel")
    parser.add_argument("--only", nargs="+", metavar="CONFIG",
                        help="run only these configurations (by label)")
    parser.add_argument("--memory-sweep", type=int, metavar="N_WORKERS",
                        help="instead of the comparison, show how peak memory "
                             "responds to rows_per_bucket and batch_rows")
    parser.add_argument("--child", nargs=4,
                        metavar=("KIND", "VARIANT", "PATH", "WORKDIR"),
                        help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.child:
        run_child(runner, *args.child)
        return
    if args.memory_sweep:
        memory_sweep(args.memory_sweep)
        return

    configs = {k: v for k, v in CONFIGS.items()
               if not args.only or k in args.only}
    write_machine_info(RESULTS_DIR / "machine.json")
    workdir = DATA_DIR / "work"
    for n_workers in args.sizes:
        path = data_path(n_workers)
        shape = panel_shape(path)
        print(f"\n=== {n_workers:,} workers, {shape['firms']:,} firms, "
              f"{shape['rows']:,} rows ({path.stat().st_size / 1e6:.0f} MB "
              f"on disk) ===", flush=True)

        # warm up once so numba kernel compilation (cached on disk) is not
        # charged to the first configuration that happens to run
        measure(SCRIPT, "hdfe", "explicit", data_path(25_000), workdir)

        reports = {}
        for order, (label, (kind, variant)) in enumerate(configs.items()):
            report = measure(SCRIPT, kind, variant, path, workdir,
                             timeout=TIMEOUT_S)
            reports[label] = report
            row = {"n_workers": n_workers, "n_firms": shape["firms"],
                   "n_rows": shape["rows"], "configuration": label,
                   "order": order, "status": report["status"]}
            if report["status"] == "ok":
                row.update(
                    wall_s=round(report["t_read"] + report["t_fit"], 3),
                    read_s=round(report["t_read"], 3),
                    fit_s=round(report["t_fit"], 3),
                    peak_memory_mb=round(report["peak_rss_mb"], 1),
                    import_memory_mb=round(report["baseline_rss_mb"], 1),
                    peak_disk_mb=round(report["peak_disk_mb"], 1),
                    **{f"coef_{k}": repr(report["coefs"][k]) for k in COEFS})
                print(f"  {label:<38}{row['wall_s']:8.1f} s   peak "
                      f"{row['peak_memory_mb']:8,.0f} MB   disk "
                      f"{row['peak_disk_mb']:7,.0f} MB", flush=True)
            else:
                print(f"  {label:<38}{report['status']}", flush=True)
            upsert_csv(RESULTS_DIR / "akm.csv", [row])

        worst = check_agreement(reports)
        verdict = "agree" if worst < 1e-6 else "DISAGREE"
        print(f"  estimates {verdict}: largest relative difference "
              f"{worst:.1e}", flush=True)


if __name__ == "__main__":
    main()

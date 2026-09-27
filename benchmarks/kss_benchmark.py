"""Time, peak memory and peak disk for the KSS leave-out variance decomposition:
hdfe_stream against xhdfe.

    python benchmarks/kss_benchmark.py                  # the default sizes
    python benchmarks/kss_benchmark.py 25000 100000     # pick your own
    python benchmarks/plot_benchmarks.py                # redraw the figures

The model is the two-way AKM specification with controls, on the same simulated
panels as the regression benchmark (firms = workers / 15):

    log_earn ~ age_squared + age_cubed | worker_id + firm_id

and the job is the whole leave-out decomposition: prune to the leave-one-out
connected set, fit, partial out the controls, collapse to worker-firm matches,
estimate leverages by random projection, and correct the variance components.
Both libraries leave out a match (each one's default), follow LeaveOutTwoWay's
conventions, and use 200 random-projection draws (xhdfe's default; hdfe_stream's
is 250). "standard errors" adds each library's component standard errors at its
own defaults for that step: xhdfe simulates the trace term with 1,000 draws of
the full quadratic form, hdfe_stream estimates it with 200 Hutchinson draws
(docs/kss_methodological_differences.md, 5.2).

xhdfe needs the data in memory, so reading it is counted, as in the regression
benchmark; hdfe_stream reads the Parquet file itself. xhdfe runs on its CPU
backend, its default. Results go to benchmarks/results/kss.csv; the script
reports how far the two libraries' estimates are apart, which is random-
projection noise, not disagreement about the estimator.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from _harness import (DATA_DIR, RESULTS_DIR, data_path, measure, panel_shape,
                      run_child, upsert_csv, write_machine_info)

SCRIPT = Path(__file__).resolve()
FML = "log_earn ~ age_squared + age_cubed | worker_id + firm_id"
DRAWS = 200
COMPONENTS = {"var_psi": "var(psi)", "var_alpha": "var(alpha)",
              "cov_alpha_psi": "cov(psi, alpha)"}

DEFAULT_SIZES = [25_000, 100_000, 400_000, 1_000_000, 5_000_000]

CONFIGS = {
    "xhdfe": ("xhdfe", "point"),
    "xhdfe (standard errors)": ("xhdfe", "se"),
    "hdfe_stream": ("hdfe", "point"),
    "hdfe_stream (standard errors)": ("hdfe", "se"),
    "hdfe_stream (low memory)": ("hdfe", "point+lowmem"),
}

LOW_MEMORY = {"rows_per_bucket": 250_000, "batch_rows": 200_000}
TIMEOUT_S = 3 * 3600


# --------------------------------------------------------------------------
# the child process
# --------------------------------------------------------------------------

def run_xhdfe(variant, path):
    import numpy as np
    import pandas as pd
    import xhdfe.akm as akm

    t0 = time.perf_counter()
    frame = pd.read_parquet(path, columns=["worker_id", "firm_id", "log_earn",
                                           "age_squared", "age_cubed"])
    t_read = time.perf_counter() - t0

    t0 = time.perf_counter()
    X = np.column_stack([frame["age_squared"].to_numpy(float),
                         frame["age_cubed"].to_numpy(float)])
    res = akm.akm_kss(frame["log_earn"].to_numpy(float),
                      frame["worker_id"].to_numpy(),
                      frame["firm_id"].to_numpy(), X=X, leverages="jla",
                      jla_draws=DRAWS, compute_se=(variant == "se"))
    t_fit = time.perf_counter() - t0
    out = {"t_read": t_read, "t_fit": t_fit,
           "n_obs": int(res["sample"]["n_obs"]),
           "kss": {k: float(res["kss"][k]) for k in COMPONENTS}}
    if variant == "se":
        out["se"] = {k: float(res["component_se"][f"se_{k}"])
                     for k in COMPONENTS}
    return out


def run_hdfe(variant, path, workdir):
    from hdfe_stream import leave_out_kss

    options = dict(LOW_MEMORY) if variant.endswith("+lowmem") else {}
    se = variant.startswith("se")
    t0 = time.perf_counter()
    lo = leave_out_kss(FML, str(path), workdir=str(workdir), n_draws=DRAWS,
                       se=se, verbose=False, **options)
    t_fit = time.perf_counter() - t0
    out = {"t_read": 0.0, "t_fit": t_fit, "n_obs": int(lo.n_obs),
           "kss": {k: float(lo.leave_out[v]) for k, v in COMPONENTS.items()}}
    if se:
        out["se"] = {k: float(lo.se.se[v]) for k, v in COMPONENTS.items()}
    return out


def runner(kind, variant, path, workdir):
    if kind == "xhdfe":
        return run_xhdfe(variant, path)
    return run_hdfe(variant, path, workdir)


# --------------------------------------------------------------------------
# the parent process
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sizes", nargs="*", type=int, default=DEFAULT_SIZES,
                        help="number of workers in each panel")
    parser.add_argument("--only", nargs="+", metavar="CONFIG",
                        help="run only these configurations (by label)")
    parser.add_argument("--child", nargs=4,
                        metavar=("KIND", "VARIANT", "PATH", "WORKDIR"),
                        help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.child:
        run_child(runner, *args.child)
        return

    configs = {k: v for k, v in CONFIGS.items()
               if not args.only or k in args.only}
    write_machine_info(RESULTS_DIR / "machine.json")
    workdir = DATA_DIR / "work_kss"
    for n_workers in args.sizes:
        path = data_path(n_workers)
        shape = panel_shape(path)
        print(f"\n=== {n_workers:,} workers, {shape['firms']:,} firms, "
              f"{shape['rows']:,} rows ===", flush=True)
        # compile numba kernels before timing anything
        measure(SCRIPT, "hdfe", "point", data_path(25_000), workdir)

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
                    n_obs_pruned=report["n_obs"],
                    **{f"kss_{k}": repr(v) for k, v in report["kss"].items()},
                    **{f"se_{k}": repr(v)
                       for k, v in report.get("se", {}).items()})
                print(f"  {label:<32}{row['wall_s']:9.1f} s   peak "
                      f"{row['peak_memory_mb']:8,.0f} MB   disk "
                      f"{row['peak_disk_mb']:7,.0f} MB   var(psi) "
                      f"{report['kss']['var_psi']:.5f}", flush=True)
            else:
                print(f"  {label:<32}{report['status']}", flush=True)
            upsert_csv(RESULTS_DIR / "kss.csv", [row])

        ok = {k: r for k, r in reports.items() if r["status"] == "ok"}
        if "xhdfe" in ok and "hdfe_stream" in ok:
            gap = {k: abs(ok["hdfe_stream"]["kss"][k] - ok["xhdfe"]["kss"][k])
                   / abs(ok["xhdfe"]["kss"][k]) for k in COMPONENTS}
            print("  hdfe_stream vs xhdfe, relative difference: "
                  + ", ".join(f"{COMPONENTS[k]} {v:.2%}" for k, v in gap.items()),
                  flush=True)


if __name__ == "__main__":
    main()

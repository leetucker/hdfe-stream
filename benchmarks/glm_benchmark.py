"""Time, peak memory and peak disk on Poisson and logit regressions with AKM
fixed effects: fepois_stream and feglm_stream against pyfixest.

    python benchmarks/glm_benchmark.py                  # the default sizes
    python benchmarks/glm_benchmark.py 25000 100000     # pick your own
    python benchmarks/plot_benchmarks.py                # redraw the figures

The panels are those of the AKM benchmark, at the same sizes, with a count and
a binary outcome driven by the same worker effects, firm effects and age profile
as log_earn (`_harness.glm_data_path`). The models are

    y_count  ~ age_squared + age_cubed | worker_id + firm_id + year    (Poisson)
    y_binary ~ age_squared + age_cubed | worker_id + firm_id + year    (logit)

clustered by worker. Results go to benchmarks/results/glm.csv, one row per
(size, configuration), replacing that size's rows when it is re-run.

pyfixest runs with its LSMR demeaner. Its default, MAP, stops with "Demeaning
failed after 10000 iterations" on these panels at every size tried, from
25,000 workers up, for both models. xhdfe fits linear models only.

Both libraries run at their defaults otherwise, and the script checks that the
coefficients agree before reporting a size. For logit they agree to the
tolerance the iterations reach rather than to rounding: pyfixest keeps workers
whose outcome is 1 in every year, whose effects then drift toward infinity,
and hdfe_stream drops them (see docs/glm.md).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from _harness import (DATA_DIR, RESULTS_DIR, glm_data_path, measure,
                      panel_shape, run_child, upsert_csv, write_machine_info)

SCRIPT = Path(__file__).resolve()
RHS = "age_squared + age_cubed | worker_id + firm_id + year"
OUTCOME = {"poisson": "y_count", "logit": "y_binary"}
COEFS = ("age_squared", "age_cubed")

DEFAULT_SIZES = [25_000, 100_000, 400_000, 1_000_000, 5_000_000]

# label -> (family, library, variant)
CONFIGS = {
    "pyfixest fepois (LSMR)": ("poisson", "pyfixest", "lsmr"),
    "fepois_stream": ("poisson", "hdfe", "auto"),
    "fepois_stream (explicit)": ("poisson", "hdfe", "explicit"),
    "fepois_stream (low memory)": ("poisson", "hdfe", "auto+lowmem"),
    "pyfixest feglm logit (LSMR)": ("logit", "pyfixest", "lsmr"),
    "feglm_stream logit": ("logit", "hdfe", "auto"),
    "feglm_stream logit (explicit)": ("logit", "hdfe", "explicit"),
    "feglm_stream logit (low memory)": ("logit", "hdfe", "auto+lowmem"),
}

# the AKM benchmark's low-memory setting
LOW_MEMORY = {"rows_per_bucket": 250_000, "batch_rows": 200_000}

TIMEOUT_S = 3 * 3600


# --------------------------------------------------------------------------
# the child process: one configuration, one size
# --------------------------------------------------------------------------

def run_pyfixest(family, path):
    import pandas as pd
    import pyfixest as pf

    fml = f"{OUTCOME[family]} ~ {RHS}"
    t0 = time.perf_counter()
    frame = pd.read_parquet(path)
    t_read = time.perf_counter() - t0

    t0 = time.perf_counter()
    if family == "poisson":
        fit = pf.fepois(fml, data=frame, vcov={"CRV1": "worker_id"},
                        demeaner=pf.LsmrDemeaner())
    else:
        fit = pf.feglm(fml, data=frame, family="logit", vcov={"CRV1": "worker_id"},
                       demeaner=pf.LsmrDemeaner())
    t_fit = time.perf_counter() - t0
    coefs = fit.coef().to_dict()
    return {"t_read": t_read, "t_fit": t_fit, "n_obs": int(fit._N),
            "coefs": {k: float(coefs[k]) for k in COEFS}}


def run_hdfe(family, solver, path, workdir):
    from hdfe_stream import feglm_stream, fepois_stream

    options = {}
    if solver.endswith("+lowmem"):
        solver = solver.removesuffix("+lowmem")
        options = dict(LOW_MEMORY)
    fml = f"{OUTCOME[family]} ~ {RHS}"
    common = dict(workdir=str(workdir), solver=solver, vcov={"CRV1": "worker_id"},
                  verbose=False, save_resid=False, **options)

    t0 = time.perf_counter()
    if family == "poisson":
        fit = fepois_stream(fml, str(path), **common)
    else:
        fit = feglm_stream(fml, str(path), "logit", **common)
    t_fit = time.perf_counter() - t0
    coefs = fit.coef()
    out = {"t_read": 0.0, "t_fit": t_fit, "n_obs": int(fit.n_obs),
           "irls_steps": int(fit.diagnostics["irls"]["iterations"]),
           "coefs": {k: float(coefs[k]) for k in COEFS},
           "reported_disk_mb": fit.diagnostics["disk_peak_gb"] * 1000}
    fit.cleanup()
    return out


def runner(kind, variant, path, workdir):
    family, library = kind.split(":")
    if library == "pyfixest":
        return run_pyfixest(family, path)
    return run_hdfe(family, variant, path, workdir)


# --------------------------------------------------------------------------
# the parent process
# --------------------------------------------------------------------------

def agreement(reports):
    """The largest relative difference between configurations' coefficients."""
    ok = [r for r in reports if r["status"] == "ok"]
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
    parser.add_argument("--child", nargs=4,
                        metavar=("KIND", "VARIANT", "PATH", "WORKDIR"),
                        help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.child:
        run_child(runner, *args.child)
        return

    configs = {k: v for k, v in CONFIGS.items() if not args.only or k in args.only}
    write_machine_info(RESULTS_DIR / "machine.json")
    workdir = DATA_DIR / "work_glm"
    for n_workers in args.sizes:
        path = glm_data_path(n_workers)
        shape = panel_shape(path)
        print(f"\n=== {n_workers:,} workers, {shape['firms']:,} firms, "
              f"{shape['rows']:,} rows ({path.stat().st_size / 1e6:.0f} MB "
              f"on disk) ===", flush=True)

        # compile numba kernels before timing anything
        for family in OUTCOME:
            measure(SCRIPT, f"{family}:hdfe", "auto", glm_data_path(25_000), workdir)

        reports = {}
        for label, (family, library, variant) in configs.items():
            order = list(CONFIGS).index(label)
            report = measure(SCRIPT, f"{family}:{library}", variant, path, workdir,
                             timeout=TIMEOUT_S)
            reports[label] = report
            row = {"n_workers": n_workers, "n_firms": shape["firms"],
                   "n_rows": shape["rows"], "family": family,
                   "configuration": label, "order": order,
                   "status": report["status"]}
            if report["status"] == "ok":
                row.update(
                    wall_s=round(report["t_read"] + report["t_fit"], 3),
                    read_s=round(report["t_read"], 3),
                    fit_s=round(report["t_fit"], 3),
                    peak_memory_mb=round(report["peak_rss_mb"], 1),
                    import_memory_mb=round(report["baseline_rss_mb"], 1),
                    peak_disk_mb=round(report["peak_disk_mb"], 1),
                    n_obs=report["n_obs"],
                    irls_steps=report.get("irls_steps", ""),
                    **{f"coef_{k}": repr(report["coefs"][k]) for k in COEFS})
                print(f"  {label:<34}{row['wall_s']:8.1f} s   peak "
                      f"{row['peak_memory_mb']:8,.0f} MB   disk "
                      f"{row['peak_disk_mb']:7,.0f} MB", flush=True)
            else:
                print(f"  {label:<34}{report['status']}", flush=True)
            upsert_csv(RESULTS_DIR / "glm.csv", [row])

        for family in OUTCOME:
            worst = agreement([reports[label] for label, c in configs.items()
                               if c[0] == family])
            print(f"  {family} estimates: largest relative difference "
                  f"{worst:.1e}", flush=True)


if __name__ == "__main__":
    main()

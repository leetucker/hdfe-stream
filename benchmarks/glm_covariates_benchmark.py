"""Time, peak memory and peak disk on Poisson and logit regressions as the
number of covariates grows: fepois_stream and feglm_stream against pyfixest,
on one panel.

    python benchmarks/glm_covariates_benchmark.py            # every width
    python benchmarks/glm_covariates_benchmark.py 12 3       # pick bin widths
    python benchmarks/plot_benchmarks.py                     # redraw the figures

The design is that of the covariates benchmark (covariates_benchmark.py, which
explains the age bins and why year effects are left out), with the count and
binary outcomes of the GLM benchmark (`_harness.glm_data_path`):

    y_count  ~ i(age_bin, ref=<youngest>) | worker_id + firm_id    (Poisson)
    y_binary ~ i(age_bin, ref=<youngest>) | worker_id + firm_id    (logit)

clustered by worker, on the 1,000,000-worker panel (8.5 million rows), with age
bins from five years wide (8 indicators) to one month wide (503). hdfe_stream
runs at its defaults and sized to the design, with the covariates benchmark's
rule. pyfixest runs with its LSMR demeaner, as in glm_benchmark.py. Results go
to benchmarks/results/glm_covariates.csv.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from _harness import (DATA_DIR, RESULTS_DIR, glm_data_path, measure, run_child,
                      upsert_csv, write_machine_info)
from covariates_benchmark import DEFAULT_WIDTHS, N_WORKERS, level_of, sized_options

SCRIPT = Path(__file__).resolve()
OUTCOME = {"poisson": "y_count", "logit": "y_binary"}

# label -> (family, library, variant)
CONFIGS = {
    "pyfixest fepois (LSMR)": ("poisson", "pyfixest", "lsmr"),
    "fepois_stream": ("poisson", "hdfe", "default"),
    "fepois_stream (sized to the design)": ("poisson", "hdfe", "sized"),
    "pyfixest feglm logit (LSMR)": ("logit", "pyfixest", "lsmr"),
    "feglm_stream logit": ("logit", "hdfe", "default"),
    "feglm_stream logit (sized to the design)": ("logit", "hdfe", "sized"),
}

TIMEOUT_S = 3 * 3600


def covariate_path():
    """The 1M-worker GLM panel with the covariates benchmark's age bins."""
    path = DATA_DIR / f"glm_covariates_{N_WORKERS}.parquet"
    if not path.exists():
        import polars as pl

        month = (pl.struct("worker_id", "year").hash(seed=20260927) % 12).cast(pl.Int32)
        age_months = (pl.col("age") * 12).cast(pl.Int32) + month
        (pl.scan_parquet(glm_data_path(N_WORKERS))
           .with_columns(**{f"age_bin_{w}": (age_months // w).cast(pl.Int32)
                            for w in DEFAULT_WIDTHS})
           .sink_parquet(path))
        print(f"  wrote {path.name}", flush=True)
    return path


def formula(family, width, path):
    """(formula, number of indicators)."""
    import polars as pl

    column = f"age_bin_{width}"
    levels = (pl.scan_parquet(path).select(pl.col(column).unique().sort())
              .collect()[column].to_list())
    return (f"{OUTCOME[family]} ~ i({column}, ref={levels[0]}) | worker_id + firm_id",
            len(levels) - 1)


# --------------------------------------------------------------------------
# the child process
# --------------------------------------------------------------------------

def run_pyfixest(family, width, path):
    import pandas as pd
    import pyfixest as pf

    fml, _ = formula(family, width, path)
    t0 = time.perf_counter()
    frame = pd.read_parquet(path, columns=["worker_id", "firm_id", OUTCOME[family],
                                           f"age_bin_{width}"])
    t_read = time.perf_counter() - t0
    t0 = time.perf_counter()
    if family == "poisson":
        fit = pf.fepois(fml, data=frame, vcov={"CRV1": "worker_id"},
                        demeaner=pf.LsmrDemeaner())
    else:
        fit = pf.feglm(fml, data=frame, family="logit", vcov={"CRV1": "worker_id"},
                       demeaner=pf.LsmrDemeaner())
    t_fit = time.perf_counter() - t0
    return {"t_read": t_read, "t_fit": t_fit,
            "coef": {str(level_of(k)): float(v) for k, v in fit.coef().items()},
            "n_coef": int(len(fit.coef()))}


def run_hdfe(family, variant, width, path, workdir):
    from hdfe_stream import feglm_stream, fepois_stream

    fml, k = formula(family, width, path)
    options = sized_options(k) if variant == "sized" else {}
    common = dict(workdir=str(workdir), vcov={"CRV1": "worker_id"},
                  verbose=False, save_resid=False, **options)
    t0 = time.perf_counter()
    if family == "poisson":
        fit = fepois_stream(fml, str(path), **common)
    else:
        fit = feglm_stream(fml, str(path), "logit", **common)
    t_fit = time.perf_counter() - t0
    out = {"t_read": 0.0, "t_fit": t_fit,
           "coef": {str(level_of(k)): float(v) for k, v in fit.coef().items()},
           "n_coef": len(fit.coefnames),
           "irls_steps": int(fit.diagnostics["irls"]["iterations"]),
           "reported_disk_mb": fit.diagnostics["disk_peak_gb"] * 1e3}
    fit.cleanup()
    return out


def runner(kind, variant, path, workdir):
    family, library, width = kind.split(":")
    width = int(width)
    if library == "pyfixest":
        return run_pyfixest(family, width, path)
    return run_hdfe(family, variant, width, path, workdir)


# --------------------------------------------------------------------------
# the parent process
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("widths", nargs="*", type=int, default=DEFAULT_WIDTHS,
                        help="age bin widths in months")
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
    path = covariate_path()
    workdir = DATA_DIR / "work_glm_covariates"
    # compile numba kernels before timing anything
    for family in OUTCOME:
        measure(SCRIPT, f"{family}:hdfe:60", "default", path, workdir)
    for width in args.widths:
        _, k = formula("poisson", width, path)
        print(f"\n=== age bins {width} months wide: {k} indicators ===", flush=True)
        reports = {}
        for label, (family, library, variant) in configs.items():
            order = list(CONFIGS).index(label)
            report = measure(SCRIPT, f"{family}:{library}:{width}", variant, path,
                             workdir, timeout=TIMEOUT_S)
            reports[label] = report
            row = {"n_covariates": k, "bin_months": width, "n_workers": N_WORKERS,
                   "family": family, "configuration": label, "order": order,
                   "status": report["status"]}
            if report["status"] == "ok":
                row.update(wall_s=round(report["t_read"] + report["t_fit"], 3),
                           read_s=round(report["t_read"], 3),
                           fit_s=round(report["t_fit"], 3),
                           peak_memory_mb=round(report["peak_rss_mb"], 1),
                           import_memory_mb=round(report["baseline_rss_mb"], 1),
                           peak_disk_mb=round(report["peak_disk_mb"], 1),
                           n_coef=report["n_coef"],
                           irls_steps=report.get("irls_steps", ""))
                print(f"  {label:<42}{row['wall_s']:9.1f} s   peak "
                      f"{row['peak_memory_mb']:8,.0f} MB   disk "
                      f"{row['peak_disk_mb']:7,.0f} MB", flush=True)
            else:
                print(f"  {label:<42}{report['status']}", flush=True)
            upsert_csv(RESULTS_DIR / "glm_covariates.csv", [row],
                       key=("n_covariates", "configuration"))

        for family in OUTCOME:
            ok = [reports[label] for label, c in configs.items()
                  if c[0] == family and reports[label]["status"] == "ok"]
            if len(ok) > 1:
                base = ok[0]["coef"]
                gap = max(abs(r["coef"].get(level, float("nan")) - value)
                          / max(abs(value), 1e-3)
                          for r in ok[1:] for level, value in base.items())
                print(f"  {family} estimates: largest relative difference "
                      f"{gap:.1e}", flush=True)


if __name__ == "__main__":
    main()

"""Time, peak memory and peak disk as the number of covariates grows: hdfe_stream
against pyfixest and xhdfe, on one panel.

    python benchmarks/covariates_benchmark.py                 # every width
    python benchmarks/covariates_benchmark.py 12 3            # pick bin widths
    python benchmarks/plot_benchmarks.py                      # redraw the figures

The model is the two-way AKM specification with age entered as a set of
indicators rather than a polynomial:

    log_earn ~ i(age_bin, ref=<youngest>) | worker_id + firm_id

and the x axis of every figure is how fine the age bins are, from five years
wide (8 indicators) to one month wide (503). The panel is the 1,000,000-worker
panel of the regression benchmark (8.5 million rows, 66,665 firms), so this
benchmark varies the width of the design at a fixed height.

The simulated panel records age in whole years, so the benchmark gives each
observation an interview month -- a hash of the worker id and year, uniform
over the twelve months and unrelated to anything else in the data -- and bins
age at interview in months. The outcome does not depend on it; the fine bins
are there for their number, not their coefficients. The month has to vary
from year to year within a worker: were it fixed (a birth month, say), a
worker's observations would sit exactly twelve months apart, the fine bins
would split into classes whose indicators sum to a function of the worker, and
the design would be singular given the worker effects.

Year effects are left out on purpose. Age is year less birth year, so with
worker and year effects in the model one combination of the age indicators is
collinear with the fixed effects, and the benchmark would be timing each
library's collinearity handling as much as its estimation.

Each library gets the categorical term in its own formula syntax (xhdfe's is
formulaic's C()) and builds its own design, as a user would. pyfixest and
xhdfe need the data in memory; hdfe_stream reads the Parquet file itself, in
batches. hdfe_stream runs twice: at its defaults, and with bucket, batch and
row-group sizes scaled down with the width of the design (`sized_options`
below), which is the setting to use for a wide model. Results go to
benchmarks/results/covariates.csv.
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

from _harness import (DATA_DIR, RESULTS_DIR, data_path, measure, run_child,
                      upsert_csv, write_machine_info)

SCRIPT = Path(__file__).resolve()
N_WORKERS = 1_000_000

# bin width in months -> the number of indicators is (age range / width) - 1
DEFAULT_WIDTHS = [60, 12, 6, 3, 1]

CONFIGS = {
    "pyfixest (MAP)": ("pyfixest", "map"),
    "pyfixest (LSMR)": ("pyfixest", "lsmr"),
    "xhdfe": ("xhdfe", "default"),
    "hdfe_stream": ("hdfe", "default"),
    "hdfe_stream (sized to the design)": ("hdfe", "sized"),
}

# "sized to the design": rows held at once scale down with the width of the
# design, so that a bucket (sorted in memory) and a batch or row group (read in
# memory) stay near these many bytes of design however wide it is. Never above
# the defaults, so a narrow design runs exactly as the default configuration.
SIZED_BUCKET_BYTES = 1e9
SIZED_BATCH_BYTES = 250e6


def sized_options(k):
    per_row = 8 * (k + 4)           # the design, the outcome and the id codes
    batch = int(min(2_000_000, SIZED_BATCH_BYTES // per_row))
    return {"rows_per_bucket": int(min(20_000_000, SIZED_BUCKET_BYTES // per_row)),
            "batch_rows": batch, "row_group_size": min(500_000, batch)}

TIMEOUT_S = 3 * 3600


def covariate_path():
    """The 1M-worker panel with birth months and age bins, made once."""
    path = DATA_DIR / f"covariates_{N_WORKERS}.parquet"
    if not path.exists():
        import polars as pl

        month = (pl.struct("worker_id", "year").hash(seed=20260927) % 12).cast(pl.Int32)
        age_months = (pl.col("age") * 12).cast(pl.Int32) + month
        (pl.scan_parquet(data_path(N_WORKERS))
           .with_columns(**{f"age_bin_{w}": (age_months // w).cast(pl.Int32)
                            for w in DEFAULT_WIDTHS})
           .sink_parquet(path))
        print(f"  wrote {path.name}", flush=True)
    return path


def formulas(width, path):
    """(pyfixest/hdfe_stream formula, xhdfe formula, number of indicators)."""
    import polars as pl

    column = f"age_bin_{width}"
    levels = (pl.scan_parquet(path).select(pl.col(column).unique().sort())
              .collect()[column].to_list())
    ref = levels[0]
    fml = f"log_earn ~ i({column}, ref={ref}) | worker_id + firm_id"
    # formulaic's treatment coding drops the first level, the same reference
    fml_xhdfe = f"log_earn ~ C({column}) | worker_id + firm_id"
    return fml, fml_xhdfe, len(levels) - 1


def level_of(name):
    """The age bin a coefficient belongs to, from any library's naming:
    'age_bin_3::101' (pyfixest), 'C(age_bin_3)[T.101]' (formulaic)."""
    match = re.search(r"(?:::|\[T\.)(-?\d+)\]?$", name)
    return int(match.group(1)) if match else None


# --------------------------------------------------------------------------
# the child process
# --------------------------------------------------------------------------

def columns_for(width):
    return ["worker_id", "firm_id", "log_earn", f"age_bin_{width}"]


def run_pyfixest(variant, width, path):
    import pandas as pd
    import pyfixest as pf

    fml, _, _ = formulas(width, path)
    t0 = time.perf_counter()
    frame = pd.read_parquet(path, columns=columns_for(width))
    t_read = time.perf_counter() - t0
    demeaner = pf.MapDemeaner() if variant == "map" else pf.LsmrDemeaner()
    t0 = time.perf_counter()
    fit = pf.feols(fml, data=frame, vcov={"CRV1": "worker_id"}, demeaner=demeaner)
    t_fit = time.perf_counter() - t0
    return {"t_read": t_read, "t_fit": t_fit,
            "coef": {str(level_of(k)): float(v) for k, v in fit.coef().items()},
            "n_coef": int(len(fit.coef()))}


def run_xhdfe(variant, width, path):
    import pandas as pd
    import xhdfe

    _, fml, _ = formulas(width, path)
    t0 = time.perf_counter()
    frame = pd.read_parquet(path, columns=columns_for(width))
    t_read = time.perf_counter() - t0
    t0 = time.perf_counter()
    fit = xhdfe.feols(fml, data=frame, se_type="cluster", clusters="worker_id")
    t_fit = time.perf_counter() - t0
    coef = dict(zip(fit.coef_names_, fit.coef_))
    return {"t_read": t_read, "t_fit": t_fit,
            "coef": {str(level_of(k)): float(v) for k, v in coef.items()},
            "n_coef": int(len(coef))}


def run_hdfe(variant, width, path, workdir):
    from hdfe_stream import feols_stream

    fml, _, k = formulas(width, path)
    options = sized_options(k) if variant == "sized" else {}
    t0 = time.perf_counter()
    fit = feols_stream(fml, str(path), workdir=str(workdir),
                       vcov={"CRV1": "worker_id"}, verbose=False,
                       save_resid=False, **options)
    t_fit = time.perf_counter() - t0
    out = {"t_read": 0.0, "t_fit": t_fit,
           "coef": {str(level_of(k)): float(v) for k, v in fit.coef().items()},
           "n_coef": len(fit.coefnames),
           "reported_disk_mb": fit.diagnostics["disk_peak_gb"] * 1e3}
    fit.cleanup()
    return out


def runner(kind, variant, path, workdir):
    library, width = kind.split(":")
    width = int(width)
    if library == "pyfixest":
        return run_pyfixest(variant, width, path)
    if library == "xhdfe":
        return run_xhdfe(variant, width, path)
    return run_hdfe(variant, width, path, workdir)


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
    workdir = DATA_DIR / "work_covariates"
    # compile numba kernels before timing anything
    measure(SCRIPT, "hdfe:60", "default", path, workdir)
    for width in args.widths:
        _, _, k = formulas(width, path)
        print(f"\n=== age bins {width} months wide: {k} indicators ===", flush=True)
        reports = {}
        for order, (label, (library, variant)) in enumerate(configs.items()):
            report = measure(SCRIPT, f"{library}:{width}", variant, path, workdir,
                             timeout=TIMEOUT_S)
            reports[label] = report
            row = {"n_covariates": k, "bin_months": width, "n_workers": N_WORKERS,
                   "configuration": label, "order": order,
                   "status": report["status"]}
            if report["status"] == "ok":
                row.update(wall_s=round(report["t_read"] + report["t_fit"], 3),
                           read_s=round(report["t_read"], 3),
                           fit_s=round(report["t_fit"], 3),
                           peak_memory_mb=round(report["peak_rss_mb"], 1),
                           import_memory_mb=round(report["baseline_rss_mb"], 1),
                           peak_disk_mb=round(report["peak_disk_mb"], 1),
                           n_coef=report["n_coef"])
                print(f"  {label:<36}{row['wall_s']:9.1f} s   peak "
                      f"{row['peak_memory_mb']:8,.0f} MB   disk "
                      f"{row['peak_disk_mb']:7,.0f} MB   {report['n_coef']} coefficients",
                      flush=True)
            else:
                print(f"  {label:<36}{report['status']}", flush=True)
            upsert_csv(RESULTS_DIR / "covariates.csv", [row],
                       key=("n_covariates", "configuration"))

        ok = [r for r in reports.values() if r["status"] == "ok"]
        if len(ok) > 1:
            base = ok[0]["coef"]
            gap = max(abs(r["coef"].get(level, float("nan")) - value)
                      / max(abs(value), 1e-3)
                      for r in ok[1:] for level, value in base.items())
            print(f"  estimates agree: largest relative difference {gap:.1e}", flush=True)


if __name__ == "__main__":
    main()

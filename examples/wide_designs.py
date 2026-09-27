"""Many covariates: a categorical expanded into hundreds of indicators.

The in-memory libraries build the whole design -- rows x covariates -- before
they start. hdfe_stream never does: it holds one bucket or one batch of rows at
a time, so its memory grows with the number of covariates times the batch, not
times the data. On the 8.5-million-row benchmark panel, 167 age indicators ran
pyfixest and xhdfe out of 26 GB, while hdfe_stream finished in 7 GB (see
benchmarks/covariates_benchmark.py and the README).

Three things matter for a wide design:

1. Let the design be built by hdfe_stream. A formula term like i(age_bin) is
   carried through the partition as the one column it comes from, and the
   indicators are built bucket by bucket. (With the low-level interface the
   same happens for any row-wise Polars expression; one that needs the whole
   column, like x - x.mean(), makes the fit build the design up front instead,
   with a warning. `diagnostics["design_evaluated"]` says which happened.)
2. Scale the batch sizes down with the width. Every step that holds rows holds
   all the covariates, so rows_per_bucket, batch_rows and row_group_size are
   the knobs; `sized_options` below is the rule the benchmark uses.
3. Expect time to grow with the square of the number of covariates: the
   cross-products are k x k per row.

This example is small so it runs in seconds; the point is the pattern.

    python examples/wide_designs.py
"""

import polars as pl

from hdfe_stream import feols_stream
from simulated_data import OUTPUT, panel_path, workdir

# Age in months at an interview month that varies from year to year (were it
# fixed within a worker, fine age bins would be collinear with the worker
# effects), in bins of one to twelve months.
wide = OUTPUT / "wide.parquet"
if not wide.exists():
    month = (pl.struct("worker_id", "year").hash(seed=1) % 12).cast(pl.Int32)
    age_months = (pl.col("age") * 12).cast(pl.Int32) + month
    (pl.scan_parquet(panel_path())
       .with_columns(**{f"age_bin_{w}": age_months // w for w in (12, 3, 1)})
       .sink_parquet(wide))


def sized_options(k, bucket_bytes=1e9, batch_bytes=250e6):
    """Bucket, batch and row-group sizes for a design of k covariates, so that
    each holds roughly the given bytes of design whatever k is. Never above
    the defaults, so a narrow design is unaffected."""
    per_row = 8 * (k + 4)
    batch = int(min(2_000_000, batch_bytes // per_row))
    return {"rows_per_bucket": int(min(20_000_000, bucket_bytes // per_row)),
            "batch_rows": batch, "row_group_size": min(500_000, batch)}


print(f"{'bins':>9} {'covariates':>10} {'seconds':>8} {'disk GB':>8}  design evaluated")
for width in (12, 3, 1):
    column = f"age_bin_{width}"
    levels = pl.scan_parquet(wide).select(pl.col(column).unique().sort()).collect()[column]
    k = len(levels) - 1
    fit = feols_stream(f"log_earn ~ i({column}, ref={levels[0]}) | worker_id + firm_id",
                       str(wide), workdir=workdir("wide"), vcov={"CRV1": "worker_id"},
                       verbose=False, save_resid=False, **sized_options(k))
    d = fit.diagnostics
    print(f"{width:>6} mo {len(fit.coefnames):>10} {d['seconds_total']:>8.1f} "
          f"{d['disk_peak_gb']:>8.3f}  {d['design_evaluated']}")
    fit.cleanup()

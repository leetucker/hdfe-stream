# Out of core: memory, disk, solvers and logging

Running on data that does not fit in memory is the reason the library exists,
so it is worth being concrete about what is held in memory and what is not.
Read this before pointing the library at something large.

Never in memory
: all the rows at once. They are read in batches, hash-partitioned into buckets
  by the streamed dimension, sorted one bucket at a time, and reduced to a table
  of cells. Row-level output (residuals, per-row fixed effects) is written
  straight to Parquet.

In memory
: one vector per level of each *non-streamed* dimension, times a block of at
  most `rhs_block` variables: so firms × years, not workers × rows. Plus, for
  the explicit solver, the reduced matrix S, whose size depends on how levels
  co-occur rather than on the number of rows. And whatever rows are in flight:
  one bucket while it is sorted, one batch while it is read, each with all of
  its variables, which is why a [wide design](wide_designs.md) wants smaller
  buckets and batches.

The streamed dimension, normally the biggest one (workers), costs no memory at
all. That asymmetry is the whole design; [how it works](../index.md#how-it-works-in-one-paragraph)
describes the passes.

## What a fit costs

Every fit reports what it used:

```python
fit = feols_stream(fml, data, workdir="scratch")
fit.diagnostics["disk_peak_gb"]       # 0.01
fit.diagnostics["disk_results_gb"]    # 0.006
fit.diagnostics["seconds_total"]      # 0.67
fit.diagnostics["cells_per_obs"]      # 1.0
fit.solver_info                       # solver, iterations, convergence, ...
```

`disk_peak_gb` is the number to watch when sizing a scratch volume:
intermediates are deleted as soon as they are no longer needed, so the peak is
well below the sum of everything written. Expect about five times the size of
the Parquet input (the sorted working copy of the rows is less compressible than
typical input).

`cells_per_obs` is worth understanding: rows are reduced to cells, one per
combination of fixed-effect levels within a worker. Pass 0 sorts each worker's
rows by the other dimensions' levels, so a cell is a run of rows, and pass 1
collapses the runs as it reads them. Well below 1.0 the later passes get much
cheaper. At 1.0 every row is its own cell and the reduction buys nothing, as
with worker, firm and year effects on a worker-year panel, or worker and
firm-by-year effects. The fit notices this while it first scans the data, from
an approximate count of the distinct combinations, and then leaves the rows
unsorted by cell, saving the wider sort. `collapse_cells=True` or `False`
overrides that, and `diagnostics["cells_collapsed"]` says which happened; the
estimates are the same either way.

## Choosing a solver

| `solver` | what it does | when |
|---|---|---|
| `"auto"` (default) | `explicit`, falling back to `stream_cg` if the reduced matrix would exceed `max_s_gb` | leave it alone |
| `"explicit"` | builds the reduced matrix S once and solves it in memory | fastest when it fits; its size depends on how levels co-occur, not on rows |
| `"stream_cg"` | applies the reduced matrix by streaming the cell table each iteration | when the reduced matrix is the thing that does not fit |
| `"within"` | hands the reduced system to `within-py` (the `within` extra) | an alternative; no varying-slopes support |

`precond="amg"` (the `amg` extra) replaces the default Jacobi preconditioner
with algebraic multigrid, for hard geometries; it needs the explicit S. With a
budget too small for S, `auto` falls back, and `solver_info["fallback"]` says
why:

```text
  explicit                        -> explicit     0.03s  16 iterations
  stream_cg                       -> stream_cg    0.03s  16 iterations
  auto (tiny budget, falls back)  -> stream_cg    0.05s  16 iterations
      fell back because: S would take 0.000701 GB > max_s_gb=1e-07 (57,969 nonzeros)
```

## Memory budgets

`batch_rows`
: rows read per batch, and per slice of whole groups when building the cell
  table (default 2,000,000).

`rows_per_bucket`
: target rows per bucket of the streamed dimension (default 20,000,000). More
  buckets means less peak memory in the sort.

`row_group_size`
: rows per row group of the working files (default 500,000).

`rhs_block`
: variables solved for at a time (default 8). This bounds the biggest
  in-memory array, about 6 × (non-streamed levels) × `rhs_block` doubles.

`max_s_gb`
: budget for the explicit S (default: a quarter of physical memory), above
  which `solver="auto"` falls back to `stream_cg`.

None of them changes the answer: with `batch_rows=20_000`,
`rows_per_bucket=50_000` and `rhs_block=2`, the coefficients agree with the
defaults' to 5e-15. The defaults are tuned for a machine with room to spare. If
memory is the binding constraint, set `rows_per_bucket` to something near what
you can afford and leave the rest alone; with many covariates, scale
`batch_rows` and `row_group_size` down too (see [wide designs](wide_designs.md)).
The saving comes from the first steps down. On a million workers, going from the
default to `rows_per_bucket=250_000` and `batch_rows=200_000` cut the peak from
2.3 GB to 1.3 GB for 8% more time, and smaller still raised it again
([trading memory for time](../benchmarks/akm.md#trading-memory-for-time)).

Leave-out estimation has a memory knob of its own, `scratch_mb`; see
[KSS benchmarks](../benchmarks/kss.md#scratch_mb).

## The disk

Each fit works in its own run directory under `workdir`, so concurrent fits
never collide. Point `workdir=` at a disk with room, or set the
`HDFE_STREAM_WORKDIR` environment variable once (for example to a cluster's
scratch filesystem) and leave `workdir` out; a `workdir` passed to a call takes
precedence, and with neither, the system temporary directory is used.

Result files live as long as the result object; everything else is deleted as
soon as the fit is done with it, and everything is removed if the fit fails.
After the example's fit, what is left is the fixed effects and the residuals:

```text
files kept after the fit, under hdfe_run_20261015_090859_b46f1ea8:
  .hdfe_stream_run  (0.00 MB)
  models/m000/fe_firm_id.parquet  (0.01 MB)
  models/m000/fe_worker_id.parquet  (0.33 MB)
  models/m000/fe_year.parquet  (0.00 MB)
  models/m000/resid.parquet  (5.49 MB)
```

- `outputs="keep"` keeps the result files after the result object is gone,
  for reading in another process, until `result.cleanup()` or
  `hdfe_stream.cleanup()`.
- `save_resid=False` skips the row-level file, which is the big one.
- A result is a context manager, which releases its files at the end of the
  block:

  ```python
  with feols_stream(fml, data, workdir="scratch") as fit:
      firm_effects = fit.fixef("firm_id").collect()      # collect what you need
  ```

- `hdfe_stream.cleanup()` sweeps up after killed jobs. Only directories this
  library created are touched, and by default only ones whose process is gone;
  `dry_run=True` lists what it would remove.

## Logging

A long out-of-core fit is not interactive. Pass a logger, and progress, warnings
and summaries all go to it as they happen, with nothing on stdout:

```python
import logging

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("akm")
fit = feols_stream(fml, data, workdir="scratch", logger=log)
fit.summary(logger=log)           # the whole summary as one record
```

```text
2026-10-16 10:13:03,023 akm INFO planning 'log_earn ~ age_squared + age_cubed | worker_id + firm_id + year' (formula expansion, level discovery)
2026-10-16 10:13:03,071 akm INFO pass 0: streaming worker_id (highest cardinality (worker_id ~20,474, firm_id ~1,194, year ~10))
2026-10-16 10:13:03,089 akm INFO pass 0: hash-partitioning rows into 1 worker_id buckets
2026-10-16 10:13:03,135 akm INFO pass 1: rows as cells (~100.0% of the rows are distinct combinations of fixed-effect levels), extracting identifying cells
2026-10-16 10:13:03,164 akm INFO step 2: solving for 3 variables (blocks of 8), solver=auto
2026-10-16 10:13:03,503 akm INFO pass 2: log_earn ~ age_squared + age_cubed | worker_id + firm_id + year
2026-10-16 10:13:03,583 akm INFO done in 0.6s; peak disk use 0.01 GB, result files 0.01 GB in ...
```

Per-iteration conjugate-gradient residuals go to the logger at DEBUG.

## Reproducibility

Everything in a fit is deterministic: the same data and the same options give
the same numbers, with no random component anywhere, to floating-point rounding.
Repeated fits normally agree bit for bit, but aggregation runs in parallel, and
once, under heavy CPU load from another process, two identical fits differed in
the last digit. Treat agreement to about 1e-12 as the guarantee, not bitwise
identity.

The exception is leave-out estimation, which uses random projection and random
draws. Its results are reproducible from the seed but not deterministic;
[draws and reproducibility](../kss/draws.md) says exactly what they depend on
and what to report in a paper.

:::{dropdown} The full script: `examples/out_of_core.py`
```{literalinclude} ../../examples/out_of_core.py
:language: python
```
:::

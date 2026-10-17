# Wide designs

The in-memory libraries build the whole design, rows × covariates, before they
start. hdfe_stream never does: it holds one bucket or one batch of rows at a
time, so its memory grows with the number of covariates times the batch, not
times the data. On the 8.5-million-row benchmark panel, 167 age indicators ran
pyfixest and xhdfe out of memory on a 30.8 GB machine, while hdfe_stream finished in 5.3 GB (see
[many covariates](../benchmarks/covariates.md)).

Three things matter for a wide design.

## 1. Let hdfe_stream build the design

A formula term such as `i(age_bin)` is carried through the partitioning as the
one column it is computed from, and the indicators are built one bucket at a
time. That works for any expression whose value on a row depends on that row
alone, which formula terms always do. With the low-level
[`StreamingHDFE`](../reference/lowlevel.md), an expression that needs the whole
column (`pl.col("x") - pl.col("x").mean()`, a window, a shift) is detected and
the design is built before partitioning instead, with a warning: the estimates
are the same, the memory is not. Define such a column in the input LazyFrame to
avoid it. `result.diagnostics["design_evaluated"]` says which happened:
`"per bucket"` is the one you want.

## 2. Scale the batches down with the width

At its defaults hdfe_stream holds a whole bucket of up to 20 million rows, with
every covariate, while it sorts. That is fine for a handful of covariates and
not for hundreds. Every step that holds rows holds all the covariates, so scale
`rows_per_bucket`, `batch_rows` and `row_group_size` down with the width. The
benchmarks aim at about 1 GB of design per bucket and 250 MB per batch and row
group, never above the defaults, so a narrow design is unaffected:

```python
def sized_options(k, bucket_bytes=1e9, batch_bytes=250e6):
    """Bucket, batch and row-group sizes for a design of k covariates."""
    per_row = 8 * (k + 4)                   # k covariates, plus outcome and ids
    batch = int(min(2_000_000, batch_bytes // per_row))
    return {"rows_per_bucket": int(min(20_000_000, bucket_bytes // per_row)),
            "batch_rows": batch, "row_group_size": min(500_000, batch)}

fit = feols_stream(f"log_earn ~ i(age_bin, ref={first}) | worker_id + firm_id",
                   data, workdir="scratch", vcov={"CRV1": "worker_id"},
                   save_resid=False, **sized_options(k))
```

## 3. Expect time to grow with the square of the width

The cross-products cost k × k per row. They are formed chunk by chunk as BLAS
matrix products, so they are fast, but 503 indicators take several times as long
as 167. Disk grows with the width too, since the working copy of the rows
carries every covariate.

The example builds age in months at an interview month that varies from year to
year (were it fixed within a worker, fine age bins would be collinear with the
worker effects), in bins of twelve, three and one months. It is small so that it
runs in seconds; the point is the pattern:

```text
     bins covariates  seconds  disk GB  design evaluated
    12 mo         41      0.9    0.013  per bucket
     3 mo        167      2.9    0.039  per bucket
     1 mo        503      9.6    0.111  per bucket
```

:::{dropdown} The full script: `examples/wide_designs.py`
```{literalinclude} ../../examples/wide_designs.py
:language: python
```
:::

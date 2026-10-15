# Quickstart

The smallest useful thing: fit a three-way fixed-effects model on a Parquet
file and read the results.

## Fitting

```python
from hdfe_stream import feols_stream

fit = feols_stream(
    "log_earn ~ age_squared + age_cubed | worker_id + firm_id + year",
    "data/panel.parquet",              # a glob like "data/part-*.parquet" works too
    workdir="scratch",
    vcov={"CRV1": "worker_id"},        # cluster by worker
)
fit.summary()
```

The formula follows pyfixest/fixest syntax, `outcome ~ covariates | fixed
effects`: here, log earnings on an age profile, with worker, firm and year
effects. The data is a Parquet path, never loaded into memory as a whole.
`workdir` is where intermediates go. It needs disk, not memory, and it is the one
argument with no pyfixest equivalent; leave it out to use the
`HDFE_STREAM_WORKDIR` environment variable or, failing that, the system
temporary directory.

The fit prints its progress as it goes (`verbose=False` turns that off, and
`logger=` sends it to a `logging` logger instead). The summary reports the
sample, the fixed effects, the fit statistics, the solver and the coefficients:

```text
### log_earn ~ age_squared + age_cubed | worker_id + firm_id + year
Streaming HDFE   vcov: CRV1:worker_id
obs: 169,833   worker_id: 20,000   firm_id: 1,333   year: 10
streamed dimension: worker_id (highest cardinality (worker_id ~20,474, firm_id ~1,194, year ~10))
identifying worker_id groups: 20,000   (worker_id x firm_id) components: 1
FE dof: 21,341   RSS: 13795.1   RMSE: 0.285   R2: 0.794504   within R2: 0.018886
solver: {'S_nnz': 57969, 'S_mb': 0.7, 'S_build_seconds': 0.04, 'solver': 'explicit', 'iterations': 16, ...}
┌─────────────┬──────────┬────────────┬───────────┬────────────┐
│ Coefficient ┆ Estimate ┆ Std. Error ┆ t value   ┆ Pr(>|t|)   │
╞═════════════╪══════════╪════════════╪═══════════╪════════════╡
│ age_squared ┆ 0.081721 ┆ 0.013262   ┆ 6.162279  ┆ 7.3077e-10 │
│ age_cubed   ┆ -0.01211 ┆ 0.001035   ┆ -11.69974 ┆ 1.6196e-31 │
└─────────────┴──────────┴────────────┴───────────┴────────────┘
```

## Reading the results

The result is an [`HDFEResult`](../reference/results.md). The coefficients come
back as a dict (`fit.coef()`), as arrays (`fit.beta`, `fit.se`), or as a Polars
DataFrame (`fit.tidy()`); `fit.n_levels` counts the levels of each dimension,
`fit.k_fe` the fixed-effect parameters net of the redundant ones, and
`fit.n_obs` the observations used.

The estimated fixed effects and the row-level output are Parquet files in the
fit's run directory, and come back as lazy Polars scans, so nothing is read
until you ask:

```python
fit.fixef("firm_id").sort("fe_firm_id", descending=True).head(5).collect()
```

```text
┌─────────┬────────────┬───────┬───────────┐
│ firm_id ┆ fe_firm_id ┆ n_obs ┆ component │
╞═════════╪════════════╪═══════╪═══════════╡
│ F1106   ┆ 1.094846   ┆ 82    ┆ 0         │
│ F1128   ┆ 0.863573   ┆ 58    ┆ 0         │
│ F710    ┆ 0.835668   ┆ 46    ┆ 0         │
│ ...     ┆            ┆       ┆           │
└─────────┴────────────┴───────┴───────────┘
```

`fixef()` gives one row per level, with the number of observations behind each
effect and its connected component. `resid()` gives one row per observation,
with the outcome, the covariate contribution `xb`, each fixed effect and the
residual, so aggregates such as a [variance decomposition](akm_variance.md) run
as a streaming pass over it. The files live as long as the result object:
collect or sink what you need before dropping it (or fit with
`outputs="keep"`; see [out of core](out_of_core.md#the-disk)).

The model-level information of the summary (observations, fixed-effect counts,
fit statistics, solver) is also available as JSON with `summary_json()`,
optionally written to a file, and as a dict with `summary_dict()`;
`coefficients=True` adds the coefficient table, and `summary_text()` returns the
printed summary as a string:

```python
fit.summary_json("fit.json", coefficients=True)
```

To put the results in a regression table, or to plot them, see
[reporting](reporting.md); to match the residuals back to the rows of the
source, see [the sample](formulas.md#missing-values-and-the-sample).

:::{dropdown} The full script: `examples/quickstart.py`
```{literalinclude} ../../examples/quickstart.py
:language: python
```
:::

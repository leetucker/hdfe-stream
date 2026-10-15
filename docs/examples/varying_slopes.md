# Varying slopes

A worker fixed effect allows each worker their own level. Varying slopes allow
each worker their own *trend* as well (fixed effects with individual slopes,
FEIS): `worker_id[t]` is a worker intercept plus a worker-specific slope on
`t`, and `worker_id[t, t2]` adds a second slope. That is a lot of parameters,
one per worker per slope, which is exactly the case that does not fit in memory
as a design matrix, and exactly the case this library handles by keeping the
work inside each worker's own rows: projecting out the streamed dimension means
removing each worker's own regression on `[1, t, t2]`, which is local to the
worker.

## Worker-specific trends

In the example's panel each worker has their own linear and quadratic trend, so
`y ~ x | worker_id[t, t2] + firm_id` is the correctly specified model, and the
coefficient on `x` identifies 0.36:

```python
intercepts = feols_stream("y ~ x | worker_id + firm_id", data, workdir="scratch")
linear = feols_stream("y ~ x | worker_id[t] + firm_id", data, workdir="scratch")
quadratic = feols_stream("y ~ x | worker_id[t, t2] + firm_id", data, workdir="scratch")
```

```text
specification                               x        se         RSS  FE params
------------------------------------------------------------------------------
worker_id (intercepts only)          0.356903  0.005124      5842.6      2,059
worker_id[t] (+ linear trend)        0.356310  0.005435      4936.0      4,059
worker_id[t, t2] (+ quadratic)       0.356917  0.005895      4163.3      6,059
```

Adding the trends the data actually has cuts the residual sum of squares, at
the cost of many more fixed-effect parameters. The fixed-effect file carries the
intercept and one column per slope (`fe_worker_id`, `fe_worker_id[t]`), and the
residual file's `fe_worker_id` column is the worker's total contribution for that
row.

A covariate inside the span of the worker intercepts and their trends cannot be
identified, and is dropped as collinear. Here `age` is birth year plus `t`, so
`age_squared` lies in the span of `{1, t, t2}` within a worker.

Varying slopes are available on the streamed dimension only, and only one
dimension may carry them. They are not available with `solver="within"`, or for
Poisson, logit and probit.

## The streamed dimension

One dimension is never held in memory as a vector: it is streamed. It should
be the one with the most levels whose groups each touch only a few levels of
the others: workers, not firms. It is chosen for you: an explicit `stream=` if
given, else the dimension carrying varying slopes, else the highest
approximate cardinality (counted while the data is first scanned; ties go to
the dimension listed first). `result.diagnostics["stream"]` says which and why:

```text
  fe=['worker_id', 'firm_id']        default           -> worker_id
      because: highest cardinality (worker_id ~1,957, firm_id ~61)
  fe=['firm_id', 'worker_id']        default           -> worker_id
      because: highest cardinality (firm_id ~61, worker_id ~1,957)
  fe=['worker_id[t]', 'firm_id']     default           -> worker_id
      because: varying slopes
  fe=['worker_id', 'firm_id']        stream='firm_id'  -> firm_id
      because: stream= option
```

Pass `stream=` explicitly when cardinality is a poor guide, for example
firm-year effects that outnumber workers in a subsample.

## The low-level interface

[`StreamingHDFE`](../reference/lowlevel.md) takes column names or
`(name, Polars expression)` pairs instead of a formula, which is handy when a
covariate is easier to write in Polars, and it needs no pyfixest at all:

```python
import polars as pl
from hdfe_stream import StreamingHDFE

est = StreamingHDFE(
    y="y",
    x=["x", ("late_period", (pl.col("year") >= 2010).cast(pl.Float64))],
    fe=["worker_id[t]", "firm_id"],
    workdir="scratch",
)
est.fit(data, vcov={"CRV1": "firm_id"}).tidy()
```

:::{dropdown} The full script: `examples/varying_slopes.py`
```{literalinclude} ../../examples/varying_slopes.py
:language: python
```
:::

# AKM variance decomposition

This is the application the library was written for. Having estimated worker
and firm effects, you want to know how much of the variance of log earnings
each one explains, and how much comes from the covariance between them:
whether good workers sort into good firms.

Fitted values split additively into the covariate part and one term per
fixed-effect dimension,

$$
y = X\beta + \text{worker} + \text{firm} + \text{year} + \text{residual},
$$

so the variance of $y$ is the sum of those five variances plus twice every
pairwise covariance between them. Each term is an aggregate over rows, and
`resid()` hands back a lazy scan of the row-level output, so Polars computes
them in a streaming pass: the numbers come back without the rows ever being
collected.

## The fit

```python
fit = feols_stream(
    "log_earn ~ age_squared + age_cubed | worker_id + firm_id + year",
    "data/panel.parquet", workdir="scratch", vcov={"CRV1": "worker_id"},
    keep=["region"],         # carry region into the row-level file
)
```

The row-level file holds, per observation, the outcome, the covariate
contribution `xb`, one column per fixed-effect dimension with that row's
estimated effect, and the residual, along with the fixed-effect ids and
`row_id`. `keep=` carries further columns of the source into it, here `region`,
for the decomposition by region below.

## The decomposition, in one streaming pass

```python
import polars as pl

PARTS = ["xb", "fe_worker_id", "fe_firm_id", "fe_year", "resid"]
stats = fit.resid().select(
    pl.col("log_earn").var().alias("var_y"),
    *[pl.col(c).var().alias(f"var_{c}") for c in PARTS],
    pl.cov("fe_worker_id", "fe_firm_id").alias("cov_worker_firm"),
    pl.corr("fe_worker_id", "fe_firm_id").alias("corr_worker_firm"),
).collect(engine="streaming").row(0, named=True)
```

`engine="streaming"` keeps Polars from materializing the frame. On the
simulated panel:

```text
Variance of log earnings: 0.3953

component                            variance    share
------------------------------------------------------
worker effects                         0.1738   44.0%
firm effects                           0.0759   19.2%
year effects                           0.0002    0.1%
covariates (Xb)                        0.0054    1.4%
residual                               0.0812   20.5%
2 x covariances between the parts      0.0587   14.8%
------------------------------------------------------
total                                  0.3953  100.0%

2 x cov(worker, firm) alone: 0.0596 (15.1% of Var(y))
Correlation of worker and firm effects: 0.259
```

A positive correlation means assortative matching: better workers at better
firms. The same moments by group are a `group_by` in the same streaming pass;
`region` is in the file because the fit was asked to keep it. Any column of the
source could be, or the residuals can be joined back onto the source on
`row_id` instead (see [the sample](formulas.md#missing-values-and-the-sample)).

## Identification

Worker and firm effects are only comparable within a connected set of firms,
linked by workers who moved between them. With more than one component, effects
in different components are not comparable. The fit reports how many components
the worker–firm network has (`fit.n_components`), and `fixef("firm_id")` gives
each firm's `component`, numbered by size so that component 0 is the largest.
[Connected sets](connected_sets.md) shows how to find them before fitting, and
how to restrict a fit to the largest one.

## The bias in these numbers

With few observations per worker, the variance of the worker effects is biased
up and their covariance with the firm effects biased down, by estimation error:
the "limited mobility bias" of Andrews et al. (2008). This example reports the
plug-in decomposition, which is what the estimated effects give you. The number
of observations per worker, 8.5 here, is the one to judge it by, since the bias
grows as it falls. [`leave_out_kss`](leave_out_kss.md) removes the bias; the
[leave-out variance components](../kss/index.md) section explains how.

:::{dropdown} The full script: `examples/akm_variance.py`
```{literalinclude} ../../examples/akm_variance.py
:language: python
```
:::

# Formulas, standard errors and the sample

Formulas are parsed by pyfixest's own parser, so the syntax and the resulting
coefficient names are pyfixest's. Every covariate term is then compiled into a
Polars expression, so the design is built while the data streams, one bucket at
a time, and never has to be built in memory: a categorical expanded into
hundreds of indicators is carried through the partitioning as the one column it
comes from.

## Covariates

**Transformations and interactions.** `I()` for arithmetic, `log()`, `exp()`
and `sqrt()`, and `:` and `*` for interactions:

```python
feols_stream("log_earn ~ I(age ** 2) + log(age) + age_squared:age_cubed"
             " | worker_id + firm_id + year", data, workdir="scratch")
```

**Categoricals.** `C(x)` for a categorical covariate; string columns work too.
One level is absorbed as the reference, exactly as pyfixest chooses it, and the
coefficients are named as in pyfixest (`C(occ)[T.1]`, `cat[k0]:age_squared`):

```python
feols_stream("log_earn ~ C(occ) + cat:age_squared | worker_id + firm_id + year",
             data, workdir="scratch")
```

**Event studies.** `i(over, by, ref=...)` gives one coefficient per level of
`over`, interacted with `by`, dropping the reference level. The simulated data
has a treatment effect of 0.1 that switches on in 2010:

```python
event = feols_stream(
    "log_earn ~ i(year, treat, ref=2009) + age_squared | worker_id + firm_id",
    data, workdir="scratch", vcov={"CRV1": "firm_id"})
```

```text
│ year::2008:treat ┆ -0.017075 ┆ 0.006682   ┆ -2.555356 ┆ 0.010718   │
│ year::2010:treat ┆ 0.098927  ┆ 0.007478   ┆ 13.228988 ┆ 1.2587e-37 │
│ year::2011:treat ┆ 0.102635  ┆ 0.00711    ┆ 14.435942 ┆ 5.3516e-44 │
```

[Reporting](reporting.md) plots it with pyfixest's `iplot`.

## Fixed effects

Any number of dimensions works, including one (a within regression) and none
(OLS with an intercept); see [one fixed effect, or none](fewer_fixed_effects.md).
`^` combines columns into one dimension, with a separate effect per firm–year
cell:

```python
feols_stream("log_earn ~ age_squared + age_cubed | worker_id + firm_id^year",
             data, workdir="scratch", vcov={"CRV1": "firm_id"})
```

Varying slopes on the streamed dimension use fixest syntax, `worker_id[t]`; see
[varying slopes](varying_slopes.md).

**Singletons.** Observations alone in their level of some fixed effect are
dropped, as in pyfixest, repeating until none is left, since dropping one can
create another in a different dimension (`fixef_rm="singleton"`, the default;
`fixef_rm="none"` keeps them). They contribute nothing to the coefficients but
count toward N, the number of clusters and the fixed-effect parameters. Finding
them costs a few extra passes over the fixed-effect columns and memory for the
singleton levels, not for the rows.

**Connected components** of the worker–firm graph are computed and reported,
and a fit can be restricted to the largest connected set with
`connected="largest"`; see [connected sets](connected_sets.md).

## Several models at once

Several outcomes (`y1 + y2 ~ ...`), cumulative or plain stepwise covariate sets
(`csw()`, `sw()`), and stepwise fixed-effect sets:

```python
multi = feols_stream(
    "log_earn + y2 ~ csw(age_squared, age_cubed)"
    " | sw(worker_id + firm_id, worker_id + firm_id + year)",
    data, workdir="scratch", vcov="hetero")
len(multi)                 # 8 models
multi.tidy()               # one table, with a column for the formula
multi.fetch_model(0)       # or iterate over multi
```

Models sharing a fixed-effect set share one pass over the data and one solve,
which is the point of asking for them together. They are estimated on the rows
where *all* of their variables are non-missing (pyfixest drops missing values
model by model). The result is an [`HDFEMulti`](../reference/results.md).

## Missing values and the sample

Rows with a null or non-finite value in any variable the formula uses are
dropped, as pyfixest drops them. In the simulated data `x_na` is null in about
5% of rows:

```python
fit = feols_stream("log_earn ~ age_squared + x_na | worker_id + firm_id + year",
                   data, workdir="scratch")
fit.n_obs                  # 161,340 of the file's 169,833 rows
```

`fit.sample()` says which rows were used and why the others were not. It returns
the source, lazily, with three columns added: `row_id`, each row's position in
the source before any row was dropped; `in_sample`; and `dropped_because`, null
for the rows used and otherwise `"missing"`, `"min_degree"`, `"disconnected"`,
`"leave_one_out"`, `"singleton"` or `"separation"`:

```python
fit.sample().group_by("dropped_because").len().collect()
```

```text
┌─────────────────┬────────┐
│ dropped_because ┆ len    │
╞═════════════════╪════════╡
│ null            ┆ 161340 │
│ missing         ┆ 8493   │
└─────────────────┴────────┘
```

The residual file is in the estimator's order, not the source's, so every row
carries `row_id` (`row_id=` renames the column, for a source that already has
one by that name). `resid()` joins back onto the source on it, for looking at
the data net of what the fit dropped, or at the residuals by a column the model
did not use:

```python
rows = fit.sample().join(fit.resid().select("row_id", "resid"), on="row_id", how="left")
rows.group_by("occ").agg(pl.col("resid").mean()).collect()
```

Nothing is stored per row for this: the frame is the source plus the small
tables of dropped levels, and `row_id` adds a few bytes per row to the working
and residual files. It relies on the source reading back in the same order,
which a file, a glob or a DataFrame does and a LazyFrame does only if its plan
does (not after a `group_by`, an unordered join or a random sample). For a model
list, `multi.sample()` returns `{formula: frame}`; models fitted on the same
fixed effects share one sample.

## Degrees of freedom

The standard errors count the fixed-effect parameters, net of the redundant
ones. The correction either removes the redundant levels it can identify
(`fe_dof="exact"`, the default) or follows pyfixest's convention
(`fe_dof="pyfixest"`, one per fixed effect beyond the first). `"exact"` counts
the connected components of each pair of fixed-effect dimensions and uses the
pair with the most, plus one redundant level for each further dimension. With
two fixed effects this is the exact number of identified parameters. With more
it is an upper bound, so standard errors are conservative. For clustered
standard errors, fixed effects nested within the clusters are dropped from the
count as in pyfixest, and a pair of two nested fixed effects is not used.

On a connected panel the two agree exactly. On a panel whose network falls
apart, pyfixest's count includes a parameter for every component past the first
that cannot be estimated, so its standard errors are a little too large. On the
example's thin panel, with 381 components:

```text
panel       fe_dof     components  FE params       df  se(age_squared)
connected   exact               1     21,332  148,500        0.0003078
connected   pyfixest            1     21,332  148,500        0.0003078
thin        exact             381     21,405   78,439        0.0004183
thin        pyfixest          381     21,785   78,059        0.0004193
```

## Collinear terms

Terms spanned by the fixed effects are dropped exactly as pyfixest drops them,
with a warning. Here `i(year)` duplicates the year effect, and `treat` is
constant within worker, so `C(treat)` duplicates the worker effect:

```python
collinear = feols_stream(
    "log_earn ~ i(year) + C(treat) + age_squared | worker_id + firm_id + year",
    data, workdir="scratch")
collinear.collin_vars      # ['year::2006', ..., 'year::2014', 'C(treat)[T.1]']
```

A covariate the streamed dimension absorbs entirely is recognized during the
solve rather than costing iterations, and is listed in
`solver_info["fe_spanned"]`.

## Standard errors

`iid`, heteroskedasticity-robust (`hetero`/HC1), and CRV1 clustered on any
column, any `^` interaction, or several dimensions at once (multi-way
Cameron–Gelbach–Miller, any number of ways). The default is `iid`, as in
pyfixest. Ask for several at fit time with `cluster=` and switch between them
afterwards with `with_vcov()`: they are all computed in the one residual pass.

```python
fit = feols_stream(
    "log_earn ~ age_squared + age_cubed | worker_id + firm_id + year",
    data, workdir="scratch",
    vcov={"CRV1": "worker_id"},
    cluster=["firm_id", "state", "worker_id+firm_id", "firm_id^year"])
fit.with_vcov("CRV1:state").se
```

```text
vcov                         se(age_squared)        clusters
iid                                 0.013373               -
hetero                              0.013304               -
CRV1:worker_id                      0.013262         [20000]
CRV1:firm_id                        0.013027          [1333]
CRV1:state                          0.008219            [12]
CRV1:worker_id+firm_id              0.012925   [20000, 1333]
CRV1:firm_id^year                   0.013115         [13296]
```

**CRV3**, the cluster jackknife (`vcov={"CRV3": var}`), is computed by
downdating the fit, which is exact when every fixed effect is nested within the
clusters (firm-by-year effects clustered by firm, say) or there are none. It is
one-way and OLS only. With fixed effects that are not nested, as in worker and
firm effects clustered by worker, the jackknife would have to refit once per
cluster, and the fit refuses; use CRV1, or pyfixest, which refits.

```python
nested = feols_stream("log_earn ~ age_squared + age_cubed | firm_id^year", data,
                      workdir="scratch", vcov={"CRV3": "firm_id"}, cluster=["firm_id"])
```

:::{dropdown} The full script: `examples/formulas.py`
```{literalinclude} ../../examples/formulas.py
:language: python
```
:::

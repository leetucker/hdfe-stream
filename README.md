# hdfe-stream

Linear regression with several high-dimensional fixed effects, on data that does
not fit in memory.

```python
from hdfe_stream import feols_stream

fit = feols_stream(
    "log_earn ~ age_squared + age_cubed | worker_id + firm_id + year",
    "data/*.parquet",              # a Parquet path or glob, never fully loaded
    workdir="scratch",             # where intermediates go: disk, not memory
    vcov={"CRV1": "worker_id"},
)
fit.summary()
```

Formulas follow [pyfixest](https://github.com/py-econometrics/pyfixest)/fixest
syntax and are parsed by pyfixest's own parser, so the syntax, the coefficient
names and the results all match. The difference is where the work happens: rows
are streamed off disk in batches and never held as a design matrix.

## When to use this, and when not to

**Use pyfixest.** It is the better tool whenever your data fits in memory:
mature, broader in scope, more inference options, and no scratch directory to
think about. If a `pandas` DataFrame of your data fits comfortably in RAM, stop
reading.

**Use this** when it doesn't. The case it was built for is a matched
employer–employee panel — tens of millions of rows, millions of worker effects,
an AKM variance decomposition — on a machine or a secure enclave where the data
is larger than the memory you are allowed. Concretely, reach for it when:

- the data does not fit in memory, or fits so tightly that nothing else does;
- you have one very high-cardinality dimension (workers, patients, students)
  whose groups each touch only a few levels of the others;
- you want **worker-specific slopes** as well as intercepts (`worker_id[t]`),
  which multiplies the parameter count by the number of slopes;
- you need a hard, predictable memory ceiling — a shared cluster, or a job that
  must not be the one that gets killed.

You are trading memory for disk, and you need scratch space: about five times
the size of your Parquet input while the fit runs, and only the result files
once it finishes. See [benchmarks](#benchmarks) for what that buys.

## How it works, in one paragraph

One fixed-effect dimension — normally the largest — is **streamed**: it
is never represented as a vector in memory. Every other dimension is held as a
vector sized by its number of *levels*, not rows. Rows are read in batches,
hash-partitioned into buckets by the streamed dimension, sorted one bucket at a
time, and reduced to a table of cells (one per combination of fixed-effect
levels). The reduced normal equations for the non-streamed dimensions are then
solved by conjugate gradient, and the streamed effects are recovered group by
group in a final pass that writes residuals and fixed effects straight to
Parquet. So memory scales with firms × years, not with workers × rows, and that
asymmetry is the whole design. Full detail is in
[`hdfe_stream/__init__.py`](hdfe_stream/__init__.py).

## Install

```bash
pip install hdfe-stream[formula]     # formula syntax needs pyfixest
```

`polars`, `numpy`, `numba`, `pyarrow` and `scipy` are required. The extras:

| extra | what it adds |
|---|---|
| `formula` | `feols_stream` formula syntax and pyfixest reporting (pyfixest, formulaic, pandas) |
| `within` | the `solver="within"` backend |
| `amg` | algebraic multigrid preconditioning (`precond="amg"`) for hard geometries |
| `all` | all of the above |
| `test` | the test suite |

Without `formula` you still get `StreamingHDFE`, the lower-level interface that
takes column names or Polars expressions instead of a formula string.

## Features

**Fixed effects.** Any number of dimensions. Interactions with `^`
(`firm_id^year`). Varying slopes on the streamed dimension in fixest syntax
(`worker_id[t]`, `worker_id[t, t2]`). Connected components of the
worker–firm graph are computed and reported, and the degrees-of-freedom
correction counts the fixed-effect parameters that are actually identified
(`fe_dof="exact"`) or follows pyfixest's convention (`fe_dof="pyfixest"`).

**Covariates.** Transformations (`I(age**2)`, `log(x)`), categoricals (`C(x)`,
string columns), interactions (`:`, `*`), event-study terms
(`i(year, treat, ref=2009)`). Terms spanned by the fixed effects are dropped
exactly as pyfixest drops them. Every term compiles to a Polars expression, so
the design is built while the data streams.

**Standard errors.** `iid`, heteroskedasticity-robust (`hetero`/HC1), and CRV1
clustered on any column, any `^` interaction, or several dimensions at once
(multi-way Cameron–Gelbach–Miller, any number of ways). Ask for several at fit
time with `cluster=` and switch between them afterwards with `with_vcov()` —
they are all computed in the one residual pass.

**Estimators.** OLS, weighted least squares (`weights=`, analytic or frequency),
and 2SLS (`y ~ exog | fe | endog ~ instruments`) with a first-stage F.

**Multiple models.** Several outcomes (`y1 + y2 ~ ...`), stepwise covariate sets
(`sw()`, `csw()`) and stepwise fixed-effect sets. Models sharing a fixed-effect
set share one pass over the data and one solve.

**Output.** Coefficients as a Polars DataFrame (`tidy()`); residuals and
per-row fixed effects as lazy Polars scans, so aggregates like a variance
decomposition run as a streaming pass; estimated effects per dimension via
`fixef()`. `to_pyfixest()` converts a result into a pyfixest model for
`pf.etable`, `pf.summary`, `pf.coefplot` and `pf.iplot`, and `hdfe_stream.etable`
mixes streaming and in-memory models in one table.

**Operations.** Progress, warnings and summaries to a `logging` logger as they
happen. Each fit works in its own run directory; intermediates are deleted as
soon as they are no longer needed, everything is removed if the fit fails, and
`hdfe_stream.cleanup()` sweeps up after killed jobs.

## Benchmarks

A standard three-way AKM specification,
`log_earn ~ age_squared + age_cubed | worker_id + firm_id + year`, clustered by
worker, on the simulated panel that ships with the library. Reproduce with:

```bash
python benchmarks/akm_benchmark.py
```

Wall time includes reading the data, because pyfixest needs it as a pandas
DataFrame before it can start and hdfe_stream reads it itself — that difference
is the comparison, not an artifact. Peak memory is `VmHWM` for the whole
process. Every configuration is run in a separate process, and all of them agree
on the coefficients to within 3e-11, so these are times for equally good
answers.

**8,500,353 rows, 1,000,000 worker effects, 66,666 firms**

| configuration | wall time | peak memory | peak disk |
|---|---:|---:|---:|
| pyfixest (MapDemeaner, default) | 87.0 s | 4,519 MB | — |
| pyfixest (LsmrDemeaner) | 13.6 s | 5,656 MB | — |
| hdfe_stream (`solver="explicit"`) | 8.4 s | 3,797 MB | 779 MB |
| hdfe_stream (`solver="stream_cg"`) | 7.5 s | 3,816 MB | 779 MB |
| hdfe_stream (`solver="within"`) | 11.7 s | 3,870 MB | 779 MB |
| **hdfe_stream (`stream_cg`, low memory)** | **8.3 s** | **1,318 MB** | 795 MB |

**3,399,908 rows, 400,000 worker effects**

| configuration | wall time | peak memory | peak disk |
|---|---:|---:|---:|
| pyfixest (MapDemeaner, default) | 33.0 s | 2,132 MB | — |
| pyfixest (LsmrDemeaner) | 5.1 s | 2,670 MB | — |
| hdfe_stream (`solver="explicit"`) | 3.9 s | 2,128 MB | 313 MB |
| hdfe_stream (`solver="stream_cg"`) | 3.5 s | 2,203 MB | 313 MB |
| hdfe_stream (`solver="within"`) | 4.7 s | 2,101 MB | 313 MB |
| **hdfe_stream (`stream_cg`, low memory)** | **3.6 s** | **1,041 MB** | 317 MB |

**850,191 rows, 100,000 worker effects**

| configuration | wall time | peak memory | peak disk |
|---|---:|---:|---:|
| pyfixest (MapDemeaner, default) | 7.0 s | 845 MB | — |
| pyfixest (LsmrDemeaner) | 1.8 s | 949 MB | — |
| hdfe_stream (`solver="explicit"`) | 1.5 s | 1,286 MB | 78 MB |
| hdfe_stream (`solver="stream_cg"`) | 1.4 s | 1,226 MB | 78 MB |
| hdfe_stream (`solver="within"`) | 1.6 s | 1,318 MB | 78 MB |
| hdfe_stream (`stream_cg`, low memory) | 1.5 s | 924 MB | 78 MB |

**212,307 rows, 25,000 worker effects**

| configuration | wall time | peak memory | peak disk |
|---|---:|---:|---:|
| pyfixest (MapDemeaner, default) | 1.5 s | 505 MB | — |
| pyfixest (LsmrDemeaner) | 1.2 s | 526 MB | — |
| hdfe_stream (`solver="explicit"`) | 0.9 s | 779 MB | 19 MB |
| hdfe_stream (`solver="stream_cg"`) | 0.9 s | 783 MB | 19 MB |
| hdfe_stream (`solver="within"`) | 1.0 s | 813 MB | 19 MB |
| hdfe_stream (`stream_cg`, low memory) | 0.9 s | 796 MB | 19 MB |

### Reading the tables

**The default pyfixest demeaner is not the one to compare against.** `MapDemeaner`
(alternating projections) is about 6x slower than `LsmrDemeaner` on the two
larger panels here, and rather closer on the small ones: a million worker
effects is exactly the case alternating projections struggles with. If you are
comparing, compare against LSMR.

**The last row of each table is the point.** `rows_per_bucket` and `batch_rows`
control how much data is in flight at once. Turning them down cuts peak memory
by a factor of three at 8.5M rows, for about 10% more wall time and the same
answer to ten decimal places. pyfixest has no equivalent knob — its memory is
whatever the design matrix needs.

**Below about a million rows, hdfe_stream uses more memory, not less.** Running
any fit at all costs about 540 MB — 223 MB of it just importing polars, numba
and scipy, the rest Polars' streaming engine and numba's thread pools — and on
small data that floor swamps everything else. This is the concrete version of
"use pyfixest if your data fits": at 212,307 rows pyfixest uses about a third
less memory. With the settings tuned, hdfe_stream draws level around a million
rows and pulls away above that.

**CPU time can be comparable to `pyfixest`, depending on your data.** With
AKM panel data, the extra disk traffic involved in using this library is offset
by reducing rows to cells before solving. The simulated panel has about 8.5 rows
per worker and one cell per row, which is friendly to this approach. A
specification where the cell table is no smaller than the data and the fixed
effects are less local may perform worse.

### Trading memory for time

Same data, same solver, same answer — only how much is in flight at once:

```bash
python benchmarks/akm_benchmark.py --memory-sweep 1000000
```

**8,500,353 rows, 1,000,000 worker effects**

| `rows_per_bucket` | `batch_rows` | wall time | peak memory | peak disk |
|---:|---:|---:|---:|---:|
| 20,000,000 (default) | 2,000,000 | 7.4 s | 3,772 MB | 779 MB |
| 1,000,000 | 500,000 | 7.2 s | 1,552 MB | 780 MB |
| 250,000 | 200,000 | 8.4 s | 1,339 MB | 795 MB |
| 100,000 | 100,000 | 9.5 s | 1,268 MB | 789 MB |

Most of the saving comes from the first step down. The defaults are tuned for a
machine with room to spare; if memory is the binding constraint, set
`rows_per_bucket` to something near what you can afford and leave the rest
alone. `rhs_block` bounds the other big array (levels × variables), and
`max_s_gb` caps the explicit reduced matrix, above which `solver="auto"` falls
back to `stream_cg` by itself.

Measured on an Intel Core Ultra 7 258V, 8 cores, 15 GB RAM, Linux (WSL2);
Python 3.13.5, polars 1.44.2, numpy 2.5.3, numba 0.67.0, scipy 1.18.1,
pyfixest 0.60.0. Numba kernels are compiled on first use and cached to disk; the
benchmark runs a warm-up fit so compilation is not charged to any configuration.

## Choosing a solver

| `solver` | what it does | when |
|---|---|---|
| `"auto"` (default) | `explicit`, falling back to `stream_cg` if the reduced matrix would exceed `max_s_gb` | leave it alone |
| `"explicit"` | builds the reduced matrix once and solves it in memory | fastest when it fits; its size depends on how levels co-occur, not on rows |
| `"stream_cg"` | applies the reduced matrix by streaming the cell table each iteration | when the reduced matrix is the thing that does not fit |
| `"within"` | hands the reduced system to `within-py` | an alternative; no varying-slopes support |

The streamed dimension is chosen for you: an explicit `stream=` if given, else
the dimension carrying varying slopes, else the highest approximate cardinality.
`result.diagnostics["stream"]` says which and why.

## Examples

Runnable, on simulated data, no setup — see [examples/](examples/):

| | |
|---|---|
| [`quickstart.py`](examples/quickstart.py) | fit a model and read the results |
| [`akm_variance.py`](examples/akm_variance.py) | the variance decomposition, in a streaming pass |
| [`formulas.py`](examples/formulas.py) | the formula syntax, end to end |
| [`weights_and_iv.py`](examples/weights_and_iv.py) | weights and 2SLS |
| [`varying_slopes.py`](examples/varying_slopes.py) | worker-specific trends, and the low-level interface |
| [`out_of_core.py`](examples/out_of_core.py) | memory, disk, solvers, logging |
| [`reporting.py`](examples/reporting.py) | tables and plots via pyfixest |

## Simulated data

`hdfe_stream.simulate` ships with the package, so you can try the library, or
check your own code, against data whose answer you know:

```python
from hdfe_stream.simulate import simulate_akm

simulate_akm(n_workers=1_000_000).write_parquet("sim.parquet")
```

`simulate_akm` draws worker and firm effects explicitly and positively assorted,
with enough mobility to connect the firms into one component. `simulate_rich`
adds categoricals, non-fixed-effect cluster variables, weights, an IV block and
missing values; `simulate_trends` adds worker-specific time trends. All are
deterministic given a `seed`.

## Correctness

The test suite checks every feature against pyfixest on simulated data —
coefficients, standard errors under each vcov, residuals, fixed effects, fit
statistics, the sample kept after dropping missing values, and the exact set of
coefficient names and dropped collinear terms. Varying slopes have no pyfixest
equivalent, so they are checked against a brute-force regression with explicit
worker-by-slope dummies; three-way clustering is checked against an
inclusion–exclusion sum built from pyfixest's own score matrix.

```bash
pip install -e ".[test]"
pytest
```

## Limitations

- **Two fixed-effect dimensions minimum.** For one, use pyfixest.
- **Varying slopes only on the streamed dimension**, and only one dimension may
  carry them.
- **CRV1 only** for clustered errors: no CRV3, no wild bootstrap, no jackknife.
- **A converted result cannot recompute its own vcov**, because it holds no
  data. Ask for what you need at fit time via `cluster=`.
- **Formula support pins a pyfixest range** (`>=0.50,<0.61`), because it uses
  pyfixest's internal formula modules. `StreamingHDFE` has no such dependency.
- **Needs scratch disk**, roughly the size of your data during the fit.
- **Limited-mobility bias** is not corrected for. The variance decomposition you
  get is the plug-in one; if you need bias-corrected AKM estimands, compute them
  from the output yourself.

# hdfe-stream

Regression with several high-dimensional fixed effects, on data that does not
fit in memory: linear (`feols_stream`), and Poisson, logit and probit
(`fepois_stream`, `feglm_stream`), with the Kline–Saggio–Sølvsten bias
correction for AKM variance decompositions (`leave_out_kss`).

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

The data can be a Parquet path or glob, a Polars LazyFrame, or a Polars
DataFrame. Using file paths or LazyFrames keeps memory usage to a minimum, but
using DataFrames works well for smaller data and for trying things out.

## Why

The leading example of a model with several high-dimensional fixed effects is
the AKM model of Abowd, Kramarz and Margolis (1999). It decomposes log earnings
into a worker effect, an employer effect and observable characteristics, and is
now estimated routinely on matched employer–employee data that cover entire
national labor markets: tens of millions of workers, hundreds of thousands of
firms, and hundreds of millions of person-years. The same structure arises
whenever two large populations are linked: students and teachers, patients and
physicians, exporters and importers.

The current generation of software (reghdfe, fixest, pyfixest, xhdfe) solved
the problem of the parameter count: none of them forms the normal equations,
and they handle three or more dimensions of fixed effects. What they share is
that the data, and usually several working copies of the variables being
demeaned, are held in memory. The binding constraint has moved from the
parameters to the observations, but it has not disappeared. Published work still
reports design choices made for computational reasons, and many researchers who
work with administrative data do so inside secure environments where memory is
fixed per project.

hdfe-stream holds only a bounded number of rows in memory at any time. It rests
on two elementary observations. Least squares never needs the full data matrix,
since $X'X$ and $X'y$ are sums over rows; by the Frisch–Waugh–Lovell theorem,
the same holds for a model with fixed effects once they have been partialled
out. And partialling them out needs much less than the data: removing the
largest dimension (workers) is local to each worker's own rows, and the rest
depend on the data only through counts and sums over cells of the fixed-effect
design.

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
- the design is **wide**: hundreds of covariates, such as a categorical
  expanded into fine indicators or many interactions. An in-memory library
  needs rows × covariates in memory before it starts; this needs one batch of
  rows × covariates at a time, with any number of fixed effects, including one
  or none (see [many covariates](benchmarks/covariates.md));
- you need a hard, predictable memory ceiling — a shared cluster, or a job that
  must not be the one that gets killed.

You are trading memory for disk, and you need scratch space: about five times
the size of your Parquet input while the fit runs, and only the result files
once it finishes. See [benchmarks](benchmarks/index.md) for what that buys:
on a 42.5-million-row panel, the AKM regression took 22 seconds and 3.9 GB,
where the other libraries needed 9–27 GB. Point `workdir=` at the scratch space,
or set the `HDFE_STREAM_WORKDIR` environment variable once (for example to a
cluster's scratch filesystem) and leave `workdir` out; a `workdir` passed to a
call takes precedence, and with neither, the system temporary directory is
used.

## Should I use a GPU?

Probably not for this. hdfe_stream has experimental GPU options (`device="cuda"`
for the solve, `polars_engine="gpu"` for the Polars queries), but on an NVIDIA
H100 their gains were modest throughout and appeared only at large sizes:
about 10% at 5 million workers with both options on, and no consistent gain
below that. A GPU-oriented library such as
[xhdfe](https://github.com/reisportela/xhdfe-xfe) is the better choice if GPU
performance is the priority: its CUDA backend took 24 s at 5 million workers,
against 107 s on its CPU backend. But hdfe_stream on a 16-vCPU CPU machine took
22 s, and that machine costs about an eighth as much per hour as AWS's smallest
H100 machine. [CPU and GPU](benchmarks/gpu.md) has the numbers, and how to try
the options.

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
asymmetry is the whole design. The covariates travel with the rows: what is
held at once is one bucket or batch of rows × covariates, never all of them,
which is why a wide design fits where an in-memory one does not.
[Out of core](examples/out_of_core.md) says what that means in practice, and
the full detail is in the docstring of
[`hdfe_stream/__init__.py`](https://github.com/leetucker/hdfe-stream/blob/main/hdfe_stream/__init__.py).

## Install

```bash
pip install "hdfe-stream[formula]"     # formula syntax needs pyfixest
```

`polars`, `numpy`, `numba`, `pyarrow` and `scipy` are required. The extras:

| extra | what it adds |
|---|---|
| `formula` | `feols_stream` formula syntax and pyfixest reporting (pyfixest, formulaic, pandas) |
| `within` | the `solver="within"` backend |
| `amg` | algebraic multigrid preconditioning (`precond="amg"`) for hard geometries |
| `all` | all of the above |
| `test` | the test suite |
| `docs` | building this documentation (Sphinx) |

On Python 3.15, pyfixest 0.60 (for the `formula` extra) has no wheels yet. It
builds from source with a Rust toolchain and
`PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1` set; everything else installs from
wheels.

Without `formula` you still get [`StreamingHDFE`](reference/lowlevel.md), the
lower-level interface that takes column names or Polars expressions instead of a
formula string.

The experimental GPU options need CuPy and cudf-polars for your CUDA version
(`pip install cupy-cuda13x cudf-polars-cu13`); there is no extra for them,
since the packages differ by CUDA version. See
[CPU and GPU](benchmarks/gpu.md#using-the-gpu-options).

## Where to go next

- **[Examples](examples/index.md)** walk through the library one feature at a
  time, on simulated data that ships with the package: formulas, standard
  errors, weights and IV, Poisson and logit, connected sets, running on data
  that does not fit in memory, and reporting.
- **[Leave-out variance components](kss/index.md)** covers the KSS bias
  correction: why it is needed, how to use it, its standard errors and
  weak-identification diagnostic, and every way it departs from the reference
  implementations.
- **[Benchmarks](benchmarks/index.md)** compare time, memory and disk with
  pyfixest, xhdfe, fixest and FixedEffectModels.jl, from 212,000 to 42.5
  million rows and from 8 to 503 covariates, and on a GPU.
- **[Function reference](reference/index.md)** documents every option.

## Limitations

- **Varying slopes only on the streamed dimension**, and only one dimension may
  carry them.
- **CRV3 only with nested fixed effects.** The cluster jackknife is computed by
  downdating the fit, which is exact when every fixed effect is nested within
  the clusters (worker effects clustered by worker, firm × year effects by
  firm) or there are none. With fixed effects that are not nested, as in an
  AKM model clustered by worker, the jackknife has to re-estimate the fixed
  effects once per cluster, and the fit refuses; use CRV1, or pyfixest, which
  refits. CRV3 is also one-way and OLS only (pyfixest has no IV CRV3 either).
  No CRV2 and no wild bootstrap.
- **Poisson, logit and probit:** no CRV3, IV, varying slopes or leave-out
  estimation; no detection of separation by the covariates (pyfixest's `"ir"`
  check), only by the fixed effects; and no correction for the incidental
  parameter bias of logit and probit with fixed effects estimated from few
  observations, as in pyfixest and fixest. During the iterations the streamed
  dimension's effects are held in memory, one float per group per coefficient
  set. See [GLM: not available](examples/glm.md#not-available).
- **A converted result cannot recompute its own vcov**, because it holds no
  data. Ask for what you need at fit time via `cluster=`.
- **Formula support pins a pyfixest range** (`>=0.60,<0.61`), because it uses
  pyfixest's internal formula modules. `StreamingHDFE` has no such dependency.
- **Needs scratch disk**, roughly the size of your data during the fit. Leave-out
  estimation adds to that: it memory-maps its row-sized accumulators rather than
  holding them, which is what keeps it inside a bounded memory footprint, at
  about 90 bytes per row of scratch while it runs.
- **Wide designs need the batch sizes scaled down** (see
  [wide designs](examples/wide_designs.md)): the defaults hold a whole bucket of
  rows with every covariate. And the cross-products cost k × k per row, so time
  grows with the square of the number of covariates.
- **Leave-out estimation handles one outcome at a time**, and its standard
  errors omit the split-sample refinement of KSS §4.2 (so they are conservative,
  as their Lemma 5 provides for). Under weak identification it supplies the
  q = 1 interval, not KSS's q > 1 generalization. Leaving out a match, as it
  does by default, var(alpha) has no standard error and the covariance no q = 1
  interval. See [what it does not do yet](kss/inference.md#what-it-does-not-do-yet).
- **Column names starting with `__hdfe_` are reserved** for the estimator's own
  columns, and a source that has one is refused. Any other name is safe.

```{toctree}
:hidden:

examples/index
kss/index
benchmarks/index
reference/index
changelog
```

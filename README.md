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

**Documentation:
[leetucker.github.io/hdfe-stream](https://leetucker.github.io/hdfe-stream/)**,
with examples, the KSS bias correction, benchmarks and the function reference.

## Features

- [Any number of fixed effects](https://leetucker.github.io/hdfe-stream/examples/formulas.html#fixed-effects),
  including [one or none](https://leetucker.github.io/hdfe-stream/examples/fewer_fixed_effects.html),
  and interactions (`firm_id^year`)
- [pyfixest formula syntax](https://leetucker.github.io/hdfe-stream/examples/formulas.html#covariates):
  transformations, categoricals, interactions, event studies
- [Several models in one call](https://leetucker.github.io/hdfe-stream/examples/formulas.html#several-models-at-once),
  sharing passes over the data
- [Standard errors](https://leetucker.github.io/hdfe-stream/examples/formulas.html#standard-errors):
  iid, robust, multi-way clustered, and one-way CRV3
- [Weighted least squares and 2SLS](https://leetucker.github.io/hdfe-stream/examples/weights_and_iv.html)
- [Poisson, logit and probit](https://leetucker.github.io/hdfe-stream/examples/glm.html)
- [Varying slopes in fixed effects](https://leetucker.github.io/hdfe-stream/examples/varying_slopes.html)
  (`worker_id[t]`), on the streamed dimension only
- [Wide designs](https://leetucker.github.io/hdfe-stream/examples/wide_designs.html),
  with hundreds of covariates
- [Connected components](https://leetucker.github.io/hdfe-stream/examples/connected_sets.html),
  the largest connected set, and `min_degree` pruning
- [AKM variance decompositions](https://leetucker.github.io/hdfe-stream/examples/akm_variance.html),
  in a streaming pass
- [KSS leave-out variance components](https://leetucker.github.io/hdfe-stream/kss/index.html),
  with standard errors and weak-identification diagnostics
- [The sample, row by row](https://leetucker.github.io/hdfe-stream/examples/formulas.html#missing-values-and-the-sample):
  which rows were dropped and why, and residuals joined back to the source
- [Regression tables and plots](https://leetucker.github.io/hdfe-stream/examples/reporting.html)
  through pyfixest
- [Memory and disk controls](https://leetucker.github.io/hdfe-stream/examples/out_of_core.html),
  a choice of solvers, and logging
- [Experimental GPU options](https://leetucker.github.io/hdfe-stream/benchmarks/gpu.html#using-the-gpu-options): the
  solve on an NVIDIA GPU (`device="cuda"`) and the Polars queries on
  cudf-polars (`polars_engine="gpu"`)
- [Simulated panels](https://leetucker.github.io/hdfe-stream/examples/index.html#the-simulated-data)
  with known effects, for testing

## When to use this, and when not to

**When your data do not fit in memory, use this.** It was built for that case:
a matched employer–employee panel — tens of millions of rows, millions of
worker effects, an AKM variance decomposition — on a machine or a secure
enclave where the data are larger than the memory you are allowed. Every other
library compared here needs the whole panel in memory before it starts; this
one holds a bounded slice of it at a time, and uses scratch disk for the rest.

**When they do fit, it is still a sensible default for medium and large
regressions.** From about 400,000 workers (3.4 million rows) up, it was the
fastest of the libraries benchmarked: against pyfixest, xhdfe, R's fixest and
Julia's FixedEffectModels.jl on the AKM regression and on wide designs, against
pyfixest on Poisson and logit, and against xhdfe on the KSS decomposition, and
at the largest sizes it used a fraction of their memory. At 5 million workers it fitted the AKM
regression in 22 seconds on a 16-vCPU machine, about as fast as xhdfe on an
NVIDIA H100 (24 seconds), without a GPU (see [should I use a GPU?](#should-i-use-a-gpu)). The
benchmarks are simulated employer–employee panels, a shape that suits its
design ([how it works](#how-it-works-in-one-paragraph)); one whose fixed-effect cells are no smaller than
the data may do less well.

**On small data it gains you nothing.** Below about 400,000 workers the
in-memory libraries were faster — by a second or less on the regression, and
by six seconds on the smallest KSS decomposition — and lighter: an hdfe_stream
process starts at about 0.8 GB.

**Use [pyfixest](https://github.com/py-econometrics/pyfixest) when you need
what a streaming design does not offer.** It is more mature and more widely
used, and has inference and clustering options this library lacks or limits:
CRV3 with fixed effects that are not nested in the clusters, separation checks
on the covariates and bias corrections for Poisson and logit, and more (see
[limitations](https://leetucker.github.io/hdfe-stream/index.html#limitations)). **Use [xhdfe](https://github.com/reisportela/xhdfe-xfe)
when GPU speed is the priority**: its CUDA backend made the most of the GPU in
the [GPU benchmark](https://leetucker.github.io/hdfe-stream/benchmarks/gpu.html).

This library is at its strongest when:

- the data do not fit in memory, or fit so tightly that nothing else does;
- you have one very high-cardinality dimension (workers, patients, students)
  whose groups each touch only a few levels of the others;
- you want **worker-specific slopes** as well as intercepts (`worker_id[t]`),
  which multiplies the parameter count by the number of slopes;
- the design is **wide**: hundreds of covariates, such as a categorical
  expanded into fine indicators or many interactions. An in-memory library
  needs rows × covariates in memory before it starts; this needs one batch of
  rows × covariates at a time, with any number of fixed effects, including one
  or none (see [many covariates](https://leetucker.github.io/hdfe-stream/benchmarks/covariates.html));
- you need a hard, predictable memory ceiling — a shared cluster, or a job that
  must not be the one that gets killed.

You are trading memory for disk, and you need scratch space: about five times
the size of your Parquet input while the fit runs, and only the result files
once it finishes. See [performance](#performance) for what that buys. Point
`workdir=` at it, or set the `HDFE_STREAM_WORKDIR` environment variable once
(for example to a cluster's scratch filesystem) and leave `workdir` out; a
`workdir` passed to a call takes precedence, and with neither, the system
temporary directory is used.

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
which is why a wide design fits where an in-memory one does not. Full detail is in
[`hdfe_stream/__init__.py`](https://github.com/leetucker/hdfe-stream/blob/main/hdfe_stream/__init__.py).

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
| `docs` | building the documentation (Sphinx) |

On Python 3.15, pyfixest 0.60 (for the `formula` extra) has no wheels yet. It
builds from source with a Rust toolchain and
`PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1` set; everything else installs from
wheels.

Without `formula` you still get `StreamingHDFE`, the lower-level interface that
takes column names or Polars expressions instead of a formula string.

The experimental GPU options need CuPy and cudf-polars for your CUDA version
(`pip install cupy-cuda13x cudf-polars-cu13`); there is no extra for them,
since the packages differ by CUDA version. See
[CPU and GPU](https://leetucker.github.io/hdfe-stream/benchmarks/gpu.html#using-the-gpu-options).

## Performance

The standard three-way AKM regression,
`log_earn ~ age_squared + age_cubed | worker_id + firm_id + year`, clustered by
worker, on a simulated panel of 5 million workers and 42.5 million rows,
against [pyfixest](https://github.com/py-econometrics/pyfixest),
[xhdfe](https://github.com/reisportela/xhdfe-xfe), R's
[fixest](https://lrberge.github.io/fixest/) and Julia's
[FixedEffectModels.jl](https://github.com/FixedEffects/FixedEffectModels.jl).
Every configuration agrees on the coefficients to within 2e-9.

| configuration | wall time | peak memory | peak disk |
|---|---:|---:|---:|
| pyfixest (MAP, its default) | 919 s | 20.8 GB | — |
| pyfixest (LSMR) | 74 s | 27.0 GB | — |
| xhdfe | 79 s | 19.3 GB | — |
| fixest (R) | 152 s | 9.1 GB | — |
| FixedEffectModels.jl (Julia) | 79 s | 9.5 GB | — |
| hdfe_stream (`explicit`, its default) | 22 s | 3.9 GB | 2.6 GB |
| hdfe_stream (`stream_cg`) | 22 s | 3.8 GB | 2.6 GB |
| **hdfe_stream (`stream_cg`, low memory)** | **24 s** | **3.7 GB** | 2.7 GB |

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/leetucker/hdfe-stream/main/docs/figures/akm_memory.dark.svg">
  <img alt="AKM regression: peak memory against the number of workers, one line per configuration, log-log" src="https://raw.githubusercontent.com/leetucker/hdfe-stream/main/docs/figures/akm_memory.light.svg">
</picture>

On small data the in-memory libraries use less memory: an hdfe_stream process
costs about 0.8 GB before it has done anything. The
[benchmarks](https://leetucker.github.io/hdfe-stream/benchmarks/index.html) cover every size from 212,000 rows up,
designs with up to 503 covariates, Poisson and logit, and the KSS
decomposition, with how each was measured.

### Should I use a GPU?

Probably not for this. hdfe_stream's GPU options are experimental, and on an
NVIDIA H100 their gains were modest throughout and appeared only at large
sizes: about 4% at 5 million workers with both options on, and no consistent
gain below that. A GPU-oriented library such as
[xhdfe](https://github.com/reisportela/xhdfe-xfe) is the better choice if GPU
performance is the priority: its CUDA backend took 24 s at 5 million workers,
against 107 s on its CPU backend. But hdfe_stream on a 16-vCPU CPU machine took
22 s, and that machine costs about an eighth as much per hour as AWS's
smallest H100 machine. [CPU and GPU](https://leetucker.github.io/hdfe-stream/benchmarks/gpu.html) has the numbers.

## Documentation

- [**Examples**](https://leetucker.github.io/hdfe-stream/examples/index.html): one page per feature, on simulated
  data that ships with the package: formulas and standard errors, weights and
  IV, Poisson and logit, wide designs, varying slopes, connected sets, running
  out of core, and reporting.
- [**Leave-out variance components**](https://leetucker.github.io/hdfe-stream/kss/index.html): the
  Kline–Saggio–Sølvsten bias correction for the AKM decomposition, its standard
  errors and weak-identification diagnostic, and every way it differs from the
  reference implementations.
- [**Benchmarks**](https://leetucker.github.io/hdfe-stream/benchmarks/index.html): time, memory and disk against
  pyfixest, xhdfe, fixest and FixedEffectModels.jl.
- [**Function reference**](https://leetucker.github.io/hdfe-stream/reference/index.html): every option.
- [**Limitations**](https://leetucker.github.io/hdfe-stream/index.html#limitations): what it does not do.
- [**Changelog**](https://github.com/leetucker/hdfe-stream/blob/main/CHANGELOG.md).

The scripts behind the examples are in [`examples/`](https://github.com/leetucker/hdfe-stream/tree/main/examples), and the
benchmarks in [`benchmarks/`](https://github.com/leetucker/hdfe-stream/tree/main/benchmarks).

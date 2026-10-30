# Benchmarks

Eight benchmarks compare hdfe_stream with in-memory libraries. Five use the
simulated AKM panel that ships with the package. Three vary the height of the data, at
five sizes from 25,000 to 5,000,000 workers (212,000 to 42.5 million rows); the
number of firms is always workers / 15, so the number of workers is the only
thing that changes along their x axis. The other two hold the data at 1,000,000
workers (8.5 million rows) and vary its width: the number of covariates, from 8
to 503. A sixth, on a machine of its own, measures hdfe_stream's experimental
GPU options and the other libraries' GPU paths. A seventh and an eighth leave
fixed effects behind: [OLS, IV, logit and probit](regression.md), with zero or
one fixed effect, on simulated rows from 100,000 to 20 million, against
scikit-learn, polars-ols, polars-ds, statsmodels, linearmodels, glum, pyfixest
and R; and the same models on a GPU.

| benchmark | job | compared with |
|---|---|---|
| [AKM regression](akm.md) | `log_earn ~ age_squared + age_cubed \| worker_id + firm_id + year`, clustered by worker | pyfixest (MAP and LSMR demeaners), xhdfe, fixest (R), FixedEffectModels.jl (Julia) |
| [Many covariates](covariates.md) | `log_earn ~ i(age_bin) \| worker_id + firm_id`, clustered by worker, with age bins from five years to one month wide | pyfixest (both demeaners), xhdfe, fixest (R), FixedEffectModels.jl (Julia) |
| [Poisson and logit](glm.md) | a count and a binary outcome on `age_squared + age_cubed \| worker_id + firm_id + year`, clustered by worker | pyfixest's `fepois` and `feglm` (LSMR demeaner) |
| [Poisson and logit, many covariates](glm_covariates.md) | the same two outcomes on `i(age_bin) \| worker_id + firm_id`, clustered by worker | pyfixest's `fepois` and `feglm` (LSMR demeaner) |
| [KSS](kss.md) | the leave-out variance decomposition of `log_earn ~ age_squared + age_cubed \| worker_id + firm_id`, with and without standard errors | xhdfe |
| [OLS, IV, logit and probit](regression.md) | OLS, 2SLS, logit and probit on 10 covariates, with no fixed effects and with one, on simulated rows (100,000 to 20 million) | scikit-learn, polars-ols, polars-ds, statsmodels, linearmodels, glum, pyfixest, fixest and `lm`/`glm` (R), NumPy |
| [CPU and GPU](gpu.md) | the AKM regression on a machine with an NVIDIA H100, with hdfe_stream's experimental GPU options | pyfixest (LSMR, on PyTorch on the GPU), xhdfe (CUDA backend), each also on the CPU |
| [CPU and GPU, plain regressions](gpu.md#plain-regressions-ols-iv-and-logit) | the OLS, 2SLS and logit models above, with no fixed effects, on the same H100 machine | cuML, CuPy, polars-ols, polars-ds |

In brief: hdfe_stream uses a fraction of the memory at scale, and is not slower
for it. At 5 million workers the AKM regression took 22 s and 3.9 GB, against
74 s and 27.0 GB for pyfixest (LSMR), 79 s and 19.3 GB for xhdfe, 79 s and
9.5 GB for FixedEffectModels.jl, and 152 s and 9.1 GB for fixest. On
small data the in-memory libraries use less memory, since an hdfe_stream process
costs about 0.8 GB before it has done anything.

On plain regressions, without the fixed-effects structure it was built for, it
is not the fastest, but not far behind and far lighter: at 20 million rows OLS
took 6.5 s and 2.5 GB, against 5.4 s and 7.5 GB for scikit-learn, 8.5 s and
5.7 GB for polars-ols, 9.1 s and 15.7 GB for pyfixest, and 2.9 s and 7.8 GB for
fixest. A logit took 9.4 s and 2.4 GB, against 12.8 s and 7.7 GB for
statsmodels, 15.9 s and 20.6 GB for pyfixest, and 4.2 s and 5.5 GB for glum.

## How it is measured

Every configuration runs in its own subprocess, since peak memory is a
per-process high-water mark. The measurements are:

- **Wall time:** end to end, including reading the data. The in-memory
  libraries need it in a data frame first (pandas, R's data.frame read with
  arrow, or Julia's DataFrame read with Parquet2.jl); hdfe_stream reads the
  Parquet file itself, in batches. That difference is the comparison, not an artifact of
  how it is measured.
- **Peak memory:** the process's `VmHWM`, which includes the roughly 0.5 GB
  that importing the libraries costs.
- **Peak disk:** the allocated size of the configuration's own work directory,
  sampled every 0.2 s. For hdfe_stream it is also taken from the fit's own
  bookkeeping, whichever is larger.

A numba warm-up run comes first, so kernel compilation is not charged to any
configuration. Configurations that run past three hours are recorded as timed
out, and processes that reach 96% of the machine's memory, which are killed, as out of
memory. Both appear in the results, and in a note under the figure.

Before reporting a size, the regression and GLM benchmarks check that every
configuration's coefficients agree. The KSS benchmark reports how far
hdfe_stream's and xhdfe's estimates are apart; that gap is random-projection
noise, well under 1% at every size measured.

fixest and FixedEffectModels.jl run as R and Julia processes, measured the
same way. Both use every logical CPU, as the Python libraries do (fixest's
default is half of them, and Julia's one thread); FixedEffectModels.jl is given
a tolerance of 1e-8, since at its default of 1e-6 its estimates fall short of
the others' accuracy; and singletons are kept or dropped as pyfixest does in
each benchmark. Julia compiles on first use, so FixedEffectModels.jl fits a
small panel first, untimed, as hdfe_stream's numba kernels are compiled before
it is timed.

xhdfe is compared on its CPU backend, which is its default; no GPU was
involved, except in the [CPU and GPU](gpu.md) benchmark, which has its own
machine. The GLM benchmarks compare with pyfixest alone, since xhdfe fits
linear models only. The [regression benchmarks](regression.md) also say what
each library returns: some give coefficients only, and some cannot absorb a
fixed effect or fit 2SLS, so that their row includes the demeaning or the two
stages. They time out at 30 minutes, not three hours. pytwoway was also considered for KSS, but version 0.3.21
needs numpy < 2 and an older scipy, so it cannot run alongside hdfe_stream's
dependencies and is not included.

## The machine

Measured on an AWS EC2 `c8id.4xlarge` instance: an Intel Xeon 6975P-C with 16
vCPUs and 30.8 GB of RAM, Linux, and the instance's local NVMe drive for the
data and work directories, so that disk traffic is not network traffic. Python
3.14.8, polars 2.0.0, numpy 2.5.3, numba 0.68.0, scipy 1.18.1, pyfixest 0.60.0,
xhdfe 2.28.1 (CPU backend) and hdfe_stream 0.6.1; R 4.5.3 with fixest 0.14.2,
and Julia 1.13.2 with FixedEffectModels.jl 2.1.0. The other libraries'
rows were measured on 2026-10-16 and 17, all five benchmarks one after the
other as a single job on a spot instance (fixest and FixedEffectModels.jl
followed on 2026-10-19, in a second job). hdfe_stream's were measured again on
2026-10-29, on the release 0.6.1, on the same instance type set up the same
way, with the same Python environment, in a third job that ran only
hdfe_stream's configurations.
The [regression benchmarks](regression.md) were measured on 2026-10-30 on the
same instance type, in one job on a spot instance, with
hdfe_stream 0.7.0, scikit-learn 1.9.1, polars-ols 0.3.5, polars-ds 0.13.0,
statsmodels 0.15.0, linearmodels 7.1 and glum 3.4.1, and the rest as above; their
versions are in
[benchmarks/results/general_machine.json](https://github.com/leetucker/hdfe-stream/blob/main/benchmarks/results/general_machine.json).
The benchmarks run in a service limited to 96% of the machine's memory, with no
swap; a process that reaches the limit is killed, and one that asks for a
single block larger than the machine can back is refused it, so "out of memory"
in the results means needing more than about 29.6 GB. AWS reclaimed the spot instance
several times; a step that is interrupted starts again from the beginning, so
every row is from one uninterrupted run. The exact versions are in
[benchmarks/results/machine.json](https://github.com/leetucker/hdfe-stream/blob/main/benchmarks/results/machine.json).

## Reproducing them

The scripts are in
[`benchmarks/`](https://github.com/leetucker/hdfe-stream/tree/main/benchmarks)
of the repository, and the numbers behind every figure are in
[`benchmarks/results/`](https://github.com/leetucker/hdfe-stream/tree/main/benchmarks/results),
one CSV per benchmark;
[benchmarks/README.md](https://github.com/leetucker/hdfe-stream/blob/main/benchmarks/README.md)
describes their columns.

```bash
pip install -e ".[benchmark]"
python benchmarks/akm_benchmark.py          # all sizes; or list sizes: 25000 100000
python benchmarks/kss_benchmark.py
python benchmarks/covariates_benchmark.py   # or list bin widths in months: 12 3
python benchmarks/glm_benchmark.py          # sizes, as for akm_benchmark.py
python benchmarks/glm_covariates_benchmark.py
python benchmarks/ols_benchmark.py          # OLS, with 0 and 1 fixed effects; sizes are rows
python benchmarks/iv_benchmark.py
python benchmarks/binary_benchmark.py       # logit and probit
python benchmarks/general_gpu_benchmark.py  # with an NVIDIA GPU, and cuML in its own environment
python benchmarks/plot_benchmarks.py        # redraw docs/figures/ from the CSVs
```

xhdfe is not on PyPI in a form that can be declared as a dependency. It builds
from source, and needs CMake on the path while it does:

```bash
pip install cmake ninja
pip install "git+https://github.com/reisportela/xhdfe-xfe.git"
```

The figures used commit `f71420b` (2.28.1).

## Correctness

The test suite checks every feature against pyfixest on simulated data:
coefficients, standard errors under each vcov, residuals, fixed effects, fit
statistics, the sample kept after dropping missing values, and the exact set of
coefficient names and dropped collinear terms. Varying slopes have no pyfixest
equivalent, so they are checked against a brute-force regression with explicit
worker-by-slope dummies; three-way clustering is checked against an
inclusion–exclusion sum built from pyfixest's own score matrix. Poisson, logit
and probit are checked against pyfixest's `fepois` and `feglm`, and their
frequency weights against pyfixest on the data with each row repeated. The KSS
correction is validated separately, against known truth, a dense prototype,
pytwoway and xhdfe ([how it is checked](../kss/index.md#how-it-is-checked)).

```bash
pip install -e ".[test]"
pytest
```

```{toctree}
:hidden:

akm
covariates
glm
glm_covariates
kss
regression
gpu
```

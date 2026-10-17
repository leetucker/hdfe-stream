# Benchmarks

Five benchmarks compare hdfe_stream with in-memory libraries on the simulated
AKM panel that ships with the package. Three vary the height of the data, at
five sizes from 25,000 to 5,000,000 workers (212,000 to 42.5 million rows); the
number of firms is always workers / 15, so the number of workers is the only
thing that changes along their x axis. The other two hold the data at 1,000,000
workers (8.5 million rows) and vary its width: the number of covariates, from 8
to 503.

| benchmark | job | compared with |
|---|---|---|
| [AKM regression](akm.md) | `log_earn ~ age_squared + age_cubed \| worker_id + firm_id + year`, clustered by worker | pyfixest (MAP and LSMR demeaners), xhdfe |
| [Many covariates](covariates.md) | `log_earn ~ i(age_bin) \| worker_id + firm_id`, clustered by worker, with age bins from five years to one month wide | pyfixest (both demeaners), xhdfe |
| [Poisson and logit](glm.md) | a count and a binary outcome on `age_squared + age_cubed \| worker_id + firm_id + year`, clustered by worker | pyfixest's `fepois` and `feglm` (LSMR demeaner) |
| [Poisson and logit, many covariates](glm_covariates.md) | the same two outcomes on `i(age_bin) \| worker_id + firm_id`, clustered by worker | pyfixest's `fepois` and `feglm` (LSMR demeaner) |
| [KSS](kss.md) | the leave-out variance decomposition of `log_earn ~ age_squared + age_cubed \| worker_id + firm_id`, with and without standard errors | xhdfe |

In brief: hdfe_stream uses a fraction of the memory at scale, and is not slower
for it. At 5 million workers the AKM regression took 22 s and 3.9 GB, against
74 s and 27.0 GB for pyfixest (LSMR) and 79 s and 19.3 GB for xhdfe. On
small data the in-memory libraries use less memory, since an hdfe_stream process
costs about 0.8 GB before it has done anything.

## How it is measured

Every configuration runs in its own subprocess, since peak memory is a
per-process high-water mark. The measurements are:

- **Wall time:** end to end, including reading the data. The in-memory
  libraries need it in a DataFrame first; hdfe_stream reads the Parquet file
  itself, in batches. That difference is the comparison, not an artifact of
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

xhdfe is compared on its CPU backend, which is its default; no GPU was
involved. The GLM benchmarks compare with pyfixest alone, since xhdfe fits
linear models only. pytwoway was also considered for KSS, but version 0.3.21
needs numpy < 2 and an older scipy, so it cannot run alongside hdfe_stream's
dependencies and is not included.

## The machine

Measured on an AWS EC2 `c8id.4xlarge` instance: an Intel Xeon 6975P-C with 16
vCPUs and 30.8 GB of RAM, Linux, and the instance's local NVMe drive for the
data and work directories, so that disk traffic is not network traffic. Python
3.14.8, polars 2.0.0, numpy 2.5.3, numba 0.68.0, scipy 1.18.1, pyfixest 0.60.0,
xhdfe 2.28.1 (CPU backend) and hdfe_stream 0.4.0. All five benchmarks ran one
after the other on 2026-10-16 and 17, as a single job on a spot instance.
The benchmarks run in a service limited to 96% of the machine's memory, with no
swap; a process that reaches the limit is killed, so "out of memory" in the
results means needing more than about 29.6 GB. AWS reclaimed the spot instance
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
```

# Benchmarks

Five benchmarks compare hdfe_stream with in-memory libraries on the simulated
AKM panel that ships with the package (three more, on simulated rows with no
panel structure, are below the table). Three vary the height of the data, at
five sizes from 25,000 to 5,000,000 workers; the number of firms is always
workers / 15, so the number of workers is the only thing that changes along
their x axis. The other two hold the data at 1,000,000 workers (8.5 million
rows) and vary its width: the number of covariates, from 8 to 503.

| script | job | compared with |
|---|---|---|
| `akm_benchmark.py` | `log_earn ~ age_squared + age_cubed \| worker_id + firm_id + year`, clustered by worker | pyfixest (MAP and LSMR demeaners), xhdfe, fixest (R), FixedEffectModels.jl (Julia) |
| `kss_benchmark.py` | the KSS leave-out variance decomposition of `log_earn ~ age_squared + age_cubed \| worker_id + firm_id`, with and without standard errors | xhdfe (its standard errors up to 1,000,000 workers) |
| `covariates_benchmark.py` | `log_earn ~ i(age_bin) \| worker_id + firm_id`, clustered by worker, with age bins from five years to one month wide | pyfixest (both demeaners), xhdfe, fixest (R), FixedEffectModels.jl (Julia) |
| `glm_benchmark.py` | Poisson and logit regressions of a count and a binary outcome on `age_squared + age_cubed \| worker_id + firm_id + year`, clustered by worker | pyfixest's `fepois` and `feglm` (LSMR demeaner) |
| `glm_covariates_benchmark.py` | the same two outcomes on `i(age_bin) \| worker_id + firm_id`, clustered by worker, with the covariates benchmark's age bins | pyfixest's `fepois` and `feglm` (LSMR demeaner) |

The OLS, IV, logit and probit benchmarks leave fixed effects behind. They run
on `_harness.general_data_path`, n rows with 10 covariates, one group variable
(n / 20 groups) as the fixed effect, an endogenous regressor with two
instruments, and continuous and binary outcomes; their x axis is the number of
rows, from 100,000 to 20 million.

| script | job | compared with |
|---|---|---|
| `ols_benchmark.py` | OLS with 10 covariates, with no fixed effects and with one | scikit-learn, polars-ols, polars-ds, statsmodels, pyfixest, linearmodels, fixest and `lm` (R), NumPy |
| `iv_benchmark.py` | 2SLS, one endogenous regressor and two instruments, with no fixed effects and with one | linearmodels, pyfixest, fixest (R), polars-ols (two stages by hand), NumPy |
| `binary_benchmark.py` | logit and probit, no fixed effects | statsmodels, pyfixest, fixest and `glm` (R), scikit-learn, glum, polars-ds (logit only for the last three) |

Each library is asked for the classical standard errors where it offers them.
scikit-learn, glum and polars-ds's `logistic_reg` return coefficients only, and
a library with no absorbing option or no 2SLS is given the data demeaned in
Polars or pandas, or runs the two stages by hand, and that is in its time.
`_regression.py` holds the runners they share. Results go to `results/ols.csv`,
`results/iv.csv` and `results/binary.csv`, keyed on (`n_rows`, configuration);
their `task`, `fixed_effects` and `se_*` columns say what was fitted and which
standard errors were reported, and `results/general_machine.json` records the
versions. Trial runs can be kept out of `results/` with the
`HDFE_BENCH_RESULTS` environment variable (a directory).

```bash
pip install -e ".[benchmark]"
python benchmarks/akm_benchmark.py          # all sizes; or list sizes: 25000 100000
python benchmarks/kss_benchmark.py
python benchmarks/covariates_benchmark.py   # or list bin widths in months: 12 3
python benchmarks/glm_benchmark.py          # sizes, as for akm_benchmark.py
python benchmarks/glm_covariates_benchmark.py
python benchmarks/ols_benchmark.py          # sizes are rows: 100000 1000000
python benchmarks/iv_benchmark.py
python benchmarks/binary_benchmark.py
python benchmarks/plot_benchmarks.py        # redraw docs/figures/ from the CSVs
```

xhdfe is not on PyPI in a form that can be declared as a dependency. It builds
from source, and needs CMake on the path while it does:

```bash
pip install cmake ninja
pip install "git+https://github.com/reisportela/xhdfe-xfe.git"
```

The figures here used commit `f71420b` (2.28.1), on its CPU backend, which is
its default; no GPU was involved.

`gpu_benchmark.py` is separate and experimental: the AKM regression above,
on one machine with an NVIDIA GPU, for hdfe_stream on the CPU, with its solve
on the GPU (`device="cuda"`), and with its Polars queries on the GPU as well
(`polars_engine="gpu"`), against pyfixest's LSMR demeaner on its default
backend and on PyTorch on the GPU (`LsmrDemeaner(backend="torch",
device="cuda")`), and xhdfe on the CPU and on its CUDA backend. It needs CuPy
and cudf-polars for the hdfe_stream configurations, a CUDA build of PyTorch
for pyfixest's GPU one, and a CUDA build of xhdfe for its GPU one:

```bash
XHDFE_ENABLE_CUDA=ON CMAKE_CUDA_ARCHITECTURES=90 \
    pip install --no-deps --reinstall "git+https://github.com/reisportela/xhdfe-xfe.git@f71420b"
```

(90 for an H100: the GPU's compute capability without the dot). With the CUDA
13 toolkit, xhdfe 2.28.1 needs one change to build: `cub::Max()` in
`src/fe_absorption_cuda.cu`, which CUDA 13 removed, becomes
`cuda::maximum<>{}`, the same operation. Its build also checks the compiled
GPU code with `cuobjdump` and `nvdisasm`, which some CUDA installs leave out
(`pip install nvidia-cuda-cuobjdump nvidia-cuda-nvdisasm` at the toolkit's
version provides them, with `CUOBJDUMP` pointing at the first). The published
GPU results used that change, as `results/gpu_machine.json` records.

`general_gpu_benchmark.py` does the same for OLS, 2SLS and logit with no fixed
effects, on the data of the benchmarks above: hdfe_stream on the CPU, with
`polars_engine="gpu"`, and for logit with `device="cuda"`; the fastest CPU
library for each model; cuML's `LinearRegression` and `LogisticRegression`;
and the normal equations, 2SLS and Newton's method in CuPy. cuML runs in a
virtual environment of its own (`HDFE_BENCH_CUML_PYTHON`, or
`/opt/bench/cuml-venv/bin/python`), with NumPy below 2.4, which dropped
`row_stack`, still used by cuML 26.06; its configurations are skipped without
it. Results go to `results/general_gpu.csv` and
`results/general_gpu_machine.json`.

On a machine without a GPU the benchmark says so and measures nothing.
Results go to `results/gpu.csv`, with the GPU and package versions in
`results/gpu_machine.json`, and are drawn as `docs/figures/gpu_*.svg`.

The GLM benchmarks compare with pyfixest alone, since xhdfe fits linear models
only, and with its LSMR demeaner alone: its default MAP demeaner stops with
"Demeaning failed after 10000 iterations" on these panels at every size tried,
for both models. The count and binary outcomes are simulated from the same
worker effects, firm effects and age profile as `log_earn`
(`_harness.glm_data_path`).

pytwoway was also considered. Version 0.3.21 needs numpy < 2 (it uses
`np.bool8`) and a scipy old enough to accept `minres(tol=)`, so it cannot run
alongside hdfe_stream's dependencies and is not included.

### R and Julia

The regression and covariates benchmarks also compare with R's
[fixest](https://lrberge.github.io/fixest/) and Julia's
[FixedEffectModels.jl](https://github.com/FixedEffects/FixedEffectModels.jl).
Neither language is a dependency of anything else here: where one is not
installed, its configurations print "skipped" and nothing is written for them.
The versions are pinned in [`external/`](external/), R's in a conda
environment file and Julia's in a manifest:

```bash
# R 4.5.3 and fixest 0.14.2, from conda-forge (micromamba, mamba or conda)
micromamba create -p benchmarks/external/.r -f benchmarks/external/environment.yml

# Julia 1.13.2 (e.g. `juliaup add 1.13.2`), then FixedEffectModels.jl 2.1.0
julia +1.13.2 --project=benchmarks/external/julia -e 'using Pkg; Pkg.instantiate()'
```

The benchmarks find `Rscript` in `benchmarks/external/.r`, or else on the
path, and `julia` on the path; `HDFE_BENCH_RSCRIPT` and `HDFE_BENCH_JULIA`
point them anywhere else (with juliaup, a `julia` launcher for 1.13.2). Each
fit runs as one R or Julia process, from `external/fixest.R` and
`external/fixedeffectmodels.jl`, which measure themselves the way the Python
child processes do.

Four settings differ from those libraries' defaults, so that each does the
same job as the others:

- Both use every logical CPU, as the Python libraries do; fixest's own default
  is half of them, and Julia's is one thread.
- FixedEffectModels.jl is given a tolerance of 1e-8. At its default, 1e-6, its
  coefficients on the covariates benchmark were up to 1e-4 (relative) from the
  others', which agree with each other to about 1e-8; at 1e-8 it agrees as
  closely, for about 1.5 times its default time.
- Singletons are treated as pyfixest treats them: kept in the regression
  benchmark (FixedEffectModels.jl drops them by default) and dropped in the
  covariates benchmark (fixest keeps them by default). The panels have almost
  none, and dropping them does not change the estimates.
- Julia compiles each function on first use, so FixedEffectModels.jl reads and
  fits a small panel first, untimed, in the same process: the times are for
  compiled code, as they are for numba in hdfe_stream. The compiled code's
  memory stays in the peak.

## How it is measured

Every configuration runs in its own subprocess, since peak memory is a
per-process high-water mark. The measurements are:

- **Wall time:** end to end, including reading the data. The in-memory
  libraries need it in a DataFrame first (pandas for pyfixest and xhdfe, a
  data.frame read with arrow for fixest, a DataFrame read with Parquet2.jl for
  FixedEffectModels.jl); hdfe_stream reads the Parquet file itself, in
  batches. That difference is the comparison, not an artifact of how it is
  measured.
- **Peak memory:** the process's `VmHWM`, which includes the roughly 0.5 GB
  that importing the libraries costs.
- **Peak disk:** the allocated size of the configuration's own work directory,
  sampled every 0.2 s. For hdfe_stream it is also taken from the fit's own
  bookkeeping, whichever is larger.

A numba warm-up run comes first, so kernel compilation is not charged to any
configuration. Configurations that run past three hours are recorded as timed
out, and processes that run out of memory as out of memory: those the kernel
kills, and those refused a single allocation larger than the machine can back
(R's "cannot allocate vector", Julia's `OutOfMemoryError`), which stop before
the kernel would kill them. Both appear in the CSV, and in a note under the
figure.

Before reporting a size, the regression and GLM benchmarks check that every
configuration's coefficients agree. The KSS benchmark reports how far
hdfe_stream's and xhdfe's estimates are apart; that gap is random-projection
noise, well under 1% at every size measured.

## Results files

`results/akm.csv`, `results/kss.csv`, `results/glm.csv` and `results/gpu.csv`
hold one row per (size, configuration), and `results/covariates.csv` and
`results/glm_covariates.csv` one per (number of covariates, configuration).
Re-running a size replaces that size's rows. `results/machine.json` records the
hardware and package versions (R's and Julia's too, where installed). The
columns are:

| column | meaning |
|---|---|
| `n_workers`, `n_firms`, `n_rows` | the panel |
| `configuration` | library and setting, as labeled in the figures |
| `status` | `ok`, or why there is no measurement |
| `wall_s`, `read_s`, `fit_s` | seconds: total, reading the data, estimation |
| `peak_memory_mb`, `import_memory_mb` | peak resident memory, and the part present before any work began |
| `peak_disk_mb` | peak disk written |
| `coef_*` (regression) | the estimates, used for the agreement check |
| `kss_*`, `se_*` (KSS) | the leave-out components and their standard errors |
| `n_obs_pruned` (KSS) | observations left after pruning to the leave-one-out connected set |
| `family`, `n_obs`, `irls_steps` (GLM) | `poisson` or `logit`; observations after dropping separated fixed-effect levels; IRLS steps, for hdfe_stream |
| `n_covariates`, `bin_months`, `n_coef` (covariates) | indicators in the design, the age-bin width, and coefficients reported: the same number, except that xhdfe also reports an intercept alongside the absorbed effects, as reghdfe does |
| `solve_s`, `solver`, `device` (GPU) | for hdfe_stream: seconds in the solve, the solver used, and where it ran |
| `peak_gpu_memory_mb` (GPU) | the process's peak GPU memory, polled from `nvidia-smi` |
| `polars_cpu_fallbacks`, `polars_cpu_fallback_reasons` (GPU) | with `polars_engine="gpu"`: the queries the GPU engine could not run, which ran on the CPU, and why |
| `notes` (GPU) | warnings a library printed while fitting |

`plot_benchmarks.py` reads only these files, so the figures can be restyled
without re-running anything.

`akm_benchmark.py --memory-sweep N_WORKERS` is a separate measurement: how
hdfe_stream's peak memory responds to `rows_per_bucket` and `batch_rows` on one
panel. Its table is in the documentation, under
[AKM regression](https://leetucker.github.io/hdfe-stream/benchmarks/akm.html#trading-memory-for-time).

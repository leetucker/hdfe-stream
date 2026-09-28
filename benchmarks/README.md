# Benchmarks

Five benchmarks compare hdfe_stream with in-memory libraries on the simulated
AKM panel that ships with the package. Three vary the height of the data, at
five sizes from 25,000 to 5,000,000 workers; the number of firms is always
workers / 15, so the number of workers is the only thing that changes along
their x axis. The other two hold the data at 1,000,000 workers (8.5 million
rows) and vary its width: the number of covariates, from 8 to 503.

| script | job | compared with |
|---|---|---|
| `akm_benchmark.py` | `log_earn ~ age_squared + age_cubed \| worker_id + firm_id + year`, clustered by worker | pyfixest (MAP and LSMR demeaners), xhdfe |
| `kss_benchmark.py` | the KSS leave-out variance decomposition of `log_earn ~ age_squared + age_cubed \| worker_id + firm_id`, with and without standard errors | xhdfe |
| `covariates_benchmark.py` | `log_earn ~ i(age_bin) \| worker_id + firm_id`, clustered by worker, with age bins from five years to one month wide | pyfixest (both demeaners), xhdfe |
| `glm_benchmark.py` | Poisson and logit regressions of a count and a binary outcome on `age_squared + age_cubed \| worker_id + firm_id + year`, clustered by worker | pyfixest's `fepois` and `feglm` (LSMR demeaner) |
| `glm_covariates_benchmark.py` | the same two outcomes on `i(age_bin) \| worker_id + firm_id`, clustered by worker, with the covariates benchmark's age bins | pyfixest's `fepois` and `feglm` (LSMR demeaner) |

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

The figures here used commit `e0c2362` (2.28.0), on its CPU backend, which is
its default; no GPU was involved.

The GLM benchmarks compare with pyfixest alone, since xhdfe fits linear models
only, and with its LSMR demeaner alone: its default MAP demeaner stops with
"Demeaning failed after 10000 iterations" on these panels at every size tried,
for both models. The count and binary outcomes are simulated from the same
worker effects, firm effects and age profile as `log_earn`
(`_harness.glm_data_path`).

pytwoway was also considered. Version 0.3.21 needs numpy < 2 (it uses
`np.bool8`) and a scipy old enough to accept `minres(tol=)`, so it cannot run
alongside hdfe_stream's dependencies and is not included.

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
out, and processes the kernel kills for lack of memory as out of memory. Both
appear in the CSV, and in a note under the figure.

Before reporting a size, the regression and GLM benchmarks check that every
configuration's coefficients agree. The KSS benchmark reports how far
hdfe_stream's and xhdfe's estimates are apart; that gap is random-projection
noise, well under 1% at every size measured.

## Results files

`results/akm.csv`, `results/kss.csv` and `results/glm.csv` hold one row per
(size, configuration), and `results/covariates.csv` and
`results/glm_covariates.csv` one per (number of covariates, configuration).
Re-running a size replaces that size's rows. `results/machine.json` records the
hardware and package versions. The columns are:

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

`plot_benchmarks.py` reads only these files, so the figures can be restyled
without re-running anything.

`akm_benchmark.py --memory-sweep N_WORKERS` is a separate measurement: how
hdfe_stream's peak memory responds to `rows_per_bucket` and `batch_rows` on one
panel. Its table is in the main README.

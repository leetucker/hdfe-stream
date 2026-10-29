# Poisson and logit

`fepois_stream` and `feglm_stream` against pyfixest's `fepois` and `feglm`,
on the simulated AKM panel of the [AKM regression benchmark](akm.md), with two
outcomes simulated from the same worker effects, firm effects and age profile
as its `log_earn`: a count (Poisson) and a binary outcome (logit). The model is
`y ~ age_squared + age_cubed | worker_id + firm_id + year`, clustered by worker,
from 25,000 to 5,000,000 workers (212,000 to 42.5 million rows), with firms =
workers / 15. [Poisson and logit, many covariates](glm_covariates.md) holds the
data fixed and varies the width instead.

pyfixest runs with its LSMR demeaner. Its default, MAP, stops with "Demeaning
failed after 10000 iterations" on these panels at every size tried, from 25,000
workers up, for both models. xhdfe fits linear models only. hdfe_stream runs at
its defaults, with `solver="explicit"`, and with the OLS benchmark's low-memory
setting (`rows_per_bucket=250_000`, `batch_rows=200_000`).

Wherever both libraries finished, the coefficients agree to within 1e-7, the
tolerance the iterations reach. For logit, pyfixest keeps workers whose outcome
is 1 in every year, which this package drops (see [differences from
pyfixest](../examples/glm.md#differences-from-pyfixest)), so its N is larger:
8,404,384 against 8,283,651 at 1,000,000 workers. Each configuration runs in
its own process, and wall time includes reading the data ([how it is
measured](index.md#how-it-is-measured)). The numbers are in
[benchmarks/results/glm.csv](https://github.com/leetucker/hdfe-stream/blob/main/benchmarks/results/glm.csv).

```bash
python benchmarks/glm_benchmark.py
python benchmarks/plot_benchmarks.py
```

```{image} ../figures/glm_time.light.svg
:class: only-light benchmark
:alt: Poisson and logit regression: wall time against the number of workers, one panel per model, one line per configuration, log-log
```
```{image} ../figures/glm_time.dark.svg
:class: only-dark benchmark
:alt: Poisson and logit regression: wall time against the number of workers, one panel per model, one line per configuration, log-log
```

```{image} ../figures/glm_memory.light.svg
:class: only-light benchmark
:alt: Poisson and logit regression: peak memory against the number of workers, one panel per model, one line per configuration, log-log
```
```{image} ../figures/glm_memory.dark.svg
:class: only-dark benchmark
:alt: Poisson and logit regression: peak memory against the number of workers, one panel per model, one line per configuration, log-log
```

```{image} ../figures/glm_disk.light.svg
:class: only-light benchmark
:alt: Poisson and logit regression: peak disk use of hdfe_stream against the number of workers, one panel per model, log-log
```
```{image} ../figures/glm_disk.dark.svg
:class: only-dark benchmark
:alt: Poisson and logit regression: peak disk use of hdfe_stream against the number of workers, one panel per model, log-log
```

Wall time and peak memory, hdfe_stream at its defaults:

| workers | rows | Poisson: `fepois_stream` | Poisson: pyfixest | logit: `feglm_stream` | logit: pyfixest |
|---:|---:|---:|---:|---:|---:|
| 25,000 | 212,000 | 1.6 s, 0.8 GB | 2.3 s, 0.6 GB | 1.6 s, 0.9 GB | 2.3 s, 0.6 GB |
| 100,000 | 850,000 | 3.1 s, 1.1 GB | 9.3 s, 1.2 GB | 3.0 s, 1.1 GB | 8.8 s, 1.2 GB |
| 400,000 | 3.4 million | 9.3 s, 1.7 GB | 40 s, 3.8 GB | 9.1 s, 1.7 GB | 40 s, 3.8 GB |
| 1,000,000 | 8.5 million | 22 s, 2.2 GB | 126 s, 8.0 GB | 22 s, 2.1 GB | 111 s, 8.2 GB |
| 5,000,000 | 42.5 million | 145 s, 4.0 GB | out of memory | 137 s, 4.0 GB | out of memory |

- **Time.** From 100,000 workers up, hdfe_stream is 3 to 6 times faster.
- **Memory.** At 5,000,000 workers pyfixest was killed at this machine's
  limit of 29.6 GB (of 30.8 GB), for both models. hdfe_stream peaked at 4.0
  GB, or 3.7 GB with the low-memory setting, in 2 to 6% more time. At
  25,000 workers pyfixest uses less, as with OLS: an hdfe_stream process costs
  about 0.8 GB before it has done anything.
- **Disk.** About 67 bytes per row: 2.8 GB at 42.5 million rows, a little
  more than OLS on the same design writes (2.6 GB).
- **Against OLS.** `feols_stream` fits the same design in 4.9 s at 1,000,000
  workers and 22 s at 5,000,000, so a Poisson or logit fit costs 4 to 7 OLS
  fits.

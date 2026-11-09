# Poisson and logit

`fepois_stream` and `feglm_stream` against pyfixest's `fepois` and `feglm`,
on the simulated AKM panel of the [AKM regression benchmark](akm.md), with two
outcomes simulated from the same worker effects, firm effects and age profile
as its `log_earn`: a count (Poisson) and a binary outcome (logit). The model is
`y ~ age_squared + age_cubed | worker_id + firm_id + year`, clustered by worker,
from 25,000 to 5,000,000 workers (212,000 to 42.5 million rows), with firms =
workers / 15. [Poisson and logit, many covariates](glm_covariates.md) holds the
data fixed and varies the width instead.

pyfixest runs with its default demeaner, LSMR. xhdfe fits linear models only.
hdfe_stream runs at
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
| 25,000 | 212,000 | 1.4 s, 0.8 GB | 1.4 s, 0.6 GB | 1.5 s, 0.8 GB | 1.3 s, 0.6 GB |
| 100,000 | 850,000 | 2.9 s, 1.1 GB | 4.4 s, 1.4 GB | 2.9 s, 1.1 GB | 4.9 s, 1.3 GB |
| 400,000 | 3.4 million | 9.4 s, 1.7 GB | 18 s, 3.7 GB | 9.6 s, 1.7 GB | 20 s, 3.7 GB |
| 1,000,000 | 8.5 million | 23 s, 2.2 GB | failed | 22 s, 2.1 GB | 65 s, 7.7 GB |
| 5,000,000 | 42.5 million | 151 s, 4.0 GB | failed | 152 s, 4.0 GB | out of memory |

- **Time.** From 100,000 workers up, hdfe_stream is faster: about 1.5 to 2 times
  at 100,000 and 400,000 workers, and 2.9 times for logit at 1,000,000.
- **Memory.** At 5,000,000 workers pyfixest's logit fit was killed at this
  machine's limit of 29.6 GB (of 30.8 GB). hdfe_stream peaked at 4.0 GB, or
  3.6 to 3.7 GB with the low-memory setting, in 6 to 15% more time. At 25,000
  workers pyfixest uses less, as with OLS: an hdfe_stream process costs about
  0.8 GB before it has done anything.
- **Disk.** About 68 bytes per row: 2.9 GB at 42.5 million rows, a little
  more than OLS on the same design writes (2.6 GB).
- **Against OLS.** `feols_stream` fits the same design in 4.9 s at 1,000,000
  workers and 23 s at 5,000,000, so a Poisson or logit fit costs 5 to 7 OLS
  fits.
- **pyfixest's Poisson fits failed from 1,000,000 workers up.** pyfixest 0.70.0
  stops with `ValueError: signed component between term 1 column 0 and term 2
  column 0 is not diagonally scalable to SDDM form`, raised by the LSMR
  demeaner's preconditioner, at both sizes. pyfixest 0.60.0 completed the
  1,000,000-worker Poisson fit (126 s, 8.0 GB), and ran out of memory at
  5,000,000. Its logit fits, which use the same demeaner, were not affected.

# Poisson and logit, many covariates

`fepois_stream` and `feglm_stream` against pyfixest's `fepois` and `feglm`,
holding the data fixed and varying its width: the two outcomes of [Poisson and
logit](glm.md) on `y ~ i(age_bin) | worker_id + firm_id`, clustered by worker,
on the 1,000,000-worker panel (8.5 million rows), with age bins from five years
wide (8 indicators) to one month wide (503), as in the [many
covariates](covariates.md) benchmark for OLS.

pyfixest runs with its LSMR demeaner (its default, MAP, does not converge on
these panels). hdfe_stream runs at its defaults, and "sized to the design", with
bucket, batch and row-group sizes scaled down as the design widens (the rule is
in [wide designs](../examples/wide_designs.md#2-scale-the-batches-down-with-the-width)).
Wherever both libraries finished, the coefficients agree to within 1e-7, the
tolerance the iterations reach, except for Poisson at 41 covariates, where they
agree to within 3e-6. The numbers are in
[benchmarks/results/glm_covariates.csv](https://github.com/leetucker/hdfe-stream/blob/main/benchmarks/results/glm_covariates.csv).

```bash
python benchmarks/glm_covariates_benchmark.py
python benchmarks/plot_benchmarks.py
```

```{image} ../figures/glm_covariates_time.light.svg
:class: only-light benchmark
:alt: Poisson and logit, many covariates: wall time against the number of covariates, one panel per model, one line per configuration, log-log
```
```{image} ../figures/glm_covariates_time.dark.svg
:class: only-dark benchmark
:alt: Poisson and logit, many covariates: wall time against the number of covariates, one panel per model, one line per configuration, log-log
```

```{image} ../figures/glm_covariates_memory.light.svg
:class: only-light benchmark
:alt: Poisson and logit, many covariates: peak memory against the number of covariates, one panel per model, one line per configuration, log-log
```
```{image} ../figures/glm_covariates_memory.dark.svg
:class: only-dark benchmark
:alt: Poisson and logit, many covariates: peak memory against the number of covariates, one panel per model, one line per configuration, log-log
```

```{image} ../figures/glm_covariates_disk.light.svg
:class: only-light benchmark
:alt: Poisson and logit, many covariates: peak disk use of hdfe_stream against the number of covariates, one panel per model, log-log
```
```{image} ../figures/glm_covariates_disk.dark.svg
:class: only-dark benchmark
:alt: Poisson and logit, many covariates: peak disk use of hdfe_stream against the number of covariates, one panel per model, log-log
```

Wall time and peak memory, hdfe_stream sized to the design:

| covariates | Poisson: `fepois_stream` | Poisson: pyfixest | logit: `feglm_stream` | logit: pyfixest |
|---:|---:|---:|---:|---:|
| 8 | 36 s, 2.8 GB | 321 s, 9.9 GB | 40 s, 2.9 GB | 266 s, 10.3 GB |
| 41 | 113 s, 4.1 GB | 1,256 s, 25.4 GB | 127 s, 4.1 GB | 905 s, 25.6 GB |
| 83 | 251 s, 4.5 GB | out of memory | 268 s, 4.4 GB | out of memory |
| 167 | 484 s, 5.8 GB | out of memory | 515 s, 5.8 GB | out of memory |
| 503 | 1,957 s, 10.6 GB | out of memory | 2,052 s, 10.6 GB | out of memory |

- **pyfixest** needs about 25.5 GB at 41 covariates, nearly all of this
  machine's 26 GB, and runs out of memory from 83 up.
- **hdfe_stream at its defaults** needs more memory as the design widens than
  when sized to it: 12.8 GB at 83 covariates, 24.0 GB at 167, and more than
  the machine has at 503. Its batches and buckets hold a fixed number of rows
  however wide the rows are. For a wide design, size them as the benchmark
  does, as for OLS.
- **Disk** grows with the width, to 6.8 GB at 503 covariates, a little more
  than OLS writes for the same design (5.5 GB).
- **Against OLS.** At 503 covariates, sized the same way, OLS takes 329 s and
  8.6 GB, and the GLM 1,957 s and 10.6 GB.

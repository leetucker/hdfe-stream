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
tolerance the iterations reach. The numbers are in
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
| 8 | 18 s, 3.0 GB | 161 s, 11.2 GB | 16 s, 3.0 GB | 142 s, 11.8 GB |
| 41 | 48 s, 4.9 GB | out of memory | 50 s, 4.2 GB | out of memory |
| 83 | 97 s, 4.7 GB | out of memory | 101 s, 4.8 GB | out of memory |
| 167 | 212 s, 6.2 GB | out of memory | 225 s, 6.1 GB | out of memory |
| 503 | 707 s, 10.9 GB | out of memory | 735 s, 10.9 GB | out of memory |

- **pyfixest** needs 11–12 GB at 8 covariates and was killed at this machine's
  limit of 29.6 GB (of 30.8 GB) from 41 up.
- **hdfe_stream at its defaults** now needs less memory than sized to the
  design up to 167 covariates, in the same time: 2.5 GB at 41 covariates, 3.3 GB
  at 83, 5.0 GB at 167 and 12.5 GB at 503 (sized, Poisson: 4.9, 4.7, 6.2 and
  10.9 GB),
  because GLMs default to chunks of about 64 MB of the design. Only at 503 is
  sizing the buckets and batches, as the benchmark does and as for OLS, worth
  the trouble.
- **Disk** grows with the width, to 6.8 GB at 503 covariates, a little more
  than OLS writes for the same design (5.3 GB).
- **Against OLS.** At 503 covariates, sized the same way, OLS takes 114 s and
  10.9 GB, and the GLM 707 s and 10.9 GB.

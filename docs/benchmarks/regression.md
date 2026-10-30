# OLS, IV, logit and probit

hdfe_stream is built for high-dimensional fixed effects, but its estimators
also fit plain regressions: OLS and weighted least squares, two-stage least
squares, and binary logit and probit, with no fixed effects or with one. These
benchmarks ask how it compares there, against the Python and R implementations
people use for that job, some of them built for speed rather than for
inference: scikit-learn, the Polars plugins polars-ols and polars-ds, NumPy,
statsmodels, linearmodels, glum, pyfixest, and R's lm, glm and fixest.

The data are simulated rows with no panel structure: 10 standard-normal
covariates, a group variable with one group per 20 rows (the fixed effect), an
endogenous regressor with two instruments, and a continuous and a binary
outcome, from 100,000 to 20 million rows
(`_harness.general_data_path`). The models are

```
y   ~ x0 + ... + x9                           OLS, no fixed effects
y   ~ x0 + ... + x9 | g                       OLS, one fixed effect
yiv ~ x0 + ... + x9 | d ~ z1 + z2             2SLS, and with  | g |  for one fixed effect
yb  ~ x0 + ... + x9                           logit and probit
```

Each configuration runs in its own process on the [same
machine](index.md#the-machine) as the other CPU benchmarks, and wall time
includes reading the Parquet file and turning it into what the library takes
([how it is measured](index.md#how-it-is-measured)). Estimates agree across
libraries to 5e-13 (OLS and IV) and 2.4e-7 (logit and probit, where the
solvers stop at different tolerances). Everything below is one run per point.
The numbers are in
[ols.csv](https://github.com/leetucker/hdfe-stream/blob/main/benchmarks/results/ols.csv),
[iv.csv](https://github.com/leetucker/hdfe-stream/blob/main/benchmarks/results/iv.csv) and
[binary.csv](https://github.com/leetucker/hdfe-stream/blob/main/benchmarks/results/binary.csv).

```bash
python benchmarks/ols_benchmark.py
python benchmarks/iv_benchmark.py
python benchmarks/binary_benchmark.py
python benchmarks/plot_benchmarks.py
```

Libraries do different amounts of work, and the rows say so. hdfe_stream and
most others return coefficients with classical standard errors.
scikit-learn, glum and polars-ds's `logistic_reg` return coefficients only.
Where a library cannot absorb a fixed effect or fit 2SLS, the row demeans the
data in Polars or pandas first, or runs the two stages by hand, and that work
is in its time; such rows return coefficients, not standard errors that are
right for the fixed effect or the second stage. "NumPy" is the normal
equations on arrays: about the least work an OLS or 2SLS fit can do, and a
floor, not a competitor.

## OLS

```{image} ../figures/ols_time.light.svg
:class: only-light benchmark
:alt: OLS regression: wall time against the number of rows, one panel for no fixed effects and one for one fixed effect, one line per library, log-log
```
```{image} ../figures/ols_time.dark.svg
:class: only-dark benchmark
:alt: OLS regression: wall time against the number of rows, one panel for no fixed effects and one for one fixed effect, one line per library, log-log
```

```{image} ../figures/ols_memory.light.svg
:class: only-light benchmark
:alt: OLS regression: peak memory against the number of rows, one panel for no fixed effects and one for one fixed effect, one line per library, log-log
```
```{image} ../figures/ols_memory.dark.svg
:class: only-dark benchmark
:alt: OLS regression: peak memory against the number of rows, one panel for no fixed effects and one for one fixed effect, one line per library, log-log
```

No fixed effects, wall time and peak memory:

| rows | hdfe_stream | scikit-learn | polars-ols | polars-ds | statsmodels | pyfixest | fixest (R) | lm (R) | NumPy |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100,000 | 1.00 s, 0.7 GB | 0.72 s, 0.3 GB | 0.09 s, 0.2 GB | 0.07 s, 0.2 GB | 0.05 s, 0.4 GB | 1.0 s, 0.5 GB | 0.12 s, 0.3 GB | 0.17 s, 0.3 GB | 0.02 s, 0.2 GB |
| 1 million | 1.3 s, 1.3 GB | 0.69 s, 0.6 GB | 0.38 s, 0.5 GB | 0.21 s, 0.6 GB | 0.35 s, 0.9 GB | 1.3 s, 1.3 GB | 0.39 s, 0.7 GB | 1.2 s, 0.8 GB | 0.10 s, 0.4 GB |
| 5 million | 3.0 s, 1.8 GB | 1.7 s, 2.1 GB | 2.1 s, 1.6 GB | 0.51 s, 2.4 GB | 1.9 s, 3.0 GB | 2.8 s, 4.3 GB | 0.88 s, 2.2 GB | 4.8 s, 3.1 GB | 0.32 s, 1.5 GB |
| 20 million | 6.5 s, 2.5 GB | 5.4 s, 7.5 GB | 8.5 s, 5.7 GB | timeout | 7.9 s, 10.9 GB | 9.1 s, 15.7 GB | 2.9 s, 7.8 GB | 18.6 s, 11.7 GB | 1.2 s, 5.3 GB |

One fixed effect:

| rows | hdfe_stream | pyfixest | fixest (R) | linearmodels | polars-ols, demeaned first |
|---:|---:|---:|---:|---:|---:|
| 100,000 | 1.0 s, 0.7 GB | 1.0 s, 0.6 GB | 0.14 s, 0.3 GB | 0.80 s, 0.4 GB | 0.10 s, 0.2 GB |
| 1 million | 1.6 s, 1.7 GB | 1.5 s, 1.6 GB | 0.56 s, 0.8 GB | 2.3 s, 1.3 GB | 0.41 s, 0.6 GB |
| 5 million | 5.0 s, 2.2 GB | 3.7 s, 5.4 GB | 1.6 s, 2.8 GB | 10.8 s, 5.0 GB | 2.2 s, 2.1 GB |
| 20 million | 9.5 s, 2.6 GB | 12.7 s, 20.0 GB | 5.6 s, 10.4 GB | 44.3 s, 18.9 GB | 9.2 s, 6.5 GB |

- **Time.** At 20 million rows `feols_stream` (6.5 s) is within about 1.2 times
  of scikit-learn (5.4 s) and faster than statsmodels (7.9 s), polars-ols
  (8.5 s) and pyfixest (9.1 s). fixest is faster (2.9 s, with 3 times the
  memory), and so is the NumPy floor (1.2 s). At 1 to 5 million rows
  hdfe_stream is slower than the fast libraries, by 1.4 to 3.7 times:
  scikit-learn, polars-ols and statsmodels fit 5 million rows in 1.7 to 2.1 s,
  against 3.0 s. With one fixed effect, at 20 million rows, hdfe_stream (9.5 s) matches
  demeaning in Polars and then calling polars-ols (9.2 s), and is ahead of
  pyfixest (12.7 s) and well ahead of linearmodels' `AbsorbingLS` (44 s);
  fixest is again faster (5.6 s).
- **Memory.** At 20 million rows hdfe_stream peaked at 2.5 GB with no fixed
  effect and 2.6 GB with one. The others needed 5.3 to 15.7 GB (5.3 GB for the
  NumPy floor; of the libraries, polars-ols was leanest at 5.7 GB and
  pyfixest the largest at 15.7 GB); with one fixed effect, 6.5 to 20.0 GB. The gap widens with size, since hdfe_stream holds a batch of
  rows, not the data.
- **Small data.** Below about a million rows it is slower and heavier: a fit
  takes about a second at 100,000 rows, against 0.05 to 0.7 s for most of the
  others, and its process peaks at 0.7 GB there.
- **polars-ds** ran out of the 30 minutes allowed at 20 million rows. Its
  `lin_reg_report` (the call that returns standard errors) takes about a second
  up to 14 million rows and, on this data, over a minute from 15 million, a
  slowdown we reproduced on a second machine and did not look into further; its
  coefficients-only `lin_reg` fits 20 million rows in 0.6 s. The benchmark
  times the call with standard errors, as it does for the others.

## Two-stage least squares

```{image} ../figures/iv_time.light.svg
:class: only-light benchmark
:alt: Two-stage least squares: wall time against the number of rows, one panel for no fixed effects and one for one fixed effect, one line per library, log-log
```
```{image} ../figures/iv_time.dark.svg
:class: only-dark benchmark
:alt: Two-stage least squares: wall time against the number of rows, one panel for no fixed effects and one for one fixed effect, one line per library, log-log
```

```{image} ../figures/iv_memory.light.svg
:class: only-light benchmark
:alt: Two-stage least squares: peak memory against the number of rows, one panel for no fixed effects and one for one fixed effect, one line per library, log-log
```
```{image} ../figures/iv_memory.dark.svg
:class: only-dark benchmark
:alt: Two-stage least squares: peak memory against the number of rows, one panel for no fixed effects and one for one fixed effect, one line per library, log-log
```

No fixed effects:

| rows | hdfe_stream | linearmodels | pyfixest | fixest (R) | polars-ols, two stages | NumPy |
|---:|---:|---:|---:|---:|---:|---:|
| 100,000 | 1.3 s, 0.7 GB | 0.70 s, 0.5 GB | 1.0 s, 0.7 GB | 0.23 s, 0.4 GB | 0.12 s, 0.2 GB | 0.03 s, 0.2 GB |
| 1 million | 1.9 s, 1.4 GB | 2.9 s, 1.8 GB | 1.8 s, 2.2 GB | 1.0 s, 1.2 GB | 0.69 s, 0.5 GB | 0.19 s, 0.8 GB |
| 5 million | 4.2 s, 2.0 GB | 13.4 s, 7.3 GB | 5.5 s, 9.2 GB | 3.2 s, 5.1 GB | 4.3 s, 1.8 GB | 0.82 s, 3.1 GB |
| 20 million | 12.5 s, 3.2 GB | 56.3 s, 28.0 GB | out of memory | 11.6 s, 19.3 GB | 17.5 s, 6.7 GB | 3.2 s, 10.4 GB |

One fixed effect:

| rows | hdfe_stream | pyfixest | fixest (R) | linearmodels, demeaned first | polars-ols, two stages |
|---:|---:|---:|---:|---:|---:|
| 100,000 | 1.4 s, 0.8 GB | 1.1 s, 0.7 GB | 0.28 s, 0.4 GB | 0.72 s, 0.5 GB | 0.13 s, 0.2 GB |
| 1 million | 2.2 s, 1.8 GB | 2.2 s, 2.4 GB | 1.1 s, 1.5 GB | 3.1 s, 1.9 GB | 0.71 s, 0.7 GB |
| 5 million | 5.1 s, 2.7 GB | 8.3 s, 9.3 GB | 3.8 s, 6.6 GB | 14.7 s, 7.5 GB | 4.2 s, 2.5 GB |
| 20 million | 18.0 s, 3.1 GB | out of memory | 13.6 s, 25.1 GB | 59.5 s, 26.7 GB | 17.4 s, 7.0 GB |

- **Time.** At 20 million rows `feols_stream` took 12.5 s without a fixed
  effect and 18.0 s with one. fixest took 11.6 s and 13.6 s, the hand-run two
  stages in polars-ols 17.5 s and 17.4 s, linearmodels 56 s and 60 s. pyfixest
  ran out of memory in both. At 5 million rows hdfe_stream (4.2 s) is between
  fixest (3.2 s) and pyfixest (5.5 s), and three times as fast as
  linearmodels (13.4 s).
- **Memory.** 3.2 GB (3.1 GB with one fixed effect) at 20 million rows,
  against 6.7 to 28 GB, and 19 to 25 GB for fixest. pyfixest was killed at this
  machine's memory.

## Logit and probit

```{image} ../figures/binary_time.light.svg
:class: only-light benchmark
:alt: Logit and probit regression: wall time against the number of rows, one panel per model, one line per library, log-log
```
```{image} ../figures/binary_time.dark.svg
:class: only-dark benchmark
:alt: Logit and probit regression: wall time against the number of rows, one panel per model, one line per library, log-log
```

```{image} ../figures/binary_memory.light.svg
:class: only-light benchmark
:alt: Logit and probit regression: peak memory against the number of rows, one panel per model, one line per library, log-log
```
```{image} ../figures/binary_memory.dark.svg
:class: only-dark benchmark
:alt: Logit and probit regression: peak memory against the number of rows, one panel per model, one line per library, log-log
```

Logit, no fixed effects:

| rows | hdfe_stream | statsmodels | pyfixest | fixest (R) | glm (R) | scikit-learn | glum | polars-ds |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100,000 | 1.1 s, 0.6 GB | 0.14 s, 0.3 GB | 1.0 s, 0.5 GB | 0.20 s, 0.3 GB | 0.40 s, 0.3 GB | 0.74 s, 0.3 GB | 0.32 s, 0.3 GB | 0.22 s, 0.2 GB |
| 1 million | 1.8 s, 1.3 GB | 0.56 s, 0.7 GB | 1.6 s, 1.4 GB | 1.1 s, 0.9 GB | 2.9 s, 1.4 GB | 0.75 s, 0.5 GB | 0.58 s, 0.6 GB | 0.51 s, 0.4 GB |
| 5 million | 3.6 s, 1.9 GB | 3.1 s, 2.2 GB | 4.5 s, 5.5 GB | 5.0 s, 3.3 GB | 13.2 s, 5.4 GB | 2.0 s, 1.7 GB | 1.4 s, 1.8 GB | 2.5 s, 2.3 GB |
| 20 million | 9.4 s, 2.4 GB | 12.8 s, 7.7 GB | 15.9 s, 20.6 GB | 19.0 s, 12.1 GB | 58.2 s, 18.2 GB | 6.4 s, 6.1 GB | 4.2 s, 5.5 GB | 10.0 s, 8.7 GB |

Probit:

| rows | hdfe_stream | statsmodels | pyfixest | fixest (R) | glm (R) |
|---:|---:|---:|---:|---:|---:|
| 100,000 | 1.0 s, 0.6 GB | 0.10 s, 0.3 GB | 1.0 s, 0.5 GB | 0.34 s, 0.3 GB | 0.48 s, 0.3 GB |
| 1 million | 1.7 s, 1.2 GB | 0.72 s, 0.7 GB | 1.9 s, 1.4 GB | 2.2 s, 1.0 GB | 3.2 s, 1.3 GB |
| 5 million | 3.6 s, 1.7 GB | 4.1 s, 2.3 GB | 6.8 s, 5.5 GB | 10.4 s, 4.0 GB | 14.7 s, 5.4 GB |
| 20 million | 9.9 s, 2.3 GB | 17.0 s, 8.1 GB | 25.2 s, 20.6 GB | 42.2 s, 15.0 GB | 65.0 s, 20.9 GB |

- **Time.** At 20 million rows `feglm_stream` took 9.4 s for logit and 9.9 s for
  probit. Among libraries that give standard errors, that is faster than
  statsmodels (12.8 s, 17.0 s), pyfixest (15.9 s, 25.2 s), fixest (19.0 s,
  42.2 s) and R's `glm` (58 s, 65 s). The coefficients-only fits are faster
  for logit: glum 4.2 s and scikit-learn 6.4 s; polars-ds (10.0 s) is not. At
  1 to 5 million rows statsmodels and those fits are faster than hdfe_stream
  (glum 1.4 s and statsmodels 3.1 s, against 3.6 s, at 5 million).
- **Memory.** 2.4 GB at 20 million rows (2.3 GB for probit), against 5.5 to 20.9 GB.

## What to take from this

On data that fits in memory, hdfe_stream is a competitive general-purpose
regression engine, not the fastest one. Libraries that do less work or
fit only one model are faster below about 5 million rows; at 20 million, fixest
is about 2.2 times faster at OLS and glum about 2.2 times faster at logit. The
package's case is that, from a few million rows, it is within a small factor
of most of them and ahead of some, while using a fraction of their memory, with one interface for OLS, WLS, 2SLS
and binary GLMs, standard errors that include clustering, and formulas, and
that its time and memory grow slowly enough to keep going where the in-memory
libraries stop.

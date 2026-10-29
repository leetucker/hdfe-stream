# Poisson, logit and probit

`fepois_stream` and `feglm_stream` fit generalized linear models with
high-dimensional fixed effects on data that does not fit in memory, with the
same formula syntax, inputs and results as `feols_stream`.

```python
from hdfe_stream import fepois_stream, feglm_stream

pois = fepois_stream("visits ~ x | worker_id + firm_id + year", "data/*.parquet",
                     offset="log_exposure", workdir="scratch")
logit = feglm_stream("promoted ~ x | firm_id + year", "data/*.parquet", "logit",
                     workdir="scratch", vcov={"CRV1": "firm_id"})
```

The estimates, standard errors, deviance and log-likelihood match pyfixest's
`fepois` and `feglm` to about 1e-12 on the test panels, with none, one, two
and three fixed effects, interacted fixed effects, every vcov, weights and
offsets (`tests/test_glm.py`). The exceptions, all deliberate, are listed
under [differences from pyfixest](#differences-from-pyfixest). Benchmarks
against pyfixest are under [GLM](../benchmarks/glm.md) and
[GLM, many covariates](../benchmarks/glm_covariates.md).

## The example

The example's data are simulated with a known coefficient of 0.3 on `x`, so the
estimates can be checked against it.

**Poisson with an offset.** The offset enters the linear predictor with a
coefficient of one (here, log exposure). Workers whose count is zero in every
year have a worker effect of minus infinity; they are dropped before the fit,
with a warning, and the result says so:

```python
pois = fepois_stream("visits ~ x | worker_id + firm_id + year", data,
                     offset="log_exposure", workdir="scratch")
pois.diagnostics["separation"]     # 4,755 rows of 600 workers dropped
pois.diagnostics["irls"]           # 9 steps, and the deviance after each
pois.resid().select("visits", "eta", "fitted", "resid").head(3).collect()
```

The coefficient on `x` is 0.2958 (se 0.0040). The residual file holds the
linear predictor, the fitted mean and the response residual for every row.

**Logit and probit.** With firm and year effects only, there are many rows per
firm, the effects are well estimated, and the coefficient is close to 0.3
(probit's is on a different scale, about 0.3 / 1.6):

```text
logit   firm + year effects:   x = 0.3156  (se 0.0055, clustered by firm)
probit  firm + year effects:   x = 0.1871  (se 0.0032, clustered by firm)
logit   + worker effects:      x = 0.3626  (12,700 rows of workers with an unchanging outcome dropped)
```

Adding worker effects, estimated from about eight rows each, moves the logit
coefficient away from the truth even though the model is correct: the
[incidental parameter bias](#not-available), which neither this package nor
pyfixest corrects.

**The low-level interface.** [`StreamingGLM`](../reference/lowlevel.md) is to
`fepois_stream` and `feglm_stream` what `StreamingHDFE` is to `feols_stream`:
column names, or `(name, Polars expression)` pairs, instead of a formula, so
pyfixest is not needed:

```python
from hdfe_stream import StreamingGLM

low = StreamingGLM("visits", ["x"], ["worker_id", "firm_id", "year"], family="poisson",
                   offset=pl.col("log_exposure"), workdir="scratch").fit(data)
```

## How it works

A GLM is fitted by iteratively reweighted least squares (IRLS): each step is
a weighted linear regression of a *working response* on the covariates and
the fixed effects, with weights that depend on the current fit. That
regression is the one `feols_stream` solves, so the GLM reuses its machinery:

- **Once:** pass 0 reads, filters, partitions and sorts the rows, as for OLS,
  and pass 1 finds the cell structure (which worker–firm cells identify the
  firm effects, the connected components).
- **Each step:** one read of the sorted rows computes every row's linear
  predictor from the current coefficients, then its IRLS weight and working
  response, and sums them per cell; the reduced system for the non-streamed
  effects is solved from those sums, starting from the previous step's
  solution; and the coefficients and effects are updated from the cells.
- **Once more:** a final read writes the residual and fixed-effect files and
  computes the standard errors, as for OLS.

Nothing row-sized is written during the iterations. The linear predictor is
rebuilt from the coefficients on each read, and what is rewritten each step
are the cells' weights and sums (memory-mapped, like OLS's identifying
cells). The disk needed is therefore no more than an OLS fit of the same data
needs. A fit usually needs 6 to 10 steps, so it takes several times as long as
OLS on the same data: 4 to 7 times, in the benchmarks.

**Memory.** One difference from OLS: the streamed dimension's effects are
held in memory during the iterations, one float per group for each of the
two or three coefficient sets alive at once (for 5 million workers, about
120 MB). OLS never holds a vector over the streamed dimension.

**Several models.** Each model is its own sequence of IRLS steps: the weights
depend on the model, so models cannot share a solve the way `feols_stream`'s
do. Models of the same outcome share pass 0 (the separation check depends on
the outcome, so each outcome gets its own).

**Solver.** The reduced matrix S depends on the weights, so the `explicit`
solver has to rebuild it at every step. Streaming the cells instead costs one
pass over them per conjugate-gradient iteration, and each solve starts from the
previous step's solution, so the count falls from step to step: from 19 at the
first step to 6 at the eighth on one 3.4-million-row panel. With
`solver="auto"`, the default, S is therefore built for the first step only, and
`stream_cg` is used after that. From 400,000 to 1,000,000 workers, `explicit`
takes within 5% of the time `auto` takes; at 5,000,000, it takes 12% less. With `precond="amg"`, which needs S, it is rebuilt at every step. Setting
`solver=` explicitly applies it to every step.

## Options

| option | default | what it does |
|---|---|---|
| `family` | | `feglm_stream` only: `"logit"` or `"probit"` |
| `offset` | none | `fepois_stream` only: a column added to the linear predictor with a coefficient of one |
| `iwls_tol` | `1e-8` | converged when the relative change in deviance, \|dev − dev_old\| / (0.1 + \|dev\|), falls below this |
| `iwls_maxiter` | `25` | the most IRLS steps; a fit that reaches it warns, and `diagnostics["irls"]["converged"]` is False |
| `separation_check` | `True` | drop fixed-effect levels whose effect would be infinite; see below |
| `weights`, `weights_type` | none | analytic or frequency weights, multiplying each row's log-likelihood |
| `bias_correction` | none | logit and probit: correct the coefficients for the fixed effects' incidental-parameter bias; see below |
| `within_pseudo_r2` | `False` | also fit the fixed-effects-only model when it is not free (two or more fixed effects, or an offset) |

Everything else (`vcov`, `cluster`, `fe_dof`, `solver`, `batch_rows`,
`workdir`, ...) is as in `feols_stream`, except that GLMs read chunks of
about 64 MB of the design unless `batch_rows` is given, with BLAS on one
thread up to 128 columns (faster, and smaller chunks then cost nothing). `assembly` does not apply. The
outcome must be nonnegative for Poisson, and 0 or 1 for logit and probit.

## Results

The result is the same [`HDFEResult`](../reference/results.md) as for OLS, with:

- `family`, `deviance`, `loglik`, `loglik_null` (the constant-only model's),
  `pseudo_r2` (McFadden's, one minus their ratio), and `aic` and `bic`
  (counting the fixed effects' degrees of freedom, as fixest);
- with fixed effects, `loglik_fe_null` and `pseudo_r2_within`: the model with
  the fixed effects and no covariates, and what the covariates add beyond it
  (fixest's within pseudo R²). With one fixed effect and no offset it is the
  groups' means, computed in the first pass; otherwise it is a fit of its
  own, made with `within_pseudo_r2=True`;
- `rss`, `r2`, `r2_within` and `rmse` set to nan;
- p-values and confidence intervals from the normal distribution, as in
  pyfixest (`tidy()`, `to_pyfixest()`);
- `diagnostics["irls"]`: the number of steps, whether they converged, the
  deviance after each, and how many step-halvings were needed;
- `diagnostics["separation"]`: how many observations and levels were dropped,
  and in how many rounds;
- residual-file columns `eta` (the linear predictor, including any offset),
  `fitted` (the mean), `resid` (the response residual, y − fitted) and
  `resid_working` (the working residual of the final step), besides the
  outcome, `xb` and each row's fixed effects. The fixed effects are on the
  scale of the linear predictor, and `eta` is `xb` plus the fixed effects plus
  the offset.

The vcov is the one pyfixest computes: that of the final weighted
least-squares step. The bread is the inverse of X̃′WX̃, the scores are the
weighted working residuals times the residualized covariates, and the
small-sample factors are pyfixest's. The iid vcov is the bread alone times
(N − 1) / (N − K), since no dispersion parameter is estimated.

## Separation

A fixed-effect level whose outcome is zero on every row (Poisson), or the same
on every row (logit, probit), has an infinite effect: the likelihood keeps
increasing as the effect grows. Such levels are dropped before anything is
written to disk, along with their rows.

- **Poisson:** one pass over the dimensions finds every such level, since
  dropping rows whose outcome is zero leaves every other level's positive rows
  where they were. This is pyfixest's `"fe"` check.
- **Logit and probit:** dropping a level can leave another with a constant
  outcome. Workers whose only 1s were at a firm where everyone had a 1 are left
  with only 0s once that firm is dropped. The passes repeat until one drops
  nothing, and `diagnostics["separation"]["rounds"]` says how many dropped
  something.

A warning reports how many observations were dropped, and `sample()` labels
their rows `"separation"`. `separation_check=False` turns the check off. The
iterations then push those effects toward infinity, and the fit may not
converge.

**Separation by the covariates is not detected.** An outcome that a
combination of covariates predicts perfectly (pyfixest's `"ir"` check, after
Correia, Guimarães and Zylkin) makes some coefficients diverge. The symptom is
a fit that does not converge, or coefficients and standard errors that are
very large.

## Incidental-parameter bias

With fixed effects estimated from few observations each, such as worker
effects in a panel of a few years, logit and probit coefficients are biased,
by O(1/T) in the rows per level. The example shows it: the true coefficient
is 0.3, firm and year effects estimate 0.32, and adding worker effects (about
eight rows each) gives 0.36. pyfixest and fixest leave this alone.

`bias_correction=True` applies the analytic correction of Fernández-Val and
Weidner (2016), computed as R's alpaca computes it (`biasCorr`; the tests
match it to 1e-7). On the example's data with worker and year effects, it
moves the coefficient from 0.358 to 0.315.

```python
fit = feglm_stream("promoted ~ x | worker_id + year", data, "logit",
                   bias_correction=True)
fit.coef()                                       # corrected
fit.diagnostics["bias_correction"]               # uncorrected, the bias, ...
```

- **Which fixed effects:** one or two, all of them. For strictly exogenous
  regressors the two terms have the same form, so nothing says which is the
  unit and which the period.
- **Predetermined regressors** (a lagged outcome, a regressor that responds
  to past shocks) add a term from the scores' correlation over time within
  each unit: `bias_correction={"time": "year", "lags": 2}` names the period
  effect and the window L, and the other fixed effect is the unit. Each unit
  and period must then have at most one row, the period must be numeric or a
  date, and the unit is made the streamed dimension, so its rows arrive in
  period order. Lags count observed periods: a warning reports units with
  gaps, and units with no more periods than L.
- **After the correction**, the fixed effects are re-estimated with the
  corrected coefficients held (one more IRLS fit, from where the first
  ended), and the variances, residuals, fixed effects and predictions are
  those of that fit, as in alpaca. The fit statistics (deviance,
  log-likelihood, AIC) remain the maximum-likelihood fit's.
- **Checks:** logit or probit; one or two fixed effects; with lags, the
  period among them, and one column. A warning when the median rows per
  level of a corrected effect is below five: the correction is first-order
  and unreliable for very short panels.
- **Not for Poisson:** with one or two fixed effects and strictly exogenous
  regressors its coefficients have no such bias. The three-way (gravity)
  case, where they do (Weidner and Zylkin 2021), is not implemented.

## Differences from pyfixest

**Logit and probit separation.** pyfixest drops levels whose outcome is all 0,
in one pass, and keeps levels whose outcome is all 1. Their effects then grow
without bound while their rows contribute almost nothing. This package drops
both kinds, repeatedly.

- **Coefficients:** the same, to the tolerance the iterations reach, since those
  rows carry no information about them.
- **N and the fixed-effect count:** differ. The standard errors therefore differ
  by the small-sample factors, and pyfixest reports effects for levels this
  package drops.
- **Tests:** they give pyfixest this package's estimation sample.

**Frequency weights.** Here they mean what they mean in OLS: a row with weight
3 counts as three identical rows. The fit, N and every vcov then equal those of
the data with each row repeated, which is what the tests check.

- **pyfixest's `fepois`** counts N as the number of rows, and scales the
  heteroskedastic meat differently. On the test panel its robust standard
  errors differ from those of the repeated data by 3.6%.
- **pyfixest's `feglm`** takes no weights.

**Weights in logit and probit.** They multiply each row's log-likelihood, as in
`fepois` and fixest. Analytic weights give the same coefficients as frequency
weights with the same values; only N and the small-sample factors differ.

**Analytic weights are rescaled to average 1** over the estimation sample, as
Stata does for maximum-likelihood estimators. A GLM has no dispersion to
absorb their scale, so otherwise the iid vcov, the log-likelihood and the bias
correction would change when every weight is multiplied by a constant
(survey weights summing to a population would give iid standard errors far
too small). The coefficients and the robust and clustered vcovs do not
depend on the scale. pyfixest's `fepois` uses the weights as given, so its
iid standard errors differ from these by the square root of the average
weight; with the weights rescaled first, the two agree (the tests check
both).

**Convergence.** The stopping rule uses the weighted deviance; pyfixest's
`fepois` uses the unweighted one. pyfixest's `feglm` halves a step that raises
the deviance, and this package does the same for all three families, except
Poisson's first step (its starting point is not a set of coefficients, as in
pyfixest). A step that changes the deviance by less than the tolerance counts
as converged without halving, so rounding at the optimum does not trigger a
search.

At tight tolerances (`iwls_tol=1e-12`) both reach the same optimum to about
1e-14. At the default of 1e-8, where each stops can differ in the last few
digits of the coefficients.

## Not available

- **CRV3.** The cluster jackknife is computed by downdating a least-squares
  fit, which is not the jackknife of a GLM. Use CRV1. (pyfixest's CRV3 for
  `feglm` refits each subsample with `fepois`, so it is not a reference for
  logit or probit either.)
- **IV, varying slopes, leave-out (KSS) variance components.** These are linear
  models only.
- **Bias correction with three or more fixed effects,** or for Poisson (see
  above).

:::{dropdown} The full script: `examples/glm.py`
```{literalinclude} ../../examples/glm.py
:language: python
```
:::

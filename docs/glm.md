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
under [differences from pyfixest](#differences-from-pyfixest).

## How it works

A GLM is fitted by iteratively reweighted least squares (IRLS): each step is
a weighted linear regression of a *working response* on the covariates and
the fixed effects, with weights that depend on the current fit. That
regression is the one `feols_stream` solves, so the GLM reuses its machinery:

- **Once:** pass 0 reads, filters, partitions and sorts the rows, as for OLS,
  and passes 1 and 1b find the cell structure (which worker–firm cells
  identify the firm effects, the connected components).
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
cells). The disk needed is therefore about what an OLS fit of the same data
needs.

**Memory.** One difference from OLS: the streamed dimension's effects are
held in memory during the iterations, one float per group for each of the
two or three coefficient sets alive at once (for 5 million workers, about
120 MB). OLS never holds a vector over the streamed dimension.

**Several models.** Each model is its own sequence of IRLS steps: the weights
depend on the model, so models cannot share a solve the way `feols_stream`'s
do. Models of the same outcome share pass 0 (the separation check depends on
the outcome, so each outcome gets its own).

## Options

| option | default | what it does |
|---|---|---|
| `family` | | `feglm_stream` only: `"logit"` or `"probit"` |
| `offset` | none | `fepois_stream` only: a column added to the linear predictor with a coefficient of one |
| `iwls_tol` | `1e-8` | converged when the relative change in deviance, \|dev − dev_old\| / (0.1 + \|dev\|), falls below this |
| `iwls_maxiter` | `25` | the most IRLS steps; a fit that reaches it warns, and `diagnostics["irls"]["converged"]` is False |
| `separation_check` | `True` | drop fixed-effect levels whose effect would be infinite; see below |
| `weights`, `weights_type` | none | analytic or frequency weights, multiplying each row's log-likelihood |

Everything else (`vcov`, `cluster`, `fe_dof`, `solver`, `batch_rows`,
`workdir`, ...) is as in `feols_stream`. `assembly` does not apply.

## Results

The result is the same `HDFEResult` as for OLS, with:

- `family`, `deviance`, `loglik`, and `pseudo_r2` (McFadden's, one minus the
  ratio of the log-likelihood to the constant-only model's);
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

A warning reports how many observations were dropped.
`separation_check=False` turns the check off. The iterations then push those
effects toward infinity, and the fit may not converge.

**Separation by the covariates is not detected.** An outcome that a
combination of covariates predicts perfectly (pyfixest's `"ir"` check, after
Correia, Guimarães and Zylkin) makes some coefficients diverge. The symptom is
a fit that does not converge, or coefficients and standard errors that are
very large.

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
- **Incidental parameter bias.** With fixed effects estimated from few
  observations each, such as worker effects in a panel of a few years, logit and
  probit coefficients are biased away from zero. Neither pyfixest nor this
  package corrects for it. `examples/glm.py` shows it: the true coefficient is
  0.3, firm and year effects estimate 0.32, and adding worker effects (about
  eight rows each) gives 0.36.

## Performance

A step costs one read of the sorted rows and one solve of the reduced system.
A fit typically takes 6 to 10 steps, so it costs a few times what an OLS fit
of the same data costs.

**Solver.** The reduced matrix S depends on the weights, so the `explicit`
solver has to rebuild it at every step. On the 3.4-million-row panel below,
each rebuild sorted 23 million entries, and rebuilding took 40% of the fit.

Streaming the cells instead costs one pass over them per conjugate-gradient
iteration. Each solve starts from the previous step's solution, so the count
falls from step to step: from 19 at the first step to 6 at the eighth on that
panel. With `solver="auto"`, the default, S is therefore built for the first
step only, and `stream_cg` is used after that.

With `precond="amg"`, which needs S, it is rebuilt at every step. Setting
`solver=` explicitly applies it to every step.

Wall time on the simulated AKM panel, with a Poisson outcome and a logit
outcome both driven by the worker and firm effects. The model is
`y ~ x + age_cubed | worker_id + firm_id + year`, with defaults on both sides,
except that pyfixest uses its LSMR demeaner. Its default demeaner did not
converge on this panel.

| rows | `feols_stream` | `fepois_stream` | pyfixest `fepois` | `feglm_stream` logit | pyfixest `feglm` logit |
|---|---|---|---|---|---|
| 850,000 | 0.9 s | 3.6 s (8 steps) | 7.9 s | 4.5 s (10 steps) | 12.8 s |
| 3,400,000 | 3.9 s | 12.4 s (8 steps) | 46.2 s | 15.5 s (10 steps) | 54.2 s |

These are single runs on one machine, not the benchmark suite, and they do not
measure memory. Both libraries' coefficients agree.

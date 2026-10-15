# Leave-out variance components (KSS)

A common use of the AKM model is to decompose the variance of earnings into the
variance of worker effects, the variance of firm effects and twice their
covariance. The plug-in decomposition, computed from the estimated effects, is
biased. Worker and firm effects are each estimated from a handful of
observations, and that estimation error inflates their measured variance and
attenuates the covariance between them, so the plug-in numbers overstate
heterogeneity and understate sorting. The shorter the panel, the worse it gets:
this is the limited-mobility bias of Andrews et al. (2008).

[Kline, Saggio and Sølvsten (2020)](https://doi.org/10.3982/ECTA16410) (KSS)
remove it without assuming homoskedasticity, and
[`leave_out_kss`](../reference/kss.md) implements that, streaming like the rest
of the package. It is an implementation of their standard cases, following
Saggio's reference code,
[LeaveOutTwoWay](https://github.com/rsaggio87/LeaveOutTwoWay). It is not an
extension of the methodology, and every departure from LeaveOutTwoWay,
VarianceComponentsHDFE.jl, xhdfe and pytwoway is listed, with its motivation and
measured effect, in [methodological differences](differences.md).

```python
from hdfe_stream import leave_out_kss

lo = leave_out_kss("log_earn ~ age_squared | worker_id + firm_id",
                   "data/*.parquet", workdir="scratch", n_draws=250)
lo.summary()
lo.tidy()        # component / plug_in / bias / leave_out, as a Polars frame
```

```text
component                plug-in        bias   leave-out
--------------------------------------------------------
var(psi)                0.063973    0.002997    0.060976
var(alpha)              0.178427    0.012475    0.165952
cov(psi, alpha)         0.027097   -0.002642    0.029739
```

This section covers:

- this page: the problem, what one call does, the choices it offers, and how it
  is validated;
- [how the correction streams](streaming.md): why every piece of it can be
  computed with passes over sorted rows, and what that costs;
- [standard errors and weak identification](inference.md): the standard
  errors, their measured coverage, KSS's diagnostic for when the normal
  interval is not justified, and the interval that stays valid when it is not;
- [draws and reproducibility](draws.md): how many random-projection draws are
  enough, what the results depend on, and what to report in a paper;
- [methodological differences](differences.md): every way this implementation
  differs from the reference implementations;
- the [KSS benchmarks](../benchmarks/kss.md): time, memory and disk against
  xhdfe, up to 42.5 million rows.

[The example](../examples/leave_out_kss.md) runs the correction against the
effects the data was generated from.

## The problem and the correction

Write the two-way model as $y = Z\delta + \varepsilon$, where $Z = [D_0, D_1, X]$
collects the worker indicators, the firm indicators and any covariates, and
$\delta = (\alpha', \psi', \beta')'$. Each variance component is a quadratic
form in the parameters, $\theta = \delta' Q \delta$; for example, the variance
of firm effects across person-years is
$\theta = \frac{1}{N}\sum_i (\psi_{J(i)} - \bar\psi)^2$, which corresponds to a
particular positive semi-definite $Q$. The plug-in estimator
$\hat\theta_{PI} = \hat\delta' Q \hat\delta$ is biased, because the estimation
error in $\hat\delta$ enters quadratically:

$$
\mathbb{E}\big[\hat\theta_{PI}\big] = \theta + \operatorname{tr}\big(Q\,\mathrm{Var}(\hat\delta)\big)
= \theta + \sum_{i=1}^{N} B_{ii}\,\sigma_i^2,
\qquad B_{ii} = z_i' S_{zz}^{-} Q\, S_{zz}^{-} z_i,
$$

where $S_{zz} = Z'Z$ and $\sigma_i^2 = \mathrm{Var}(\varepsilon_i)$. When worker
and firm effects are estimated from few observations, as they are in panels with
short spells and limited mobility, this bias is large: it inflates the estimated
variances of worker and firm effects and biases their covariance downwards.

KSS remove the bias without assuming homoskedasticity. Their estimator replaces
$\sigma_i^2$ by the leave-one-out estimate

$$
\hat\sigma_i^2 = y_i\big(y_i - z_i'\hat\delta_{-i}\big)
               = y_i\,\frac{\hat\varepsilon_i}{1-P_{ii}},
\qquad P_{ii} = z_i' S_{zz}^{-} z_i,
$$

where $\hat\delta_{-i}$ is the estimate without observation $i$ and $P_{ii}$ its
statistical leverage, and computes
$\hat\theta_{KSS} = \hat\theta_{PI} - \sum_i B_{ii}\hat\sigma_i^2$ on a centered
outcome. Two features of this estimator matter for computation. It requires
$P_{ii} < 1$ for every observation, so the sample has to be pruned first (below).
And computing $P_{ii}$ and $B_{ii}$ exactly needs a linear solve for every
observation, which is infeasible for millions of them: KSS approximate them by
random projection, and so does this package ([how the correction
streams](streaming.md)).

## What one call does

One call does three things, in an order that is not negotiable:

1. **Prunes to the leave-one-out connected set.** The correction needs every
   effect to stay estimable when any one match is dropped, which fails at
   workers who are the only link between two parts of the worker–firm network.
   They are removed, and the pass repeated until none is left, since removing
   one can make another; workers observed only once are dropped as well. This
   is what LeaveOutTwoWay and xhdfe do, and it matches xhdfe row for row.
2. **Fits on what survives.**
3. **Approximates each match's leverage** by random projection, and from those
   computes the bias.

**Pruning changes the estimation sample, which is why it comes first.** On the
simulated panel it costs 0.1% of rows; in KSS's own application it removes about
half the firms. A decomposition computed from a fit made *before* pruning would
be describing a different model, so there is no "fit then decompose it" path
that skips this. `prune=False` checks instead of pruning, and fails if the panel
is not already connected, for callers who pruned upstream.

The pruning is the fit option `connected="leave_one_out"`, so it runs on the
rows the model can use, and the fit's `sample()` labels the rows it drops
`"leave_one_out"` (`lo.fit.sample()`); `lo.pruning` has the counts. To find the
set ahead of time, or to fit other things on it,
`connected_components(data, "worker_id + firm_id", leave_one_out=True)` returns
it as a table of the levels kept, and its `.largest()` is the pruned panel (see
[connected sets](../examples/connected_sets.md)).

`psi=` names the non-streamed dimension whose variance is wanted (the firms);
the streamed dimension plays the part of alpha (the workers). Both default to
the first two fixed effects in the formula. Extra options are passed to the fit,
so `weights=` and the memory knobs remain available. Several outcomes at once
are not supported: fit them separately.

## What is left out: a match, or an observation

By default the correction leaves out a whole **worker–firm match** — every year
of a spell at once — as LeaveOutTwoWay, VarianceComponentsHDFE.jl and xhdfe all
do. Earnings errors are correlated within a spell; leaving out one year treats
the others as independent of it, and the leave-out variance estimate is then
biased. In a Monte Carlo with half the error variance a spell-level shock, the
observation-level estimates of var(psi) and the covariance were off by 6% and
15% (t = 7 and 12), and the match-level ones by nothing measurable.

It is computed the reference's way: partial out everything but the two effects,
collapse to one row per match weighted by its length, and run the correction on
that weighted regression — whose leverages are the matches'. The components are
still person-year moments, divided by person-years − 1.

```python
lo = leave_out_kss(fml, data, workdir="scratch")                            # a match
lo = leave_out_kss(fml, data, workdir="scratch", leave_out="observation")   # one year
```

What differs between them:

| | `leave_out="match"` (default) | `leave_out="observation"` |
|---|---|---|
| robust to errors correlated within a spell | yes | no |
| a stayer's variance | its own years' residuals, averaged over the spell (LeaveOutTwoWay's `sigma_for_stayers`) | its own leave-out estimate, like any other row (`stayers=` to change) |
| standard errors (`se=True`) | var(psi) and cov, following `leave_out_COMPLETE`'s beta match-level option ([details](inference.md#standard-errors-when-leaving-out-a-match)) | all three components |
| q = 1 interval under weak identification | var(psi) | all three components |
| fit needs `keep_intermediates=True` | no | yes |

Match-level standard errors follow LeaveOutTwoWay's `leave_out_COMPLETE`, whose
match-level option is marked beta there. The differences are listed in
[standard errors when leaving out a match](inference.md#standard-errors-when-leaving-out-a-match).
One caution that no leave-out scheme removes: when a spell-level shock is
present, var(alpha) is biased at both levels, because a stayer's worker effect
cannot be told apart from its one spell's shock.

**`centering=`**, at match level only. The default, `centering="reference"`,
follows LeaveOutTwoWay: it centers the outcome after transforming by `√w`,
`√w·ȳ − mean(√w·ȳ)`. `centering="weighted"` is a **deviation from the
reference**: it centers `ȳ` at its weighted mean first, `√w·(ȳ − ȳ_w)`.

- *When it matters.* The two give the same result when spells are all the same
  length, and nearly the same when the outcome's mean is near zero. Both are
  unbiased.
- *Benefit.* Otherwise the reference's centering carries the outcome's *level*
  into `σ̂²`, and so depends on where the outcome is located. With log earnings
  (mean about 10), its estimates were 1.7–3 times as variable across simulated
  samples, and 5–70 times as variable across seeds at a fixed number of draws.
- *Cost.* The weighted centering does not reproduce LeaveOutTwoWay's or xhdfe's
  numbers on the same data. Keep the default when replicating their results.

See [methodological differences](differences.md) §3.4.

## If you want the regression too

`leave_out_kss` is a wrapper over three steps you can take yourself. Leaving
out a match reads only the fit's residual file, which every fit keeps; leaving
out an observation reads the sorted rows and cell tables a fit otherwise deletes
when it finishes, so that needs `keep_intermediates=True`:

```python
fit = feols_stream(fml, "data/*.parquet", workdir="scratch",
                   connected="leave_one_out")
lo  = fit.leave_out_kss(n_draws=250)
```

The two give identical numbers.

## Weights

`weights=` works, and then the estimand is the **weighted** variance
decomposition. Everything follows from that one choice:

- leverages are those of the square-root-weight hat matrix,
  `P_ii = w_i x_i' (A'WA)^- x_i` — of the two hat matrices a weighted fit
  admits, only that one is symmetric and idempotent, so only that one gives
  leverages;
- the plug-in variances and covariance are weighted, because the bias term is a
  bias *for those*;
- `sigma2` carries a factor of `w`, since the sandwich
  `S^- A'W Ω W A S^-` has two of them and the trace supplies one itself.

Unweighted, every one of those reduces to the plain KSS expression, and the
unweighted results are unchanged to floating point. The weighted path is
validated the same way the rest is: Monte Carlo against known effects in the
dense prototype, then streaming against the prototype.

## How it is checked

The correction is subtle and easy to get quietly wrong, so it is validated three
ways. A dense, exact implementation in
[`prototypes/`](https://github.com/leetucker/hdfe-stream/tree/main/prototypes)
is checked against **known truth** by Monte Carlo — the simulator draws the
worker and firm effects, so the estimand is known — confirming the plug-in
estimator is biased in the documented direction and the corrected one is not.
The streaming version is then checked against that prototype (agreeing to 3e-5
at 1024 draws), and independently against
[pytwoway](https://github.com/py-econometrics/pytwoway) on an identical sample
(plug-in exact, leave-out within 0.2%). The prototype's match-level estimator
reproduces [xhdfe](https://github.com/reisportela/xhdfe-xfe) — itself a port of
LeaveOutTwoWay — to machine precision at both levels, and its pruning matches
xhdfe's row for row. For match-level standard errors, a literal port of xhdfe's
code reproduces xhdfe's, and the package's own version is checked against the
dense kernel exactly (the quadratic form reproduces the point estimate to
10⁻¹⁰) and by Monte Carlo coverage.

[The example](../examples/leave_out_kss.md) runs the whole thing against the
effects the data was generated from, so you can watch the bias appear and be
removed.

```{toctree}
:hidden:

streaming
inference
draws
differences
```

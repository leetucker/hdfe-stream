# Leave-out variance components

The plug-in AKM decomposition is biased. Worker and firm effects are estimated
from a handful of observations each, and that estimation error inflates their
measured variance and attenuates the covariance between them, so the plug-in
numbers overstate how much of wage inequality is worker and firm heterogeneity,
and understate how much is sorting. [`leave_out_kss`](../reference/kss.md)
applies the bias correction of Kline, Saggio and Sølvsten (2020). This example
runs it on a simulated panel whose true effects are known, so you can watch the
bias appear and be removed. The [leave-out variance
components](../kss/index.md) section has the full story; this page follows the
script.

It is the slowest of the examples, because it runs the correction seven times.

## One call

```python
from hdfe_stream import leave_out_kss
from hdfe_stream.simulate import simulate_akm

simulate_akm(n_workers=8_000, seed=17, keep_effects=True).write_parquet("panel.parquet")
lo = leave_out_kss("log_earn ~ age_squared + age_cubed | worker_id + firm_id",
                   "panel.parquet", workdir="scratch", n_draws=250, seed=0)
print(lo.summary())
```

```text
leave-out variance components (firm_id and worker_id)
  67,913 observations in 13,404 matches, 9,491 of them movers'
  leaving out a match, 250 draws; max mover leverage 0.937202; mean sigma2 0.018147
  sample: 8,000 -> 7,993 worker_id, 533 -> 531 firm_id after pruning to the leave-one-out connected set

component                plug-in        bias   leave-out
--------------------------------------------------------
var(psi)                0.064284    0.001372    0.062912
var(alpha)              0.174041    0.007164    0.166877
cov(psi, alpha)         0.027284   -0.001314    0.028598
```

This prunes to the leave-one-out connected set, fits, and decomposes, in that
order: pruning changes the estimation sample, so it has to happen before the
fit. What is left out is a whole worker–firm match, every year of a spell, as
the reference implementations do. `keep_effects=True` makes the simulator return
the effects it drew, as `true_worker_effect` and `true_firm_effect`.

## Against the truth

The estimand is an observation-level moment of the true effects, taken over the
sample that survived pruning, so it is computed on `lo.fit.sample()`'s rows in
the sample:

```text
component              truth    plug-in  leave-out  plug-in err  leave-out err
------------------------------------------------------------------------------
var(psi)             0.05864    0.06428    0.06291      0.00564        0.00427
var(alpha)           0.15976    0.17404    0.16688      0.01428        0.00712
cov(psi, alpha)      0.03051    0.02728    0.02860      0.00323        0.00191
```

The plug-in errors run in the direction theory predicts: both variances too
high, the covariance too low. With 8.5 observations per worker the bias is
modest; it grows as that number falls, so a short panel needs the correction
more, not less.

## What pruning cost

Here pruning removed 7 workers (2 of them articulation points) and 2 firms,
59 rows, since the simulated panel is well connected. `lo.pruning` has the
counts, and `lo.fit.sample()` labels the rows it dropped `"leave_one_out"`. On
administrative data KSS report losing about half the firms.

## In steps, keeping the regression

`leave_out_kss` is a wrapper. If you want the regression in its own right, do
the steps yourself: [`connected_components`](connected_sets.md) finds the
leave-one-out connected set ahead of time, and the fit's own `leave_out_kss`
method gives the same components.

```python
from hdfe_stream import connected_components, feols_stream

panel = connected_components("panel.parquet", "worker_id + firm_id",
                             leave_one_out=True).largest()
fit = feols_stream(fml, panel, workdir="scratch", stream="worker_id")
step = fit.leave_out_kss(n_draws=250, seed=0)
```

## How much the approximation moves the answer

The leverages are approximated by random projection, so the answer carries
Monte Carlo noise, and the way to know how much is to vary the seed. Across four
seeds at 250 draws the spread was 0.2–1.4% of the estimates, small next to their
sampling uncertainty. [Draws and reproducibility](../kss/draws.md) measures it
properly and says what to report in a paper.

## Standard errors and weak identification

`se=True` adds standard errors and 95% intervals. The example leaves out an
observation here (`leave_out="observation"`), so that all three components get
one; leaving out a match reports them for var(psi) and the covariance. With
standard errors comes a check on whether the normal interval is justified at
all, which on this well-connected panel it is (`q = 0` for every component):

```text
component                plug-in        bias   leave-out         se              95% interval
---------------------------------------------------------------------------------------------
var(psi)                0.064284    0.003421    0.060863   0.001336      [0.058245, 0.063481]
var(alpha)              0.174041    0.013201    0.160841   0.001301      [0.158291, 0.163391]
cov(psi, alpha)         0.027284   -0.002529    0.029813   0.000462      [0.028908, 0.030718]
```

All three intervals contain the truth. The last part of the example builds a
panel with a bottleneck, two blocks of firms joined by six movers, where the
normal interval is not justified, and the output adds KSS's q = 1 interval,
which stays valid there. Both are explained in [standard errors and weak
identification](../kss/inference.md).

:::{dropdown} The full script: `examples/leave_out_kss.py`
```{literalinclude} ../../examples/leave_out_kss.py
:language: python
```
:::

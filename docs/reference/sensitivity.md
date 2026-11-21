# Sensitivity to parallel trends

An event study's pre-treatment coefficients show whether treated and control
units were trending alike before treatment. They cannot show that the trends
would have stayed alike afterwards, and a pre-trend test that does not reject
is weak evidence that they would. Rambachan and Roth (2023, *A More Credible
Approach to Parallel Trends*, Review of Economic Studies 90(5)) replace "the
trends are parallel" with "the trends differ after treatment by no more than
some multiple M of how much they differed before". `trend_sensitivity` gives
confidence intervals for a post-treatment effect that are valid under that
weaker assumption, for each value of M you ask about. The largest M at which
the interval still excludes zero (the *breakdown value*) says how much
departure from parallel trends the conclusion can take.

This is the method of the authors' R package
[HonestDiD](https://github.com/asheshrambachan/HonestDiD), with the same
restrictions, intervals and defaults, and it gives the same intervals. It is
much faster: the R package solves hundreds of thousands of small linear
programs for a relative-magnitudes analysis, building each one in R; here they
are solved in compiled code, each starting from the previous solution. A
relative-magnitudes interval takes about a hundred times less time than in R
(0.06 second instead of 6.6 with 4 pre-treatment periods, 0.45 instead of 51
with 32), and a smoothness FLCI a few hundredths of a second instead of 4 (see
[the benchmark](#speed)). It also uses next to no memory: a few megabytes,
where the R package's smoothness intervals use about a gigabyte.

## Use it

On a regression with leads and lags, named by `i()`:

```python
from hdfe_stream import feols_stream

fit = feols_stream("y ~ i(rel_time, ref=-1) | unit + year", data,
                   vcov={"CRV1": "unit"})
s = fit.trend_sensitivity(M=[0.5, 1, 1.5, 2])
s.table          # M, lower, upper: the robust interval at each M
s.original       # the usual interval under exact parallel trends
s.breakdown      # largest M in the grid at which the effect stays significant
s.plot()         # needs matplotlib
fit.plot_event_study()   # the event study itself, with its reference period
```

The event study is every coefficient named `<variable>::<event time>`, under
the fit's current vcov (`with_vcov` chooses another). The reference event time
(default `reference=-1`) is omitted from the regression, and the event times
are consecutive on each side of it. `keep=` selects the coefficients when the
fit has more than one such variable. The effect bounded is the one at event
time 0 by default; `target=` takes another event time, `"average"` (the mean
over all post-reference event times) or weights `{event time: weight}`.

From any event study's coefficients and covariance:

```python
from hdfe_stream.core.sensitivity import trend_sensitivity

s = trend_sensitivity(beta, vcov, n_pre=4, restriction="smoothness", M=[0, 0.01, 0.02])
```

`beta` lists the pre-treatment coefficients from the earliest to the one just
before the reference period, then the post-treatment ones; `target` is then a
vector of weights on the post-treatment coefficients (default: the first).

diff-stream's staggered-adoption estimators have the same method, which finds
the event study, its reference period and anticipation in the fit.

## Restrictions

`restriction=` says how the violation of parallel trends after treatment
(δ_post) may relate to the one before it (δ_pre, which the pre-treatment
coefficients estimate).

| `restriction` | R | The violation after treatment | M = 0 means |
|---|---|---|---|
| `"relative_magnitudes"` (default) | Δ^RM, `createSensitivityResults_relativeMagnitudes` | no change from one period to the next larger than M times the largest such change before treatment | exact parallel trends after treatment |
| `"smoothness"` | Δ^SD, `createSensitivityResults` | its slope changes by at most M between consecutive periods (M is in the outcome's units) | a linear trend, extrapolated from before treatment |
| `"smoothness_relative_magnitudes"` | Δ^SDRM, `bound = "deviation from linear trend"` | changes in slope no larger than M times the largest one before treatment | a linear trend |

Either of two shape restrictions can be added:

- `bias="positive"` or `"negative"`: the violation after treatment has this sign
  (R's Δ^…PB, Δ^…NB).
- `monotonicity="increasing"` or `"decreasing"`: the violation is monotone over
  every period (R's Δ^…I, Δ^…D).

The default values of M are R's: for smoothness, ten values from 0 to an upper
bound on the largest second difference of the pre-treatment coefficients; for
the relative-magnitudes restrictions, ten values from 0 to 2. Choose M from
what you know about the setting. The R package's vignette discusses how.

## Intervals

| `ci_method` | R | Available for | Default for |
|---|---|---|---|
| `"fixed_length"` | FLCI | smoothness | smoothness |
| `"conditional"` | Conditional | all | |
| `"hybrid_fixed_length"` | C-F | smoothness | smoothness with a sign or monotonicity restriction |
| `"hybrid_least_favorable"` | C-LF | all | the relative-magnitudes restrictions |

The fixed-length interval (FLCI) is an affine estimator plus and minus the
half-length that covers the worst-case bias. It is short when the restriction
is smoothness, but ignores sign and monotonicity restrictions (you get a
warning). It is not consistent for the relative-magnitudes restrictions,
which therefore allow only the other two.
The conditional test of Andrews, Roth and Pakes (2023) is inverted over a grid
of 1,000 values of the effect (`grid_points=`, `grid_bounds=`). The hybrids
add a first-stage test of size α/10: the FLCI's, or the least-favorable
critical value's, which is simulated (`simulations=1000`, `seed=0`).

## Agreement with R

The tests compare every restriction, sign and monotonicity variant and
interval with HonestDiD 0.2.8, on the package's two example event studies and
on simulated event studies with one to five pre-treatment and one to four
post-treatment periods, at every point of R's grid. They use R's own
least-favorable critical values, since the simulated draws differ between the
languages. The differences that remain are deliberate:

- **The folded-normal quantile is exact.** The FLCI's half-length is a quantile
  of |N(t, 1)|, which R estimates from 10⁶ simulated draws. Computed exactly,
  the half-length differs from R's by up to about 0.1%.
- **The least-favorable critical value uses NumPy's random numbers.** It takes
  the same number of draws as R (1,000) from a fixed seed, so it is
  reproducible, but differs from R's by simulation error. That can move a
  hybrid interval's end by a grid step.
- **The FLCI is optimized exactly.** The optimal estimator's worst-case bias
  and variance trade off along a path that is computed exactly (it is the
  solution path of a lasso), and the half-length is minimized along it. R
  bisects on a numerical derivative and, when that fails, takes the best of
  100 grid points; the interval here is then never longer than R's.
- **The grid reaches the interval.** R centers its grid for the
  relative-magnitudes restrictions on zero, ±20 standard errors wide. When the
  accepted set reaches an end of the grid, R reports the grid's end with a
  warning; here the grid is extended at the same spacing until it closes.
  When the grid misses the interval altogether (an effect more than about 20
  standard errors from zero), R reports an empty interval; here the same grid
  is centered on the estimate. Neither case arises when R's interval is
  closed.
- **Monotonicity with one pre-treatment period.** R's constraint matrix for
  this case requires δ₋₁ ≤ δ₁ where the restriction is δ₋₁ ≤ 0 (an indexing
  slip in `.create_A_M`). This package uses the restriction as stated. With two
  or more pre-treatment periods the two agree.
- **No errors where R stops.** For a few event studies, R's conditional test
  stops with "system is exactly singular" at some grid point (its closed form
  for the truncation bounds needs a matrix that is singular there). Here those
  points use the dual computation of the bounds that R uses for degenerate
  solutions.
- **An empty interval is NaN**, with a warning, where R reports `Inf` and `-Inf`.

## Speed

Seconds for one value of M (the default grid of 1,000 points), on simulated
event studies with 4 post-treatment periods, on one laptop (8 cores; R 4.5.3,
HonestDiD 0.2.8, diff-diff 4.0.0). diff-diff is shown for the FLCI only: its
relative-magnitudes interval is not the method's (it widens the estimated
identified set by z standard errors, its conditional test being disabled), so
its time is not comparable. `benchmarks/sensitivity_benchmark.py` reproduces
the table.

```{include} ../benchmarks/sensitivity_table.md
```

The work is linear in the number of pre-treatment periods for the
relative-magnitudes restrictions (one polyhedron per pre-treatment period and
sign). The polyhedra and the values of M run in parallel (`n_threads=`), which
pays off for a grid of M values more than for one. Memory is negligible:
everything is of the size of the event study.

## Reference

```{eval-rst}
.. currentmodule:: hdfe_stream.core.sensitivity

.. autofunction:: trend_sensitivity

.. autofunction:: event_study_sensitivity

.. autofunction:: event_study_inputs

.. autofunction:: default_M

.. autoclass:: TrendSensitivityResult
   :no-members:

   .. autoproperty:: breakdown
   .. autoproperty:: M
   .. automethod:: plot
   .. automethod:: summary_text
   .. automethod:: summary_dict

.. autoclass:: TrendSensitivityWarning

.. currentmodule:: hdfe_stream

.. automethod:: HDFEResult.trend_sensitivity
   :no-index:

.. automethod:: HDFEResult.plot_event_study
   :no-index:

.. autofunction:: hdfe_stream.core.plots.event_study_plot
```

# Weights and IV

## Weights

`weights=` names a column (or, in the [low-level
interface](../reference/lowlevel.md), takes a Polars expression) of strictly
positive weights; rows with a missing weight are dropped. There are two kinds,
as in pyfixest, and the difference is not cosmetic:

`weights_type="aweights"` (the default)
: analytic or precision weights. N stays the number of rows.

`weights_type="fweights"`
: frequency weights: each row stands for `w` identical observations, so N is
  the sum of the weights, and every degrees-of-freedom correction changes with
  it. The fit, N and every vcov equal those of the data with each row repeated.

```python
analytic = feols_stream(fml, data, workdir="scratch", weights="wa")
frequency = feols_stream(fml, data, workdir="scratch", weights="wf",
                         weights_type="fweights")
```

```text
model                                N   age_squared          se
----------------------------------------------------------------
unweighted                     169,833      0.081721    0.013373
aweights (wa)                  169,833      0.077746    0.013346
fweights (wf)                  423,811      0.080056    0.008126
```

Frequency weights inflate N to the weight total, which is why their standard
errors are smaller: the fit is told there are more observations.

## Two-stage least squares

An IV formula has four parts:
`outcome ~ exogenous | fixed effects | endogenous ~ instruments`. In the
simulated data `x2` is correlated with an unobservable that also enters `y_iv`,
so OLS on `x2` is biased upward; `z1` and `z2` are excluded instruments, and the
true coefficient on `x2` is 0.2:

```python
iv = feols_stream("y_iv ~ age_squared | worker_id + firm_id | x2 ~ z1 + z2",
                  data, workdir="scratch", vcov={"CRV1": "worker_id"})
```

```text
true coefficient on x2:   0.200000
OLS  (biased):            0.543949
2SLS (consistent):        0.196701
```

The result carries its first-stage regressions, each a fitted model in its own
right (`iv.first_stage[0].tidy()`), and `iv.f_stat_1st_stage`, the Wald
statistic on the excluded instruments computed under the same vcov as the
model: the usual weak-instrument diagnostic. It is 33,476 here, against a
conventional rule of thumb of 10. Weights and IV combine.

:::{dropdown} The full script: `examples/weights_and_iv.py`
```{literalinclude} ../../examples/weights_and_iv.py
:language: python
```
:::

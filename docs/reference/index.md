# Function reference

Everything here is importable from `hdfe_stream`, except the simulators, which
are in `hdfe_stream.simulate`.

| | |
|---|---|
| **[Estimation](estimation.md)** | |
| `feols_stream` | OLS, weighted least squares and 2SLS with high-dimensional fixed effects, from a formula |
| `fepois_stream` | Poisson regression, from a formula |
| `feglm_stream` | logit and probit, from a formula |
| **[Results](results.md)** | |
| `HDFEResult` | one model: coefficients, standard errors, fixed effects, residuals, the sample |
| `HDFEMulti` | a formula that expands to several models |
| `HDFEResult.predict`, `.confint`, `.wald_test`, `.wildboottest` | predictions on new data, confidence intervals (pointwise or simultaneous), Wald tests, the wild cluster bootstrap |
| **[Leave-out variance components](kss.md)** | |
| `leave_out_kss` | the Kline–Saggio–Sølvsten decomposition, end to end |
| `LeaveOutComponents`, `ComponentSE`, `WeakIdDiagnostics` | its result, standard errors and weak-identification diagnostic |
| **[Connected sets](connected.md)** | |
| `connected_components` | the connected components of two fixed effects, `min_degree` pruning and the leave-one-out connected set |
| `ConnectedComponents` | its result |
| **[Low-level interface](lowlevel.md)** | |
| `StreamingHDFE` | the estimator, with column names or Polars expressions instead of a formula |
| `StreamingGLM` | the same for Poisson, logit and probit |
| **[Utilities](utilities.md)** | |
| `etable` | regression tables, through pyfixest |
| `bonferroni`, `holm`, `rwolf`, `wyoung` | multiple-testing corrections across models; the last two from the wild cluster bootstrap |
| `cleanup` | remove run directories left by killed jobs |
| `simulate_akm`, `simulate_rich`, `simulate_trends`, `simulate_bottleneck` | simulated panels with known effects |

```{toctree}
:hidden:

estimation
results
kss
connected
lowlevel
utilities
```

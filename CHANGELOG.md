# Changelog

## 0.1.0

First release.

- `feols_stream`: out-of-core OLS with several high-dimensional fixed effects,
  varying slopes, weights, IV, and CRV1/CRV3 and other variance estimators.
- `fepois_stream` and `feglm_stream`: Poisson, logit and probit by iteratively
  reweighted least squares over the same passes.
- AKM variance decomposition and leave-out (KSS) bias correction with
  standard errors and weak-identification diagnostics.
- Reporting through pyfixest (`to_pyfixest()`, `etable`).

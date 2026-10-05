# Changelog

## Unreleased

- **Polars 1.41 is excluded** from the supported versions. Its streaming join
  panics (`called Option::unwrap() on a None value`) once a frame has about
  130 columns, which a model with more than about 120 covariates reaches in
  pass 0. Polars 1.40 and earlier, and 1.42 and later, are unaffected.
- **A panic inside Polars is reported as a `RuntimeError`** that names the
  Polars version, says it is a Polars bug, and suggests a fix, instead of a
  bare `PanicException`.

## 0.2.0

Changes to defaults, so results can differ from 0.1.0 without any change to
your code.

- **Default `vcov` is now `"iid"`**, with or without fixed effects, matching
  pyfixest 0.60. Before, a model with fixed effects defaulted to CRV1 clustered
  on the first fixed effect. Pass `vcov={"CRV1": "worker_id"}` for the old
  behavior.
- **Singleton observations are dropped by default** (`fixef_rm="singleton"`),
  as in pyfixest, repeating until none is left. `fixef_rm="none"` keeps them.
  `leave_out_kss` fits with `fixef_rm="none"`, since its pruning defines the
  sample.

New features.

- **`row_id` in `resid()`**: each row carries its position in the source before
  any row was dropped, so residuals join back onto the source (`row_id=` renames
  the column).
- **`fit.sample()`** returns the source, lazily, with `row_id`, `in_sample` and
  `dropped_because` (`"missing"`, `"singleton"` or `"separation"`).
  `multi.sample()` returns `{formula: frame}`.
- **Polars `DataFrame` inputs** are accepted wherever a Parquet path or
  `LazyFrame` was.
- **Reserved column prefix `__hdfe_`.** All of the estimator's internal columns
  now use it, and a source with a column starting with it is refused. This fixes
  fits failing when a user column was named `w` (with weights), `v0`, `c1`,
  `gcode` and similar.

## 0.1.0

First release.

- `feols_stream`: out-of-core OLS with several high-dimensional fixed effects,
  varying slopes, weights, IV, and CRV1/CRV3 and other variance estimators.
- `fepois_stream` and `feglm_stream`: Poisson, logit and probit by iteratively
  reweighted least squares over the same passes.
- AKM variance decomposition and leave-out (KSS) bias correction with
  standard errors and weak-identification diagnostics.
- Reporting through pyfixest (`to_pyfixest()`, `etable`).
- `summary_json()` and `summary_dict()`: the model-level information of
  `summary()` as JSON or a dict, for saving to disk.

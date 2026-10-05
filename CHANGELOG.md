# Changelog

## Unreleased

- **`fit.sample()` labels every singleton.** With three or more fixed effects
  that each had singleton levels, rows dropped only for the third or a later
  dimension came back with `in_sample=True`, so the counts disagreed with
  `n_obs` and the summary's singleton count. The fit itself was right. GLM
  `"separation"` labels had the same fault.
- **A NaN in a fixed-effect or cluster column is missing**, as a null is,
  rather than a level of its own. For fixed effects this matches pyfixest;
  for clusters, where pyfixest raises an error, the row is dropped as it is
  for a null cluster. `sample()` reports these rows as `"missing"`.

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

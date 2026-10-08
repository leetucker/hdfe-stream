# Changelog

## 0.2.2 (in progress)

- **Polars 2.0 support.** `fit.sample()` returns the source's rows in the source's
  order again. Polars 2.0 collects with the streaming engine by default, whose
  joins do not keep row order unless asked, so the rows came back shuffled.
- **Dropping singletons no longer inflates memory with a wide design.** The
  dropped levels were anti-joined after the design was evaluated, so the
  streaming join buffered every design column; with 503 indicators that took
  the first scan to about 14 GB under Polars 2.0. The rows are now dropped from
  the source columns before the design is evaluated, and the peak is back to
  about 8.6 GB. The rows dropped are the same.
- **Sorting the buckets takes less memory under Polars 2.0.** Each bucket is
  sorted with Polars' in-memory engine and then written, rather than sunk
  through the streaming engine, whose sort peaked about 1.5x higher. With a
  narrow design in one bucket (8 covariates, 8.5 million rows) the peak falls
  from 4.6 GB to 3.3 GB; the sort is also faster with a wide design.
- **Empty run directories no longer pile up on NFS.** A file deleted while
  still open or memory-mapped becomes a hidden .nfs file on NFS, so its run
  directory could not be removed and was left empty once the process exited.
  A failed removal is now retried after collecting garbage on every platform
  (it was Windows only); the leave-out functions release their memory maps as
  soon as they are done (`reload_intermediates()` maps them back in); and
  empty run directories more than a minute old are removed by every new fit
  in the same working directory and by `hdfe_stream.cleanup()`.
- **The explicit solver builds its matrix faster and in less memory.** The
  reduced matrix S is now written directly in compressed form, one row at a
  time, instead of being assembled from chunks of coordinate triples that
  were merged with scipy. Each row's size is counted first, so S is allocated
  once and nothing larger than S is held while it is built; `max_s_gb` is
  checked before S is allocated rather than partway through. Estimates are
  unchanged up to rounding.
- **Removed `triple_budget` and `dense_max_levels`** from `StreamingHDFE`. They
  tuned the old way of building S and have no counterpart in the new one.
- **Finding singletons takes fewer scans.** The search used to stop only
  after a full round over the fixed effects found nothing. It now stops once
  every fixed effect has been checked since the last drop, since a dimension's
  own drop cannot leave a singleton in it: four scans instead of six for a
  typical three-way model, about 40% less time spent on singletons, and the
  same memory. The rows dropped are the same.

## 0.2.1

- **`fe_dof="exact"` no longer depends on the order of the fixed effects.**
  With three or more fixed effects it counted the connected components of the
  streamed dimension and whichever came next in the formula, so reordering the
  formula could change the standard errors. It now counts the components of
  every pair of dimensions and uses the pair with the most, which is never
  more than the true number of redundant levels. With two fixed effects
  nothing changes, except in the clustered case below.
- **Clustered standard errors with fixed effects nested in the clusters.**
  Under `fe_dof="exact"`, the components of a pair of fixed effects that are
  both nested within the clusters were subtracted from K, although those
  fixed effects' levels are already dropped from it. K could then fall below
  the number of covariates and the standard errors were too small. Such a pair
  is no longer used, and with no other pair K is pyfixest's.
- **`fit.sample()` labels every singleton.** With three or more fixed effects
  that each had singleton levels, rows dropped only for the third or a later
  dimension came back with `in_sample=True`, so the counts disagreed with
  `n_obs` and the summary's singleton count. The fit itself was right. GLM
  `"separation"` labels had the same fault.
- **A NaN in a fixed-effect or cluster column is missing**, as a null is,
  rather than a level of its own. For fixed effects this matches pyfixest;
  for clusters, where pyfixest raises an error, the row is dropped as it is
  for a null cluster. `sample()` reports these rows as `"missing"`.
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

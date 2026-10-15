# Changelog

## 0.3.0

New features.

- **`connected_components`** finds the connected components of the graph of
  two fixed-effect dimensions (either may be an interaction, `firm_id^year`)
  by union-find over the rows as they stream, holding level-sized arrays
  only. It returns a table per dimension giving each level its component,
  numbered by size so that component 0 is the largest (ties go to the
  component with the smallest level of the first dimension), and a table of
  component sizes. `largest()` filters the data to the largest connected
  set and `label()` adds each row's component, both lazily (on the data the
  components were found in, or on other data passed to them). `min_degree`
  removes levels with fewer distinct partners in the other dimension,
  repeating until none is left, before the components are found: `(2, 1)`
  keeps only movers, `2` the 2-core. `leave_one_out=True` prunes to the
  leave-one-out connected set of KSS estimation instead. `rank_by` says
  whether "largest" counts rows (the default) or the levels of one dimension.
- **`connected=`, `min_degree=` and `connected_fe=`** in `feols_stream`,
  `fepois_stream`, `feglm_stream` and `StreamingHDFE` restrict the fit to a
  pair of fixed effects (`connected_fe`, by default the first two listed),
  among the rows with no missing values and before singletons are dropped:
  `min_degree` prunes levels with too few distinct partners, then
  `connected="largest"` keeps the largest connected set of what is left, or
  `connected="leave_one_out"` the leave-one-out connected set, as
  `connected_components` does. `sample()` reports the rows they drop as
  `"min_degree"`, `"disconnected"` and `"leave_one_out"`, and
  `diagnostics["connected"]` and the summary count them.
- **Python 3.15** is supported and tested in CI. pyfixest, which the
  `formula` extra needs, has no 3.15 wheels yet (0.60.0), and building it
  from source needs Rust and `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1`; the
  rest of the package installs from wheels.

Changes that can affect existing code.

- **`leave_one_out_connected` and `LeaveOutSet` are removed.** Use
  `connected_components(data, "worker_id + firm_id", leave_one_out=True)`,
  whose `.largest()` is the pruned panel, or `connected="leave_one_out"`
  in a fit. The set is the same: on simulated panels with many components,
  workers observed once and several rounds of cut workers, the workers and
  firms kept are identical. It is found 5-6 times faster (38s against 218s on
  150 million rows), and a null worker or firm id no longer makes it fail.
- **`leave_out_kss` prunes in the fit**, with `connected="leave_one_out"`, so
  the leave-one-out connected set is that of the rows the model can use:
  before, rows with a missing outcome or covariate were part of the network
  being pruned. `lo.pruning` is now a dict of the pruning's counts (the
  fit's `diagnostics["connected"]["leave_one_out"]`), and the rows it dropped
  are those `lo.fit.sample()` labels `"leave_one_out"`.
- **The `component` column of `fixef()` is numbered by size**, so component
  0 is the largest (by rows in the estimation sample), as in
  `connected_components`. Ties go to the component with the smallest level of
  that dimension. It was numbered by each component's smallest level.
- **Removed `hdfe_stream.iter_group_chunks` and `StreamingHDFE.from_formula`.**
  The first read only the fit's internal row files, sorted by its internal
  group code, and stays internal; the second duplicated `feols_stream`.

Fixes.

- **The leave-out operator's null space is right with several connected
  components.** `apply_inverse` built one null vector for every number up to
  the largest component label, which with several components left gaps, and
  each gap added a spurious vector. Leave-out estimation runs on a connected
  set and was not affected.

Documentation.

- **New example, `connected_sets.py`**, and additions to the others:
  `sample()` and joining the residuals back on `row_id`, `fe_dof`, and CRV3
  (`formulas.py`); `summary_json` (`quickstart.py`); `keep=` and a
  decomposition by group (`akm_variance.py`); and `StreamingGLM` (`glm.py`).

## 0.2.2

- **Polars 2.0 support.** `fit.sample()` returns the source's rows in the source's
  order again. Polars 2.0 collects with the streaming engine by default, whose
  joins do not keep row order unless asked, so the rows came back shuffled.
  Under Polars 2.0, repeated fits of the same model can differ in the last
  digit or so (around 1e-16 relative): its streaming engine adds floating-point
  sums in an order that depends on thread scheduling. Under Polars 1.x they
  are identical bit for bit.
- **Dropping singletons no longer inflates memory with a wide design.** The
  dropped levels were anti-joined after the design was evaluated, so the
  streaming join buffered every design column. The rows are now dropped from
  the source columns before the design is evaluated.
- **Finding singletons takes fewer scans.** The search used to stop only
  after a full round over the fixed effects found nothing. It now stops once
  every fixed effect has been checked since the last drop, since a dimension's
  own drop cannot leave a singleton in it: four scans instead of six for a
  typical three-way model, about 40% less time spent on singletons, and the
  same memory. The rows dropped are the same.
- **Sorting the buckets and building the cell table takes less memory under
  Polars 2.0.** Each bucket is sorted with Polars' in-memory engine and then
  written, rather than sunk through the streaming engine, whose sort peaked
  about 1.5x higher. Similarly, Pass 1 grouped each bucket with a streaming
  `group_by`, which under Polars 2.0 gives every thread its own hash table: with
  as many cells as rows, as in an AKM panel, the fit's peak more than doubled.
  The rows are sorted by fe[0] group, so each bucket is now grouped a slice of
  whole groups at a time and the cells appended in order. This reduces memory
  use in AKM specifications and leaves wide designs unchanged or a little
  better.
- **The explicit solver builds its matrix faster and in less memory.** The
  reduced matrix S is now written directly in compressed form, one row at a
  time, instead of being assembled from chunks of coordinate triples that
  were merged with scipy. Each row's size is counted first, so S is allocated
  once and nothing larger than S is held while it is built; `max_s_gb` is
  checked before S is allocated rather than partway through. Estimates are
  unchanged up to rounding.
- **Removed `triple_budget` and `dense_max_levels`** from `StreamingHDFE`. They
  tuned the old way of building S and have no counterpart in the new one.
- **Importing hdfe_stream costs a little less memory.** p-values, critical
  values and the weak-identification interval use `scipy.special`'s
  distribution functions instead of `scipy.stats`, which is no longer
  imported. The values are the same.
- **Empty run directories no longer pile up on NFS.** A file deleted while
  still open or memory-mapped becomes a hidden .nfs file on NFS, so its run
  directory could not be removed and was left empty once the process exited.
  A failed removal is now retried after collecting garbage on every platform
  (it was Windows only); the leave-out functions release their memory maps as
  soon as they are done (`reload_intermediates()` maps them back in); and
  empty run directories more than a minute old are removed by every new fit
  in the same working directory and by `hdfe_stream.cleanup()`.
- **Slightly less verbose default logging.** CG iteration messages are now
  logged at DEBUG level and therefore not printed by default. Lower your log
  level to obtain this information.


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

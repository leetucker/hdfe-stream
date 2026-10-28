# Changelog

## Unreleased: 0.6.0

A supported API for packages built on `hdfe_stream.core` (logit-stream is
the first), and a row store for them. Nothing changes for code that imports
from `hdfe_stream` itself, and no estimate changes.

- **Stable names in `hdfe_stream.core`**, listed in its docstring and pinned
  by a test: they change only in a minor release, with a changelog entry.
  Nine helpers lose their leading underscore on the way: `core.formula`'s
  `compile_design`, `categorical_vars`, `numeric_expr` and
  `pyfixest_formula_api`; `core.feterms.parse_fe`; `core.report.warn`;
  `core.utils.as_lazy`; `core.workspace.Run`; and
  `core.results_base.vcov_key`. The type aliases move from `core._types` to
  `core.typing`. The old names are gone (they were internal).
- **`hdfe_stream.core.store`** (new): a block store of rows (`BlockStore`)
  for estimators that read every row in each iteration, with what pass 0
  needs to fill it -- the narrow copy of the sample (`write_sample`,
  `iter_batches`), level tables (`factorize`), frames of whole groups,
  sorted in buckets when the groups are not consecutive (`iter_groups`), and
  Polars queries on the CPU or the GPU engine with fallback
  (`PolarsEngine`). It is logit-stream's pass 0, moved here; hdfe-stream's
  own estimators do not use it. `BlockStore(binary=mask)` stores the
  design's 0/1 columns as one byte each and puts X back together, float64,
  as each block is read (`iterate(raw=True)` yields them apart, for a
  consumer that puts X together elsewhere, as on a GPU).
- **Results record each vcov's small-sample factor**
  (`diagnostics["vcov_factors"]`, by vcov key), for a caller that converts
  to other conventions (logit-stream's multinomial logit with several
  fixed effects, fitted by `fepois_stream`).
- **Faster row reads.** Stacking a chunk's columns into the row-major
  design (`_stack`, every pass of every model) is a blocked parallel copy:
  0.04 s instead of 0.47 s for 2 million rows of 39 columns. An ACS logit of
  3 million rows without fixed effects went from 16.7 s to 11.4 s from this
  alone.
- **GLMs (IRLS): BLAS on one thread, smaller chunks.** OpenBLAS's threads and
  numba's contended in the row passes, the more the smaller the chunks; up
  to 128 columns the passes now hold BLAS to one thread
  (`core.utils.single_blas`, new dependency threadpoolctl), which is faster
  at every chunk size. GLMs then default to chunks of about 64 MB of the
  design (`batch_rows=None`, between 100,000 and 2,000,000 rows; linear
  models keep 2,000,000). Estimates are unchanged to rounding.
- **Buckets sized by the rows' width.** Pass 0 sorts each bucket in memory;
  `rows_per_bucket` now defaults to about 384 MB of rows at their width
  (500,000 to 20,000,000 rows) instead of 20,000,000 rows. An ACS logit of 3
  million rows and 41 columns with PUMA fixed effects: peak 3.3 GB -> about
  1.7 GB, slightly faster. The sort itself is unchanged.
- Linear models without fixed effects also default to 64 MB chunks: on 3
  million rows of 40 columns, peak memory 3.9 GB -> 2.1 GB at the same time.
- **Log-likelihoods and information criteria.** Results have `aic` and
  `bic` (the coefficients plus the fixed effects' degrees of freedom, as
  fixest counts them), OLS results the normal `loglik` (with analytic
  weights, sum(log w) / 2 added, as statsmodels' WLS), GLM results
  `loglik_null` (constant only), and GLMs with fixed effects
  `loglik_fe_null` and `pseudo_r2_within` (fixest's within pseudo R2: what
  the covariates add beyond the fixed effects). With one fixed effect and no
  offset the fixed-effects-only model is the groups' means, computed in the
  first row pass; otherwise it is a fit of its own, made with
  `within_pseudo_r2=True`.
- **GLMs rescale analytic weights to average 1** over the estimation sample
  (Stata's convention for maximum likelihood): before, the iid vcov and the
  log-likelihood depended on the weights' scale (weights x 10: iid standard
  errors / sqrt(10)). Coefficients and robust/clustered vcovs are unchanged;
  `fepois` iid standard errors with analytic weights now differ from
  pyfixest's, which uses the weights as given.
- **Incidental-parameter bias correction** for logit and probit with one or
  two fixed effects (`bias_correction=True`, or `{"time": ..., "lags": L}`
  for predetermined regressors): the analytic correction of Fernández-Val
  and Weidner (2016), as R's alpaca computes it (`biasCorr`), matching it to
  1e-7 with the same standard errors; the fixed effects are then
  re-estimated with the corrected coefficients held (from the first fit's,
  or from zero if that fails, as it can when the correction is large, in
  very short panels). Checks the family, the
  number of fixed effects, and with lags one row per unit and period and an
  orderable period; warns for very short panels, gaps and units no longer
  than the window. Not for Poisson (no such bias with one or two fixed
  effects), nor three-way models.

## 0.5.1

Internal restructuring. Nothing changes for code that imports from
`hdfe_stream` itself, the documented `hdfe_stream.simulate` and
`hdfe_stream.leaveout` paths included, and no estimate changes.

- **`hdfe_stream.core`**: the machinery the estimators share now lives in its
  own subpackage, for other estimators built on the same streaming passes to
  import without the leave-out (KSS) estimators. It holds the passes, the
  solve, inference, the numba kernels, formula parsing, results and
  reporting. Its names are internal, not a supported API.
- `StreamingHDFE` is the core driver (`core/estimator.py`) with the leave-out
  mixins added. `StreamingGLM`'s weighted least-squares steps, which do not
  depend on the GLM family, are in `core/irls.py`. Per-group kernels that do
  not depend on the model have moved from `kernels_inverse.py` to
  `core/kernels_groups.py`.
- `HDFEResult` and `HDFEMulti` derive from new base classes
  (`core.results_base.RegressionResult` and `RegressionMulti`). These hold
  what follows from a vector of estimates and its vcov alone: `tidy`,
  `confint`, `wald_test`, the summaries, the pyfixest view behind `etable`,
  and the multiple-testing corrections. Output is unchanged. `repr()` of a
  result lists its fields in a new order, the base class's first.
- The modules that moved were undocumented, and their old paths no longer
  import: `hdfe_stream.results`, `reporting`, `connected`, `multcomp`,
  `workspace`, `families`, `utils`, `gpu`, `wildboot`, `formula`, `feterms`,
  `passes`, `solve`, `inference`, `report`, `_columns`, `_types` and the
  `kernels_*` modules other than `kernels_inverse` and `kernels_se`. Import
  public names from `hdfe_stream` (`hdfe_stream.HDFEResult`,
  `hdfe_stream.connected_components`, `hdfe_stream.cleanup`, ...).
- The numba kernels are cached under their new paths, so the first fit after
  upgrading compiles them once more.

## 0.5.0

- **Experimental GPU options.** `device="cuda"` runs the solve on an NVIDIA
  GPU through CuPy, and `polars_engine="gpu"` runs the fit's Polars queries on
  the cudf-polars GPU engine, with any query it cannot run run on the CPU and
  listed in `diagnostics["polars_gpu_fallbacks"]`. Neither changes the
  estimates beyond the solver tolerance. On an NVIDIA H100 they gained about
  10% at 5 million workers and nothing consistent below; see the new "CPU and
  GPU" benchmark, which also measures pyfixest's and xhdfe's GPU paths.
- The regression and many-covariates benchmarks also compare with R's fixest
  and Julia's FixedEffectModels.jl. The AKM figures no longer draw
  `solver="within"`, whose results stay in `akm.csv`.
- The choice of Polars engine is now always explicitly specified in the
  codebase. This fixes an issue where the in-memory engine would be used in
  Polars 1.x.x despite the streaming engine being used in Polars 2.x.x.

## 0.4.0

New features.

- **`HDFEResult.predict()`** predicts on new data (a DataFrame, a LazyFrame, or
  a Parquet path or glob) or on the fit's source, lazily, so the data can be
  larger than memory. It returns the data with a prediction column: the linear
  predictor, or with `type="response"` a GLM's mean. Rows with a fixed-effect
  level the fit did not estimate get null. It supports `i()` covariates,
  interacted fixed effects, varying slopes, and IV models.
- **`HDFEResult.confint()`**: confidence intervals, pointwise or, with
  `joint=True`, simultaneous over the selected coefficients, as pyfixest's.
- **`HDFEResult.wald_test()`**: Wald tests of R beta = q, or that the
  coefficients selected by `keep`/`drop` are zero, under the current vcov, with
  F or chi-squared p-values. F is reported by default for any OLS hypothesis
  (chi-squared for GLMs), with `df_t` denominator degrees of freedom, so a
  one-coefficient test has `tidy()`'s p-value.
- **`bonferroni()` and `holm()`**: multiple-testing corrections of one or
  more coefficients' p-values across models (results, `HDFEMulti`, or lists
  of them). Holm's step-down adjustment controls the familywise error rate as
  Bonferroni's does, and never rejects less often.
- **The wild cluster bootstrap: `bootcluster=`, `HDFEResult.wildboottest()`,
  `rwolf()` and `wyoung()`.** With `bootcluster=`, the fit keeps, for each
  cluster, the sums the bootstrap needs (about G x k^2 numbers, collected in the
  residual pass it makes anyway; nothing without the option). After the fit,
  `wildboottest()` runs the restricted or unrestricted wild cluster bootstrap of
  any coefficient from those sums, without reading the data, and `rwolf()` and
  `wyoung()` give Romano-Wolf and Westfall-Young adjusted p-values across models
  on the same draws. One-way, OLS only, and every fixed effect must be nested
  within the clusters (a fixed effect with few levels, such as year, can go in
  as covariates instead).

Fixes.

- `cluster="firm_id"`, a single string, was read as a list of characters; it
  now means one clustering, as documented.

Performance.

- **Cells are collapsed from sorted rows, without a hash aggregation.** Pass 0
  now sorts each worker's rows by the other fixed effects' levels too, so that
  every cell (one combination of levels) is a run of consecutive rows, and pass
  1 collapses the runs as it reads the sorted rows, in a multithreaded kernel.
  This replaces Polars' group_by over each slice of rows, and the cell table it
  wrote to disk and read back. Where nearly every row is a cell of its own
  (e.g., worker, firm and year effects on a worker-year panel), pass 0 sorts by
  the streamed dimension alone. The new `collapse_cells` option (`"auto"`, the
  default; True; False) overrides that choice, and
  `diagnostics["cells_collapsed"]` reports it. Together with the faster
  iterations below, this reduced compute time by 5-31% on large AKM
  configurations, and peak disk by up to 40%, with the same or lower memory.
- **Faster conjugate-gradient iterations.** The vector work of each iteration
  (dot products, norms, and the updates of the solution, residual and search
  direction) runs as fused, multithreaded kernels instead of single-threaded
  numpy operations.
- **Faster IRLS steps for Poisson, logit and probit.** Each step's read of the
  rows (every row's linear predictor, the family's mean, weight and working
  response, the deviance and log-likelihood, and the sums per cell) runs as one
  fused, multithreaded kernel, instead of a dozen single-threaded numpy
  operations over each chunk. On large regression benchmarks, compute time
  declined by 11-32%.
- **Pass 1 builds the Jacobi diagonal and step 2's right-hand side** chunk by
  chunk while it has the cells in memory, instead of reading the identifying
  cells back for each, when the variables are solved in one block (up to
  `rhs_block` of them). Each of those reads is a pass over data as large as
  the cell table, which matters most where the scratch disk is slow.
- **A memory budget for per-thread accumulators.** `stream_cg` and the
  leave-out passes give each thread its own level-sized accumulator, which
  with many threads and many levels could take more memory than the rest of
  the fit (64 threads, 30 million firm-years and 8 columns: about 120 GB,
  allocated afresh every iteration). They now get one copy per thread only
  while all of them fit in half of `max_s_gb`, with fewer threads on those
  passes when that binds, and `stream_cg` allocates its copies once per
  solve. Lowering `max_s_gb` tightens it.

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

- **A documentation site**, at
  [leetucker.github.io/hdfe-stream](https://leetucker.github.io/hdfe-stream/),
  built with Sphinx from `docs/` and published by a new GitHub Actions
  workflow. It has a page for each example, a section on leave-out variance
  components (bringing together `docs/kss.md`, the methodological differences
  and a derivation of the correction), a page for each benchmark, a function
  reference built from the docstrings, and this changelog. `docs/kss.md`,
  `docs/glm.md` and `docs/kss_methodological_differences.md` moved into it,
  and the README is shorter, pointing to it. The public docstrings are now
  numpydoc, documenting every option, including `tol`, `maxiter`,
  `row_group_size` and `log_level` of `StreamingHDFE`, and the attributes of
  `HDFEResult` and the other result classes. `pip install
  "hdfe-stream[docs]"` installs what building it needs.
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

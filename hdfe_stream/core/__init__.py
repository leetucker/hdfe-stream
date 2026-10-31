"""The estimation machinery that hdfe_stream's estimators share, and that
other estimators built on the same streaming passes can import.

Stable names
    The names below are the supported API for packages built on this layer
    (logit-stream is one). They change only in a minor release (0.x -> 0.y),
    with the change in the changelog; tests/test_core_api.py pins them.

    core.typing        PathLike, Source, Vcov, Variables
    core.formula       compile_design, categorical_vars, numeric_expr,
                       pyfixest_formula_api
    core.feterms       parse_fe
    core.report        warn
    core.utils         as_lazy, single_blas
    core.profile       Profile, Component, Request, ScanStats, scan_source,
                       combine_profiles, source_row_group_rows
    core.resources     detect_cpus, detect_memory_gb, limit_threads, limit_memory,
                       thread_scope, peak_rss_gb, reset_peak_rss, current_rss_gb,
                       describe, MemoryBudget
    core.workspace     Run, cleanup
    core.gpu           check_polars_gpu, cuda_backend
    core.store         BlockStore, Block, PolarsEngine, block_rows,
                       factorize, not_missing, write_sample, iter_batches,
                       iter_groups
    core.results_base  RegressionResult, RegressionMulti, vcov_key
    core.reporting     etable
    core.multcomp      bonferroni, holm
    core.ivtests       kp_rk, identification, iid_omegas, weak_f, hansen_j,
                       sargan, partial_r2, effective_f, anderson_rubin,
                       c_statistic

    The hooks a `RegressionResult` subclass implements (its docstring) are
    stable too.

    Everything else -- names with a leading underscore, and the engine
    modules (estimator, passes, solve, inference, irls, the kernels) -- is
    internal and can change in any release. Import the estimators' public
    names from `hdfe_stream`.

Layering: no module in this package imports from `hdfe_stream` outside
`core` at import time (tests/test_layering.py checks it). The only upward
references are the lazy imports inside `HDFEResult.leave_out_kss`, which
reach the leave-out (KSS) estimators in the parent package.

Infrastructure
    _columns.py        names of the columns the estimator adds to the rows
    typing.py          type aliases of the public signatures
    report.py          logging / printing
    utils.py           small shared helpers; Parquet iteration in whole groups
    profile.py         `Profile`: estimated memory/disk, the scan, the request
    profile_streaming.py  the memory model of the shared passes (`_ProfileMixin`)
    resources.py       CPUs and memory a job may use, thread limits, peak memory
                       (re-exports the `stream-limits` package)
    store.py           a block store of rows, for estimators that read every
                       row in each iteration (dense Newton)
    workspace.py       run directories, `cleanup()`
    gpu.py             the CuPy solve and the Polars GPU engine check
Input side
    feterms.py         'worker_id[t]' / 'firm_id^year' parsing
    formula.py         pyfixest formula -> Polars expressions
Engine
    estimator.py       `_StreamingBase`: options, layout, the driver
    passes.py          pass 0/1      (mixin)
    solve.py           step 2        (mixin)
    inference.py       steps 3/4     (mixin)
    irls.py            `_IRLSMixin`: the weighted least-squares steps of IRLS
    connected.py       connected components, the largest connected set
    families.py        GLM families: Poisson, logit, probit
Kernels (numba)
    kernels_base.py    cell sums, the solve, assembly, residuals, CRV3
    kernels_slopes.py  the same, with varying slopes
    kernels_glm.py     GLM row pass and group effects
    kernels_graph.py   connected components, articulation points
    kernels_groups.py  per-group sums, distinct counts, counter-based draws
Results and export
    results_base.py    `RegressionResult`, `RegressionMulti`: inference,
                       tests, reports and export from coefficients and vcov
    results.py         `HDFEResult`, `HDFEMulti`
    reporting.py       pyfixest views, `etable`
    wildboot.py        the wild cluster bootstrap
    multcomp.py        multiple-testing corrections
    ivtests.py         IV identification and overidentification tests from sums
"""

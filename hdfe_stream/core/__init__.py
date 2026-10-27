"""The estimation machinery that hdfe_stream's estimators share, and that
other estimators built on the same streaming passes can import.

Everything here is internal: names keep their leading underscores, and none
of it is a supported public API. Import public names from `hdfe_stream`.

Layering: no module in this package imports from `hdfe_stream` outside
`core` at import time (tests/test_layering.py checks it). The only upward
references are the lazy imports inside `HDFEResult.leave_out_kss`, which
reach the leave-out (KSS) estimators in the parent package.

Infrastructure
    _columns.py        names of the columns the estimator adds to the rows
    _types.py          type aliases of the public signatures
    report.py          logging / printing
    utils.py           small shared helpers; Parquet iteration in whole groups
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
"""

"""`feols_stream`: fit a pyfixest-style formula out of core."""

from __future__ import annotations

import polars as pl

from .estimator import StreamingHDFE
from .feterms import _parse_fe_term
from .report import _log
from .formula import _plan_formula, _pyfixest_formula_api
from .results import HDFEMulti


class _FormulaEstimator:
    def __init__(self, fml, workdir, options):
        self.fml, self.workdir, self.options = fml, workdir, options

    def fit(self, data, vcov=None, cluster=(), fe_dof="exact"):
        return feols_stream(self.fml, data, self.workdir, vcov=vcov, cluster=cluster,
                            fe_dof=fe_dof, **self.options)


def feols_stream(fml, data, workdir=None, vcov=None, cluster=(), fe_dof="exact", **options):
    """
    Out-of-core OLS with high-dimensional fixed effects from a pyfixest-style
    formula, e.g. "y ~ x1 + i(year, treat, ref=2010) | worker_id + firm_id^year".

    data    : Parquet path/glob or Polars LazyFrame.
    workdir : directory under which run directories are created (default:
              the system temporary directory); see StreamingHDFE for the
              outputs=, save_resid= and keep_intermediates= options.
    vcov    : as in pyfixest; default {'CRV1': <first FE>} like pyfixest.
    cluster : extra cluster specs to compute CRV1 for (one-way or 'a+b').
    **options : passed to StreamingHDFE (stream, weights, weights_type,
              solver, precond, keep, verbose, logger, log_level, ...).

    Varying slopes on the streamed dimension: "y ~ x | worker_id[t] + firm_id".
    The worker FE file then has fe_worker_id (intercept) and fe_worker_id[t] (slope)
    columns; the residual file's fe_worker_id column is the worker's total
    contribution for that row.

    IV formulas ("y ~ x | fe | endog ~ z") are estimated by 2SLS; each result
    carries its first-stage regressions (`first_stage`) and the Wald F of the
    excluded instruments computed with the same vcov (`f_stat_1st_stage`).

    All models with the same fixed effects share one set of passes and one
    solve, and are estimated on the rows where *all* of their variables are
    non-missing (pyfixest drops missing values model by model).
    """
    lf = data if isinstance(data, pl.LazyFrame) else pl.scan_parquet(data)
    _log(options.get("verbose", True), f"planning {fml!r} (formula expansion, level discovery)",
         options.get("logger"), options.get("log_level"))
    groups = _plan_formula(fml, lf)
    results = []
    try:
        for fe, g in groups.items():
            est = StreamingHDFE(y=list(g["y"].items()), x=list(g["x"].items()), fe=list(fe),
                                workdir=workdir, models=g["models"], **options)
            # pyfixest's default: cluster by the first fixed effect as written
            res = est.fit(lf, vcov=vcov if vcov is not None else
                          {"CRV1": _parse_fe_term(fe[0])[0]}, cluster=cluster, fe_dof=fe_dof)
            results += list(res) if isinstance(res, HDFEMulti) else [res]
    except BaseException:
        # a later FE set failed: don't leave the earlier sets' files behind
        if not options.get("keep_intermediates"):
            for r in results:
                r.cleanup()
        raise
    Formula, _ = _pyfixest_formula_api()
    order = {s.formula: i for i, s in enumerate(Formula.parse(fml))}
    results.sort(key=lambda r: order.get(r.fml, len(order)))
    return results[0] if len(results) == 1 else HDFEMulti(results)

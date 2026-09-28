"""`feols_stream`, `fepois_stream`, `feglm_stream`: fit a pyfixest-style
formula out of core."""

from __future__ import annotations

import polars as pl

from .estimator import StreamingHDFE
from .feterms import _parse_fe_term
from .glm import StreamingGLM
from .report import _log
from .formula import _plan_formula, _pyfixest_formula_api
from .results import HDFEMulti, _vcov_key


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
    Any number of fixed effects works, including one ("y ~ x | worker_id",
    a within regression) and none ("y ~ x", OLS with an intercept).

    data    : Parquet path/glob or Polars LazyFrame.
    workdir : directory under which run directories are created (default:
              $HDFE_STREAM_WORKDIR if set, else the system temporary
              directory); see StreamingHDFE for the
              outputs=, save_resid= and keep_intermediates= options.
    vcov    : as in pyfixest; default {'CRV1': <first FE>} like pyfixest, or
              iid without fixed effects. {'CRV3': var} (one-way, OLS) needs
              every fixed effect nested within the clusters, or none.
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
    def estimators(fe, g):
        yield StreamingHDFE(y=list(g["y"].items()), x=list(g["x"].items()), fe=list(fe),
                            workdir=workdir, models=g["models"], **options)
    return _fit_formula(fml, data, estimators, vcov, cluster, fe_dof, options)


def _fit_formula(fml, data, estimators, vcov, cluster, fe_dof, options):
    """Plan the formula, fit each estimator that `estimators(fe, group)`
    makes for each set of fixed effects, and return the results in the
    formula's order."""
    lf = data if isinstance(data, pl.LazyFrame) else pl.scan_parquet(data)
    _log(options.get("verbose", True), f"planning {fml!r} (formula expansion, level discovery)",
         options.get("logger"), options.get("log_level"))
    groups = _plan_formula(fml, lf)
    results = []
    try:
        for fe, g in groups.items():
            # pyfixest's default: cluster by the first fixed effect as written,
            # or iid when there is none
            default = {"CRV1": _parse_fe_term(fe[0])[0]} if fe else "iid"
            for est in estimators(fe, g):
                res = est.fit(lf, vcov=vcov if vcov is not None else default, cluster=cluster,
                              fe_dof=fe_dof)
                results += list(res) if isinstance(res, HDFEMulti) else [res]
    except BaseException:
        # a later fit failed: don't leave the earlier fits' files behind
        if not options.get("keep_intermediates"):
            for r in results:
                r.cleanup()
        raise
    Formula, _ = _pyfixest_formula_api()
    order = {s.formula: i for i, s in enumerate(Formula.parse(fml))}
    results.sort(key=lambda r: order.get(r.fml, len(order)))
    return results[0] if len(results) == 1 else HDFEMulti(results)


def fepois_stream(fml, data, workdir=None, vcov=None, cluster=(), fe_dof="exact",
                  offset=None, iwls_tol=1e-8, iwls_maxiter=25, separation_check=True,
                  **options):
    """
    Out-of-core Poisson regression (log link) with high-dimensional fixed
    effects, from a pyfixest-style formula, e.g.
    "trade ~ log_dist | exporter^year + importer^year". Estimated by
    iteratively reweighted least squares; each step is one read of the rows
    and one solve of the fixed effects (see `StreamingGLM`).

    data, workdir, vcov, cluster, fe_dof, **options : as in `feols_stream`,
              except that the vcov may not be CRV3, and IV formulas and
              varying slopes are not available. weights= and weights_type=
              work as in pyfixest's fepois.
    offset  : column name added to the linear predictor with a coefficient of
              one (e.g. log exposure).
    iwls_tol, iwls_maxiter : convergence tolerance on the relative change in
              deviance, and the most IRLS steps (pyfixest's defaults).
    separation_check : drop fixed-effect levels whose outcome is zero on
              every row (their effect would be minus infinity), as pyfixest's
              "fe" check, which it runs by default. Covariates that separate
              the outcome (its "ir" check) are not detected.

    The dependent variable must be nonnegative. Each dependent variable is
    its own set of IRLS fits (the separation check depends on it); models of
    the same outcome share pass 0.
    """
    return _fit_glm(fml, data, "poisson", workdir, vcov, cluster, fe_dof,
                    dict(options, offset=offset, iwls_tol=iwls_tol, iwls_maxiter=iwls_maxiter,
                         separation_check=separation_check))


def feglm_stream(fml, data, family, workdir=None, vcov=None, cluster=(), fe_dof="exact",
                 iwls_tol=1e-8, iwls_maxiter=25, separation_check=True, **options):
    """
    Out-of-core logit or probit regression with high-dimensional fixed
    effects, from a pyfixest-style formula: `family` is "logit" or "probit".
    Estimated by iteratively reweighted least squares, with step-halving, as
    pyfixest's feglm (see `StreamingGLM`).

    Arguments as in `fepois_stream`, without the offset. The dependent
    variable must be 0 or 1. The separation check drops fixed-effect levels
    whose outcome is the same on every row (all 0 or all 1), repeating until
    none is left; pyfixest's "fe" check drops only levels whose outcome is
    all 0, once.

    Weights (weights=, weights_type=) multiply each row's log-likelihood, as
    in fepois; pyfixest's feglm takes none. Frequency weights give exactly
    the fit of the data with each row repeated that many times.

    No correction is made for the incidental parameter problem: with fixed
    effects estimated from few observations each, the coefficients are
    biased, as in pyfixest and fixest.
    """
    if str(family).lower() not in ("logit", "probit"):
        raise ValueError(f"family must be 'logit' or 'probit', got {family!r}"
                         + ("; use fepois_stream" if str(family).lower() == "poisson" else "")
                         + ("; for a linear model use feols_stream"
                            if str(family).lower() == "gaussian" else ""))
    return _fit_glm(fml, data, str(family).lower(), workdir, vcov, cluster, fe_dof,
                    dict(options, iwls_tol=iwls_tol, iwls_maxiter=iwls_maxiter,
                         separation_check=separation_check))


def _fit_glm(fml, data, family, workdir, vcov, cluster, fe_dof, options):
    key = _vcov_key(vcov)
    if key is not None and key.startswith("CRV3:"):
        raise ValueError("CRV3 standard errors are not available for GLMs; use CRV1")

    def estimators(fe, g):
        by_y = {}
        for model in g["models"]:
            if model.get("iv"):
                raise ValueError(f"{model['fml']}: IV (2SLS) is not available for GLMs")
            by_y.setdefault(model["y"], []).append(model)
        for yname, models in by_y.items():
            xs = list(dict.fromkeys(x for model in models for x in model["x"]))
            yield StreamingGLM(y=[(yname, g["y"][yname])], x=[(x, g["x"][x]) for x in xs],
                               fe=list(fe), family=family, workdir=workdir, models=models,
                               **options)
    return _fit_formula(fml, data, estimators, vcov, cluster, fe_dof, options)

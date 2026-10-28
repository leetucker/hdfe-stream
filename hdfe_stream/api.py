"""`feols_stream`, `fepois_stream`, `feglm_stream`: fit a pyfixest-style
formula out of core."""

from __future__ import annotations

from typing import Any, Sequence

import polars as pl

from .core.typing import PathLike, Source, Vcov

from .estimator import StreamingHDFE
from .glm import StreamingGLM
from .core.report import _log
from .core.formula import _plan_formula, pyfixest_formula_api
from .core.results import HDFEMulti, HDFEResult
from .core.results_base import vcov_key
from .core.utils import as_lazy


def feols_stream(fml: str, data: Source, workdir: PathLike | None = None,
                 vcov: Vcov = "iid", cluster: Sequence[str] | str = (),
                 fe_dof: str = "exact", bootcluster: str | None = None,
                 **options: Any) -> HDFEResult | HDFEMulti:
    """Out-of-core OLS with high-dimensional fixed effects, from a
    pyfixest-style formula.

    Any number of fixed effects works, including one (``"y ~ x | worker_id"``,
    a within regression) and none (``"y ~ x"``, OLS with an intercept).

    Parameters
    ----------
    fml : str
        The formula, in pyfixest syntax, e.g.
        ``"y ~ x1 + i(year, treat, ref=2010) | worker_id + firm_id^year"``.
        Several outcomes (``y1 + y2 ~ ...``) and stepwise sets (``sw()``,
        ``csw()``, ``sw0()``, ``csw0()``) give several models. Varying slopes
        on the streamed dimension: ``"y ~ x | worker_id[t] + firm_id"``. IV:
        ``"y ~ x | fe | endog ~ z"``, estimated by 2SLS.
    data : str, Path, polars.LazyFrame or polars.DataFrame
        A Parquet path or glob, or a Polars LazyFrame. A DataFrame also works,
        but is already in memory, which the library is built to avoid.
    workdir : str or Path, optional
        Directory under which run directories are created. Default: the
        ``HDFE_STREAM_WORKDIR`` environment variable if set, else the system
        temporary directory. See `StreamingHDFE` for the ``outputs=``,
        ``save_resid=`` and ``keep_intermediates=`` options.
    vcov : str or dict, default "iid"
        As in pyfixest: ``"iid"``, ``"hetero"`` (or ``"HC1"``),
        ``{"CRV1": "var"}`` or ``"CRV1:var"``, with multi-way clustering as
        ``{"CRV1": "a+b"}``. ``{"CRV3": var}`` (one-way, OLS) needs every fixed
        effect nested within the clusters, or none.
    cluster : str or sequence of str, optional
        Further cluster specifications to compute CRV1 for (one-way, or
        ``"a+b"``), so that `HDFEResult.with_vcov` can switch to them.
    fe_dof : {"exact", "pyfixest"}, default "exact"
        How the fixed-effect parameters are counted for the degrees of
        freedom; see `StreamingHDFE.fit`.
    bootcluster : str, optional
        A clustering to keep the per-cluster sums for, so that
        `HDFEResult.wildboottest`, `hdfe_stream.rwolf` and
        `hdfe_stream.wyoung` can bootstrap it after the fit. One-way, OLS,
        and every fixed effect nested within the clusters; see
        `StreamingHDFE.fit`.
    **options
        Passed to `StreamingHDFE`: ``fixef_rm``, ``connected``,
        ``min_degree``, ``connected_fe``, ``stream``, ``weights``,
        ``weights_type``, ``solver``, ``precond``, ``keep``, the memory
        options, ``verbose``, ``logger``, ``log_level``, and the rest.
        ``fixef_rm="singleton"`` (the default, as in pyfixest) drops singleton
        observations; ``"none"`` keeps them. ``connected="largest"`` keeps only
        the largest connected set of the first two fixed effects listed, and
        ``min_degree=`` prunes their levels with too few distinct partners
        first (see `connected_components`).

    Returns
    -------
    HDFEResult or HDFEMulti
        One result, or several when the formula expands to several models.

    Notes
    -----
    With varying slopes, the worker fixed-effect file has ``fe_worker_id``
    (intercept) and ``fe_worker_id[t]`` (slope) columns; the residual file's
    ``fe_worker_id`` column is the worker's total contribution for that row.

    IV results carry their first-stage regressions (``first_stage``) and the
    Wald F of the excluded instruments computed with the same vcov
    (``f_stat_1st_stage``).

    All models with the same fixed effects share one set of passes and one
    solve, and are estimated on the rows where *all* of their variables are
    non-missing (pyfixest drops missing values model by model).

    Examples
    --------
    >>> fit = feols_stream(
    ...     "log_earn ~ age_squared + age_cubed | worker_id + firm_id + year",
    ...     "data/*.parquet", workdir="scratch", vcov={"CRV1": "worker_id"})
    >>> fit.summary()
    """
    def estimators(fe, g):
        yield StreamingHDFE(y=list(g["y"].items()), x=list(g["x"].items()), fe=list(fe),
                            workdir=workdir, models=g["models"], **options)
    return _fit_formula(fml, data, estimators, vcov, cluster, fe_dof, options,
                        bootcluster)


def _fit_formula(fml, data, estimators, vcov, cluster, fe_dof, options, bootcluster=None):
    """Plan the formula, fit each estimator that `estimators(fe, group)`
    makes for each set of fixed effects, and return the results in the
    formula's order."""
    lf = as_lazy(data)
    _log(options.get("verbose", True), f"planning {fml!r} (formula expansion, level discovery)",
         options.get("logger"), options.get("log_level"))
    groups = _plan_formula(fml, lf)
    results = []
    try:
        for fe, g in groups.items():
            for est in estimators(fe, g):
                res = est.fit(lf, vcov=vcov, cluster=cluster, fe_dof=fe_dof,
                              bootcluster=bootcluster)
                results += list(res) if isinstance(res, HDFEMulti) else [res]
    except BaseException:
        # a later fit failed: don't leave the earlier fits' files behind
        if not options.get("keep_intermediates"):
            for r in results:
                r.cleanup()
        raise
    Formula, _ = pyfixest_formula_api()
    order = {s.formula: i for i, s in enumerate(Formula.parse(fml))}
    results.sort(key=lambda r: order.get(r.fml, len(order)))
    return results[0] if len(results) == 1 else HDFEMulti(results)


def fepois_stream(fml: str, data: Source, workdir: PathLike | None = None,
                  vcov: Vcov = "iid", cluster: Sequence[str] | str = (),
                  fe_dof: str = "exact", offset: str | pl.Expr | None = None,
                  iwls_tol: float = 1e-8, iwls_maxiter: int = 25,
                  separation_check: bool = True, **options: Any) -> HDFEResult | HDFEMulti:
    """Out-of-core Poisson regression (log link) with high-dimensional fixed
    effects, from a pyfixest-style formula.

    Estimated by iteratively reweighted least squares; each step is one read
    of the rows and one solve of the fixed effects (see `StreamingGLM`).

    Parameters
    ----------
    fml : str
        The formula, e.g. ``"trade ~ log_dist | exporter^year + importer^year"``.
        IV formulas and varying slopes are not available.
    data, workdir, cluster, fe_dof
        As in `feols_stream`.
    vcov : str or dict, default "iid"
        As in `feols_stream`, except that it may not be CRV3.
    offset : str or polars.Expr, optional
        A column added to the linear predictor with a coefficient of one
        (e.g. log exposure).
    iwls_tol : float, default 1e-8
        Convergence tolerance on the relative change in deviance,
        ``|dev - dev_old| / (0.1 + |dev|)`` (pyfixest's default).
    iwls_maxiter : int, default 25
        The most IRLS steps (pyfixest's default). A fit that reaches it warns,
        and ``diagnostics["irls"]["converged"]`` is False.
    separation_check : bool, default True
        Drop fixed-effect levels whose outcome is zero on every row (their
        effect would be minus infinity), as pyfixest's ``"fe"`` check, which
        it runs by default. Covariates that separate the outcome (its ``"ir"``
        check) are not detected.
    **options
        As in `feols_stream`. ``weights=`` and ``weights_type=`` work as in
        pyfixest's ``fepois``.

    Returns
    -------
    HDFEResult or HDFEMulti

    Notes
    -----
    The dependent variable must be nonnegative. Each dependent variable is its
    own set of IRLS fits (the separation check depends on it); models of the
    same outcome share pass 0.
    """
    return _fit_glm(fml, data, "poisson", workdir, vcov, cluster, fe_dof,
                    dict(options, offset=offset, iwls_tol=iwls_tol, iwls_maxiter=iwls_maxiter,
                         separation_check=separation_check))


def feglm_stream(fml: str, data: Source, family: str, workdir: PathLike | None = None,
                 vcov: Vcov = "iid", cluster: Sequence[str] | str = (),
                 fe_dof: str = "exact", iwls_tol: float = 1e-8, iwls_maxiter: int = 25,
                 separation_check: bool = True, **options: Any) -> HDFEResult | HDFEMulti:
    """Out-of-core logit or probit regression with high-dimensional fixed
    effects, from a pyfixest-style formula.

    Estimated by iteratively reweighted least squares, with step-halving, as
    pyfixest's ``feglm`` (see `StreamingGLM`).

    Parameters
    ----------
    fml : str
        The formula. IV formulas and varying slopes are not available.
    data : str, Path, polars.LazyFrame or polars.DataFrame
        As in `feols_stream`.
    family : {"logit", "probit"}
        The model.
    workdir, vcov, cluster, fe_dof, iwls_tol, iwls_maxiter, **options
        As in `fepois_stream`.
    separation_check : bool, default True
        Drop fixed-effect levels whose outcome is the same on every row (all 0
        or all 1), repeating until none is left. pyfixest's ``"fe"`` check
        drops only levels whose outcome is all 0, once.

    Returns
    -------
    HDFEResult or HDFEMulti

    Notes
    -----
    The dependent variable must be 0 or 1.

    Weights (``weights=``, ``weights_type=``) multiply each row's
    log-likelihood, as in ``fepois``; pyfixest's ``feglm`` takes none.
    Frequency weights give exactly the fit of the data with each row repeated
    that many times.

    With fixed effects estimated from few observations each, the
    coefficients have an incidental-parameter bias; ``bias_correction=True``
    (an option of `StreamingGLM`) corrects it as R's alpaca does, which
    pyfixest and fixest do not.
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
    key = vcov_key(vcov)
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

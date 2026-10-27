"""Multiple-testing corrections across models: `bonferroni` and `holm`, from
the p-values; `rwolf` and `wyoung`, from the wild cluster bootstrap."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import polars as pl

from .results import HDFEMulti, HDFEResult
from .results_base import RegressionMulti, RegressionResult


def _family(models, param):
    """One row per (model, coefficient): the hypotheses being corrected for."""
    if isinstance(models, (RegressionResult, RegressionMulti)):
        models = [models]
    fits = []
    for m in models:
        fits += list(m) if isinstance(m, RegressionMulti) else [m]
    if not fits:
        raise ValueError("no models given")
    params = [param] if isinstance(param, str) else list(param)
    rows = []
    for i, fit in enumerate(fits):
        missing = [p for p in params if p not in fit.coefnames]
        if missing:
            raise ValueError(f"{missing} not among the coefficients of model {i} "
                             f"({fit.fml}): {fit.coefnames}")
        tidy = fit.tidy().filter(pl.col("Coefficient").is_in(params))
        rows.append(tidy.with_columns(pl.lit(i).alias("model"), pl.lit(fit.fml).alias("fml"))
                        .sort(pl.col("Coefficient").replace_strict(
                            params, list(range(len(params))))))
    table = pl.concat(rows).select("model", "fml", pl.all().exclude("model", "fml"))
    p = table["Pr(>|t|)"].to_numpy()
    if np.isnan(p).any():
        raise ValueError("some p-values are nan (a coefficient with a zero or undefined "
                         "standard error)")
    return table, p


def bonferroni(models: HDFEResult | HDFEMulti | Iterable, param: str | list) -> pl.DataFrame:
    """Bonferroni-adjusted p-values for ``param = 0`` across models.

    Each p-value, from `HDFEResult.tidy`, is multiplied by the number of
    hypotheses (models times parameters), capped at 1. As pyfixest's
    ``bonferroni``, which takes one parameter.

    Parameters
    ----------
    models : HDFEResult, HDFEMulti, or a list of them
        The models.
    param : str or list of str
        The coefficient(s) tested in every model; each must be in every model.

    Returns
    -------
    polars.DataFrame
        One row per model and parameter: ``model`` (its position), ``fml``,
        the `tidy` columns, and ``Bonferroni Pr(>|t|)``.
    """
    table, p = _family(models, param)
    return table.with_columns(pl.Series("Bonferroni Pr(>|t|)", np.minimum(1.0, p * len(p))))


def holm(models: HDFEResult | HDFEMulti | Iterable, param: str | list) -> pl.DataFrame:
    """Holm's step-down adjusted p-values for ``param = 0`` across models.

    The p-values are sorted from smallest to largest, the j-th of m is
    multiplied by m - j + 1, and each adjusted p-value is the running maximum
    of those, capped at 1. Like Bonferroni's, the adjustment controls the
    familywise error rate under any dependence between the tests, and no
    adjusted p-value is larger than Bonferroni's, so it rejects at least as
    often (Holm, 1979).

    Parameters
    ----------
    models, param
        As in `bonferroni`.

    Returns
    -------
    polars.DataFrame
        As in `bonferroni`, with ``Holm Pr(>|t|)``.
    """
    table, p = _family(models, param)
    return table.with_columns(pl.Series("Holm Pr(>|t|)", _holm(p)))


def _holm(p):
    p = np.asarray(p, dtype=float)
    m = len(p)
    order = np.argsort(p, kind="stable")
    adjusted = np.empty(m)
    adjusted[order] = np.minimum(1.0, np.maximum.accumulate((m - np.arange(m)) * p[order]))
    return adjusted


def rwolf(models: HDFEResult | HDFEMulti | Iterable, param: str | list, reps: int = 9999,
          seed: int | None = None, weights_type: str = "rademacher") -> pl.DataFrame:
    """Romano-Wolf adjusted p-values for ``param = 0`` across models, from
    the wild cluster bootstrap.

    Every model is bootstrapped on the same draws (each cluster gets the same
    weight in every model), and the hypotheses are stepped down from the
    largest absolute t statistic: each one's p-value is the share of draws
    whose largest absolute bootstrap t statistic, over the hypotheses not yet
    stepped past, reaches its t statistic, with one added to the count and to
    the draws, and the p-values are then made monotone (Romano and Wolf 2005;
    Clarke, Romano and Wolf 2020). This controls the familywise error rate
    while using the dependence between the tests, so it rejects more than
    Bonferroni or Holm when they are correlated. As pyfixest's ``rwolf``,
    with the same draws for the same seed.

    Parameters
    ----------
    models : HDFEResult, HDFEMulti, or a list of them
        Fit with the same ``bootcluster=``.
    param : str or list of str
        The coefficient(s) tested in every model.
    reps : int, default 9999
        Bootstrap draws.
    seed : int, optional
        Seed for the draws.
    weights_type : {"rademacher", "mammen", "webb", "norm"}, default "rademacher"
        The distribution of the cluster weights.

    Returns
    -------
    polars.DataFrame
        As in `bonferroni`, with ``RW Pr(>|t|)``.
    """
    table, t, tstar = _resampled(models, param, reps, seed, weights_type)
    return table.with_columns(pl.Series("RW Pr(>|t|)", _rwolf_p(t, tstar)))


def wyoung(models: HDFEResult | HDFEMulti | Iterable, param: str | list, reps: int = 9999,
           seed: int | None = None, weights_type: str = "rademacher") -> pl.DataFrame:
    """Westfall-Young adjusted p-values for ``param = 0`` across models, from
    the wild cluster bootstrap.

    The free step-down resampling method (Westfall and Young 1993): each
    test's t statistic, and each draw's, is turned into a p-value
    from the t distribution with G - 1 degrees of freedom; in each draw, the
    p-values are replaced by their running minimum over the tests with larger
    original p-values; and a test's adjusted p-value is the share of draws in
    which that minimum is at most its own p-value, made monotone. As
    pyfixest's ``wyoung``, with the same draws for the same seed.

    Parameters
    ----------
    models, param, reps, seed, weights_type
        As in `rwolf`.

    Returns
    -------
    polars.DataFrame
        As in `bonferroni`, with ``WY Pr(>|t|)``.
    """
    from scipy import special

    table, t, tstar = _resampled(models, param, reps, seed, weights_type)
    df = np.array([fit._boot["df"] for fit in _fits(models)
                   for _ in ([param] if isinstance(param, str) else param)], dtype=float)
    p = 2 * special.stdtr(df, -np.abs(t))
    pstar = 2 * special.stdtr(df, -np.abs(tstar))
    return table.with_columns(pl.Series("WY Pr(>|t|)", _wyoung_p(p, pstar)))


def _fits(models):
    if isinstance(models, (RegressionResult, RegressionMulti)):
        models = [models]
    fits = []
    for m in models:
        fits += list(m) if isinstance(m, RegressionMulti) else [m]
    return fits


def _resampled(models, param, reps, seed, weights_type):
    """The family's table, and its t statistics and bootstrap t statistics
    (B x S), in the table's row order."""
    from .wildboot import _bootstrap

    table, _ = _family(models, param)
    params = [param] if isinstance(param, str) else list(param)
    tests = [(fit, fit.coefnames.index(p)) for fit in _fits(models) for p in params]
    for fit, _ in tests:
        if getattr(fit, "_boot", None) is None:
            raise ValueError(f"{fit.fml}: fit with bootcluster= to bootstrap it")
    t, tstar, _ = _bootstrap(tests, reps, weights_type, True, seed)
    return table, t, tstar


def _rwolf_p(t, tstar):
    """Romano-Wolf step-down p-values, as pyfixest computes them."""
    t, tstar = np.abs(t), np.abs(tstar)
    B, S = tstar.shape
    order = np.argsort(t)[::-1]
    p = np.empty(S)
    for s in range(S):
        top = tstar[:, order[s:]].max(axis=1)
        p[s] = min(1.0, (np.sum(top >= t[order[s]] * (1 - 1e-9)) + 1) / (B + 1))
    out = np.empty(S)
    out[order] = np.maximum.accumulate(p)
    return out


def _wyoung_p(p, pstar):
    """Westfall-Young free step-down p-values, as pyfixest computes them."""
    B, S = pstar.shape
    order = np.argsort(p)
    Q = np.minimum.accumulate(pstar[:, order[::-1]], axis=1)[:, ::-1]
    # a draw that reproduces a p-value (to rounding) is a tie, and counts
    adjusted = np.maximum.accumulate((Q <= p[order] * (1 + 1e-9)).sum(axis=0) / B)
    out = np.empty(S)
    out[order] = adjusted
    return out

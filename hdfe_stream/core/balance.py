"""A weighted two-group balance table, out of core.

For each term (a numeric column or expression), the weighted means of two
groups (1 and 0), the standardized mean difference, the variance ratio, and
the mean and maximum gap between the groups' weighted empirical CDFs. Any
weighting works: matching weights, inverse-probability weights, entropy
balancing, or none (weight 1). Everything is streamed sums and one sort per
term (for the eCDF), so the data are never held in memory.

Conventions (MatchIt's, which cobalt shares; match-stream checks them
against MatchIt 4.8.2):

- weighted variance: weights normalized to sum to 1 within the group,
  sum w (x - m)^2 / (1 - sum w^2); for a 0/1 term, m (1 - m);
- the standardized mean difference is (m1 - m0) / scale, with `scale` the
  term's standard deviation as the caller defines it (MatchIt: the focal
  group's, before weighting; cobalt's default for the ATE: the pooled
  one), else computed here from group 1, group 0 or both pooled (the
  square root of the mean of the two variances); differences below
  sqrt(eps) are reported unscaled, as MatchIt;
- the eCDF gaps are evaluated at each distinct value of the term among all
  rows given (a row with weight 0 contributes its value but no mass), each
  group's weights summing to 1; for a 0/1 term both equal |m1 - m0|.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

_G, _W = "__bal_g", "__bal_w"
EPS_DIFF = np.sqrt(np.finfo(float).eps)


def _prepare(lf, group, weight, terms):
    lf = lf.lazy()
    g = pl.col(group).cast(pl.Boolean) if isinstance(group, str) else group.cast(pl.Boolean)
    if weight is None:
        w = pl.lit(1.0)
    else:
        w = pl.col(weight) if isinstance(weight, str) else weight
    names = [t if isinstance(t, str) else t.meta.output_name() for t in terms]
    exprs = [(pl.col(t) if isinstance(t, str) else t).cast(pl.Float64).alias(f"__bal_x{i}")
             for i, t in enumerate(terms)]
    return lf.select(g.alias(_G), w.cast(pl.Float64).alias(_W), *exprs), names


def group_moments(lf, n_terms: int) -> dict:
    """Per group (1, 0) and term i: sum w, sum w^2, sum w x, sum w x^2, and
    whether the term only takes 0 and 1 (over all rows given)."""
    aggs = []
    for arm, cond in (("1", pl.col(_G)), ("0", ~pl.col(_G))):
        w = pl.col(_W).filter(cond)
        aggs += [w.sum().alias(f"sw{arm}"), (w * w).sum().alias(f"sww{arm}")]
        for i in range(n_terms):
            xv = pl.col(f"__bal_x{i}").filter(cond)
            aggs += [(w * xv).sum().alias(f"sx{arm}_{i}"),
                     (w * xv * xv).sum().alias(f"sxx{arm}_{i}")]
    for i in range(n_terms):
        aggs.append(pl.col(f"__bal_x{i}").is_in([0.0, 1.0]).all().alias(f"bin_{i}"))
    return lf.select(aggs).collect(engine="streaming").row(0, named=True)


def mean_var(r: dict, arm: str, i: int, binary: bool) -> tuple[float, float]:
    """The weighted mean and variance of term i in group `arm` ("1"/"0")."""
    sw, sww = r[f"sw{arm}"], r[f"sww{arm}"]
    if not sw:
        return np.nan, np.nan
    m = r[f"sx{arm}_{i}"] / sw
    if binary:
        return m, m * (1 - m)
    denom = 1 - sww / sw**2
    var = (r[f"sxx{arm}_{i}"] / sw - m * m) / denom if denom > 0 else np.nan
    return m, var


def ecdf_gaps(lf, i: int) -> tuple[float, float]:
    """Mean and maximum |F_1(v) - F_0(v)| over the distinct values v of term i."""
    norm = pl.col(_W) / pl.col(_W).sum().over(_G)
    signed = pl.when(pl.col(_G)).then(norm).otherwise(-norm)
    d = (lf.select(pl.col(f"__bal_x{i}").alias("x"), signed.alias("s"))
         .sort("x")
         .with_columns(pl.col("s").cum_sum().abs().alias("gap"))
         .filter(pl.col("x") != pl.col("x").shift(-1).fill_null(np.inf))
         .select(pl.col("gap").mean().alias("mean"), pl.col("gap").max().alias("max"))
         .collect(engine="streaming"))
    return float(d["mean"][0]), float(d["max"][0])


@dataclass
class GroupSizes:
    """The rows of groups 1 and 0 (`n1`, `n0`) and their Kish effective
    sample sizes (`ess1`, `ess0`): (sum w)^2 / sum w^2, the number of
    equally weighted rows that would give a mean the same variance; equal
    to the rows without weights, 0 for a group with no weight."""

    n1: int
    n0: int
    ess1: float
    ess0: float


def effective_sizes(lf: pl.LazyFrame | pl.DataFrame, group, weight=None) -> GroupSizes:
    """The rows and Kish effective sample sizes of the two groups in `lf`,
    `group` and `weight` as in `weighted_balance` (a column or expression;
    no weight: 1). One streaming pass."""
    lf, _ = _prepare(lf, group, weight, [])
    aggs = []
    for arm, cond in (("1", pl.col(_G)), ("0", ~pl.col(_G))):
        w = pl.col(_W).filter(cond)
        aggs += [w.len().alias(f"n{arm}"), w.sum().alias(f"sw{arm}"),
                 (w * w).sum().alias(f"sww{arm}")]
    r = lf.select(aggs).collect(engine="streaming").row(0, named=True)

    def ess(arm):
        return r[f"sw{arm}"] ** 2 / r[f"sww{arm}"] if r[f"sww{arm}"] else 0.0
    return GroupSizes(n1=int(r["n1"]), n0=int(r["n0"]), ess1=ess("1"), ess0=ess("0"))


def weighted_balance(lf: pl.LazyFrame | pl.DataFrame, group, terms, weight=None, *,
                     scale: str | dict = "pooled", ecdf: bool = True) -> pl.DataFrame:
    """The balance table: one row per term with ``mean1``, ``mean0``,
    ``var1``, ``var0``, ``smd``, ``var_ratio`` (var1 / var0; null for a 0/1
    term), ``ecdf_mean`` and ``ecdf_max`` (|m1 - m0| for a 0/1 term).

    Parameters
    ----------
    lf : the rows (a LazyFrame streams)
    group : column or boolean expression, true for group 1
    terms : columns or expressions (aliased) to compare
    weight : column or expression; None weighs every row 1
    scale : "group1", "group0", "pooled", or a dict term -> standard
        deviation (e.g. computed before weighting, as MatchIt)
    ecdf : compute the eCDF gaps (one sort per term)
    """
    data, names = _prepare(lf, group, weight, list(terms))
    r = group_moments(data, len(names))
    rows = []
    for i, name in enumerate(names):
        binary = bool(r[f"bin_{i}"])
        m1, v1 = mean_var(r, "1", i, binary)
        m0, v0 = mean_var(r, "0", i, binary)
        if isinstance(scale, dict):
            sd = float(scale[name])
        elif scale == "group1":
            sd = np.sqrt(v1)
        elif scale == "group0":
            sd = np.sqrt(v0)
        elif scale == "pooled":
            sd = np.sqrt((v1 + v0) / 2)
        else:
            raise ValueError(f"scale must be 'group1', 'group0', 'pooled' or a dict, "
                             f"not {scale!r}")
        diff = m1 - m0
        smd = (diff / sd if sd > 0 else np.copysign(np.inf, diff)) if abs(diff) > EPS_DIFF else diff
        if binary:
            ratio, emean, emax = None, abs(diff), abs(diff)
        else:
            ratio = v1 / v0 if v0 > 0 else (np.inf if v1 > 0 else None)
            emean, emax = ecdf_gaps(data, i) if ecdf else (np.nan, np.nan)
        rows.append({"term": name, "mean1": m1, "mean0": m0, "var1": v1, "var0": v0,
                     "smd": smd, "var_ratio": ratio, "ecdf_mean": emean, "ecdf_max": emax})
    return pl.DataFrame(rows, schema={"term": pl.String, "mean1": pl.Float64,
                                      "mean0": pl.Float64, "var1": pl.Float64,
                                      "var0": pl.Float64, "smd": pl.Float64,
                                      "var_ratio": pl.Float64, "ecdf_mean": pl.Float64,
                                      "ecdf_max": pl.Float64})

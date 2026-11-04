"""Inverse-probability weights from a fitted score, and the Hajek
(normalized) IPW difference in means with an influence-function variance
that accounts for the estimated score.

Weights for a treatment D with score p = P(D = 1 | x):

    estimand  D = 1          D = 0
    "att"     1              p / (1 - p)
    "atc"     (1 - p) / p    1
    "ate"     1 / p          1 / (1 - p)

The estimate is the difference of the two groups' weighted means, each
normalized by its own total weight (Hajek; the Horvitz-Thompson version,
dividing by N, is not offered: it is not invariant to adding a constant
to the outcome). Its influence function is `weighted_mean_difference`'s
plus the first-stage term J H^-1 s_i (`mestimation`), with

    J = d theta / d gamma = sum_D=1 w'_i f_i x_i (y_i - mu1) / W1
                          - sum_D=0 w'_i f_i x_i (y_i - mu0) / W0

(w' = dw/dp, f = dp/deta), so the variance is the stacked M-estimation
sandwich of Stata's ``teffects ipw`` (normalized weights) or WeightIt's
M-estimation, written so that it streams and clusters (`if_vcov`).

Trimming (`trim`): units whose score lies outside [lo, hi] are dropped
(Crump, Hotz, Imbens and Mitnik 2009 use [0.1, 0.9]); the trimmed sample
is then taken as given, as is usual, which changes the estimand to the
effect on the units kept. Trimming happens before the weights are
computed; the first stage is not refitted. This is not R did / DRDID's
ATT convention, which keeps every treated unit, drops only controls with a
high score, and caps the score at 1 - 1e-6; an estimator that follows
DRDID trims itself and passes ``trim=None``.

Written for any estimator that weights by a fitted score (matching,
Callaway and Sant'Anna's IPW and DR estimators with their own outcome
contrasts as `y`, ...); no estimator-specific names.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from .influence import IFVariance, if_vcov
from .mestimation import BinaryFirstStage, corrected_contributions

ESTIMANDS = ("att", "atc", "ate")


def ipw_weight(treatment: str, p: pl.Expr, estimand: str) -> pl.Expr:
    """The raw (unnormalized) inverse-probability weight as an expression."""
    d = pl.col(treatment).cast(pl.Boolean)
    if estimand == "att":
        return pl.when(d).then(1.0).otherwise(p / (1 - p))
    if estimand == "atc":
        return pl.when(d).then((1 - p) / p).otherwise(1.0)
    if estimand == "ate":
        return pl.when(d).then(1 / p).otherwise(1 / (1 - p))
    raise ValueError(f"estimand must be one of {ESTIMANDS}, not {estimand!r}")


def _dweight(treatment: str, p: pl.Expr, estimand: str) -> pl.Expr:
    """dw/dp for each row."""
    d = pl.col(treatment).cast(pl.Boolean)
    if estimand == "att":
        return pl.when(d).then(0.0).otherwise(1 / (1 - p) ** 2)
    if estimand == "atc":
        return pl.when(d).then(-1 / p**2).otherwise(0.0)
    return pl.when(d).then(-1 / p**2).otherwise(1 / (1 - p) ** 2)


def trim_bounds(trim) -> tuple[float, float] | None:
    """None, a (lo, hi) pair, or a number a meaning [a, 1 - a]."""
    if trim is None:
        return None
    lo, hi = (float(trim), 1 - float(trim)) if np.isscalar(trim) else map(float, trim)
    if not 0 <= lo < hi <= 1:
        raise ValueError(f"trim must give 0 <= lo < hi <= 1, not {trim!r}")
    return lo, hi


def ipw_weights(lf: pl.LazyFrame | pl.DataFrame, treatment: str, score, estimand: str = "att",
                *, trim=None, normalize: bool = True, name: str = "weight") -> pl.LazyFrame:
    """`lf` with the inverse-probability weight in column `name`, lazily.

    `score` is a column of probabilities or a `BinaryFirstStage`. Rows
    outside the `trim` bounds are dropped. With `normalize`, each group's
    weights are rescaled to sum to its number of rows (the Hajek scale;
    one streaming pass), else they are the raw weights in the table above.
    Scores of exactly 0 or 1 give infinite weights: trim them."""
    lf = lf.lazy()
    fs = isinstance(score, BinaryFirstStage)
    if fs:
        lf = score.with_eta(lf)
    p = score.prob() if fs else pl.col(score).cast(pl.Float64)
    lf = lf.with_columns(p.alias("__p"))
    b = trim_bounds(trim)
    if b is not None:
        lf = lf.filter((pl.col("__p") >= b[0]) & (pl.col("__p") <= b[1]))
    lf = lf.with_columns(ipw_weight(treatment, pl.col("__p"), estimand).alias(name))
    if normalize:
        g = pl.col(treatment).cast(pl.Boolean)
        lf = lf.with_columns((pl.col(name) * pl.len().over(g) / pl.col(name).sum().over(g))
                             .alias(name))
    return lf.drop("__p", *(["__eta"] if fs else []))


@dataclass
class IPWEstimate:
    """`estimate` (mean1 - mean0, Hajek), its `variance` (`IFVariance`,
    corrected for the estimated score when `corrected`), the derivative
    `J` of the estimate with respect to the score's coefficients (the
    covariates; not the fixed effects), and the
    sample after trimming (`n_obs`)."""

    estimate: float
    mean1: float
    mean0: float
    variance: IFVariance
    corrected: bool
    J: np.ndarray
    n_obs: int


def ipw_mean_difference(lf: pl.LazyFrame | pl.DataFrame, y: str, treatment: str,
                        first: BinaryFirstStage, estimand: str = "att", *, cluster=None,
                        trim=None, correct: bool = True,
                        df_k: int | None = None, workdir=None) -> IPWEstimate:
    """The Hajek IPW estimate of the `estimand` effect on `y`, with an
    influence-function variance (clustered on `cluster`) that accounts for
    the estimated score unless ``correct=False`` (then the weights are held
    fixed, as `weighted_mean_difference`). `first` must be the fitted score
    model, with ``first.outcome == treatment``, and `lf` its estimation
    sample (the first-stage term sums over it; trimmed rows contribute only
    that term). Small-sample factor: G/(G-1)
    (``df_k=None``, Stata's ``_robust``); see `if_vcov`. With absorbed fixed
    effects in the score, the correction runs a fixed-effect regression
    (`BinaryFirstStage.project`; scratch files under `workdir`)."""
    if first.outcome != treatment:
        raise ValueError("the first stage's outcome must be the treatment")
    if estimand not in ESTIMANDS:
        raise ValueError(f"estimand must be one of {ESTIMANDS}, not {estimand!r}")
    lf = first.with_eta(lf.lazy())
    p = first.prob()
    lf = lf.with_columns(p.alias("__p"))
    b = trim_bounds(trim)
    # trimmed rows weigh 0 but stay: the score was fitted on every row, so its
    # contributions s_i (and H) are over all of them
    kept = (pl.lit(True) if b is None
            else (pl.col("__p") >= b[0]) & (pl.col("__p") <= b[1]))
    d = pl.col(treatment).cast(pl.Boolean)
    Y = pl.col(y).cast(pl.Float64)
    w = pl.when(kept).then(ipw_weight(treatment, pl.col("__p"), estimand)).otherwise(0.0)
    lf = lf.with_columns(w.alias("__w"), kept.alias("__kept"))
    if not correct:
        lf = lf.filter(pl.col("__kept"))
    t = (lf.select(pl.col("__w").filter(d).sum().alias("W1"),
                   pl.col("__w").filter(~d).sum().alias("W0"),
                   (pl.col("__w") * Y).filter(d).sum().alias("S1"),
                   (pl.col("__w") * Y).filter(~d).sum().alias("S0"),
                   Y.filter(pl.col("__kept")).is_null().sum().alias("nulls"),
                   pl.col("__kept").sum().alias("n"),
                   pl.col("__w").is_infinite().sum().alias("inf"))
         .collect(engine="streaming").row(0, named=True))
    if t["nulls"]:
        raise ValueError(f"{t['nulls']:,} rows have a missing {y!r}")
    if t["inf"]:
        raise ValueError(f"{t['inf']:,} rows have a score of 0 or 1 (infinite weight); "
                         "use trim=")
    if not (t["W1"] > 0 and t["W0"] > 0):
        raise ValueError("both groups need a positive total weight")
    m1, m0 = t["S1"] / t["W1"], t["S0"] / t["W0"]
    phi = (pl.when(~pl.col("__kept")).then(0.0)
           .when(d).then(pl.col("__w") * (Y - m1) / t["W1"])
           .otherwise(-pl.col("__w") * (Y - m0) / t["W0"]).alias("__phi"))
    lf = lf.with_columns(phi)
    k = len(first.x)
    J = np.zeros(k)
    if correct:
        # dtheta/dgamma: d mu_g / d gamma = sum_g w' f x (y - mu_g) / W_g
        lf = first.with_parts(lf)
        dw = _dweight(treatment, pl.col("__p"), estimand) * pl.col("__f")
        c = (pl.when(~pl.col("__kept")).then(0.0)
             .when(d).then(dw * (Y - m1) / t["W1"]).otherwise(-dw * (Y - m0) / t["W0"]))
        # J over the covariates (reported); with fixed effects the correction
        # projects c itself (see mestimation)
        J = np.array(lf.select([(c * pl.col(f"__x{j}")).sum().alias(f"J{j}")
                                for j in range(k)])
                     .collect(engine="streaming").row(0))
        if first.fe:
            lf = corrected_contributions(lf, "__phi", first, name="__phi_c", c=c,
                                         workdir=workdir)
        else:
            lf = corrected_contributions(lf, "__phi", first, J, name="__phi_c")
        col = "__phi_c"
    else:
        col = "__phi"
    var = if_vcov(lf, [col], cluster, df_k=df_k)
    return IPWEstimate(estimate=m1 - m0, mean1=m1, mean0=m0, variance=var, corrected=correct,
                       J=J, n_obs=int(t["n"]))

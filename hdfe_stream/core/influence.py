"""Influence-function variances from streamed sums, with one-way and
multi-way clustering.

An estimator that is (asymptotically) a sum of per-row contributions,
``theta_hat - theta ~= sum_i phi_i``, has the sandwich variance

    V = c * sum_g (sum_{i in g} phi_i)(sum_{i in g} phi_i)'

over clusters g (each row its own cluster when there is no clustering).
Only per-cluster sums of phi are needed, so `if_vcov` is one streaming
``group_by`` per clustering term and holds G x p numbers, never the rows.
Here phi is the influence function divided by N: it is on the scale of the
estimate, and V needs no further 1/N^2.

Multi-way clustering is Cameron, Gelbach and Miller's (2011)
inclusion-exclusion: for dimensions A and B, V_A + V_B - V_(A and B), and so
on for more. The small-sample factor follows hdfe-stream (pyfixest's
defaults), so that an estimator that is also a least-squares coefficient
gets the regression's standard error exactly:

    c = G / (G - 1) * (N - 1) / (N - k)

with G the smallest one-way cluster count (G = N without clustering) and k
the parameters of that regression (`df_k`; None drops the second factor,
leaving Stata's ``_robust`` convention G / (G - 1)). Every term of a
multi-way sum takes the same c. A multi-way variance can have a negative
diagonal; it is returned as is, with a warning, as in hdfe-stream.

`ClusterScoreAccumulator` gives the same one-way variance from batches of
whole clusters (rows already grouped by cluster, as a panel store keeps
them), in p x p memory whatever the number of clusters.

`weighted_mean_difference` is the first consumer: the difference between
two groups' weighted means with the weights held fixed (matching weights,
IPW with a known score, ...).
"""

from __future__ import annotations

import itertools
import warnings
from dataclasses import dataclass

import numpy as np
import polars as pl


@dataclass
class IFVariance:
    """A variance from `if_vcov`: `vcov` (p x p), the rows `n_obs`, the
    one-way cluster counts `n_clusters` (by dimension; empty without
    clustering), the small-sample factor `adj` and the degrees of freedom
    `df` (smallest G minus 1, or N minus 1)."""

    vcov: np.ndarray
    n_obs: int
    n_clusters: dict
    adj: float
    df: int


def cluster_dims(cluster) -> list[str]:
    """``None`` -> [], ``"a"`` -> ["a"], ``"a+b"`` or ``["a", "b"]`` -> ["a", "b"]."""
    if cluster is None:
        return []
    if isinstance(cluster, str):
        dims = [c.strip() for c in cluster.split("+")]
    else:
        dims = [str(c).strip() for c in cluster]
    if not dims or any(not c for c in dims):
        raise ValueError(f"bad cluster specification {cluster!r}")
    if len(set(dims)) != len(dims):
        raise ValueError(f"a cluster dimension is repeated in {cluster!r}")
    return dims


def if_vcov(lf: pl.LazyFrame | pl.DataFrame, phi: list[str], cluster=None, *,
            df_k: int | None = None) -> IFVariance:
    """The sandwich variance of an estimator with per-row contributions in
    the columns `phi` of `lf` (see the module docstring), clustered on
    `cluster` (a column, ``"a+b"`` or a list for multi-way; None: by row).

    Rows with a null in a cluster column are an error (they belong to no
    cluster)."""
    lf = lf.lazy()
    dims = cluster_dims(cluster)
    p = len(phi)
    f = [pl.col(c).cast(pl.Float64) for c in phi]
    if dims:
        nulls = (lf.select([pl.col(d).null_count().alias(d) for d in dims])
                 .collect(engine="streaming").row(0, named=True))
        bad = {d: n for d, n in nulls.items() if n}
        if bad:
            raise ValueError(f"cluster columns have missing values: {bad}")
    n = int(lf.select(pl.len()).collect(engine="streaming").item())

    def meat(by: list[str] | None) -> tuple[np.ndarray, int]:
        if by is None:      # each row its own cluster: sum of outer products
            prods = [(f[a] * f[b]).sum().alias(f"{a}_{b}")
                     for a in range(p) for b in range(a, p)]
            vals = lf.select(prods).collect(engine="streaming").row(0)
            M = np.zeros((p, p))
            it = iter(vals)
            for a in range(p):
                for b in range(a, p):
                    M[a, b] = M[b, a] = next(it)
            return M, n
        s = (lf.group_by(by).agg([x.sum().alias(f"__s{j}") for j, x in enumerate(f)])
             .collect(engine="streaming"))
        A = s.select([f"__s{j}" for j in range(p)]).to_numpy()
        return A.T @ A, s.height

    V = np.zeros((p, p))
    counts = {}
    if not dims:
        V, G = meat(None)
    else:
        for r in range(1, len(dims) + 1):
            sign = 1.0 if r % 2 else -1.0
            for sub in itertools.combinations(dims, r):
                M, G_sub = meat(list(sub))
                V += sign * M
                if r == 1:
                    counts[sub[0]] = G_sub
        G = min(counts.values())
    adj = G / (G - 1) if G > 1 else np.nan
    if df_k is not None:
        adj *= (n - 1) / (n - df_k) if n > df_k else np.nan
    V = adj * (V + V.T) / 2
    if np.any(np.diag(V) < 0):
        warnings.warn("the multi-way clustered variance has negative diagonal entries",
                      stacklevel=2)
    return IFVariance(vcov=V, n_obs=n, n_clusters=counts, adj=float(adj), df=G - 1)


class ClusterScoreAccumulator:
    """`if_vcov`'s one-way clustered variance, accumulated from batches that
    each hold whole clusters. It keeps a p x p matrix whatever the number of
    clusters, where `if_vcov`'s ``group_by`` holds every cluster's sums at
    once (2 million clusters, p = 20, fed by `iter_groups`: 0.8 GB above
    the baseline against 4.5 GB, and no slower); it
    needs the rows grouped by cluster, which `if_vcov` does not.

    Feed it batch by batch, with either

    - ``add_rows(cluster, *blocks)``: a batch's rows and their cluster keys
      (an array or a Polars Series), each cluster's rows consecutive and
      none continuing into another batch, as `core.store.iter_groups`
      yields them; the rows are summed within cluster here; or
    - ``add_clusters(*blocks, n_obs=None)``: rows that are already one
      cluster's sums each (`n_obs`: the observations behind them, for the
      small-sample factor; default one per cluster). Without clustering,
      pass the rows themselves.

    A block is an (n, width) array, or 1-d for width 1. The blocks of a
    call are placed side by side, so the variance's columns are the blocks'
    columns in order, cross-block entries included (an estimator's scores
    and its weights' scores, say). Every call takes the same total width.

    ``variance(df_k=None)`` returns an `IFVariance` with `if_vcov`'s
    small-sample factor; `sums` is the matrix without it. `name` labels
    `IFVariance.n_clusters` (None: an empty dict, as without clustering).

    The check that clusters do not span batches is partial: a batch whose
    first key equals the previous batch's last key, or a key that appears
    in two separate runs of one batch, is an error; a cluster that comes
    back in a later batch is not detected (it would count as two clusters)."""

    def __init__(self, name: str | None = None):
        self.name = name
        self.n_clusters = 0
        self.n_obs = 0
        self._M: np.ndarray | None = None
        self._last = None
        self._started = False

    @staticmethod
    def _stack(blocks) -> np.ndarray:
        if not blocks:
            raise ValueError("no score blocks given")
        cols = [np.asarray(b, dtype=np.float64) for b in blocks]
        cols = [c.reshape(len(c), -1) for c in cols]
        if len({c.shape[0] for c in cols}) != 1:
            raise ValueError("score blocks have different numbers of rows")
        return cols[0] if len(cols) == 1 else np.hstack(cols)

    def _accumulate(self, S: np.ndarray) -> None:
        if self._M is None:
            self._M = np.zeros((S.shape[1], S.shape[1]))
        elif S.shape[1] != self._M.shape[0]:
            raise ValueError(f"score blocks have total width {S.shape[1]}, "
                             f"earlier ones {self._M.shape[0]}")
        self._M += S.T @ S

    def add_clusters(self, *blocks, n_obs: int | None = None) -> None:
        """Add rows that are one cluster's score sums each."""
        S = self._stack(blocks)
        self._accumulate(S)
        self.n_clusters += S.shape[0]
        self.n_obs += S.shape[0] if n_obs is None else int(n_obs)

    def add_rows(self, cluster, *blocks) -> None:
        """Add a batch of rows of whole clusters, keys in `cluster`."""
        if hasattr(cluster, "null_count") and cluster.null_count():
            raise ValueError("cluster keys have missing values")
        keys = cluster.to_numpy() if hasattr(cluster, "to_numpy") else np.asarray(cluster)
        A = self._stack(blocks)
        if keys.shape[0] != A.shape[0]:
            raise ValueError("cluster keys and score blocks have different numbers of rows")
        if not keys.shape[0]:
            return
        starts = np.r_[0, np.flatnonzero(keys[1:] != keys[:-1]) + 1]
        if np.unique(keys[starts]).shape[0] != starts.shape[0]:
            raise ValueError("a cluster's rows are not consecutive within the batch")
        if self._started and keys[0] == self._last:
            raise ValueError("a cluster continues from the previous batch; give whole "
                             "clusters per batch (core.store.iter_groups)")
        self._started, self._last = True, keys[-1]
        self._accumulate(np.add.reduceat(A, starts, axis=0))
        self.n_clusters += starts.shape[0]
        self.n_obs += keys.shape[0]

    @property
    def sums(self) -> np.ndarray:
        """sum over clusters of s_g s_g', without the small-sample factor."""
        if self._M is None:
            raise ValueError("no scores were added")
        return (self._M + self._M.T) / 2

    def variance(self, df_k: int | None = None) -> IFVariance:
        """The variance, with `if_vcov`'s small-sample factor (`df_k` as there)."""
        M = self.sums
        G, n = self.n_clusters, self.n_obs
        adj = G / (G - 1) if G > 1 else np.nan
        if df_k is not None:
            adj *= (n - 1) / (n - df_k) if n > df_k else np.nan
        return IFVariance(vcov=adj * M, n_obs=n,
                          n_clusters={self.name: G} if self.name is not None else {},
                          adj=float(adj), df=G - 1)


@dataclass
class WeightedMeanDifference:
    """`estimate` = mean1 - mean0 (weighted means of the two groups), its
    `variance` (an `IFVariance`), and the group totals of the weights."""

    estimate: float
    mean1: float
    mean0: float
    weight1: float
    weight0: float
    variance: IFVariance


def weighted_mean_difference(lf: pl.LazyFrame | pl.DataFrame, y: str, w: str, group: str,
                             cluster=None, *, df_k: int | None = 2,
                             allow_negative: bool = False) -> WeightedMeanDifference:
    """The difference of the weighted means of `y` between rows with `group`
    true (1) and false (0), with weights `w` held fixed, and its
    influence-function variance (`if_vcov`, clustered on `cluster`).

    Each row contributes phi_i = w_i (y_i - mean1) / W1 in group 1 and
    -w_i (y_i - mean0) / W0 in group 0 (W: the group's total weight). This is
    the coefficient on `group` in a weighted least-squares regression of `y`
    on (1, `group`) and, with the default ``df_k=2``, its HC1 / CRV1
    standard error exactly (as hdfe-stream, pyfixest, or R's
    ``sandwich::vcovCL``). Rows with zero weight change nothing but N in the
    small-sample factor; leave them out. Negative weights (ridge or
    local-linear matching) are an error unless `allow_negative`; each
    group's total must be positive. Two streaming passes: the group totals,
    then the clustered sums."""
    lf = lf.lazy()
    g = pl.col(group).cast(pl.Boolean)
    W, Y = pl.col(w).cast(pl.Float64), pl.col(y).cast(pl.Float64)
    t = (lf.select((W * g).sum().alias("W1"), (W * ~g).sum().alias("W0"),
                   (W * Y * g).sum().alias("S1"), (W * Y * ~g).sum().alias("S0"),
                   (W.is_null() | Y.is_null() | pl.col(group).is_null()).sum().alias("nulls"),
                   (W < 0).sum().alias("neg"))
         .collect(engine="streaming").row(0, named=True))
    if t["nulls"]:
        raise ValueError(f"{t['nulls']:,} rows have a missing {y!r}, {w!r} or {group!r}")
    if t["neg"] and not allow_negative:
        raise ValueError(f"{t['neg']:,} rows have a negative weight")
    if not (t["W1"] > 0 and t["W0"] > 0):
        raise ValueError("both groups need a positive total weight")
    m1, m0 = t["S1"] / t["W1"], t["S0"] / t["W0"]
    phi = (pl.when(g).then(W * (Y - m1) / t["W1"])
           .otherwise(-W * (Y - m0) / t["W0"]).alias("__phi"))
    var = if_vcov(lf.with_columns(phi), ["__phi"], cluster, df_k=df_k)
    return WeightedMeanDifference(estimate=m1 - m0, mean1=m1, mean0=m0, weight1=t["W1"],
                                  weight0=t["W0"], variance=var)


def group_mean_phis(lf: pl.LazyFrame | pl.DataFrame, y: str, w: str, group: str,
                    wmd: WeightedMeanDifference) -> pl.DataFrame:
    """The per-row influence contributions behind `weighted_mean_difference`
    (`wmd` is its result for the same columns): ``phi1`` for the mean of group
    1 (w_i (y_i - mean1) / W1 on group-1 rows, else 0) and ``phi0`` for the
    mean of group 0. The difference's contribution is ``phi1 - phi0``."""
    g = pl.col(group).cast(pl.Boolean)
    W, Y = pl.col(w).cast(pl.Float64), pl.col(y).cast(pl.Float64)
    return lf.lazy().select(
        pl.when(g).then(W * (Y - wmd.mean1) / wmd.weight1).otherwise(0.0).alias("phi1"),
        pl.when(~g).then(W * (Y - wmd.mean0) / wmd.weight0).otherwise(0.0).alias("phi0"),
    ).collect(engine="streaming")


def joint_if_vcov(parts: list[tuple[pl.DataFrame, dict[int, str]]], p: int, cluster,
                  factors: np.ndarray) -> IFVariance:
    """The joint variance of p estimators that live on different tables.

    Each part is ``(table, {j: column})``: a table holding the cluster
    columns and, for each estimator j it contributes to, its per-row
    influence contribution in `column`. Estimator j has contributions in one
    part only, so a row of another part adds zero to it; the tables may share
    clusters (the same unit, the same source column value), which is what
    makes the estimators covary. The sums per cluster are those of `if_vcov`
    over the stacked tables, without its small-sample factor; entry (a, b) is
    then scaled by sqrt(factors[a] * factors[b]), where `factors[j]` is the
    factor of estimator j's own fit (`IFVariance.adj`), so that the diagonal
    equals each estimator's own variance exactly. The returned `adj` is 1."""
    dims = cluster_dims(cluster)
    keep = [d for d in dims if d != "__unit"] + (["__unit"] if "__unit" in dims else [])
    frames = []
    for tab, cols in parts:
        frames.append(tab.select(*keep, *(pl.col(c).alias(f"__phi{j}") for j, c in cols.items())))
    stack = pl.concat(frames, how="diagonal").with_columns(
        [pl.col(f"__phi{j}").fill_null(0.0) for j in range(p)])
    raw = if_vcov(stack, [f"__phi{j}" for j in range(p)], dims or None, df_k=None)
    f = np.sqrt(np.asarray(factors, dtype=float))
    return IFVariance(vcov=raw.vcov / raw.adj * np.outer(f, f), n_obs=raw.n_obs,
                      n_clusters=raw.n_clusters, adj=1.0, df=raw.df)

"""The wild cluster bootstrap, from per-cluster sums kept at fit time.

A fit with ``bootcluster=`` keeps, for each cluster g of that clustering,
H_g = sum w h h' and s_g = sum w e h over its rows, where h are the covariates
with the fixed effects removed and e the residuals (`HDFEResult._boot`). With
every fixed effect nested within the clusters, these are all the wild cluster
bootstrap needs (Roodman, MacKinnon, Nielsen and Webb 2019, "Fast and
wild"). For a test of beta_j = 0, with A = (sum_g H_g)^-1:

    restricted fit       delta = beta - beta_r = A_j beta_j / A_jj
    its scores           s_g = s_g + H_g delta     (s_g itself if unrestricted)
    a draw v (one weight per cluster), w = sum_g v_g s_g:
    numerator            R (beta* - beta_r) = sum_g v_g a_g,    a_g = A_j' s_g
    cluster scores       v_g a_g - c_g' w,                       c_g = A H_g A_j

so a draw costs O(G k), the denominator is the CRV1 variance from those
scores (with the fit's small-sample factor, which cancels from the p-value),
and nothing is read from disk.

The weights are drawn as the `wildboottest` package draws them (which
pyfixest's `wildboottest` uses): one G x B matrix from
``numpy.random.default_rng(seed)``, the clusters in sorted order, or every
sign pattern when 2^G < B with Rademacher weights. The same seed therefore
gives the same draws, and so the same p-values, up to rounding. When G x B
is too large to draw at once, the draws come in blocks from the same
generator, which no longer matches but is still a valid bootstrap.
"""

from __future__ import annotations

import warnings
from itertools import product

import numpy as np
import polars as pl

# at most this many weights in memory at once (8 bytes each)
_BLOCK = 1 << 25


def _rademacher(n, rng):
    return rng.choice([-1, 1], size=n, replace=True)


def _mammen(n, rng):
    return rng.choice(np.array([-1, 1]) * (np.sqrt(5) + np.array([-1, 1])) / 2, size=n,
                      replace=True, p=(np.sqrt(5) + np.array([1, -1])) / (2 * np.sqrt(5)))


def _webb(n, rng):
    return rng.choice(np.concatenate([-np.sqrt(np.array([3, 2, 1]) / 2),
                                      np.sqrt(np.array([1, 2, 3]) / 2)]),
                      replace=True, size=n)


def _norm(n, rng):
    return rng.normal(size=n)


WEIGHTS = {"rademacher": _rademacher, "mammen": _mammen, "webb": _webb, "norm": _norm}


def _draws(weights_type, G, reps, seed):
    """Yield blocks of weights, G x (draws in the block), and say whether
    they are a full enumeration."""
    if weights_type not in WEIGHTS:
        raise ValueError(f"weights_type must be one of {sorted(WEIGHTS)}, got "
                         f"{weights_type!r}")
    if weights_type == "rademacher" and 2 ** G < reps:
        yield np.array(list(product([-1, 1], repeat=G)), dtype=float).T
        return
    fun, rng = WEIGHTS[weights_type], np.random.default_rng(seed)
    per = reps if G * reps <= _BLOCK else max(1, _BLOCK // G)
    left = reps
    while left:
        b = min(per, left)
        yield np.asarray(fun(G * b, rng), dtype=float).reshape(G, b)
        left -= b


class _Test:
    """One model's test of beta_j = 0, ready for draws."""

    def __init__(self, res, j, impose_null):
        boot = res._boot
        AG, sG, adj = boot["AG"], boot["sG"], boot["adj"]
        A = np.linalg.inv(AG.sum(axis=0))
        Aj = A[:, j]
        beta = res.beta
        S = sG + np.einsum("gab,b->ga", AG, Aj * (beta[j] / A[j, j])) if impose_null else sG
        self.a = S @ Aj
        self.S = S
        self.C = np.einsum("gab,b->ga", AG, Aj) @ A
        self.adj = adj
        self.t = beta[j] / np.sqrt(adj * np.sum((sG @ Aj) ** 2))

    def tstar(self, V):
        """Bootstrap t statistics for the draws V (G x b)."""
        Vt = V.T
        num = Vt @ self.a
        U = Vt * self.a - (Vt @ self.S) @ self.C.T
        return num / np.sqrt(self.adj * np.einsum("bg,bg->b", U, U))


def _bootstrap(tests, reps, weights_type, impose_null, seed):
    """Bootstrap t statistics for several (result, coefficient) tests, all on
    the same draws: each cluster gets the same weight in every model.

    Returns the t statistics (S,), the bootstrap ones (B x S), and whether
    the draws were a full enumeration.
    """
    cols = tests[0][0]._boot["cols"]
    for res, _ in tests:
        if res._boot is None:
            raise ValueError(f"{res.fml}: fit with bootcluster= to bootstrap it")
        if res._boot["cols"] != cols:
            raise ValueError("every model must be bootstrapped on the same clustering; "
                             f"got {cols} and {res._boot['cols']}")
    # the union of the models' clusters, sorted, is the order the weights are
    # drawn in; each model takes the rows of its own clusters
    union = (pl.concat([res._boot["levels"] for res, _ in tests], how="vertical_relaxed")
             .unique().sort(cols).with_row_index("__u"))
    rows = []
    for res, _ in tests:
        lv = res._boot["levels"].with_row_index("__m")
        rows.append(lv.join(union, on=cols, how="left").sort("__m")["__u"].to_numpy())
    prepared = [_Test(res, j, impose_null) for res, j in tests]
    blocks = []
    enumerated = weights_type == "rademacher" and 2 ** union.height < reps
    for V in _draws(weights_type, union.height, reps, seed):
        blocks.append(np.column_stack([p.tstar(V[r]) for p, r in zip(prepared, rows)]))
    if enumerated:
        warnings.warn(f"2^G = {2 ** union.height:,} is fewer than reps={reps:,}: every "
                      "Rademacher sign pattern is used once instead", stacklevel=3)
    return np.array([p.t for p in prepared]), np.vstack(blocks), enumerated


def _coef_index(res, param):
    if param not in res.coefnames:
        raise ValueError(f"{param!r} is not a coefficient of {res.fml}: {res.coefnames}")
    return res.coefnames.index(param)

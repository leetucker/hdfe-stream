"""Sensitivity of an event study to violations of parallel trends
(Rambachan and Roth, 2023, *A More Credible Approach to Parallel Trends*,
Review of Economic Studies 90(5)): confidence intervals for a post-treatment
effect that remain valid when the trend difference after treatment is only
restricted to resemble the one before it.

The inputs are an event study's coefficients `beta`, ordered from the
earliest pre-treatment period to the last post-treatment period with the
reference period omitted, and their covariance. Nothing else about the model
or the data enters, so the same code serves any estimator that produces an
event study with a joint covariance: a two-way fixed effects regression with
leads and lags, or the staggered-adoption estimators of diff-stream.

The restrictions, their sign and monotonicity variants, the four confidence
intervals and their defaults are those of the R package HonestDiD 0.2.8 by
the method's authors, which this module follows step by step; the guide
(docs/reference/sensitivity.md) lists the correspondence and the few
differences. The conditional tests solve their linear programs with the
warm-started simplex of `_sensitivity_kernels`, which is where the speed
over the R package comes from.
"""

from __future__ import annotations

import math
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import polars as pl
from scipy import optimize, special, stats

from . import _sensitivity_kernels as K
from ._sensitivity_flci import optimal_flci
from .results_base import _json_safe

RESTRICTIONS = ("smoothness", "relative_magnitudes", "smoothness_relative_magnitudes")
CI_METHODS = ("fixed_length", "conditional", "hybrid_fixed_length", "hybrid_least_favorable")
_R_METHOD = {"fixed_length": "FLCI", "conditional": "Conditional",
             "hybrid_fixed_length": "C-F", "hybrid_least_favorable": "C-LF"}


class TrendSensitivityWarning(UserWarning):
    """A sensitivity interval is unreliable as computed: every value of the
    target was rejected, or the accepted set reached the edge of the search."""


# --------------------------------------------------------------------------
# restriction sets: A delta <= d, delta = (delta_pre, delta_post), the
# reference period omitted (R's .create_A_* functions)
# --------------------------------------------------------------------------

def _drop_zero_rows(A):
    return A[(A * A).sum(axis=1) > 1e-10]


def _A_SD(n_pre, n_post):
    n = n_pre + n_post
    At = np.zeros((n - 1, n + 1))
    for r in range(n - 1):
        At[r, r:r + 3] = (1, -2, 1)
    At = np.delete(At, n_pre, axis=1)
    return np.vstack([At, -At])


def _A_B(n_pre, n_post, bias):
    A = -np.eye(n_pre + n_post)[n_pre:]
    return -A if bias == "negative" else A


def _A_M(n_pre, n_post, monotonicity):
    n = n_pre + n_post
    A = np.zeros((n, n))
    for r in range(n_pre - 1):
        A[r, r:r + 2] = (1, -1)
    A[n_pre - 1, n_pre - 1] = 1
    if n_post > 0:
        A[n_pre, n_pre] = -1
        for r in range(n_pre + 1, n):
            A[r, r - 1:r + 1] = (1, -1)
    return -A if monotonicity == "decreasing" else A


def _A_RM(n_pre, n_post, Mbar, s, positive):
    n = n_pre + n_post
    At = np.zeros((n, n + 1))
    for r in range(n):
        At[r, r:r + 2] = (-1, 1)
    v = np.zeros(n + 1)
    v[n_pre + s - 1:n_pre + s + 1] = (-1, 1)
    if not positive:
        v = -v
    UB = np.vstack([np.tile(v, (n_pre, 1)), np.tile(Mbar * v, (n_post, 1))])
    A = _drop_zero_rows(np.vstack([At - UB, -At - UB]))
    return np.delete(A, n_pre, axis=1)


def _A_SDRM(n_pre, n_post, Mbar, s, positive):
    n = n_pre + n_post
    At = np.zeros((n - 1, n + 1))
    for r in range(n - 1):
        At[r, r:r + 3] = (1, -2, 1)
    v = np.zeros(n + 1)
    v[n_pre + s - 2:n_pre + s + 1] = (1, -2, 1)
    if not positive:
        v = -v
    UB = np.vstack([np.tile(v, (n_pre - 1, 1)), np.tile(Mbar * v, (n_post, 1))])
    A = _drop_zero_rows(np.vstack([At - UB, -At - UB]))
    return np.delete(A, n_pre, axis=1)


@dataclass
class _Piece:
    """One polyhedron of the restriction (the restriction itself for the
    smoothness family; one choice of the period with the largest pre-period
    violation, and its sign, for the relative-magnitudes families)."""
    A: np.ndarray
    d: np.ndarray


def _pieces(restriction, n_pre, n_post, M, bias, monotonicity):
    extra_A, extra_d = np.zeros((0, n_pre + n_post)), np.zeros(0)
    if bias is not None:
        extra_A = _A_B(n_pre, n_post, bias)
    elif monotonicity is not None:
        extra_A = _A_M(n_pre, n_post, monotonicity)
    extra_d = np.zeros(len(extra_A))
    if restriction == "smoothness":
        A = _A_SD(n_pre, n_post)
        return [_Piece(np.vstack([A, extra_A]), np.concatenate([np.full(len(A), float(M)), extra_d]))]
    build, smin = ((_A_RM, -(n_pre - 1)) if restriction == "relative_magnitudes"
                   else (_A_SDRM, -(n_pre - 2)))
    out = []
    for s in range(smin, 1):
        for positive in (True, False):
            A = build(n_pre, n_post, M, s, positive)
            out.append(_Piece(np.vstack([A, extra_A]), np.concatenate([np.zeros(len(A)), extra_d])))
    return out


def _identified_set_at_zero(piece, n_pre, l):
    """R's .compute_IDset_* with trueBeta = 0: the range of -l'delta_post over
    the restriction with delta_pre = 0 (used only to place the grid)."""
    Ap = piece.A[:, n_pre:]
    keep = np.abs(Ap).sum(axis=1) > 0
    Ap, d = Ap[keep], piece.d[keep]
    bounds = [(None, None)] * Ap.shape[1]
    hi = optimize.linprog(-l, A_ub=Ap, b_ub=d, bounds=bounds, method="highs")
    lo = optimize.linprog(l, A_ub=Ap, b_ub=d, bounds=bounds, method="highs")
    if hi.status != 0 and lo.status != 0:
        return 0.0, 0.0
    lmax = -hi.fun if hi.status == 0 else 0.0
    lmin = lo.fun if lo.status == 0 else 0.0
    return -lmax, -lmin


# --------------------------------------------------------------------------
# the conditional tests over a grid of the target
# --------------------------------------------------------------------------

def _psd_root(S):
    vals, vecs = np.linalg.eigh((S + S.T) / 2)
    return (vecs * np.sqrt(np.clip(vals, 0, None))) @ vecs.T


def _construct_gamma(l):
    """R's .construct_Gamma: l and the identity columns that the reduced row
    echelon form of [l, I] selects as pivots, as rows."""
    T = len(l)
    cols = [l] + [np.eye(T)[:, j] for j in range(T)]
    chosen = []
    for c in cols:
        trial = np.column_stack(chosen + [c])
        if np.linalg.matrix_rank(trial, tol=1e-10) == len(chosen) + 1:
            chosen.append(c)
        if len(chosen) == T:
            break
    return np.column_stack(chosen).T


def _dual_set(sd, X):
    """The dual feasible set {gamma >= 0, sd'gamma = 1, X'gamma = 0} as
    (WT, b, scale, basis); None when it is empty (the primal program is
    unbounded below, which R treats as a solver failure: not rejecting)."""
    scale = float(sd.max())
    if X.shape[1]:
        U, s, _ = np.linalg.svd(X, full_matrices=False)
        r = int((s > max(X.shape) * np.finfo(float).eps * s[0]).sum()) if s.size and s[0] > 0 else 0
        Xc = X if r == X.shape[1] else U[:, :r]
    else:
        Xc = X
    WT = np.ascontiguousarray(np.column_stack([sd / scale, Xc]))
    b = np.zeros(WT.shape[1])
    b[0] = 1.0
    status, basis = K.phase_one(WT, b)
    if status != K.OPTIMAL:
        return None
    return WT, b, scale, basis


def _truncnorm_quantile(p, lo, up):
    lo = np.asarray(lo, float)
    up = np.asarray(up, float)
    out = np.empty_like(lo)
    narrow = up - lo <= 1e-12
    out[narrow] = lo[narrow]
    w = ~narrow
    if w.any():
        out[w] = stats.truncnorm.ppf(p, lo[w], up[w])
    return out


def _lf_cv_plain(SA, kappa, simulations, seed):
    """The least-favorable critical value without nuisance parameters (R's
    .compute_least_favorable_cv with X_T = NULL): the 1 - kappa quantile of
    the largest standardized moment under the null, from simulated draws."""
    draws = np.random.default_rng(seed).standard_normal((simulations, len(SA))) @ _psd_root(SA)
    draws /= np.sqrt(np.diag(SA))
    return float(np.quantile(draws.max(axis=1), 1 - kappa))


def _lf_cv_arp(SA, XA, dual, kappa, simulations, seed):
    """The least-favorable critical value with nuisance parameters: the
    1 - kappa quantile of eta under the least-favorable null, one linear
    program per simulated draw (failed programs dropped, as in R)."""
    WT, b, scale, basis = dual
    draws = np.random.default_rng(seed).standard_normal((simulations, len(SA))) @ _psd_root(SA)
    etas = K.lf_etas(WT, b, np.ascontiguousarray(draws), scale, basis)
    return float(np.nanquantile(etas, 1 - kappa))


@dataclass
class _Setup:
    """A piece prepared for testing: the moments, their covariance, the
    hybrid's first stage, and the least-favorable critical value."""
    nuisance: bool
    hybrid: int
    size: float
    payload: dict = field(default_factory=dict)


def _setup_piece(piece, beta, sigma, n_pre, n_post, l, family, method, alpha, kappa,
                 simulations, seed, flci_cache, lf_cache):
    A, d = piece.A, piece.d
    hybrid = {"conditional": K.ARP, "hybrid_least_favorable": K.LEAST_FAVORABLE,
              "hybrid_fixed_length": K.FIXED_LENGTH}[method]
    size = alpha if hybrid == K.ARP else (alpha - kappa) / (1 - kappa)
    if n_post == 1:
        if family != "smoothness":
            keep = A[:, -1] != 0
            A, d = A[keep], d[keep]
        pay = dict(A=np.ascontiguousarray(A), d=np.ascontiguousarray(d), lf_cv=0.0,
                   fs_l=np.zeros(n_pre + n_post), fs_h=0.0)
        if hybrid == K.LEAST_FAVORABLE:
            SA = A @ sigma @ A.T
            key = ("plain", SA.tobytes())
            if key not in lf_cache:
                lf_cache[key] = _lf_cv_plain(SA, kappa, simulations, seed)
            pay["lf_cv"] = lf_cache[key]
        elif hybrid == K.FIXED_LENGTH:
            vec, half = flci_cache()
            pay["fs_l"], pay["fs_h"] = vec, half
        return _Setup(False, hybrid, size, pay)
    rows = np.flatnonzero(np.abs(A[:, n_pre:]).sum(axis=1) > 0)
    Gam = _construct_gamma(l)
    AG = A[:, n_pre:] @ np.linalg.inv(Gam)
    a = np.ascontiguousarray(AG[:, 0])
    X = np.ascontiguousarray(AG[:, 1:])
    Y0 = A @ beta - d
    SY = A @ sigma @ A.T
    SA = SY[np.ix_(rows, rows)]
    sd = np.sqrt(np.diag(SA))
    dual = _dual_set(sd, X[rows])
    pay = dict(Y0=Y0, a=a, rows=rows.astype(np.int64), X=X, SY=np.ascontiguousarray(SY),
               lf_cv=0.0, vbar=np.zeros(len(A)), half=0.0, vbar_d=0.0, vbar_a=0.0, dual=dual)
    if dual is None:
        warnings.warn("the conditional test's linear program is unbounded for one polyhedron of "
                      "the restriction, so that polyhedron rejects no value of the target (R: "
                      "'LP for eta did not converge properly. Not rejecting')",
                      TrendSensitivityWarning, stacklevel=4)
        return _Setup(True, hybrid, size, pay)
    WT, b, scale, basis = dual
    if hybrid == K.LEAST_FAVORABLE:
        key = ("arp", SA.tobytes(), X[rows].tobytes())
        if key not in lf_cache:
            lf_cache[key] = _lf_cv_arp(SA, X[rows], dual, kappa, simulations, seed)
        pay["lf_cv"] = lf_cache[key]
    elif hybrid == K.FIXED_LENGTH:
        vec, half = flci_cache()
        vbar = np.linalg.lstsq(A.T, vec, rcond=None)[0]
        pay.update(vbar=vbar, half=half, vbar_d=float(vbar @ d), vbar_a=float(vbar @ a))
    return _Setup(True, hybrid, size, pay)


def _test_grid(setup, thetas, beta, sigma, n_pre):
    """1 where the test accepts theta, 0 where it rejects."""
    p = setup.payload
    if not setup.nuisance:
        kind, stat, mu, sd, zlo, zup = K.plain_path(
            beta, sigma, p["A"], p["d"], n_pre, thetas, setup.hybrid, p["lf_cv"], p["fs_l"], p["fs_h"])
        accept = kind == K.ACCEPT
        q = kind == K.QUANTILE
        if q.any():
            z = _truncnorm_quantile(1 - setup.size, zlo[q], zup[q])
            with np.errstate(invalid="ignore"):
                cv = np.maximum(0, mu[q] + sd[q] * z)
                accept[q] = ~(stat[q] > cv)
        return accept
    if p["dual"] is None:
        return np.ones(len(thetas), bool)
    WT, b, scale, basis = p["dual"]
    kind, ms, zlo, zup = K.arp_path(
        p["Y0"], p["a"], p["rows"], p["X"], p["SY"], thetas, setup.hybrid, p["lf_cv"],
        p["vbar"], p["half"], p["vbar_d"], p["vbar_a"], WT, b, scale, basis)
    accept = kind == K.ACCEPT
    q = kind == K.QUANTILE
    if q.any():
        z = _truncnorm_quantile(1 - setup.size, zlo[q], zup[q])
        with np.errstate(invalid="ignore"):
            accept[q] = ~(ms[q] > np.maximum(0, z))
    return accept


# --------------------------------------------------------------------------
# the result
# --------------------------------------------------------------------------

@dataclass
class TrendSensitivityResult:
    """Confidence intervals for a post-treatment effect under restrictions on
    how far parallel trends may fail, one per value of M.

    Attributes
    ----------
    table : polars.DataFrame
        ``M``, ``lower`` and ``upper``: the robust confidence interval at each
        value of M (M-bar for the relative-magnitudes restrictions).
    original : dict
        The usual interval under exact parallel trends: ``estimate``, ``se``,
        ``lower``, ``upper``.
    restriction, bias, monotonicity, ci_method : str or None
        The restriction on the violations and the interval, as passed.
    alpha : float
        One minus the coverage.
    target : numpy.ndarray
        The weights l on the post-treatment coefficients: the target is
        l'tau_post.
    n_pre, n_post : int
        Pre- and post-treatment coefficients used.
    pre, post : list
        Their labels (event times or coefficient names), when known.
    r_names : dict
        The R package's names for the restriction (``Delta``) and the
        interval (``method``), for comparison with its output.
    diagnostics : dict
        Grid size, the grid's bounds for each M and whether it was extended.
    """

    table: pl.DataFrame
    original: dict
    restriction: str
    ci_method: str
    alpha: float
    target: np.ndarray
    n_pre: int
    n_post: int
    bias: str | None = None
    monotonicity: str | None = None
    pre: list = field(default_factory=list)
    post: list = field(default_factory=list)
    r_names: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)
    # per M, the tests' decisions on R's grid (before any extension)
    _grid_accept: list = field(default_factory=list, repr=False)

    @property
    def M(self) -> np.ndarray:
        """The values of M, in order."""
        return self.table["M"].to_numpy()

    @property
    def breakdown(self) -> float | None:
        """The largest M in the grid up to which every robust interval
        excludes zero: the effect stays significant for violations up to this
        size. None when the interval at the smallest M already includes zero.
        It is read off the grid of M values; a finer grid refines it."""
        out = None
        for M, lo, hi in self.table.select("M", "lower", "upper").iter_rows():
            if not (lo > 0 or hi < 0):
                break
            out = M
        return out

    def summary_dict(self) -> dict:
        """The result as plain Python types (JSON-safe)."""
        return _json_safe({
            "restriction": self.restriction, "bias": self.bias,
            "monotonicity": self.monotonicity, "ci_method": self.ci_method,
            "alpha": self.alpha, "target": self.target, "pre": self.pre, "post": self.post,
            "original": self.original, "breakdown": self.breakdown,
            "intervals": self.table.to_dicts(), "r_names": self.r_names,
            "diagnostics": self.diagnostics,
        })

    def summary_text(self) -> str:
        """A plain-text report."""
        what = self.restriction.replace("_", " ")
        extra = f", {self.bias} bias" if self.bias else (
            f", {self.monotonicity} trend" if self.monotonicity else "")
        lines = [
            f"Sensitivity to parallel trends: {what}{extra}",
            f"Interval: {self.ci_method.replace('_', ' ')} (R: {self.r_names.get('method')}), "
            f"{(1 - self.alpha) * 100:g}%; {self.n_pre} pre- and {self.n_post} post-treatment coefficients",
            f"Original: {self.original['estimate']:.6g} [{self.original['lower']:.6g}, "
            f"{self.original['upper']:.6g}]",
            f"Breakdown M: {'none' if self.breakdown is None else f'{self.breakdown:.6g}'}",
            str(self.table),
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary_text()

    def plot(self, ax: Any = None, *, title: str | None = None):
        """The robust intervals against M, with the original interval at the
        left, as in R's createSensitivityPlot. Needs matplotlib (not a
        dependency of this package: ``pip install hdfe-stream[plot]``).
        Draws on `ax` if given; returns the figure."""
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError(
                "TrendSensitivityResult.plot needs matplotlib, which is not installed; "
                "install it with `pip install matplotlib` (or the `plot` extra, "
                "`pip install hdfe-stream[plot]`)") from exc
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 4))
        else:
            fig = ax.figure
        M = self.M
        gap = (M[1] - M[0]) if len(M) > 1 else (abs(M[0]) or 1.0)
        x0 = M[0] - gap
        o = self.original
        ax.errorbar([x0], [o["estimate"]], yerr=[[o["estimate"] - o["lower"]], [o["upper"] - o["estimate"]]],
                    fmt="none", capsize=4, color="#d95f02", label="Original")
        lo, hi = self.table["lower"].to_numpy(), self.table["upper"].to_numpy()
        mid = (lo + hi) / 2
        ax.errorbar(M, mid, yerr=[mid - lo, hi - mid], fmt="none", capsize=4, color="#1b9e77",
                    label=self.r_names.get("method", self.ci_method))
        ax.axhline(0, color="0.6", linewidth=0.8)
        ax.set_xlabel("M-bar" if self.restriction != "smoothness" else "M")
        ax.set_ylabel(f"{(1 - self.alpha) * 100:g}% confidence interval")
        ax.legend(frameon=False)
        ax.set_title(title if title is not None else self.restriction.replace("_", " "))
        return fig


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------

def _r_delta(restriction, bias, monotonicity):
    base = {"smoothness": "DeltaSD", "relative_magnitudes": "DeltaRM",
            "smoothness_relative_magnitudes": "DeltaSDRM"}[restriction]
    if bias:
        return base + ("PB" if bias == "positive" else "NB")
    if monotonicity:
        return base + ("I" if monotonicity == "increasing" else "D")
    return base


def default_M(beta, vcov, n_pre, restriction):
    """R's default grids: for smoothness, 10 values from 0 to the 95% upper
    bound of the largest pre-period second difference (to the first
    pre-period coefficient's standard error when there is one pre-period);
    for the relative-magnitudes restrictions, 10 values from 0 to 2."""
    if restriction != "smoothness":
        return np.linspace(0, 2, 10)
    if n_pre == 1:
        return np.linspace(0, math.sqrt(vcov[0, 0]), 10)
    A = _A_SD(n_pre, 0)
    diffs = A @ beta[:n_pre]
    se = np.sqrt(np.diag(A @ vcov[:n_pre, :n_pre] @ A.T))
    return np.linspace(0, float(np.max(diffs + special.ndtri(0.95) * se)), 10)


def trend_sensitivity(beta: Sequence[float], vcov: Any, n_pre: int, *,
                      restriction: str = "relative_magnitudes",
                      M: Sequence[float] | float | None = None,
                      target: Sequence[float] | None = None,
                      bias: str | None = None, monotonicity: str | None = None,
                      ci_method: str | None = None, alpha: float = 0.05,
                      grid_points: int = 1000, grid_bounds: tuple | None = None,
                      simulations: int = 1000, seed: int = 0,
                      n_threads: int | None = None,
                      pre: list | None = None, post: list | None = None) -> TrendSensitivityResult:
    """Confidence intervals for a post-treatment effect that allow parallel
    trends to fail by up to M (Rambachan and Roth, 2023).

    Parameters
    ----------
    beta : array-like
        Event-study coefficients: the `n_pre` pre-treatment ones from the
        earliest to the one just before the reference period, then the
        post-treatment ones from the first after it. The reference period is
        omitted (normalized to zero), and the periods are consecutive.
    vcov : array-like
        Their covariance.
    n_pre : int
        Number of pre-treatment coefficients.
    restriction : {"relative_magnitudes", "smoothness", "smoothness_relative_magnitudes"}
        How the post-treatment violation delta_post may differ from parallel
        trends. ``"relative_magnitudes"`` (R's Delta^RM): no period-to-period
        change after treatment larger than M times the largest one before it.
        ``"smoothness"`` (Delta^SD): the slope of the violation changes by at
        most M between consecutive periods, so M = 0 allows exactly a linear
        trend. ``"smoothness_relative_magnitudes"`` (Delta^SDRM, R's
        "deviation from linear trend"): changes in slope after treatment no
        larger than M times the largest one before it.
    M : float or array-like, optional
        The values of M (M-bar) at which to compute intervals. By default R's
        grids (see `default_M`).
    target : array-like, optional
        Weights l on the post-treatment coefficients: the target is
        l'tau_post. Default: the first post-treatment period, as in R.
    bias : {"positive", "negative"}, optional
        Also require the violation after treatment to have this sign.
    monotonicity : {"increasing", "decreasing"}, optional
        Also require the violation to be monotone in this direction over all
        periods. At most one of `bias` and `monotonicity`.
    ci_method : {"fixed_length", "conditional", "hybrid_fixed_length", "hybrid_least_favorable"}, optional
        The interval (R's FLCI, Conditional, C-F and C-LF). Default as in R:
        ``"fixed_length"`` for smoothness, ``"hybrid_fixed_length"`` for
        smoothness with a sign or monotonicity restriction,
        ``"hybrid_least_favorable"`` for the relative-magnitudes restrictions,
        which allow only it and ``"conditional"``.
    alpha : float, default 0.05
        One minus the coverage.
    grid_points : int, default 1000
        Grid of target values over which the conditional and hybrid tests are
        inverted, as in R; the interval runs from the smallest to the largest
        accepted point. When the accepted set reaches an end of R's grid, the
        grid is extended at the same spacing until it is closed.
    grid_bounds : (float, float), optional
        The grid's ends; by default R's.
    simulations : int, default 1000
        Draws for the least-favorable hybrid's first-stage critical value.
    seed : int, default 0
        Seed for those draws.
    n_threads : int, optional
        Threads for the tests (the pieces of the relative-magnitudes
        restrictions and the values of M run in parallel); by default all
        available CPUs.
    pre, post : list, optional
        Labels of the pre- and post-treatment coefficients for the report.

    Returns
    -------
    TrendSensitivityResult
    """
    beta = np.asarray(beta, dtype=float).ravel()
    sigma = np.asarray(vcov, dtype=float)
    n = len(beta)
    n_pre = int(n_pre)
    n_post = n - n_pre
    if sigma.shape != (n, n):
        raise ValueError(f"vcov must be {n} x {n} to match beta; got {sigma.shape}")
    if n_pre < 1 or n_post < 1:
        raise ValueError(f"need at least one pre- and one post-treatment coefficient; "
                         f"got {n_pre} pre and {n_post} post")
    if not np.all(np.isfinite(beta)) or not np.all(np.isfinite(sigma)):
        raise ValueError("beta and vcov must be finite")
    if restriction not in RESTRICTIONS:
        raise ValueError(f"restriction must be one of {RESTRICTIONS}; got {restriction!r}")
    if restriction == "smoothness_relative_magnitudes" and n_pre < 2:
        raise ValueError("restriction='smoothness_relative_magnitudes' needs at least two "
                         "pre-treatment coefficients")
    if bias is not None and monotonicity is not None:
        raise ValueError("give at most one of bias and monotonicity")
    if bias not in (None, "positive", "negative"):
        raise ValueError(f"bias must be 'positive' or 'negative'; got {bias!r}")
    if monotonicity not in (None, "increasing", "decreasing"):
        raise ValueError(f"monotonicity must be 'increasing' or 'decreasing'; got {monotonicity!r}")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1); got {alpha}")
    asym = np.max(np.abs(sigma - sigma.T)) if n else 0.0
    if asym > 100 * np.finfo(float).eps * max(np.max(np.abs(sigma)), np.finfo(float).tiny):
        warnings.warn(f"vcov is not exactly symmetric (largest asymmetry {asym:g})", stacklevel=2)
    lam = np.linalg.eigvalsh((sigma + sigma.T) / 2)
    if lam.min() < 0:
        warnings.warn(f"vcov is not numerically positive semi-definite (smallest eigenvalue "
                      f"{lam.min():g})", stacklevel=2)
    if ci_method is None:
        if restriction == "smoothness":
            ci_method = "fixed_length" if bias is None and monotonicity is None else "hybrid_fixed_length"
        else:
            ci_method = "hybrid_least_favorable"
    if ci_method not in CI_METHODS:
        raise ValueError(f"ci_method must be one of {CI_METHODS}; got {ci_method!r}")
    if restriction != "smoothness" and ci_method not in ("conditional", "hybrid_least_favorable"):
        raise ValueError(f"restriction={restriction!r} allows ci_method 'hybrid_least_favorable' "
                         f"or 'conditional'; got {ci_method!r}")
    if ci_method == "fixed_length" and (bias or monotonicity):
        warnings.warn("ci_method='fixed_length' does not use the sign or monotonicity restriction",
                      stacklevel=2)
    l = np.zeros(n_post) if target is None else np.asarray(target, dtype=float).ravel()
    if target is None:
        l[0] = 1.0
    if l.shape != (n_post,):
        raise ValueError(f"target must have {n_post} weights, one per post-treatment coefficient")
    if not np.any(l != 0):
        raise ValueError("target weights are all zero")
    Ms = (default_M(beta, sigma, n_pre, restriction) if M is None
          else np.atleast_1d(np.asarray(M, dtype=float)))
    if np.any(Ms < 0) or not np.all(np.isfinite(Ms)):
        raise ValueError("M must be finite and non-negative")
    grid_points = int(grid_points)
    if grid_points < 2:
        raise ValueError("grid_points must be at least 2")
    kappa = alpha / 10
    sd_theta = float(np.sqrt(l @ sigma[n_pre:, n_pre:] @ l))
    estimate = float(l @ beta[n_pre:])
    z = special.ndtri(1 - alpha / 2)
    original = {"estimate": estimate, "se": sd_theta,
                "lower": estimate - z * sd_theta, "upper": estimate + z * sd_theta}
    family = restriction

    rows_out, grids, extended, grid_accept = [], [], [], []
    if ci_method == "fixed_length":
        for m in Ms:
            vec, half = optimal_flci(sigma, m, n_pre, n_post, l, alpha)
            c = float(vec @ beta)
            rows_out.append((float(m), c - half, c + half))
    else:
        lf_cache: dict = {}
        flci_store: dict = {}
        jobs = []
        for mi, m in enumerate(Ms):
            pieces = _pieces(restriction, n_pre, n_post, m, bias, monotonicity)

            def flci_for(m=m):
                if m not in flci_store:
                    flci_store[m] = optimal_flci(sigma, m, n_pre, n_post, l, kappa)
                return flci_store[m]

            # R's grid
            if grid_bounds is not None:
                lb, ub = map(float, grid_bounds)
            elif family == "smoothness" and ci_method == "hybrid_fixed_length":
                vec, half = flci_for()
                c = float(vec @ beta)
                lb, ub = c - half, c + half
            elif family == "smoothness":
                idl, idu = _identified_set_at_zero(pieces[0], n_pre, l)
                if bias == "negative" and n_post > 1:
                    idl, idu = -idu, -idl
                lb, ub = idl - 20 * sd_theta, idu + 20 * sd_theta
            else:
                lb, ub = -20 * sd_theta, 20 * sd_theta
            jobs.append((mi, m, pieces, flci_for, lb, ub))

        setups = {}
        for mi, m, pieces, flci_for, lb, ub in jobs:
            for pj, piece in enumerate(pieces):
                setups[mi, pj] = (piece, flci_for)

        def prepare(key):
            piece, flci_for = setups[key]
            return key, _setup_piece(piece, beta, sigma, n_pre, n_post, l, family, ci_method,
                                     alpha, kappa, simulations, seed, flci_for, lf_cache)

        threads = n_threads or _cpus()
        # FLCI directions first (shared by the pieces of one M), then pieces
        if ci_method == "hybrid_fixed_length":
            for _, _, _, flci_for, _, _ in jobs:
                flci_for()
        with ThreadPoolExecutor(max_workers=threads) as ex:
            prepared = dict(ex.map(prepare, list(setups)))

            def run(args):
                key, thetas = args
                return key, _test_grid(prepared[key], thetas, beta, sigma, n_pre)

            grid_of = {mi: np.linspace(lb, ub, grid_points) for mi, _, _, _, lb, ub in jobs}
            accepted = dict(ex.map(run, [(key, grid_of[key[0]]) for key in prepared]))

            for mi, m, pieces, _, lb, ub in jobs:
                thetas = grid_of[mi]
                acc = np.zeros(len(thetas), bool)
                for pj in range(len(pieces)):
                    acc |= accepted[mi, pj]
                grid_accept.append(acc.copy())
                step = thetas[1] - thetas[0]
                ext = False
                if not acc.any() and ci_method != "hybrid_fixed_length" and grid_bounds is None:
                    # R's grid can miss the interval entirely (it is centered on
                    # zero, not on the estimate): try the same grid centered on
                    # the estimate before reporting an empty interval
                    half_w = (thetas[-1] - thetas[0]) / 2
                    thetas = np.linspace(estimate - half_w, estimate + half_w, len(thetas))
                    res = list(ex.map(run, [((mi, pj), thetas) for pj in range(len(pieces))]))
                    acc = np.zeros(len(thetas), bool)
                    for _, a_ in res:
                        acc |= a_
                    ext = True
                lower_open = upper_open = False
                if ci_method != "hybrid_fixed_length" and step > 0:
                    for side in (-1, 1):
                        for _ in range(50):
                            if not (acc[0] if side < 0 else acc[-1]):
                                break
                            ext = True
                            edge = thetas[0] if side < 0 else thetas[-1]
                            new = edge + side * step * np.arange(1, grid_points + 1)
                            if side < 0:
                                new = new[::-1]
                            res = list(ex.map(run, [((mi, pj), new) for pj in range(len(pieces))]))
                            add = np.zeros(len(new), bool)
                            for _, a in res:
                                add |= a
                            thetas = np.concatenate([new, thetas]) if side < 0 else np.concatenate([thetas, new])
                            acc = np.concatenate([add, acc]) if side < 0 else np.concatenate([acc, add])
                        else:
                            if side < 0:
                                lower_open = True
                            else:
                                upper_open = True
                if acc.any():
                    lo_v = -np.inf if lower_open else float(thetas[acc].min())
                    hi_v = np.inf if upper_open else float(thetas[acc].max())
                    if lower_open or upper_open:
                        warnings.warn(f"at M = {m:g} the accepted set did not close within 50 grid "
                                      "extensions; reporting an infinite bound",
                                      TrendSensitivityWarning, stacklevel=2)
                else:
                    lo_v = hi_v = np.nan
                    warnings.warn(f"at M = {m:g} the test rejects every value of the target on the "
                                  "grid; the interval is empty (NaN)", TrendSensitivityWarning,
                                  stacklevel=2)
                rows_out.append((float(m), lo_v, hi_v))
                grids.append((float(lb), float(ub)))
                extended.append(ext)
    table = pl.DataFrame(rows_out, schema=["M", "lower", "upper"], orient="row")
    diagnostics = {"grid_points": grid_points if ci_method != "fixed_length" else None}
    if grids:
        diagnostics["grid_bounds"] = grids
        diagnostics["grid_extended"] = extended
    if ci_method == "hybrid_least_favorable":
        diagnostics["simulations"] = simulations
        diagnostics["seed"] = seed
    return TrendSensitivityResult(
        table=table, original=original, restriction=restriction, ci_method=ci_method,
        alpha=alpha, target=l, n_pre=n_pre, n_post=n_post, bias=bias,
        monotonicity=monotonicity, pre=list(pre) if pre is not None else [],
        post=list(post) if post is not None else [],
        r_names={"Delta": _r_delta(restriction, bias, monotonicity), "method": _R_METHOD[ci_method]},
        diagnostics=diagnostics, _grid_accept=grid_accept)


def _cpus():
    from .resources import detect_cpus
    try:
        return max(1, int(detect_cpus()))
    except Exception:
        return 1


def _event_time(name):
    tail = str(name).rsplit("::", 1)[-1]
    try:
        v = float(tail)
    except ValueError:
        return None
    return int(v) if v.is_integer() else None


def event_study_inputs(times: Sequence[int], beta: Any, vcov: Any, *, reference: int = -1,
                       target: Any = None):
    """The inputs of `trend_sensitivity` from an event study labeled by
    integer event times: the coefficients before `reference` and after it,
    in event-time order, and the target's weights.

    `target` is an event time after the reference (default: 0 if estimated,
    else the first after the reference), ``"average"`` (equal weights on every
    post-reference coefficient), or ``{event time: weight}``.

    Returns (beta, vcov, n_pre, weights, pre_times, post_times)."""
    times = [int(t) for t in times]
    if len(set(times)) != len(times):
        raise ValueError(f"event times are repeated: {times}")
    if reference in times:
        raise ValueError(f"the reference event time {reference} has a coefficient; the event "
                         "study must omit (normalize to zero) the reference period")
    beta = np.asarray(beta, dtype=float).ravel()
    vcov = np.asarray(vcov, dtype=float)
    order = np.argsort(times)
    st = [times[i] for i in order]
    pre = [t for t in st if t < reference]
    post = [t for t in st if t > reference]
    if not pre:
        raise ValueError(f"no pre-treatment coefficients (event times before the reference "
                         f"{reference}); sensitivity analysis needs at least one")
    if not post:
        raise ValueError(f"no coefficients after the reference event time {reference}")
    want_pre = list(range(reference - len(pre), reference))
    want_post = list(range(reference + 1, reference + 1 + len(post)))
    if pre != want_pre or post != want_post:
        raise ValueError(
            "the event times must be consecutive on each side of the reference "
            f"{reference}; got {pre} before it and {post} after it. Select consecutive event "
            "times (e.g. with keep=, or horizons= when aggregating)")
    b = beta[order]
    V = vcov[np.ix_(order, order)]
    n_post = len(post)
    w = np.zeros(n_post)
    if target is None:
        w[post.index(0) if 0 in post else 0] = 1.0
    elif isinstance(target, str):
        if target != "average":
            raise ValueError(f"target must be an event time, 'average' or {{event time: weight}}; "
                             f"got {target!r}")
        w[:] = 1.0 / n_post
    elif isinstance(target, dict):
        for t, wt in target.items():
            if int(t) not in post:
                raise ValueError(f"target event time {t} is not among the post-reference "
                                 f"event times {post}")
            w[post.index(int(t))] = float(wt)
    elif np.ndim(target) == 0 and float(target).is_integer():
        if int(target) not in post:
            raise ValueError(f"target event time {target} is not among the post-reference "
                             f"event times {post}")
        w[post.index(int(target))] = 1.0
    else:
        raise ValueError(f"target must be an event time, 'average' or {{event time: weight}}; "
                         f"got {target!r}")
    return b, V, len(pre), w, pre, post


def event_study_sensitivity(times: Sequence[int], beta: Any, vcov: Any, *, reference: int = -1,
                            target: Any = None, **options: Any) -> TrendSensitivityResult:
    """`trend_sensitivity` for an event study labeled by integer event times
    (see `event_study_inputs` for `reference` and `target`; `options` are
    `trend_sensitivity`'s)."""
    b, V, n_pre, w, pre, post = event_study_inputs(times, beta, vcov, reference=reference,
                                                   target=target)
    return trend_sensitivity(b, V, n_pre, target=w, pre=pre, post=post, **options)

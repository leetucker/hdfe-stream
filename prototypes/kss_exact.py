"""Kline-Saggio-Solvsten leave-out variance components: exact, dense, small-n.

STATUS: prototype. Not part of the hdfe_stream package, not streaming, and not
usable at scale. Its only job is to pin down *what* the estimator is, on panels
small enough that every quantity can be computed exactly and checked, so that a
later streaming implementation has something trustworthy to be tested against.

The estimand
------------
With y = J psi + W alpha + e, where J selects firms and W selects workers, the
objects of interest are observation-level moments of the fitted effects:

    var(psi)        Var_i( psi[firm(i)] )
    var(alpha)      Var_i( alpha[worker(i)] )
    cov(psi, alpha) Cov_i( psi[firm(i)], alpha[worker(i)] )

all taken across *observations*, dividing by n (not n-1), which is the
convention in KSS and in pytwoway's `VarPsi` / `VarAlpha` / `CovPsiAlpha`.

Each is a quadratic form theta = b' Q b in the stacked coefficient vector
b = [psi; alpha]. The plug-in estimator b_hat' Q b_hat is biased upward for the
variances (and, for cov, biased toward zero), because b_hat carries estimation
error:

    E[b_hat' Q b_hat] = theta + sum_i sigma2_i * B_ii,
    B_ii = a_i' S^-1 Q S^-1 a_i,        S = A' A,   A = [J, W]

KSS remove exactly that term, using a leave-one-out estimate of sigma2_i that is
unbiased without assuming homoskedasticity:

    sigma2_hat_i = (y_i - y_bar) * e_hat_i / (1 - P_ii),    P_ii = a_i' S^-1 a_i

so the corrected estimator is

    theta_hat = b_hat' Q b_hat - sum_i B_ii sigma2_hat_i.

Two details that are easy to get wrong:

  * sigma2_hat_i uses the *demeaned* outcome (y_i - y_bar), not y_i, as in
    LeaveOutTwoWay's `leave_out_KSS`, VarianceComponentsHDFE.jl, xhdfe and
    pytwoway (the older `leave_out_COMPLETE` uses y_i);
  * stayers -- workers seen at a single firm -- use their own sigma2_hat_i when
    an observation is left out, as LeaveOutTwoWay does: a stayer seen twice or
    more still has a well-defined leave-out residual. pytwoway instead imputes
    the movers' mean at the same firm, a rule designed for spell-level data;
    it is available as stayers="firm_mean".

What this file does NOT do
--------------------------
  * scale: S^-1 is formed densely, so K = (#firms - 1) + #workers must be small
  * Johnson-Lindenstrauss approximation of P_ii and B_ii (the whole point of the
    eventual streaming version), including its non-linearity bias correction
  * the proper leave-one-out connected set -- see `prune_to_leave_out` for the
    brute-force stand-in and why it is not the same thing
  * inference on the components themselves
  * covariates beyond the two fixed effects, weights, or IV
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp


@dataclass
class Panel:
    """A two-way panel, coded to dense contiguous ids.

    `extra` carries per-observation arrays along through subsetting and
    pruning. That is how the tests keep hold of the true worker and firm
    effects: the estimand is a moment over the *surviving* rows, so ground
    truth has to be computed after sample selection, not before.
    """

    worker: np.ndarray          # int, 0 .. n_workers-1
    firm: np.ndarray            # int, 0 .. n_firms-1
    y: np.ndarray
    weight: np.ndarray = None   # None means unweighted
    extra: dict = field(default_factory=dict)

    @property
    def w(self):
        """Weights, as an array either way, so callers need no special case."""
        return np.ones(self.n_obs) if self.weight is None else self.weight

    @property
    def n_obs(self):
        return len(self.y)

    @property
    def n_workers(self):
        return int(self.worker.max()) + 1 if self.n_obs else 0

    @property
    def n_firms(self):
        return int(self.firm.max()) + 1 if self.n_obs else 0

    @property
    def movers(self):
        """Workers observed at more than one firm, as a per-observation mask."""
        n_firms_per_worker = np.zeros(self.n_workers, dtype=np.int64)
        order = np.lexsort((self.firm, self.worker))
        w, f = self.worker[order], self.firm[order]
        new = np.ones(len(w), bool)
        new[1:] = (w[1:] != w[:-1]) | (f[1:] != f[:-1])
        np.add.at(n_firms_per_worker, w[new], 1)
        return n_firms_per_worker[self.worker] > 1

    def recode(self):
        """Renumber workers and firms to be contiguous again after a subset."""
        _, worker = np.unique(self.worker, return_inverse=True)
        _, firm = np.unique(self.firm, return_inverse=True)
        return Panel(worker=worker, firm=firm, y=self.y, weight=self.weight,
                     extra=self.extra)

    def subset(self, mask):
        return Panel(worker=self.worker[mask], firm=self.firm[mask],
                     y=self.y[mask],
                     weight=None if self.weight is None else self.weight[mask],
                     extra={k: v[mask] for k, v in self.extra.items()}).recode()

    @classmethod
    def from_frame(cls, frame, worker="worker_id", firm="firm_id", y="log_earn",
                   weight=None, extra=()):
        """Build from a Polars DataFrame (or anything indexable by column)."""
        _, w = np.unique(np.asarray(frame[worker]), return_inverse=True)
        _, f = np.unique(np.asarray(frame[firm]), return_inverse=True)
        return cls(worker=w, firm=f, y=np.asarray(frame[y], dtype=float),
                   weight=(None if weight is None
                           else np.asarray(frame[weight], dtype=float)),
                   extra={c: np.asarray(frame[c], dtype=float) for c in extra})


def design(panel):
    """A = [J, W], with the last firm dummy dropped for identification.

    Dropping one firm column is the usual AKM normalization: it sets the last
    firm's effect to zero. The variance components are location-invariant, so
    they do not depend on which column goes.
    """
    n, n_firms, n_workers = panel.n_obs, panel.n_firms, panel.n_workers
    rows = np.arange(n)
    J = sp.csr_matrix((np.ones(n), (rows, panel.firm)), shape=(n, n_firms))
    W = sp.csr_matrix((np.ones(n), (rows, panel.worker)), shape=(n, n_workers))
    return sp.hstack([J[:, :-1], W], format="csr"), J, W


def _dense_inverse(A, w=None):
    """(A'WA)^-1, densely. The prototype's scale limit lives here."""
    Ad = A.toarray()
    S = Ad.T @ Ad if w is None else Ad.T @ (w[:, None] * Ad)
    return np.linalg.pinv(S, hermitian=True)


def leverages(A, S_inv=None, w=None):
    """Exact leverages, in the square-root-weight metric.

    Weighted least squares is ordinary least squares on the transformed design
    W^(1/2) A, and it is *that* hat matrix which is symmetric and idempotent --
    so it is that one whose diagonal is the leverage a leave-one-out residual
    divides by:

        P_ii = w_i a_i' (A'WA)^-1 a_i

    Without weights w is one and this is the usual expression.
    """
    S_inv = _dense_inverse(A, w) if S_inv is None else S_inv
    Ad = A.toarray()
    quadratic = np.einsum("ij,ij->i", Ad @ S_inv, Ad)
    return quadratic if w is None else w * quadratic


def quadratic_forms(panel, J, W, person_years=None):
    """Q matrices for var(psi), var(alpha) and cov(psi, alpha).

    Each is built in the stacked [psi (minus the dropped firm); alpha] basis, so
    that theta = b' Q b reproduces the observation-level moment. The demeaning
    matrix is what makes these *variances* rather than second moments; it is
    applied to the n-row selector matrices before the product.

    With weights the moment is weighted too -- the mean subtracted is the
    weighted mean, and rows enter in proportion to their weight -- so that the
    estimand matches the population the weights describe.
    """
    weights = panel.w
    total = weights.sum()
    # the references divide by n - 1 person-year observations, not n; with
    # weights the same factor scales the weight total. Centering still uses it.
    n = panel.n_obs if person_years is None else person_years
    norm = total * (n - 1) / n
    Jd = J[:, :-1].toarray()            # matches the dropped column in `design`
    Wd = W.toarray()

    def centered(M):
        return M - (weights @ M) / total

    Jc, Wc = centered(Jd), centered(Wd)
    Jw, Ww = weights[:, None] * Jc, weights[:, None] * Wc
    n_psi, n_alpha = Jc.shape[1], Wc.shape[1]
    K = n_psi + n_alpha

    def blocks(top_left=None, top_right=None, bottom_right=None):
        Q = np.zeros((K, K))
        if top_left is not None:
            Q[:n_psi, :n_psi] = top_left
        if bottom_right is not None:
            Q[n_psi:, n_psi:] = bottom_right
        if top_right is not None:
            Q[:n_psi, n_psi:] = top_right
            Q[n_psi:, :n_psi] = top_right.T
        return Q

    return {
        "var(psi)": blocks(top_left=Jc.T @ Jw / norm),
        "var(alpha)": blocks(bottom_right=Wc.T @ Ww / norm),
        # symmetrized so that b' Q b is the covariance itself
        "cov(psi, alpha)": blocks(top_right=Jc.T @ Ww / (2 * norm)),
    }


def sigma2_leave_one_out(panel, y_hat, P_ii, stayers="own"):
    """Leave-one-out variance estimates, unbiased under heteroskedasticity.

    For movers, sigma2_i = w_i (y_i - y_bar) e_i / (1 - P_ii): this is the
    algebraic form of the leave-one-out product in the square-root-weight
    metric, and its expectation is w_i Var(e_i) without assuming the errors are
    identically distributed.

    That extra w_i is deliberate and is what the bias term wants. The sandwich
    is S^-1 A'W Omega W A S^-1 with Omega = diag(Var(e_i)), so the quantity
    entering it per observation is w_i^2 Var(e_i) -- and the trace applies one
    factor of W itself, so what it needs handed to it is w_i Var(e_i). Without
    weights w is one and this is the usual expression.

    `stayers="own"` (default) gives stayers their own sigma2_hat_i, like any
    other row -- LeaveOutTwoWay's rule at observation level. "firm_mean" is
    pytwoway's: the movers' mean at the same firm, weighted so that the imputed
    value is on the same per-observation scale as the movers' own. "drop" sets
    stayers to zero.
    """
    weights = panel.w
    resid = panel.y - y_hat
    movers = panel.movers
    y_bar = (weights @ panel.y) / weights.sum()
    sigma2 = np.full(panel.n_obs, np.nan)
    rows = np.ones(panel.n_obs, bool) if stayers == "own" else movers
    sigma2[rows] = (weights[rows] * (panel.y[rows] - y_bar)
                    * resid[rows] / (1.0 - P_ii[rows]))

    if stayers == "own":
        # LeaveOutTwoWay's rule when leaving out an observation: a stayer seen
        # twice or more has a well-defined leave-out residual of its own
        pass
    elif stayers == "firm_mean":
        # sigma2 is on a w_i scale, so averaging it across observations of
        # different weight has to divide that out and put it back
        totals = np.zeros(panel.n_firms)
        counts = np.zeros(panel.n_firms)
        np.add.at(totals, panel.firm[movers],
                  sigma2[movers] * weights[movers])
        np.add.at(counts, panel.firm[movers], weights[movers] ** 2)
        firm_mean = np.divide(totals, counts, out=np.full(panel.n_firms, np.nan),
                              where=counts > 0)
        sigma2[~movers] = firm_mean[panel.firm[~movers]] * weights[~movers]
    elif stayers == "drop":
        sigma2[~movers] = 0.0
    else:
        raise ValueError(f"unknown stayers rule {stayers!r}")
    return sigma2, movers


@dataclass
class KSSResult:
    n_obs: int
    n_workers: int
    n_firms: int
    n_movers: int
    plug_in: dict
    kss: dict
    bias: dict
    max_leverage: float
    sigma2_mean: float
    diagnostics: dict = field(default_factory=dict)

    def summary(self):
        lines = [f"KSS leave-out variance components  "
                 f"(n={self.n_obs:,}, workers={self.n_workers:,}, "
                 f"firms={self.n_firms:,}, movers={self.n_movers:,})",
                 f"max leverage {self.max_leverage:.6f}   "
                 f"mean sigma2 {self.sigma2_mean:.6f}",
                 "",
                 f"{'component':<20}{'plug-in':>12}{'bias':>12}{'KSS':>12}"]
        lines.append("-" * 56)
        for key in self.plug_in:
            lines.append(f"{key:<20}{self.plug_in[key]:>12.6f}"
                         f"{self.bias[key]:>12.6f}{self.kss[key]:>12.6f}")
        return "\n".join(lines)


def kss(panel, stayers="own", leverage_tol=1e-9):
    """Plug-in and KSS-corrected variance components for a two-way panel.

    Everything is exact: S^-1 is formed densely and the leverages and bias terms
    follow from it directly. Intended for panels of at most a few thousand
    coefficients.

    Raises if the panel is not leave-one-out connected. The check is on the
    leverages themselves rather than on whether sigma2 came out finite: a
    leverage of one is reached as 1 - 1e-16 rather than exactly 1, which yields
    an enormous but perfectly finite sigma2 and would otherwise sail through.
    """
    A, J, W = design(panel)
    weights = panel.w
    S_inv = _dense_inverse(A, panel.weight)

    beta = S_inv @ (A.T @ (weights * panel.y))
    y_hat = A @ beta
    P_ii = leverages(A, S_inv, panel.weight)

    movers_mask = panel.movers
    if movers_mask.any():
        worst = float(P_ii[movers_mask].max())
        if worst > 1.0 - leverage_tol:
            raise ValueError(
                f"a mover observation has leverage {worst:.12f}, i.e. 1 to within "
                f"{leverage_tol:g}: the panel is not leave-one-out connected, so "
                "its leave-one-out residual is undefined. Prune with "
                "prune_to_leave_out() first.")

    sigma2, movers = sigma2_leave_one_out(panel, y_hat, P_ii, stayers)
    if not np.all(np.isfinite(sigma2)):
        raise ValueError(
            "sigma2 is not finite for every observation: some firm has no "
            "movers, so there is nothing to impute a stayer's variance from. "
            "Prune with prune_to_leave_out() first.")

    Ad = A.toarray()
    plug_in, corrected, bias = {}, {}, {}
    for name, Q in quadratic_forms(panel, J, W).items():
        plug_in[name] = float(beta @ Q @ beta)
        # The bias is tr(Q S^-1 A'W Omega W A S^-1), which rowwise is
        # sum_i w_i * sigma2_i * a_i' S^-1 Q S^-1 a_i -- one factor of w here
        # and the other already inside sigma2 (see sigma2_leave_one_out).
        M = S_inv @ Q @ S_inv
        B_ii = np.einsum("ij,jk,ik->i", Ad, M, Ad)
        bias[name] = float((weights * B_ii) @ sigma2)
        corrected[name] = plug_in[name] - bias[name]

    return KSSResult(
        n_obs=panel.n_obs, n_workers=panel.n_workers, n_firms=panel.n_firms,
        n_movers=int(movers.sum()), plug_in=plug_in, kss=corrected, bias=bias,
        max_leverage=float(P_ii[movers].max()) if movers.any() else float("nan"),
        sigma2_mean=float(np.mean(sigma2 / weights)),
        diagnostics={"n_coef": A.shape[1], "rank_deficiency":
                     A.shape[1] - int(np.linalg.matrix_rank(Ad))},
    )


# --------------------------------------------------------------------------
# leaving out a match (LeaveOutTwoWay's default)
# --------------------------------------------------------------------------
#
# LeaveOutTwoWay's `leave_out_KSS` leaves out a whole worker-firm match by
# collapsing the data to match means, weighting each by the length of the spell,
# and running KSS on the weighted collapsed regression. In the square-root-weight
# metric that *is* leaving out a match: the collapsed row's leverage is the
# match's leverage. Everything below follows that file line by line, including
# two details that change numbers:
#
#   * sigma2 is centered in the transformed metric: (sqrt(w) ybar - mean over
#     matches of sqrt(w) ybar) times the transformed leave-out residual;
#   * a stayer's single match cannot be left out -- its leverage is 1 -- so it
#     gets `sigma_for_stayers.m`: the worker's own person-year residuals from the
#     collapsed fit, with leverage 1/T_i, (y_it - ybar) e_it / (1 - 1/T_i),
#     averaged over the match.
#
# The quadratic forms stay person-year moments, divided by person-years - 1.


def collapse_matches(panel):
    """One row per worker-firm match, and each person-year row's match index.

    The match row carries the weighted mean outcome and, as its weight, the
    spell's total weight -- its length when unweighted.
    """
    key = panel.worker.astype(np.int64) * max(panel.n_firms, 1) + panel.firm
    uniq, match = np.unique(key, return_inverse=True)
    w = panel.w
    total = np.bincount(match, weights=w)
    ybar = np.bincount(match, weights=w * panel.y) / total
    collapsed = Panel(worker=uniq // max(panel.n_firms, 1),
                      firm=uniq % max(panel.n_firms, 1), y=ybar, weight=total)
    return collapsed, match


def kss_match(panel, leverage_tol=1e-9, centering="reference"):
    """KSS leaving out a match, exactly, following LeaveOutTwoWay.

    `centering="reference"` is LeaveOutTwoWay's: sqrt(w) ybar minus its mean over
    matches. "weighted" centers ybar at its weighted mean before transforming,
    w (ybar - ybar_w) -- identical when spells are of equal length.
    """
    mp, match = collapse_matches(panel)
    A, J, W = design(mp)
    w = mp.w
    S_inv = _dense_inverse(A, w)
    beta = S_inv @ (A.T @ (w * mp.y))
    fitted = A @ beta
    P_ii = leverages(A, S_inv, w)

    matches_per_worker = np.bincount(mp.worker)
    stayer = matches_per_worker[mp.worker] == 1
    if (~stayer).any():
        worst = float(P_ii[~stayer].max())
        if worst > 1.0 - leverage_tol:
            raise ValueError(f"a mover's match has leverage {worst:.12f}: the "
                             "panel is not leave-one-match-out connected")

    root = np.sqrt(w)
    y_t = root * mp.y
    e_t = root * (mp.y - fitted)
    sigma2 = np.full(len(w), np.nan)
    movers = ~stayer
    centered = (y_t - y_t.mean() if centering == "reference"
               else root * (mp.y - (w @ mp.y) / w.sum()))
    sigma2[movers] = centered[movers] * e_t[movers] / (1.0 - P_ii[movers])

    # sigma_for_stayers.m, at person-year level
    wr = panel.w
    worker_total = np.bincount(panel.worker, weights=wr)
    p_it = wr / worker_total[panel.worker]
    e_it = panel.y - fitted[match]
    y_mean = (wr @ panel.y) / wr.sum()
    r_it = (panel.y - y_mean) * e_it / (1.0 - p_it)
    # the match row's error variance in the transformed metric is
    # sum w^2 Var(e) / sum w -- a plain mean of r when unweighted
    sigma_stay = np.bincount(match, weights=wr ** 2 * r_it) / w
    sigma2[stayer] = sigma_stay[stayer]

    forms = quadratic_forms(mp, J, W, person_years=panel.n_obs)
    Ad = A.toarray()
    plug_in, corrected, bias = {}, {}, {}
    for name, Q in forms.items():
        plug_in[name] = float(beta @ Q @ beta)
        M = S_inv @ Q @ S_inv
        B_ii = np.einsum("ij,jk,ik->i", Ad, M, Ad)
        bias[name] = float((w * B_ii) @ sigma2)
        corrected[name] = plug_in[name] - bias[name]
    return KSSResult(
        n_obs=panel.n_obs, n_workers=panel.n_workers, n_firms=panel.n_firms,
        n_movers=int(movers.sum()), plug_in=plug_in, kss=corrected, bias=bias,
        max_leverage=float(P_ii[movers].max()) if movers.any() else float("nan"),
        sigma2_mean=float(np.mean(sigma2 / w)),
        diagnostics={"n_matches": len(w), "stayer_matches": int(stayer.sum()),
                     "worker": mp.worker, "firm": mp.firm, "sigma2": sigma2,
                     "stayer": stayer})


# --------------------------------------------------------------------------
# standard errors when leaving out a match: the reference, literally
# --------------------------------------------------------------------------
#
# LeaveOutTwoWay's `leave_out_COMPLETE`, as xhdfe ports it (akm_kss.cpp, the
# "Component standard errors" block), works in person-year space with the
# block leave-out kernel. Every regressor is constant within a match, so the
# person-year hat matrix's block for match m is p_m 1 1' with p_m = x_m' S^- x_m,
# and leaving out the block gives eta_h = eta + p/(1 - T p) (1'eta) on it.
# Its conventions:
#
#   * the outcome is the raw outcome (after partialling out any controls), not
#     centered, in sigma, in C y and in theta_c;
#   * stayers: p = 1/T, b = 0 for var(psi) and cov (exact: a stayer's outcome
#     does not move psi-hat), eta_h = eta (the block is singular), and var(alpha)
#     is not reported;
#   * sigma-tilde is the person-year sig_i = y_i eta_h,i, averaged within cells
#     of a 1,000 x 1,000 grid in quantile of (p, b) ("llr_fit mode 4");
#   * V = (4 sum_i sigma-tilde_i W_i^2 - Var_sim) / n^2, W = C y, with Var_sim
#     the variance across 1,000 draws of v^H C v, v = sqrt(sigma-tilde) z,
#     complex where sigma-tilde < 0 (MATLAB's sqrt), and the standard error
#     truncated to zero when that difference is negative.
#
# `xhdfe_match_se` reproduces all of it, with the expectation of Var_sim in
# place of its simulated value, so it can be compared with xhdfe run at many
# draws. It exists to validate against the reference; the package's own
# match-level standard errors are `kss_match_se`.

def _group_equally(x, n_groups):
    """LeaveOutTwoWay's quantile binning (xhdfe `group_equally`), literally."""
    s = np.sort(x)
    N = len(s)
    cuts = np.empty(n_groups - 1)
    for i in range(1, n_groups):
        q = (100.0 * i / n_groups) * (N + 1.0) / 100.0
        w = int(np.floor(q))
        f = q - w
        w = min(max(w, 1), N - 1)
        cuts[i - 1] = (1.0 - f) * s[w - 1] + f * s[w]
    idx = np.searchsorted(cuts, x, side="right")
    return np.where(idx < len(cuts), idx + 1, n_groups)


def _person_year_forms(panel, J, W):
    """xhdfe's unscaled quadratic forms: sum_i (d_i - dbar)(d_i - dbar), no
    divisor, in the stacked [psi (minus the last firm); alpha] basis."""
    n = panel.n_obs
    Jc = J[:, :-1].toarray()
    Wd = W.toarray()
    Jc = Jc - Jc.mean(axis=0)
    Wc = Wd - Wd.mean(axis=0)
    n_psi = Jc.shape[1]
    K = n_psi + Wc.shape[1]
    fe, cov = np.zeros((K, K)), np.zeros((K, K))
    fe[:n_psi, :n_psi] = Jc.T @ Jc
    cov[:n_psi, n_psi:] = Jc.T @ Wc / 2.0
    cov[n_psi:, :n_psi] = Wc.T @ Jc / 2.0
    return {"var(psi)": fe, "cov(psi, alpha)": cov}, n


def xhdfe_match_se(panel, grid=1000):
    """xhdfe's `compute_se` at match level, dense and exact.

    Unweighted panels only (xhdfe refuses weights with standard errors).
    Returns per component: `theta_c`, `first` (4 sum sigma-tilde W^2),
    `var_sim` (the expectation of the simulated variance), `se` and the
    unclamped `se_numerator`, all on xhdfe's scale (the standard error is
    divided by n, the plug-in by n - 1).
    """
    A, J, W = design(panel)
    X = A.toarray()
    y = panel.y
    n = panel.n_obs
    S_inv = np.linalg.pinv(X.T @ X, hermitian=True)
    beta = S_inv @ (X.T @ y)
    eta = y - X @ beta

    key = panel.worker.astype(np.int64) * max(panel.n_firms, 1) + panel.firm
    _, match = np.unique(key, return_inverse=True)
    n_match = match.max() + 1
    T = np.bincount(match).astype(float)
    first_row = np.zeros(n_match, dtype=np.int64)
    first_row[match[::-1]] = np.arange(n)[::-1]
    worker_of = panel.worker[first_row]
    matches_per_worker = np.bincount(worker_of)
    stayer = matches_per_worker[worker_of] < 2
    worker_T = np.bincount(panel.worker).astype(float)

    Xm = X[first_row]                                   # one row per match
    Z = S_inv @ Xm.T                                    # S^- x_m, columns
    p = np.einsum("km,km->m", Xm.T, Z)
    p[stayer] = 1.0 / worker_T[worker_of[stayer]]
    den = 1.0 - T * p
    gain = np.where(den > 1e-12, p / np.where(den > 1e-12, den, 1.0), 0.0)
    eta_sum = np.bincount(match, weights=eta)
    eta_h = eta + gain[match] * eta_sum[match]
    sig_raw = y * eta_h
    snap = lambda v: v.astype(np.float32).astype(np.float64)  # noqa: E731
    gP = _group_equally(snap(p[match]), grid)

    forms, _ = _person_year_forms(panel, J, W)
    M = np.eye(n) - X @ S_inv @ X.T
    E = sp.csr_matrix((np.ones(n), (np.arange(n), match)),
                      shape=(n, n_match)).toarray()
    y_sum = np.bincount(match, weights=y)
    plug_scale = n - 1.0
    out = {}
    for name, Q in forms.items():
        b = np.einsum("km,kl,lm->m", Z, Q, Z)
        b[stayer] = 0.0
        theta_c = (float(beta @ Q @ beta) / plug_scale
                   - float(np.sum(b * y_sum * eta_sum * (1.0 + T * gain))) / n)

        gB = _group_equally(snap(b[match]), grid)
        cell = gB.astype(np.int64) * 1000003 + gP
        _, cell_id = np.unique(cell, return_inverse=True)
        sig_t = (np.bincount(cell_id, weights=sig_raw)
                 / np.bincount(cell_id))[cell_id]

        B = X @ S_inv @ Q @ S_inv @ X.T
        full_gain = 1.0 + T * gain
        G = (E * (b * full_gain)[None, :]) @ E.T @ M      # Lambda_B (I-L_P)^+ M
        C_ns = B - G
        # W = By - (Lambda_B eta_h + xi)/2 is the symmetrized kernel times y
        Wv = 0.5 * (C_ns + C_ns.T) @ y
        first = 4.0 * float(sig_t @ Wv ** 2)

        a = np.where(sig_t >= 0, np.sqrt(np.abs(sig_t)), 0.0) \
            + 1j * np.where(sig_t < 0, np.sqrt(np.abs(sig_t)), 0.0)
        K = np.conj(a)[:, None] * C_ns * a[None, :]
        re = 0.5 * (K.real + K.real.T)
        im = 0.5 * (K.imag + K.imag.T)
        var_sim = 2.0 * float((re ** 2).sum() + (im ** 2).sum())
        numerator = first - var_sim
        out[name] = {"theta_c": theta_c, "first": first, "var_sim": var_sim,
                     "se_numerator": numerator,
                     "se": float(np.sqrt(max(numerator, 0.0)) / n)}
    return out


# --------------------------------------------------------------------------
# standard errors when leaving out a match: the package's version
# --------------------------------------------------------------------------
#
# The reference's person-year block kernel is the collapsed regression's own
# zero-diagonal kernel. Write E for the person-year-by-match indicator and
# D = diag(sqrt(T)). Every piece of the person-year kernel C is E (.) E', and
# conjugating by D turns it into
#
#     C~ = B~ - (1/2)(diag(b~) M~ + M~ diag(b~)),   b~_m = B~_mm / (1 - P~_mm)
#
# on the collapsed rows, in the square-root-weight metric -- the kernel the
# observation-level code already applies. A stayer's match has P~ = 1 and, for
# var(psi) and cov, B~_mm = 0 exactly, so b~ = 0 there: the reference's
# convention, and the only value that keeps C~'s diagonal zero.
#
# v'Cv depends on v only through E'v, whose covariance is diag(sum over the
# match of Var(e)) for independent errors -- or the whole block 1'Omega_m 1 if
# errors are correlated within the match. In the collapsed metric that is
# Om~_m = 1'Omega_m 1 / T_m, the variance of sqrt(T) times the match mean, and
# the collapsed leave-match-out sigma2 estimates exactly that. The reference's
# person-year sig_i estimates only the diagonal, tr(Omega_m) / T_m, so under
# within-match correlation it understates V. `se_variance` chooses.
#
# The reported point estimate must be the quadratic form the variance
# describes. With the weighted centering it is y~' C~ y~, y~ = sqrt(T)(ybar -
# ybar_w). With the reference's centering, c = mean over matches of sqrt(T)
# ybar, it is exactly y_r' K y_r with y_r = sqrt(T) ybar and
#
#     K = C~ + (1/2M) (1 u' + u 1'),   u = M~ b~
#
# a rank-two update whose diagonal, u_m / M, is O(1/M) rather than zero: the
# price of centering at an estimated constant that is not in the column space.


def kss_match_se(panel, centering="reference", se_variance="person_year",
                 n_bins=None):
    """Match-level standard errors for var(psi) and cov(psi, alpha), exactly.

    `centering` must match the point estimate's (`kss_match`); `theta` is the
    point estimate recomputed as the quadratic form, which must reproduce
    `kss_match(...).kss` for these two components. `se_variance` is "match"
    (the collapsed leave-match-out sigma2, weighted-centered, with
    `sigma_for_stayers` for stayers) or "person_year" (the reference's
    person-year y eta_h summed over the match and divided by T, outcome
    demeaned). var(alpha) is not reported, following leave_out_COMPLETE.
    """
    mp, match = collapse_matches(panel)
    A, J, W = design(mp)
    w = mp.w
    S_inv = _dense_inverse(A, w)
    beta = S_inv @ (A.T @ (w * mp.y))
    fitted = A @ beta
    P_ii = leverages(A, S_inv, w)
    stayer = np.bincount(mp.worker)[mp.worker] == 1
    movers = ~stayer

    root = np.sqrt(w)
    X = root[:, None] * A.toarray()
    Mt = np.eye(len(w)) - X @ S_inv @ X.T
    y_r = root * mp.y
    y_w = root * (mp.y - (w @ mp.y) / w.sum())
    e_t = root * (mp.y - fitted)
    n_match = len(w)

    # the variance each collapsed row carries
    sigma2 = np.zeros(n_match)
    if se_variance == "match":
        sigma2[movers] = y_w[movers] * e_t[movers] / (1.0 - P_ii[movers])
        wr = panel.w
        worker_total = np.bincount(panel.worker, weights=wr)
        p_it = wr / worker_total[panel.worker]
        e_it = panel.y - fitted[match]
        y_mean = (wr @ panel.y) / wr.sum()
        r_it = (panel.y - y_mean) * e_it / (1.0 - p_it)
        stay = np.bincount(match, weights=wr ** 2 * r_it) / w
        sigma2[stayer] = stay[stayer]
    elif se_variance == "person_year":
        if not np.allclose(panel.w, 1.0):
            raise ValueError("se_variance='person_year' is the reference's, "
                             "unweighted")
        T = w
        p = np.where(stayer, 1.0, P_ii) / T
        den = 1.0 - T * p
        gain = np.where(movers, p / np.where(movers, den, 1.0), 0.0)
        e_it = panel.y - fitted[match]
        eta_sum = np.bincount(match, weights=e_it)
        eta_h = e_it + gain[match] * eta_sum[match]
        y_c = panel.y - panel.y.mean()
        sigma2 = np.bincount(match, weights=y_c * eta_h) / T
    else:
        raise ValueError("se_variance must be 'match' or 'person_year'")

    forms = quadratic_forms(mp, J, W, person_years=panel.n_obs)
    out = {}
    for name in ("var(psi)", "cov(psi, alpha)"):
        Q = forms[name]
        B = X @ S_inv @ Q @ S_inv @ X.T
        B_ii = np.diag(B).copy()
        b = np.zeros(n_match)
        b[movers] = B_ii[movers] / (1.0 - P_ii[movers])
        C = B - 0.5 * (b[:, None] * Mt + Mt * b[None, :])
        if centering == "weighted":
            K, y = C, y_w
        elif centering == "reference":
            u = Mt @ b
            K = C + (np.outer(np.ones(n_match), u)
                     + np.outer(u, np.ones(n_match))) / (2.0 * n_match)
            y = y_r
        else:
            raise ValueError("centering must be 'reference' or 'weighted'")
        smooth = smooth_sigma2(sigma2, P_ii, B_ii, n_bins)
        floor = np.maximum(smooth, 0.0)
        Ky = K @ y
        first = 4.0 * float(smooth @ Ky ** 2)
        trace = float(smooth @ (K ** 2) @ floor)
        V = first - 2.0 * trace
        out[name] = {
            "theta": float(y @ K @ y), "V": V, "first": first,
            "trace": 2.0 * trace,
            "se": float(np.sqrt(V)) if V > 0 else float("nan"),
            "se_conservative": float(np.sqrt(first)) if first > 0
            else float("nan"),
            "max_abs_diagonal": float(np.abs(np.diag(K)).max()),
        }
    return out


# --------------------------------------------------------------------------
# inference
# --------------------------------------------------------------------------
#
# Everything here lives in the square-root-weight metric, where the weighted
# problem *is* the unweighted one: with X~ = W^(1/2) X and y~ = W^(1/2) y, the
# errors have variance w_i Var(e_i), which is exactly what
# `sigma2_leave_one_out` returns. So the algebra below is written once, with no
# weights in it, and the transform is applied at the edges.
#
# The estimator is a quadratic form with a zero diagonal. Write
#
#     C = B - (1/2) (diag(b) M + M diag(b)),   b_i = B_ii / (1 - P_ii)
#
# with B = X~ S^- Q S^- X~', M = I - P~. Then C_ii = B_ii - M_ii b_i = 0, and
#
#     y~' C y~ = y~' B y~ - sum_i b_i y~_i (M y~)_i = plug-in - sum_i B_ii s2_i
#
# which is the KSS estimator itself. That identity is exact and is what
# `c_matrix` is tested against: no Monte Carlo needed to check it.
#
# Because the diagonal vanishes, E[y~'Cy~] = mu'C mu = theta with no trace
# term, and for independent errors
#
#     V[theta-hat] = 4 sum_i s2_i (C mu)_i^2 + 2 tr(C Om C Om),   Om = diag(s2)
#
# with no appeal to normality -- the third and fourth moment terms carry a
# factor of C_ii and drop out. This is the variance in KSS Theorem 2, whose
# second term they write as 2 sum_i sum_{l != i} C_il^2 s2_i s2_l; that equals
# the trace precisely because the diagonal is zero.


def bii_rows(A, Q, S_inv, w=None):
    """B_ii per observation, in the square-root-weight metric.

    B_ii = w_i a_i' S^- Q S^- a_i. This is the per-row quantity whose weighted
    sum against sigma2 is the bias term, and the streaming code deliberately
    does *not* form it -- it estimates the sum directly in coefficient space,
    which is cheaper and less noisy. Inference needs the individual values.
    """
    Ad = A.toarray()
    quadratic = np.einsum("ij,jk,ik->i", Ad, S_inv @ Q @ S_inv, Ad)
    return quadratic if w is None else w * quadratic


def c_matrix(A, Q, S_inv, P_ii, w=None):
    """The dense n x n kernel C of the leave-out quadratic form.

    Only ever used as a reference: it is n x n, so it is what limits the
    prototype's size. The streaming version applies C to a vector instead.
    """
    Ad = A.toarray()
    X = Ad if w is None else np.sqrt(w)[:, None] * Ad
    B = X @ S_inv @ Q @ S_inv @ X.T
    M = np.eye(len(X)) - X @ S_inv @ X.T
    b = np.diag(B) / (1.0 - P_ii)
    return B - 0.5 * (b[:, None] * M + M * b[None, :])


def variance_of_theta(C, sigma2, y_tilde):
    """(V-hat, its two terms) for one component, exactly.

    V-hat = 4 sum_i s2_i (C y~)_i^2 - 2 sum_ij C_ij^2 s2_i s2_j.

    The first term is the plug-in for 4 sum s2_i (C mu)_i^2, and it is upward
    biased: E[(C y~)_i^2] = (C mu)_i^2 + sum_l C_il^2 s2_l, so its expectation
    overshoots by exactly 4 tr(C Om C Om). Since the second term of V is
    +2 tr(C Om C Om), subtracting 2 tr(C Om C Om) leaves the whole thing
    unbiased -- which is why the correction enters with the sign it does, and
    why the first term alone is conservative rather than anticonservative.
    """
    Cy = C @ y_tilde
    plug_in = 4.0 * float(sigma2 @ (Cy ** 2))
    trace = float(sigma2 @ (C ** 2) @ sigma2)
    return plug_in - 2.0 * trace, plug_in, trace


def trace_by_simulation(C, sigma2, n_draws=1000, seed=0, rademacher=False):
    """tr(C Om C Om) by random projection, the way the streaming code must.

    With v = Om^(1/2) z and E[zz'] = I, E[sum_i s2_i (C v)_i^2] = tr(C Om C Om).
    That needs one application of C per draw and no distributional assumption;
    Saggio's reference instead takes Var(v'Cv) across Gaussian draws, which is
    the same target by way of 2 tr(.) but relies on normality and spends the
    same solve. Both are here so the cheaper one can be checked.
    """
    rng = np.random.default_rng(seed)
    root = np.sqrt(np.maximum(sigma2, 0.0))
    direct = np.empty(n_draws)
    quadratic = np.empty(n_draws)
    for draw in range(n_draws):
        z = (rng.integers(0, 2, len(root)) * 2.0 - 1.0 if rademacher
             else rng.standard_normal(len(root)))
        v = root * z
        Cv = C @ v
        direct[draw] = sigma2 @ (Cv ** 2)
        quadratic[draw] = v @ Cv
    return float(direct.mean()), float(quadratic.var())


def smooth_sigma2(sigma2, P_ii, B_ii, n_bins=None):
    """sigma2 smoothed against (P_ii, B_ii): the package's own smoother.

    The smoother is a modeling choice rather than something this reference
    exists to check, so it is shared: dense and streaming then differ only in
    exact versus estimated leverages, which is the comparison that means
    something. It stands in for the reference's lowess -- quantile-rank grid at
    the lowess bandwidth, cell means interpolated bilinearly, so the fit is
    continuous in its regressors. See hdfe_stream.leaveout_se.fit_smoother.
    """
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from hdfe_stream.leaveout_se import fit_smoother, smooth_eval

    return smooth_eval(fit_smoother(sigma2, P_ii, B_ii, n_bins), P_ii, B_ii)


def kss_se(panel, stayers="own", sigma2_rule="raw", n_bins=None):
    """Standard errors for the three components, exactly (dense C).

    `sigma2_rule` is "raw" for the leave-out sigma2 itself or "smooth" for the
    binned smooth standing in for the reference's lowess fit.

    Returns dicts keyed the same way as `kss`: the point estimate recomputed as
    y~'Cy~ (which must equal plug-in minus bias), the variance, its two terms,
    and the standard error.
    """
    A, J, W = design(panel)
    weights = panel.w
    S_inv = _dense_inverse(A, panel.weight)
    beta = S_inv @ (A.T @ (weights * panel.y))
    P_ii = leverages(A, S_inv, panel.weight)
    sigma2, movers = sigma2_leave_one_out(panel, A @ beta, P_ii, stayers)
    # demeaned, to match sigma2: C1 = -M b / 2 is not zero, so y'Cy does depend
    # on the location of y. sigma2_leave_one_out uses (y - ybar) following
    # Saggio, which trades an O(1/n) bias for a large variance reduction -- y_i
    # itself carries the whole mean of the outcome. Whatever convention the point
    # estimate uses, the variance has to describe *that* estimator, so the two
    # are tied together here rather than chosen independently.
    y_tilde = np.sqrt(weights) * (panel.y - (weights @ panel.y) / weights.sum())

    out = {}
    for name, Q in quadratic_forms(panel, J, W).items():
        B_ii = bii_rows(A, Q, S_inv, panel.weight)
        C = c_matrix(A, Q, S_inv, P_ii, panel.weight)
        used = (smooth_sigma2(sigma2, P_ii, B_ii, n_bins)
                if sigma2_rule == "smooth" else sigma2)
        V, first, trace = variance_of_theta(C, used, y_tilde)
        out[name] = {
            "theta": float(y_tilde @ C @ y_tilde),
            "plug_in": float(beta @ Q @ beta),
            "bias": float(B_ii @ sigma2),
            "V": V, "V_first_term": first, "V_trace_term": 2.0 * trace,
            "se": float(np.sqrt(V)) if V > 0 else float("nan"),
            "se_conservative": float(np.sqrt(first)),
            "max_abs_C_diagonal": float(np.abs(np.diag(C)).max()),
        }
    return out


# --------------------------------------------------------------------------
# weak identification: the eigenvalue diagnostic of KSS Section 5
# --------------------------------------------------------------------------
#
# The normal interval rests on Theorem 2 condition (ii): lambda_1^2 / sum
# lambda^2 = o(1), for the eigenvalues of A~ = S^(-1/2) Q S^(-1/2) -- the same as
# those of S^- Q. KSS's threshold rule (Section 6.2) takes q to be the number of
# leading eigenvalues with lambda_l^2 / sum lambda^2 >= 1/10; q = 0 is the case
# where the normal interval is justified.
#
# The first-stage F is b1^2 / V[b1], with b1 the OLS estimate of the leading
# eigen-direction's nuisance parameter: b1 = sum_i x1bar_i y~_i, where
# x1bar_i = sqrt(w_i) x_i' u1 and Q u1 = lambda_1 S u1. It is scale-free, so the
# eigenvector's normalization does not matter. V[b1] is linear in sigma2, so the
# ordinary leave-out sigma2 is unbiased for it -- but it is often negative, and
# the leading direction can load on a few rows, so the sum can come out negative.
# Like the reference, V[b1] therefore uses sigma2 smoothed: here against the
# leverage alone, in quantile bins.

WEAK_ID_THRESHOLD = 0.1


def _smooth_on_leverage(sigma2, P_ii, n_bins=None):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from hdfe_stream.leaveout_se import fit_smoother, smooth_eval

    return smooth_eval(fit_smoother(sigma2, P_ii, None, n_bins), P_ii)


def weak_id(panel, stayers="own", top=3):
    """Exact eigenvalue diagnostic for all three components, densely."""
    A, J, W = design(panel)
    weights = panel.w
    S_inv = _dense_inverse(A, panel.weight)
    beta = S_inv @ (A.T @ (weights * panel.y))
    P_ii = leverages(A, S_inv, panel.weight)
    sigma2, _movers = sigma2_leave_one_out(panel, A @ beta, P_ii, stayers)
    y_tilde = np.sqrt(weights) * (panel.y - (weights @ panel.y) / weights.sum())
    Ad = A.toarray()

    out = {}
    for name, Q in quadratic_forms(panel, J, W).items():
        T = S_inv @ Q
        lam, vec = np.linalg.eig(T)
        order = np.argsort(-np.abs(lam))
        lam = np.real(lam[order])
        sum_sq = float(np.real(np.trace(T @ T)))
        u1 = np.real(vec[:, order[0]])
        x1bar = np.sqrt(weights) * (Ad @ u1)
        b1 = float(x1bar @ y_tilde)
        v1 = float((x1bar ** 2) @ _smooth_on_leverage(sigma2, P_ii))
        ratios = (lam[:top] ** 2) / sum_sq
        q = 0
        for r in ratios:
            if r >= WEAK_ID_THRESHOLD:
                q += 1
            else:
                break
        out[name] = {"eigenvalues": lam[:top].tolist(), "sum_sq": sum_sq,
                     "ratios": ratios.tolist(), "q": q,
                     "f_stat": b1 * b1 / v1 if v1 > 0 else float("nan")}
    return out


# --------------------------------------------------------------------------
# the q = 1 weak-identification interval (KSS Sections 5-6)
# --------------------------------------------------------------------------
#
# With u1 the leading eigenvector of S^- Q (u1' S u1 = 1) and x1 = X~ u1 its row
# values, B = sum_l lambda_l x_l x_l', so deflating one rank,
#
#     B2 = B - lambda_1 x1 x1',   C2 = B2 - (1/2)(diag(b2) M + M diag(b2)),
#     b2 = diag(B2) / (1 - P_ii),
#
# gives y~' C2 y~ = theta-hat - lambda_1 (b1-hat^2 - sum x1^2 sigma2-hat) --
# exactly KSS's theta_1-hat, again a zero-diagonal quadratic form. So the stage-0
# algebra carries over: V[theta_1] = 4 sum s2 (C2 mu)^2 + 2 tr(C2 Om C2 Om), and
# Cov(b1, theta_1) = 2 sum x1 s2 (C2 mu), with no appeal to normality.


def weak_design(panel):
    """Everything in the q = 1 interval that depends on the design, not on y."""
    A, J, W = design(panel)
    weights = panel.w
    S_inv = _dense_inverse(A, panel.weight)
    Ad = A.toarray()
    Xt = np.sqrt(weights)[:, None] * Ad
    S = Xt.T @ Xt
    P = Xt @ S_inv @ Xt.T
    P_ii = np.diag(P).copy()
    M = np.eye(len(Xt)) - P
    parts = {"A": A, "S_inv": S_inv, "P_ii": P_ii, "components": {}}
    for name, Q in quadratic_forms(panel, J, W).items():
        lam, vec = np.linalg.eig(S_inv @ Q)
        top = int(np.argmax(np.abs(lam)))
        u1 = np.real(vec[:, top])
        u1 /= np.sqrt(u1 @ S @ u1)
        lam1 = float(np.real(lam[top]))
        x1 = Xt @ u1
        B = Xt @ S_inv @ Q @ S_inv @ Xt.T
        B2 = B - lam1 * np.outer(x1, x1)
        b = np.diag(B) / (1.0 - P_ii)
        b2 = np.diag(B2) / (1.0 - P_ii)
        parts["components"][name] = {
            "Q": Q, "lam1": lam1, "x1": x1, "B_ii": np.diag(B).copy(),
            "C": B - 0.5 * (b[:, None] * M + M * b[None, :]),
            "C2": B2 - 0.5 * (b2[:, None] * M + M * b2[None, :]),
        }
    return parts


def kss_weak(panel, parts=None, alpha=0.05, stayers="own", n_bins=None,
             smooth_input="y_resid"):
    """Normal and q = 1 (Andrews-Mikusheva) intervals for all three components.

    The center is the estimator the package reports -- plug-in minus bias, with
    the stayer imputation -- and theta_1-hat = theta-hat - lambda_1 (b1^2 -
    sum x1^2 sigma2-hat) is formed from it with the same sigma2-hat. The variances
    use sigma2 smoothed on (P_ii, B_ii) per component, as the standard errors do.
    """
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from hdfe_stream.am_interval import am_interval

    parts = weak_design(panel) if parts is None else parts
    A, S_inv, P_ii = parts["A"], parts["S_inv"], parts["P_ii"]
    weights = panel.w
    beta = S_inv @ (A.T @ (weights * panel.y))
    sigma2, _movers = sigma2_leave_one_out(panel, A @ beta, P_ii, stayers)
    y_tilde = np.sqrt(weights) * (panel.y - (weights @ panel.y) / weights.sum())

    # What the smoother averages. "y_resid" is the reference's choice, the
    # leave-out sigma2 itself; "resid_sq" is w e^2 / (1 - P_ii), whose
    # expectation is a local weighted average of sigma2 (weights M_il^2 / M_ii,
    # summing to one, M_ii on the row itself) and which drops the factor of the
    # outcome's mean that makes the other so noisy.
    resid = panel.y - A @ beta
    smooth_on = (sigma2 if smooth_input == "y_resid"
                 else weights * resid ** 2 / (1.0 - P_ii))

    out = {}
    for name, comp in parts["components"].items():
        C, C2, x1, lam1 = comp["C"], comp["C2"], comp["x1"], comp["lam1"]
        theta = float(beta @ comp["Q"] @ beta) - float(comp["B_ii"] @ sigma2)
        s = smooth_sigma2(smooth_on, P_ii, comp["B_ii"], n_bins)

        Cy = C @ y_tilde
        V = 4 * float(s @ Cy ** 2) - 2 * float(s @ (C ** 2) @ s)
        se = np.sqrt(V) if V > 0 else np.nan

        b1 = float(x1 @ y_tilde)
        theta1 = theta - lam1 * (b1 * b1 - float((x1 ** 2) @ sigma2))
        # Variance weights must be nonnegative. A bin mean of the raw leave-out
        # sigma2 can come out negative when bins are small, and with s >= 0
        # Cauchy-Schwarz guarantees s12^2 <= s11 * first, so the conservative
        # fallback below is always positive definite.
        s = np.maximum(s, 0.0)
        C2y = C2 @ y_tilde
        s11 = float((x1 ** 2) @ s)
        s12 = 2 * float((x1 * s) @ C2y)
        first = 4 * float(s @ C2y ** 2)
        s22 = first - 2 * float(s @ (C2 ** 2) @ s)
        conservative = not (s22 > 0 and s12 * s12 < s11 * s22 * (1 - 1e-12))
        if conservative:
            # KSS Remark 9: a variance estimate biased upward keeps the interval
            # valid, only wider. The first term alone is exactly that.
            s22 = first
        Sigma = np.array([[s11, s12], [s12, s22]])
        lo, hi, info = am_interval(lam1, b1, theta1, Sigma, alpha)
        out[name] = {"theta": theta, "se": float(se), "normal": (theta - 1.96 * se,
                                                                  theta + 1.96 * se),
                     "weak": (lo, hi), "lam1": lam1, "b1": b1, "theta1": theta1,
                     "Sigma": Sigma, "conservative": conservative, **info}
    return out


# --------------------------------------------------------------------------
# sample selection
# --------------------------------------------------------------------------

def largest_connected_set(panel):
    """Restrict to the largest set of firms linked by workers who moved.

    Worker and firm effects are only comparable within such a set; across
    components the normalization is arbitrary.
    """
    n_firms = panel.n_firms
    parent = np.arange(n_firms)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    order = np.lexsort((panel.firm, panel.worker))
    w, f = panel.worker[order], panel.firm[order]
    for i in range(1, len(w)):
        if w[i] == w[i - 1]:
            a, b = find(f[i]), find(f[i - 1])
            if a != b:
                parent[max(a, b)] = min(a, b)
    labels = np.array([find(x) for x in range(n_firms)])

    sizes = np.bincount(labels[panel.firm])
    biggest = int(np.argmax(sizes))
    return panel.subset(labels[panel.firm] == biggest)


def prune_to_leave_out(panel, tol=1e-9, max_rounds=50):
    """Drop observations until every mover has leverage strictly below one.

    NOTE this is a stand-in, not KSS's leave-one-out connected set. KSS define
    the sample by a graph condition -- the largest set of firms that stays
    connected after removing any single worker's observations -- and prune to it
    before estimating. Here we instead detect the *consequence* of violating
    that condition (a mover with P_ii = 1, whose leave-one-out residual is
    degenerate) and iterate until it is gone.

    The two coincide often but not always, which matters when comparing against
    an implementation that does the graph version: do the comparison on a sample
    that has already been pruned by *one* of them, not on raw data. See
    `compare_pytwoway.py`.
    """
    panel = largest_connected_set(panel)
    for round_index in range(max_rounds):
        A, _, _ = design(panel)
        P_ii = leverages(A)
        movers = panel.movers
        bad = movers & (P_ii > 1.0 - tol)
        if not bad.any():
            return panel, round_index
        panel = panel.subset(~bad)
        panel = largest_connected_set(panel)
    raise RuntimeError(f"leverage pruning did not settle in {max_rounds} rounds")

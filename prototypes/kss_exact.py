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

Two details that are easy to get wrong, both taken from pytwoway's `fe.py`
(`_estimate_Sii_he`), which follows Saggio's reference MATLAB:

  * sigma2_hat_i uses the *demeaned* outcome (y_i - y_bar), not y_i;
  * it is only defined for movers. Stayers -- workers seen at a single firm --
    have no within-worker firm variation, so they get an imputed value: by
    default the mean of the movers' estimates at the same firm.

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
    extra: dict = field(default_factory=dict)

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
        return Panel(worker=worker, firm=firm, y=self.y, extra=self.extra)

    def subset(self, mask):
        return Panel(worker=self.worker[mask], firm=self.firm[mask],
                     y=self.y[mask],
                     extra={k: v[mask] for k, v in self.extra.items()}).recode()

    @classmethod
    def from_frame(cls, frame, worker="worker_id", firm="firm_id", y="log_earn",
                   extra=()):
        """Build from a Polars DataFrame (or anything indexable by column)."""
        _, w = np.unique(np.asarray(frame[worker]), return_inverse=True)
        _, f = np.unique(np.asarray(frame[firm]), return_inverse=True)
        return cls(worker=w, firm=f, y=np.asarray(frame[y], dtype=float),
                   extra={c: np.asarray(frame[c], dtype=float) for c in extra})


def design(panel):
    """A = [J, W], with the last firm dummy dropped for identification.

    Dropping one firm column is the usual AKM normalisation: it sets the last
    firm's effect to zero. The variance components are location-invariant, so
    they do not depend on which column goes.
    """
    n, n_firms, n_workers = panel.n_obs, panel.n_firms, panel.n_workers
    rows = np.arange(n)
    J = sp.csr_matrix((np.ones(n), (rows, panel.firm)), shape=(n, n_firms))
    W = sp.csr_matrix((np.ones(n), (rows, panel.worker)), shape=(n, n_workers))
    return sp.hstack([J[:, :-1], W], format="csr"), J, W


def _dense_inverse(A):
    """(A'A)^-1, densely. The prototype's scale limit lives here."""
    S = (A.T @ A).toarray()
    return np.linalg.pinv(S, hermitian=True)


def leverages(A, S_inv=None):
    """Exact P_ii = a_i' (A'A)^-1 a_i for every observation."""
    S_inv = _dense_inverse(A) if S_inv is None else S_inv
    # rows of A are sparse (two nonzeros), so do it as a rowwise quadratic form
    AS = A @ S_inv                      # n x K
    return np.einsum("ij,ij->i", AS, A.toarray())


def quadratic_forms(panel, J, W):
    """Q matrices for var(psi), var(alpha) and cov(psi, alpha).

    Each is built in the stacked [psi (minus the dropped firm); alpha] basis, so
    that theta = b' Q b reproduces the observation-level moment. The demeaning
    matrix M = I - ii'/n is what makes these *variances* rather than second
    moments; it is applied to the n-row selector matrices before the product.
    """
    n = panel.n_obs
    Jd = J[:, :-1].toarray()            # matches the dropped column in `design`
    Wd = W.toarray()

    def centered(M):
        return M - M.mean(axis=0, keepdims=True)

    Jc, Wc = centered(Jd), centered(Wd)
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
        "var(psi)": blocks(top_left=Jc.T @ Jc / n),
        "var(alpha)": blocks(bottom_right=Wc.T @ Wc / n),
        # symmetrised so that b' Q b is the covariance itself
        "cov(psi, alpha)": blocks(top_right=Jc.T @ Wc / (2 * n)),
    }


def sigma2_leave_one_out(panel, y_hat, P_ii, stayers="firm_mean"):
    """Leave-one-out variance estimates, unbiased under heteroskedasticity.

    For movers, sigma2_i = (y_i - y_bar) * e_i / (1 - P_ii): this is the
    algebraic form of y_i (y_i - x_i' b_hat_{-i}), which has expectation
    sigma2_i without assuming the errors are identically distributed.

    Stayers have P_ii determined entirely by their own worker effect, so the
    leave-one-out residual is degenerate. `stayers="firm_mean"` imputes the mean
    of the movers' estimates at the same firm, which is pytwoway's default.
    """
    resid = panel.y - y_hat
    movers = panel.movers
    sigma2 = np.full(panel.n_obs, np.nan)
    sigma2[movers] = ((panel.y[movers] - panel.y.mean())
                      * resid[movers] / (1.0 - P_ii[movers]))

    if stayers == "firm_mean":
        totals = np.zeros(panel.n_firms)
        counts = np.zeros(panel.n_firms)
        np.add.at(totals, panel.firm[movers], sigma2[movers])
        np.add.at(counts, panel.firm[movers], 1.0)
        firm_mean = np.divide(totals, counts, out=np.full(panel.n_firms, np.nan),
                              where=counts > 0)
        sigma2[~movers] = firm_mean[panel.firm[~movers]]
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


def kss(panel, stayers="firm_mean", leverage_tol=1e-9):
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
    S_inv = _dense_inverse(A)

    beta = S_inv @ (A.T @ panel.y)
    y_hat = A @ beta
    P_ii = leverages(A, S_inv)

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
        # B_ii = a_i' S^-1 Q S^-1 a_i, rowwise
        M = S_inv @ Q @ S_inv
        B_ii = np.einsum("ij,jk,ik->i", Ad, M, Ad)
        bias[name] = float(B_ii @ sigma2)
        corrected[name] = plug_in[name] - bias[name]

    return KSSResult(
        n_obs=panel.n_obs, n_workers=panel.n_workers, n_firms=panel.n_firms,
        n_movers=int(movers.sum()), plug_in=plug_in, kss=corrected, bias=bias,
        max_leverage=float(P_ii[movers].max()) if movers.any() else float("nan"),
        sigma2_mean=float(np.mean(sigma2)),
        diagnostics={"n_coef": A.shape[1], "rank_deficiency":
                     A.shape[1] - int(np.linalg.matrix_rank(Ad))},
    )


# --------------------------------------------------------------------------
# sample selection
# --------------------------------------------------------------------------

def largest_connected_set(panel):
    """Restrict to the largest set of firms linked by workers who moved.

    Worker and firm effects are only comparable within such a set; across
    components the normalisation is arbitrary.
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

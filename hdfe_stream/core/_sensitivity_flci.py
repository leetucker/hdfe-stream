"""The optimal fixed-length confidence interval (FLCI) of Rambachan and Roth
(2023, section 4.1; Armstrong and Kolesar, 2018) for the smoothness
restriction Delta^SD(M).

The interval is an affine estimator a + v'beta, v = (v_pre, l), plus and
minus a half-length chosen so that it covers theta = l'tau_post for every
violation delta in Delta^SD(M). R's HonestDiD writes v_pre through weights w
on the pre-period coefficients and, for each standard deviation h of the
estimator, minimizes the worst-case bias subject to sd <= h with a conic
solver, then searches over h. Writing z for the cumulative sums of w (R's
L w), the worst-case bias is

    M * (constant + |z_K| + sum_{k<K} |z_k|),     z_K = sum_k w_k fixed,

and the variance is a quadratic in z. The pairs (variance, bias) that are
optimal for some h are therefore the solutions of the lasso

    min_u  u'P u / 2 + q'u + mu ||u||_1,        u = (z_1, ..., z_{K-1}),

as mu runs from 0 (the smallest variance, R's hMin) to max|q| and beyond
(the smallest bias, R's h0). The lasso's solution is piecewise linear in mu,
so its whole path is computed exactly by the homotopy (LARS) algorithm, and
the half-length h * cv(M * bias / h) is minimized on each linear piece, where
cv is the 1 - alpha quantile of the folded normal |N(t, 1)|. R estimates
that quantile from 10^6 simulated draws; here it is computed exactly.
"""

from __future__ import annotations

import numpy as np
from scipy import optimize, special


def folded_normal_quantile(p: float, t: float) -> float:
    """The p quantile of |X|, X ~ N(t, 1)."""
    t = abs(float(t))
    if not np.isfinite(t):
        return np.inf

    def f(c):
        return special.ndtr(c - t) - special.ndtr(-c - t) - p

    hi = t + 10.0
    while f(hi) < 0:
        hi *= 2
    return optimize.brentq(f, 0.0, hi, xtol=1e-14, rtol=4 * np.finfo(float).eps)


def _bias_constant(l):
    T = len(l)
    s = sum(abs(np.dot(np.arange(1, k + 1), l[T - k:])) for k in range(1, T + 1))
    return s - np.dot(np.arange(1, T + 1), l)


def _lasso_path(P, q):
    """Pieces (mu_lo, mu_hi, active, a, b) of the lasso path, u_A = a + mu b,
    from mu = max|q| down to 0."""
    n = len(q)
    mu = float(np.max(np.abs(q))) if n else 0.0
    pieces = []
    if n == 0 or mu <= 0:
        return pieces, mu
    j0 = int(np.argmax(np.abs(q)))
    active = [j0]
    signs = {j0: -np.sign(q[j0])}
    for _ in range(20 * n + 20):
        A = np.array(active)
        sA = np.array([signs[j] for j in active])
        PAA = P[np.ix_(A, A)]
        a = -np.linalg.solve(PAA, q[A])
        b = -np.linalg.solve(PAA, sA)
        tol = 1e-12 * max(1.0, mu)
        best, event = 0.0, None
        inactive = [j for j in range(n) if j not in signs]
        for j in inactive:
            cj = q[j] + P[j, A] @ a
            dj = P[j, A] @ b
            for sgn in (1.0, -1.0):
                den = sgn - dj
                if den != 0:
                    m = cj / den
                    if tol < m < mu - tol and m > best:
                        best, event = m, ("join", j, -sgn)
        for idx, j in enumerate(active):
            if b[idx] != 0:
                m = -a[idx] / b[idx]
                if tol < m < mu - tol and m > best:
                    best, event = m, ("leave", j, 0.0)
        pieces.append((best, mu, A.copy(), a, b))
        if event is None:
            break
        kind, j, s = event
        if kind == "join":
            active.append(j)
            signs[j] = s
        else:
            active.remove(j)
            del signs[j]
        mu = best
    return pieces, float(np.max(np.abs(q)))


def optimal_flci(sigma, M, n_pre, n_post, l, alpha):
    """R's `.findOptimalFLCI_helper`: the optimal FLCI's direction
    v = (v_pre, l) (length n_pre + n_post) and half-length."""
    sigma = np.asarray(sigma, dtype=float)
    l = np.asarray(l, dtype=float)
    K, T = n_pre, n_post
    D = np.eye(K) - np.eye(K, k=-1)
    D2 = D @ D
    Spre = sigma[:K, :K]
    Spp = sigma[:K, K:]
    v0 = float(l @ sigma[K:, K:] @ l)
    Pf = D2.T @ Spre @ D2
    rf = D2.T @ (Spp @ l)
    cw = float(np.dot(np.arange(1, T + 1), l))
    base_bias = _bias_constant(l) + abs(cw)
    p = 1 - alpha

    def half_length(h, bias):
        return h * folded_normal_quantile(p, M * bias / h) if h > 0 else np.inf

    if K == 1:
        z = np.array([cw])
        h = np.sqrt(Pf[0, 0] * cw * cw + 2 * rf[0] * cw + v0)
        return np.concatenate([D2 @ z, l]), half_length(h, base_bias)
    P = Pf[:-1, :-1]
    q = Pf[:-1, -1] * cw + rf[:-1]
    const = Pf[-1, -1] * cw * cw + 2 * rf[-1] * cw + v0

    def point(u):
        var = u @ P @ u + 2 * q @ u + const
        h = np.sqrt(max(var, 0.0))
        return h, base_bias + np.abs(u).sum()

    pieces, mu_max = _lasso_path(P, q)
    u0 = np.zeros(K - 1)
    best_u = u0
    best = half_length(*point(u0))

    def u_at(piece, mu):
        _, _, A, a, b = piece
        u = np.zeros(K - 1)
        u[A] = a + mu * b
        return u

    for piece in pieces:
        lo, hi = piece[0], piece[1]
        if M == 0 and lo > 0:
            continue
        grid = np.linspace(lo, hi, 9)
        vals = [half_length(*point(u_at(piece, m))) for m in grid]
        i = int(np.argmin(vals))
        if vals[i] < best:
            best, best_u = vals[i], u_at(piece, grid[i])
        a_, b_ = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
        if b_ > a_:
            res = optimize.minimize_scalar(
                lambda m: half_length(*point(u_at(piece, m))), bounds=(a_, b_),
                method="bounded", options={"xatol": 1e-12 * max(1.0, mu_max)})
            if res.fun < best:
                best, best_u = float(res.fun), u_at(piece, res.x)
    z = np.concatenate([best_u, [cw]])
    return np.concatenate([D2 @ z, l]), best

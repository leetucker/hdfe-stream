"""numba kernels for `core.newton`: one fused pass over a chunk of rows that
gives, for each of M models, the log-likelihood on each split, the gradient,
and (optionally) the information, for the logit, probit and Poisson with no
absorbed fixed effect or one.

The linear predictor of row i in model m is

    eta = off_i + a[m, g(i)] + x_i'b[m]

with a[m, g] the effect of the row's group (the intercept when there is one
group). The information has an arrow shape: a block for b (the `columns`),
a diagonal for a, and the cross block between them. The kernel forms the
three for every model in one read of the rows.

Each row's nonzeros are found once and shared by every model; with
`use_nnz` the products then run over them only, which is what makes designs
of indicators (rows with a few nonzeros among many columns) cheap: a row's
information costs (nonzeros)^2 / 2 instead of c^2 / 2.

The groups' sums are accumulated one of two ways. When the rows of a chunk
come grouped (each group's rows consecutive) and the parts handed to the
threads start at group boundaries (`part_starts`), a thread owns whole
groups: it sums a group's rows locally and adds them to the shared arrays
once, at the end of the group's run, so no entry is written by two threads.
Otherwise every part keeps its own copy of the groups' sums.
"""

from __future__ import annotations

import math

import numba as nb
import numpy as np

from .kernels_glm import LOGIT, POISSON, PROBIT, _expit, _log_expit, _log_ndtr

_LOG_2PI_HALF = 0.9189385332046727        # log(2 pi) / 2

__all__ = ["LOGIT", "POISSON", "PROBIT", "newton_pass_rows", "newton_row_terms",
           "part_starts", "row_terms"]


@nb.njit(cache=True, inline="always", error_model="numpy")
def row_terms(family, yi, eta, observed=False):
    """One row's log-likelihood, score (d loglik / d eta) and expected
    information (-E d2 loglik / d eta2) at eta, or with `observed` the
    observed one (-d2 loglik / d eta2): (ll, r, h).

    For the logit and Poisson the two are the same; for the probit the
    expected information is phi^2 / (Phi (1 - Phi)), Fisher scoring's weight,
    and the observed one r (r + eta) (y = 1) or r (r - eta) with r the
    inverse Mills ratio, which Newton's method converges quadratically with.
    The binary families take 1 - mu from -eta and the logarithms of the tails
    from series that do not underflow, so that rows far in a tail keep their
    score."""
    if family == LOGIT:
        lp = _log_expit(eta)
        lq = _log_expit(-eta)
        p = _expit(eta)
        q = _expit(-eta)
        if yi > 0.0:
            return lp, q, p * q
        return lq, -p, p * q
    if family == PROBIT:
        lp = _log_ndtr(eta)
        lq = _log_ndtr(-eta)
        lphi = -0.5 * eta * eta - _LOG_2PI_HALF
        if yi > 0.0:
            m = math.exp(lphi - lp)
            return lp, m, (m * (m + eta) if observed else math.exp(2.0 * lphi - lp - lq))
        m = math.exp(lphi - lq)
        return lq, -m, (m * (m - eta) if observed else math.exp(2.0 * lphi - lp - lq))
    # Poisson
    mu = math.exp(eta)
    return yi * eta - mu - math.lgamma(yi + 1.0), yi - mu, mu


def part_starts(f, n_parts):
    """Row positions that cut a chunk into `n_parts` parts of about equal
    numbers of rows, each cut moved forward to the start of a group (the
    codes `f` come in runs): ``n_parts + 1`` nondecreasing positions from 0
    to n."""
    n = len(f)
    cuts = np.flatnonzero(f[1:] != f[:-1]) + 1
    targets = (np.arange(1, n_parts, dtype=np.int64) * n) // n_parts
    if len(cuts):
        idx = np.searchsorted(cuts, targets)
        snapped = np.where(idx < len(cuts), cuts[np.minimum(idx, len(cuts) - 1)], n)
    else:
        snapped = np.full(len(targets), n)
    return np.r_[0, snapped, n].astype(np.int64)


@nb.njit(parallel=True, cache=True, error_model="numpy")
def newton_pass_rows(family, X, y, w, off, f, s, B, A, inc, colpos, info, use_nnz, grouped,
                     starts, ll_t, sk_t, gb_t, H_t, ga_t, D_t, C_t, ga, D, C,
                     observed=False, meat=0):
    """Add a chunk's rows to the sums (see the module docstring).

    X (n, k), y (n,), w and off (n,) or empty, f the group codes and s the
    split codes (n,); B (M, k) and A (M, G) each model's coefficients and
    effects; inc (M, S) the splits each model's gradient and information
    cover; colpos (k,) each covariate's position among the information's
    columns, or -1. `starts` cuts the rows into the parts the threads take
    (at group boundaries when `grouped`).

    Per part: ll_t and sk_t (P, M, S) the log-likelihood and the rows
    skipped (a non-finite linear predictor: a group held out of the model),
    gb_t (P, M, k) the gradient, H_t (P, M, c, c) the information's upper
    triangle. The groups' sums -- the effects' gradient, information and
    cross block -- go to ga_t, D_t, C_t ((P, M, G), (P, M, G), (P, M, G, c)),
    or, when `grouped`, to the shared ga, D, C ((M, G), (M, G), (M, G, c)).

    With `meat` the "information" sums take each row's squared score in
    place of its information -- (w r)^2 (``meat=2``: no weights or analytic
    ones) or w r^2 (``meat=1``: frequency weights, a row standing for w
    rows) -- which gives the heteroskedasticity-robust variance's meat in
    the same arrow shape.
    """
    n, k = X.shape
    M = B.shape[0]
    c = H_t.shape[2]
    n_parts = len(starts) - 1
    has_w = len(w) > 0
    has_off = len(off) > 0
    for t in nb.prange(n_parts):
        nz = np.empty(k, np.int64)
        xv = np.empty(k)
        loc_s = np.zeros(M)
        loc_h = np.zeros(M)
        loc_c = np.zeros((M, c))
        cur = -1
        for i in range(starts[t], starts[t + 1]):
            gi = f[i]
            if grouped and gi != cur:
                if cur >= 0:
                    for m in range(M):
                        ga[m, cur] += loc_s[m]
                        loc_s[m] = 0.0
                        if info:
                            D[m, cur] += loc_h[m]
                            loc_h[m] = 0.0
                            for p in range(c):
                                C[m, cur, p] += loc_c[m, p]
                                loc_c[m, p] = 0.0
                cur = gi
            cnt = 0
            if use_nnz:
                for j in range(k):
                    v = X[i, j]
                    if v != 0.0:
                        nz[cnt] = j
                        xv[cnt] = v
                        cnt += 1
            else:
                for j in range(k):
                    nz[j] = j
                    xv[j] = X[i, j]
                cnt = k
            wt = w[i] if has_w else 1.0
            yi = y[i]
            si = s[i]
            base = off[i] if has_off else 0.0
            for m in range(M):
                eta = base + A[m, gi]
                for q in range(cnt):
                    eta += B[m, nz[q]] * xv[q]
                if not np.isfinite(eta):
                    sk_t[t, m, si] += 1.0
                    continue
                ll, r, h = row_terms(family, yi, eta, observed)
                ll_t[t, m, si] += wt * ll
                if not inc[m, si]:
                    continue
                if meat == 2:
                    h = (wt * r) * (wt * r)
                elif meat == 1:
                    h = wt * r * r
                else:
                    h *= wt
                r *= wt
                for q in range(cnt):
                    gb_t[t, m, nz[q]] += r * xv[q]
                if grouped:
                    loc_s[m] += r
                else:
                    ga_t[t, m, gi] += r
                if not info:
                    continue
                if grouped:
                    loc_h[m] += h
                else:
                    D_t[t, m, gi] += h
                for q in range(cnt):
                    pa = colpos[nz[q]]
                    if pa < 0:
                        continue
                    hx = h * xv[q]
                    if grouped:
                        loc_c[m, pa] += hx
                    else:
                        C_t[t, m, gi, pa] += hx
                    for q2 in range(q, cnt):
                        pb = colpos[nz[q2]]
                        if pb < 0:
                            continue
                        if pa <= pb:
                            H_t[t, m, pa, pb] += hx * xv[q2]
                        else:
                            H_t[t, m, pb, pa] += hx * xv[q2]
        if grouped and cur >= 0:
            for m in range(M):
                ga[m, cur] += loc_s[m]
                if info:
                    D[m, cur] += loc_h[m]
                    for p in range(c):
                        C[m, cur, p] += loc_c[m, p]


@nb.njit(parallel=True, cache=True, error_model="numpy")
def newton_row_terms(family, y, w, eta, s, inc, starts, ll_t, sk_t, R, Hh, observed=False,
                     meat=0):
    """Each row's terms for every model, for the passes that form the sums
    with BLAS products instead (dense designs): eta (n, M) the linear
    predictors; into R and Hh (n, M) the weighted score and information (or,
    with `meat`, squared score, as in `newton_pass_rows`), zero for a row
    outside the model's splits (`inc`) or with a non-finite eta; ll_t and
    sk_t (P, M, S) each part's log-likelihood and rows skipped."""
    M = eta.shape[1]
    has_w = len(w) > 0
    for t in nb.prange(len(starts) - 1):
        for i in range(starts[t], starts[t + 1]):
            wt = w[i] if has_w else 1.0
            si = s[i]
            for m in range(M):
                e = eta[i, m]
                R[i, m] = 0.0
                Hh[i, m] = 0.0
                if not np.isfinite(e):
                    sk_t[t, m, si] += 1.0
                    continue
                ll, r, h = row_terms(family, y[i], e, observed)
                ll_t[t, m, si] += wt * ll
                if not inc[m, si]:
                    continue
                R[i, m] = wt * r
                if meat == 2:
                    Hh[i, m] = (wt * r) * (wt * r)
                elif meat == 1:
                    Hh[i, m] = wt * r * r
                else:
                    Hh[i, m] = wt * h

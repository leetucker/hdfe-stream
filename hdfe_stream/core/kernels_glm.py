"""numba kernels for the GLM (IRLS) path; see glm.py."""

from __future__ import annotations

import math

import numba as nb
import numpy as np


@nb.njit(parallel=True, cache=True)
def _nb_group_effects(starts, codes, n, sums, offs, fe, beta, out):
    """fe[0] effect of each group from its cells: the weighted group mean of
    z - x'beta - (other fixed effects),

        out[g] = sum_c (s_z,c - s_x,c beta - n_c sum_d fe[level_d,c]) / sum_c n_c,

    where column 0 of `sums` is the working response and the rest the
    covariates. A group whose weight has underflowed to zero gets 0."""
    D, k = codes.shape[1], beta.shape[0]
    for g in nb.prange(len(starts) - 1):
        num = 0.0
        den = 0.0
        for c in range(starts[g], starts[g + 1]):
            v = sums[c, 0]
            for j in range(k):
                v -= sums[c, j + 1] * beta[j]
            f = 0.0
            for d in range(D):
                f += fe[offs[d] + codes[c, d]]
            num += v - n[c] * f
            den += n[c]
        out[g] = num / den if den > 0 else 0.0


# --------------------------------------------------------------------------
# the IRLS row pass, fused
# --------------------------------------------------------------------------
# Scalar versions of what families.py computes with scipy, written as scipy
# computes them, so that the fused row pass gives the same numbers: the
# binary families take 1 - mu from -eta, and the logarithms of the tails come
# from series that do not underflow.

POISSON, LOGIT, PROBIT = 0, 1, 2
FAMILY_CODES = {"poisson": POISSON, "logit": LOGIT, "probit": PROBIT}

_SQRT1_2 = 0.7071067811865476
_LOG_2PI_HALF = 0.9189385332046727        # log(2 pi) / 2
_INV_SQRT_2PI = 0.3989422804014327
_EPS = 2.220446049250313e-16


@nb.njit(cache=True)
def _expit(x):
    if x < 0.0:
        e = np.exp(x)
        return e / (1.0 + e)
    return 1.0 / (1.0 + np.exp(-x))


@nb.njit(cache=True)
def _log_expit(x):
    if x < 0.0:
        return x - np.log1p(np.exp(x))
    return -np.log1p(np.exp(-x))


@nb.njit(cache=True)
def _ndtr(a):
    x = a * _SQRT1_2
    z = abs(x)
    if z < _SQRT1_2:
        return 0.5 + 0.5 * math.erf(x)
    y = 0.5 * math.erfc(z)
    return 1.0 - y if x > 0.0 else y


@nb.njit(cache=True)
def _log_ndtr(a):
    """scipy's log_ndtr: the asymptotic series below -20."""
    if a > 6.0:
        return -_ndtr(-a)
    if a > -20.0:
        return np.log(_ndtr(a))
    log_lhs = -0.5 * a * a - np.log(-a) - _LOG_2PI_HALF
    last, rhs, numerator, factor = 0.0, 1.0, 1.0, 1.0
    cons = 1.0 / (a * a)
    sign, i = 1.0, 0
    while abs(last - rhs) > _EPS:
        i += 1
        last = rhs
        sign = -sign
        factor *= cons
        numerator *= 2 * i - 1
        rhs += sign * numerator * factor
    return log_lhs + np.log(rhs)


@nb.njit(cache=True, inline="always")
def _row_terms(family, yi, eta):
    """One row's IRLS weight W, working-response update u (z = eta + u),
    deviance, log-likelihood and, for Poisson, log(y!), as families.py
    computes them."""
    lg = 0.0
    if family == POISSON:
        mu = np.exp(eta)
        Wi = mu
        u = yi / mu - 1.0 if mu > 0.0 else 0.0
        xlogy = yi * np.log(yi) if yi != 0.0 else 0.0
        dev = 2.0 * (xlogy - yi * eta - yi + mu)
        lg = math.lgamma(yi + 1.0)
        ll = yi * eta - mu - lg
    elif family == LOGIT:
        p = _expit(eta)
        q = _expit(-eta)
        Wi = p * q
        resid = q if yi > 0.0 else -p
        u = resid / Wi if Wi > 0.0 else 0.0
        dev = -2.0 * (_log_expit(eta) if yi > 0.0 else _log_expit(-eta))
        ll = -0.5 * dev
    else:
        p = _ndtr(eta)
        q = _ndtr(-eta)
        pdf = np.exp(-0.5 * eta * eta) * _INV_SQRT_2PI
        var = p * q
        Wi = pdf * pdf / var if var > 0.0 else 0.0
        resid = q if yi > 0.0 else -p
        u = resid / pdf if Wi > 0.0 and pdf > 0.0 else 0.0
        dev = -2.0 * (_log_ndtr(eta) if yi > 0.0 else _log_ndtr(-eta))
        ll = -0.5 * dev
    return Wi, u, dev, ll, lg


@nb.njit(parallel=True, cache=True)
def _nb_glm_rows(family, init, ybar, starts, gcell, cstarts, y, X, f, off, codes,
                 offs, beta, fe, a, V, w, nc, sc, parts):
    """One chunk of whole fe[0] groups through an IRLS row pass, in one sweep
    of each row: its linear predictor at the coefficients (beta, the other
    fixed effects `fe`, its group's effect `a`, the offset), the family's
    mean, IRLS weight W and working-response update u, the deviance and
    log-likelihood; then per cell (rows cstarts[c]:cstarts[c+1]; group g's
    cells are gcell[g]:gcell[g+1]) the weight nc[c] = sum w and sums
    sc[c] = sum w v of v = [z, x], z = eta + u less the offset, w = W f.
    V is left holding the weighted within-cell deviations sqrt(w)(v - vbar_c),
    whose cross-products the caller forms with one BLAS product, and w the
    row weights.

    With `init` the linear predictor is the family's starting value instead.
    f (frequency or analytic weights) and off may be empty. parts[b] holds
    block b's sums of the deviance, the log-likelihood, w y... (see below),
    for the caller to add up; the blocks are fixed, so the sums do not depend
    on the number of threads:

        0 deviance   1 log-likelihood   2 f y   3 f   4 f log(y!) (Poisson)
        5.. between-cell sums of squares, sc^2 / nc, per column of v
    """
    nblk = parts.shape[0]
    G = len(starts) - 1
    k = X.shape[1]
    D = codes.shape[1]
    m = k + 1
    weighted = f.shape[0] > 0
    offset = off.shape[0] > 0
    for b in nb.prange(nblk):
        for j in range(parts.shape[1]):
            parts[b, j] = 0.0
        for g in range(b * G // nblk, (b + 1) * G // nblk):
            for c in range(gcell[g], gcell[g + 1]):
                s, e = cstarts[c], cstarts[c + 1]
                for j in range(m):
                    sc[c, j] = 0.0
                tw = 0.0
                for i in range(s, e):
                    yi = y[i]
                    fi = f[i] if weighted else 1.0
                    if init:
                        eta = np.log(0.5 * (yi + ybar)) if family == POISSON else 0.0
                    else:
                        eta = a[g]
                        for j in range(k):
                            eta += X[i, j] * beta[j]
                        for d in range(D):
                            eta += fe[offs[d] + codes[i, d]]
                        if offset:
                            eta += off[i]
                    Wi, u, dev, ll, lg = _row_terms(family, yi, eta)
                    if family == POISSON:
                        parts[b, 4] += fi * lg
                    parts[b, 0] += fi * dev
                    parts[b, 1] += fi * ll
                    parts[b, 2] += fi * yi
                    parts[b, 3] += fi
                    wi = Wi * fi
                    w[i] = wi
                    z = eta + u
                    if offset:
                        z -= off[i]
                    V[i, 0] = z
                    for j in range(k):
                        V[i, j + 1] = X[i, j]
                    tw += wi
                    for j in range(m):
                        sc[c, j] += wi * V[i, j]
                nc[c] = tw
                for j in range(m):
                    if tw > 0.0:
                        parts[b, 5 + j] += sc[c, j] * sc[c, j] / tw
                for i in range(s, e):
                    r = np.sqrt(w[i])
                    for j in range(m):
                        V[i, j] = r * (V[i, j] - sc[c, j] / tw) if tw > 0.0 else 0.0


@nb.njit(parallel=True, cache=True)
def _nb_glm_rows_plain(family, init, ybar, y, X, f, off, beta, V, Wb, parts):
    """`_nb_glm_rows` for a model without fixed effects, where the rows are
    not grouped: one sweep of each row gives its linear predictor (beta and
    the offset), the family's terms, and V[i] = sqrt(w) v of v = [z, x],
    z = eta + u less the offset, w = W f. With `Wb` of one matrix per block,
    block b also forms its cross-products V_b'V_b in Wb[b] (a BLAS call on
    its own thread); with none (many columns) the caller forms V'V with
    one BLAS product on all threads.

    With `init` the linear predictor is the family's starting value; f, off
    and (with `init`) beta may be empty. The rows are cut into the fixed
    blocks of `parts`, so the sums do not depend on the number of threads:

        0 deviance   1 log-likelihood   2 f y   3 f   4 f log(y!) (Poisson)
        5 sum w   6.. sum w v, per column of v
    """
    nblk = parts.shape[0]
    n = len(y)
    k = X.shape[1]
    weighted = f.shape[0] > 0
    offset = off.shape[0] > 0
    cross = Wb.shape[0] > 0
    for b in nb.prange(nblk):
        for j in range(parts.shape[1]):
            parts[b, j] = 0.0
        lo = b * n // nblk
        hi = (b + 1) * n // nblk
        for i in range(lo, hi):
            yi = y[i]
            fi = f[i] if weighted else 1.0
            if init:
                eta = np.log(0.5 * (yi + ybar)) if family == POISSON else 0.0
            else:
                eta = 0.0
                for j in range(k):
                    eta += X[i, j] * beta[j]
                if offset:
                    eta += off[i]
            Wi, u, dev, ll, lg = _row_terms(family, yi, eta)
            parts[b, 0] += fi * dev
            parts[b, 1] += fi * ll
            parts[b, 2] += fi * yi
            parts[b, 3] += fi
            parts[b, 4] += fi * lg
            wi = Wi * fi
            z = eta + u
            if offset:
                z -= off[i]
            parts[b, 5] += wi
            parts[b, 6] += wi * z
            for j in range(k):
                parts[b, 7 + j] += wi * X[i, j]
            r = np.sqrt(wi)
            V[i, 0] = r * z
            for j in range(k):
                V[i, j + 1] = r * X[i, j]
        if cross:
            if hi > lo:
                Vb = V[lo:hi]
                Wb[b] = np.dot(Vb.T, Vb)
            else:
                Wb[b] = 0.0

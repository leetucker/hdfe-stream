"""numba kernels for `core.sensitivity`: a dense simplex for the small linear
programs of the conditional tests, and the tests themselves over a grid of
values of the target parameter.

The conditional test (Andrews, Roth and Pakes, 2023; Rambachan and Roth,
2023, section 3) solves, at each value theta of the target, the linear
program

    eta(theta) = min_{eta, delta} eta   s.t.   y(theta) - X delta <= eta * sd,

and its dual,

    max_gamma gamma' y(theta)   s.t.   gamma >= 0,  sd' gamma = 1,  X' gamma = 0.

The dual's feasible set does not depend on theta, only its objective does, so
the kernels solve the dual with a revised primal simplex that starts each
value of theta from the previous optimal basis: neighboring grid points
usually share their optimal vertex, and the warm start costs one pricing pass.
The same set is the feasible set of the programs that find the truncation
bounds of a degenerate solution, and of the least-favorable critical value's
simulated programs, which therefore start from a basis that is close too.

Every step follows the R package HonestDiD 0.2.8 (`arp-nuisance.R`,
`arp-nonuisance.R`): the same branches, tolerances and truncation bounds, so
that, at each grid point, the decision is the one the R code makes. The
truncated-normal quantile is left to the caller (scipy), which computes it for
all grid points at once.
"""

from __future__ import annotations

import numba as nb
import numpy as np

OPTIMAL, UNBOUNDED, ITERATION_LIMIT, INFEASIBLE, REDUNDANT = 0, 1, 2, 3, 4

ACCEPT, REJECT, QUANTILE = 0, 1, 2
ARP, LEAST_FAVORABLE, FIXED_LENGTH = 0, 1, 2

_MAX_ITER = 5000
_PIVOT_TOL = 1e-9


# The systems below have at most a few dozen rows. They are solved here rather than
# by LAPACK: OpenBLAS's LAPACK takes a process-wide lock for its work buffers on
# every call, even with one BLAS thread, and with several threads solving their
# own small programs the lock, not the arithmetic, set the time (eight threads
# were slower than one).

@nb.njit(cache=True, nogil=True, error_model="numpy")
def _solve(B, rhs, transpose):
    """B x = rhs (or B' x = rhs), by Gaussian elimination with partial pivoting.
    B is not changed."""
    r = B.shape[0]
    L = np.empty((r, r))
    for i in range(r):
        for j in range(r):
            L[i, j] = B[j, i] if transpose else B[i, j]
    x = rhs.copy()
    for col in range(r):
        piv = col
        big = abs(L[col, col])
        for i in range(col + 1, r):
            if abs(L[i, col]) > big:
                big = abs(L[i, col])
                piv = i
        if piv != col:
            for j in range(r):
                t = L[col, j]
                L[col, j] = L[piv, j]
                L[piv, j] = t
            t = x[col]
            x[col] = x[piv]
            x[piv] = t
        p = L[col, col]
        for i in range(col + 1, r):
            f = L[i, col] / p
            if f != 0.0:
                for j in range(col + 1, r):
                    L[i, j] -= f * L[col, j]
                x[i] -= f * x[col]
    for i in range(r - 1, -1, -1):
        s = x[i]
        for j in range(i + 1, r):
            s -= L[i, j] * x[j]
        x[i] = s / L[i, i]
    return x


@nb.njit(cache=True, nogil=True, error_model="numpy")
def _inverse(B):
    r = B.shape[0]
    out = np.empty((r, r))
    e = np.zeros(r)
    for j in range(r):
        e[:] = 0.0
        e[j] = 1.0
        col = _solve(B, e, False)
        for i in range(r):
            out[i, j] = col[i]
    return out


@nb.njit(cache=True, nogil=True, error_model="numpy")
def _singular_values(Mx):
    """The singular values of a small matrix, largest first, by one-sided
    Jacobi rotations of its columns (of its transpose's, when it is wide)."""
    m, n = Mx.shape
    if n > m:
        U = Mx.T.copy()
        m, n = n, m
    else:
        U = Mx.copy()
    eps = 2.220446049250313e-16
    for _ in range(80):
        rotated = False
        for p in range(n - 1):
            for q in range(p + 1, n):
                alpha = 0.0
                beta = 0.0
                gamma = 0.0
                for i in range(m):
                    alpha += U[i, p] * U[i, p]
                    beta += U[i, q] * U[i, q]
                    gamma += U[i, p] * U[i, q]
                if gamma == 0.0 or abs(gamma) <= eps * np.sqrt(alpha * beta):
                    continue
                rotated = True
                zeta = (beta - alpha) / (2.0 * gamma)
                t = (1.0 if zeta >= 0 else -1.0) / (abs(zeta) + np.sqrt(1.0 + zeta * zeta))
                c = 1.0 / np.sqrt(1.0 + t * t)
                s = c * t
                for i in range(m):
                    up = U[i, p]
                    uq = U[i, q]
                    U[i, p] = c * up - s * uq
                    U[i, q] = s * up + c * uq
        if not rotated:
            break
    sv = np.empty(n)
    for j in range(n):
        acc = 0.0
        for i in range(m):
            acc += U[i, j] * U[i, j]
        sv[j] = np.sqrt(acc)
    return np.sort(sv)[::-1]


@nb.njit(cache=True, nogil=True, error_model="numpy")
def _gather_basis(WT, basis, B):
    r = basis.shape[0]
    for i in range(r):
        col = WT[basis[i]]
        for k in range(r):
            B[k, i] = col[k]


@nb.njit(cache=True, nogil=True, error_model="numpy")
def simplex(WT, b, c, basis, max_iter):
    """max c'x subject to W x = b, x >= 0, where `WT` is W transposed (one row
    per variable), from the feasible basis `basis` (r column indices of W,
    updated in place to the optimal basis).

    Dantzig pricing, switching to Bland's rule after 20 degenerate pivots in a
    row so that the method cannot cycle. Returns (status, x_B): OPTIMAL,
    UNBOUNDED or ITERATION_LIMIT, and the basic variables' values."""
    n, r = WT.shape
    isbasic = np.zeros(n, np.bool_)
    for i in range(r):
        isbasic[basis[i]] = True
    cscale = 1.0
    for j in range(n):
        cscale = max(cscale, abs(c[j]))
    tol_d = 1e-10 * cscale
    B = np.empty((r, r))
    cB = np.empty(r)
    xB = np.zeros(r)
    degenerate = 0
    for _ in range(max_iter):
        _gather_basis(WT, basis, B)
        for i in range(r):
            cB[i] = c[basis[i]]
        xB = _solve(B, b, False)
        pi = _solve(B, cB, True)
        bland = degenerate > 20
        enter = -1
        best = tol_d
        for j in range(n):
            if isbasic[j]:
                continue
            row = WT[j]
            dj = c[j]
            for k in range(r):
                dj -= pi[k] * row[k]
            if dj > tol_d:
                if bland:
                    enter = j
                    break
                if dj > best:
                    best = dj
                    enter = j
        if enter < 0:
            return OPTIMAL, xB
        u = _solve(B, WT[enter], False)
        leave = -1
        tmin = np.inf
        for i in range(r):
            if u[i] > _PIVOT_TOL:
                t = max(xB[i], 0.0) / u[i]
                if t < tmin - 1e-13:
                    tmin = t
                    leave = i
                elif t <= tmin + 1e-13 and leave >= 0 and basis[i] < basis[leave]:
                    leave = i
        if leave < 0:
            return UNBOUNDED, xB
        degenerate = degenerate + 1 if tmin <= 1e-13 else 0
        isbasic[basis[leave]] = False
        basis[leave] = enter
        isbasic[enter] = True
    return ITERATION_LIMIT, xB


@nb.njit(cache=True, nogil=True, error_model="numpy")
def phase_one(WT, b):
    """A feasible basis of {x >= 0 : W x = b}, by the artificial-variable
    phase one, with the artificial variables pivoted out at the end. Returns
    (status, basis): OPTIMAL, INFEASIBLE, or REDUNDANT when W's rows are
    linearly dependent."""
    n, r = WT.shape
    W1T = np.zeros((n + r, r))
    b1 = np.empty(r)
    for k in range(r):
        sgn = -1.0 if b[k] < 0 else 1.0
        b1[k] = sgn * b[k]
        for j in range(n):
            W1T[j, k] = sgn * WT[j, k]
        W1T[n + k, k] = 1.0
    c1 = np.zeros(n + r)
    for k in range(r):
        c1[n + k] = -1.0
    basis = np.arange(n, n + r).astype(np.int64)
    status, xB = simplex(W1T, b1, c1, basis, 20 * _MAX_ITER)
    if status != OPTIMAL:
        return INFEASIBLE, basis
    infeas = 0.0
    for i in range(r):
        if basis[i] >= n:
            infeas += xB[i]
    if infeas > 1e-9 * max(1.0, np.abs(b1).max()):
        return INFEASIBLE, basis
    B = np.empty((r, r))
    for i in range(r):
        if basis[i] < n:
            continue
        _gather_basis(W1T, basis, B)
        e = np.zeros(r)
        e[i] = 1.0
        rho = _solve(B, e, True)
        best = 1e-9
        enter = -1
        for j in range(n):
            used = False
            for q in range(r):
                if basis[q] == j:
                    used = True
                    break
            if used:
                continue
            v = 0.0
            for k in range(r):
                v += rho[k] * W1T[j, k]
            if abs(v) > best:
                best = abs(v)
                enter = j
        if enter < 0:
            return REDUNDANT, basis
        basis[i] = enter
    return OPTIMAL, basis


@nb.njit(cache=True, nogil=True, error_model="numpy")
def _dual_value(WT, b, c, basis):
    """Solve max c'gamma over the dual set from `basis`; returns (status,
    gamma (all variables), value)."""
    n, r = WT.shape
    status, xB = simplex(WT, b, c, basis, _MAX_ITER)
    gamma = np.zeros(n)
    val = 0.0
    for i in range(r):
        gamma[basis[i]] = xB[i]
        val += c[basis[i]] * xB[i]
    return status, gamma, val


@nb.njit(cache=True, nogil=True, error_model="numpy")
def lf_etas(WT, b, draws, scale, basis0):
    """eta of each row of `draws` (one simulated y each): the optimal value of
    the dual program, NaN when the program fails (R drops those draws)."""
    nd = draws.shape[0]
    out = np.empty(nd)
    basis = basis0.copy()
    c = np.empty(draws.shape[1])
    for i in range(nd):
        for j in range(draws.shape[1]):
            c[j] = draws[i, j] / scale
        status, gamma, val = _dual_value(WT, b, c, basis)
        if status != OPTIMAL:
            out[i] = np.nan
            basis[:] = basis0
        else:
            out[i] = val
    return out


@nb.njit(cache=True, nogil=True, error_model="numpy")
def _roundeps(x):
    eps = 2.220446049250313e-16 ** 0.75
    return 0.0 if abs(x) < eps else x


@nb.njit(cache=True, nogil=True, error_model="numpy")
def _check_solution(c, WT, b, s_T, bvec, scale, basis):
    """R's `.check_if_solution_helper`: is c the maximum over the dual set of
    (s_T + bvec c)' gamma? Returns (honest, gamma in R's units, status)."""
    m = s_T.shape[0]
    f = np.empty(m)
    for j in range(m):
        f[j] = (s_T[j] + bvec[j] * c) / scale
    status, gamma, val = _dual_value(WT, b, f, basis)
    for j in range(m):
        gamma[j] /= scale
    if status != OPTIMAL:
        return False, gamma, status
    return abs(c - val) <= 1e-6, gamma, status


@nb.njit(cache=True, nogil=True, error_model="numpy")
def _shortcut(gamma, s_T, bvec):
    gs = 0.0
    gb = 0.0
    for j in range(gamma.shape[0]):
        gs += gamma[j] * s_T[j]
        gb += gamma[j] * bvec[j]
    return _roundeps(gs) / (1.0 - gb)


@nb.njit(cache=True, nogil=True, error_model="numpy")
def vlo_vup_dual(eta, s_T, gamma_t, sig, WT, b, scale, basis0):
    """R's `.vlo_vup_dual_fn` (ARP 2023, appendix D, algorithm 1): the
    truncation bounds of a degenerate solution, from the interval of c over
    which c is the maximum of (s_T + sig gamma c / gamma'sig gamma)' gamma."""
    tol_c = 1e-6
    m = s_T.shape[0]
    sg = sig @ gamma_t
    v = 0.0
    for j in range(m):
        v += gamma_t[j] * sg[j]
    sigma_B = np.sqrt(v)
    bvec = sg / v
    low_initial = min(-100.0, eta - 20.0 * sigma_B)
    high_initial = max(100.0, eta + 20.0 * sigma_B)
    switchiters = 10
    maxiters = 10000
    basis = basis0.copy()
    ok, g, st = _check_solution(eta, WT, b, s_T, bvec, scale, basis)
    if not ok:
        return eta, np.inf
    # vup
    ok, g, st = _check_solution(high_initial, WT, b, s_T, bvec, scale, basis)
    if ok:
        vup = np.inf
    else:
        dif = 0.0
        iters = 1
        mid = _shortcut(g, s_T, bvec)
        while True:
            if not np.isfinite(mid):
                dif = tol_c + 1.0
                mid = high_initial
                break
            ok, g, st = _check_solution(mid, WT, b, s_T, bvec, scale, basis)
            if ok or iters >= maxiters:
                break
            iters += 1
            if iters >= switchiters:
                dif = tol_c + 1.0
                break
            mid = _shortcut(g, s_T, bvec)
        low = eta
        high = mid
        while dif > tol_c and iters < maxiters:
            iters += 1
            mid = (high + low) / 2.0
            ok, g, st = _check_solution(mid, WT, b, s_T, bvec, scale, basis)
            if ok:
                low = mid
            else:
                high = mid
            dif = high - low
        vup = mid
    # vlo
    basis[:] = basis0
    ok, g, st = _check_solution(low_initial, WT, b, s_T, bvec, scale, basis)
    if ok:
        vlo = -np.inf
    else:
        dif = 0.0
        iters = 1
        mid = _shortcut(g, s_T, bvec)
        while True:
            if not np.isfinite(mid):
                dif = tol_c + 1.0
                mid = low_initial
                break
            ok, g, st = _check_solution(mid, WT, b, s_T, bvec, scale, basis)
            if ok or iters >= maxiters:
                break
            iters += 1
            if iters >= switchiters:
                dif = tol_c + 1.0
                break
            mid = _shortcut(g, s_T, bvec)
        low = mid
        high = eta
        while dif > tol_c and iters < maxiters:
            mid = (low + high) / 2.0
            iters += 1
            ok, g, st = _check_solution(mid, WT, b, s_T, bvec, scale, basis)
            if ok:
                high = mid
            else:
                low = mid
            dif = high - low
        vlo = mid
    return vlo, vup


@nb.njit(cache=True, nogil=True, error_model="numpy")
def _flci_bounds(vbar, dbar0, dbar1, S, cvec):
    """R's `.FLCI_computeVloVup`."""
    vS = 0.0
    vc = 0.0
    for j in range(vbar.shape[0]):
        vS += vbar[j] * S[j]
        vc += vbar[j] * cvec[j]
    vlo = -np.inf
    vup = np.inf
    # rows (vbar, dbar0) and (-vbar, dbar1)
    if vc != 0.0:
        r0 = (dbar0 - vS) / vc
        r1 = (dbar1 + vS) / (-vc)
        if vc < 0:
            vlo = max(vlo, r0)
            vup = min(vup, r1)
        else:
            vup = min(vup, r0)
            vlo = max(vlo, r1)
    return vlo, vup


@nb.njit(cache=True, nogil=True, error_model="numpy")
def _rank(Mx):
    """Matrix::rankMatrix's default: singular values above
    max(dim) * eps * the largest."""
    if Mx.shape[0] == 0 or Mx.shape[1] == 0:
        return 0
    s = _singular_values(Mx)
    tol = max(Mx.shape[0], Mx.shape[1]) * 2.220446049250313e-16 * s[0]
    k = 0
    for v in s:
        if v > tol:
            k += 1
    return k


@nb.njit(cache=True, nogil=True, error_model="numpy")
def arp_path(Y0, a, rows, X, SY, thetas, hybrid, lf_cv, vbar, half, vbar_d, vbar_a,
             WT, b, scale, basis0):
    """The nuisance-parameter conditional test (R's `.lp_conditional_test_fn`)
    at each theta, y(theta) = Y0 - a theta.

    `rows` are the moments the test uses (R's rowsForARP), `X` the nuisance
    columns of all moments, `SY` the moments' covariance; the fixed-length
    hybrid's first stage uses every moment. `WT`, `b`, `scale` and `basis0`
    are the dual set of the rows' program, its first row divided by `scale`,
    and a feasible basis of it.

    Returns, per theta, (kind, maxstat, zlo, zup): kind ACCEPT or REJECT when
    the test decides without a truncated-normal quantile; QUANTILE when it
    rejects if maxstat > max(0, the quantile on [zlo, zup])."""
    G = thetas.shape[0]
    mA = rows.shape[0]
    mf = Y0.shape[0]
    k = X.shape[1]
    kind = np.zeros(G, np.int8)
    maxstat = np.zeros(G)
    zlo = np.zeros(G)
    zup = np.zeros(G)
    XA = np.empty((mA, k))
    SA = np.empty((mA, mA))
    sdA = np.empty(mA)
    for i in range(mA):
        for q in range(k):
            XA[i, q] = X[rows[i], q]
        for j in range(mA):
            SA[i, j] = SY[rows[i], rows[j]]
        sdA[i] = np.sqrt(SA[i, i])
    basis = basis0.copy()
    y = np.empty(mf)
    yA = np.empty(mA)
    c = np.empty(mA)
    eps = 2.220446049250313e-16
    MB = np.empty((k + 1, k + 1))
    Bidx = np.empty(k + 1, np.int64)
    # a variance this small is rounding error of zero (R: below machine epsilon,
    # in absolute terms; here relative to the moments' scale, for any units)
    dmax = 0.0
    for i in range(mA):
        dmax = max(dmax, SA[i, i])
    zero_var = max(eps, 1e3 * eps * dmax)
    for g in range(G):
        th = thetas[g]
        for i in range(mf):
            y[i] = Y0[i] - a[i] * th
        for i in range(mA):
            yA[i] = y[rows[i]]
            c[i] = yA[i] / scale
        status, gs, eta = _dual_value(WT, b, c, basis)
        if status != OPTIMAL:
            # R: "LP for eta did not converge properly. Not rejecting"
            kind[g] = ACCEPT
            basis[:] = basis0
            continue
        lam = gs / scale
        if hybrid == LEAST_FAVORABLE:
            if eta > lf_cv:
                kind[g] = REJECT
                continue
        elif hybrid == FIXED_LENGTH:
            dbar0 = half - vbar_d + (1.0 - vbar_a) * th
            dbar1 = half + vbar_d - (1.0 - vbar_a) * th
            vy = 0.0
            for i in range(mf):
                vy += vbar[i] * y[i]
            if max(vy - dbar0, -vy - dbar1) > 0:
                kind[g] = REJECT
                continue
        nB = 0
        for i in range(mA):
            if lam[i] > 1e-6:
                nB += 1
        degenerate = nB != k + 1
        XB = np.empty((nB, k))
        q = 0
        for i in range(mA):
            if lam[i] > 1e-6:
                XB[q] = XA[i]
                q += 1
        Xdim = min(nB, k)
        full_rank = Xdim > 0 and _rank(XB) == Xdim
        if not degenerate and full_rank:
            q = 0
            for i in range(mA):
                if lam[i] > 1e-6:
                    Bidx[q] = i
                    MB[q, 0] = sdA[i]
                    for p in range(k):
                        MB[q, p + 1] = XA[i, p]
                    q += 1
            # R stops with an error when [sd_B, X_B] is singular although X_B
            # has full rank; the dual computation still applies
            if _rank(MB) < k + 1:
                full_rank = False
        if degenerate or not full_rank:
            sl = SA @ lam
            s2 = 0.0
            for i in range(mA):
                s2 += lam[i] * sl[i]
            if abs(s2) <= zero_var:
                kind[g] = REJECT if eta > 0 else ACCEPT
                continue
            s_T = yA - sl * ((lam @ yA) / s2)
            vlo, vup = vlo_vup_dual(eta, s_T, lam, SA, WT, b, scale, basis)
            gamma = lam
            sB = np.sqrt(s2)
        else:
            MBinv = _inverse(MB)
            gamma = np.zeros(mA)
            for q in range(k + 1):
                gamma[Bidx[q]] = MBinv[0, q]
            sv = SA @ gamma
            s2 = 0.0
            for i in range(mA):
                s2 += gamma[i] * sv[i]
            if s2 <= zero_var:
                # the binding moments' combination has no variance (a moment and its
                # negation both bind): R's rule for this case in its dual branch;
                # its closed form stops with an error here
                kind[g] = REJECT if eta > 0 else ACCEPT
                continue
            sB = np.sqrt(s2)
            gy = gamma @ yA
            vlo = -np.inf
            vup = np.inf
            # rho is zero for a moment that duplicates a binding one (relative
            # magnitudes at M = 0 with a sign or monotonicity restriction); in
            # floating point it is rounding error, whose sign must not make the
            # moment a bound (R lets it, and then accepts values it should reject)
            svmax = 0.0
            for i in range(mA):
                svmax = max(svmax, abs(sv[i]))
            rho_tol = 1e-10 * svmax / s2
            inB = np.zeros(mA, np.bool_)
            for q in range(k + 1):
                inB[Bidx[q]] = True
            for i in range(mA):
                if inB[i]:
                    continue
                # row i of Gamma_B: [sd_i, X_i] MB^{-1} S_B - e_i
                coef = np.zeros(k + 1)
                for q in range(k + 1):
                    acc = sdA[i] * MBinv[0, q]
                    for p in range(k):
                        acc += XA[i, p] * MBinv[p + 1, q]
                    coef[q] = acc
                gsv = -sv[i]
                gyv = -yA[i]
                for q in range(k + 1):
                    gsv += coef[q] * sv[Bidx[q]]
                    gyv += coef[q] * yA[Bidx[q]]
                rho = gsv / s2
                if abs(rho) <= rho_tol:
                    continue
                val = -gyv / rho + gy
                if rho > 0:
                    vlo = max(vlo, val)
                else:
                    vup = min(vup, val)
        ms = eta / sB
        if hybrid == LEAST_FAVORABLE:
            lo = vlo / sB
            up = min(vup, lf_cv) / sB
        elif hybrid == FIXED_LENGTH:
            gfull = np.zeros(mf)
            for i in range(mA):
                gfull[rows[i]] = gamma[i]
            sgf = SY @ gfull
            den = 0.0
            gy_full = 0.0
            for i in range(mf):
                den += gfull[i] * sgf[i]
                gy_full += gfull[i] * y[i]
            cvec = sgf / den
            S = y - cvec * gy_full
            dbar0 = half - vbar_d + (1.0 - vbar_a) * th
            dbar1 = half + vbar_d - (1.0 - vbar_a) * th
            flo, fup = _flci_bounds(vbar, dbar0, dbar1, S, cvec)
            lo = max(vlo, flo) / sB
            up = min(vup, fup) / sB
        else:
            lo = vlo / sB
            up = vup / sB
        # The observed statistic lies in its truncation interval by construction.
        # R does not reject when it falls outside; within rounding of a bound (a
        # theta on the boundary of the hybrid's first stage, where the bound is
        # the statistic itself) it is on the bound, and is tested there.
        tol = 1e-9 * max(1.0, abs(ms))
        if not (lo - tol <= ms and ms <= up + tol):
            kind[g] = ACCEPT
            continue
        if ms > up:
            ms = up
        if ms < lo:
            ms = lo
        kind[g] = QUANTILE
        maxstat[g] = ms
        zlo[g] = lo
        zup[g] = up
    return kind, maxstat, zlo, zup


@nb.njit(cache=True, nogil=True, error_model="numpy")
def plain_path(beta, Sigma, A, d, post_index, thetas, hybrid, lf_cv, fs_l, fs_h):
    """The conditional test without nuisance parameters (one post-treatment
    period; R's `.testInIdentifiedSet` and its two hybrids) at each theta,
    y(theta) = beta - e theta with e the post-period basis vector.

    Returns, per theta, (kind, stat, mu, sd, zlo, zup): QUANTILE means reject
    when stat > max(0, mu + sd * the quantile on [zlo, zup])."""
    G = thetas.shape[0]
    m, n = A.shape
    kind = np.zeros(G, np.int8)
    stat = np.zeros(G)
    mus = np.zeros(G)
    sds = np.zeros(G)
    zlo = np.zeros(G)
    zup = np.zeros(G)
    AS = A @ Sigma
    st = np.empty(m)
    for i in range(m):
        acc = 0.0
        for j in range(n):
            acc += AS[i, j] * A[i, j]
        st[i] = np.sqrt(acc)
    At = np.empty((m, n))
    dt = np.empty(m)
    for i in range(m):
        for j in range(n):
            At[i, j] = A[i, j] / st[i]
        dt[i] = d[i] / st[i]
    extra = 2 if hybrid == FIXED_LENGTH else 0
    Abar = np.empty((m + extra, n))
    dbar = np.empty(m + extra)
    y = np.empty(n)
    for g in range(G):
        for j in range(n):
            y[j] = beta[j]
        y[post_index] -= thetas[g]
        mom = At @ y - dt
        loc = 0
        for i in range(1, m):
            if mom[i] > mom[loc]:
                loc = i
        if hybrid == LEAST_FAVORABLE:
            if mom[loc] > lf_cv:
                kind[g] = REJECT
                continue
        elif hybrid == FIXED_LENGTH:
            ly = fs_l @ y
            if max(ly - fs_h, -ly - fs_h) > 0:
                kind[g] = REJECT
                continue
        gam = At[loc].copy()
        for i in range(m):
            for j in range(n):
                Abar[i, j] = At[i, j] - gam[j]
            dbar[i] = dt[i] - dt[loc]
        if hybrid == FIXED_LENGTH:
            for j in range(n):
                Abar[m, j] = fs_l[j]
                Abar[m + 1, j] = -fs_l[j]
            dbar[m] = fs_h
            dbar[m + 1] = fs_h
        sg = Sigma @ gam
        s2 = gam @ sg
        sbar = np.sqrt(s2)
        cvec = sg / s2
        gy = gam @ y
        z = y - cvec * gy
        Ac = Abar @ cvec
        Az = Abar @ z
        vlo = -np.inf
        vup = np.inf
        for i in range(m + extra):
            if Ac[i] < 0:
                vlo = max(vlo, (dbar[i] - Az[i]) / Ac[i])
            elif Ac[i] > 0:
                vup = min(vup, (dbar[i] - Az[i]) / Ac[i])
        mu = dt[loc]
        kind[g] = QUANTILE
        stat[g] = gy
        mus[g] = mu
        sds[g] = sbar
        zlo[g] = (vlo - mu) / sbar
        zup[g] = (vup - mu) / sbar
    return kind, stat, mus, sds, zlo, zup

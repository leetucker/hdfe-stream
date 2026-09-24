"""numba kernels for varying slopes on the streamed dimension (FEIS).

Each kernel here mirrors one in `kernels_base`, with the group mean replaced
by the fit of a small within-group weighted regression.
"""

from __future__ import annotations

import numpy as np
import numba as nb



# --------------------------------------------------------------------------
# numba kernels for varying slopes on fe[0] (FEIS)
#
# With slopes, projecting out fe[0] is no longer "subtract the weighted group
# mean" but "subtract the fitted values of a small weighted regression on
# T = [1, slope variables] within the group". Per identifying group g we keep
# Ainv[g] = pinv(sum w t t') (p x p) and C[g] = sum w t v' (p x m); per cell
# st[c] = sum w t (p). Everything stays local to the group.
# --------------------------------------------------------------------------

@nb.njit(cache=True)
def _nb_rhs_sl(starts, codes, sums, st, Ainv, C, offs, b):
    """b += D_o' W M_0 V with the slope projection."""
    D, m, p = codes.shape[1], sums.shape[1], st.shape[1]
    coef = np.empty((p, m))
    for gi in range(len(starts) - 1):
        s, e = starts[gi], starts[gi + 1]
        for r in range(p):
            for j in range(m):
                acc = 0.0
                for r2 in range(p):
                    acc += Ainv[gi, r, r2] * C[gi, r2, j]
                coef[r, j] = acc
        for i in range(s, e):
            for d in range(D):
                a = offs[d] + codes[i, d]
                for j in range(m):
                    v = sums[i, j]
                    for r in range(p):
                        v -= st[i, r] * coef[r, j]
                    b[a, j] += v


@nb.njit(cache=True)
def _nb_diag_sl(starts, codes, n, st, Ainv, offs, diag):
    """Exact diagonal of S with the slope projection."""
    D, p = codes.shape[1], st.shape[1]
    for gi in range(len(starts) - 1):
        s, e = starts[gi], starts[gi + 1]
        lev = np.empty((e - s) * D, np.int64)
        u = np.zeros(((e - s) * D, p))
        q = 0
        for i in range(s, e):
            for d in range(D):
                a = offs[d] + codes[i, d]
                diag[a] += n[i]
                r = 0
                while r < q and lev[r] != a:
                    r += 1
                if r == q:
                    lev[q] = a
                    q += 1
                for c in range(p):
                    u[r, c] += st[i, c]
        for r in range(q):
            quad = 0.0
            for c1 in range(p):
                for c2 in range(p):
                    quad += u[r, c1] * Ainv[gi, c1, c2] * u[r, c2]
            diag[lev[r]] -= quad


@nb.njit(parallel=True, cache=True)
def _nb_emit_triples_sl(starts, codes, n, st, Ainv, offs, doffs, g0, g1, toff, R, C, V, dense):
    """As _nb_emit_triples, with each group's correction  -U Ainv U'  where U
    holds the per-level sums of w t (q x p)."""
    D, p = codes.shape[1], st.shape[1]
    nt = dense.shape[0]
    G = g1 - g0
    for t in nb.prange(nt):
        dn = dense[t]
        for gi in range(g0 + t * G // nt, g0 + (t + 1) * G // nt):
            s, e = starts[gi], starts[gi + 1]
            pp = toff[gi - g0]
            lev = np.empty((e - s) * D, np.int64)
            dlev = np.empty((e - s) * D, np.int64)
            u = np.zeros(((e - s) * D, p))
            q = 0
            for i in range(s, e):
                for d in range(D):
                    a = offs[d] + codes[i, d]
                    for d2 in range(D):
                        if doffs[d] >= 0 and doffs[d2] >= 0:
                            dn[doffs[d] + codes[i, d], doffs[d2] + codes[i, d2]] += n[i]
                        else:
                            R[pp] = a
                            C[pp] = offs[d2] + codes[i, d2]
                            V[pp] = n[i]
                            pp += 1
                    r = 0
                    while r < q and lev[r] != a:
                        r += 1
                    if r == q:
                        lev[q] = a
                        dlev[q] = doffs[d] + codes[i, d] if doffs[d] >= 0 else -1
                        q += 1
                    for c in range(p):
                        u[r, c] += st[i, c]
            for r1 in range(q):
                for r2 in range(q):
                    v = 0.0
                    for c1 in range(p):
                        for c2 in range(p):
                            v -= u[r1, c1] * Ainv[gi, c1, c2] * u[r2, c2]
                    if dlev[r1] >= 0 and dlev[r2] >= 0:
                        dn[dlev[r1], dlev[r2]] += v
                    else:
                        R[pp] = lev[r1]
                        C[pp] = lev[r2]
                        V[pp] = v
                        pp += 1


@nb.njit(parallel=True, cache=True)
def _nb_stream_matvec_sl(P, starts, codes, n, st, Ainv, offs, acc):
    """(D_o' W M_0 D_o) P with the slope projection, streamed."""
    nt, k, D, p = acc.shape[0], P.shape[1], codes.shape[1], st.shape[1]
    G = len(starts) - 1
    for t in nb.prange(nt):
        out = acc[t]
        out[:] = 0.0
        Cg = np.empty((p, k))
        Mg = np.empty((p, k))
        tv = np.empty(k)
        for gi in range(t * G // nt, (t + 1) * G // nt):
            s, e = starts[gi], starts[gi + 1]
            Cg[:] = 0.0
            for i in range(s, e):
                for j in range(k):
                    x = 0.0
                    for d in range(D):
                        x += P[offs[d] + codes[i, d], j]
                    for r in range(p):
                        Cg[r, j] += st[i, r] * x
            for r in range(p):
                for j in range(k):
                    x = 0.0
                    for r2 in range(p):
                        x += Ainv[gi, r, r2] * Cg[r2, j]
                    Mg[r, j] = x
            for i in range(s, e):
                for j in range(k):
                    x = 0.0
                    for d in range(D):
                        x += P[offs[d] + codes[i, d], j]
                    tv[j] = x
                for j in range(k):
                    v = n[i] * tv[j]
                    for r in range(p):
                        v -= st[i, r] * Mg[r, j]
                    for d in range(D):
                        out[offs[d] + codes[i, d], j] += v


@nb.njit(cache=True)
def _group_pinv(T, w, s, e, A, Ainv):
    """A = sum w t t' over rows s..e; Ainv = its pseudo-inverse."""
    p = T.shape[1]
    A[:] = 0.0
    for i in range(s, e):
        for r1 in range(p):
            for r2 in range(p):
                A[r1, r2] += w[i] * T[i, r1] * T[i, r2]
    Ainv[:] = np.linalg.pinv(A, rcond=1e-12)


@nb.njit(parallel=True, cache=True)
def _nb_assemble_rows_sl(starts, codes, offs, w, T, V, Gamma, acc):
    """Row assembly of V' M_D V with the slope projection."""
    nt, D, m, p = acc.shape[0], codes.shape[1], V.shape[1], T.shape[1]
    G = len(starts) - 1
    for t in nb.prange(nt):
        A = np.empty((p, p))
        Ainv = np.empty((p, p))
        c = np.empty((p, m))
        coef = np.empty((p, m))
        q = np.empty(m)
        for gi in range(t * G // nt, (t + 1) * G // nt):
            s, e = starts[gi], starts[gi + 1]
            _group_pinv(T, w, s, e, A, Ainv)
            c[:] = 0.0
            for i in range(s, e):
                for j in range(m):
                    v = V[i, j]
                    for d in range(D):
                        v -= Gamma[offs[d] + codes[i, d], j]
                    for r in range(p):
                        c[r, j] += w[i] * T[i, r] * v
            for r in range(p):
                for j in range(m):
                    x = 0.0
                    for r2 in range(p):
                        x += Ainv[r, r2] * c[r2, j]
                    coef[r, j] = x
            for i in range(s, e):
                for j in range(m):
                    v = V[i, j]
                    for d in range(D):
                        v -= Gamma[offs[d] + codes[i, d], j]
                    for r in range(p):
                        v -= T[i, r] * coef[r, j]
                    q[j] = v
                for j1 in range(m):
                    for j2 in range(m):
                        acc[t, j1, j2] += w[i] * q[j1] * q[j2]


@nb.njit(parallel=True, cache=True)
def _nb_pass2_sl(starts, codes, offs, w, T, y, X, beta, gam_y, gam_y0, Z, gam_z, Pi, yc,
                 fweights, g_coef, e_out, h_out, acc_s, acc_B, acc_hc, acc_g):
    """_nb_pass2 with the slope projection; g_coef[g] receives the group's
    intercept and slopes (in terms of the centered slope variables)."""
    nt, D, k, q, p = acc_B.shape[0], codes.shape[1], X.shape[1], Z.shape[1], T.shape[1]
    G = len(starts) - 1
    for t in nb.prange(nt):
        A = np.empty((p, p))
        Ainv = np.empty((p, p))
        cu = np.empty(p)
        cy = np.empty(p)
        cz = np.empty((p, q))
        bu = np.empty(p)
        by = np.empty(p)
        bz = np.empty((p, q))
        zt = np.empty(q)
        h = np.empty(k)
        sg = np.empty(k)
        for gi in range(t * G // nt, (t + 1) * G // nt):
            s, e = starts[gi], starts[gi + 1]
            _group_pinv(T, w, s, e, A, Ainv)
            cu[:] = 0.0
            cy[:] = 0.0
            cz[:] = 0.0
            for i in range(s, e):
                wi = w[i]
                u = y[i]
                yy = y[i]
                for j in range(k):
                    u -= X[i, j] * beta[j]
                for d in range(D):
                    a = offs[d] + codes[i, d]
                    u -= gam_y[a]
                    yy -= gam_y0[a]
                e_out[i] = u
                for r in range(p):
                    cu[r] += wi * T[i, r] * u
                    cy[r] += wi * T[i, r] * yy
                for j in range(q):
                    v = Z[i, j]
                    for d in range(D):
                        v -= gam_z[offs[d] + codes[i, d], j]
                    for r in range(p):
                        cz[r, j] += wi * T[i, r] * v
            for r in range(p):
                x1 = 0.0
                x2 = 0.0
                for r2 in range(p):
                    x1 += Ainv[r, r2] * cu[r2]
                    x2 += Ainv[r, r2] * cy[r2]
                bu[r] = x1
                by[r] = x2
                g_coef[gi, r] = x1
                for j in range(q):
                    x = 0.0
                    for r2 in range(p):
                        x += Ainv[r, r2] * cz[r2, j]
                    bz[r, j] = x
            sg[:] = 0.0
            for i in range(s, e):
                wi = w[i]
                yy = y[i]
                for d in range(D):
                    yy -= gam_y0[offs[d] + codes[i, d]]
                ei = e_out[i]
                for r in range(p):
                    yy -= T[i, r] * by[r]
                    ei -= T[i, r] * bu[r]
                e_out[i] = ei
                for j in range(q):
                    v = Z[i, j]
                    for d in range(D):
                        v -= gam_z[offs[d] + codes[i, d], j]
                    for r in range(p):
                        v -= T[i, r] * bz[r, j]
                    zt[j] = v
                for c in range(k):
                    hc = 0.0
                    for j in range(q):
                        hc += zt[j] * Pi[j, c]
                    h[c] = hc
                    h_out[i, c] = hc
                acc_s[t, 0] += wi * ei * ei
                acc_s[t, 1] += wi * yy * yy
                acc_s[t, 2] += wi * (y[i] - yc)
                acc_s[t, 3] += wi * (y[i] - yc) * (y[i] - yc)
                hw = wi if fweights else wi * wi
                for c1 in range(k):
                    sg[c1] += wi * h[c1] * ei
                    for c2 in range(k):
                        acc_B[t, c1, c2] += wi * h[c1] * h[c2]
                        acc_hc[t, c1, c2] += hw * ei * ei * h[c1] * h[c2]
            for c1 in range(k):
                for c2 in range(k):
                    acc_g[t, c1, c2] += sg[c1] * sg[c2]

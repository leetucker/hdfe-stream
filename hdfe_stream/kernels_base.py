"""numba kernels for the standard (no varying slopes) path."""

from __future__ import annotations

import numpy as np
import numba as nb



# --------------------------------------------------------------------------
# numba kernels
#
# Identifying cells live in flat arrays, sorted by fe[0] group:
#   starts (G+1,) int64 group boundaries      codes (M, D) uint32 level codes
#   n      (M,)   float64 cell counts         sums  (M, m) float64 centered sums
# `offs[d]` is the start of dimension d's block in the stacked level vector.
# Parallel kernels split the groups into `nt` contiguous blocks, one per
# thread; anything scattered into level-sized vectors uses per-thread
# accumulators so threads never write to the same memory.
# --------------------------------------------------------------------------

@nb.njit(cache=True)
def _nb_rhs(starts, codes, n, sums, offs, b):
    """b += D_o' M_0 V for the given columns of V (cell sums)."""
    D, m = codes.shape[1], sums.shape[1]
    vbar = np.empty(m)
    for gi in range(len(starts) - 1):
        s, e = starts[gi], starts[gi + 1]
        Ng = 0.0
        vbar[:] = 0.0
        for i in range(s, e):
            Ng += n[i]
            for j in range(m):
                vbar[j] += sums[i, j]
        for j in range(m):
            vbar[j] /= Ng
        for i in range(s, e):
            for d in range(D):
                a = offs[d] + codes[i, d]
                for j in range(m):
                    b[a, j] += sums[i, j] - n[i] * vbar[j]


@nb.njit(cache=True)
def _nb_diag(starts, codes, n, offs, diag):
    """Exact diagonal of S = D_o' M_0 D_o (Jacobi preconditioner)."""
    D = codes.shape[1]
    for gi in range(len(starts) - 1):
        s, e = starts[gi], starts[gi + 1]
        Ng = 0.0
        lev = np.empty((e - s) * D, np.int64)
        u = np.empty((e - s) * D)
        q = 0
        for i in range(s, e):
            Ng += n[i]
            for d in range(D):
                a = offs[d] + codes[i, d]
                diag[a] += n[i]
                r = 0
                while r < q and lev[r] != a:
                    r += 1
                if r == q:
                    lev[q] = a
                    u[q] = 0.0
                    q += 1
                u[r] += n[i]
        for r in range(q):
            diag[lev[r]] -= u[r] * u[r] / Ng


@nb.njit(parallel=True, cache=True)
def _nb_assemble_rows(starts, codes, offs, w, V, Gamma, acc):
    """acc[t] += sum over rows of w v~ v~', with v~ = v - sum_d Gamma_d[level]
    minus its weighted fe[0]-group mean: the fully residualized
    cross-products."""
    nt, D, m = acc.shape[0], codes.shape[1], V.shape[1]
    G = len(starts) - 1
    for t in nb.prange(nt):
        q = np.empty(m)
        qg = np.empty(m)
        for gi in range(t * G // nt, (t + 1) * G // nt):
            s, e = starts[gi], starts[gi + 1]
            qg[:] = 0.0
            Wg = 0.0
            for i in range(s, e):
                Wg += w[i]
                for j in range(m):
                    v = V[i, j]
                    for d in range(D):
                        v -= Gamma[offs[d] + codes[i, d], j]
                    qg[j] += w[i] * v
            for j in range(m):
                qg[j] /= Wg
            for i in range(s, e):
                for j in range(m):
                    v = V[i, j]
                    for d in range(D):
                        v -= Gamma[offs[d] + codes[i, d], j]
                    q[j] = v - qg[j]
                for j1 in range(m):
                    for j2 in range(m):
                        acc[t, j1, j2] += w[i] * q[j1] * q[j2]


@nb.njit(cache=True)
def _uf_find(parent, x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


@nb.njit(cache=True)
def _nb_components(starts, codes, n_levels):
    """Union-find over the first non-streamed dimension: levels that share
    an fe[0] group are connected. Returns a root label per level."""
    parent = np.arange(n_levels)
    for gi in range(len(starts) - 1):
        s, e = starts[gi], starts[gi + 1]
        r0 = _uf_find(parent, codes[s, 0])
        for i in range(s + 1, e):
            r = _uf_find(parent, codes[i, 0])
            if r != r0:
                if r < r0:
                    parent[r0] = r
                    r0 = r
                else:
                    parent[r] = r0
    for x in range(n_levels):
        parent[x] = _uf_find(parent, x)
    return parent


@nb.njit(cache=True)
def _nb_group_levels(s, e, codes, offs, doffs, lev, dlev, u, n):
    """Distinct levels touched by one group, with their observation counts.
    dlev[r] is the level's index in the dense block (-1 if not dense).
    Returns (number of distinct levels, number of those that are dense)."""
    D = codes.shape[1]
    q = 0
    qd = 0
    for i in range(s, e):
        for d in range(D):
            a = offs[d] + codes[i, d]
            r = 0
            while r < q and lev[r] != a:
                r += 1
            if r == q:
                lev[q] = a
                dlev[q] = doffs[d] + codes[i, d] if doffs[d] >= 0 else -1
                if dlev[q] >= 0:
                    qd += 1
                u[q] = 0.0
                q += 1
            u[r] += n[i]
    return q, qd


@nb.njit(parallel=True, cache=True)
def _nb_count_triples(starts, codes, n, offs, doffs, g0, g1, out):
    """Number of COO triples each group contributes to the explicit S (pairs
    where both levels are dense go to the dense block instead)."""
    D = codes.shape[1]
    Dd = 0
    for d in range(D):
        if doffs[d] >= 0:
            Dd += 1
    for gi in nb.prange(g0, g1):
        s, e = starts[gi], starts[gi + 1]
        lev = np.empty((e - s) * D, np.int64)
        dlev = np.empty((e - s) * D, np.int64)
        u = np.empty((e - s) * D)
        q, qd = _nb_group_levels(s, e, codes, offs, doffs, lev, dlev, u, n)
        out[gi - g0] = (e - s) * (D * D - Dd * Dd) + q * q - qd * qd


@nb.njit(parallel=True, cache=True)
def _nb_emit_triples(starts, codes, n, offs, doffs, g0, g1, toff, R, C, V, dense):
    """Each group g adds  sum_cells n a a'  -  u u' / N_g  to S, where a is the
    cell's level indicator and u the group's level counts. Group g writes its
    sparse entries to its own slice [toff[g], toff[g+1]); entries between two
    dense (small-dimension) levels go to the per-thread dense[t] block."""
    D = codes.shape[1]
    nt = dense.shape[0]
    G = g1 - g0
    for t in nb.prange(nt):
        dn = dense[t]
        for gi in range(g0 + t * G // nt, g0 + (t + 1) * G // nt):
            s, e = starts[gi], starts[gi + 1]
            p = toff[gi - g0]
            lev = np.empty((e - s) * D, np.int64)
            dlev = np.empty((e - s) * D, np.int64)
            u = np.empty((e - s) * D)
            q, qd = _nb_group_levels(s, e, codes, offs, doffs, lev, dlev, u, n)
            Ng = 0.0
            for i in range(s, e):
                Ng += n[i]
                for d in range(D):
                    a = offs[d] + codes[i, d]
                    for d2 in range(D):
                        if doffs[d] >= 0 and doffs[d2] >= 0:
                            dn[doffs[d] + codes[i, d], doffs[d2] + codes[i, d2]] += n[i]
                        else:
                            R[p] = a
                            C[p] = offs[d2] + codes[i, d2]
                            V[p] = n[i]
                            p += 1
            for r1 in range(q):
                for r2 in range(q):
                    v = -u[r1] * u[r2] / Ng
                    if dlev[r1] >= 0 and dlev[r2] >= 0:
                        dn[dlev[r1], dlev[r2]] += v
                    else:
                        R[p] = lev[r1]
                        C[p] = lev[r2]
                        V[p] = v
                        p += 1


@nb.njit(parallel=True, cache=True)
def _nb_csr_matmat(indptr, indices, data, X, out):
    """out = S @ X for CSR S and dense X (L, k); rows split across threads."""
    k = X.shape[1]
    for i in nb.prange(len(indptr) - 1):
        for j in range(k):
            out[i, j] = 0.0
        for p in range(indptr[i], indptr[i + 1]):
            c, v = indices[p], data[p]
            for j in range(k):
                out[i, j] += v * X[c, j]


@nb.njit(parallel=True, cache=True)
def _nb_stream_matvec(P, starts, codes, n, offs, acc):
    """(D_o' M_0 D_o) P without forming the matrix. acc: (nt, L, k) per-thread
    accumulators, zeroed here; the caller sums over axis 0."""
    nt, k, D = acc.shape[0], P.shape[1], codes.shape[1]
    G = len(starts) - 1
    for t in nb.prange(nt):
        out = acc[t]
        out[:] = 0.0
        mg = np.empty(k)
        for gi in range(t * G // nt, (t + 1) * G // nt):
            s, e = starts[gi], starts[gi + 1]
            mg[:] = 0.0
            Ng = 0.0
            for i in range(s, e):
                Ng += n[i]
                for j in range(k):
                    tv = 0.0
                    for d in range(D):
                        tv += P[offs[d] + codes[i, d], j]
                    mg[j] += n[i] * tv
            for j in range(k):
                mg[j] /= Ng
            for i in range(s, e):
                for j in range(k):
                    tv = 0.0
                    for d in range(D):
                        tv += P[offs[d] + codes[i, d], j]
                    v = n[i] * (tv - mg[j])
                    for d in range(D):
                        out[offs[d] + codes[i, d], j] += v


@nb.njit(parallel=True, cache=True)
def _nb_assemble(starts, codes, n, sums, offs, Gamma, acc):
    """acc[t] += sum over identifying cells of n_c r_c r_c', with
    r_c = vbar_c - sum_d Gamma_d[level] - (group mean of the same)."""
    nt, D, m = acc.shape[0], codes.shape[1], sums.shape[1]
    G = len(starts) - 1
    for t in nb.prange(nt):
        q = np.empty(m)
        qg = np.empty(m)
        for gi in range(t * G // nt, (t + 1) * G // nt):
            s, e = starts[gi], starts[gi + 1]
            qg[:] = 0.0
            Ng = 0.0
            for i in range(s, e):
                Ng += n[i]
                for j in range(m):
                    v = sums[i, j] / n[i]
                    for d in range(D):
                        v -= Gamma[offs[d] + codes[i, d], j]
                    qg[j] += n[i] * v
            for j in range(m):
                qg[j] /= Ng
            for i in range(s, e):
                for j in range(m):
                    v = sums[i, j] / n[i]
                    for d in range(D):
                        v -= Gamma[offs[d] + codes[i, d], j]
                    q[j] = v - qg[j]
                for j1 in range(m):
                    for j2 in range(m):
                        acc[t, j1, j2] += n[i] * q[j1] * q[j2]


@nb.njit(parallel=True, cache=True)
def _nb_pass2(starts, codes, offs, w, y, X, beta, gam_y, gam_y0, Z, gam_z, Pi, yc,
              fweights, g_eff, e_out, h_out, acc_s, acc_B, acc_hc, acc_g):
    """Row pass for one chunk of complete fe[0] groups.

    Residuals use the regressors X:  e = y - X beta - FEs, with the fe[0]
    effect the weighted group mean of y - X beta - (other FEs). The scores
    use h = Pi' z~, where z~ are the residualized instruments Z; for OLS,
    Z = X and Pi = I, so h = x~. Per-thread accumulators:
      acc_s[t]  = [sum w e^2, sum w y~^2, sum w (y-yc), sum w (y-yc)^2]
      acc_B[t]  = sum w h h'                     (bread^-1)
      acc_hc[t] = sum w^2 e^2 h h'  (w e^2 h h' for frequency weights)
      acc_g[t]  = sum over fe[0] groups of s_g s_g', s_g = sum w h e
    """
    nt, D, k, q = acc_B.shape[0], codes.shape[1], X.shape[1], Z.shape[1]
    G = len(starts) - 1
    for t in nb.prange(nt):
        mz = np.empty(q)
        zt = np.empty(q)
        h = np.empty(k)
        sg = np.empty(k)
        for gi in range(t * G // nt, (t + 1) * G // nt):
            s, e = starts[gi], starts[gi + 1]
            Wg = 0.0
            mu = 0.0
            my = 0.0
            mz[:] = 0.0
            for i in range(s, e):
                wi = w[i]
                Wg += wi
                u = y[i]
                yy = y[i]
                for j in range(k):
                    u -= X[i, j] * beta[j]
                for d in range(D):
                    a = offs[d] + codes[i, d]
                    u -= gam_y[a]
                    yy -= gam_y0[a]
                for j in range(q):
                    v = Z[i, j]
                    for d in range(D):
                        v -= gam_z[offs[d] + codes[i, d], j]
                    mz[j] += wi * v
                mu += wi * u
                my += wi * yy
                e_out[i] = u
            mu /= Wg
            my /= Wg
            for j in range(q):
                mz[j] /= Wg
            g_eff[gi] = mu
            sg[:] = 0.0
            for i in range(s, e):
                wi = w[i]
                yy = y[i]
                for d in range(D):
                    yy -= gam_y0[offs[d] + codes[i, d]]
                yy -= my
                ei = e_out[i] - mu
                e_out[i] = ei
                for j in range(q):
                    v = Z[i, j]
                    for d in range(D):
                        v -= gam_z[offs[d] + codes[i, d], j]
                    zt[j] = v - mz[j]
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

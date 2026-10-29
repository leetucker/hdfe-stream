"""numba kernels for the standard (no varying slopes) path."""

from __future__ import annotations

import numba as nb
import numpy as np

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

@nb.njit(parallel=True, cache=True)
def _nb_stack(cols, out):
    """out[:, j] = cols[j] (a typed list of 1-D float64 arrays), in blocks
    of rows that stay in cache, the blocks in parallel. Writing whole columns
    into a C-ordered array instead touches every row's cache line once per
    column: on a 2-million-row, 39-column chunk that took 0.47 s, this 0.04 s."""
    n, k = out.shape
    nblk = (n + 4095) // 4096
    for b in nb.prange(nblk):
        r0 = b * 4096
        r1 = min(n, r0 + 4096)
        for j in range(k):
            a = cols[j]
            for i in range(r0, r1):
                out[i, j] = a[i]


@nb.njit(parallel=True, cache=True)
def _nb_unpack(Xd, dcols, Xb, bcols, out):
    """out[:, dcols] = Xd and out[:, bcols] = Xb (uint8), row by row, rows
    in parallel: a design stored with its 0/1 columns as bytes, put back
    together in one pass over out."""
    n = out.shape[0]
    kd, kb = len(dcols), len(bcols)
    for i in nb.prange(n):
        for j in range(kd):
            out[i, dcols[j]] = Xd[i, j]
        for j in range(kb):
            out[i, bcols[j]] = Xb[i, j]


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
def _nb_residualize_rows(starts, codes, offs, w, V, Gamma, demean, nt):
    """Overwrite V with the fully residualized design: v - sum_d
    Gamma_d[level], less its weighted fe[0]-group mean. The cross-products
    V' W V are then one BLAS product in the caller, rather than k x k scalar
    updates per row here.

    With `demean` False there is no fe[0] (no fixed effects at all) and the
    group mean is zero; `starts` then only splits the rows among threads.
    """
    D, m = codes.shape[1], V.shape[1]
    G = len(starts) - 1
    for t in nb.prange(nt):
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
                    V[i, j] = v
                    qg[j] += w[i] * v
            if demean:
                for j in range(m):
                    qg[j] /= Wg
                for i in range(s, e):
                    for j in range(m):
                        V[i, j] -= qg[j]


@nb.njit(cache=True)
def _uf_find(parent, x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


@nb.njit(cache=True)
def _nb_components(starts, codes, n_levels, col=0):
    """Union-find over non-streamed dimension `col` (the first by default):
    levels that share an fe[0] group are connected. Returns a root label
    per level."""
    parent = np.arange(n_levels)
    for gi in range(len(starts) - 1):
        s, e = starts[gi], starts[gi + 1]
        r0 = _uf_find(parent, codes[s, col])
        for i in range(s + 1, e):
            r = _uf_find(parent, codes[i, col])
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
def _nb_union_pairs(parent, a, b, off):
    """Union-find over the graph of two non-streamed dimensions: each cell
    joins its level `a[i]` of the first with its level `b[i]` of the second,
    stored at `off + b[i]`. `parent` carries over between chunks of cells."""
    for i in range(len(a)):
        ra = _uf_find(parent, a[i])
        rb = _uf_find(parent, off + b[i])
        if ra != rb:
            if ra < rb:
                parent[rb] = ra
            else:
                parent[ra] = rb


# The explicit S = D_o' M_0 D_o is built directly in CSR, one row (level) at
# a time. Row r collects, from every fe[0] group g that contains level r,
#     sum over g's cells with level r of  n a'   -   u_r' Ainv_g U_g'
# where a is a cell's level indicator and U_g holds the group's per-level
# sums of the projection's regressors: without slopes those are the cell
# counts and Ainv_g = 1 / N_g. A first pass counts each row's distinct
# columns so S is allocated once at its final size; a second fills it.

@nb.njit(cache=True)
def _nb_level_groups(starts, codes, offs, L):
    """For each level, the fe[0] groups that contain it, ascending: groups of
    level a are grp[ptr[a]:ptr[a+1]]."""
    D = codes.shape[1]
    G = len(starts) - 1
    mark = np.full(L, -1, np.int64)
    ptr = np.zeros(L + 1, np.int64)
    for g in range(G):
        for i in range(starts[g], starts[g + 1]):
            for d in range(D):
                a = offs[d] + codes[i, d]
                if mark[a] != g:
                    mark[a] = g
                    ptr[a + 1] += 1
    for a in range(L):
        ptr[a + 1] += ptr[a]
    grp = np.empty(ptr[L], np.int32)
    fill = ptr[:-1].copy()
    mark[:] = -1
    for g in range(G):
        for i in range(starts[g], starts[g + 1]):
            for d in range(D):
                a = offs[d] + codes[i, d]
                if mark[a] != g:
                    mark[a] = g
                    grp[fill[a]] = g
                    fill[a] += 1
    return ptr, grp


@nb.njit(parallel=True, cache=True)
def _nb_row_work(starts, lptr, lgrp, out):
    """Cells scanned to build each row: the sizes of the groups it is in."""
    for r in nb.prange(len(lptr) - 1):
        w = 0
        for k in range(lptr[r], lptr[r + 1]):
            g = lgrp[k]
            w += starts[g + 1] - starts[g]
        out[r] = w


@nb.njit(parallel=True, cache=True)
def _nb_row_nnz(starts, codes, offs, lptr, lgrp, work, wide, bounds, out):
    """Distinct columns of each row: every level of every group the row's
    level is in. Rows are split into blocks [bounds[k], bounds[k+1]) of
    similar work; rows with work > `wide` (levels of small dimensions, which
    are in most groups) mark columns in a dense flag array instead."""
    D = codes.shape[1]
    L = len(lptr) - 1
    for k in nb.prange(len(bounds) - 1):
        mark = np.full(L, -1, np.int32)
        flag = np.zeros(0, np.uint8)
        for r in range(bounds[k], bounds[k + 1]):
            cnt = 0
            if work[r] > wide:
                if len(flag) == 0:
                    flag = np.zeros(L, np.uint8)
                else:
                    flag[:] = 0
                for j in range(lptr[r], lptr[r + 1]):
                    g = lgrp[j]
                    for i in range(starts[g], starts[g + 1]):
                        for d in range(D):
                            flag[offs[d] + codes[i, d]] = 1
                for c in range(L):
                    cnt += flag[c]
            else:
                for j in range(lptr[r], lptr[r + 1]):
                    g = lgrp[j]
                    for i in range(starts[g], starts[g + 1]):
                        for d in range(D):
                            c = offs[d] + codes[i, d]
                            if mark[c] != r:
                                mark[c] = r
                                cnt += 1
            out[r] = cnt


@nb.njit(parallel=True, cache=True)
def _nb_row_fill(starts, codes, n, st, Ainv, Ng, offs, lptr, lgrp, work, wide, bounds,
                 indptr, indices, data):
    """Fill the rows of S (see above), each with sorted column indices.

    Without slopes `Ainv` is empty and the correction divides by Ng rather
    than multiplying by 1 / Ng, so that it cancels exactly where it should
    (a worker who never moves). Wide rows (see `_nb_row_nnz`) accumulate in a
    dense array and come out sorted; the rest accumulate in place and are
    sorted at the end."""
    D, p = codes.shape[1], st.shape[1]
    L = len(lptr) - 1
    plain = Ainv.shape[0] == 0
    for k in nb.prange(len(bounds) - 1):
        mark = np.full(L, -1, np.int32)
        pos = np.empty(L, np.int32)
        acc = np.zeros(0)
        flag = np.zeros(0, np.uint8)
        ur = np.empty(p)
        w = np.empty(p)
        for r in range(bounds[k], bounds[k + 1]):
            dr = 0
            while dr + 1 < D and offs[dr + 1] <= r:
                dr += 1
            lr = r - offs[dr]
            p0 = indptr[r]
            dense = work[r] > wide
            if dense and len(acc) == 0:
                acc = np.zeros(L)
                flag = np.zeros(L, np.uint8)
            q = 0
            for j in range(lptr[r], lptr[r + 1]):
                g = lgrp[j]
                s, e = starts[g], starts[g + 1]
                ur[:] = 0.0
                for i in range(s, e):
                    if codes[i, dr] == lr:
                        for c in range(p):
                            ur[c] += st[i, c]
                if plain:
                    w[0] = ur[0] / Ng[g]
                else:
                    for c1 in range(p):
                        v = 0.0
                        for c2 in range(p):
                            v += Ainv[g, c1, c2] * ur[c2]
                        w[c1] = v
                for i in range(s, e):
                    v = 0.0
                    for c in range(p):
                        v -= st[i, c] * w[c]
                    if codes[i, dr] == lr:
                        v += n[i]
                    for d in range(D):
                        col = offs[d] + codes[i, d]
                        if dense:
                            acc[col] += v
                            flag[col] = 1
                        elif mark[col] != r:
                            mark[col] = r
                            pos[col] = q
                            indices[p0 + q] = col
                            data[p0 + q] = v
                            q += 1
                        else:
                            data[p0 + pos[col]] += v
            if dense:
                for c in range(L):
                    if flag[c]:
                        indices[p0 + q] = c
                        data[p0 + q] = acc[c]
                        q += 1
                        acc[c] = 0.0
                        flag[c] = 0
            else:
                order = np.argsort(indices[p0:p0 + q])
                ci = indices[p0:p0 + q][order]
                cv = data[p0:p0 + q][order]
                indices[p0:p0 + q] = ci
                data[p0:p0 + q] = cv


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
def _nb_run_sums(cs, ZT, w, ST, Wparts):
    """Collapse runs of rows into cells (pass 1).

    cs     (C+1,)    start of each cell's run of rows, and the end
    ZT     (m, n)    the centered variables, one contiguous row per variable
    w      (n,)      row weights, or empty for none
    ST     (m, C)    out: each cell's weighted sums
    Wparts (B, m, m) out, when it has columns: block sums of the within-cell
                     scatter, sum_i w_i (z_i - zbar_c)(z_i - zbar_c)', for the
                     caller to add up; B fixes the blocks the cells are split
                     into, so the result does not depend on the thread count
    """
    m, C = ST.shape
    nblk = Wparts.shape[0]
    weighted = w.shape[0] > 0
    within = Wparts.shape[1] > 0
    for b in nb.prange(nblk):
        mean = np.empty(m)
        dev = np.empty(m)
        if within:
            Wparts[b, :, :] = 0.0
        for c in range(b * C // nblk, (b + 1) * C // nblk):
            s, e = cs[c], cs[c + 1]
            for j in range(m):
                acc = 0.0
                if weighted:
                    for i in range(s, e):
                        acc += w[i] * ZT[j, i]
                else:
                    for i in range(s, e):
                        acc += ZT[j, i]
                ST[j, c] = acc
            if within:
                n = 0.0
                if weighted:
                    for i in range(s, e):
                        n += w[i]
                else:
                    n = e - s
                for j in range(m):
                    mean[j] = ST[j, c] / n
                for i in range(s, e):
                    wi = w[i] if weighted else 1.0
                    for j in range(m):
                        dev[j] = ZT[j, i] - mean[j]
                    for j1 in range(m):
                        for j2 in range(m):
                            Wparts[b, j1, j2] += wi * dev[j1] * dev[j2]


# Conjugate-gradient vector updates, fused so that each iteration reads the
# level-sized arrays once or twice instead of once per numpy operation, and
# split across threads. Column sums are reduced in a fixed number of blocks
# (`parts.shape[0]`), not one per thread, so the result does not depend on the
# thread count.

@nb.njit(parallel=True, cache=True)
def _nb_cg_dot(A, B, parts):
    """parts[k, j] = sum over block k's rows of A[i, j] * B[i, j]; the caller
    sums over axis 0."""
    nb_, L, m = parts.shape[0], A.shape[0], A.shape[1]
    for k in nb.prange(nb_):
        for j in range(m):
            parts[k, j] = 0.0
        for i in range(k * L // nb_, (k + 1) * L // nb_):
            for j in range(m):
                parts[k, j] += A[i, j] * B[i, j]


@nb.njit(parallel=True, cache=True)
def _nb_cg_step(X, R, P, AP, alpha, dinv, parts):
    """X += P alpha and R -= AP alpha, column by column, and in the same pass
    the block sums of R**2 (parts[k, 0]) and, with a Jacobi preconditioner
    `dinv` (empty otherwise), of R**2 * dinv = R'Z (parts[k, 1])."""
    nb_, L, m = parts.shape[0], X.shape[0], X.shape[1]
    jacobi = dinv.shape[0] > 0
    for k in nb.prange(nb_):
        for j in range(m):
            parts[k, 0, j] = 0.0
            parts[k, 1, j] = 0.0
        for i in range(k * L // nb_, (k + 1) * L // nb_):
            for j in range(m):
                X[i, j] += alpha[j] * P[i, j]
                r = R[i, j] - alpha[j] * AP[i, j]
                R[i, j] = r
                parts[k, 0, j] += r * r
                if jacobi:
                    parts[k, 1, j] += r * r * dinv[i]


@nb.njit(parallel=True, cache=True)
def _nb_cg_direction(P, Z, dinv, beta):
    """P = Z + P beta, with Z = dinv * R (Jacobi: pass R as Z and the
    diagonal as dinv) or a preconditioned residual (pass an empty dinv)."""
    L, m = P.shape[0], P.shape[1]
    jacobi = dinv.shape[0] > 0
    for i in nb.prange(L):
        for j in range(m):
            z = Z[i, j] * dinv[i] if jacobi else Z[i, j]
            P[i, j] = z + beta[j] * P[i, j]


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
def _nb_pass2(starts, codes, offs, w, y, X, beta, gam_y, gam_y0, Z, gam_z, yc,
              demean, g_eff, e_out, zt_out, sg_out, acc_s):
    """Row pass for one chunk of complete fe[0] groups.

    Residuals use the regressors X:  e = y - X beta - FEs, with the fe[0]
    effect the weighted group mean of y - X beta - (other FEs). zt_out
    receives the residualized instruments Z (for OLS, Z = X); the caller forms
    the scores h = Pi' z~ and every k x k sum from them with BLAS. With
    `demean` False there is no fe[0] (no fixed effects at all): that effect,
    and every other group mean, is zero, and `starts` only splits the rows
    among threads. sg_out[g], when it has a row per group, receives the
    group's sum of w e z~ (the scores for clustering on fe[0], before Pi).
    Per-thread accumulators:
      acc_s[t] = [sum w e^2, sum w y~^2, sum w (y-yc), sum w (y-yc)^2]
    """
    want_sg = sg_out.shape[0] > 0
    D, k, q = codes.shape[1], X.shape[1], Z.shape[1]
    G = len(starts) - 1
    nt = acc_s.shape[0]
    for t in nb.prange(nt):
        mz = np.empty(q)
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
                    zt_out[i, j] = v
                    mz[j] += wi * v
                mu += wi * u
                my += wi * yy
                e_out[i] = u
            if demean:
                mu /= Wg
                my /= Wg
                for j in range(q):
                    mz[j] /= Wg
            else:
                mu = 0.0
                my = 0.0
                mz[:] = 0.0
            g_eff[gi] = mu
            if want_sg:
                sg_out[gi, :] = 0.0
            for i in range(s, e):
                wi = w[i]
                yy = y[i]
                for d in range(D):
                    yy -= gam_y0[offs[d] + codes[i, d]]
                yy -= my
                ei = e_out[i] - mu
                e_out[i] = ei
                for j in range(q):
                    zt_out[i, j] -= mz[j]
                if want_sg:
                    for j in range(q):
                        sg_out[gi, j] += wi * ei * zt_out[i, j]
                acc_s[t, 0] += wi * ei * ei
                acc_s[t, 1] += wi * yy * yy
                acc_s[t, 2] += wi * (y[i] - yc)
                acc_s[t, 3] += wi * (y[i] - yc) * (y[i] - yc)




# --------------------------------------------------------------------------
# CRV3: the cluster jackknife, by downdating
#
# Leaving cluster g out of a (weighted) least-squares fit moves beta by
#     beta_(-g) - beta = -(A - A_g)^+ s_g,
# with A = sum w h h' over all rows, A_g the same over g's rows and
# s_g = sum w e h over g's rows (exact, since sum w e h = 0 over all rows).
# The pseudo-inverse follows pyfixest's own downdate, so a cluster that holds
# all of a covariate's variation gives the same answer in both.
# --------------------------------------------------------------------------

@nb.njit(parallel=True, cache=True)
def _nb_crv3_groups(starts, w, e, H, A, acc):
    """acc[t] += d_g d_g' over the groups of a chunk, d_g = (A - A_g)^+ s_g:
    for clusters that are the fe[0] groups themselves, which a chunk holds
    whole."""
    nt, k = acc.shape[0], H.shape[1]
    G = len(starts) - 1
    for t in nb.prange(nt):
        M = np.empty((k, k))
        sg = np.empty(k)
        for gi in range(t * G // nt, (t + 1) * G // nt):
            M[:, :] = A
            sg[:] = 0.0
            for i in range(starts[gi], starts[gi + 1]):
                wi = w[i]
                for a in range(k):
                    sg[a] += wi * e[i] * H[i, a]
                    for b in range(k):
                        M[a, b] -= wi * H[i, a] * H[i, b]
            d = np.linalg.pinv(M) @ sg
            for a in range(k):
                for b in range(k):
                    acc[t, a, b] += d[a] * d[b]


@nb.njit(cache=True)
def _nb_crv3_scatter(code, w, e, H, AG, sG):
    """Per-cluster sums over a chunk's rows, for clusters spread across
    chunks: AG[c] += w h h', sG[c] += w e h. Serial: rows of one cluster
    land anywhere, and per-thread copies of AG would cost G x k^2 each."""
    k = H.shape[1]
    for i in range(len(code)):
        c = code[i]
        wi = w[i]
        for a in range(k):
            sG[c, a] += wi * e[i] * H[i, a]
            for b in range(k):
                AG[c, a, b] += wi * H[i, a] * H[i, b]


@nb.njit(parallel=True, cache=True)
def _nb_crv3_finish(A, AG, sG, acc):
    """acc[t] += d_g d_g' over all clusters, from the sums _nb_crv3_scatter
    collected."""
    nt, k = acc.shape[0], A.shape[0]
    G = AG.shape[0]
    for t in nb.prange(nt):
        for g in range(t * G // nt, (t + 1) * G // nt):
            d = np.linalg.pinv(A - AG[g]) @ sG[g]
            for a in range(k):
                for b in range(k):
                    acc[t, a, b] += d[a] * d[b]

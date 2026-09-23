"""
hdfe_stream: out-of-core OLS with several high-dimensional fixed effects.

    from hdfe_stream import feols_stream
    fit = feols_stream("log_earn ~ age_squared + age_cubed | pik + sein + year",
                       "data/*.parquet", workdir="hdfe_work", vcov={"CRV1": "pik"})
    fit.summary()

Formulas follow pyfixest syntax and are parsed with pyfixest's own parser:
covariates, interactions (`:` and `*`), `I()`, `log()`, `C()`, `i()`,
several dependent variables (`y1 + y2 ~ ...`), `sw()` / `csw()` stepwise
syntax, fixed-effect interactions (`sein^year`), and IV
(`y ~ x | fe | endog ~ z`). Every covariate term is compiled to a Polars
expression, so constructing the design never leaves the streaming engine.
Weighted least squares via `weights=` (aweights or fweights). The
lower-level `StreamingHDFE` class takes column names or Polars expressions
directly.

Reporting: `result.to_pyfixest()` returns a pyfixest Feols/Feiv view for
pf.etable, pf.summary, pf.coefplot and pf.iplot; `hdfe_stream.etable(...)`
does the conversion for you and accepts ordinary pyfixest models too.

Model:  y = X b + sum_d  gamma_d[level_d(i)] + e     for d = 0 ... D-1

One fixed-effect dimension (`fe[0]` internally) is *streamed*: it is never
represented as a vector in memory. It should be the one with the most levels
whose groups each touch only a few levels of the other dimensions (workers,
not firms). It is chosen as: the `stream=` option if given; else the
dimension with varying slopes; else the highest approximate cardinality
(HyperLogLog counts gathered during pass 0's first scan), ties going to the
dimension listed first. Every other dimension is represented by level-sized
vectors in RAM.

Varying slopes (FEIS) on the streamed dimension use fixest syntax:
`| pik[exper] + sein` gives worker effects plus worker-specific slopes on
exper (several: `pik[exper, exper2]`). Projecting out the streamed dimension
then means removing each worker's own weighted regression on [1, slopes],
which is still local to the worker.

Memory: rows are only ever streamed. The arrays held in RAM are sized by the
non-streamed dimensions' level counts (times a block of at most `rhs_block`
variables at a time), plus, for the explicit solver, the sparse reduced
matrix S, whose size depends on how levels co-occur and not on rows. Polars
steps stream; the sort and cell group_by run one fe[0] hash bucket at a time.

Pipeline
--------
Pass 0 (Polars, streaming)   evaluate the design as Polars expressions; drop
                             rows with missing / non-finite values; factorize
                             the non-streamed FE and cluster ids;
                             hash-partition rows into fe[0] buckets; sort each
                             bucket and assign dense fe[0] codes.
Pass 1 (Polars, streaming)   per bucket: group_by(fe[0], other codes) -> cell
                             table with n and sums (plus within-cell
                             cross-products when assembly="cells").
Pass 1b (streamed + numba)   keep cells of fe[0] groups with more than one
                             cell as memory-mapped arrays; Jacobi diagonal.
Step 2 (solver, per block)   solve  S Gamma = D_o' M_0 V,  S = D_o' M_0 D_o,
                             for all variables V in blocks of `rhs_block`
                             columns: "explicit" (sparse S built once, block
                             PCG in memory), "stream_cg" (S applied by
                             streaming the cells), or "within".
Step 3 (numba)               assemble V' M_D V (from cells, or from a row
                             pass when there are many variables); drop
                             collinear covariates per model; beta.
Step 4 (streamed + numba)    per model: stream rows for fe[0] effects,
                             residuals, RSS and the meat for HC1 / CRV1.
"""

from __future__ import annotations

import ast
import logging
import os
import re
import shutil
import socket
import tempfile
import time
import uuid
import warnings
import weakref
from dataclasses import dataclass, field
from pathlib import Path

import numba as nb
import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import scipy.sparse as sp
from scipy import stats


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _log(verbose, msg, logger=None, level=None):
    """Progress message: to `logger` if given (its own formatting and
    timestamps apply), else printed to stdout with a timestamp."""
    if not verbose:
        return
    if logger is not None:
        logger.log(level if level is not None else logging.INFO, msg)
    else:
        print(f"[hdfe {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _warn(msg, logger=None):
    """Warnings go to logger.warning when a logger is given (so they show up
    in the log in real time), else through the warnings module."""
    if logger is not None:
        logger.warning(msg)
    else:
        warnings.warn(msg, stacklevel=3)


def _emit(text, logger=None, level=None):
    """Multi-line report text (summaries): one log record, or print."""
    if logger is not None:
        logger.log(level if level is not None else logging.INFO, text)
    else:
        print(text)


def _phys_mem_gb():
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    except (ValueError, OSError, AttributeError):
        return 16.0


def _safe(name):
    """File-name-safe version of a dimension name like 'sein^year'."""
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name.replace("^", "_x_"))


def iter_group_chunks(paths, columns, batch_rows=2_000_000):
    """Yield dicts of numpy arrays from Parquet file(s) sorted by `gcode`.

    Every yielded chunk contains *complete* fe[0] groups: rows of the last
    group in a batch are carried into the next one. With several files,
    `gcode` must be increasing across files (as the bucket files are).
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]
    carry = None
    for batch in (b for p in paths
                  for b in pq.ParquetFile(p).iter_batches(batch_size=batch_rows,
                                                          columns=columns)):
        tbl = pa.Table.from_batches([batch])
        if carry is not None:
            tbl = pa.concat_tables([carry, tbl])
        g = tbl.column("gcode").to_numpy()
        cut = int(np.searchsorted(g, g[-1], side="left"))
        if cut == 0:  # the whole buffer is one group; keep accumulating
            carry = tbl
            continue
        yield _tbl_to_np(tbl.slice(0, cut))
        carry = tbl.slice(cut)
    if carry is not None and carry.num_rows:
        yield _tbl_to_np(carry)


def _tbl_to_np(tbl):
    return {c: tbl.column(c).to_numpy() for c in tbl.column_names}


def _segments(g):
    """Segment starts and local segment index for a sorted code vector."""
    brk = np.flatnonzero(g[1:] != g[:-1]) + 1
    starts = np.concatenate(([0], brk))
    local = np.zeros(len(g), dtype=np.int64)
    local[brk] = 1
    np.cumsum(local, out=local)
    return starts, local


def _scatter(idx, vals, n):
    """Column-wise bincount: out[j] = sum of the rows of vals with idx == j."""
    vals = np.asarray(vals)
    if vals.ndim == 1:
        return np.bincount(idx, weights=vals, minlength=n)
    return np.column_stack(
        [np.bincount(idx, weights=vals[:, j], minlength=n) for j in range(vals.shape[1])]
    )


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


# --------------------------------------------------------------------------
# formula front end: pyfixest syntax -> Polars expressions
#
# pyfixest's parser splits formulas and expands multiple-estimation syntax.
# To get exactly pyfixest's coefficient names and categorical coding, the
# covariate side is run through pyfixest/formulaic on a tiny *synthetic*
# frame that contains every level of every categorical variable. Each
# resulting column name is then compiled back to a Polars expression:
# components are split on ':' and are either an indicator for one level
# ("C(g)[T.2]", "cat[a]", "year::2005") or a numeric expression ("x",
# "I(age ** 2)", "log(age)") translated through a small AST compiler.
# --------------------------------------------------------------------------

def _pyfixest_formula_api():
    try:
        from pyfixest.estimation.formula.model_matrix import create_model_matrix
        from pyfixest.estimation.formula.parse import Formula
    except ImportError as err:  # internal API; pin the pyfixest version
        raise ImportError(
            "formula support uses pyfixest's formula parser "
            "(pyfixest.estimation.formula), which this pyfixest version does not "
            "provide; pin a compatible pyfixest or use StreamingHDFE directly") from err
    return Formula, create_model_matrix


_FUNCS = {
    "log": lambda e: e.log(), "log1p": lambda e: e.log1p(), "log10": lambda e: e.log10(),
    "exp": lambda e: e.exp(), "sqrt": lambda e: e.sqrt(), "abs": lambda e: e.abs(),
}
_BINOPS = {
    ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b, ast.Div: lambda a, b: a / b,
    ast.Pow: lambda a, b: a.pow(b), ast.FloorDiv: lambda a, b: a // b,
    ast.Mod: lambda a, b: a % b,
}
_CMPOPS = {
    ast.Eq: lambda a, b: a == b, ast.NotEq: lambda a, b: a != b,
    ast.Lt: lambda a, b: a < b, ast.LtE: lambda a, b: a <= b,
    ast.Gt: lambda a, b: a > b, ast.GtE: lambda a, b: a >= b,
}


def _ast_to_expr(node, numeric=True):
    """Compile a small, safe subset of Python expressions to Polars."""
    if isinstance(node, ast.Expression):
        return _ast_to_expr(node.body, numeric)
    if isinstance(node, ast.Name):
        return pl.col(node.id).cast(pl.Float64) if numeric else pl.col(node.id)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (bool, int, float)) and numeric:
            return pl.lit(float(node.value))
        return pl.lit(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        return _BINOPS[type(node.op)](_ast_to_expr(node.left), _ast_to_expr(node.right))
    if isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.USub):
            return -_ast_to_expr(node.operand)
        if isinstance(node.op, ast.UAdd):
            return _ast_to_expr(node.operand)
    if isinstance(node, ast.Compare) and len(node.ops) == 1 and type(node.ops[0]) in _CMPOPS:
        out = _CMPOPS[type(node.ops[0])](_ast_to_expr(node.left, False),
                                         _ast_to_expr(node.comparators[0], False))
        return out.cast(pl.Float64)
    if isinstance(node, ast.Call) and len(node.args) == 1 and not node.keywords:
        f = node.func
        fname = f.id if isinstance(f, ast.Name) else (f.attr if isinstance(f, ast.Attribute) else None)
        if fname == "I":
            return _ast_to_expr(node.args[0])
        if fname in _FUNCS:
            return _FUNCS[fname](_ast_to_expr(node.args[0]))
    raise ValueError(f"unsupported expression in formula: {ast.unparse(node)!r}. Supported: "
                     "columns, + - * / ** // %, comparisons, I(), "
                     f"{', '.join(sorted(_FUNCS))}, C(), i(). Construct anything else "
                     "as a column in the input LazyFrame.")


def _numeric_expr(src):
    return _ast_to_expr(ast.parse(src, mode="eval")).cast(pl.Float64)


def _split_components(name):
    """Split a model-matrix column name on top-level ':' (keeping '::')."""
    s = name.replace("::", "\x00")
    parts, depth, cur = [], 0, ""
    for ch in s:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        if ch == ":" and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    parts.append(cur)
    return [p.replace("\x00", "::") for p in parts]


def _categorical_vars(rhs, schema):
    """Variables that need level discovery: first argument of C()/i(), a
    non-numeric second argument of i(), and non-numeric columns."""
    import formulaic
    cats = set()
    for term in formulaic.Formula(rhs):
        for fac in term.factors:
            src = fac.expr
            if src in schema:
                if not (schema[src].is_numeric() or schema[src] == pl.Boolean):
                    cats.add(src)
                continue
            try:
                tree = ast.parse(src, mode="eval")
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                        and node.func.id in ("C", "i"):
                    kw = {k.arg for k in node.keywords}
                    if node.func.id == "C" and (len(node.args) != 1 or kw):
                        raise ValueError(f"only plain C(var) is supported, got {src!r}")
                    if node.func.id == "i" and kw & {"bin", "bin2"}:
                        raise ValueError("i(..., bin=) is not supported; bin the variable "
                                         "in the input LazyFrame instead")
                    for j, arg in enumerate(node.args[:2]):
                        if not isinstance(arg, ast.Name):
                            raise ValueError(f"arguments of {node.func.id}() must be column "
                                             f"names, got {src!r}")
                        if j == 0 or not (schema[arg.id].is_numeric()
                                          or schema[arg.id] == pl.Boolean):
                            cats.add(arg.id)
    return cats


def _compile_design(depvar, rhs, schema, levels):
    """Return (depvar expr, [(coefname, expr), ...]) for one model."""
    import formulaic
    Formula, create_model_matrix = _pyfixest_formula_api()
    needed = {v for v in formulaic.Formula(rhs).required_variables if v in schema}
    missing = {v for v in formulaic.Formula(rhs).required_variables
               if v not in schema and v not in ("i", "C", "I", "np", *_FUNCS)}
    if missing:
        raise ValueError(f"unknown variables in formula: {sorted(missing)}")

    # synthetic frame: every categorical level appears at least once
    import pandas as pd
    R = max([3] + [len(levels[v]) for v in needed if v in levels])
    rng = np.random.default_rng(0)
    synth = {"__y__": rng.uniform(1, 2, R)}
    for v in needed:
        if v in levels:
            synth[v] = np.resize(np.array(levels[v], dtype=object if isinstance(
                levels[v][0], str) else None), R)
        else:
            synth[v] = rng.uniform(1, 2, R)
    mm = create_model_matrix(Formula.parse(f"__y__ ~ {rhs}")[0], pd.DataFrame(synth),
                             drop_intercept=True)
    names = [c for c in mm.independent.columns if c != "Intercept"]

    cat_labels = {}
    for v in levels:
        cat_labels[f"C({v})"] = v
        if v in schema and not schema[v].is_numeric():
            cat_labels[v] = v
    lvl_lookup = {v: {str(x): x for x in lv} for v, lv in levels.items()}

    def component(c):
        m = re.fullmatch(r"(.+)\[(T\.)?(.*)\]", c)
        if m and m.group(1) in cat_labels:
            v = cat_labels[m.group(1)]
            return (pl.col(v) == pl.lit(lvl_lookup[v][m.group(3)])).cast(pl.Float64)
        if "::" in c:
            v, lv = c.split("::", 1)
            if v in lvl_lookup and lv in lvl_lookup[v]:
                return (pl.col(v) == pl.lit(lvl_lookup[v][lv])).cast(pl.Float64)
        return _numeric_expr(c)

    cols = []
    for nm in names:
        expr = None
        for c in _split_components(nm):
            e = component(c)
            expr = e if expr is None else expr * e
        cols.append((nm, expr))
    return (depvar, _numeric_expr(depvar)), cols


_FE_TERM = re.compile(r"^\s*([^\[\]]+?)\s*(\[\[?)(.*?)(\]\]?)\s*$")


def _parse_fe_term(term):
    """'pik[exper, exper2]' -> ('pik', ['exper', 'exper2']); 'sein' -> ('sein', [])."""
    m = _FE_TERM.match(term)
    if not m:
        return "^".join(c.strip() for c in term.split("^")), []
    name, opening, inner, closing = m.groups()
    if opening == "[[" or closing == "]]":
        raise NotImplementedError(f"{term!r}: slopes without the FE intercept ([[...]]) are not "
                                  "supported; use name[x] (intercept + slopes)")
    slopes = [v.strip() for v in inner.split(",") if v.strip()]
    if not slopes:
        raise ValueError(f"{term!r}: no slope variables inside [...]")
    return "^".join(c.strip() for c in name.split("^")), slopes


def _parse_fe(fe_str):
    return [t.strip() for t in fe_str.split("+") if t.strip()] if fe_str else []


def _plan_formula(fml, lf):
    """Expand a pyfixest formula into groups of models that share a set of
    fixed effects (and therefore one pass 0/1 and one solve)."""
    Formula, _ = _pyfixest_formula_api()
    schema = lf.collect_schema()
    specs = Formula.parse(fml)
    for s in specs:
        if len(_parse_fe(s.fixed_effects)) < 2:
            raise ValueError(f"'{s.formula}': at least two fixed-effect dimensions are "
                             "required (the first is streamed)")

    # level discovery for all categorical variables (one streaming pass each)
    cats = set()
    for s in specs:
        cats |= _categorical_vars(s.second_stage.split("~", 1)[1], schema)
        if s.first_stage is not None:
            cats |= _categorical_vars(s.first_stage.split("~", 1)[1], schema)
    levels = {}
    for v in sorted(cats):
        u = lf.select(pl.col(v).drop_nulls().unique().sort()).collect(engine="streaming")
        levels[v] = u[v].to_list()

    groups = {}
    for s in specs:
        dep, rhs = (t.strip() for t in s.second_stage.split("~", 1))
        (dname, dexpr), cols = _compile_design(dep, rhs, schema, levels)
        fe = tuple(_parse_fe(s.fixed_effects))
        g = groups.setdefault(fe, {"y": {}, "x": {}, "models": []})
        g["y"][dname] = dexpr
        model = {"fml": s.formula, "y": dname, "x": [nm for nm, _ in cols]}
        extra = []
        if s.first_stage is not None:
            # first stage "endog ~ instruments + exogenous": its columns are the
            # instrument set Z; the endogenous columns come from compiling the
            # left-hand side as a right-hand side
            endog, fs_rhs = (t.strip() for t in s.first_stage.split("~", 1))
            _, fs_cols = _compile_design(dep, fs_rhs, schema, levels)
            _, en_cols = _compile_design(dep, endog, schema, levels)
            model["iv"] = {"endog": [nm for nm, _ in en_cols], "z": [nm for nm, _ in fs_cols]}
            missing = [e for e in model["iv"]["endog"] if e not in model["x"]]
            if missing:
                raise ValueError(f"endogenous columns {missing} not found among regressors")
            extra = fs_cols + en_cols
        for nm, ex in cols + extra:
            g["x"].setdefault(nm, ex)
        g["models"].append(model)
    return groups


# --------------------------------------------------------------------------
# disk management
#
# Each fit works in its own run directory, <workdir>/hdfe_run_<time>_<id>/,
# so concurrent fits never collide and cleanup never touches anything else in
# workdir. Intermediates (the sorted copy of the rows, cell tables, memory-
# mapped arrays, the solution Gamma, factor maps) are deleted as soon as the
# fit no longer needs them. Result files (residuals and fixed effects, read
# lazily through resid()/fixef()) live in <run>/models/ and are removed when
# the results are garbage-collected or the interpreter exits
# (outputs="auto"), when .cleanup() is called, or never (outputs="keep").
# On any exception, the whole run directory is removed.
# --------------------------------------------------------------------------

_MARKER = ".hdfe_stream_run"
_ACTIVE_RUNS = set()      # run directories of fits currently running in this process


def _rmtree_quiet(path):
    _ACTIVE_RUNS.discard(str(path))
    shutil.rmtree(path, ignore_errors=True)


def _dir_bytes(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


class _Run:
    """A run directory, shared by all results of one fit."""

    def __init__(self, base, auto_cleanup):
        base = Path(base)
        base.mkdir(parents=True, exist_ok=True)
        self.path = base / f"hdfe_run_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        self.path.mkdir()
        (self.path / _MARKER).write_text(f"{socket.gethostname()} {os.getpid()} {time.time()}\n")
        _ACTIVE_RUNS.add(str(self.path))
        self._fin = (weakref.finalize(self, _rmtree_quiet, str(self.path))
                     if auto_cleanup else None)

    def cleanup(self):
        if self._fin is not None:
            self._fin()                      # runs once; detaches itself
        else:
            _rmtree_quiet(str(self.path))

    @property
    def exists(self):
        return self.path.exists()


def cleanup(workdir=None, older_than_hours=None, force=False, dry_run=False):
    """Remove leftover run directories (e.g. from killed processes) in
    `workdir` (default: the system temporary directory). Only directories
    created by hdfe_stream (they carry a marker file) are touched.

    Runs of this process are removed unless a fit is still running in them
    (this includes result files of finished fits; their resid()/fixef() stop
    working). Skipped unless force=True: runs of other processes still alive
    on this host, runs from other hosts (their liveness can't be checked),
    and runs younger than `older_than_hours` if given.
    Returns a list of (path, bytes) that were (or, with dry_run, would be)
    removed.
    """
    base = Path(workdir) if workdir else Path(tempfile.gettempdir())
    host, removed = socket.gethostname(), []
    if not base.exists():
        return removed
    for d in sorted(base.glob("hdfe_run_*")):
        marker = d / _MARKER
        if not marker.exists():
            continue
        try:
            h, pid, started = marker.read_text().split()
            pid, started = int(pid), float(started)
        except ValueError:
            h, pid, started = "?", -1, 0.0
        if str(d) in _ACTIVE_RUNS:
            continue                        # a fit is running in it right now
        mine = h == host and pid == os.getpid()
        if not force and not mine:
            if older_than_hours is not None and time.time() - started < older_than_hours * 3600:
                continue
            if h != host:
                continue
            if pid > 0:
                try:
                    os.kill(pid, 0)
                    continue                # process still running
                except ProcessLookupError:
                    pass
                except PermissionError:
                    continue                # exists, owned by someone else
        size = _dir_bytes(d)
        if not dry_run:
            _rmtree_quiet(str(d))
        removed.append((str(d), size))
    return removed


# --------------------------------------------------------------------------
# results
# --------------------------------------------------------------------------

@dataclass
class HDFEResult:
    fml: str
    depvar: str
    coefnames: list
    fe_names: list
    beta: np.ndarray
    vcov: np.ndarray
    vcov_type: str
    df_t: int
    n_obs: int
    n_levels: dict
    n_identifying: int
    n_components: int
    k_fe: int
    rss: float
    r2_within: float
    solver_info: dict
    paths: dict
    collin_vars: list = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)
    all_vcovs: dict = field(default_factory=dict)
    r2: float = np.nan
    adj_r2: float = np.nan
    adj_r2_within: float = np.nan
    rmse: float = np.nan
    is_iv: bool = False
    first_stage: list = field(default_factory=list)
    f_stat_1st_stage: list = field(default_factory=list)
    n_clusters: dict = field(default_factory=dict)
    weights: str | None = None
    weights_type: str = "aweights"
    _run: object = field(default=None, repr=False, compare=False)

    @property
    def se(self):
        return np.sqrt(np.diag(self.vcov))

    def coef(self) -> dict:
        return dict(zip(self.coefnames, self.beta))

    def tidy(self) -> pl.DataFrame:
        se = self.se
        t = np.divide(self.beta, se, out=np.full_like(self.beta, np.nan), where=se > 0)
        p = 2 * stats.t.sf(np.abs(t), self.df_t)
        return pl.DataFrame({
            "Coefficient": self.coefnames,
            "Estimate": self.beta,
            "Std. Error": se,
            "t value": t,
            "Pr(>|t|)": p,
        }, schema_overrides={"Coefficient": pl.Utf8})

    def with_vcov(self, vcov) -> "HDFEResult":
        """Switch among the vcovs computed in pass 2: 'iid', 'hetero', or a
        cluster variable given as {'CRV1': var} or 'CRV1:var'."""
        import copy
        key = _vcov_key(vcov)
        if key not in self.all_vcovs:
            raise KeyError(f"{key} was not computed; available: {list(self.all_vcovs)}. "
                           "Request it at fit time with vcov= or cluster=.")
        out = copy.copy(self)
        v, dft = self.all_vcovs[key]
        out.vcov, out.vcov_type, out.df_t = v, key, dft
        return out

    def to_pyfixest(self):
        """A pyfixest Feols (Feiv for IV) view of this result, for
        pf.etable, pf.summary, pf.coefplot, ... Only the stored results are
        available; methods that need the data (predict, re-computing vcov,
        wild bootstrap) are not."""
        return _to_pyfixest(self)

    def _scan(self, path, what):
        if path is None:
            raise FileNotFoundError(f"{what} was not saved (save_resid=False)")
        if not Path(path).exists():
            raise FileNotFoundError(
                f"{what} file is gone: result files were cleaned up (cleanup() was called, "
                "or, with outputs='auto', every result object of the fit was garbage-"
                "collected). Keep the result object alive while using its lazy frames, "
                "collect/sink what you need, or fit with outputs='keep'.")
        return pl.scan_parquet(path)

    def fixef(self, name: str) -> pl.LazyFrame:
        """Lazy scan of one dimension's estimated fixed effects. The file lives
        as long as this result object (outputs='auto'): collect or sink what
        you need before dropping the result."""
        return self._scan(self.paths["fe"][name], f"fixed effects for {name!r}")

    def resid(self) -> pl.LazyFrame:
        """Lazy scan of row-level output (FE contributions + residuals); same
        lifetime caveat as fixef()."""
        return self._scan(self.paths["resid"], "residual")

    @property
    def files_dir(self):
        """Directory holding this model's result files."""
        return self.paths.get("dir")

    def cleanup(self):
        """Delete this model's result files (and its first-stage files for
        IV); the run directory goes once no model files remain."""
        for fs in self.first_stage:
            fs.cleanup()
        if self.paths.get("dir"):
            shutil.rmtree(self.paths["dir"], ignore_errors=True)
        run = self._run
        if run is not None and run.exists:
            models = run.path / "models"
            if not models.exists() or not any(models.iterdir()):
                run.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.cleanup()
        return False

    def summary_text(self) -> str:
        """The summary() report as a string."""
        lines = [f"### {self.fml}", f"Streaming HDFE   vcov: {self.vcov_type}",
                 "obs: {:,}   ".format(self.n_obs)
                 + "   ".join(f"{k}: {v:,}" for k, v in self.n_levels.items())]
        st = self.diagnostics.get("stream")
        if st:
            lines.append(f"streamed dimension: {st['dim']} ({st['reason']})")
        lines.append(f"identifying {self.fe_names[0]} groups: {self.n_identifying:,}   "
                     f"({self.fe_names[0]} x {self.fe_names[1]}) components: "
                     f"{self.n_components:,}")
        lines.append(f"FE dof: {self.k_fe:,}   RSS: {self.rss:.6g}   RMSE: {self.rmse:.4g}   "
                     f"R2: {self.r2:.6f}   within R2: {self.r2_within:.6f}")
        if self.weights:
            lines.append(f"weights: {self.weights} ({self.weights_type})")
        if self.is_iv:
            lines.append("first-stage F (excluded instruments, same vcov): "
                         + ", ".join(f"{fs.depvar}: {f:.4g}"
                                     for fs, f in zip(self.first_stage, self.f_stat_1st_stage)))
        if self.collin_vars:
            lines.append(f"dropped as collinear: {self.collin_vars}")
        lines.append(f"solver: {self.solver_info}")
        lines.append(str(self.tidy()))
        return "\n".join(lines)

    def summary(self, logger=None, level=logging.INFO):
        """Print the report, or write it to `logger` (one record, at `level`)."""
        _emit(self.summary_text(), logger, level)

class HDFEMulti:
    """Results of a formula that expands to several models."""

    def __init__(self, results):
        self.all_fitted_models = {r.fml: r for r in results}

    def fetch_model(self, i) -> HDFEResult:
        return list(self.all_fitted_models.values())[i]

    def to_pyfixest(self) -> list:
        return [r.to_pyfixest() for r in self]

    def cleanup(self):
        """Delete the result files of all models."""
        for r in self:
            r.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.cleanup()
        return False

    def tidy(self) -> pl.DataFrame:
        return pl.concat([r.tidy().with_columns(pl.lit(f).alias("fml"))
                          for f, r in self.all_fitted_models.items()]).select("fml", pl.all().exclude("fml"))

    def summary_text(self) -> str:
        return "\n\n".join(r.summary_text() for r in self.all_fitted_models.values())

    def summary(self, logger=None, level=logging.INFO, per_model=False):
        """Print all reports, or write them to `logger`: one record in total,
        or one per model with per_model=True."""
        if logger is not None and per_model:
            for r in self.all_fitted_models.values():
                r.summary(logger, level)
        else:
            _emit(self.summary_text(), logger, level)

    def __len__(self):
        return len(self.all_fitted_models)

    def __iter__(self):
        return iter(self.all_fitted_models.values())


def _canon_cluster(var):
    """'pik + sein' -> 'pik+sein'; 'sein ^ year' -> 'sein^year'."""
    return "+".join("^".join(c.strip() for c in part.split("^")) for part in var.split("+"))


def _vcov_key(v):
    if v is None:
        return None
    if isinstance(v, dict):
        (kind, var), = v.items()
        if kind.upper() != "CRV1":
            raise ValueError(f"only CRV1 clustering is supported, got {kind}")
        return f"CRV1:{_canon_cluster(var)}"
    v = str(v)
    if v.lower() in ("iid",):
        return "iid"
    if v.lower() in ("hetero", "hc1"):
        return "hetero"
    if v.upper().startswith("CRV1:"):
        return "CRV1:" + _canon_cluster(v.split(":", 1)[1])
    raise ValueError(f"unsupported vcov {v!r}: use 'iid', 'hetero'/'HC1' or {{'CRV1': var}} "
                     "(multi-way: {'CRV1': 'a+b'})")


# --------------------------------------------------------------------------
# pyfixest adapter (pf.etable, pf.summary, pf.coefplot, ...)
#
# pyfixest's reporting functions accept Feols/Feiv instances, and maketables
# reads a documented set of attributes plus tidy(). The adapter subclasses
# Feols/Feiv without running their constructors and fills those attributes
# from an HDFEResult. Anything that needs the data itself is unavailable.
# --------------------------------------------------------------------------

_ADAPTER_CLASSES = {}


def _adapter_classes():
    if not _ADAPTER_CLASSES:
        import pandas as pd
        from pyfixest.estimation.models.feiv_ import Feiv
        from pyfixest.estimation.models.feols_ import Feols

        class _StreamMixin:
            def __init__(self, res):            # no Feols.__init__: no data here
                _fill_pyfixest(self, res)

            def _index(self):
                return pd.Index(self._coefnames, name="Coefficient")

            def coef(self):
                return pd.Series(self._beta_hat, index=self._index(), name="Estimate")

            def se(self):
                return pd.Series(self._se, index=self._index(), name="Std. Error")

            def tstat(self):
                return pd.Series(self._tstat, index=self._index(), name="t value")

            def pvalue(self):
                return pd.Series(self._pvalue, index=self._index(), name="Pr(>|t|)")

            def confint(self, alpha=0.05, **_):
                crit = stats.t.ppf(1 - alpha / 2, self._df_t)
                lo, hi = self._beta_hat - crit * self._se, self._beta_hat + crit * self._se
                return pd.DataFrame({f"{alpha / 2 * 100:.1f}%": lo,
                                     f"{(1 - alpha / 2) * 100:.1f}%": hi}, index=self._index())

            def tidy(self, alpha=0.05):
                ci = self.confint(alpha)
                return pd.DataFrame({"Estimate": self._beta_hat, "Std. Error": self._se,
                                     "t value": self._tstat, "Pr(>|t|)": self._pvalue,
                                     ci.columns[0]: ci.iloc[:, 0].to_numpy(),
                                     ci.columns[1]: ci.iloc[:, 1].to_numpy()},
                                    index=self._index())

            def vcov(self, *args, **kwargs):
                raise NotImplementedError(
                    "streaming results cannot recompute the vcov; use "
                    "HDFEResult.with_vcov() and convert again")

        class StreamFeols(_StreamMixin, Feols):
            pass

        class StreamFeiv(_StreamMixin, Feiv):
            pass

        _ADAPTER_CLASSES.update(ols=StreamFeols, iv=StreamFeiv)
    return _ADAPTER_CLASSES["ols"], _ADAPTER_CLASSES["iv"]


def _fill_pyfixest(obj, r):
    se = r.se
    t = np.divide(r.beta, se, out=np.full_like(r.beta, np.nan), where=se > 0)
    kind = r.vcov_type
    if kind.startswith("CRV1:"):
        cv = kind.split(":", 1)[1]
        Gs = list(r.n_clusters[cv])
        # pyfixest reports min(G) for every term of a multi-way vcov
        vtype, detail, clustervar, G = ("CRV", "CRV1", cv.split("+"),
                                        Gs if len(Gs) == 1 else [min(Gs)] * len(Gs))
    else:
        vtype, detail, clustervar, G = kind, kind, None, None
    crit = stats.t.ppf(0.975, r.df_t)
    obj.__dict__.update({
        "_fml": r.fml, "_depvar": r.depvar,
        # the FE string as written in the formula: maketables matches FE rows
        # across models by the text between '+' signs, spaces included
        "_fixef": r.fml.split("|")[1].strip() if "|" in r.fml else " + ".join(r.fe_names),
        "_has_fixef": True, "_coefnames": list(r.coefnames), "_k": len(r.coefnames),
        "_beta_hat": np.asarray(r.beta), "_se": se, "_tstat": t,
        "_pvalue": 2 * stats.t.sf(np.abs(t), r.df_t),
        "_conf_int": np.vstack([r.beta - crit * se, r.beta + crit * se]),
        "_vcov": r.vcov, "_vcov_type": vtype, "_vcov_type_detail": detail,
        "_clustervar": clustervar, "_G": G, "_df_t": r.df_t,
        "_N": int(r.n_obs) if float(r.n_obs).is_integer() else r.n_obs,
        "_r2": r.r2, "_adj_r2": r.adj_r2, "_r2_adj": r.adj_r2,
        "_r2_within": r.r2_within, "_adj_r2_within": r.adj_r2_within, "_rmse": r.rmse,
        "_F_stat": None, "deviance": None, "_method": "feols", "_is_iv": r.is_iv,
        "_f_stat_1st_stage": (r.f_stat_1st_stage[0] if r.is_iv and len(r.f_stat_1st_stage) == 1
                              else None),
        "_use_mundlak": False, "_sample_split_var": None, "_sample_split_value": "all",
        "_quantile": None, "_data": None, "_weights_name": r.weights,
        "_weights_type": r.weights_type, "_collin_vars": list(r.collin_vars),
        "_model_name": r.fml, "_model_name_plot": r.fml,
        "_icovars": [c for c in r.coefnames if "::" in c] or None,     # for pf.iplot
        "_stream_result": r,
    })


def _to_pyfixest(r):
    ols, iv = _adapter_classes()
    return (iv if r.is_iv else ols)(r)


def etable(models, **kwargs):
    """pf.etable for streaming results (HDFEResult, HDFEMulti, or a list that
    may mix them with ordinary pyfixest models). kwargs go to pf.etable."""
    import pyfixest as pf
    if isinstance(models, (HDFEResult, HDFEMulti)):
        models = [models]
    out = []
    for m in models:
        if isinstance(m, HDFEMulti):
            out += m.to_pyfixest()
        elif isinstance(m, HDFEResult):
            out.append(m.to_pyfixest())
        else:
            out.append(m)
    return pf.etable(out, **kwargs)


# --------------------------------------------------------------------------
# estimator
# --------------------------------------------------------------------------

class _STooLarge(Exception):
    pass


def _norm_vars(items):
    """Normalize variables to an ordered {name: Float64 Polars expression}."""
    if isinstance(items, dict):
        return {k: v.cast(pl.Float64) for k, v in items.items()}
    if isinstance(items, (str, tuple)):
        items = [items]
    out = {}
    for it in items:
        if isinstance(it, str):
            out[it] = pl.col(it).cast(pl.Float64)
        else:
            name, expr = it
            out[name] = expr.cast(pl.Float64)
    return out


class StreamingHDFE:
    """
    Low-level interface (see `feols_stream` for formulas).

    Parameters
    ----------
    y : dependent variable(s): a column name, or a list of names and/or
        (name, Polars expression) pairs. Each y is estimated as its own model
        (sharing one solve) unless `models` says otherwise.
    x : covariates, as for `y`. May be empty (FE-only model).
    fe : fixed-effect dimensions, e.g. ["pik", "sein", "year"] or
         ["pik", "sein^year"] ('^' interacts columns). One dimension may carry
         varying slopes, "pik[exper]" or "pik[exper, exper2]" (fixest syntax;
         FE intercepts plus slopes); it is then the streamed dimension.
    stream : name of the dimension to stream. Default: the dimension with
         slopes, else the highest approximate cardinality (ties: listed
         first). The choice is logged and stored in diagnostics["stream"];
         pass it explicitly when cardinality is a poor guide (e.g. firm-year
         effects that outnumber workers in a subsample).
    workdir : directory under which each fit creates its own run directory
         (hdfe_run_<time>_<id>/) for intermediate and result files. Default:
         the system temporary directory (honours TMPDIR). Put it on a disk
         with room for several times the input's Parquet size (the sorted
         working copy of the rows is less compressible than typical input;
         about 4-5x in tests); diagnostics["disk_peak_gb"] reports it.
    outputs : "auto" (default): result files (residuals, fixed effects) are
             deleted when all result objects of the fit are garbage-collected
             or the interpreter exits; keep the result alive while using
             resid()/fixef(), or collect/sink what you need.
         "keep": result files stay until result.cleanup() or
             hdfe_stream.cleanup(workdir). Intermediates are always removed.
    save_resid : write the row-level residual file (the largest output;
         default True). With False, resid() is unavailable.
    keep_intermediates : keep intermediate files and, on errors, the run
         directory (for debugging only).
    solver : "auto" (default): "explicit" unless S would exceed `max_s_gb`,
             then "stream_cg".
         "explicit": build S = D_o' M_0 D_o once as a sparse matrix and
             iterate in memory. RAM ~ nnz(S): grows with distinct level
             co-occurrences (firm-year pairs, firm pairs sharing a mover),
             not with rows.
         "stream_cg": never form S; each iteration streams the identifying
             cells. Fallback when S is too big.
         "within": identifying cells in RAM, solved by `within`.
    precond : "jacobi" (default) or "amg" (pyamg smoothed aggregation;
         explicit solver only).
    max_s_gb : size cap for S (default: 25% of physical memory).
    rhs_block : variables solved at a time; bounds solver memory at about
         6 x (total non-streamed levels) x rhs_block doubles.
    assembly : "cells" (within-cell cross-products computed in pass 1),
         "rows" (an extra row pass after the solve; no m^2 aggregation
         columns), or "auto" (cells when there are at most 8 variables).
    collin_tol : tolerance for dropping collinear covariates (pyfixest's
         algorithm applied to the residualized X'X).
    collin_tol_rel : also drop a covariate when its variation left after
         removing the fixed effects is below this share of its raw variation
         (default 1e-6, i.e. residual SD < 0.1% of raw SD).
    weights : column name or Polars expression with strictly positive
         weights (weighted least squares; rows with missing weights are
         dropped).
    weights_type : "aweights" (default; N = rows) or "fweights" (frequency
         weights; N = sum of weights), as in pyfixest.
    keep : extra columns to carry into the residual output.
    n_buckets / rows_per_bucket : rows are hash-partitioned by fe[0] into
         buckets so the sort and the cell group_by are bucket-sized.
    batch_rows : rows per chunk when streaming Parquet.
    cells_in_memory : load the identifying cells into RAM instead of
         memory-mapping them.
    triple_budget, dense_max_levels : tuning for building S (see
         `_build_explicit`).
    n_threads : numba threads (default: all available).
    verbose : emit progress messages (default True).
    logger : a logging.Logger; if given, progress messages go to it (at
         `log_level`, default INFO) as they happen, and warnings (dropped
         collinear variables, non-convergence, negative variances) go to
         logger.warning. Default: print to stdout / warnings module.
    models : list of {"fml", "y", "x"} dicts selecting which of the variables
         enter each model; all models share passes 0-1 and the solve. An IV
         model adds "iv": {"endog": [...], "z": [...]}, where z lists the
         full instrument set (excluded instruments and exogenous regressors).
    """

    def __init__(self, y, x, fe, workdir=None, solver="auto", precond="jacobi", keep=(),
                 tol=1e-10, maxiter=5000, batch_rows=2_000_000, row_group_size=500_000,
                 n_buckets=None, rows_per_bucket=20_000_000, cells_in_memory=False,
                 triple_budget=5_000_000, dense_max_levels=1000, rhs_block=8,
                 assembly="auto", max_s_gb=None, collin_tol=1e-10, collin_tol_rel=1e-6,
                 n_threads=None,
                 weights=None, weights_type="aweights", stream=None, models=None,
                 verbose=True, logger=None, log_level=logging.INFO, outputs="auto",
                 save_resid=True, keep_intermediates=False):
        ys, xs = _norm_vars(y), _norm_vars(x if x is not None else [])
        self.var_names = list(ys) + [k for k in xs if k not in ys]
        self.var_exprs = {**xs, **ys}
        self.m = len(self.var_names)
        self.vidx = {nm: j for j, nm in enumerate(self.var_names)}
        parsed = [_parse_fe_term(t) for t in fe]
        self.fe_user = [nm for nm, _ in parsed]
        if len(self.fe_user) < 2:
            raise ValueError("need at least two fixed-effect dimensions")
        if len(set(self.fe_user)) < len(self.fe_user):
            raise ValueError(f"duplicate fixed-effect dimensions in {self.fe_user}")
        self.slopes = {nm: sl for nm, sl in parsed if sl}
        if len(self.slopes) > 1:
            raise ValueError("varying slopes are supported on one fixed-effect dimension only "
                             f"(the streamed one); got {list(self.slopes)}")
        self.stream = "^".join(c.strip() for c in stream.split("^")) if stream else None
        if self.stream is not None and self.stream not in self.fe_user:
            raise ValueError(f"stream={stream!r} is not one of the fixed effects {self.fe_user}")
        if self.slopes and self.stream is not None and self.stream not in self.slopes:
            raise ValueError("varying slopes are only supported on the streamed dimension; "
                             f"slopes are on {list(self.slopes)[0]!r}, stream={self.stream!r}")
        self.slope_vars = next(iter(self.slopes.values())) if self.slopes else []
        self.p = 1 + len(self.slope_vars)
        self.fe_cols = {d: [c.strip() for c in d.split("^")] for d in self.fe_user}
        if models is None:
            rhs = " + ".join(xs) or "1"
            models = [{"fml": f"{yn} ~ {rhs} | {' + '.join(fe)}", "y": yn, "x": list(xs)}
                      for yn in ys]
        self.models = models
        self.base_dir = Path(workdir) if workdir else Path(tempfile.gettempdir())
        if outputs not in ("auto", "keep"):
            raise ValueError("outputs must be 'auto' or 'keep'")
        self.outputs, self.save_resid = outputs, save_resid
        self.keep_intermediates = keep_intermediates
        if solver not in ("auto", "explicit", "stream_cg", "within"):
            raise ValueError("solver must be 'auto', 'explicit', 'stream_cg' or 'within'")
        if precond not in ("jacobi", "amg"):
            raise ValueError("precond must be 'jacobi' or 'amg'")
        if assembly not in ("auto", "cells", "rows"):
            raise ValueError("assembly must be 'auto', 'cells' or 'rows'")
        self.solver, self.precond = solver, precond
        if self.slopes and solver == "within":
            raise ValueError("solver='within' does not support varying slopes; "
                             "use 'auto', 'explicit' or 'stream_cg'")
        if self.slopes and assembly == "cells":
            raise ValueError("varying slopes need assembly='rows'")
        self.assembly = (assembly if assembly != "auto" else
                         ("cells" if self.m <= 8 and not self.slopes else "rows"))
        fe_src = {c for d in self.fe_user for c in self.fe_cols[d]}
        self.keep = [c for c in dict.fromkeys(keep) if c not in fe_src]
        self.tol, self.maxiter = tol, maxiter
        self.batch_rows, self.rgs = batch_rows, row_group_size
        self.n_buckets, self.rows_per_bucket = n_buckets, rows_per_bucket
        self.cells_in_memory = cells_in_memory
        self.triple_budget = triple_budget
        self.dense_max_levels = dense_max_levels
        self.rhs_block = max(1, int(rhs_block))
        self.max_s_gb = max_s_gb if max_s_gb is not None else 0.25 * _phys_mem_gb()
        self.collin_tol, self.collin_tol_rel = collin_tol, collin_tol_rel
        if weights_type not in ("aweights", "fweights"):
            raise ValueError("weights_type must be 'aweights' or 'fweights'")
        self.weights = (None if weights is None else
                        (pl.col(weights) if isinstance(weights, str) else weights).cast(pl.Float64))
        self.weights_name = weights if isinstance(weights, str) else ("<expr>" if weights is not None else None)
        self.weights_type = weights_type
        if n_threads:
            nb.set_num_threads(n_threads)
        self.verbose = verbose
        self.logger, self.log_level = logger, log_level

    def _init_run(self):
        """Create this fit's run directory and the intermediate file paths."""
        self._run = _Run(self.base_dir, self.outputs == "auto")
        self.workdir = p = self._run.path
        self.disk_peak = 0
        self.paths = {
            "maps": {},
            "partitioned": str(p / "rows_partitioned"),
            "rows": [], "cells": [],
            "ident": {"codes": str(p / "ident_codes.u32"), "n": str(p / "ident_n.f64"),
                      "sums": str(p / "ident_sums.f64"), "starts": str(p / "ident_starts.npy"),
                      "st": str(p / "ident_st.f64"), "ainv": str(p / "ident_ainv.f64"),
                      "cmat": str(p / "ident_cmat.f64")},
            "gamma": str(p / "gamma.npy"),
        }

    def _log(self, msg):
        _log(self.verbose, msg, self.logger, self.log_level)

    def _track_disk(self):
        self.disk_peak = max(self.disk_peak, _dir_bytes(self.workdir))

    def _release_intermediates(self):
        """Drop memory maps and delete everything in the run directory
        except the per-model result files."""
        for a in ("codes", "n", "sums", "st", "ainv", "cmat", "starts"):
            if hasattr(self, a):
                setattr(self, a, None)
        if self.keep_intermediates or not self.workdir.exists():
            return
        for child in self.workdir.iterdir():
            if child.name in ("models", _MARKER):
                continue
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                child.unlink(missing_ok=True)

    def _setup_dims(self, g):
        """Fix the streamed dimension g; the others keep the user's order."""
        self.fe = [g] + [d for d in self.fe_user if d != g]
        self.g_fe, self.o_fe = self.fe[0], self.fe[1:]
        self.ccols = [f"c{d}" for d in range(1, len(self.fe))]   # code columns
        self.code_of = {d: cc for d, cc in zip(self.o_fe, self.ccols)}
        self.code_of[self.g_fe] = "gcode"
        self.paths["maps"] = {d: str(self.workdir / f"map_{_safe(d)}.parquet") for d in self.o_fe}

    def _choose_stream(self, approx):
        """Streamed dimension: explicit `stream`, else the dimension with
        varying slopes, else the highest (approximate) cardinality, ties going
        to the dimension listed first."""
        if self.stream is not None:
            return self.stream, "stream= option"
        if self.slopes:
            return next(iter(self.slopes)), "varying slopes"
        best = max(self.fe_user, key=lambda d: (approx[d], -self.fe_user.index(d)))
        return best, "highest cardinality (" + ", ".join(
            f"{d} ~{approx[d]:,}" for d in self.fe_user) + ")"

    # --------------------------------------------------------------- layout
    def _offsets(self):
        """Start offset of each other-dimension block in the stacked vector."""
        off, o = {}, 0
        for f in self.o_fe:
            off[f] = o
            o += self.n_levels[f]
        return off, o

    def _src_cols(self):
        cols = [c for d in self.fe_user for c in self.fe_cols[d]]
        cols += list(self.cluster_cols)
        return list(dict.fromkeys(cols))

    # ------------------------------------------------------------------ pass 0
    def _pass0_code(self, source, keys, extra):
        lf = source if isinstance(source, pl.LazyFrame) else pl.scan_parquet(source)
        schema = lf.collect_schema()
        src = self._src_cols()
        missing = [c for c in src + self.keep + self.slope_vars if c not in schema]
        if missing:
            raise ValueError(f"columns not found in data: {missing}")
        m = self.m
        wexpr = [self.weights.alias("w")] if self.weights is not None else []
        texpr = [pl.col(v).cast(pl.Float64).alias(f"t{j + 1}")
                 for j, v in enumerate(self.slope_vars)]
        finite = [pl.col(f"v{j}").is_finite() for j in range(m)]
        finite += [pl.col(f"t{j + 1}").is_finite() for j in range(len(self.slope_vars))]
        if self.weights is not None:
            finite.append(pl.col("w").is_finite())
        lf = (lf.select([pl.col(c) for c in src + self.keep] + wexpr + texpr
                        + [self.var_exprs[nm].alias(f"v{j}") for j, nm in enumerate(self.var_names)])
                .drop_nulls(src)
                .filter(pl.all_horizontal(finite)))

        # One scan for counts, means, weight checks and (if the streamed
        # dimension must be chosen by cardinality) HyperLogLog counts.
        wstats = ([pl.col("w").sum().alias("wsum"), pl.col("w").min().alias("wmin")]
                  if self.weights is not None else [])
        tstats = [pl.col(f"t{j + 1}").mean() for j in range(len(self.slope_vars))]
        need_card = self.stream is None and not self.slopes
        card = []
        if need_card:
            for d in self.fe_user:
                cols = self.fe_cols[d]
                key = pl.col(cols[0]) if len(cols) == 1 else pl.struct(cols).hash(seed=7)
                card.append(key.approx_n_unique().alias(f"__card_{d}"))
        self._log("pass 0: scanning data (counts, means"
                  + (", approximate cardinalities)" if need_card else ")"))
        mom = (lf.select([pl.len().alias("n")] + wstats + tstats + card
                         + [pl.col(f"v{j}").mean() for j in range(m)]
                         + [pl.col(f"v{j}").var().alias(f"var{j}") for j in range(m)])
                 .collect(engine="streaming"))
        approx = {d: mom[f"__card_{d}"].item() for d in self.fe_user} if need_card else {}
        g, why = self._choose_stream(approx)
        self._setup_dims(g)
        self.stream_choice = {"dim": g, "reason": why}
        self._log(f"pass 0: streaming {g} ({why})")
        self._resolve_clusters(keys, extra)
        self.tmeans = np.array([mom[f"t{j + 1}"].item() for j in range(len(self.slope_vars))])
        self.n_obs = mom["n"].item()
        if self.n_obs == 0:
            raise ValueError("no complete observations")
        if self.weights is not None and mom["wmin"].item() <= 0:
            raise ValueError("weights must be strictly positive")
        self.wsum = mom["wsum"].item() if self.weights is not None else float(self.n_obs)
        self.N = self.wsum if self.weights_type == "fweights" else self.n_obs
        self.means = np.array([mom[f"v{j}"].item() for j in range(m)])
        self.raw_ss = np.array([(mom[f"var{j}"].item() or 0.0) * max(self.n_obs - 1, 1)
                                for j in range(m)])
        P = self.n_buckets or max(1, -(-self.n_obs // self.rows_per_bucket))
        self.n_buckets = P

        # Codes for non-streamed dimensions and extra cluster variables:
        # one level-sized hash table each.
        self.n_levels = {}
        maps = [(d, self.fe_cols[d], cc, self.paths["maps"][d])
                for d, cc in zip(self.o_fe, self.ccols)]
        maps += [(name, cm["cols"], cm["code"], cm["map"]) for name, cm in self.cmaps.items()]
        coded = lf
        for name, cols, cc, path in maps:
            self._log(f"pass 0: factorizing {name}")
            lf.select(cols).unique().sort(cols).with_row_index(cc).sink_parquet(path)
            nl = pl.scan_parquet(path).select(pl.len()).collect().item()
            if name in self.o_fe:
                self.n_levels[name] = nl
            else:
                self.cmaps[name]["G"] = nl
            coded = coded.join(pl.scan_parquet(path), on=cols)

        # Hash-partition rows by fe[0]: every group lands in exactly one bucket.
        g_cols = self.fe_cols[self.g_fe]
        key = pl.col(g_cols[0]) if len(g_cols) == 1 else pl.struct(g_cols)
        self._log(f"pass 0: hash-partitioning rows into {P} {self.g_fe} buckets")
        shutil.rmtree(self.paths["partitioned"], ignore_errors=True)
        (coded.with_columns(_bucket=(key.hash(seed=20260921) % P).cast(pl.UInt32))
              .sink_parquet(pl.PartitionBy(self.paths["partitioned"], key="_bucket",
                                           include_key=False)))

        # Sort each bucket by fe[0] and assign dense codes that keep
        # increasing across buckets.
        self._log(f"pass 0: sorting buckets and assigning {self.g_fe} codes")
        new_group = pl.any_horizontal([(pl.col(c) != pl.col(c).shift()).fill_null(True)
                                       for c in g_cols])
        offset = 0
        self.paths["rows"] = []
        for bkt in range(P):
            srcdir = Path(self.paths["partitioned"]) / f"_bucket={bkt}"
            if not srcdir.exists():
                continue
            out = str(self.workdir / f"rows_b{bkt:04d}.parquet")
            (pl.scan_parquet(str(srcdir / "*.parquet"))
               .sort(g_cols)
               .with_columns(gcode=(new_group.cast(pl.UInt32).cum_sum() - 1 + offset)
                             .cast(pl.UInt32))
               .sink_parquet(out, row_group_size=self.rgs))
            offset = pl.scan_parquet(out).select(pl.col("gcode").max()).collect().item() + 1
            self.paths["rows"].append(out)
        self.n_levels = {self.g_fe: offset, **self.n_levels}
        self._track_disk()
        shutil.rmtree(self.paths["partitioned"], ignore_errors=True)
        for spec in self.clusters.values():
            if spec["kind"] in ("seg", "fe"):
                spec["G"] = self.n_levels[spec["dim"]]
            elif spec["kind"] == "extra":
                spec["G"] = self.cmaps[spec["map"]]["G"]
            else:                                   # segsub: counted in pass 2
                spec["G"] = None
                spec["xL"] = (self.n_levels[spec["xdim"]] if spec["xdim"] is not None
                              else self.cmaps[spec["map"]]["G"])

    # ------------------------------------------------------------------ pass 1
    def _pass1_cells(self):
        """group_by(fe[0], other codes) -> cell table with counts and sums of
        the (globally centered) variables, plus within-cell cross-products
        when assembly == 'cells'."""
        self._log(f"pass 1: group_by({self.g_fe}, {', '.join(self.o_fe)}) -> cells")
        m = self.m
        z = [(pl.col(f"v{j}") - mu).alias(f"z{j}") for j, mu in enumerate(self.means)]
        weighted = self.weights is not None
        if weighted:
            z.append(pl.col("w"))
        wz = (lambda e: pl.col("w") * e) if weighted else (lambda e: e)
        # "n" is the cell's weight sum (its count when unweighted); "cnt" the count
        aggs = [(pl.col("w").sum() if weighted else pl.len().cast(pl.Float64)).alias("n"),
                pl.len().cast(pl.UInt32).alias("cnt")]
        aggs += [wz(pl.col(f"z{j}")).sum().alias(f"s{j}") for j in range(m)]
        if self.assembly == "cells":
            aggs += [wz(pl.col(f"z{i}") * pl.col(f"z{j}")).sum().alias(f"cp{i}_{j}")
                     for i in range(m) for j in range(i, m)]
        ps = len(self.slope_vars)
        if ps:
            # slope variables centered at their means (keeps the per-group
            # p x p systems well conditioned); moments needed for the
            # within-group regression on [1, slopes]
            z += [(pl.col(f"t{j + 1}") - mu).alias(f"u{j + 1}") for j, mu in enumerate(self.tmeans)]
            aggs += [wz(pl.col(f"u{a}")).sum().alias(f"sl{a}") for a in range(1, ps + 1)]
            aggs += [wz(pl.col(f"u{a}") * pl.col(f"u{b}")).sum().alias(f"sll{a}_{b}")
                     for a in range(1, ps + 1) for b in range(a, ps + 1)]
            aggs += [wz(pl.col(f"u{a}") * pl.col(f"z{j}")).sum().alias(f"slz{a}_{j}")
                     for a in range(1, ps + 1) for j in range(m)]
        self.paths["cells"] = []
        for rows in self.paths["rows"]:
            out = rows.replace("rows_b", "cells_b")
            (pl.scan_parquet(rows)
               .select("gcode", *self.ccols, *z)
               .group_by("gcode", *self.ccols)
               .agg(aggs)
               .sort("gcode", *self.ccols)
               .sink_parquet(out, row_group_size=self.rgs))
            self.paths["cells"].append(out)

        exprs = [pl.len().alias("n_cells")]
        if self.assembly == "cells":
            exprs += [(pl.col(f"cp{i}_{j}") - pl.col(f"s{i}") * pl.col(f"s{j}") / pl.col("n"))
                      .sum().alias(f"w{i}_{j}") for i in range(m) for j in range(i, m)]
        w = pl.scan_parquet(self.paths["cells"]).select(exprs).collect(engine="streaming")
        self.n_cells = w["n_cells"].item()
        if self.assembly == "cells":
            W = np.zeros((m, m))
            for i in range(m):
                for j in range(i, m):
                    W[i, j] = W[j, i] = w[f"w{i}_{j}"].item()
            self.W_within_cell = W

    # ----------------------------------------------------------------- pass 1b
    def _pass1b_identifying(self):
        """Keep cells of fe[0] groups with more than one cell (the only ones
        that identify the other effects) as flat binary arrays for memory
        mapping, and build the Jacobi diagonal."""
        self._log("pass 1b: extracting identifying cells")
        m = self.m
        off, tot = self._offsets()
        scols = [f"s{j}" for j in range(m)]
        obs = np.zeros(tot)             # weight per level (all cells)
        cnt = np.zeros(tot)             # observations per level (all cells)
        sizes = []
        ps, p = len(self.slope_vars), self.p
        slcols = ([f"sl{a}" for a in range(1, ps + 1)]
                  + [f"sll{a}_{b}" for a in range(1, ps + 1) for b in range(a, ps + 1)]
                  + [f"slz{a}_{j}" for a in range(1, ps + 1) for j in range(m)])
        kinds = ("codes", "n", "sums") + (("st", "ainv", "cmat") if ps else ())
        files = {k: open(self.paths["ident"][k], "wb") for k in kinds}
        self.fe0_rank = 0               # fe[0] parameters: sum of per-group ranks
        for ch in iter_group_chunks(self.paths["cells"],
                                    ["gcode", *self.ccols, "n", "cnt", *scols, *slcols],
                                    self.batch_rows):
            for f, cc in zip(self.o_fe, self.ccols):
                sl = slice(off[f], off[f] + self.n_levels[f])
                obs[sl] += np.bincount(ch[cc], weights=ch["n"], minlength=self.n_levels[f])
                cnt[sl] += np.bincount(ch[cc], weights=ch["cnt"], minlength=self.n_levels[f])
            starts, local = _segments(ch["gcode"])
            ncells = np.diff(np.append(starts, len(local)))
            keep = ncells[local] > 1
            if ps:
                # per cell: st = sum w [1, u]; stt = sum w [1,u][1,u]'; stz = sum w [1,u] z'
                Mc = len(local)
                st = np.empty((Mc, p))
                st[:, 0] = ch["n"]
                for a in range(1, p):
                    st[:, a] = ch[f"sl{a}"]
                stt = np.empty((Mc, p, p))
                stt[:, 0, :] = st
                stt[:, :, 0] = st
                for a in range(1, p):
                    for b in range(a, p):
                        stt[:, a, b] = stt[:, b, a] = ch[f"sll{a}_{b}"]
                stz = np.empty((Mc, p, m))
                for j in range(m):
                    stz[:, 0, j] = ch[f"s{j}"]
                    for a in range(1, p):
                        stz[:, a, j] = ch[f"slz{a}_{j}"]
                A = np.add.reduceat(stt, starts, axis=0)
                self.fe0_rank += int(np.linalg.matrix_rank(A, hermitian=True).sum())
            else:
                self.fe0_rank += len(starts)
            if not keep.any():
                continue
            sizes.append(ncells[ncells > 1])
            np.column_stack([ch[cc][keep] for cc in self.ccols]).astype(np.uint32).tofile(files["codes"])
            ch["n"][keep].astype(np.float64).tofile(files["n"])
            np.column_stack([ch[c][keep] for c in scols]).tofile(files["sums"])
            if ps:
                gk = ncells > 1
                st[keep].tofile(files["st"])
                np.linalg.pinv(A[gk], rcond=1e-12, hermitian=True).tofile(files["ainv"])
                np.add.reduceat(stz, starts, axis=0)[gk].tofile(files["cmat"])
        for fh in files.values():
            fh.close()
        sizes = np.concatenate(sizes) if sizes else np.zeros(0, np.int64)
        np.save(self.paths["ident"]["starts"],
                np.concatenate(([0], np.cumsum(sizes))).astype(np.int64))
        self.obs, self.cnt, self.n_identifying = obs, cnt, len(sizes)
        self._track_disk()
        if not self.keep_intermediates:         # cell tables are no longer needed
            for f in self.paths["cells"]:
                Path(f).unlink(missing_ok=True)
        self._load_ident()
        self.diag = np.zeros(tot)
        if ps:
            _nb_diag_sl(self.starts, self.codes, self.n, self.st, self.ainv, self.offs, self.diag)
        else:
            _nb_diag(self.starts, self.codes, self.n, self.offs, self.diag)
        self.fe_params = {self.g_fe: self.fe0_rank,
                          **{d: self.n_levels[d] for d in self.o_fe}}

    def _load_ident(self):
        """Memory-map the identifying cells (or load them, if cells_in_memory)."""
        m, D = self.m, len(self.o_fe)
        self.starts = np.load(self.paths["ident"]["starts"])
        M = int(self.starts[-1])
        load = np.array if self.cells_in_memory else (lambda a: a)

        def mm(key, dtype, shape):
            if M == 0:
                return np.zeros(shape, dtype)
            return load(np.memmap(self.paths["ident"][key], dtype=dtype, mode="r", shape=shape))
        self.codes = mm("codes", np.uint32, (M, D))
        self.n = mm("n", np.float64, (M,))
        self.sums = mm("sums", np.float64, (M, m))
        if self.slope_vars:
            p, G = self.p, len(self.starts) - 1
            self.st = mm("st", np.float64, (M, p))
            self.ainv = (load(np.memmap(self.paths["ident"]["ainv"], dtype=np.float64, mode="r",
                                        shape=(G, p, p))) if G else np.zeros((0, p, p)))
            self.cmat = (load(np.memmap(self.paths["ident"]["cmat"], dtype=np.float64, mode="r",
                                        shape=(G, p, m))) if G else np.zeros((0, p, m)))
        off, _ = self._offsets()
        self.offs = np.array([off[f] for f in self.o_fe], np.int64)
        self.n_ident_cells = M

    # ----------------------------------------------------------- components
    def _components(self):
        """Connected components of the (fe[0], fe[1]) graph by union-find.
        Only the first two dimensions get exact redundancy accounting; any
        further dimension contributes one more restriction (see `fe_dof`)."""
        self._log("computing connected components")
        lab = _nb_components(self.starts, self.codes, self.n_levels[self.o_fe[0]])
        self.comp = lab
        self.n_components = int(np.count_nonzero(lab == np.arange(len(lab))))

    # --------------------------------------------------------------- step 2
    def _pcg(self, matvec, precond, b):
        """Block PCG: one independent CG per right-hand-side column, run in
        lockstep so every operator application covers all columns."""
        bnorm = np.linalg.norm(b, axis=0)
        bnorm[bnorm == 0] = 1.0
        X = np.zeros_like(b)
        R = b.copy()
        Z = precond(R)
        P = Z.copy()
        rz = np.einsum("ij,ij->j", R, Z)
        active = np.ones(b.shape[1], bool)
        rel = np.ones(b.shape[1])
        it = 0
        for it in range(1, self.maxiter + 1):
            AP = matvec(P)
            pAp = np.einsum("ij,ij->j", P, AP)
            alpha = np.where(active & (pAp > 0), rz / np.where(pAp > 0, pAp, 1), 0.0)
            X += P * alpha
            R -= AP * alpha
            rel = np.linalg.norm(R, axis=0) / bnorm
            active = rel > self.tol
            if self.verbose and (it % 25 == 0 or not active.any()):
                self._log(f"  CG iter {it}: max rel. residual {rel.max():.2e}")
            if not active.any():
                break
            Z = precond(R)
            rz_new = np.einsum("ij,ij->j", R, Z)
            beta = np.where(active, rz_new / np.where(rz != 0, rz, 1), 0.0)
            P = Z + P * beta
            rz = rz_new
        return X, it, bool(not active.any()), float(rel.max())

    def _jacobi(self):
        pos = self.diag > 1e-12
        dinv = np.where(pos, 1.0 / np.where(pos, self.diag, 1.0), 0.0)
        return lambda R: R * dinv[:, None]

    def _build_explicit(self):
        """Form S = D_o' M_0 D_o as a CSR matrix in one parallel pass.

        Entries between two levels of *small* dimensions (at most
        `dense_max_levels` levels each, e.g. years) are accumulated in a dense
        block; everything else is emitted as COO triples. Groups are processed
        in chunks whose triples fit in max(triple_budget, nnz(S so far)); each
        chunk is converted to CSR (summing duplicates) and added to S, which
        keeps the total merge cost linear in the number of triples. Raises
        _STooLarge as soon as S exceeds `max_s_gb`.
        """
        off, L = self._offsets()
        G = len(self.starts) - 1
        doffs, nd = [], 0
        for f in self.o_fe:
            if self.n_levels[f] <= self.dense_max_levels and nd + self.n_levels[f] <= 4096:
                doffs.append(nd)
                nd += self.n_levels[f]
            else:
                doffs.append(-1)
        doffs = np.array(doffs, np.int64)
        nt = nb.get_num_threads()
        dense = np.zeros((nt, max(nd, 1), max(nd, 1)))
        cap = self.max_s_gb * 1e9

        def nbytes(S):
            return S.data.nbytes + S.indices.nbytes + S.indptr.nbytes

        S = sp.csr_matrix((L, L))
        t0 = time.time()
        n_triples = 0
        block = 2_000_000                  # groups per counting block
        for b0 in range(0, G, block):
            b1 = min(G, b0 + block)
            cnt = np.empty(b1 - b0, np.int64)
            _nb_count_triples(self.starts, self.codes, self.n, self.offs, doffs, b0, b1, cnt)
            cum = np.concatenate(([0], np.cumsum(cnt)))
            g0 = b0
            while g0 < b1:
                budget = max(self.triple_budget, S.nnz)
                base = cum[g0 - b0]
                g1 = b0 + int(np.searchsorted(cum, base + budget, side="right")) - 1
                g1 = min(max(g1, g0 + 1), b1)
                toff = cum[g0 - b0:g1 - b0 + 1] - base
                mt = int(toff[-1])
                R = np.empty(mt, np.int32)
                C = np.empty(mt, np.int32)
                V = np.empty(mt)
                if self.slope_vars:
                    _nb_emit_triples_sl(self.starts, self.codes, self.n, self.st, self.ainv,
                                        self.offs, doffs, g0, g1, toff, R, C, V, dense)
                else:
                    _nb_emit_triples(self.starts, self.codes, self.n, self.offs, doffs,
                                     g0, g1, toff, R, C, V, dense)
                if mt:
                    S = S + sp.coo_matrix((V, (R, C)), shape=(L, L)).tocsr()
                del R, C, V
                n_triples += mt
                g0 = g1
                if nbytes(S) > cap:
                    raise _STooLarge(
                        f"S exceeded max_s_gb={self.max_s_gb:.1f} after {g1:,} of {G:,} "
                        f"groups ({S.nnz:,} nonzeros)")
        if nd:
            dsum = dense.sum(axis=0)
            gidx = np.concatenate([off[f] + np.arange(self.n_levels[f])
                                   for f, d in zip(self.o_fe, doffs) if d >= 0])
            rr, cc = np.nonzero(dsum)
            S = S + sp.coo_matrix((dsum[rr, cc], (gidx[rr], gidx[cc])), shape=(L, L)).tocsr()
        S.sum_duplicates()
        S.sort_indices()
        mb = nbytes(S) / 1e6
        self._log(f"  explicit S: {L:,} x {L:,}, nnz {S.nnz:,} ({mb:,.0f} MB), "
                           f"{n_triples:,} triples + {nd}x{nd} dense block, "
                           f"{time.time()-t0:.1f}s")
        return S, {"S_nnz": int(S.nnz), "S_mb": round(mb, 1),
                   "S_build_seconds": round(time.time() - t0, 2)}

    def _make_block_solver(self):
        """Return (solve(b) -> (X, iterations, converged, rel), info dict)."""
        solver = self.solver
        info = {}
        if solver in ("auto", "explicit"):
            try:
                S, sinfo = self._build_explicit()
                info.update(sinfo)
                solver = "explicit"
            except _STooLarge as err:
                if self.solver == "explicit":
                    raise MemoryError(f"{err}; raise max_s_gb or use solver='stream_cg'") from None
                self._log(f"  {err}; falling back to stream_cg")
                info["fallback"] = str(err)
                solver = "stream_cg"
        info["solver"] = solver

        if solver == "explicit":
            indptr, indices, data = S.indptr, S.indices, S.data

            def matvec(P):
                out = np.empty_like(P)
                _nb_csr_matmat(indptr, indices, data, np.ascontiguousarray(P), out)
                return out
            if self.precond == "amg":
                import pyamg
                t0 = time.time()
                ml = pyamg.smoothed_aggregation_solver(S, symmetry="symmetric", max_coarse=500)
                M = ml.aspreconditioner(cycle="V")
                info["amg_setup_seconds"] = round(time.time() - t0, 2)
                info["solver"] = "explicit+amg"

                def precond(R):
                    return np.column_stack([M @ R[:, j] for j in range(R.shape[1])])
            else:
                precond = self._jacobi()
            return (lambda b: self._pcg(matvec, precond, b)), info

        if solver == "stream_cg":
            nt = nb.get_num_threads()
            L = self.diag.shape[0]

            def matvec_s(P):
                acc = np.empty((nt, L, P.shape[1]))
                if self.slope_vars:
                    _nb_stream_matvec_sl(np.ascontiguousarray(P), self.starts, self.codes, self.n,
                                         self.st, self.ainv, self.offs, acc)
                else:
                    _nb_stream_matvec(np.ascontiguousarray(P), self.starts, self.codes, self.n,
                                      self.offs, acc)
                return acc.sum(axis=0)
            jac = self._jacobi()
            return (lambda b: self._pcg(matvec_s, jac, b)), info

        # within: full system on identifying cells, keep the non-streamed blocks
        import within
        sizes = np.diff(self.starts)
        design = np.asfortranarray(np.column_stack(
            [np.repeat(np.arange(len(sizes), dtype=np.uint32), sizes),
             np.asarray(self.codes)]).astype(np.uint32))
        nvec = np.asarray(self.n)
        off, tot = self._offsets()

        def solve_within(j0, j1):
            Y = np.asarray(self.sums[:, j0:j1]) / nvec[:, None]
            res = within.solve_batch(design, Y, weights=nvec,
                                     options=within.LsmrOptions(tol=self.tol, maxiter=self.maxiter))
            x, lay = np.asarray(res.x), res.layout
            Gm = np.zeros((tot, j1 - j0))
            for t, f in enumerate(self.o_fe, start=1):
                start, nl_seen = lay.index(t, 0, 0), lay.n_levels(t)
                Gm[off[f]:off[f] + nl_seen] = x[start:start + nl_seen]
            return Gm, list(res.iterations), bool(all(res.converged)), float("nan")
        return solve_within, info

    def _solve(self):
        """Solve for Gamma (levels x variables) in column blocks; the result
        is written to a memory-mapped .npy file."""
        off, L = self._offsets()
        m = self.m
        t0 = time.time()
        solve, info = self._make_block_solver()
        gamma = np.lib.format.open_memmap(self.paths["gamma"], mode="w+",
                                          dtype=np.float64, shape=(L, m))
        its, conv, rel = [], True, 0.0
        for j0 in range(0, m, self.rhs_block):
            j1 = min(m, j0 + self.rhs_block)
            if info["solver"] == "within":
                X, it, c, r = solve(j0, j1)
            else:
                b = np.zeros((L, j1 - j0))
                if self.slope_vars:
                    _nb_rhs_sl(self.starts, self.codes, self.sums[:, j0:j1], self.st, self.ainv,
                               self.cmat[:, :, j0:j1], self.offs, b)
                else:
                    _nb_rhs(self.starts, self.codes, self.n, self.sums[:, j0:j1], self.offs, b)
                X, it, c, r = solve(b)
            gamma[:, j0:j1] = self._normalize(X)
            its.append(it)
            conv &= c
            rel = max(rel, r)
        gamma.flush()
        info.update({"iterations": its if len(its) > 1 else its[0], "converged": conv,
                     "blocks": len(its), "seconds": round(time.time() - t0, 2)})
        if not np.isnan(rel):
            info["max_rel_residual"] = rel
        if not conv:
            _warn(f"solver did not converge within maxiter={self.maxiter}", self.logger)
        return np.load(self.paths["gamma"], mmap_mode="r"), info

    def _normalize(self, Gamma):
        """fe[1] effects have obs-weighted mean zero within each connected
        component; further dimensions have global obs-weighted mean zero. The
        constants move into the fe[0] effects; fitted values are unchanged."""
        off, _ = self._offsets()
        f1 = self.o_fe[0]
        nl, o = self.n_levels[f1], off[f1]
        w = self.obs[o:o + nl]
        cw = np.bincount(self.comp, weights=w, minlength=nl)
        cw[cw == 0] = 1.0
        mean = _scatter(self.comp, w[:, None] * Gamma[o:o + nl], nl) / cw[:, None]
        Gamma[o:o + nl] -= mean[self.comp]
        for f in self.o_fe[1:]:
            nl, o = self.n_levels[f], off[f]
            w = self.obs[o:o + nl]
            Gamma[o:o + nl] -= (w @ Gamma[o:o + nl]) / max(w.sum(), 1.0)
        return Gamma

    # --------------------------------------------------------------- step 3
    def _assemble(self, gamma):
        """V' M_D V for all variables (m x m)."""
        m, nt = self.m, nb.get_num_threads()
        acc = np.zeros((nt, m, m))
        if self.assembly == "cells":
            _nb_assemble(self.starts, self.codes, self.n, self.sums, self.offs, gamma, acc)
            return self.W_within_cell + acc.sum(axis=0)
        self._log("step 3: row pass for V' M_D V")
        wcol = ["w"] if self.weights is not None else []
        tcols = [f"t{j + 1}" for j in range(len(self.slope_vars))]
        cols = ["gcode", *self.ccols, *wcol, *tcols, *[f"v{j}" for j in range(m)]]
        for ch in iter_group_chunks(self.paths["rows"], cols, self.batch_rows):
            gc = ch["gcode"]
            starts = np.concatenate(([0], np.flatnonzero(gc[1:] != gc[:-1]) + 1, [len(gc)]))
            codes = np.column_stack([ch[cc] for cc in self.ccols]).astype(np.int64)
            V = np.column_stack([ch[f"v{j}"] for j in range(m)]).astype(np.float64)
            w = (np.ascontiguousarray(ch["w"], dtype=np.float64) if wcol
                 else np.ones(len(gc)))
            if tcols:
                _nb_assemble_rows_sl(starts, codes, self.offs, w, self._slope_matrix(ch), V,
                                     gamma, acc)
            else:
                _nb_assemble_rows(starts, codes, self.offs, w, V, gamma, acc)
        return acc.sum(axis=0)

    def _slope_matrix(self, ch):
        """T = [1, centered slope variables] for a chunk of rows."""
        n = len(ch["gcode"])
        T = np.empty((n, self.p))
        T[:, 0] = 1.0
        for j, mu in enumerate(self.tmeans):
            T[:, j + 1] = ch[f"t{j + 1}"] - mu
        return T

    # ------------------------------------------------------- clusters / dof
    def _resolve_clusters(self, keys, extra):
        """Turn cluster requests into cluster *terms*.

        A one-way request 'a' is one term; a multi-way request 'a+b' expands
        to the terms a, b and a^b (and all higher intersections for three or
        more ways), combined by inclusion-exclusion (Cameron, Gelbach and
        Miller 2011), as in pyfixest/fixest. Each term is one of:
          seg     the fe[0] groups themselves: scores summed per group
          segsub  a refinement of fe[0] (e.g. pik^sein): rows of a cluster
                  are always in the same chunk, so scores are summed per
                  (group, code) inside each chunk; no global array
          fe      another FE dimension: its codes, level-sized accumulator
          extra   anything else: factorized in pass 0, level-sized accumulator
        """
        reqs = [k.split(":", 1)[1] for k in keys if k.startswith("CRV1:")]
        reqs += [c for c in extra if c not in reqs]
        g0 = set(self.fe_cols[self.g_fe])
        dim_of = {frozenset(self.fe_cols[d]): d for d in self.o_fe}
        self.clusters, self.cmaps, self.cluster_reqs = {}, {}, {}

        def cmap(cols):
            name = "^".join(cols)
            if name not in self.cmaps:
                self.cmaps[name] = {"cols": list(cols), "code": f"k{len(self.cmaps)}",
                                    "map": str(self.workdir / f"cluster_{_safe(name)}.parquet")}
            return name

        def term(cols):
            key = "^".join(cols)
            if key in self.clusters:
                return key
            cs = set(cols)
            if cs == g0:
                spec = {"kind": "seg", "dim": self.g_fe, "code": "gcode"}
            elif cs > g0:
                xcols = [c for c in cols if c not in g0]
                xdim = dim_of.get(frozenset(xcols))
                spec = {"kind": "segsub", "xdim": xdim,
                        "code": self.code_of[xdim] if xdim else None,
                        "map": None if xdim else cmap(xcols)}
                if not xdim:
                    spec["code"] = self.cmaps[spec["map"]]["code"]
            elif frozenset(cols) in dim_of:
                d = dim_of[frozenset(cols)]
                spec = {"kind": "fe", "dim": d, "code": self.code_of[d]}
            else:
                name = cmap(cols)
                spec = {"kind": "extra", "map": name, "code": self.cmaps[name]["code"]}
            spec["cols"] = list(cols)
            self.clusters[key] = spec
            return key

        from itertools import combinations
        for req in reqs:
            parts = [[c.strip() for c in p.split("^")] for p in req.split("+")]
            terms = []
            for r in range(1, len(parts) + 1):
                for combo in combinations(range(len(parts)), r):
                    cols = list(dict.fromkeys(c for i in combo for c in parts[i]))
                    terms.append((term(cols), (-1) ** (r + 1), r == 1))
            self.cluster_reqs[req] = terms

    def _nested_dims(self, key):
        """FE dimensions whose every level lies within a single cluster of
        this term (not counted in K for CRV1, as in fixest/pyfixest)."""
        spec = self.clusters[key]
        ccols = set(spec["cols"])
        out = []
        for d in self.fe:
            if ccols <= set(self.fe_cols[d]):
                out.append(d)
                continue
            if spec["kind"] == "segsub":    # no single code column; subset rule only
                continue
            # a level is inside one cluster iff min == max of the cluster code
            # over its rows; min/max keep the aggregation state O(1) per level
            ccol, dcol = spec["code"], self.code_of[d]
            q = [pl.col(ccol).min().alias("lo"), pl.col(ccol).max().alias("hi")]
            chk = (pl.col("lo") != pl.col("hi")).any()
            if d == self.g_fe:      # bucket by bucket: rows are bucketed by fe[0]
                ok = not any(pl.scan_parquet(f).group_by("gcode").agg(q).select(chk)
                             .collect(engine="streaming").item() for f in self.paths["rows"])
            else:
                ok = not (pl.scan_parquet(self.paths["rows"]).group_by(dcol).agg(q).select(chk)
                          .collect(engine="streaming").item())
            if ok:
                out.append(d)
        return out

    def _redundant_slopes(self):
        """Slope variables that are a function of another FE dimension (e.g.
        worker trends in `year` alongside year effects). The worker slopes on
        such a variable sum to a trend that the other dimension already
        spans, so one of them is redundant; each counts as one more
        restriction in the FE degrees of freedom."""
        out = []
        for j, v in enumerate(self.slope_vars):
            for d in self.o_fe:
                spread = (pl.scan_parquet(self.paths["rows"])
                          .group_by(self.code_of[d])
                          .agg((pl.col(f"t{j + 1}").max() - pl.col(f"t{j + 1}").min()).alias("r"))
                          .select(pl.col("r").max()).collect(engine="streaming").item())
                if spread == 0:
                    out.append((v, d))
                    break
        self.slope_redundancy = out
        return out

    def _collinear(self, Axx, idx=None):
        """pyfixest's collinearity check on the residualized X'X, plus a
        relative check: a variable whose residual sum of squares is below
        collin_tol_rel of its raw (centered) sum of squares is treated as
        absorbed by the fixed effects. The iterative solve leaves noise of
        order sqrt(tol * condition number) in exactly collinear variables,
        so an absolute test alone can miss them."""
        k = Axx.shape[0]
        if k == 0:
            return np.zeros(0, bool)
        mask = self._collinear_pf(Axx)
        if idx is not None:
            raw = self.raw_ss[idx]
            mask |= (raw > 0) & (np.diag(Axx) <= self.collin_tol_rel * raw)
            if mask.any() and not mask.all():   # re-check the rest without them
                keep = np.flatnonzero(~mask)
                mask[keep] = self._collinear_pf(Axx[np.ix_(keep, keep)])
        return mask

    def _collinear_pf(self, Axx):
        k = Axx.shape[0]
        try:
            from pyfixest.core import find_collinear_variables
            mask, _, all_removed = find_collinear_variables(np.ascontiguousarray(Axx),
                                                            self.collin_tol)
            mask = np.asarray(mask, bool)
        except ImportError:   # fallback: sequential pivot test on the Cholesky
            mask = np.zeros(k, bool)
            for j in range(k):
                keep = np.flatnonzero(~mask[:j])
                r = Axx[j, j]
                if len(keep):
                    r -= Axx[j, keep] @ np.linalg.solve(Axx[np.ix_(keep, keep)], Axx[keep, j])
                mask[j] = r <= self.collin_tol * max(Axx[j, j], 1e-300)
        return mask

    # --------------------------------------------------------------- step 4
    def _pass2(self, tag, ycol_name, gamma, yj, xk, beta, zk, Pi):
        """Row pass for one model: fe[0] effects, residuals, moments, meats;
        writes the residual file and one FE file per dimension.

        xk: regressor columns (residuals use X beta); zk, Pi: instrument
        columns and first-stage coefficients, so the scores use Pi' z~
        (for OLS, zk = xk and Pi = I).
        """
        mdir = self.workdir / "models" / tag
        mdir.mkdir(parents=True, exist_ok=True)
        paths = {"dir": str(mdir),
                 "resid": str(mdir / "resid.parquet") if self.save_resid else None,
                 "fe": {d: str(mdir / f"fe_{_safe(d)}.parquet") for d in self.fe}}
        off, _ = self._offsets()
        k, q, nt = len(xk), len(zk), nb.get_num_threads()
        L = gamma.shape[0]
        gam_x = np.ascontiguousarray(gamma[:, xk]) if k else np.zeros((L, 0))
        gam_z = np.ascontiguousarray(gamma[:, zk]) if q else np.zeros((L, 0))
        gam_y0 = np.ascontiguousarray(gamma[:, yj])
        gam_y = gam_y0 - gam_x @ beta                  # FEs of the y-equation
        beta_c = np.ascontiguousarray(beta, dtype=np.float64)
        Pi_c = np.ascontiguousarray(Pi, dtype=np.float64).reshape(q, k)
        fweights = self.weights is not None and self.weights_type == "fweights"
        weighted = self.weights is not None

        acc_s = np.zeros((nt, 4))
        acc_B = np.zeros((nt, k, k))
        acc_hc = np.zeros((nt, k, k))
        acc_g = np.zeros((nt, k, k))
        scores = {t: np.zeros((spec["G"], k)) for t, spec in self.clusters.items()
                  if spec["kind"] in ("fe", "extra")}
        subs = {t: spec for t, spec in self.clusters.items() if spec["kind"] == "segsub"}
        meat_sub = {t: np.zeros((k, k)) for t in subs}
        G_sub = {t: 0 for t in subs}
        ccodes = [spec["code"] for spec in self.clusters.values() if spec["kind"] != "seg"]
        src = [c for c in self._src_cols() if c not in self.keep]
        tcols = [f"t{j + 1}" for j in range(len(self.slope_vars))]
        cols = list(dict.fromkeys(["gcode", *self.ccols, *ccodes, *src, *self.keep, *tcols,
                                   *(["w"] if weighted else []), f"v{yj}",
                                   *[f"v{j}" for j in xk], *[f"v{j}" for j in zk]]))
        g_cols = self.fe_cols[self.g_fe]
        yc = float(self.means[yj])
        r_writer = g_writer = None
        for ch in iter_group_chunks(self.paths["rows"], cols, self.batch_rows):
            gc = ch["gcode"]
            starts = np.concatenate(([0], np.flatnonzero(gc[1:] != gc[:-1]) + 1, [len(gc)]))
            codes = np.column_stack([ch[cc] for cc in self.ccols]).astype(np.int64)
            y = np.ascontiguousarray(ch[f"v{yj}"], dtype=np.float64)
            n = len(y)
            w = np.ascontiguousarray(ch["w"], dtype=np.float64) if weighted else np.ones(n)
            X = (np.column_stack([ch[f"v{j}"] for j in xk]).astype(np.float64) if k
                 else np.zeros((n, 0)))
            Z = (np.column_stack([ch[f"v{j}"] for j in zk]).astype(np.float64) if q
                 else np.zeros((n, 0)))
            G = len(starts) - 1
            e = np.empty(n)
            h = np.empty((n, k))
            if tcols:
                T = self._slope_matrix(ch)
                g_coef = np.empty((G, self.p))
                _nb_pass2_sl(starts, codes, self.offs, w, T, y, X, beta_c, gam_y, gam_y0, Z,
                             gam_z, Pi_c, yc, fweights, g_coef, e, h, acc_s, acc_B, acc_hc, acc_g)
                # back to the original scale of the slope variables
                g_eff = g_coef[:, 0] - g_coef[:, 1:] @ self.tmeans
                g_row = np.einsum("ij,ij->i", T, np.repeat(g_coef, np.diff(starts), axis=0))
            else:
                g_eff = np.empty(G)
                _nb_pass2(starts, codes, self.offs, w, y, X, beta_c, gam_y, gam_y0, Z, gam_z,
                          Pi_c, yc, fweights, g_eff, e, h, acc_s, acc_B, acc_hc, acc_g)
                g_row = None
            Ng = np.diff(starts)
            if scores or subs:
                he = h * (w * e)[:, None]
                for t, sc in scores.items():
                    sc += _scatter(ch[self.clusters[t]["code"]].astype(np.int64), he, sc.shape[0])
                if subs:
                    local = np.repeat(np.arange(G, dtype=np.int64), Ng)
                for t, spec in subs.items():
                    # clusters nested in fe[0] groups are complete within the chunk
                    key = local * spec["xL"] + ch[spec["code"]].astype(np.int64)
                    uk, inv = np.unique(key, return_inverse=True)
                    S = _scatter(inv, he, len(uk))
                    meat_sub[t] += S.T @ S
                    G_sub[t] += len(uk)

            if self.save_resid:
                out = {c: ch[c] for c in src}
                out.update({c: ch[c] for c in self.keep})
                if weighted:
                    out["weights"] = w
                out[ycol_name] = y
                # row-level fe[0] contribution (intercept + slopes * variables)
                out[f"fe_{self.g_fe}"] = np.repeat(g_eff, Ng) if g_row is None else g_row
                for j, f in enumerate(self.o_fe):
                    out[f"fe_{f}"] = gam_y[self.offs[j] + codes[:, j]]
                out.update({"xb": X @ beta_c, "resid": e})
                tbl = pa.table(out)
                if r_writer is None:
                    r_writer = pq.ParquetWriter(paths["resid"], tbl.schema)
                r_writer.write_table(tbl, row_group_size=self.rgs)

            slopes_out = ({f"fe_{self.g_fe}[{v}]": g_coef[:, j + 1]
                           for j, v in enumerate(self.slope_vars)} if tcols else {})
            gt = pa.table({**{c: ch[c][starts[:-1]] for c in g_cols},
                           f"fe_{self.g_fe}": g_eff, **slopes_out, "n_obs": Ng.astype(np.int64)})
            if g_writer is None:
                g_writer = pq.ParquetWriter(paths["fe"][self.g_fe], gt.schema)
            g_writer.write_table(gt, row_group_size=self.rgs)
        if r_writer is not None:
            r_writer.close()
        g_writer.close()

        for f in self.o_fe:
            nl, o = self.n_levels[f], off[f]
            extra = {"component": pl.Series(self.comp)} if f == self.o_fe[0] else {}
            (pl.scan_parquet(self.paths["maps"][f])
               .with_columns(**{f"fe_{f}": pl.Series(gam_y[o:o + nl]),
                                "n_obs": pl.Series(self.cnt[o:o + nl]).cast(pl.Int64)}, **extra)
               .sink_parquet(paths["fe"][f]))

        meat = {t: sc.T @ sc for t, sc in scores.items()}
        meat.update(meat_sub)
        for t, spec in self.clusters.items():
            if spec["kind"] == "seg":
                meat[t] = acc_g.sum(axis=0)
            elif spec["kind"] == "segsub":
                spec["G"] = G_sub[t]
        rss, tss_w, sy, syy = acc_s.sum(axis=0)
        ssy = syy - sy * sy / self.wsum     # weighted total SS around the weighted mean
        return rss, tss_w, ssy, acc_B.sum(axis=0), acc_hc.sum(axis=0), meat, paths

    # ------------------------------------------------------------ estimation
    def _estimate(self, tag, model, A, gamma, ctx):
        """OLS or 2SLS for one model from the assembled V' M_D V, then its
        row pass and variance estimates."""
        fml, yname = model["fml"], model["y"]
        yj = self.vidx[yname]
        xj = [self.vidx[x] for x in model["x"]]
        mask = self._collinear(A[np.ix_(xj, xj)], xj)
        if len(xj) and mask.all():
            raise ValueError(f"{fml}: all covariates are collinear with the fixed effects")
        dropped = [x for x, m_ in zip(model["x"], mask) if m_]
        xk = [j for j, m_ in zip(xj, mask) if not m_]
        names = [x for x, m_ in zip(model["x"], mask) if not m_]
        k = len(xk)
        iv = model.get("iv")
        first_stage, f_stats = [], []
        if iv:
            lost = [e for e in iv["endog"] if e in dropped]
            if lost:
                raise ValueError(f"{fml}: endogenous variables {lost} are collinear with "
                                 "the fixed effects / exogenous regressors")
            zj = [self.vidx[z] for z in iv["z"]]
            zmask = self._collinear(A[np.ix_(zj, zj)], zj)
            zdrop = [z for z, m_ in zip(iv["z"], zmask) if m_]
            dropped += [z for z in zdrop if z not in dropped]
            zk = [j for j, m_ in zip(zj, zmask) if not m_]
            znames = [z for z, m_ in zip(iv["z"], zmask) if not m_]
            if len(zk) < k:
                raise ValueError(f"{fml}: under-identified ({len(zk)} instruments for "
                                 f"{k} regressors after dropping collinear columns)")
            Azz, Azx, Azy = A[np.ix_(zk, zk)], A[np.ix_(zk, xk)], A[zk, yj]
            Pi = np.linalg.solve(Azz, Azx)             # first-stage coefficients
            XhXh = Azx.T @ Pi
            beta = np.linalg.solve(XhXh, Pi.T @ Azy)
            bread_asm = XhXh
        else:
            zk, Pi = xk, np.eye(k)
            bread_asm = A[np.ix_(xk, xk)]
            beta = np.linalg.solve(bread_asm, A[xk, yj]) if k else np.zeros(0)
        if dropped:
            _warn(f"{fml}: {len(dropped)} variables dropped due to "
                  f"multicollinearity: {dropped}", self.logger)
        Axy = A[xk, yj]
        rss_asm = A[yj, yj] - 2 * beta @ Axy + beta @ A[np.ix_(xk, xk)] @ beta if k else A[yj, yj]

        self._log(f"pass 2: {fml}")
        rss, tss_w, ssy, B, meat_hc, meat, paths = self._pass2(tag, yname, gamma, yj, xk,
                                                               beta, zk, Pi)
        N, k_fe, nested = self.N, ctx["k_fe"], ctx["nested"]
        # Small-sample factors follow pyfixest's ssc defaults: iid
        # (N-1)/(N-K); hetero N/(N-K); CRV1 G/(G-1)*(N-1)/(N-Kc), with the
        # levels of FEs nested in any of the request's cluster terms dropped
        # from K (plus one back per nested FE). Multi-way: inclusion-exclusion
        # over the terms, each scaled with G = the smallest one-way cluster
        # count (pyfixest's G_df="min"); t-tests use G_min - 1 df.
        K = k + k_fe
        Binv = np.linalg.inv(B) if k else np.zeros((0, 0))
        vc = {"iid": (rss / (N - K) * Binv, N - K),
              "hetero": (N / (N - K) * Binv @ meat_hc @ Binv, N - K)}
        n_clusters = {}
        for req, terms in self.cluster_reqs.items():
            Gs = [self.clusters[t]["G"] for t, _, single in terms if single]
            Gm = min(Gs)
            nest = list(dict.fromkeys(d for t, _, _ in terms for d in nested[t]))
            Kc = K - sum(self.fe_params[d] for d in nest) + len(nest)
            adj = Gm / (Gm - 1) * (N - 1) / (N - Kc)
            Vc = sum(sign * adj * (Binv @ meat[t] @ Binv) for t, sign, _ in terms)
            vc[f"CRV1:{req}"] = (Vc, Gm - 1)
            n_clusters[req] = Gs
        V, dft = vc[ctx["default"]]
        if np.any(np.diag(V) < 0):
            _warn(f"{fml}: the {ctx['default']} vcov has negative variances "
                  "(possible with multi-way clustering)", self.logger)

        if iv:
            # one first-stage OLS per endogenous variable (same vcov type), and
            # the Wald F on the excluded instruments, as pyfixest reports it
            excluded = [z for z in znames if z not in names]
            for ei, en in enumerate(iv["endog"]):
                fs_model = {"fml": f"{en} ~ {' + '.join(znames)} | {' + '.join(self.fe)}",
                            "y": en, "x": znames}
                fs = self._estimate(f"{tag}_fs{ei}", fs_model, A, gamma, ctx)
                first_stage.append(fs)
                idx = [fs.coefnames.index(z) for z in excluded if z in fs.coefnames]
                b = fs.beta[idx]
                Vs = fs.vcov[np.ix_(idx, idx)]
                f_stats.append(float(b @ np.linalg.solve(Vs, b) / len(idx)) if idx else np.nan)

        # goodness of fit, with pyfixest's definitions (not reported for IV)
        k_fe_r2 = sum(self.fe_params.values()) - len(self.fe) + 1
        r2 = 1 - rss / ssy if ssy > 0 and not iv else np.nan
        r2w = 1 - rss / tss_w if tss_w > 0 and not iv else np.nan
        diag = {"rss_assembled": float(rss_asm), "rss_rows": float(rss),
                "bread_rel_diff": (float(np.abs(bread_asm - B).max() / np.abs(B).max()) if k
                                   else 0.0),
                "cells_per_obs": round(self.n_cells / self.n_obs, 4),
                "assembly": self.assembly, "nested_in_cluster": nested,
                "stream": self.stream_choice, "fe_params": dict(self.fe_params),
                "slope_redundancy": getattr(self, "slope_redundancy", []),
                "seconds_total": round(time.time() - ctx["t0"], 2)}
        return HDFEResult(
            fml=fml, depvar=yname, coefnames=names, fe_names=self.fe, beta=beta, vcov=V,
            vcov_type=ctx["default"], df_t=dft, n_obs=N, n_levels=dict(self.n_levels),
            n_identifying=self.n_identifying, n_components=self.n_components, k_fe=k_fe,
            rss=float(rss), r2_within=float(r2w), solver_info=dict(ctx["info"]), paths=paths,
            collin_vars=dropped, diagnostics=diag, all_vcovs=vc, r2=float(r2),
            adj_r2=float(1 - (1 - r2) * (N - 1) / (N - k - k_fe_r2)),
            adj_r2_within=float(1 - (1 - r2w) * (N - k_fe_r2) / (N - k - k_fe_r2)),
            rmse=float(np.sqrt(rss / N)) if not iv else np.nan, is_iv=bool(iv), first_stage=first_stage,
            f_stat_1st_stage=f_stats,
            n_clusters=n_clusters,
            weights=self.weights_name, weights_type=self.weights_type, _run=self._run)

    def _fit_passes(self, source, keys, extra, default, fe_dof, t0):
        self._pass0_code(source, keys, extra)
        self._pass1_cells()
        self._pass1b_identifying()
        self._components()
        self._log("n={:,} cells={:,} identifying groups={:,} cells={:,} "
                           "components={:,} threads={} ".format(
            self.n_obs, self.n_cells, self.n_identifying, self.n_ident_cells,
            self.n_components, nb.get_num_threads())
            + " ".join(f"{f}={v:,}" for f, v in self.n_levels.items()))

        self._log(f"step 2: solving for {self.m} variables "
                           f"(blocks of {self.rhs_block}), solver={self.solver}")
        gamma, info = self._solve()
        self._log("step 3: assembling normal equations")
        A = self._assemble(gamma)
        nested = {t: self._nested_dims(t) for t in self.clusters}
        total = sum(self.fe_params.values())   # fe[0]: one per group, or its rank with slopes
        n_red = (self.n_components + len(self.o_fe) - 1) if fe_dof == "exact" else len(self.o_fe)
        n_red += len(self._redundant_slopes())
        ctx = {"k_fe": total - n_red, "nested": nested, "default": default, "t0": t0,
               "info": info}
        self._track_disk()
        results = [self._estimate(f"m{mi:03d}", model, A, gamma, ctx)
                   for mi, model in enumerate(self.models)]
        return results

    # --------------------------------------------------------------- driver
    def fit(self, source, vcov="iid", cluster=(), fe_dof="exact"):
        """
        source : Parquet path/glob or a Polars LazyFrame.
        vcov   : default vcov: 'iid', 'hetero'/'HC1', {'CRV1': var} or
                 'CRV1:var'; multi-way clustering as {'CRV1': 'a+b'} (any
                 number of ways). iid and hetero are always computed; switch
                 with result.with_vcov(...).
        cluster: further cluster specs to compute CRV1 for, e.g.
                 ["sein", "pik+sein"]. Any column or 'a^b' combination works;
                 fe[0] itself and the other FE dimensions reuse their codes.
        fe_dof : 'exact' counts one redundant level per connected component of
                 the (fe[0], fe[1]) graph, plus one per further dimension;
                 'pyfixest' counts one per dimension beyond the first.
        Returns an HDFEResult, or an HDFEMulti when there are several models.
        """
        t0 = time.time()
        default = _vcov_key(vcov)
        keys = list(dict.fromkeys(["iid", "hetero", default]))
        extra = [_canon_cluster(c) for c in cluster]
        self.cluster_cols = list(dict.fromkeys(
            c for req in [k.split(":", 1)[1] for k in keys if k.startswith("CRV1:")] + extra
            for part in req.split("+") for c in part.split("^")))
        self._init_run()
        try:
            results = self._fit_passes(source, keys, extra, default, fe_dof, t0)
        except BaseException:
            # nothing from a failed fit is kept (unless debugging)
            self._release_intermediates()
            if not self.keep_intermediates:
                self._run.cleanup()
            raise
        self._release_intermediates()
        _ACTIVE_RUNS.discard(str(self.workdir))
        peak = self.disk_peak / 1e9
        kept = _dir_bytes(self.workdir) / 1e9
        for r in results:
            r.diagnostics["disk_peak_gb"] = round(peak, 3)
            r.diagnostics["disk_results_gb"] = round(kept, 3)
        self._log(f"done in {time.time()-t0:.1f}s; peak disk use {peak:.2f} GB, result files "
                  f"{kept:.2f} GB in {self.workdir}"
                  + (" (removed with the results)" if self.outputs == "auto" else ""))
        return results[0] if len(results) == 1 else HDFEMulti(results)

    @classmethod
    def from_formula(cls, fml, workdir=None, **options):
        """Estimator for a pyfixest-style formula; call .fit(data, vcov=...)."""
        return _FormulaEstimator(fml, workdir, options)


class _FormulaEstimator:
    def __init__(self, fml, workdir, options):
        self.fml, self.workdir, self.options = fml, workdir, options

    def fit(self, data, vcov=None, cluster=(), fe_dof="exact"):
        return feols_stream(self.fml, data, self.workdir, vcov=vcov, cluster=cluster,
                            fe_dof=fe_dof, **self.options)


def feols_stream(fml, data, workdir=None, vcov=None, cluster=(), fe_dof="exact", **options):
    """
    Out-of-core OLS with high-dimensional fixed effects from a pyfixest-style
    formula, e.g. "y ~ x1 + i(year, treat, ref=2010) | pik + sein^year".

    data    : Parquet path/glob or Polars LazyFrame.
    workdir : directory under which run directories are created (default:
              the system temporary directory); see StreamingHDFE for the
              outputs=, save_resid= and keep_intermediates= options.
    vcov    : as in pyfixest; default {'CRV1': <first FE>} like pyfixest.
    cluster : extra cluster specs to compute CRV1 for (one-way or 'a+b').
    **options : passed to StreamingHDFE (stream, weights, weights_type,
              solver, precond, keep, verbose, logger, log_level, ...).

    Varying slopes on the streamed dimension: "y ~ x | pik[exper] + sein".
    The worker FE file then has fe_pik (intercept) and fe_pik[exper] (slope)
    columns; the residual file's fe_pik column is the worker's total
    contribution for that row.

    IV formulas ("y ~ x | fe | endog ~ z") are estimated by 2SLS; each result
    carries its first-stage regressions (`first_stage`) and the Wald F of the
    excluded instruments computed with the same vcov (`f_stat_1st_stage`).

    All models with the same fixed effects share one set of passes and one
    solve, and are estimated on the rows where *all* of their variables are
    non-missing (pyfixest drops missing values model by model).
    """
    lf = data if isinstance(data, pl.LazyFrame) else pl.scan_parquet(data)
    _log(options.get("verbose", True), f"planning {fml!r} (formula expansion, level discovery)",
         options.get("logger"), options.get("log_level"))
    groups = _plan_formula(fml, lf)
    results = []
    try:
        for fe, g in groups.items():
            est = StreamingHDFE(y=list(g["y"].items()), x=list(g["x"].items()), fe=list(fe),
                                workdir=workdir, models=g["models"], **options)
            # pyfixest's default: cluster by the first fixed effect as written
            res = est.fit(lf, vcov=vcov if vcov is not None else
                          {"CRV1": _parse_fe_term(fe[0])[0]}, cluster=cluster, fe_dof=fe_dof)
            results += list(res) if isinstance(res, HDFEMulti) else [res]
    except BaseException:
        # a later FE set failed: don't leave the earlier sets' files behind
        if not options.get("keep_intermediates"):
            for r in results:
                r.cleanup()
        raise
    Formula, _ = _pyfixest_formula_api()
    order = {s.formula: i for i, s in enumerate(Formula.parse(fml))}
    results.sort(key=lambda r: order.get(r.fml, len(order)))
    return results[0] if len(results) == 1 else HDFEMulti(results)
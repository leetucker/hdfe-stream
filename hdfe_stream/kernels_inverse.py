"""numba kernels for applying S^- and the projection A S^- A' out of core.

`S = A'WA` is the normal-equations matrix of the full design, streamed fixed
effect included. The existing solver handles only right-hand sides built from
the design's own cell sums, and only for the *reduced* system; these kernels are
what generalize it to an arbitrary vector, which is what leave-out estimation
needs (statistical leverages, and from them the Kline-Saggio-Solvsten bias
correction).

Both kernels work on *rows*, not on the cell table, because the vectors being
projected are row-level and are regenerated rather than stored. Rows arrive
grouped by the streamed dimension -- `iter_group_chunks` yields whole groups --
which is what makes the group means below computable in one pass.

Writing A = [D_0, D_o] for the streamed block and everything else, the block
inverse turns on D_0'WD_0 being diagonal, one entry per streamed group. That
leaves

    T   = D_o' M_0 D_o                (the reduced matrix, already solved by
                                       _SolveMixin)
    u_o = T^- (v_o - D_o'WD_0 P^-1 v_0)
    u_0 = P^-1 (v_0 - D_0'WD_o u_o)

`_nb_center_rows` forms that reduced right-hand side; `_nb_project_rows` takes
the solution back to row level. Neither ever holds a vector indexed by the
streamed dimension, which is the whole point.
"""

from __future__ import annotations

import numba as nb
import numpy as np


@nb.njit(parallel=True, cache=True)
def _nb_center_rows(starts, codes, offs, w, X, R, acc, acc_x):
    """acc += D_o' W M_0 R and acc_x += X' W M_0 R, for one chunk of rows.

    M_0 removes each streamed group's weighted mean, so what gets scattered into
    the accumulators is w_i (R_i - Rbar_g): the reduced right-hand side, for the
    fixed-effect block and the covariate block respectively.

    starts (G+1,)      group boundaries within this chunk
    codes  (n, D)      level codes of the non-streamed dimensions
    offs   (D,)        start of each dimension's block in the stacked vector
    w      (n,)        weights
    X      (n, k)      covariates; k = 0 for a design of fixed effects alone
    R      (n, m)      the vectors being projected
    acc    (nt, L, m)  per-thread level accumulators, so threads never share
    acc_x  (nt, k, m)  per-thread covariate accumulators
    """
    G, nt = len(starts) - 1, acc.shape[0]
    D, m, k = codes.shape[1], R.shape[1], X.shape[1]
    for t in nb.prange(nt):
        for gi in range(G * t // nt, G * (t + 1) // nt):
            s, e = starts[gi], starts[gi + 1]
            sw = 0.0
            for i in range(s, e):
                sw += w[i]
            if sw <= 0.0:
                continue
            for j in range(m):
                total = 0.0
                for i in range(s, e):
                    total += w[i] * R[i, j]
                mean = total / sw
                for i in range(s, e):
                    v = w[i] * (R[i, j] - mean)
                    for d in range(D):
                        acc[t, offs[d] + codes[i, d], j] += v
                    for c in range(k):
                        acc_x[t, c, j] += X[i, c] * v


@nb.njit(parallel=True, cache=True)
def _nb_project_rows(starts, codes, offs, w, X, C, R, U, out):
    """out[i, j] = (A u_j)_i, given u_j's covariate block C[:, j] and
    non-streamed fixed-effect block U[:, j].

    The streamed block is recovered group by group rather than stored:

        h_i     = x_i'C + sum_d u_o[level_d(i)]
        u_0[g]  = weighted mean over the group of (R_i - h_i)
        (A u)_i = u_0[g(i)] + h_i

    `out` doubles as scratch for h_i, so no per-chunk allocation is needed.
    Groups are independent and each writes only its own rows, so plain prange
    over groups is safe without per-thread accumulators.
    """
    G = len(starts) - 1
    D, m, k = codes.shape[1], R.shape[1], X.shape[1]
    for gi in nb.prange(G):
        s, e = starts[gi], starts[gi + 1]
        sw = 0.0
        for i in range(s, e):
            sw += w[i]
        for j in range(m):
            total = 0.0
            for i in range(s, e):
                h = 0.0
                for d in range(D):
                    h += U[offs[d] + codes[i, d], j]
                for c in range(k):
                    h += X[i, c] * C[c, j]
                out[i, j] = h
                total += w[i] * (R[i, j] - h)
            mean = total / sw if sw > 0.0 else 0.0
            for i in range(s, e):
                out[i, j] += mean


@nb.njit(cache=True)
def _nb_rademacher(ordinal, n_rows, draws, seed, first_draw, out):
    """Rademacher +/-1 entries for rows `ordinal` to `ordinal + n_rows`.

    Derived from the row's position in the (fixed, sorted) row files and the
    draw index, so the same vector comes back in the second pass without having
    been stored anywhere. splitmix64 on (seed, ordinal, draw) is counter-based,
    so entry (i, j) is reproducible on its own, independently of iteration order
    or chunk boundaries.

    `first_draw` offsets the draw index, which is what lets a long run of draws
    be processed in blocks: block b asks for draws b*block .. (b+1)*block and
    gets its own stretch of one single well-defined sequence, rather than
    repeating the same vectors with a different seed.
    """
    for i in range(n_rows):
        row = np.uint64(ordinal + i)
        for j in range(draws):
            d = np.uint64(first_draw + j)
            x = (np.uint64(seed)
                 ^ (row * np.uint64(0x9E3779B97F4A7C15))
                 ^ (d * np.uint64(0xBF58476D1CE4E5B9)))
            x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
            x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
            x = x ^ (x >> np.uint64(31))
            out[i, j] = 1.0 if (x >> np.uint64(63)) == np.uint64(0) else -1.0


@nb.njit(parallel=True, cache=True)
def _nb_reduce_rows(starts, codes, offs, w, X, R, acc, acc_x):
    """acc += D_o' W R and acc_x += X' W R, with no centering.

    The centered version above is what a right-hand side built *from rows* needs.
    This plain one is what a right-hand side given directly in coefficient space
    needs: there the streamed block contributes D_o'W rho, where rho is constant
    within each group and so survives centering untouched.
    """
    G, nt = len(starts) - 1, acc.shape[0]
    D, m, k = codes.shape[1], R.shape[1], X.shape[1]
    for t in nb.prange(nt):
        for gi in range(G * t // nt, G * (t + 1) // nt):
            for i in range(starts[gi], starts[gi + 1]):
                for j in range(m):
                    v = w[i] * R[i, j]
                    for d in range(D):
                        acc[t, offs[d] + codes[i, d], j] += v
                    for c in range(k):
                        acc_x[t, c, j] += X[i, c] * v


@nb.njit(parallel=True, cache=True)
def _nb_spread_groups(starts, w, group_values, out):
    """out[i, j] = group_values[g(i), j] / s_g, the streamed block as rows.

    s_g is the group's weight total, so this is P^-1 applied to the streamed
    block and broadcast back over the group's rows -- the form both the
    reduction and the reconstruction want it in.
    """
    G = len(starts) - 1
    m = group_values.shape[1]
    for gi in nb.prange(G):
        s, e = starts[gi], starts[gi + 1]
        sw = 0.0
        for i in range(s, e):
            sw += w[i]
        inv = 1.0 / sw if sw > 0.0 else 0.0
        for j in range(m):
            value = group_values[gi, j] * inv
            for i in range(s, e):
                out[i, j] = value


@nb.njit(parallel=True, cache=True)
def _nb_group_means(starts, w, R, out):
    """out[g, j] = weighted mean of R over group g."""
    G = len(starts) - 1
    m = R.shape[1]
    for gi in nb.prange(G):
        s, e = starts[gi], starts[gi + 1]
        sw = 0.0
        for i in range(s, e):
            sw += w[i]
        inv = 1.0 / sw if sw > 0.0 else 0.0
        for j in range(m):
            total = 0.0
            for i in range(s, e):
                total += w[i] * R[i, j]
            out[gi, j] = total * inv


@nb.njit(parallel=True, cache=True)
def _nb_project_coef(starts, codes, offs, w, X, C, RHO, U, out_rows, out_groups):
    """Row values and streamed block for a coefficient-space solution.

    Same reconstruction as `_nb_project_rows`, but it also hands back the
    streamed block it derives on the way, which a caller working in coefficient
    space needs and a caller working in row space does not:

        h_i        = x_i'C + sum_d u_o[level_d(i)]
        u_0[g]     = weighted mean over the group of (RHO_i - h_i)
        (A u)_i    = u_0[g(i)] + h_i
    """
    G = len(starts) - 1
    D, m, k = codes.shape[1], RHO.shape[1], X.shape[1]
    for gi in nb.prange(G):
        s, e = starts[gi], starts[gi + 1]
        sw = 0.0
        for i in range(s, e):
            sw += w[i]
        for j in range(m):
            total = 0.0
            for i in range(s, e):
                h = 0.0
                for d in range(D):
                    h += U[offs[d] + codes[i, d], j]
                for c in range(k):
                    h += X[i, c] * C[c, j]
                out_rows[i, j] = h
                total += w[i] * (RHO[i, j] - h)
            mean = total / sw if sw > 0.0 else 0.0
            out_groups[gi, j] = mean
            for i in range(s, e):
                out_rows[i, j] += mean


@nb.njit(parallel=True, cache=True)
def _nb_group_sums(starts, w, R, out):
    """out[g, j] = sum over group g of w_i R[i, j]."""
    G = len(starts) - 1
    m = R.shape[1]
    for gi in nb.prange(G):
        s, e = starts[gi], starts[gi + 1]
        for j in range(m):
            total = 0.0
            for i in range(s, e):
                total += w[i] * R[i, j]
            out[gi, j] = total


@nb.njit(parallel=True, cache=True)
def _nb_trace_moments(starts, codes, psi_column, psi_offset, w,
                      Z_levels, Z_groups, U_levels, U_groups, acc):
    """Cross-moments over rows for the three variance-component quadratic forms.

    Each form is an observation-level second moment of the fitted effects, so
    evaluating Z'Q u needs the two coefficient vectors read at row level: the
    chosen non-streamed dimension ("psi") from a level-indexed vector, and the
    streamed dimension ("alpha") from a group-indexed one.

    Accumulates, per draw j, into acc[t, j, :]:

        0: sum w              4: sum w a1        6: sum w b1
        1: sum w a1 a2        5: sum w a2        7: sum w b2
        2: sum w b1 b2
        3: sum w (a1 b2 + b1 a2)

    where a = the psi-dimension value and b = the alpha (streamed) value, with
    1 denoting Z and 2 denoting u. The covariances follow from these outside.
    """
    G, nt = len(starts) - 1, acc.shape[0]
    m = Z_groups.shape[1]
    for t in nb.prange(nt):
        for gi in range(G * t // nt, G * (t + 1) // nt):
            s, e = starts[gi], starts[gi + 1]
            for j in range(m):
                b1 = Z_groups[gi, j]
                b2 = U_groups[gi, j]
                for i in range(s, e):
                    level = psi_offset + codes[i, psi_column]
                    a1 = Z_levels[level, j]
                    a2 = U_levels[level, j]
                    weight = w[i]
                    acc[t, j, 0] += weight
                    acc[t, j, 1] += weight * a1 * a2
                    acc[t, j, 2] += weight * b1 * b2
                    acc[t, j, 3] += weight * (a1 * b2 + b1 * a2)
                    acc[t, j, 4] += weight * a1
                    acc[t, j, 5] += weight * a2
                    acc[t, j, 6] += weight * b1
                    acc[t, j, 7] += weight * b2


@nb.njit(parallel=True, cache=True)
def _nb_group_distinct(starts, codes, column, out):
    """out[g] = how many distinct values of codes[:, column] group g contains.

    Quadratic within a group, which is the right trade: groups are a worker's
    handful of observations, and this avoids sorting or allocating per group.
    """
    G = len(starts) - 1
    for gi in nb.prange(G):
        s, e = starts[gi], starts[gi + 1]
        distinct = 0
        for i in range(s, e):
            seen_before = False
            for j in range(s, i):
                if codes[j, column] == codes[i, column]:
                    seen_before = True
                    break
            if not seen_before:
                distinct += 1
        out[gi] = distinct

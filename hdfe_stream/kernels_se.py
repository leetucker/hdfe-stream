"""Numba kernels for leave-out standard errors.

Three row-level operations the inference layer needs and the existing kernels do
not provide: reading a coefficient-space vector back at row level, the two cross
sums the covariance quadratic form needs, and the weighted accumulation of the
three per-row diagonals.

Everything here works on a *block* of vectors at once -- the trailing axis is
the draw -- because a pass over the rows is the expensive thing and the whole
point of blocking is to amortize it.
"""

from __future__ import annotations

import numba as nb


@nb.njit(parallel=True, cache=True)
def _nb_readout(starts, codes, offs, X, U_levels, U_cov, U_groups, out):
    """out[i, j] = (A u_j)_i, the row value of a coefficient-space vector.

        (A u)_i = u_streamed[g(i)] + sum_d u_levels[level_d(i)] + x_i' u_cov

    This is the plain reconstruction, with no weighting and no group solve: the
    caller already has the whole coefficient vector and only wants to see it at
    row level. `_nb_project_coef` looks similar but is not the same thing -- it
    *derives* the streamed block on the way, which a caller holding the solution
    has no need of.
    """
    G = len(starts) - 1
    D, m, k = codes.shape[1], U_groups.shape[1], X.shape[1]
    for gi in nb.prange(G):
        s, e = starts[gi], starts[gi + 1]
        for j in range(m):
            ug = U_groups[gi, j]
            for i in range(s, e):
                h = ug
                for d in range(D):
                    h += U_levels[offs[d] + codes[i, d], j]
                for c in range(k):
                    h += X[i, c] * U_cov[c, j]
                out[i, j] = h


@nb.njit(parallel=True, cache=True)
def _nb_cov_cross(starts, codes, psi_column, psi_offset, w,
                  U_levels, U_groups, acc_psi, out_groups):
    """The two cross sums the covariance quadratic form needs.

        acc_psi[l, j]    += sum over rows at psi level l of w_i u_streamed[g(i)]
        out_groups[g, j]  = sum over rows in group g of w_i u_levels[psi(i)]

    Q_cov pairs the two dimensions, so applying it to a coefficient vector means
    summing each dimension's values against the *other* dimension's grouping --
    which is the one part of the three quadratic forms that cannot be done in
    coefficient space alone. The centering is left to the caller: the mean times
    the level's weight total is subtracted afterwards, which is exact because
    each centered selector is orthogonal to the weights by construction.
    """
    G, nt = len(starts) - 1, acc_psi.shape[0]
    m = U_groups.shape[1]
    for t in nb.prange(nt):
        for gi in range(G * t // nt, G * (t + 1) // nt):
            s, e = starts[gi], starts[gi + 1]
            for j in range(m):
                ug = U_groups[gi, j]
                total = 0.0
                for i in range(s, e):
                    level = psi_offset + codes[i, psi_column]
                    acc_psi[t, level, j] += w[i] * ug
                    total += w[i] * U_levels[level, j]
                out_groups[gi, j] = total


@nb.njit(parallel=True, cache=True)
def _nb_bii_moments(root, r1, r2, out_psi, out_alpha, out_cov):
    """Accumulate the three per-row diagonals from one block of draws.

    r1 and r2 are the row values of S^- L_psi' q and S^- L_alpha' q. With
    Q = L'L for a variance and Q = (L_psi' L_alpha + L_alpha' L_psi)/2 for the
    covariance, the diagonal of B is a squared readout in the first case and a
    product of the two in the second -- so one pair of solves per draw yields
    all three.
    """
    n, m = r1.shape
    for i in nb.prange(n):
        a = 0.0
        b = 0.0
        c = 0.0
        for j in range(m):
            x = root[i] * r1[i, j]
            y = root[i] * r2[i, j]
            a += x * x
            b += y * y
            c += x * y
        out_psi[i] += a
        out_alpha[i] += b
        out_cov[i] += c

"""numba kernels for the GLM (IRLS) path; see glm.py."""

from __future__ import annotations

import numba as nb
import numpy as np


@nb.njit(parallel=True, cache=True)
def _nb_cell_moments(cstarts, w, V, nc, sc):
    """Per cell (rows cstarts[c]:cstarts[c+1], one run of the sorted rows):
    nc[c] = sum w and sc[c] = sum w v. V is then overwritten with the
    weighted within-cell deviations sqrt(w) (v - vbar_c), whose
    cross-products the caller forms with one BLAS product."""
    m = V.shape[1]
    for c in nb.prange(len(cstarts) - 1):
        s, e = cstarts[c], cstarts[c + 1]
        W = 0.0
        for j in range(m):
            sc[c, j] = 0.0
        for i in range(s, e):
            W += w[i]
            for j in range(m):
                sc[c, j] += w[i] * V[i, j]
        nc[c] = W
        for i in range(s, e):
            r = np.sqrt(w[i])
            for j in range(m):
                V[i, j] = r * (V[i, j] - sc[c, j] / W)


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

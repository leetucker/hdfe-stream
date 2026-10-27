"""numba kernels on rows grouped by the streamed dimension (as
`iter_group_chunks` yields them, `starts` marking each group's first row), and
counter-based random draws keyed by a row's position in the row files.

Nothing here depends on the model: per-group sums are the per-group moment
contributions (scores, cluster sums) of any estimator over the same passes.
The leave-out estimators are their first users.
"""

from __future__ import annotations

import numba as nb
import numpy as np


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

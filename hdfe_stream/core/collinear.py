"""The collinearity check on a cross-product or information matrix.

`collinear_columns(A, tol)` is fixest's and pyfixest's rule -- a sequential
Cholesky factorization of A, in which a column whose pivot (the part of its
diagonal the earlier kept columns do not explain) is below `tol` is dropped
-- with a second test in the same pass: a column is also dropped when its
pivot is at most PIVOT_REL of its own diagonal.

The second test is what makes the check scale-free. An exactly collinear
column's pivot is rounding error, of order the machine epsilon times its
diagonal, and so of either sign: on a large or heavily weighted sample (an
indicator's diagonal of 1e5 or more, say), that error can be larger than the
absolute `tol`, and whether the column is dropped then depends on the order
in which the sums were added. Relative to the diagonal, an exactly collinear
column's pivot is 1e-13 or less, and a column that is not collinear has one
many orders of magnitude larger, so the test does not depend on rounding.
Every column the absolute test drops is still dropped, and which column of a
collinear set goes (the later one) does not change.
"""

from __future__ import annotations

import numba as nb
import numpy as np

# drop a column whose pivot is at most this share of its own diagonal: a
# column that is not collinear but is this close to it would be estimated
# with almost no precision anyway
PIVOT_REL = 1e-9


@nb.njit(cache=True)
def _collinear_columns(A, tol, rel):
    K = A.shape[0]
    # L[j, :j] is row j of the Cholesky factor (column j of R in fixest's
    # notation), kept contiguous for the inner products; a dropped column's
    # row stays zero, so it drops out of every later product
    L = np.zeros((K, K))
    drop = np.zeros(K, np.bool_)
    for j in range(K):
        r = A[j, j]
        for k in range(j):
            r -= L[j, k] * L[j, k]
        if r < tol or r <= rel * A[j, j]:
            drop[j] = True
            for k in range(j):
                L[j, k] = 0.0
            continue
        d = np.sqrt(r)
        L[j, j] = d
        for i in range(j + 1, K):
            v = A[i, j]
            for k in range(j):
                v -= L[i, k] * L[j, k]
            L[i, j] = v / d
    return drop


def collinear_columns(A, tol: float = 1e-10, rel: float = PIVOT_REL) -> np.ndarray:
    """Which columns of the symmetric matrix A are collinear with the ones
    before them (a boolean mask): fixest's absolute test on each pivot (below
    `tol`) or the relative one (at most `rel` of the column's diagonal)."""
    A = np.ascontiguousarray(A, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be square")
    if not A.shape[0]:
        return np.zeros(0, bool)
    return _collinear_columns(A, float(tol), float(rel))

"""The fused IRLS row pass (`kernels_glm._nb_glm_rows`) computes each family's
mean, weight and deviance with scalar versions of the scipy functions
`families.py` uses. They must agree, including far in the tails, where the
binary families' deviance comes from series that do not underflow."""

from __future__ import annotations

import numba as nb
import numpy as np
import pytest
from scipy import special

from hdfe_stream.kernels_glm import _expit, _log_expit, _log_ndtr, _ndtr

GRID = np.concatenate([np.linspace(-60.0, 60.0, 4801),
                       [-745.0, -700.0, -40.0, -20.0001, -19.9999, 5.9999, 6.0001, 709.0]])


@nb.njit
def _apply(fn, x):
    out = np.empty_like(x)
    for i in range(len(x)):
        out[i] = fn(x[i])
    return out


@pytest.mark.parametrize("mine, reference", [
    (_expit, special.expit), (_log_expit, special.log_expit),
    (_ndtr, special.ndtr), (_log_ndtr, special.log_ndtr)])
def test_scalar_functions_match_scipy(mine, reference):
    got, want = _apply(mine, GRID), reference(GRID)
    assert np.array_equal(np.isfinite(got), np.isfinite(want))
    finite = np.isfinite(want)
    # agreement to rounding: relative where the values are large (log_ndtr
    # far in the tail is in the hundreds of thousands), absolute near zero
    diff = np.abs(got[finite] - want[finite])
    assert (diff / np.maximum(np.abs(want[finite]), 1.0)).max() < 1e-13

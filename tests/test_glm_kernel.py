"""The fused IRLS row pass (`kernels_glm._nb_glm_rows`) computes each family's
mean, weight and deviance with scalar versions of the scipy functions
`families.py` uses. They must agree, including far in the tails, where the
binary families' deviance comes from series that do not underflow."""

from __future__ import annotations

import numba as nb
import numpy as np
import pytest
from scipy import special

from hdfe_stream.core.kernels_glm import _expit, _log_expit, _log_ndtr, _ndtr

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


@pytest.mark.parametrize("init", [True, False], ids=["start", "step"])
@pytest.mark.parametrize("wide", [False, True], ids=["blockwise", "one-product"])
@pytest.mark.parametrize("family", ["poisson", "logit", "probit"])
def test_plain_row_pass_matches_families(family, wide, init):
    """The row pass without fixed effects (`_nb_glm_rows_plain`), with
    frequency weights and an offset, against what families.py computes: the
    sums, the weighted cross-products of [z, x], whether each block forms
    its own (`wide`: the caller forms V'V), and independent of the threads."""
    from hdfe_stream.core.families import get_family
    from hdfe_stream.core.kernels_glm import FAMILY_CODES, _nb_glm_rows_plain

    rng = np.random.default_rng(3)
    n, k, blocks = 5_003, 3, 64
    X = np.column_stack([np.ones(n), rng.normal(size=(n, k - 1))])
    beta = np.array([0.2, 0.5, -0.4])
    off = rng.normal(0, 0.3, n)
    f = rng.integers(1, 4, n).astype(float)
    fam = get_family(family)
    eta_true = X @ beta + off
    y = (rng.poisson(np.exp(eta_true)) if family == "poisson"
         else (rng.random(n) < special.expit(eta_true))).astype(float)
    ybar = float(y.mean())

    eta = fam.init_eta(y, ybar) if init else X @ beta + off
    mu, W, u = fam.irls(y, eta)
    dev, ll = fam.deviance(y, eta, mu), fam.loglik(y, eta, mu)
    w = W * f
    v = np.column_stack([eta + u - off, X])
    want_parts = [dev @ f, ll @ f, y @ f, f.sum(),
                  special.gammaln(y + 1.0) @ f if family == "poisson" else 0.0, w.sum()]

    def run(threads):
        nb.set_num_threads(threads)
        V = np.empty((n, k + 1))
        Wb = np.empty((0 if wide else blocks, k + 1, k + 1))
        parts = np.empty((blocks, 6 + k + 1))
        _nb_glm_rows_plain(FAMILY_CODES[family], init, ybar, y, X, f, off,
                           np.zeros(0) if init else beta, V, Wb, parts)
        return parts.sum(axis=0), (V.T @ V if wide else Wb.sum(axis=0))

    try:
        p, A = run(1)
        p4, A4 = run(min(4, nb.config.NUMBA_NUM_THREADS))
    finally:
        nb.set_num_threads(nb.config.NUMBA_NUM_THREADS)
    np.testing.assert_allclose(p[:6], want_parts, rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(p[6:], w @ v, rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(A, (v * w[:, None]).T @ v, rtol=1e-12, atol=1e-9)
    assert np.array_equal(p, p4)
    if not wide:
        assert np.array_equal(A, A4)

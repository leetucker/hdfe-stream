"""The collinearity check (`core.collinear`): pyfixest's rule, and the
pivot test relative to each column's diagonal that makes it scale-free."""

import numpy as np
import pytest

from hdfe_stream.core.collinear import collinear_columns


def _design(rng, n=2000, k=12, share=0.1):
    X = (rng.random((n, k)) < share).astype(float)
    X[:, 0] = 1.0
    X[:, 5] = X[:, 2] + X[:, 3]          # exactly collinear with two earlier columns
    X[:, 9] = X[:, 0] - X[:, 7]          # and with the intercept
    return X


def test_matches_pyfixest_without_the_relative_test():
    pyfixest = pytest.importorskip("pyfixest.core")
    rng = np.random.default_rng(1)
    for trial in range(20):
        X = rng.normal(size=(300, 40))
        X[:, 17] = X[:, 3] - 2 * X[:, 11]
        X[:, 30] = X[:, 29] * (1 + 1e-7 * rng.normal(size=300))
        A = X.T @ X * 10.0 ** rng.integers(-3, 4)
        want, _, _ = pyfixest.find_collinear_variables(np.ascontiguousarray(A), 1e-10)
        np.testing.assert_array_equal(collinear_columns(A, 1e-10, rel=0.0), want)


def test_exact_collinearity_dropped_at_any_scale():
    """At a large scale the rounding in an exactly collinear column's pivot
    exceeds the absolute tolerance; the relative test still drops it."""
    rng = np.random.default_rng(2)
    X = _design(rng)
    for scale in (1.0, 1e5, 1e9, 1e13):
        A = X.T @ X * scale
        assert np.flatnonzero(collinear_columns(A)).tolist() == [5, 9], scale
    # with the absolute test alone, the largest scale keeps a collinear column
    A = X.T @ X * 1e13
    assert collinear_columns(A, rel=0.0).sum() < 2


def test_does_not_depend_on_the_order_of_the_sums():
    rng = np.random.default_rng(3)
    X = _design(rng)
    w = rng.uniform(1, 60, size=len(X)) * 1e3     # large weights, as a Poisson's mean
    for trial in range(30):
        order = rng.permutation(len(X))
        A = np.zeros((X.shape[1], X.shape[1]))
        for rows in np.array_split(order, 7):     # summed in chunks, in a random order
            Xc = X[rows] * np.sqrt(w[rows])[:, None]
            A += Xc.T @ Xc
        assert np.flatnonzero(collinear_columns(A)).tolist() == [5, 9]


def test_nearly_collinear_column_kept():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(5000, 6))
    X[:, 4] = X[:, 1] + 1e-3 * rng.normal(size=5000)   # R^2 about 1 - 1e-6
    assert not collinear_columns(X.T @ X).any()


def test_zero_columns_and_empty():
    A = np.diag([2.0, 0.0, 3.0])
    assert collinear_columns(A).tolist() == [False, True, False]
    assert collinear_columns(np.zeros((0, 0))).shape == (0,)
    with pytest.raises(ValueError):
        collinear_columns(np.zeros((2, 3)))

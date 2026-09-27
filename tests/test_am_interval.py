"""The Andrews-Mikusheva interval: pure numerics, no data.

Checked three ways: the critical value against its analytic limits and a direct
simulation of its defining distribution; the interval against the definition
(extremes over a filled grid of the ellipse); and against Saggio's closed-form
quartic from the KSS reference implementation, ported below.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from hdfe_stream.am_interval import am_critical_value, am_interval, curvature


def test_critical_value_limits():
    """kappa = 0 is the chi-squared(1) quantile, kappa -> infinity chi-squared(2)."""
    assert am_critical_value(0.0) == pytest.approx(stats.norm.ppf(0.975), abs=1e-10)
    assert am_critical_value(1e-8) == pytest.approx(stats.norm.ppf(0.975), abs=1e-6)
    assert am_critical_value(1e7) == pytest.approx(
        np.sqrt(stats.chi2.ppf(0.95, 2)), abs=1e-5)
    assert am_critical_value(1e-8, alpha=0.10) == pytest.approx(
        stats.norm.ppf(0.95), abs=1e-6)


def test_critical_value_rises_with_curvature():
    values = [am_critical_value(k) for k in (0.0, 0.1, 0.5, 1, 2, 5, 20, 100)]
    assert np.all(np.diff(values) > 0)


@pytest.mark.parametrize("kappa", [0.25, 1.0, 4.0, 20.0])
def test_critical_value_matches_simulation(kappa):
    """The (1 - alpha) quantile of sqrt(chi_1^2 + (chi_1' + 1/kappa)^2) - 1/kappa."""
    rng = np.random.default_rng(int(kappa * 100))
    n = 1_000_000
    a = np.sqrt(rng.chisquare(1, n))
    b = np.sqrt(rng.chisquare(1, n))
    rho = np.sqrt(a ** 2 + (b + 1 / kappa) ** 2) - 1 / kappa
    # compare on the probability scale, where the Monte Carlo error is exactly
    # binomial -- a simulated quantile's error depends on the density there
    share = np.mean(rho <= am_critical_value(kappa))
    assert abs(share - 0.95) < 4 * np.sqrt(0.95 * 0.05 / n), share


def _brute(lam, b1, th1, sigma, z, n=900):
    L = np.linalg.cholesky(sigma)
    r = np.sqrt(np.linspace(0, 1, n))[:, None]
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)[None, :]
    b = b1 + z * L[0, 0] * r * np.cos(t)
    th = th1 + z * (L[1, 0] * r * np.cos(t) + L[1, 1] * r * np.sin(t))
    g = lam * b * b + th
    return g.min(), g.max()


def _saggio(lam, b1, th1, sigma, z):
    """AM_CI.m from rsaggio87/LeaveOutTwoWay, with lambda already normalized.

    Written for lambda > 0 (it takes gamma = sqrt(gamma^2) >= 0)."""
    vs, vt = sigma[0, 0], sigma[1, 1]
    rho = sigma[0, 1] / np.sqrt(vs * vt)
    gamma = np.sqrt(lam ** 2 * vs ** 2 / vt)
    s = np.sqrt(vs)
    p = [4 * gamma ** 2 / vs,
         4 * gamma * rho / s - 8 * b1 * gamma ** 2 / vs,
         1 - 4 * gamma ** 2 * z ** 2 + 4 * gamma ** 2 * b1 ** 2 / vs
         - 8 * b1 * rho * gamma / s,
         4 * b1 ** 2 * gamma * rho / s - 2 * b1 - 4 * gamma * z ** 2 * rho * s,
         b1 ** 2 - rho ** 2 * vs * z ** 2]
    roots = np.roots(p)
    b = np.real(roots[np.abs(np.imag(roots)) < 1e-9])
    den = np.sqrt(1 - rho ** 2 + (2 * gamma * b / s + rho) ** 2)
    center = lam * b ** 2 + th1 - rho * np.sqrt(vt / vs) * (b1 - b)
    half = z * np.sqrt(vt) * (1 - rho ** 2) / den
    return (center - half).min(), (center + half).max()


def _random_case(rng, positive=True, b_scale=1.0):
    lam = rng.uniform(0.2, 3.0) * (1 if positive else -1)
    b1 = rng.normal(0, 1.5) * b_scale
    vb, vt = rng.uniform(0.2, 2), rng.uniform(0.2, 2)
    rho = rng.uniform(-0.9, 0.9)
    c = rho * np.sqrt(vb * vt)
    return lam, b1, rng.normal(), np.array([[vb, c], [c, vt]])


@pytest.mark.parametrize("seed", range(8))
def test_interval_matches_the_reference_quartic(seed):
    rng = np.random.default_rng(seed)
    lam, b1, th1, sigma = _random_case(rng, b_scale=seed % 3)   # includes b1 = 0
    lo, hi, info = am_interval(lam, b1, th1, sigma)
    slo, shi = _saggio(lam, b1, th1, sigma, info["z"])
    assert lo == pytest.approx(slo, abs=1e-10 * (hi - lo))
    assert hi == pytest.approx(shi, abs=1e-10 * (hi - lo))


@pytest.mark.parametrize("positive", [True, False])
def test_interval_matches_its_definition(positive):
    """Extremes over the filled ellipse, for either sign of lambda_1 -- the
    covariance's leading eigenvalue is usually negative, where the quartic's
    gamma >= 0 is not meant to apply."""
    rng = np.random.default_rng(11 if positive else 12)
    for _ in range(4):
        lam, b1, th1, sigma = _random_case(rng, positive)
        lo, hi, info = am_interval(lam, b1, th1, sigma)
        blo, bhi = _brute(lam, b1, th1, sigma, info["z"])
        assert lo <= blo + 1e-12 and hi >= bhi - 1e-12      # never inside
        assert abs(lo - blo) < 1e-4 * (hi - lo)              # grid resolution
        assert abs(hi - bhi) < 1e-4 * (hi - lo)


def test_curvature_matches_kss():
    sigma = np.array([[2.0, 0.6], [0.6, 3.0]])
    rho2 = 0.36 / 6.0
    assert curvature(-1.5, sigma) == pytest.approx(
        2 * 1.5 * 2.0 / (np.sqrt(3.0) * np.sqrt(1 - rho2)))


def test_nearly_linear_map_gives_the_normal_interval():
    """As lambda_1 -> 0 the map is linear in theta1 and the interval is the
    ordinary one for theta1 at the chi-squared(1) critical value."""
    sigma = np.array([[1.0, 0.2], [0.2, 0.5]])
    lo, hi, info = am_interval(1e-9, 0.3, 1.0, sigma)
    half = stats.norm.ppf(0.975) * np.sqrt(0.5)
    assert info["kappa"] < 1e-8
    assert lo == pytest.approx(1.0 - half, abs=1e-6)
    assert hi == pytest.approx(1.0 + half, abs=1e-6)

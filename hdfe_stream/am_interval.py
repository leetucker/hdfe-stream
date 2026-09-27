"""The Andrews-Mikusheva interval for a weakly identified quadratic form.

Pure numerics, no data: given the leading eigenvalue lambda_1 and the joint
estimate (b1-hat, theta1-hat) with its 2x2 covariance, return the q = 1
confidence interval of Kline, Saggio and Solvsten (2020), Section 6.1:

    C = [ min, max ] of  lambda_1 b^2 + t  over the ellipse
        { (b1-hat - b, theta1-hat - t) Sigma^-1 (.)' <= z_{alpha,kappa}^2 }

Projecting a 2-d ellipse through a quadratic map would ordinarily need the
chi-squared(2) critical value to control size; Andrews and Mikusheva (2016)
show the curvature of the map lets it come down, toward the chi-squared(1) value
when the map is nearly linear. KSS's curvature for q = 1 is

    kappa = 2 |lambda_1| V[b1] / ( V[theta1]^(1/2) (1 - rho^2)^(1/2) )

and z_{alpha,kappa} is the (1 - alpha) quantile of

    rho(kappa) = sqrt(chi_1^2 + (chi_1' + 1/kappa)^2) - 1/kappa

for independent chi variates (Appendix C.6.1). Saggio's reference tabulates that
quantile by simulation; for q = 1 it has an exact one-dimensional integral form,
used here. With a, b independent half-normals and c = 1/kappa,

    rho <= z  <=>  a^2 + (b + c)^2 <= (z + c)^2
    P(rho <= z) = int_0^z 2 phi(b) [2 Phi(sqrt((z - b)(z + b + 2c))) - 1] db

written in factored form so nothing cancels as kappa -> 0. The limits are the
chi-squared(1) quantile at kappa = 0 and chi-squared(2) as kappa -> infinity.
"""

from __future__ import annotations

import numpy as np


def am_cdf(z, kappa):
    """P(rho(kappa) <= z) for q = 1."""
    from scipy import integrate, stats

    if z <= 0:
        return 0.0
    if kappa <= 0:
        return float(2.0 * stats.norm.cdf(z) - 1.0)
    c = 1.0 / kappa

    def integrand(b):
        inner = np.sqrt(max((z - b) * (z + b + 2.0 * c), 0.0))
        return 2.0 * stats.norm.pdf(b) * (2.0 * stats.norm.cdf(inner) - 1.0)

    value, _err = integrate.quad(integrand, 0.0, z, epsabs=1e-13, epsrel=1e-12,
                                 limit=200)
    return float(value)


def am_critical_value(kappa, alpha=0.05):
    """z_{alpha, kappa} for q = 1: between sqrt(chi2_1) and sqrt(chi2_2)."""
    from scipy import optimize, stats

    lo = float(np.sqrt(stats.chi2.ppf(1.0 - alpha, 1)))
    if kappa <= 0:
        return lo
    hi = float(np.sqrt(stats.chi2.ppf(1.0 - alpha, 2)))
    target = 1.0 - alpha
    # the quantile rises with kappa from lo to hi; bracket just outside both
    return float(optimize.brentq(lambda z: am_cdf(z, kappa) - target,
                                 lo * (1 - 1e-9), hi * (1 + 1e-9),
                                 xtol=1e-13, rtol=1e-13))


def curvature(lam1, sigma):
    """KSS's q = 1 curvature from lambda_1 and the 2x2 covariance."""
    v_b, v_t, cov = float(sigma[0, 0]), float(sigma[1, 1]), float(sigma[0, 1])
    rho2 = cov * cov / (v_b * v_t)
    return 2.0 * abs(lam1) * v_b / np.sqrt(v_t * (1.0 - rho2))


def am_interval(lam1, b1, theta1, sigma, alpha=0.05, grid=4096):
    """(lower, upper, info) for the q = 1 weak-identification interval.

    `sigma` is the 2x2 covariance of (b1-hat, theta1-hat), which must be
    positive definite. The extremes of lambda_1 b^2 + t over the ellipse lie on
    its boundary -- the map is linear in t with coefficient one, so any interior
    point can be improved by moving t -- so the problem is one-dimensional in
    the boundary angle. The objective there is a trigonometric polynomial of
    degree two, with at most four stationary points, so a dense grid followed by
    a bounded Brent refinement finds both global extremes. KSS's closed form
    (Appendix C.6.2) solves the same first-order condition as a quartic; the
    tests check the two agree.
    """
    from scipy import optimize

    sigma = np.asarray(sigma, dtype=np.float64)
    kappa = curvature(lam1, sigma)
    z = am_critical_value(kappa, alpha)
    L = np.linalg.cholesky(sigma)

    def g(t):
        b = b1 + z * L[0, 0] * np.cos(t)
        th = theta1 + z * (L[1, 0] * np.cos(t) + L[1, 1] * np.sin(t))
        return lam1 * b * b + th

    ts = np.linspace(0.0, 2.0 * np.pi, grid, endpoint=False)
    values = g(ts)
    step = ts[1] - ts[0]

    def refine(index, sign):
        t0 = ts[index]
        res = optimize.minimize_scalar(lambda t: sign * g(t),
                                       bounds=(t0 - step, t0 + step),
                                       method="bounded",
                                       options={"xatol": 1e-14})
        return float(g(res.x)) if sign > 0 else float(g(res.x))

    lower = min(refine(int(np.argmin(values)), 1.0), float(values.min()))
    upper = max(refine(int(np.argmax(values)), -1.0), float(values.max()))
    rho = float(sigma[0, 1] / np.sqrt(sigma[0, 0] * sigma[1, 1]))
    return lower, upper, {"kappa": float(kappa), "z": float(z), "rho": rho}

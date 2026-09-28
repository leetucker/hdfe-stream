"""GLM families for `fepois_stream` and `feglm_stream`: Poisson (log link),
logit and probit.

Everything here works on one chunk of rows at a time, as numpy arrays. Each
family gives what an IRLS step needs from the linear predictor eta: the mean
mu, the IRLS weight W = 1 / (g'(mu)^2 V(mu)) and the working-response update
u = (y - mu) g'(mu), so that the working response is z = eta + u. Deviance
and log-likelihood are per row; the caller applies the user's weights.

The binary families compute 1 - mu from -eta rather than by subtraction, so
rows far in either tail keep their precision; a row whose weight underflows
to zero gets u = 0 (it carries no weight, and its z would be inf).
"""

from __future__ import annotations

import numpy as np
from scipy import special

_SQRT_2PI = np.sqrt(2.0 * np.pi)


class _Family:
    name = ""
    method = ""          # pyfixest's name for the estimator (reporting)

    def check(self, lo, hi, n_nonbinary):
        """Validate the outcome from its range and its count of values other
        than 0 and 1."""

    def init_eta(self, y, ybar):
        """Starting linear predictor, as pyfixest starts it."""
        raise NotImplementedError

    def mu(self, eta):
        raise NotImplementedError

    def irls(self, y, eta):
        """(mu, W, u) at eta."""
        raise NotImplementedError

    def deviance(self, y, eta, mu):
        raise NotImplementedError

    def loglik(self, y, eta, mu):
        raise NotImplementedError

    def loglik_null(self, sy, sw, slg):
        """Log-likelihood of the constant-only model, from sum w y, sum w and
        (Poisson) sum w log(y!)."""
        raise NotImplementedError


class Poisson(_Family):
    name, method = "poisson", "fepois"

    def check(self, lo, hi, n_nonbinary):
        if lo < 0:
            raise ValueError("Poisson regression needs a nonnegative dependent variable; "
                             f"its minimum is {lo}")
        if hi <= 0:
            raise ValueError("the dependent variable is zero on every row")

    def init_eta(self, y, ybar):
        return np.log((y + ybar) / 2.0)

    def mu(self, eta):
        return np.exp(eta)

    def irls(self, y, eta):
        mu = np.exp(eta)
        u = np.where(mu > 0, y / np.where(mu > 0, mu, 1.0) - 1.0, 0.0)
        return mu, mu, u

    def deviance(self, y, eta, mu):
        # 2 (y log(y / mu) - (y - mu)), with log mu = eta
        return 2.0 * (special.xlogy(y, y) - y * eta - y + mu)

    def loglik(self, y, eta, mu):
        return y * eta - mu - special.gammaln(y + 1.0)

    def loglik_null(self, sy, sw, slg):
        ybar = sy / sw
        return sy * np.log(ybar) - sw * ybar - slg


class _Binary(_Family):

    def check(self, lo, hi, n_nonbinary):
        if n_nonbinary:
            raise ValueError(f"{self.name} needs a dependent variable of 0s and 1s; "
                             f"{n_nonbinary:,} rows have other values")
        if lo == hi:
            raise ValueError(f"the dependent variable is {lo:g} on every row")

    def init_eta(self, y, ybar):
        return np.zeros_like(y)            # mu = 1/2

    def _cdf(self, eta):
        """(mu, 1 - mu, log mu, log(1 - mu))."""
        raise NotImplementedError

    def deviance(self, y, eta, mu):
        _, _, lp, lq = self._cdf(eta)
        return -2.0 * np.where(y > 0, lp, lq)

    def loglik(self, y, eta, mu):
        return -0.5 * self.deviance(y, eta, mu)

    def loglik_null(self, sy, sw, slg):
        p = sy / sw
        return sy * np.log(p) + (sw - sy) * np.log1p(-p)


class Logit(_Binary):
    name, method = "logit", "feglm-logit"

    def _cdf(self, eta):
        return (special.expit(eta), special.expit(-eta),
                special.log_expit(eta), special.log_expit(-eta))

    def mu(self, eta):
        return special.expit(eta)

    def irls(self, y, eta):
        p, q = special.expit(eta), special.expit(-eta)
        W = p * q
        resid = np.where(y > 0, q, -p)             # y - mu, without cancellation
        u = np.where(W > 0, resid / np.where(W > 0, W, 1.0), 0.0)
        return p, W, u


class Probit(_Binary):
    name, method = "probit", "feglm-probit"

    def _cdf(self, eta):
        return (special.ndtr(eta), special.ndtr(-eta),
                special.log_ndtr(eta), special.log_ndtr(-eta))

    def mu(self, eta):
        return special.ndtr(eta)

    def irls(self, y, eta):
        p, q = special.ndtr(eta), special.ndtr(-eta)
        pdf = np.exp(-0.5 * eta * eta) / _SQRT_2PI     # 1 / g'(mu)
        V = p * q
        W = np.where(V > 0, pdf * pdf / np.where(V > 0, V, 1.0), 0.0)
        resid = np.where(y > 0, q, -p)
        u = np.where((W > 0) & (pdf > 0), resid / np.where(pdf > 0, pdf, 1.0), 0.0)
        return p, W, u


FAMILIES = {"poisson": Poisson, "logit": Logit, "probit": Probit}


def get_family(name):
    try:
        return FAMILIES[str(name).lower()]()
    except KeyError:
        raise ValueError(f"unknown family {name!r}; use one of {sorted(FAMILIES)}"
                         + (" (for a linear model use feols_stream)"
                            if str(name).lower() == "gaussian" else "")) from None

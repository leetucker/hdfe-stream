"""Log-likelihoods, AIC/BIC and the pseudo R-squareds: against statsmodels,
the dummy-variable fits, and the groups' means."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from scipy.special import gammaln, xlogy

import hdfe_stream as hs

sm = pytest.importorskip("statsmodels.api")


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    n = 3000
    g, b = rng.integers(0, 30, n), rng.integers(0, 7, n)
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    a, c = rng.normal(0, 0.7, 30)[g], rng.normal(0, 0.5, 7)[b]
    y = 1 + 0.5 * x1 - 0.3 * x2 + a + rng.normal(size=n)
    yb = (rng.random(n) < 1 / (1 + np.exp(-(0.3 * x1 - 0.5 * x2 + a + c)))).astype(float)
    cnt = rng.poisson(np.exp(0.2 + 0.3 * x1 + a)).astype(float)
    return pl.DataFrame({"y": y, "yb": yb, "c": cnt, "x1": x1, "x2": x2, "g": g, "b": b,
                         "w": rng.uniform(0.5, 2, n)})


def _X(df, *cols):
    return sm.add_constant(df.select(cols).to_numpy())


def test_ols_matches_statsmodels(data):
    X, y = _X(data, "x1", "x2"), data["y"].to_numpy()
    r, s = hs.feols_stream("y ~ x1 + x2", data, verbose=False), sm.OLS(y, X).fit()
    assert r.loglik == pytest.approx(s.llf, abs=1e-8)
    assert (r.aic, r.bic) == pytest.approx((s.aic, s.bic), abs=1e-8)
    assert np.isnan(r.loglik_null) and np.isnan(r.pseudo_r2)
    assert {"aic", "bic", "loglik"} <= set(r.summary_dict())


def test_weighted_ols_matches_statsmodels(data):
    X, y = _X(data, "x1", "x2"), data["y"].to_numpy()
    r = hs.feols_stream("y ~ x1 + x2", data, weights="w", verbose=False)
    s = sm.WLS(y, X, weights=data["w"].to_numpy()).fit()
    assert r.loglik == pytest.approx(s.llf, abs=1e-8)
    assert r.aic == pytest.approx(s.aic, abs=1e-8)


def test_fixed_effects_count_as_parameters(data):
    """With an absorbed fixed effect, the criteria are the dummy fit's."""
    D = np.column_stack([data.select("x1", "x2").to_numpy(),
                         (data["g"].to_numpy()[:, None] == np.arange(30)).astype(float)])
    r = hs.feols_stream("y ~ x1 + x2 | g", data, verbose=False)
    s = sm.OLS(data["y"].to_numpy(), D).fit()
    assert r.k_fe == 30
    assert (r.loglik, r.aic, r.bic) == pytest.approx((s.llf, s.aic, s.bic), abs=1e-8)


def test_logit_matches_statsmodels(data):
    X, y = _X(data, "x1", "x2"), data["yb"].to_numpy()
    r = hs.feglm_stream("yb ~ x1 + x2", data, family="logit", verbose=False)
    s = sm.Logit(y, X).fit(disp=0)
    assert r.loglik == pytest.approx(s.llf, abs=1e-8)
    assert r.loglik_null == pytest.approx(s.llnull, abs=1e-5)    # statsmodels iterates it
    assert (r.aic, r.bic) == pytest.approx((s.aic, s.bic), abs=1e-8)
    assert np.isnan(r.pseudo_r2_within)


def test_within_pseudo_r2_one_fixed_effect(data):
    """One fixed effect: the null with only it is the groups' means, free."""
    r = hs.feglm_stream("yb ~ x1 + x2 | g", data, family="logit", verbose=False)
    p = data.group_by("g").agg(pl.col("yb").mean().alias("p"), pl.len().alias("n"))
    ll = sum(n * (xlogy(q, q) + xlogy(1 - q, 1 - q)) for q, n in zip(p["p"], p["n"]))
    assert r.loglik_fe_null == pytest.approx(ll, abs=1e-8)
    assert r.pseudo_r2_within == pytest.approx(1 - r.loglik / ll)
    assert 0 < r.pseudo_r2_within < r.pseudo_r2
    assert "within pseudo R2" in r.summary_text()

    rp = hs.fepois_stream("c ~ x1 | g", data, verbose=False)
    q = data.group_by("g").agg(pl.col("c").sum().alias("s"), pl.len().alias("n"))
    llp = (sum(xlogy(s, s / n) - s for s, n in zip(q["s"], q["n"]))
           - gammaln(data["c"].to_numpy() + 1).sum())
    assert rp.loglik_fe_null == pytest.approx(llp, abs=1e-7)


def test_within_pseudo_r2_two_fixed_effects_on_request(data):
    """Two fixed effects: a fit of its own, made only on request; it is the
    model with the second effect's indicators and no covariates."""
    fml = "yb ~ x1 + x2 | g + b"
    assert np.isnan(hs.feglm_stream(fml, data, family="logit",
                                    verbose=False).pseudo_r2_within)
    r = hs.feglm_stream(fml, data, family="logit", verbose=False, within_pseudo_r2=True)
    r0 = hs.feglm_stream("yb ~ C(b) | g", data, family="logit", verbose=False)
    assert r.loglik_fe_null == pytest.approx(r0.loglik, abs=1e-6)
    assert r.pseudo_r2_within == pytest.approx(1 - r.loglik / r0.loglik, rel=1e-8)

"""feglm_stream and fepois_stream with no fixed effect or one: Newton's
method on core.newton's fused passes before IRLS's final step gives the fit
IRLS alone gives when run to a tight tolerance (at IRLS's default tolerance,
1e-8 on the deviance, its coefficients can differ from the maximum by about
1e-6 relative; Newton's step stops on the decrement and is closer)."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from scipy import special

from hdfe_stream import StreamingGLM, feglm_stream, fepois_stream

pytestmark = pytest.mark.filterwarnings("ignore:.*dropped:UserWarning")


@pytest.fixture(autouse=True)
def _streaming_glm(monkeypatch):
    """These tests are of StreamingGLM's Newton start: the fast path
    (`glm_fast`, tested in test_glm_fast.py) is off."""
    from hdfe_stream import glm_fast
    monkeypatch.setattr(glm_fast, "ENABLED", False)


def _data(family, n=4000, seed=7):
    r = np.random.default_rng(seed)
    df = pl.DataFrame({"x1": r.normal(size=n), "x2": r.normal(size=n),
                       "c": r.integers(0, 8, n), "g": r.integers(0, 60, n),
                       "cl": r.integers(0, 40, n), "w": r.uniform(0.5, 2.0, n),
                       "lexp": r.normal(0, 0.3, n)})
    eta = (0.5 * df["x1"].to_numpy() - 0.3 * df["x2"].to_numpy()
           + 0.2 * (df["c"].to_numpy() % 3) + 0.4 * np.sin(df["g"].to_numpy()) - 0.3)
    if family == "poisson":
        y = r.poisson(np.exp(eta + df["lexp"].to_numpy()))
    elif family == "logit":
        y = (r.random(n) < special.expit(eta)).astype(int)
    else:
        y = (r.random(n) < special.ndtr(eta)).astype(int)
    return df.with_columns(pl.Series("y", y))


def _fit(family, fml, df, **kw):
    if family == "poisson":
        return fepois_stream(fml, df, verbose=False, **kw)
    return feglm_stream(fml, df, family, verbose=False, **kw)


@pytest.mark.parametrize("family", ["logit", "probit", "poisson"])
@pytest.mark.parametrize("fe", ["", " | g"])
@pytest.mark.parametrize("vcov", ["iid", "hetero", {"CRV1": "cl"}])
def test_newton_start_gives_the_irls_fit(family, fe, vcov, monkeypatch):
    df = _data(family)
    fml = "y ~ x1 + x2 + C(c)" + fe
    kw = dict(vcov=vcov, weights="w")
    if family == "poisson":
        kw["offset"] = "lexp"
    fast = _fit(family, fml, df, **kw)
    assert "newton" in fast.diagnostics and fast.diagnostics["newton"]["converged"]
    assert fast.diagnostics["irls"]["iterations"] <= 2
    monkeypatch.setattr(StreamingGLM, "newton", False)
    slow = _fit(family, fml, df, iwls_tol=1e-15, iwls_maxiter=100, **kw)
    assert "newton" not in slow.diagnostics
    assert fast.coefnames == slow.coefnames
    np.testing.assert_allclose(fast.beta, slow.beta, rtol=1e-7, atol=1e-8)
    np.testing.assert_allclose(fast.se, slow.se, rtol=1e-6)
    np.testing.assert_allclose(fast.loglik, slow.loglik, rtol=1e-12)
    assert fast.vcov_type == slow.vcov_type and fast.df_t == slow.df_t
    a = fast.fixef("g").collect().sort("g") if fe else None
    if fe:
        b = slow.fixef("g").collect().sort("g")
        np.testing.assert_allclose(a["fe_g"].to_numpy(), b["fe_g"].to_numpy(), atol=1e-6)


def test_collinear_covariate_dropped_as_by_irls(monkeypatch):
    df = _data("logit").with_columns(x3=2 * pl.col("x1") - pl.col("x2"))
    fast = feglm_stream("y ~ x1 + x2 + x3 | g", df, "logit", verbose=False)
    monkeypatch.setattr(StreamingGLM, "newton", False)
    slow = feglm_stream("y ~ x1 + x2 + x3 | g", df, "logit", verbose=False, iwls_tol=1e-15,
                        iwls_maxiter=100)
    assert fast.collin_vars == slow.collin_vars and fast.collin_vars
    np.testing.assert_allclose(fast.beta, slow.beta, rtol=1e-6, atol=1e-8)


def test_two_fixed_effects_keep_irls():
    df = _data("logit")
    fit = feglm_stream("y ~ x1 + x2 | g + c", df, "logit", verbose=False)
    assert "newton" not in fit.diagnostics

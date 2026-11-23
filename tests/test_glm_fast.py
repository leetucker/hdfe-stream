"""The fast path of feglm_stream and fepois_stream (`glm_fast`): the result
StreamingGLM gives, to rounding -- estimates, every vcov flavor, the fit
statistics, the files, sample() and predict()."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from scipy import special

from hdfe_stream import feglm_stream, fepois_stream, glm_fast

pytestmark = pytest.mark.filterwarnings("ignore:.*dropped:UserWarning",
                                        "ignore:invalid value:RuntimeWarning")


def _data(family, n=4000, seed=7):
    r = np.random.default_rng(seed)
    df = pl.DataFrame({"x1": r.normal(size=n), "x2": r.normal(size=n),
                       "c": r.integers(0, 8, n), "g": r.integers(0, 60, n),
                       "cl": r.integers(0, 40, n), "w": r.uniform(0.5, 2.0, n),
                       "fw": r.integers(1, 4, n), "lexp": r.normal(0, 0.3, n)})
    eta = (0.5 * df["x1"].to_numpy() - 0.3 * df["x2"].to_numpy()
           + 0.2 * (df["c"].to_numpy() % 3) + 0.4 * np.sin(df["g"].to_numpy()) - 0.3)
    if family == "poisson":
        y = r.poisson(np.exp(eta + df["lexp"].to_numpy()))
    elif family == "logit":
        y = (r.random(n) < special.expit(eta)).astype(int)
    else:
        y = (r.random(n) < special.ndtr(eta)).astype(int)
    df = df.with_columns(pl.Series("y", y))
    # a group of one row (a singleton) and a group whose outcome never varies
    # (or, for Poisson, is always zero): both dropped, for different reasons
    return pl.concat([df, df.head(1).with_columns(g=pl.lit(999, pl.Int64))]).with_columns(
        pl.when(pl.col("g") == 59).then(0).otherwise(pl.col("y")).alias("y"))


def _both(family, fml, df, **kw):
    fn = ((lambda f, d, **k: fepois_stream(f, d, **k)) if family == "poisson"
          else (lambda f, d, **k: feglm_stream(f, d, family, **k)))
    glm_fast.ENABLED = True
    try:
        fast = fn(fml, df, verbose=False, **kw)
        glm_fast.ENABLED = False
        slow = fn(fml, df, verbose=False, iwls_tol=1e-14, iwls_maxiter=100, **kw)
    finally:
        glm_fast.ENABLED = True
    assert fast.diagnostics.get("fit") == "newton"
    assert slow.diagnostics.get("fit") != "newton"
    return fast, slow


def _same(fast, slow, rtol=1e-8):
    assert fast.coefnames == slow.coefnames and fast.fml == slow.fml
    assert fast.collin_vars == slow.collin_vars
    assert (fast.n_obs, fast.k_fe, fast.df_t, fast.vcov_type) == \
           (slow.n_obs, slow.k_fe, slow.df_t, slow.vcov_type)
    assert fast.n_levels == slow.n_levels and fast.n_clusters == slow.n_clusters
    np.testing.assert_allclose(fast.beta, slow.beta, rtol=rtol, atol=1e-10)
    assert set(fast.all_vcovs) == set(slow.all_vcovs)
    for key, (V, df) in slow.all_vcovs.items():
        np.testing.assert_allclose(fast.all_vcovs[key][0], V, rtol=rtol,
                                   atol=rtol * np.nanmax(np.abs(V), initial=0.0),
                                   err_msg=key)
        assert fast.all_vcovs[key][1] == df
    for name in ["loglik", "deviance", "loglik_null", "pseudo_r2", "aic", "bic",
                 "loglik_fe_null", "pseudo_r2_within"]:
        np.testing.assert_allclose(getattr(fast, name), getattr(slow, name), rtol=1e-10,
                                   err_msg=name)


@pytest.mark.parametrize("family", ["logit", "probit", "poisson"])
@pytest.mark.parametrize("fe", ["", " | g"])
def test_the_result_of_streaming_glm(family, fe):
    df = _data(family)
    kw = dict(vcov={"CRV1": "cl"}, cluster=["c", "cl+c"], weights="w")
    if family == "poisson":
        kw["offset"] = "lexp"
    fast, slow = _both(family, "y ~ x1 + x2 + C(c)" + fe, df, **kw)
    tight = family != "probit"          # IRLS's probit (Fisher scoring) stops near 1e-8
    _same(fast, slow, rtol=1e-8 if tight else 1e-6)
    # the files: residuals (in each fit's own row order) and fixed effects
    a = fast.resid().collect().sort("row_id")
    b = slow.resid().collect().sort("row_id")
    assert a.columns == b.columns
    for c in a.columns:
        np.testing.assert_allclose(a[c].to_numpy().astype(float),
                                   b[c].to_numpy().astype(float), rtol=1e-6,
                                   atol=1e-9 if tight else 1e-6,
                                   err_msg=c)
    if fe:
        fa, fb = fast.fixef("g").collect().sort("g"), slow.fixef("g").collect().sort("g")
        assert fa.columns == fb.columns and fa["n_obs"].to_list() == fb["n_obs"].to_list()
        np.testing.assert_allclose(fa["fe_g"].to_numpy(), fb["fe_g"].to_numpy(),
                                   atol=1e-7 if tight else 1e-6)
    # which rows were used, and why not
    sa, sb = fast.sample().collect(), slow.sample().collect()
    assert sa["dropped_because"].to_list() == sb["dropped_because"].to_list()
    # predictions on new data
    pa_, pb = (r.predict(df, type="response").collect()["prediction"].to_numpy()
               for r in (fast, slow))
    np.testing.assert_allclose(pa_, pb, rtol=1e-6, atol=1e-12)


def test_frequency_weights_and_iid():
    df = _data("logit")
    fast, slow = _both("logit", "y ~ x1 + x2 | g", df, vcov="hetero", weights="fw",
                       weights_type="fweights")
    _same(fast, slow)


def test_collinear_and_dropped():
    df = _data("poisson").with_columns(x3=2 * pl.col("x1") - pl.col("x2"))
    fast, slow = _both("poisson", "y ~ x1 + x2 + x3 | g", df, vcov="iid")
    assert fast.collin_vars
    _same(fast, slow)


def test_streaming_glm_otherwise():
    """Two fixed effects, multiple estimation, CRV3 (an error for GLMs, as
    before) and IRLS-only options go to StreamingGLM."""
    df = _data("logit")
    for fml, kw in [("y ~ x1 | g + c", {}), ("y ~ sw(x1, x2) | g", {}),
                    ("y ~ x1 | g", {"bias_correction": True}),
                    ("y ~ x1 | g", {"separation_check": False})]:
        r = feglm_stream(fml, df, "logit", verbose=False, **kw)
        for one in (list(r.all_fitted_models.values()) if hasattr(r, "all_fitted_models")
                    else [r]):
            assert one.diagnostics.get("fit") != "newton", fml
    with pytest.raises(ValueError, match="CRV3"):
        feglm_stream("y ~ x1 | g", df, "logit", vcov={"CRV3": "cl"}, verbose=False)

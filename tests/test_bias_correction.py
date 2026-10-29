"""The incidental-parameter bias correction (`bias_correction=`) against R's
alpaca (`biasCorr`), and its checks of the panel design."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

import hdfe_stream as hs


def _panel(N=300, T=8, seed=11):
    """A dynamic logit panel: unit and period effects, a regressor correlated
    with the unit effect, and the lagged outcome. The alpaca numbers below
    are for this data (alpaca 0.3.5, dev.tol = center.tol = 1e-13, the rows
    sorted by unit and period), shuffled here."""
    rng = np.random.default_rng(seed)
    i, t = np.repeat(np.arange(N), T), np.tile(np.arange(T), N)
    x1, x2 = rng.normal(size=N * T), rng.normal(size=N * T)
    a, g = rng.normal(0, 1, N)[i], rng.normal(0, 0.5, T)[t]
    y, ylag = np.zeros(N * T, int), np.zeros(N * T)
    for r in range(N * T):
        yl = y[r - 1] if t[r] > 0 else 0
        ylag[r] = yl
        eta = 0.5 * yl + 0.8 * x1[r] - 0.5 * x2[r] + a[r] + g[r]
        y[r] = rng.random() < 1 / (1 + np.exp(-eta))
    return pl.DataFrame({"y": y, "x1": x1 + 0.3 * a, "x2": x2, "ylag": ylag, "i": i,
                         "t": t}).sample(fraction=1.0, shuffle=True, seed=1)


@pytest.fixture(scope="module")
def panel():
    return _panel()


# case: formula, family, bias_correction, {name: (mle, corrected, se)}
ALPACA = {
    "two-way logit": ("y ~ x1 + x2 | i + t", "logit", True,
                      {"x1": (1.0366598893, 0.8667803619, 0.0648299622),
                       "x2": (-0.6316070622, -0.5296597373, 0.0621694422)}),
    "one-way logit": ("y ~ x1 + x2 | i", "logit", True,
                      {"x1": (0.9737443478, 0.8338862882, 0.0623897612),
                       "x2": (-0.5476233233, -0.4696801339, 0.0584093465)}),
    "two-way probit": ("y ~ x1 + x2 | i + t", "probit", True,
                       {"x1": (0.6003604958, 0.5103264176, 0.0367010642),
                        "x2": (-0.3598700164, -0.3062327078, 0.0357240260)}),
    "dynamic logit, L = 2": ("y ~ ylag + x1 | i + t", "logit", {"time": "t", "lags": 2},
                             {"ylag": (-0.2675879931, 0.4267684900, 0.1299525017),
                              "x1": (0.9780102202, 0.8376368269, 0.0634796000)}),
}


@pytest.mark.filterwarnings("ignore:.*separation")
@pytest.mark.parametrize("case", list(ALPACA))
def test_matches_alpaca(panel, case):
    fml, family, bc, ref = ALPACA[case]
    f = hs.feglm_stream(fml, panel, family=family, bias_correction=bc, iwls_tol=1e-14,
                        verbose=False)
    d = f.diagnostics["bias_correction"]
    # alpaca's variance is the inverse Hessian; hdfe-stream's iid one has
    # pyfixest's (N - 1) / (N - K)
    ssc = np.sqrt((f.n_obs - 1) / (f.n_obs - len(f.beta) - f.k_fe))
    for name, (mle, corrected, se) in ref.items():
        assert d["beta_uncorrected"][name] == pytest.approx(mle, abs=1e-6)
        assert f.coef()[name] == pytest.approx(corrected, abs=1e-6)
        assert f.se[f.coefnames.index(name)] / ssc == pytest.approx(se, rel=1e-5)
        assert d["bias"][name] == pytest.approx(mle - corrected, abs=1e-6)
    assert d["refit"]["converged"]
    assert "bias-corrected" in f.summary_text()


@pytest.mark.filterwarnings("ignore:.*separation")
def test_the_refit_is_consistent(panel):
    """The fixed effects, residuals and predictions are those of the
    corrected coefficients, re-estimated with them held: the working
    score of the fixed effects is zero there, so each unit's fitted
    probabilities sum to its outcomes."""
    f = hs.feglm_stream("y ~ x1 + x2 | i", panel, family="logit", bias_correction=True,
                        iwls_tol=1e-14, verbose=False)
    r = f.resid().collect()
    by = r.group_by("i").agg((pl.col("y") - pl.col("fitted")).sum().abs().alias("s"))
    assert by["s"].max() < 1e-8
    # the residual file's X beta is at the corrected coefficients
    np.testing.assert_allclose((r["eta"] - r["fe_i"]).to_numpy(), r["xb"].to_numpy(), atol=1e-8)
    src = (panel.with_row_index("row_id")
                .join(r.select(pl.col("row_id").cast(pl.UInt32), "xb"), on="row_id"))
    np.testing.assert_allclose(src["xb"].to_numpy(),
                               (src["x1"] * f.coef()["x1"] + src["x2"] * f.coef()["x2"]).to_numpy(),
                               atol=1e-8)


@pytest.mark.filterwarnings("ignore:.*separation")
def test_checks(panel):
    with pytest.raises(ValueError, match="Poisson"):
        hs.fepois_stream("y ~ x1 | i", panel, bias_correction=True, verbose=False)
    three = panel.with_columns(k=pl.col("i") % 7)
    with pytest.raises(ValueError, match="one or two fixed effects"):
        hs.feglm_stream("y ~ x1 | i + t + k", three, family="logit", bias_correction=True,
                        verbose=False)
    with pytest.raises(ValueError, match="needs two fixed effects"):
        hs.feglm_stream("y ~ x1 | i", panel, family="logit", verbose=False,
                        bias_correction={"lags": 1})
    with pytest.raises(ValueError, match="not one of the fixed effects"):
        hs.feglm_stream("y ~ x1 | i + t", panel, family="logit", verbose=False,
                        bias_correction={"time": "year", "lags": 1})
    with pytest.raises(ValueError, match="stream='i'"):
        hs.feglm_stream("y ~ x1 | i + t", panel, family="logit", verbose=False, stream="t",
                        bias_correction={"time": "t", "lags": 1})
    with pytest.raises(ValueError, match="True, or"):
        hs.feglm_stream("y ~ x1 | i + t", panel, family="logit", verbose=False,
                        bias_correction={"period": "t"})
    # a period that is not numeric or temporal
    s = panel.with_columns(t=pl.col("t").cast(pl.String))
    with pytest.raises(ValueError, match="numeric"):
        hs.feglm_stream("y ~ x1 | i + t", s, family="logit", verbose=False,
                        bias_correction={"time": "t", "lags": 1})
    # two rows for some unit and period
    dup = pl.concat([panel, panel.head(50)])
    with pytest.raises(ValueError, match="at most one row per unit and period"):
        hs.feglm_stream("y ~ x1 | i + t", dup, family="logit", verbose=False,
                        bias_correction={"time": "t", "lags": 1})


@pytest.mark.filterwarnings("ignore:.*separation")
@pytest.mark.filterwarnings("ignore:.*very short panels")
def test_refit_from_zero_when_the_warm_start_fails():
    """Two periods: the correction is large (and poor), and re-estimating the
    effects from the first fit's overshoots beyond what step-halving
    recovers; the refit then starts them from zero."""
    rng = np.random.default_rng(1)
    N = 5000
    i = np.repeat(np.arange(N), 2)
    a = rng.normal(0, 1, N)[i]
    x1, x2 = 0.5 * a + rng.normal(size=2 * N), rng.normal(size=2 * N)
    y = (rng.random(2 * N) < 1 / (1 + np.exp(-(a + x1 - 0.5 * x2)))).astype(int)
    f = hs.feglm_stream("y ~ x1 + x2 | i", pl.DataFrame({"y": y, "x1": x1, "x2": x2, "i": i}),
                        family="logit", bias_correction=True, verbose=False)
    assert f.diagnostics["bias_correction"]["refit"]["converged"]
    r = f.resid().collect()
    by = r.group_by("i").agg((pl.col("y") - pl.col("fitted")).sum().abs().alias("s"))
    assert by["s"].max() < 1e-6


@pytest.mark.filterwarnings("ignore:.*separation")
def test_warnings():
    """Very short panels; and, with lags, gaps and units no longer than L."""
    short = _panel(N=400, T=3, seed=2)
    with pytest.warns(UserWarning, match="very short panels"):
        hs.feglm_stream("y ~ x1 | i + t", short, family="logit", bias_correction=True,
                        verbose=False)
    gappy = _panel(N=200, T=8, seed=3).filter(~((pl.col("i") % 5 == 0) & (pl.col("t") == 3)))
    with pytest.warns(UserWarning, match="gaps"):
        f = hs.feglm_stream("y ~ ylag + x1 | i + t", gappy, family="logit", verbose=False,
                            bias_correction={"time": "t", "lags": 1})
    assert f.diagnostics["bias_correction"]["units_with_gaps"] > 0
    with pytest.warns(UserWarning, match="no more periods than"):
        hs.feglm_stream("y ~ ylag + x1 | i + t", short, family="logit", verbose=False,
                        bias_correction={"time": "t", "lags": 3})

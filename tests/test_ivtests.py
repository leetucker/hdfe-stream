"""IV diagnostics (`core.ivtests`) on feols_stream's 2SLS fits.

The reference is dense: the data demeaned by their one fixed effect, the
exogenous regressor partialled out, and every statistic computed from the
rows -- the iid ones from their canonical-correlation definitions, the
robust ones from the scores' meats -- with ivreg2's formulas.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from scipy import stats

from hdfe_stream import StreamingHDFE, feols_stream
from hdfe_stream.core.ivtests import hansen_j, identification, kp_rk

N, G = 4_000, 80


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(11)
    g = rng.integers(0, G, N)
    a = rng.normal(0, 1, G)[g]
    z = rng.normal(0, 1, (N, 3))
    x = rng.normal(0, 1, N)
    v = rng.normal(0, 1, (N, 2))
    e1 = 0.8 * z[:, 0] + 0.3 * z[:, 1] + 0.4 * x + v[:, 0] + a
    e2 = 0.5 * z[:, 2] + 0.2 * z[:, 0] + v[:, 1] - a
    het = np.exp(0.5 * z[:, 1])
    y = 1.0 * e1 - 0.5 * e2 + 0.3 * x + a + het * (0.6 * v[:, 0] + rng.normal(0, 1, N))
    return pl.DataFrame({"y": y, "e1": e1, "e2": e2, "x": x, "z1": z[:, 0], "z2": z[:, 1],
                         "z3": z[:, 2], "g": g, "cl": g % 20,
                         "w": rng.uniform(0.5, 2.0, N)})


def demean(df, cols, w):
    out = {}
    for c in cols:
        s = df.select((pl.col(c) * w).sum().over("g") / w.sum().over("g")).to_series()
        out[c] = (df[c] - s).to_numpy()
    return out


def reference(df, endog, insts, vcov, weights=None):
    w = pl.Series(np.ones(df.height)) if weights is None else df[weights]
    d = demean(df.with_columns(_w=w), ["y", "x", *endog, *insts], pl.col("_w"))
    wv = w.to_numpy()
    X2 = d["x"][:, None]
    Y = np.column_stack([d[c] for c in endog])
    Z1 = np.column_stack([d[c] for c in insts])
    Zall = np.column_stack([Z1, X2])
    Xall = np.column_stack([Y, X2])

    def wcross(a, b):
        return a.T @ (wv[:, None] * b if b.ndim == 2 else wv * b)
    b = np.linalg.solve(wcross(Xall, Zall) @ np.linalg.solve(wcross(Zall, Zall), wcross(Zall, Xall)),
                        wcross(Xall, Zall) @ np.linalg.solve(wcross(Zall, Zall), wcross(Zall, d["y"])))
    e = d["y"] - Xall @ b
    gz = np.linalg.solve(wcross(X2, X2), wcross(X2, Z1))
    gy = np.linalg.solve(wcross(X2, X2), wcross(X2, Y))
    Z1t, Yt = Z1 - X2 @ gz, Y - X2 @ gy
    ZZ, ZY, YY = wcross(Z1t, Z1t), wcross(Z1t, Yt), wcross(Yt, Yt)
    Pi = np.linalg.solve(ZZ, ZY)
    Vt = Yt - Z1t @ Pi
    n = len(e)
    if vcov == "iid":
        # Anderson's LM: N times the smallest squared canonical correlation;
        # Cragg-Donald: the smallest eigenvalue of (V'V/N)^-1 Pi'Z'Z Pi
        rho = np.linalg.eigvals(np.linalg.solve(YY, ZY.T @ np.linalg.solve(ZZ, ZY))).real
        cd = np.linalg.eigvals(np.linalg.solve(wcross(Vt, Vt) / n, Pi.T @ ZZ @ Pi)).real
        return {"lm": n * rho.min(), "wald": cd.min()}
    s = np.hstack([Zall * (wv * e)[:, None],
                   (Yt[:, :, None] * Z1t[:, None, :]).reshape(n, -1) * wv[:, None],
                   (Vt[:, :, None] * Z1t[:, None, :]).reshape(n, -1) * wv[:, None]])
    if vcov == "hetero":
        Om = s.T @ s
        ng = None
    else:
        codes = df["cl"].to_numpy()
        sc = np.vstack([s[codes == c].sum(0) for c in np.unique(codes)])
        Om = sc.T @ sc
        ng = len(np.unique(codes))
    q, m1 = Zall.shape[1], Z1.shape[1] * Y.shape[1]
    out = identification(ZZ, ZY, YY, Om[q:q + m1, q:q + m1], Om[q + m1:, q + m1:],
                         n_obs=n, n_inst=q, n_clusters=ng, dof_extra=G if ng is None else 1,
                         endog=endog)
    out["J"] = hansen_j(wcross(Zall, Xall), wcross(Zall, d["y"]), Om[:q, :q])
    return out


@pytest.mark.parametrize("vcov", ["iid", "hetero", {"CRV1": "cl"}])
@pytest.mark.parametrize("weights", [None, "w"])
def test_two_endogenous(data, tmp_path, vcov, weights):
    # pyfixest's formulas take one endogenous regressor; the low-level
    # interface takes any number
    model = {"fml": "y ~ x | g | e1 + e2 ~ z1 + z2 + z3", "y": "y", "x": ["e1", "e2", "x"],
             "iv": {"endog": ["e1", "e2"], "z": ["z1", "z2", "z3", "x"]}}
    est = StreamingHDFE("y", ["e1", "e2", "x", "z1", "z2", "z3"], ["g"], models=[model],
                        weights=weights, workdir=tmp_path, verbose=False)
    fit = est.fit(data.lazy(), vcov=vcov)
    fit = fit[0] if isinstance(fit, list) else fit
    d = fit.iv_diag
    key = vcov if isinstance(vcov, str) else "cluster"
    ref = reference(data, ["e1", "e2"], ["z1", "z2", "z3"], key, weights)
    assert d["kp_lm"]["df"] == 2 and d["kp_wald"]["df"] == 2
    if key == "iid":
        assert d["kp_lm"]["statistic"] == pytest.approx(ref["lm"], rel=1e-7)
        n, L, L1 = N, 4, 3
        assert d["kp_wald"]["statistic"] == pytest.approx(ref["wald"], rel=1e-7)
        assert d["kp_wald"]["F"] == pytest.approx(ref["wald"] / n * (n - L - G) / L1,
                                                  rel=1e-7)
        assert d["overid"]["test"] == "sargan"
    else:
        for name in ("kp_lm", "kp_wald"):
            assert d[name]["statistic"] == pytest.approx(ref[name]["statistic"], rel=1e-7)
        assert d["kp_wald"]["F"] == pytest.approx(ref["kp_wald"]["F"], rel=1e-7)
        for got, want in zip(d["sanderson_windmeijer"], ref["sanderson_windmeijer"]):
            assert got["F"] == pytest.approx(want["F"], rel=1e-7)
        assert d["overid"]["test"] == "hansen_j"
        assert d["overid"]["statistic"] == pytest.approx(ref["J"][0], rel=1e-7)
        assert d["overid"]["df"] == 1
    assert any("weak identification" in line for line in fit._summary_lines())


def test_one_endogenous_is_first_stage(data, tmp_path):
    """With one endogenous regressor, the rk Wald statistic is the
    first-stage Wald test of the excluded instruments (same vcov, before
    small-sample factors), and the SW F is that F with ivreg2's factor."""
    fit = feols_stream("y ~ x | g | e1 ~ z1 + z2 + z3", data, vcov="hetero",
                       workdir=tmp_path, verbose=False)
    d = fit.iv_diag
    fs = fit.first_stage[0]
    idx = [fs.coefnames.index(z) for z in ("z1", "z2", "z3")]
    k_fs = len(fs.coefnames) + fs.k_fe
    Vraw = fs.vcov[np.ix_(idx, idx)] * (N - k_fs) / N
    b = fs.beta[idx]
    assert d["kp_wald"]["statistic"] == pytest.approx(float(b @ np.linalg.solve(Vraw, b)),
                                                      rel=1e-7)
    sw, = d["sanderson_windmeijer"]
    assert sw["F"] == pytest.approx(d["kp_wald"]["F"], rel=1e-10)


def test_kp_rk_scale_invariance():
    """The rk statistic does not depend on the scale of the instruments or
    the regressors."""
    rng = np.random.default_rng(3)
    Z = rng.normal(size=(500, 4))
    Y = Z @ rng.normal(size=(4, 2)) + rng.normal(size=(500, 2))
    Pi = np.linalg.solve(Z.T @ Z, Z.T @ Y)
    V = Y - Z @ Pi
    s = (V[:, :, None] * Z[:, None, :]).reshape(500, -1)
    a, df = kp_rk(Z.T @ Z, Pi, Y.T @ Y, s.T @ s)
    Zs, Ys = Z * 3.0, Y * 0.5
    Pis = np.linalg.solve(Zs.T @ Zs, Zs.T @ Ys)
    ss = ((Ys - Zs @ Pis)[:, :, None] * Zs[:, None, :]).reshape(500, -1)
    b, _ = kp_rk(Zs.T @ Zs, Pis, Ys.T @ Ys, ss.T @ ss)
    assert df == 3 and a == pytest.approx(b, rel=1e-10)
    assert stats.chi2.sf(a, df) < 1e-6


def dense_extra(df, endog, insts, vcov, weights=None):
    """The newer statistics, densely: Shea's partial R^2 by Godfrey's formula
    on the full regressors, the Anderson-Rubin Wald test of beta = 0, and the
    endogeneity C statistic with the OLS residuals' moment covariance."""
    w = pl.Series(np.ones(df.height)) if weights is None else df[weights]
    d = demean(df.with_columns(_w=w), ["y", "x", *endog, *insts], pl.col("_w"))
    wv = w.to_numpy()
    n = len(wv)
    X2 = d["x"][:, None]
    Y = np.column_stack([d[c] for c in endog])
    Z1 = np.column_stack([d[c] for c in insts])
    Z, X, y = np.column_stack([Z1, X2]), np.column_stack([Y, X2]), d["y"]

    def wc(a, b):
        return a.T @ (wv[:, None] * b if b.ndim == 2 else wv * b)
    Xh = Z @ np.linalg.solve(wc(Z, Z), wc(Z, X))
    shea = np.diag(np.linalg.inv(wc(X, X)))[:len(endog)] / \
        np.diag(np.linalg.inv(wc(Xh, Xh)))[:len(endog)]
    codes = df["cl"].to_numpy() if vcov == "cluster" else np.arange(n)

    def meat(sc):
        _, inv = np.unique(codes, return_inverse=True)
        agg = np.zeros((inv.max() + 1, sc.shape[1]))
        np.add.at(agg, inv, sc)
        return agg.T @ agg
    # Anderson-Rubin at 0: y on [Z1, X2], the excluded instruments' Wald
    g = np.linalg.solve(wc(Z, Z), wc(Z, y))
    u = y - Z @ g
    B = np.linalg.inv(wc(Z, Z))
    Vg = B @ meat(Z * (wv * u)[:, None]) @ B
    L1 = Z1.shape[1]
    ar = g[:L1] @ np.linalg.solve(Vg[:L1, :L1], g[:L1])
    # endogeneity: the instruments plus Y, at the OLS residuals
    bo = np.linalg.solve(wc(X, X), wc(X, y))
    eo = y - X @ bo
    Zf = np.column_stack([Z, Y])
    Sf = meat(Zf * (wv * eo)[:, None])
    Jf = hansen_j(wc(Zf, X), wc(Zf, y), Sf)[0]
    q = Z.shape[1]
    Jr = hansen_j(wc(Z, X), wc(Z, y), Sf[:q, :q])[0]
    return {"shea": shea, "ar": ar, "C": Jf - Jr}


@pytest.mark.parametrize("vcov", ["hetero", {"CRV1": "cl"}])
@pytest.mark.parametrize("weights", [None, "w"])
def test_newer_statistics(data, tmp_path, vcov, weights):
    model = {"fml": "y ~ x | g | e1 + e2 ~ z1 + z2 + z3", "y": "y", "x": ["e1", "e2", "x"],
             "iv": {"endog": ["e1", "e2"], "z": ["z1", "z2", "z3", "x"]}}
    est = StreamingHDFE("y", ["e1", "e2", "x", "z1", "z2", "z3"], ["g"], models=[model],
                        weights=weights, workdir=tmp_path, verbose=False)
    fit = est.fit(data.lazy(), vcov=vcov)
    fit = fit[0] if isinstance(fit, list) else fit
    d = fit.iv_diag
    ref = dense_extra(data, ["e1", "e2"], ["z1", "z2", "z3"],
                      "cluster" if isinstance(vcov, dict) else vcov, weights)
    np.testing.assert_allclose([d["partial_r2"][e]["shea"] for e in ("e1", "e2")],
                               ref["shea"], rtol=1e-8)
    assert d["anderson_rubin"]["statistic"] == pytest.approx(ref["ar"], rel=1e-7)
    assert d["anderson_rubin"]["df"] == 3 and d["anderson_rubin"]["confidence_set"] is None
    assert d["endogeneity"]["statistic"] == pytest.approx(ref["C"], rel=1e-6)
    assert d["endogeneity"]["df"] == 2
    assert d["effective_f"] is None                 # two endogenous regressors


@pytest.mark.parametrize("vcov", ["iid", "hetero"])
def test_one_endogenous_extras(data, tmp_path, vcov):
    """One endogenous regressor: the effective F (the first-stage F under
    iid errors), Shea's R^2 equal to the partial R^2, and the
    Anderson-Rubin set, whose ends the test just rejects at. (y less its
    e2 term, which would make the instruments invalid: the set would then be
    empty.)"""
    data = data.with_columns(y=pl.col("y") + 0.5 * pl.col("e2"))
    fit = feols_stream("y ~ x | g | e1 ~ z1 + z2 + z3", data, vcov=vcov, workdir=tmp_path,
                       verbose=False)
    d = fit.iv_diag
    pr = d["partial_r2"]["e1"]
    assert pr["shea"] == pytest.approx(pr["partial"], rel=1e-10)
    ef = d["effective_f"]
    if vcov == "iid":
        assert ef["F"] == pytest.approx(d["kp_wald"]["F"], rel=1e-8)
        assert ef["k_eff"] == pytest.approx(3.0)
    assert ef["critical_value"] > 0
    ar = d["anderson_rubin"]
    cs = ar["confidence_set"]
    (lo, hi), = cs["intervals"]                     # strong instruments: one interval
    b = fit.beta[fit.coefnames.index("e1")]
    assert lo < b < hi
    # at the ends, the test of beta0 = end is at its critical value
    crit = stats.chi2.ppf(0.95, 3)
    for end in (lo, hi):
        r = feols_stream("y ~ x | g | e1 ~ z1 + z2 + z3",
                         data.with_columns(y=pl.col("y") - end * pl.col("e1")), vcov=vcov,
                         workdir=tmp_path, verbose=False)
        assert r.iv_diag["anderson_rubin"]["statistic"] == pytest.approx(crit, rel=1e-6)

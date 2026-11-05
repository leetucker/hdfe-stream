"""hdfe_stream.core.balance (first written in match-stream) against numpy.
Its MatchIt conventions are checked end to end in match-stream's
tests/test_balance.py (MatchIt 4.8.2 to 1e-8)."""

import numpy as np
import polars as pl
import pytest

from hdfe_stream.core.balance import weighted_balance


@pytest.fixture(scope="module")
def df():
    rng = np.random.default_rng(5)
    n = 1500
    g = rng.random(n) < 0.4
    return pl.DataFrame({"g": g, "x": rng.standard_normal(n) + g, "b": rng.random(n) < 0.3,
                         "w": np.where(rng.random(n) < 0.1, 0.0, rng.exponential(1, n))})


def _wmv(x, w):
    w = w / w.sum()
    m = np.sum(w * x)
    return m, np.sum(w * (x - m) ** 2) / (1 - np.sum(w**2))


@pytest.mark.parametrize("scale", ["group1", "group0", "pooled"])
def test_against_numpy(df, scale):
    tab = weighted_balance(df, "g", ["x", pl.col("b").cast(pl.Float64).alias("b")], "w",
                           scale=scale)
    g, x, b, w = (df[c].to_numpy() for c in ("g", "x", "b", "w"))
    m1, v1 = _wmv(x[g], w[g])
    m0, v0 = _wmv(x[~g], w[~g])
    sd = {"group1": np.sqrt(v1), "group0": np.sqrt(v0), "pooled": np.sqrt((v1 + v0) / 2)}[scale]
    rx = tab.row(0, named=True)
    np.testing.assert_allclose([rx["mean1"], rx["mean0"], rx["var1"], rx["var0"]],
                               [m1, m0, v1, v0], rtol=1e-12)
    np.testing.assert_allclose(rx["smd"], (m1 - m0) / sd, rtol=1e-12)
    np.testing.assert_allclose(rx["var_ratio"], v1 / v0, rtol=1e-12)
    # eCDF gaps at every distinct value, zero-weight rows included
    vals = np.unique(x)
    F1 = np.array([w[g & (x <= v)].sum() for v in vals]) / w[g].sum()
    F0 = np.array([w[~g & (x <= v)].sum() for v in vals]) / w[~g].sum()
    np.testing.assert_allclose([rx["ecdf_mean"], rx["ecdf_max"]],
                               [np.mean(np.abs(F1 - F0)), np.max(np.abs(F1 - F0))], rtol=1e-10)
    rb = tab.row(1, named=True)                    # a 0/1 term
    mb1, mb0 = np.average(b[g], weights=w[g]), np.average(b[~g], weights=w[~g])
    assert rb["var_ratio"] is None
    np.testing.assert_allclose([rb["var1"], rb["ecdf_max"]], [mb1 * (1 - mb1), abs(mb1 - mb0)],
                               rtol=1e-12)


def test_unweighted_and_given_scale(df):
    tab = weighted_balance(df.lazy(), "g", ["x"], scale={"x": 2.0}, ecdf=False)
    g, x = df["g"].to_numpy(), df["x"].to_numpy()
    np.testing.assert_allclose(tab["smd"][0], (x[g].mean() - x[~g].mean()) / 2, rtol=1e-12)
    np.testing.assert_allclose(tab["var1"][0], x[g].var(ddof=1), rtol=1e-12)
    assert np.isnan(tab["ecdf_max"][0])
    with pytest.raises(ValueError, match="scale"):
        weighted_balance(df, "g", ["x"], scale="treated")


def test_effective_sizes(df):
    from hdfe_stream.core.balance import effective_sizes
    g, w = df["g"].to_numpy(), df["w"].to_numpy()
    s = effective_sizes(df, "g", "w")
    assert (s.n1, s.n0) == (int(g.sum()), int((~g).sum()))
    np.testing.assert_allclose([s.ess1, s.ess0],
                               [w[g].sum() ** 2 / (w[g] ** 2).sum(),
                                w[~g].sum() ** 2 / (w[~g] ** 2).sum()], rtol=1e-12)
    # no weights: the rows; an expression for the group; a group with no weight
    u = effective_sizes(df.lazy(), ~pl.col("g"))
    assert (u.n1, u.n0, u.ess1, u.ess0) == (s.n0, s.n1, s.n0, s.n1)
    z = effective_sizes(df.with_columns(pl.lit(0.0).alias("w")), "g", "w")
    assert z.ess1 == z.ess0 == 0.0 and z.n1 == s.n1


def test_zero_variance_balance_ratios_and_standardization():
    """Constant controls are valid for balance, even when the ratio is infinite."""
    d = pl.DataFrame({"d": [False, False, True, True],
                      "mixed": [2., 2., 1., 3.],
                      "same": [2., 2., 2., 2.],
                      "different": [2., 2., 3., 3.]})
    rows = {r["term"]: r for r in weighted_balance(d, "d", ["mixed", "same", "different"], ecdf=False).to_dicts()}
    assert np.isinf(rows["mixed"]["var_ratio"])
    assert rows["mixed"]["smd"] == 0
    assert rows["same"]["var_ratio"] is None
    assert rows["same"]["smd"] == 0
    assert np.isposinf(rows["different"]["smd"])
    assert rows["different"]["var_ratio"] is None

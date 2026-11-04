"""hdfe_stream.core.ipw and .mestimation (first written in match-stream):
IPW weights and the Hajek IPW estimate whose influence-function variance
accounts for the estimated score, against a brute-force stacked
M-estimation sandwich (score equations and the two weighted-mean moments,
Jacobian by central differences) that shares no code with them."""


import numpy as np
import polars as pl
import pytest
from scipy import special

from hdfe_stream import feglm_stream
from hdfe_stream.core.influence import weighted_mean_difference
from hdfe_stream.core.ipw import ipw_mean_difference, ipw_weights
from hdfe_stream.core.mestimation import BinaryFirstStage


@pytest.fixture(scope="module")
def df():
    rng = np.random.default_rng(11)
    n = 2500
    x1, x2 = rng.standard_normal(n), rng.standard_normal(n)
    d = rng.random(n) < special.expit(-0.4 + 0.8 * x1 - 0.5 * x2)
    y = 1 + 0.7 * d + x1 + 0.5 * x2**2 + rng.standard_normal(n)
    return pl.DataFrame({"d": d.astype(np.int64), "x1": x1, "x2": x2, "y": y,
                         "cl": rng.integers(0, 80, n)})


@pytest.fixture(scope="module")
def fits(df, tmp_path_factory):
    wd = tmp_path_factory.mktemp("fs")
    return {link: feglm_stream("d ~ x1 + x2", df, family=link, workdir=wd)
            for link in ("logit", "probit")}


def _F(link, eta):
    return special.expit(eta) if link == "logit" else special.ndtr(eta)


def stacked(df, gamma, link, estimand, trim=None, cluster=None):
    """HC0 (or clustered, no small-sample factor) sandwich variance of
    mu1 - mu0 from the stacked moment conditions."""
    X = np.column_stack([np.ones(df.height), df["x1"].to_numpy(), df["x2"].to_numpy()])
    D = df["d"].to_numpy().astype(float)
    Y = df["y"].to_numpy()
    k = X.shape[1]

    def moments(theta):
        g, m1, m0 = theta[:k], theta[k], theta[k + 1]
        eta = X @ g
        F = _F(link, eta)
        f = F * (1 - F) if link == "logit" else np.exp(-eta**2 / 2) / np.sqrt(2 * np.pi)
        s = ((D - F) * f / (F * (1 - F)))[:, None] * X
        p = F
        if estimand == "att":
            w = np.where(D == 1, 1.0, p / (1 - p))
        elif estimand == "atc":
            w = np.where(D == 1, (1 - p) / p, 1.0)
        else:
            w = np.where(D == 1, 1 / p, 1 / (1 - p))
        if trim is not None:
            w = np.where((p >= trim[0]) & (p <= trim[1]), w, 0.0)
        e1 = D * w * (Y - m1)
        e0 = (1 - D) * w * (Y - m0)
        return np.column_stack([s, e1, e0])

    # the estimates solve sum m = 0: gamma (the fit) and the Hajek means
    e = moments(np.r_[gamma, 0.0, 0.0])
    # mu_g = sum w y / sum w over group g: from the moments at mu = 0 and mu = 1
    e_1 = moments(np.r_[gamma, 1.0, 1.0])
    W1 = (e[:, k] - e_1[:, k]).sum()
    W0 = (e[:, k + 1] - e_1[:, k + 1]).sum()
    th = np.r_[gamma, e[:, k].sum() / W1, e[:, k + 1].sum() / W0]
    M = moments(th)
    h = 1e-6
    A = np.zeros((k + 2, k + 2))
    for j in range(k + 2):
        dp, dm = th.copy(), th.copy()
        dp[j] += h * max(1.0, abs(th[j]))
        dm[j] -= h * max(1.0, abs(th[j]))
        A[:, j] = (moments(dp).sum(0) - moments(dm).sum(0)) / (dp[j] - dm[j])
    if cluster is None:
        B = M.T @ M
    else:
        G = pl.DataFrame(M).with_columns(pl.Series("c", cluster)).group_by("c").sum()
        S = G.drop("c").to_numpy()
        B = S.T @ S
    Ai = np.linalg.inv(A)
    V = Ai @ B @ Ai.T
    c = np.zeros(k + 2)
    c[k], c[k + 1] = 1, -1
    return th[k] - th[k + 1], float(c @ V @ c)


@pytest.mark.parametrize("link", ["logit", "probit"])
def test_first_stage_pieces(df, fits, link):
    first = BinaryFirstStage.from_fit(fits[link], "d")
    assert first.names == ["Intercept", "x1", "x2"]
    # about 0 at the fit (IRLS stops a little short of the exact MLE)
    np.testing.assert_allclose(first.score_sums(df), 0, atol=1e-3)
    # the observed information, directly
    X = np.column_stack([np.ones(df.height), df["x1"].to_numpy(), df["x2"].to_numpy()])
    D = df["d"].to_numpy()
    eta = X @ first.beta
    F = _F(link, eta)
    if link == "logit":
        a = F * (1 - F)
    else:
        f = np.exp(-eta**2 / 2) / np.sqrt(2 * np.pi)
        v = F * (1 - F)
        g = f / v
        gp = (-eta * f * v - f * f * (1 - 2 * F)) / v**2
        a = f * g - (D - F) * gp
    np.testing.assert_allclose(first.information(df), (X * a[:, None]).T @ X, rtol=1e-10)


@pytest.mark.parametrize("link", ["logit", "probit"])
@pytest.mark.parametrize("estimand", ["att", "atc", "ate"])
def test_corrected_variance_is_the_stacked_sandwich(df, fits, link, estimand):
    first = BinaryFirstStage.from_fit(fits[link], "d")
    r = ipw_mean_difference(df, "y", "d", first, estimand)
    est, v = stacked(df, first.beta, link, estimand)
    np.testing.assert_allclose(r.estimate, est, rtol=1e-10)
    n = df.height
    np.testing.assert_allclose(r.variance.vcov[0, 0], n / (n - 1) * v, rtol=1e-6)


def test_trimmed_and_clustered(df, fits):
    first = BinaryFirstStage.from_fit(fits["logit"], "d")
    r = ipw_mean_difference(df, "y", "d", first, "ate", trim=0.1, cluster="cl")
    est, v = stacked(df, first.beta, "logit", "ate", trim=(0.1, 0.9),
                     cluster=df["cl"].to_numpy())
    G = df["cl"].n_unique()
    np.testing.assert_allclose(r.estimate, est, rtol=1e-10)
    np.testing.assert_allclose(r.variance.vcov[0, 0], G / (G - 1) * v, rtol=1e-6)
    assert r.n_obs < df.height


def test_uncorrected_holds_weights_fixed(df, fits):
    first = BinaryFirstStage.from_fit(fits["logit"], "d")
    r = ipw_mean_difference(df, "y", "d", first, "att", correct=False)
    w = ipw_weights(df, "d", first, "att")
    ref = weighted_mean_difference(w, "y", "weight", "d", df_k=None)
    np.testing.assert_allclose(r.estimate, ref.estimate, rtol=1e-12)
    np.testing.assert_allclose(r.variance.vcov, ref.variance.vcov, rtol=1e-10)
    # for the ATT the score's estimation lowers the variance here
    rc = ipw_mean_difference(df, "y", "d", first, "att")
    assert rc.variance.vcov[0, 0] != pytest.approx(r.variance.vcov[0, 0], rel=1e-3)


def test_weights(df, fits):
    first = BinaryFirstStage.from_fit(fits["logit"], "d")
    w = ipw_weights(df, "d", first, "ate").collect()
    sums = w.group_by("d").agg(pl.col("weight").sum(), pl.len())
    np.testing.assert_allclose(sums["weight"].to_numpy(), sums["len"].to_numpy())
    p = first.prob()
    raw = ipw_weights(df.with_columns(p.alias("p")), "d", "p", "att", normalize=False).collect()
    pp = raw["p"].to_numpy()
    want = np.where(raw["d"].to_numpy() == 1, 1.0, pp / (1 - pp))
    np.testing.assert_allclose(raw["weight"].to_numpy(), want, rtol=1e-12)
    t = ipw_weights(df.with_columns(p.alias("p")), "d", "p", "att", trim=(0.2, 0.8)).collect()
    assert t["p"].min() >= 0.2 and t["p"].max() <= 0.8


@pytest.mark.parametrize("link,rtol", [("logit", 1e-9), ("probit", 1e-8)])
@pytest.mark.parametrize("estimand", ["att", "ate", "atc"])
def test_absorbed_fixed_effects_equal_dummies(df, tmp_path, link, rtol, estimand):
    # the same score model, g absorbed (correction by a fixed-effect
    # regression weighted by the observed information) or as dummies (the
    # dense observed information): the same numbers
    kw = {"family": link, "workdir": tmp_path, "iwls_tol": 1e-12, "iwls_maxiter": 100}
    d = df.with_columns((pl.col("cl") % 8).alias("g"))
    a = feglm_stream("d ~ x1 + x2 | g", d, **kw)
    b = feglm_stream("d ~ x1 + x2 + C(g)", d, **kw)
    ra = ipw_mean_difference(d, "y", "d", BinaryFirstStage.from_fit(a, "d"), estimand,
                             workdir=tmp_path)
    rb = ipw_mean_difference(d, "y", "d", BinaryFirstStage.from_fit(b, "d"), estimand)
    np.testing.assert_allclose(ra.estimate, rb.estimate, rtol=1e-10)
    np.testing.assert_allclose(ra.variance.vcov, rb.variance.vcov, rtol=rtol)


def test_fixed_effects_with_clusters(df, tmp_path):
    # the corrected contributions cluster like any other
    d = df.with_columns((pl.col("cl") % 8).alias("g"))
    a = feglm_stream("d ~ x1 + x2 | g", d, family="logit", workdir=tmp_path)
    b = feglm_stream("d ~ x1 + x2 + C(g)", d, family="logit", workdir=tmp_path)
    ra = ipw_mean_difference(d, "y", "d", BinaryFirstStage.from_fit(a, "d"), "att",
                             cluster="cl", workdir=tmp_path)
    rb = ipw_mean_difference(d, "y", "d", BinaryFirstStage.from_fit(b, "d"), "att",
                             cluster="cl")
    np.testing.assert_allclose(ra.variance.vcov, rb.variance.vcov, rtol=1e-8)



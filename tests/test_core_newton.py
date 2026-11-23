"""core.newton: its stable names, pass 0, and the Newton pass against direct
NumPy sums and numerical derivatives, for the logit, probit and Poisson."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from scipy import special

import hdfe_stream.core.newton as newton
from hdfe_stream.core.newton import glm_design

STABLE = ["glm_design", "GLMDesign", "NewtonMoments"]

# the dropped group in `_data` (test_pass0 checks the warning itself)
pytestmark = pytest.mark.filterwarnings("ignore:.*levels of g dropped:UserWarning")
FAMILIES = ["logit", "probit", "poisson"]


def test_stable_names():
    assert sorted(newton.__all__) == sorted(STABLE)
    for name in STABLE:
        assert hasattr(newton, name)


def _data(family, n=3000, seed=11):
    r = np.random.default_rng(seed)
    df = pl.DataFrame({"x1": r.normal(size=n), "x2": r.normal(size=n),
                       "x3": r.uniform(0, 3, n), "g": r.integers(0, 30, n),
                       "c": r.integers(0, 6, n), "w": r.uniform(.5, 2, n),
                       "fold": r.integers(0, 4, n), "lexp": r.normal(0, .3, n)})
    eta = (0.7 * df["x1"].to_numpy() - 0.4 * df["x2"].to_numpy()
           + (df["g"].to_numpy() % 5 - 2) * 0.3 - 0.2)
    if family == "poisson":
        y = r.poisson(np.exp(eta + df["lexp"].to_numpy()))
    elif family == "logit":
        y = (r.random(n) < special.expit(eta)).astype(int)
    else:
        y = (r.random(n) < special.ndtr(eta)).astype(int)
    df = df.with_columns(pl.Series("y", y))
    # a group whose outcome never varies (logit, probit) or is always zero
    # (Poisson): dropped by pass 0
    return df.with_columns(pl.when(pl.col("g") == 29).then(0).otherwise(pl.col("y"))
                           .alias("y"))


def _terms(family, y, eta):
    """(loglik, score, expected information) per row, by NumPy and SciPy."""
    if family == "logit":
        p = special.expit(eta)
        return y * special.log_expit(eta) + (1 - y) * special.log_expit(-eta), y - p, p * (1 - p)
    if family == "probit":
        P, Q = special.ndtr(eta), special.ndtr(-eta)
        phi = np.exp(-0.5 * eta ** 2) / np.sqrt(2 * np.pi)
        ll = y * special.log_ndtr(eta) + (1 - y) * special.log_ndtr(-eta)
        return ll, np.where(y > 0, phi / P, -phi / Q), phi ** 2 / (P * Q)
    mu = np.exp(eta)
    return y * eta - mu - special.gammaln(y + 1), y - mu, mu


def _reference(ld, B, A, models, cols):
    """The pass's sums by NumPy over the stored rows."""
    M, S, G, k = len(B), ld.n_splits, ld.n_groups, len(ld.names)
    ll = np.zeros((M, S))
    gb = np.zeros((M, k))
    ga = np.zeros((M, G))
    H = np.zeros((M, len(cols), len(cols)))
    D = np.zeros((M, G))
    C = np.zeros((M, G, len(cols)))
    for X, y, w, off, f, s in ld.rows_with_offset():
        wt = np.ones(len(y)) if w is None else w
        o = 0.0 if off is None else off
        for m in range(M):
            eta = X @ B[m] + A[m, f] + o
            lt, r, h = _terms(ld.family, y, eta)
            ll[m] += np.bincount(s, wt * lt, S)
            inc = models[m, s]
            r, h = np.where(inc, wt * r, 0.0), np.where(inc, wt * h, 0.0)
            gb[m] += X.T @ r
            ga[m] += np.bincount(f, r, G)
            Xc = X[:, cols]
            H[m] += Xc.T @ (Xc * h[:, None])
            D[m] += np.bincount(f, h, G)
            np.add.at(C[m], f, Xc * h[:, None])
    return ll, gb, ga, H, D, C


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("fe", [False, True])
def test_pass_matches_numpy(family, fe, tmp_path, monkeypatch):
    """Several models with their own splits, weights, an offset and a column
    subset: the kernel's sums, with the nonzeros-only products and without,
    and with the groups' sums per run and per part, are NumPy's."""
    df = _data(family)
    fml = "y ~ x1 + x2 + x3 + C(c)" + (" | g" if fe else "")
    ld = glm_design(fml, df, family=family, weights="w", split_by="fold", n_splits=4,
                    offset="lexp" if family == "poisson" else None, workdir=tmp_path)
    k, G, S = len(ld.names), ld.n_groups, ld.n_splits
    assert G == (29 if fe else 1)
    assert ld.diagnostics["fe_sorted"] == fe        # the per-run accumulation is tested
    r = np.random.default_rng(3)
    M = 3
    B = r.normal(0, 0.2, (M, k))
    A = r.normal(0, 0.2, (M, G))
    models = np.array([[True] * S, [False, True, True, True], [True, False, True, False]])
    cols = np.array([4, 0, 2, 6])
    ref = _reference(ld, B, A, models, cols)
    for nnz_share, sorted_ in [(0.0, True), (1.1, True), (1.1, False)]:
        monkeypatch.setattr(newton, "NNZ_SHARE", nnz_share)
        ld.fe_sorted = sorted_ and ld.diagnostics["fe_sorted"]
        mom = ld.newton_pass(B, A, models=models, columns=cols)
        for got, want in zip([mom.loglik, mom.gradient, mom.effects_gradient, mom.information,
                              mom.effects_information, mom.cross], ref):
            np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-9)
        lite = ld.newton_pass(B, A, models=models, information=False)
        np.testing.assert_allclose(lite.loglik, ref[0], rtol=1e-12)
        assert lite.information is None
    ld.cleanup()


@pytest.mark.parametrize("family", FAMILIES)
def test_derivatives(family, tmp_path):
    """The gradient is the log-likelihood's derivative, and (logit, Poisson)
    the information is minus the gradient's: by central differences."""
    df = _data(family, n=2000)
    ld = glm_design("y ~ x1 + x2 + x3 | g", df, family=family, workdir=tmp_path,
                    storage="memory")
    k, G = len(ld.names), ld.n_groups
    b = np.array([0.3, -0.2, 0.1])
    a = np.full(G, -0.1)
    mom = ld.newton_pass(b, a)
    eps = 1e-6
    for j in range(k):
        d = np.zeros(k)
        d[j] = eps
        up = ld.newton_pass(b + d, a, information=False)
        dn = ld.newton_pass(b - d, a, information=False)
        np.testing.assert_allclose((up.loglik.sum() - dn.loglik.sum()) / (2 * eps),
                                   mom.gradient[0, j], rtol=1e-6, atol=1e-6)
        if family != "probit":
            np.testing.assert_allclose(-(up.gradient - dn.gradient)[0] / (2 * eps),
                                       mom.information[0, j], rtol=1e-5, atol=1e-5)
            np.testing.assert_allclose(-(up.effects_gradient - dn.effects_gradient)[0]
                                       / (2 * eps), mom.cross[0, :, j], rtol=1e-5, atol=1e-6)
    ld.cleanup()


def test_skipped_groups_and_profiling(tmp_path):
    """A group held out by an infinite effect contributes nothing and is
    counted; the profiled information is the Schur complement."""
    df = _data("logit")
    ld = glm_design("y ~ x1 + x2 | g", df, family="logit", workdir=tmp_path)
    G = ld.n_groups
    a = np.zeros(G)
    a[3] = -np.inf
    mom = ld.newton_pass(np.array([0.2, 0.1]), a)
    n3 = df.filter(pl.col("g") == 3).height
    assert mom.skipped.sum() == n3
    assert mom.effects_information[0, 3] == 0 and mom.effects_gradient[0, 3] == 0
    Cp, gp = mom.profiled(0)
    keep = mom.effects_information[0] > 0
    Aab = mom.cross[0][keep]
    Di = 1 / mom.effects_information[0][keep]
    np.testing.assert_allclose(Cp, mom.information[0] - (Aab * Di[:, None]).T @ Aab,
                               rtol=1e-12)
    np.testing.assert_allclose(gp, mom.gradient[0] - (Aab * Di[:, None]).T
                               @ mom.effects_gradient[0][keep], rtol=1e-10, atol=1e-12)
    ld.cleanup()


@pytest.mark.parametrize("family", FAMILIES)
def test_pass0(family, tmp_path):
    """The sample, the groups dropped, the weights' scaling and the sums."""
    df = _data(family).with_columns(
        pl.when(pl.int_range(pl.len()) % 50 == 7).then(None).otherwise(pl.col("x2"))
        .alias("x2"))
    with pytest.warns(UserWarning, match="levels of g dropped"):
        ld = glm_design("y ~ x1 + x2 | g", df, family=family, weights="w",
                        split_by="fold", n_splits=4, workdir=tmp_path)
    kept = df.filter(pl.col("x2").is_not_null())
    assert ld.diagnostics["n_raw"] == df.height
    assert ld.separation == {"checked": True, "levels": 1,
                             "observations": kept.filter(pl.col("g") == 29).height}
    used = kept.filter(pl.col("g") != 29)
    assert ld.n_obs == used.height
    # analytic weights scaled to a mean of 1 over the rows kept
    w = used["w"].to_numpy() * used.height / used["w"].sum()
    np.testing.assert_allclose(ld.weight_sum, w.sum(), rtol=1e-12)
    f = used["fold"].to_numpy()
    for j, nm in enumerate(["x1", "x2"]):
        x = used[nm].to_numpy()
        np.testing.assert_allclose(ld.x_sums[:, j], np.bincount(f, w * x, 4), rtol=1e-10)
        np.testing.assert_allclose(ld.x_squares[:, j], np.bincount(f, w * x * x, 4),
                                   rtol=1e-10)
    np.testing.assert_allclose(ld.group_outcomes.sum(), (w * used["y"].to_numpy()).sum(),
                               rtol=1e-10)
    if family == "poisson":
        assert ld.outcome_counts is None
    else:
        np.testing.assert_allclose(ld.outcome_counts.sum(axis=(0, 1)),
                                   [(w * (1 - used["y"].to_numpy())).sum(),
                                    (w * used["y"].to_numpy()).sum()], rtol=1e-10)
    assert ld.fe_levels["g"].to_list() == list(range(29))
    ld.cleanup()


def test_outcome_checks(tmp_path):
    df = _data("poisson")
    with pytest.raises(ValueError, match="0s and 1s"):
        glm_design("y ~ x1", df.with_columns(pl.col("y") * 3), family="logit",
                   workdir=tmp_path)
    with pytest.raises(ValueError, match="nonnegative"):
        glm_design("y ~ x1", df.with_columns(pl.col("y") - 5), family="poisson",
                   workdir=tmp_path)
    with pytest.raises(ValueError, match="at most one absorbed"):
        glm_design("y ~ x1 | g + c", df, family="poisson", workdir=tmp_path)

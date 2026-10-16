"""Post-estimation inference: `confint`, `wald_test`, `bonferroni`, `holm`.

All four work from the stored coefficients and vcov, so they are checked
against pyfixest's methods on the same models, fitted to a tight tolerance.
Where this package departs from pyfixest on purpose (an F test for any linear
hypothesis, F's denominator degrees of freedom without clustering), the check
is against the textbook formula instead.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from scipy import stats

from conftest import FE_DOF_PF, rel

pf = pytest.importorskip("pyfixest")

from hdfe_stream import bonferroni, feols_stream, fepois_stream, holm  # noqa: E402
from hdfe_stream.multcomp import _holm  # noqa: E402

FML = "log_earn ~ age_squared + age_cubed + x2 + z1 | worker_id + firm_id + year"
VCOVS = ["iid", {"CRV1": "firm_id"}]


def ours(fml, src, fn=feols_stream, **kwargs):
    kwargs = {"fe_dof": FE_DOF_PF, "verbose": False, "tol": 1e-12, **kwargs}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(fml, src, **kwargs)


def theirs(fml, frame, fn=None, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if fn == "fepois":
            return pf.fepois(fml, frame, iwls_tol=1e-12, demeaner=pf.LsmrDemeaner(
                fixef_atol=1e-12, fixef_btol=1e-12), **kwargs)
        return pf.feols(fml, frame, fixef_tol=1e-12, **kwargs)


@pytest.fixture(scope="module")
def pair(rich):
    """Our fit and pyfixest's, for each vcov."""
    out = {}
    for v in VCOVS:
        key = v if isinstance(v, str) else "crv"
        out[key] = (ours(FML, rich.src, vcov=v), theirs(FML, rich.pandas, vcov=v))
    yield out
    for res, _ in out.values():
        res.cleanup()


# -------------------------------------------------------------------- confint

@pytest.mark.parametrize("key", ["iid", "crv"])
def test_confint_pointwise(pair, key):
    res, ref = pair[key]
    got, want = res.confint(alpha=0.1), ref.confint(alpha=0.1)
    assert got.columns == ["Coefficient", "5.0%", "95.0%"]
    assert got["Coefficient"].to_list() == list(want.index)
    assert rel(got["5.0%"], want["5.0%"]) < 1e-7
    assert rel(got["95.0%"], want["95.0%"]) < 1e-7


def test_confint_joint_matches_pyfixest_draws(pair):
    """Same seed, same draws: the simultaneous critical value is pyfixest's."""
    res, ref = pair["crv"]
    got = res.confint(joint=True, seed=7, reps=5000)
    want = ref.confint(joint=True, seed=7, reps=5000)
    assert rel(got["2.5%"], want["2.5%"]) < 1e-7
    point = res.confint()
    assert ((got["97.5%"] - got["2.5%"]) > (point["97.5%"] - point["2.5%"])).all()


def test_confint_keep_drop(pair):
    res, _ = pair["iid"]
    assert res.confint(keep=["x2", "age"])["Coefficient"].to_list() == \
        ["x2", "age_squared", "age_cubed"]
    assert res.confint(keep="age", drop="cubed")["Coefficient"].to_list() == ["age_squared"]
    with pytest.raises(ValueError, match="no coefficient matches"):
        res.confint(keep="age", exact_match=True)
    assert res.confint(keep=["z1"], exact_match=True)["Coefficient"].to_list() == ["z1"]


# ------------------------------------------------------------------ wald_test

@pytest.mark.parametrize("key", ["iid", "crv"])
def test_wald_all_zero(pair, key):
    """R = I, q = 0: pyfixest's F statistic; the p-value from F(k, df_t)."""
    res, ref = pair[key]
    got = res.wald_test()
    want = ref.wald_test()
    assert got["distribution"] == "F" and got["df1"] == 4 and got["df2"] == res.df_t
    assert rel(got["statistic"], want["statistic"]) < 1e-7
    assert got["pvalue"] == pytest.approx(stats.f.sf(got["statistic"], 4, res.df_t),
                                          rel=1e-10, abs=1e-300)


def test_wald_general_hypothesis_chi2(pair):
    """Any R and q: W and its chi-squared p-value are pyfixest's."""
    res, ref = pair["crv"]
    R = np.array([[0, 0, 1, -1], [0, 0, 1, 0]], dtype=float)
    q = np.array([0.0, 0.01])
    got = res.wald_test(R, q, distribution="chi2")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        want = ref.wald_test(R, q, distribution="chi2")
    assert rel(got["statistic"], want["statistic"]) < 1e-7
    assert rel(got["pvalue"], want["pvalue"]) < 1e-6
    # F for the same hypothesis: W / 2 against F(2, df_t), not switched to chi2
    f = res.wald_test(R, q)
    assert f["distribution"] == "F"
    assert f["statistic"] == pytest.approx(got["statistic"] / 2)


@pytest.mark.parametrize("key", ["iid", "crv"])
def test_wald_one_coefficient_is_the_t_test(pair, key):
    res, _ = pair[key]
    tidy = res.tidy()
    for j, name in enumerate(res.coefnames):
        got = res.wald_test(keep=name, exact_match=True)
        assert got["statistic"] == pytest.approx(tidy["t value"][j] ** 2, rel=1e-12)
        assert got["pvalue"] == pytest.approx(tidy["Pr(>|t|)"][j], rel=1e-8, abs=1e-300)


def test_wald_keep_is_selected_rows_of_identity(pair):
    res, _ = pair["crv"]
    R = np.eye(4)[[0, 1]]
    assert res.wald_test(keep="age") == res.wald_test(R)
    assert res.wald_test(keep="age", q=[0.1, 0.0]) == res.wald_test(R, [0.1, 0.0])


def test_wald_follows_with_vcov(rich):
    res = ours(FML, rich.src, vcov="iid", cluster=["firm_id"])
    with res:
        assert res.with_vcov({"CRV1": "firm_id"}).wald_test()["df2"] == \
            min(res.n_clusters["firm_id"]) - 1
        assert res.wald_test()["statistic"] != \
            res.with_vcov({"CRV1": "firm_id"}).wald_test()["statistic"]


def test_wald_errors(pair):
    res, _ = pair["iid"]
    with pytest.raises(ValueError, match="4 columns"):
        res.wald_test(np.eye(3))
    with pytest.raises(ValueError, match="either R or keep"):
        res.wald_test(np.eye(4), keep="age")
    with pytest.raises(ValueError, match="q must be"):
        res.wald_test(np.eye(4)[:2], q=[0, 0, 0])
    with pytest.raises(ValueError, match="distribution"):
        res.wald_test(distribution="t")


def test_wald_glm_defaults_to_chi2(rich):
    fml = "y2 ~ age_squared + x2 | worker_id + firm_id"
    frame = rich.frame.with_columns((rich.frame["y2"] > rich.frame["y2"].median())
                                    .cast(float).alias("y2"))
    path = rich.path.with_name("postest_glm.parquet")
    frame.write_parquet(path)
    res = ours(fml, str(path), fepois_stream, iwls_tol=1e-12, vcov={"CRV1": "firm_id"})
    ref = theirs(fml, frame.to_pandas(), "fepois", vcov={"CRV1": "firm_id"})
    with res:
        got = res.wald_test()
        want = ref.wald_test(distribution="chi2")
        assert got["distribution"] == "chi2" and got["df2"] is None
        assert rel(got["statistic"], want["statistic"]) < 1e-6
        ci, ci_ref = res.confint(), ref.confint()
        assert rel(ci["2.5%"], ci_ref["2.5%"]) < 1e-6


# -------------------------------------------------------- bonferroni and holm

@pytest.fixture(scope="module")
def models(rich):
    fmls = ["log_earn ~ x2 | worker_id + firm_id",
            "log_earn ~ x2 + age_squared | worker_id + firm_id",
            "y2 ~ x2 + z1 | worker_id + firm_id + year"]
    fits = [ours(f, rich.src, vcov={"CRV1": "firm_id"}) for f in fmls]
    refs = [theirs(f, rich.pandas, vcov={"CRV1": "firm_id"}) for f in fmls]
    yield fits, refs
    for f in fits:
        f.cleanup()


def test_bonferroni_matches_pyfixest(models):
    fits, refs = models
    got = bonferroni(fits, "x2")
    want = pf.bonferroni(refs, "x2")
    assert got["fml"].to_list() == [f.fml for f in fits]
    assert rel(got["Bonferroni Pr(>|t|)"], want.loc["Bonferroni Pr(>|t|)"].to_numpy()) < 1e-6


def test_holm():
    p = np.array([0.01, 0.04, 0.03, 0.005])
    # sorted 0.005, 0.01, 0.03, 0.04 times 4, 3, 2, 1, then the running max
    assert np.allclose(_holm(p), [0.03, 0.06, 0.06, 0.02])
    assert np.allclose(_holm([0.5, 0.9]), [1.0, 1.0])
    rng = np.random.default_rng(0)
    for _ in range(20):
        p = rng.uniform(0, 0.2, 7) ** 2
        adj = _holm(p)
        assert np.all(adj >= p) and np.all(adj <= np.minimum(1, 7 * p) + 1e-15)
        order = np.argsort(p)
        assert np.all(np.diff(adj[order]) >= 0)


def test_holm_on_models(models, rich):
    fits, _ = models
    got = holm(fits, "x2")
    bonf = bonferroni(fits, "x2")
    assert np.allclose(got["Holm Pr(>|t|)"], _holm(got["Pr(>|t|)"].to_numpy()))
    assert (got["Holm Pr(>|t|)"] <= bonf["Bonferroni Pr(>|t|)"] + 1e-15).all()
    # several parameters: every (model, parameter) pair is one hypothesis
    both = holm(fits[2], ["z1", "x2"])
    assert both["Coefficient"].to_list() == ["z1", "x2"]
    assert np.allclose(bonferroni(fits[2], ["z1", "x2"])["Bonferroni Pr(>|t|)"],
                       np.minimum(1, 2 * both["Pr(>|t|)"].to_numpy()))
    with pytest.raises(ValueError, match="not among the coefficients"):
        holm(fits, "z1")
    multi = ours("log_earn ~ x2 + csw0(age_squared, age_cubed) | worker_id + firm_id", rich.src)
    with multi:
        assert holm(multi, "x2").height == 3

"""Fewer than two fixed effects: one (a within regression) and none (OLS).

With one dimension the streamed groups are demeaned and there is nothing left
for the solver; with none the rows are never grouped at all, and the model
keeps its intercept as a coefficient. Both are checked against pyfixest on the
`rich` panel, across weights, IV and every vcov flavor, with small batches so
that the multi-batch paths run.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from conftest import (FE_DOF_PF, TOL_BETA, TOL_RESID, TOL_SE, TOL_STAT,
                      ensure_interactions, pf_feols, rel, resid_sorted, scaled)

pytest.importorskip("pyfixest")

from hdfe_stream import StreamingHDFE, feols_stream  # noqa: E402

CASES = [
    # no fixed effects
    ("log_earn ~ age_squared + age_cubed + i(occ)", {}),
    ("log_earn ~ age_squared + age_cubed", {"weights": "wa"}),
    ("log_earn ~ age_squared + age_cubed",
     {"weights": "wf", "weights_type": "fweights"}),
    ("y_iv ~ age_squared | x2 ~ z1 + z2", {}),
    ("log_earn ~ age_squared + age_cubed - 1", {}),
    # one fixed effect
    ("log_earn ~ age_squared + age_cubed + i(occ) | worker_id", {}),
    ("log_earn ~ age_squared + age_cubed | firm_id^year", {}),
    ("log_earn ~ age_squared + age_cubed | worker_id", {"weights": "wa"}),
    ("log_earn ~ age_squared + age_cubed | worker_id",
     {"weights": "wf", "weights_type": "fweights"}),
    ("y_iv ~ age_squared | worker_id | x2 ~ z1 + z2", {}),
]
IDS = ["ols", "ols-aweights", "ols-fweights", "ols-iv", "ols-no-intercept",
       "within", "within-interacted", "within-aweights", "within-fweights",
       "within-iv"]
NO_FE, ONE_FE = list(range(5)), list(range(5, 10))
WITHIN_OLS = ONE_FE[:-1]

VCOVS = [("iid", "iid"), ("hetero", "hetero"),
         ("CRV1:worker_id", {"CRV1": "worker_id"}), ("CRV1:state", {"CRV1": "state"}),
         ("CRV1:worker_id+firm_id", {"CRV1": "worker_id+firm_id"})]


def reference_kwargs(options):
    return {"weights": options.get("weights"),
            "weights_type": options.get("weights_type", "aweights")}


@pytest.fixture(scope="module")
def fitted(rich, tmp_path_factory):
    """One streaming fit (every vcov flavor at once) and one pyfixest
    reference per case."""
    base = tmp_path_factory.mktemp("fewer_fe")
    ensure_interactions(rich, ["firm_id^year"])
    fits, refs = {}, {}
    for index, (fml, options) in enumerate(CASES):
        fits[index] = feols_stream(
            fml, rich.src, workdir=base / str(index), vcov="iid",
            cluster=["worker_id", "state", "worker_id+firm_id"], fe_dof=FE_DOF_PF,
            verbose=False, n_buckets=3, batch_rows=5_000, tol=1e-11,
            keep=["year"], outputs="keep", **options)
        refs[index] = pf_feols(fml, rich, vcov="iid", **reference_kwargs(options))
    yield fits, refs
    for r in fits.values():
        r.cleanup()


@pytest.mark.parametrize("index", range(len(CASES)), ids=IDS)
@pytest.mark.parametrize("key,spec", VCOVS, ids=[k for k, _ in VCOVS])
def test_coefficients_and_se_match_pyfixest(fitted, index, key, spec):
    fits, refs = fitted
    res, ref = fits[index].with_vcov(key), refs[index]
    ref.vcov(spec)
    assert res.coefnames == list(ref.coef().index)
    assert rel(res.beta, ref.coef().to_numpy()) < TOL_BETA
    assert rel(res.se, ref.se().to_numpy()) < TOL_SE


@pytest.mark.parametrize("index", range(len(CASES)), ids=IDS)
def test_fit_statistics_match_pyfixest(fitted, index):
    fits, refs = fitted
    res, ref = fits[index].with_vcov("iid"), refs[index]
    ref.vcov("iid")
    assert res.n_obs == ref._N
    for attr, ref_attr in [("r2", "_r2"), ("adj_r2", "_adj_r2"),
                           ("r2_within", "_r2_within"),
                           ("adj_r2_within", "_adj_r2_within"), ("rmse", "_rmse")]:
        mine, theirs = getattr(res, attr), getattr(ref, ref_attr)
        # no within R2 without fixed effects, and none of these under IV:
        # the two must agree on which are undefined, not just on the values
        assert np.isnan(mine) == np.isnan(theirs), attr
        if np.isfinite(theirs):
            assert rel(mine, theirs) < TOL_STAT, attr


@pytest.mark.parametrize("index", range(len(CASES)), ids=IDS)
def test_residuals_match_pyfixest(fitted, index):
    fits, refs = fitted
    theirs = np.sort(np.asarray(refs[index].resid()))
    assert scaled(resid_sorted(fits[index]), theirs) < TOL_RESID


@pytest.mark.parametrize("index", NO_FE, ids=[IDS[i] for i in NO_FE])
def test_ols_rows_add_up_without_fixed_effects(fitted, index, rich):
    """With no fixed effects the row file carries no fe_ columns, and the
    fitted part is all in xb (the intercept is one of the coefficients)."""
    fits, _ = fitted
    res = fits[index]
    out = res.resid().collect()
    assert not [c for c in out.columns if c.startswith("fe_")]
    y = out[res.depvar].to_numpy()
    assert scaled(out["xb"].to_numpy() + out["resid"].to_numpy(), y) < TOL_RESID
    assert res.fe_names == [] and res.n_levels == {} and res.k_fe == 0
    assert res.solver_info["solver"] == "none"
    with pytest.raises(KeyError, match="none"):
        res.fixef("worker_id")


@pytest.mark.parametrize("index", WITHIN_OLS, ids=[IDS[i] for i in WITHIN_OLS])
def test_within_fixed_effects_match_pyfixest(fitted, index, rich):
    """The one dimension's effects, joined back to the input rows: with xb and
    the residual they add up to the outcome, and with xb they are pyfixest's
    prediction. (pyfixest predicts for OLS only, hence no IV case.)"""
    fits, refs = fitted
    res, ref = fits[index], refs[index]
    (dim,) = res.fe_names
    keys = ["worker_id", "year"]
    mine = res.resid().select(*keys, f"fe_{dim}", "xb", "resid", res.depvar).collect()
    assert scaled(mine[f"fe_{dim}"] + mine["xb"] + mine["resid"], mine[res.depvar]) < TOL_RESID

    # pyfixest predicts for the rows it kept (singletons are dropped)
    used = rich.pandas.loc[ref._data.index, keys]
    theirs = pl.DataFrame({**{k: used[k].to_numpy() for k in keys},
                           "predict_ref": np.asarray(ref.predict())})
    joined = mine.join(theirs, on=keys)
    assert joined.height == len(used) == mine.height
    assert scaled(joined[f"fe_{dim}"] + joined["xb"], joined["predict_ref"]) < TOL_RESID
    assert res.solver_info["solver"] == "none"
    assert res.k_fe == res.n_levels[dim]


def test_csw0_expands_to_zero_one_and_two_fixed_effects(rich, workdir):
    """csw0() yields a model with no fixed effects, one, and two; each group of
    models is fitted with its own set of passes and all three must agree with
    pyfixest."""
    fml = "log_earn ~ age_squared | csw0(worker_id, firm_id)"
    multi = feols_stream(fml, rich.src, workdir=workdir, vcov="hetero",
                         fe_dof=FE_DOF_PF, verbose=False, batch_rows=5_000, tol=1e-12)
    with multi:
        refs = pf_feols(fml, rich, vcov="hetero")
        assert len(multi) == 3
        for res, ref in zip(multi, refs.all_fitted_models.values()):
            assert res.coefnames == list(ref.coef().index)
            assert rel(res.beta, ref.coef().to_numpy()) < TOL_BETA
            assert rel(res.se, ref.se().to_numpy()) < TOL_SE
        assert [len(r.fe_names) for r in multi] == [0, 1, 2]


def test_default_vcov_is_iid(rich, workdir):
    """pyfixest's default is iid, with or without fixed effects."""
    with feols_stream("log_earn ~ age_squared", rich.src, workdir=workdir,
                      verbose=False) as res:
        assert res.vcov_type == "iid"
    with feols_stream("log_earn ~ age_squared | worker_id", rich.src,
                      workdir=workdir, verbose=False) as res:
        assert res.vcov_type == "iid"


@pytest.mark.parametrize("fe", [[], None])
def test_low_level_interface_adds_an_intercept(rich, workdir, fe):
    est = StreamingHDFE("log_earn", ["age_squared"], fe, workdir=workdir, verbose=False)
    with est.fit(rich.src) as res:
        assert res.coefnames == ["Intercept", "age_squared"]
        ref = pf_feols("log_earn ~ age_squared", rich, vcov="iid")
        assert rel(res.beta, ref.coef().to_numpy()) < TOL_BETA


def test_one_fixed_effect_assembly_paths_agree(rich, workdir):
    fml = "log_earn ~ age_squared + age_cubed | worker_id"
    fits = [feols_stream(fml, rich.src, workdir=workdir, vcov="hetero", verbose=False,
                         assembly=a) for a in ("cells", "rows")]
    try:
        assert rel(fits[0].beta, fits[1].beta) < TOL_BETA
        assert rel(fits[0].se, fits[1].se) < TOL_SE
    finally:
        for r in fits:
            r.cleanup()


def test_cell_assembly_needs_a_fixed_effect(rich, workdir):
    with pytest.raises(ValueError, match="needs a fixed effect"):
        StreamingHDFE("log_earn", ["age_squared"], [], workdir=workdir, assembly="cells")


def test_summary_and_reporting_without_fixed_effects(fitted):
    fits, _ = fitted
    for index in (NO_FE[0], ONE_FE[0]):
        text = fits[index].summary_text()
        assert "components" not in text
        assert ("within R2" in text) == (index in ONE_FE)
    pf = pytest.importorskip("pyfixest")
    table = pf.etable([fits[NO_FE[0]].to_pyfixest(), fits[ONE_FE[0]].to_pyfixest()],
                      type="df")
    assert table is not None


@pytest.mark.parametrize("fml", ["log_earn ~ age_squared | worker_id",
                                 "log_earn ~ age_squared"])
def test_leave_out_needs_two_fixed_effects(rich, workdir, fml):
    with feols_stream(fml, rich.src, workdir=workdir, verbose=False) as res:
        with pytest.raises(ValueError, match="two fixed effects"):
            res.leave_out_kss()

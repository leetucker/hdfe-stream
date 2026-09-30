"""Poisson, logit and probit (`fepois_stream`, `feglm_stream`) against pyfixest.

Both sides iterate to a tight tolerance (IRLS and the fixed-effect solve), so
the comparison measures hdfe_stream's error rather than where either stopped.
The streaming fits use small batches and several buckets, so fe[0] groups
straddle chunks and the cell bookkeeping crosses chunk and file boundaries.

Two places where the reference is not pyfixest's own answer:

* logit and probit: pyfixest's separation check drops fixed-effect levels
  whose outcome is all 0 but keeps those whose outcome is all 1, whose effect
  then drifts toward infinity while their rows contribute nothing. hdfe_stream
  drops both, so pyfixest is given hdfe_stream's estimation sample.
* frequency weights: pyfixest's fepois counts N as the number of rows and
  scales the heteroskedastic meat so that it does not reproduce the fit of
  the data with each row repeated, and its feglm takes no weights at all.
  hdfe_stream's frequency weights are checked against pyfixest on the
  repeated rows instead, which is what frequency weights mean.
"""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl
import pytest

from conftest import FE_DOF_PF, TOL_BETA, TOL_SE, TOL_STAT, rel, scaled

pytest.importorskip("pyfixest")

from hdfe_stream import StreamingGLM, feglm_stream, fepois_stream  # noqa: E402
from hdfe_stream.simulate import simulate_rich  # noqa: E402

TIGHT = {"iwls_tol": 1e-12, "tol": 1e-12}
SMALL_BATCHES = {"batch_rows": 700, "n_buckets": 2}
VCOVS = ["iid", "hetero", {"CRV1": "worker_id"}, {"CRV1": "firm_id"},
         {"CRV1": "worker_id+firm_id"}]
VCOV_IDS = ["iid", "hetero", "crv-worker", "crv-firm", "crv-twoway"]


@pytest.fixture(scope="module")
def glm_panel(tmp_path_factory):
    """A small AKM panel with a count outcome `yp`, a binary outcome `yb`
    (both driven by the worker and firm effects and `x1`), a second count
    outcome `yq`, integer frequency weights `fw`, an offset `lexp` and a row
    id `rid`."""
    frame = simulate_rich(n_workers=400, n_firms=25, seed=4, keep_effects=True)
    rng = np.random.default_rng(1)
    x1 = frame["age_squared"].to_numpy()
    x1 = (x1 - x1.mean()) / x1.std()
    eta = (0.3 * x1 + 0.5 * frame["true_worker_effect"].to_numpy()
           + 0.5 * frame["true_firm_effect"].to_numpy())
    frame = frame.with_columns(
        x1=pl.Series(x1),
        yp=pl.Series(rng.poisson(np.exp(eta)).astype(float)),
        yq=pl.Series(rng.poisson(np.exp(0.5 * eta)).astype(float)),
        yb=pl.Series((rng.random(len(eta)) < 1 / (1 + np.exp(-eta))).astype(float)),
        fw=pl.col("wf").cast(pl.Int64),
        lexp=0.1 * pl.col("x2"),
    ).with_row_index("rid")
    path = tmp_path_factory.mktemp("glm") / "glm.parquet"
    frame.write_parquet(path)
    return str(path), frame.to_pandas()


def pf_glm(family, fml, data, **kwargs):
    """pyfixest reference, iterated and demeaned to a tight tolerance."""
    import pyfixest as pf
    kwargs = {"fixef_rm": "none", "iwls_tol": 1e-12,
              "demeaner": pf.LsmrDemeaner(fixef_atol=1e-12, fixef_btol=1e-12), **kwargs}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if family == "poisson":
            return pf.fepois(fml, data, **kwargs)
        return pf.feglm(fml, data, family=family, **kwargs)


def fit(family, fml, path, **kwargs):
    kwargs = {"fe_dof": FE_DOF_PF, "verbose": False, **TIGHT, **SMALL_BATCHES, **kwargs}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")     # separation and collinearity notes
        if family == "poisson":
            return fepois_stream(fml, path, **kwargs)
        return feglm_stream(fml, path, family, **kwargs)


def our_sample(res, pdf):
    """The rows of `pdf` that the streaming fit kept (fit with keep=['rid'])."""
    rid = res.resid().select("rid").collect()["rid"].to_numpy()
    return pdf[pdf["rid"].isin(rid)].reset_index(drop=True)


def repeated(pdf, col="fw"):
    return pdf.loc[pdf.index.repeat(pdf[col])].reset_index(drop=True)


def assert_agrees(res, ref):
    assert res.coefnames == list(ref.coef().index)
    assert rel(res.beta, ref.coef().to_numpy()) < TOL_BETA
    assert rel(res.se, ref.se().to_numpy()) < TOL_SE
    assert res.n_obs == ref._N
    assert res.df_t == ref._df_t
    assert rel(res.deviance, ref.deviance) < TOL_STAT


# -------------------------------------------------------------------- Poisson

POIS_FMLS = [
    "yp ~ x1 + x2 | worker_id + firm_id",
    "yp ~ x1 + x2 | worker_id + firm_id + year",
    "yp ~ x1 + x2 | firm_id^year + worker_id",
    "yp ~ x1 + x2 | worker_id",
    "yp ~ x1 + x2 + i(occ)",
]
POIS_IDS = ["two-way", "three-way", "interacted", "one-way", "no-fe"]


@pytest.mark.parametrize("vcov", VCOVS, ids=VCOV_IDS)
@pytest.mark.parametrize("fml", POIS_FMLS, ids=POIS_IDS)
def test_fepois_matches_pyfixest(glm_panel, fml, vcov):
    path, pdf = glm_panel
    with fit("poisson", fml, path, vcov=vcov) as res:
        ref = pf_glm("poisson", fml, pdf, vcov=vcov)
        assert_agrees(res, ref)
        assert res.family == "poisson" and res.diagnostics["irls"]["converged"]
        assert rel(res.loglik, ref._loglik) < TOL_STAT
        assert rel(res.pseudo_r2, ref._pseudo_r2) < 1e-8


@pytest.mark.parametrize("vcov", VCOVS[:3], ids=VCOV_IDS[:3])
def test_fepois_aweights_match_pyfixest(glm_panel, vcov):
    path, pdf = glm_panel
    fml = "yp ~ x1 + x2 | worker_id + firm_id"
    with fit("poisson", fml, path, vcov=vcov, weights="wa") as res:
        assert_agrees(res, pf_glm("poisson", fml, pdf, vcov=vcov, weights="wa"))


@pytest.mark.parametrize("vcov", VCOVS[:4], ids=VCOV_IDS[:4])
def test_fepois_fweights_match_repeated_rows(glm_panel, vcov):
    path, pdf = glm_panel
    fml = "yp ~ x1 + x2 | worker_id + firm_id"
    with fit("poisson", fml, path, vcov=vcov, weights="fw", weights_type="fweights") as res:
        assert_agrees(res, pf_glm("poisson", fml, repeated(pdf), vcov=vcov))


def test_offset_matches_pyfixest(glm_panel):
    path, pdf = glm_panel
    fml = "yp ~ x1 | worker_id + firm_id"
    with fit("poisson", fml, path, vcov="hetero", offset="lexp", keep=["lexp"]) as res:
        assert_agrees(res, pf_glm("poisson", fml, pdf, vcov="hetero", offset="lexp"))
        out = res.resid().collect()
        parts = out["xb"] + out["fe_worker_id"] + out["fe_firm_id"] + out["lexp"]
        assert scaled(parts.to_numpy(), out["eta"].to_numpy()) < 1e-10


def test_several_models(glm_panel):
    """csw0() and two outcomes: one IRLS per model, one estimator per outcome,
    results in the formula's order."""
    path, pdf = glm_panel
    fml = "yp + yq ~ csw0(x1, x2) | worker_id + firm_id"
    with fit("poisson", fml, path, vcov="hetero") as multi:
        refs = {(r._depvar, tuple(r._coefnames)): r
                for r in pf_glm("poisson", fml, pdf, vcov="hetero").to_list()}
        assert len(multi) == len(refs) == 6
        assert [res.depvar for res in multi] == ["yp"] * 3 + ["yq"] * 3
        for res in multi:
            r = refs[(res.depvar, tuple(res.coefnames))]
            if res.coefnames:
                assert_agrees(res, r)
            else:
                assert rel(res.deviance, r.deviance) < TOL_STAT


# --------------------------------------------------------------- logit/probit

BIN_FMLS = [
    "yb ~ x1 + x2 | worker_id + firm_id",
    "yb ~ x1 + x2 | firm_id",
    "yb ~ x1 + x2 + i(occ)",
]
BIN_IDS = ["two-way", "one-way", "no-fe"]


@pytest.mark.parametrize("vcov", VCOVS[:4], ids=VCOV_IDS[:4])
@pytest.mark.parametrize("fml", BIN_FMLS, ids=BIN_IDS)
@pytest.mark.parametrize("family", ["logit", "probit"])
def test_feglm_matches_pyfixest(glm_panel, family, fml, vcov):
    path, pdf = glm_panel
    with fit(family, fml, path, vcov=vcov, keep=["rid"]) as res:
        ref = pf_glm(family, fml, our_sample(res, pdf), vcov=vcov)
        assert_agrees(res, ref)
        assert res.family == family


@pytest.mark.parametrize("vcov", VCOVS[:4], ids=VCOV_IDS[:4])
@pytest.mark.parametrize("family", ["logit", "probit"])
def test_feglm_fweights_match_repeated_rows(glm_panel, family, vcov):
    path, pdf = glm_panel
    fml = "yb ~ x1 + x2 | worker_id + firm_id"
    with fit(family, fml, path, vcov=vcov, weights="fw", weights_type="fweights",
             keep=["rid"]) as res:
        ref = pf_glm(family, fml, repeated(our_sample(res, pdf)), vcov=vcov)
        assert_agrees(res, ref)


@pytest.mark.parametrize("family", ["logit", "probit"])
def test_feglm_aweights(glm_panel, family):
    """Analytic weights give the frequency-weighted coefficients, and are
    invariant to rescaling (pyfixest's feglm has no weights to compare with)."""
    path, _ = glm_panel
    fml = "yb ~ x1 + x2 | worker_id + firm_id"
    with fit(family, fml, path, weights="fw", weights_type="fweights") as f, \
            fit(family, fml, path, weights="fw") as a, \
            fit(family, fml, path, weights=pl.col("fw") * 3.0) as a3:
        assert rel(a.beta, f.beta) < TOL_BETA
        assert rel(a3.beta, a.beta) < TOL_BETA
        assert rel(a3.se, a.se) < TOL_SE
        assert a.n_obs == a3.n_obs < f.n_obs


def test_binary_separation_repeats(tmp_path):
    """Dropping firm S (all ones) leaves worker A with only zeros, so A goes
    in the second round; pyfixest, which drops only all-zero levels once,
    keeps both."""
    rng = np.random.default_rng(5)
    n = 3000
    base = pl.DataFrame({"worker": rng.integers(0, 300, n).astype(str),
                         "firm": rng.integers(0, 20, n).astype(str),
                         "x": rng.normal(size=n)})
    base = base.with_columns(y=(pl.Series(rng.random(n)) < 1 / (1 + (-pl.col("x")).exp()))
                             .cast(pl.Float64))
    special = pl.DataFrame({"worker": ["A", "B", "A", "C", "C"],
                            "firm": ["S", "S", "0", "0", "1"],
                            "x": [0.1, 0.2, 0.3, 0.4, 0.5], "y": [1.0, 1.0, 0.0, 1.0, 0.0]})
    path = tmp_path / "sep.parquet"
    frame = pl.concat([base, special])
    frame.write_parquet(path)
    with fit("logit", "y ~ x | worker + firm", str(path)) as res:
        sep = res.diagnostics["separation"]
        assert sep["rounds"] == 2
        kept = res.resid().select("worker", "firm").collect()
        assert kept.filter(pl.col("worker") == "A").height == 0
        assert kept.filter(pl.col("firm") == "S").height == 0
        assert kept.filter(pl.col("worker") == "C").height == 2
        assert res.n_obs + sep["observations"] == frame.height


# ------------------------------------------------------------------- options

@pytest.mark.parametrize("options", [
    {"solver": "stream_cg"}, {"solver": "within"}, {"cells_in_memory": True},
    {"keep_intermediates": True}, {"stream": "firm_id"}, {"precond": "amg"},
    {"batch_rows": 2_000_000, "n_buckets": 1},
], ids=["stream_cg", "within", "cells-in-memory", "keep-intermediates", "stream-firm",
        "amg", "one-chunk"])
def test_solver_and_layout_options(glm_panel, options):
    path, pdf = glm_panel
    fml = "yp ~ x1 + x2 | worker_id + firm_id + year"
    ref = pf_glm("poisson", fml, pdf, vcov={"CRV1": "worker_id"})
    with fit("poisson", fml, path, vcov={"CRV1": "worker_id"}, **options) as res:
        assert_agrees(res, ref)


def test_results_and_reporting(glm_panel):
    import pyfixest as pf
    path, pdf = glm_panel
    fml = "yp ~ x1 + x2 | worker_id + firm_id"
    with fit("poisson", fml, path, vcov="hetero") as res:
        ref = pf_glm("poisson", fml, pdf, vcov="hetero")
        # normal-distribution inference, as pyfixest has it for GLMs
        tidy = res.tidy()
        assert rel(tidy["Pr(>|t|)"].to_numpy(), ref.pvalue().to_numpy()) < 1e-6
        view = res.to_pyfixest()
        assert view._method == "fepois"
        assert rel(view.pvalue().to_numpy(), ref.pvalue().to_numpy()) < 1e-6
        assert rel(view.confint().to_numpy(), ref.confint().to_numpy()) < 1e-6
        pf.etable([view, ref])
        text = res.summary_text()
        assert "poisson" in text and "deviance" in text and "RSS" not in text
        import json
        d = json.loads(res.summary_json())
        assert d["family"] == "poisson" and d["deviance"] == res.deviance
        assert "rss" not in d and d["irls"]["iterations"] >= 1
        # the residual file: fitted = exp(eta), resid = y - fitted, and eta is
        # the sum of its parts
        out = res.resid().collect()
        assert {"eta", "fitted", "resid", "resid_working", "yp"} <= set(out.columns)
        assert scaled(out["fitted"].to_numpy(), np.exp(out["eta"].to_numpy())) < 1e-12
        assert scaled(out["resid"].to_numpy(), (out["yp"] - out["fitted"]).to_numpy()) < 1e-12
        eta = (out["xb"] + out["fe_worker_id"] + out["fe_firm_id"]).to_numpy()
        assert scaled(eta, out["eta"].to_numpy()) < 1e-10
        assert res.fixef("firm_id").collect().height == res.n_levels["firm_id"]
        with pytest.raises(ValueError, match="linear models"):
            res.leave_out_kss()


def test_low_level_estimator(glm_panel):
    path, pdf = glm_panel
    est = StreamingGLM(y="yb", x=["x1", ("x1sq", pl.col("x1") ** 2)], fe=["firm_id"],
                       family="probit", verbose=False, **TIGHT)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = est.fit(path, vcov="hetero", fe_dof=FE_DOF_PF)
    with res:
        pdf = pdf.assign(x1sq=pdf["x1"] ** 2)
        ref = pf_glm("probit", "yb ~ x1 + x1sq | firm_id", pdf, vcov="hetero")
        assert rel(res.beta, ref.coef().to_numpy()) < TOL_BETA
        assert rel(res.se, ref.se().to_numpy()) < TOL_SE


def test_not_converged_warns(glm_panel):
    path, _ = glm_panel
    with pytest.warns(UserWarning, match="did not converge"):
        res = fepois_stream("yp ~ x1 | worker_id + firm_id", path, iwls_maxiter=2,
                            verbose=False)
    with res:
        assert not res.diagnostics["irls"]["converged"]
        assert res.diagnostics["irls"]["iterations"] == 2


# ----------------------------------------------------------------- refusals

@pytest.mark.parametrize("call,message", [
    (lambda p: fepois_stream("yp ~ x1 | worker_id", p, vcov={"CRV3": "worker_id"}),
     "CRV3 .* not available for GLMs"),
    (lambda p: fepois_stream("yp ~ 1 | worker_id | x2 ~ z1", p), "IV .* not available"),
    (lambda p: fepois_stream("yp ~ x1 | worker_id[x2] + firm_id", p),
     "varying slopes are not available"),
    (lambda p: feglm_stream("yb ~ x1 | worker_id", p, "gaussian"), "feols_stream"),
    (lambda p: feglm_stream("yb ~ x1 | worker_id", p, "poisson"), "fepois_stream"),
    (lambda p: feglm_stream("yp ~ x1 | worker_id", p, "logit"), "0s and 1s"),
    (lambda p: fepois_stream("x1 ~ x2 | worker_id", p), "nonnegative"),
], ids=["crv3", "iv", "slopes", "gaussian", "poisson-in-feglm", "not-binary", "negative"])
def test_refusals(glm_panel, call, message):
    path, _ = glm_panel
    with pytest.raises(ValueError, match=message):
        call(path)

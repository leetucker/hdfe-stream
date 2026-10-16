"""`HDFEResult.predict`: predictions on new data and on the source.

On the estimation sample a prediction is the fitted value, so the first check
is internal: prediction = outcome - residual (for GLMs, the residual file's
``eta`` and ``fitted``). The second is pyfixest's `predict` on the same rows,
with its fixed effects recovered to a tight tolerance; pyfixest has no varying
slopes, so those are checked internally only. Out of sample, a row whose
fixed-effect level the fit never saw gets a null where pyfixest gives NaN.
"""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl
import pytest

from conftest import FE_DOF_PF, scaled

pf = pytest.importorskip("pyfixest")

from hdfe_stream import feglm_stream, feols_stream, fepois_stream  # noqa: E402
from hdfe_stream.simulate import simulate_rich  # noqa: E402

TOL_PRED = 1e-8


def fit(fml, source, fn=feols_stream, *args, **kwargs):
    kwargs = {"fe_dof": FE_DOF_PF, "verbose": False, "tol": 1e-12, **kwargs}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(fml, source, *args, **kwargs)


def pf_predict(fml, train, newdata, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = pf.feols(fml, train.to_pandas(), fixef_tol=1e-12, **kwargs)
        return model.predict(newdata=newdata.to_pandas(), atol=1e-14, btol=1e-14)


def in_sample_gap(res, outcome):
    """Largest |prediction - (outcome - residual)| over the estimation sample,
    relative to the outcome's scale."""
    pred = res.predict().with_row_index("row_id").select("row_id", "prediction")
    r = res.resid().select("row_id", outcome, "resid")
    j = r.join(pred, on="row_id").collect()
    return scaled(j["prediction"], j[outcome] - j["resid"])


@pytest.mark.parametrize("fml, pf_fml", [
    ("log_earn ~ age_squared + age_cubed | worker_id + firm_id + year", None),
    # pyfixest predicts neither with '^' fixed effects nor (on newdata) with
    # i(); it gets the interaction as one column, and C() for the categorical
    ("log_earn ~ age_squared + C(occ) | worker_id + firm_id^year",
     "log_earn ~ age_squared + C(occ) | worker_id + firm_year"),
    ("log_earn ~ age_squared + C(state) + x2", None),
])
def test_in_sample_matches_fitted_and_pyfixest(rich, fml, pf_fml):
    with fit(fml, rich.src) as res:
        assert in_sample_gap(res, "log_earn") < TOL_PRED
        ours = res.predict(rich.src).collect()["prediction"].to_numpy()
    frame = rich.frame.with_columns(firm_year=pl.concat_str("firm_id", "year", separator="_"))
    theirs = pf_predict(pf_fml or fml, frame, frame)
    ok = ~np.isnan(theirs)
    assert np.array_equal(ok, ~np.isnan(ours))
    assert scaled(ours[ok], theirs[ok]) < 1e-7


def test_i_syntax(rich):
    """pyfixest's predict refuses newdata with i() covariates; here they work
    like any other covariate."""
    fml = "log_earn ~ i(year, treat, ref=2009) | worker_id + firm_id"
    with fit(fml, rich.src) as res:
        assert in_sample_gap(res, "log_earn") < TOL_PRED


def test_out_of_sample(rich):
    """Fit on the early years, predict the later ones: workers and firms
    seen in training get predictions that match pyfixest's, the others null."""
    fml = "log_earn ~ age_squared + age_cubed | worker_id + firm_id"
    years = rich.frame["year"]
    cut = int(years.quantile(0.7))
    train = rich.frame.filter(pl.col("year") <= cut)
    test = rich.frame.filter(pl.col("year") > cut).with_columns(
        pl.col("worker_id") + pl.when(pl.int_range(pl.len()) % 10 == 0)
        .then(10**7).otherwise(0))                 # some workers the fit never saw
    with fit(fml, train) as res:
        ours = res.predict(test).collect()
    assert ours.columns == [*test.columns, "prediction"]
    assert ours.drop("prediction").equals(test)            # newdata's own rows and order
    theirs = pf_predict(fml, train, test)
    pred = ours["prediction"]
    assert np.array_equal(pred.is_null().to_numpy(), np.isnan(theirs))
    assert pred.null_count() > 0
    ok = ~np.isnan(theirs)
    assert scaled(pred.to_numpy()[ok], theirs[ok]) < 1e-7


def test_lazy_and_path_sources_agree(rich, tmp_path):
    fml = "log_earn ~ age_squared | worker_id + firm_id"
    with fit(fml, rich.src) as res:
        frame = res.predict(rich.frame).collect()
        lazy = res.predict(rich.frame.lazy()).collect()
        path = res.predict(rich.src).collect()
        default = res.predict().collect()
    for other in (lazy, path, default):
        assert other["prediction"].equals(frame["prediction"])


def test_varying_slopes(trends):
    fml = "y ~ x | worker_id[t] + firm_id"
    with fit(fml, trends.src) as res:
        assert in_sample_gap(res, "y") < TOL_PRED


def test_iv(rich):
    fml = "y_iv ~ age_squared | worker_id + firm_id | x2 ~ z1 + z2"
    with fit(fml, rich.src) as res:
        assert in_sample_gap(res, "y_iv") < TOL_PRED


@pytest.fixture(scope="module")
def glm_panel(tmp_path_factory):
    frame = simulate_rich(n_workers=400, n_firms=25, seed=4, keep_effects=True)
    rng = np.random.default_rng(1)
    x1 = frame["age_squared"].to_numpy()
    x1 = (x1 - x1.mean()) / x1.std()
    eta = (0.3 * x1 + 0.5 * frame["true_worker_effect"].to_numpy()
           + 0.5 * frame["true_firm_effect"].to_numpy())
    frame = frame.with_columns(
        x1=pl.Series(x1),
        yp=pl.Series(rng.poisson(np.exp(eta)).astype(float)),
        yb=pl.Series((rng.random(len(eta)) < 1 / (1 + np.exp(-eta))).astype(float)),
        lexp=0.1 * pl.col("x2"))
    path = tmp_path_factory.mktemp("predict_glm") / "glm.parquet"
    frame.write_parquet(path)
    return str(path)


@pytest.mark.parametrize("family", ["poisson", "logit", "probit"])
def test_glm_link_and_response(glm_panel, family):
    """The linear predictor (with the offset) is the residual file's eta, and
    the response is its fitted mean; rows dropped for separation are null."""
    if family == "poisson":
        res = fit("yp ~ x1 | worker_id + firm_id", glm_panel, fepois_stream,
                  offset="lexp", iwls_tol=1e-12)
    else:
        res = fit("yb ~ x1 | worker_id + firm_id", glm_panel, feglm_stream, family,
                  iwls_tol=1e-12)
    with res:
        link = res.predict().with_row_index("row_id")
        resp = res.predict(type="response", name="mu").with_row_index("row_id")
        j = (res.resid().select("row_id", "eta", "fitted")
             .join(link.select("row_id", "prediction"), on="row_id")
             .join(resp.select("row_id", "mu"), on="row_id").collect())
        assert scaled(j["prediction"], j["eta"]) < TOL_PRED
        assert scaled(j["mu"], j["fitted"]) < TOL_PRED
        dropped = res.sample().filter(pl.col("dropped_because") == "separation")
        out = res.predict(type="response").with_row_index("row_id")
        nulls = dropped.join(out, on="row_id").select(pl.col("prediction").is_null().all())
        assert nulls.collect().item()


def test_errors(rich):
    fml = "log_earn ~ age_squared | worker_id + firm_id"
    with fit(fml, rich.src) as res:
        with pytest.raises(ValueError, match="lacks columns"):
            res.predict(rich.frame.drop("age_squared"))
        with pytest.raises(ValueError, match="already has a column"):
            res.predict(rich.frame.with_columns(prediction=0.0))
        with pytest.raises(ValueError, match="type must be"):
            res.predict(type="mean")

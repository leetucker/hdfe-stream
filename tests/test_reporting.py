"""Results and reporting: the `HDFEResult` API, and the pyfixest adapter that
lets streaming results go into `pf.etable`, `pf.summary` and `pf.iplot`.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from conftest import FE_DOF_PF, TOL_SE, pf_feols, rel

pytest.importorskip("pyfixest")

from hdfe_stream import HDFEMulti, HDFEResult, etable, feols_stream  # noqa: E402

OLS_FML = "log_earn ~ age_squared + age_cubed | worker_id + firm_id + year"
IV_FML = "y_iv ~ age_squared | worker_id + firm_id | x2 ~ z1 + z2"
EVENT_FML = "log_earn ~ i(year, treat, ref=2009) + age_squared | worker_id + firm_id"


@pytest.fixture(scope="module")
def fits(rich, tmp_path_factory):
    base = tmp_path_factory.mktemp("report")
    out = {
        "ols": feols_stream(OLS_FML, rich.src, workdir=base / "ols",
                            vcov={"CRV1": "worker_id"}, cluster=["firm_id"],
                            fe_dof=FE_DOF_PF, verbose=False, outputs="keep"),
        "iv": feols_stream(IV_FML, rich.src, workdir=base / "iv",
                           vcov={"CRV1": "worker_id"}, fe_dof=FE_DOF_PF,
                           verbose=False, outputs="keep"),
        "event": feols_stream(EVENT_FML, rich.src, workdir=base / "event",
                              vcov={"CRV1": "firm_id"}, fe_dof=FE_DOF_PF,
                              verbose=False, outputs="keep"),
        "multi": feols_stream("log_earn + y2 ~ age_squared | worker_id + firm_id",
                              rich.src, workdir=base / "multi", vcov="hetero",
                              verbose=False, outputs="keep"),
    }
    yield out
    for r in out.values():
        r.cleanup()


# --------------------------------------------------------------------------
# HDFEResult
# --------------------------------------------------------------------------

def test_tidy_has_one_row_per_coefficient(fits):
    res = fits["ols"]
    tidy = res.tidy()
    assert tidy.height == len(res.coefnames)
    assert tidy["Coefficient"].to_list() == res.coefnames
    assert rel(tidy["Estimate"].to_numpy(), res.beta) < 1e-15
    assert rel(tidy["Std. Error"].to_numpy(), res.se) < 1e-15


def test_coef_maps_names_to_estimates(fits):
    res = fits["ols"]
    assert res.coef() == dict(zip(res.coefnames, res.beta))


def test_se_is_the_vcov_diagonal(fits):
    res = fits["ols"]
    assert rel(res.se, np.sqrt(np.diag(res.vcov))) < 1e-15


def test_summary_text_reports_the_model(fits):
    text = fits["ols"].summary_text()
    assert OLS_FML in text
    for token in ["worker_id", "firm_id", "RSS", "within R2", "solver"]:
        assert token in text, token


def test_residuals_and_fixef_are_lazy(fits):
    """Both come back as Polars LazyFrames, so nothing is read until collect."""
    res = fits["ols"]
    assert isinstance(res.resid(), pl.LazyFrame)
    assert isinstance(res.fixef("firm_id"), pl.LazyFrame)
    firm = res.fixef("firm_id").collect()
    assert firm.height == res.n_levels["firm_id"]
    assert {"firm_id", "fe_firm_id", "n_obs"} <= set(firm.columns)
    assert firm["n_obs"].sum() == pytest.approx(res.n_obs)


def test_fixef_exposes_only_meaningful_columns(fits):
    """Each dimension gets its levels, its estimated effect and its group size
    -- and nothing internal. The dimension the connected components were
    computed on also carries `component`, which is real information about
    identification rather than an implementation detail.
    """
    res = fits["ols"]
    expected = {
        "worker_id": {"worker_id", "fe_worker_id", "n_obs"},
        "firm_id": {"firm_id", "fe_firm_id", "n_obs", "component"},
        "year": {"year", "fe_year", "n_obs"},
    }
    for dim, columns in expected.items():
        frame = res.fixef(dim).collect()
        assert set(frame.columns) == columns, dim
        assert frame.height == res.n_levels[dim], dim


def test_interacted_fixef_expands_to_its_parts(rich, workdir):
    """An interacted dimension is keyed by the columns it was built from, not
    by an internal code."""
    res = feols_stream("log_earn ~ age_squared | worker_id + firm_id^year", rich.src,
                       workdir=workdir, verbose=False)
    frame = res.fixef("firm_id^year").collect()
    assert set(frame.columns) == {"firm_id", "year", "fe_firm_id^year", "n_obs",
                                 "component"}
    assert frame.height == res.n_levels["firm_id^year"]


# --------------------------------------------------------------------------
# HDFEMulti
# --------------------------------------------------------------------------

def test_multi_is_a_collection_of_results(fits):
    multi = fits["multi"]
    assert isinstance(multi, HDFEMulti)
    assert len(multi) == 2
    assert all(isinstance(r, HDFEResult) for r in multi)
    assert multi.fetch_model(0) is list(multi)[0]
    assert {r.depvar for r in multi} == {"log_earn", "y2"}


def test_multi_tidy_labels_each_model(fits):
    tidy = fits["multi"].tidy()
    assert "fml" in tidy.columns
    assert tidy["fml"].n_unique() == 2
    assert tidy.height == sum(len(r.coefnames) for r in fits["multi"])


def test_multi_summary_text_covers_every_model(fits):
    text = fits["multi"].summary_text()
    for r in fits["multi"]:
        assert r.fml in text


# --------------------------------------------------------------------------
# the pyfixest adapter
# --------------------------------------------------------------------------

def test_adapter_type_follows_the_model(fits):
    from pyfixest.estimation.models.feiv_ import Feiv
    from pyfixest.estimation.models.feols_ import Feols

    assert isinstance(fits["ols"].to_pyfixest(), Feols)
    adapter = fits["iv"].to_pyfixest()
    assert isinstance(adapter, Feiv)
    assert adapter._is_iv


def test_adapter_reports_the_same_numbers(fits):
    res = fits["ols"]
    adapter = res.to_pyfixest()
    assert list(adapter.coef().index) == res.coefnames
    assert rel(adapter.coef().to_numpy(), res.beta) < 1e-15
    assert rel(adapter.se().to_numpy(), res.se) < 1e-15
    assert rel(adapter.tstat().to_numpy(), res.beta / res.se) < 1e-12
    assert adapter.tidy().shape == (len(res.coefnames), 6)
    lo, hi = adapter.confint().to_numpy().T
    assert np.all(lo < res.beta) and np.all(res.beta < hi)


def test_adapter_matches_a_real_pyfixest_fit(rich, fits):
    """The adapter is only useful if pyfixest's reporting reads the same values
    out of it as out of one of its own models."""
    ref = pf_feols(OLS_FML, rich, vcov={"CRV1": "worker_id"})
    adapter = fits["ols"].to_pyfixest()
    assert list(adapter.coef().index) == list(ref.coef().index)
    assert rel(adapter.se().to_numpy(), ref.se().to_numpy()) < TOL_SE
    assert adapter._vcov_type == ref._vcov_type == "CRV"


def test_adapter_cannot_recompute_the_vcov(fits):
    """The adapter has no data behind it, so asking pyfixest to recompute the
    vcov must fail loudly and point at `with_vcov`."""
    adapter = fits["ols"].to_pyfixest()
    with pytest.raises(NotImplementedError, match="with_vcov"):
        adapter.vcov("hetero")


def test_pyfixest_summary_accepts_an_adapter(fits, capsys):
    import pyfixest as pf

    pf.summary(fits["iv"].to_pyfixest())
    out = capsys.readouterr().out
    assert "age_squared" in out
    assert "IV" in out


def test_etable_mixes_streaming_and_pyfixest_models(rich, fits):
    """`etable` accepts streaming results, multi-model results and ordinary
    pyfixest models in one call."""
    ref = pf_feols(OLS_FML, rich, vcov={"CRV1": "worker_id"})
    table = etable([fits["ols"], ref, fits["iv"]], type="df")
    assert table.shape[1] == 3
    assert "age_squared" in table.to_string()


def test_etable_expands_a_multi_result(fits):
    table = etable(fits["multi"], type="df")
    assert table.shape[1] == len(fits["multi"])


def test_etable_accepts_a_single_result(fits):
    assert etable(fits["ols"], type="df").shape[1] == 1


def test_iplot_accepts_an_event_study_adapter(fits):
    """`pf.iplot` needs the interaction terms flagged on the model; the adapter
    sets that from the `::` coefficient names."""
    import matplotlib
    matplotlib.use("Agg")
    import pyfixest as pf

    adapter = fits["event"].to_pyfixest()
    assert adapter._icovars, "interaction terms were not flagged for iplot"
    figure = pf.iplot(adapter, plot_backend="matplotlib")
    assert figure is not None

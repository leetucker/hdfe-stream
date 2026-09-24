"""The `StreamingHDFE` interface: variables as Polars expressions, option
validation, and the data sources it accepts.

These are the paths a caller uses when the formula front end is not wanted --
including when pyfixest is not installed at all.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from conftest import TOL_BETA, rel

from hdfe_stream import StreamingHDFE

FE = ["worker_id", "firm_id", "year"]


def test_covariates_can_be_polars_expressions(akm, workdir):
    """A covariate can be given as a (name, expression) pair, which is the
    point of the low-level interface: the expression is evaluated inside the
    streaming engine, so the column never has to exist on disk."""
    est = StreamingHDFE(
        y="log_earn",
        x=["age_squared", ("over_40", (pl.col("age") > 40).cast(pl.Float64))],
        fe=FE, workdir=workdir, verbose=False)
    res = est.fit(akm.src)

    assert res.coefnames == ["age_squared", "over_40"]
    assert np.all(np.isfinite(res.beta))


def test_expression_covariate_matches_a_materialized_column(akm, workdir):
    """Computing a covariate on the fly must give the same fit as computing it
    beforehand and reading it from the data."""
    materialized = akm.frame.with_columns(
        over_40=(pl.col("age") > 40).cast(pl.Float64))

    on_the_fly = StreamingHDFE(
        y="log_earn", x=["age_squared", ("over_40", (pl.col("age") > 40).cast(pl.Float64))],
        fe=FE, workdir=workdir / "expr", verbose=False).fit(akm.src)
    precomputed = StreamingHDFE(
        y="log_earn", x=["age_squared", "over_40"], fe=FE,
        workdir=workdir / "col", verbose=False).fit(materialized.lazy())

    assert on_the_fly.coefnames == precomputed.coefnames
    assert rel(on_the_fly.beta, precomputed.beta) < TOL_BETA


def test_accepts_a_lazyframe_as_well_as_a_path(akm, workdir):
    from_path = StreamingHDFE("log_earn", ["age_squared"], FE,
                              workdir=workdir / "path", verbose=False).fit(akm.src)
    from_lazy = StreamingHDFE("log_earn", ["age_squared"], FE,
                              workdir=workdir / "lazy", verbose=False).fit(akm.frame.lazy())
    assert rel(from_path.beta, from_lazy.beta) < 1e-12
    assert from_path.n_obs == from_lazy.n_obs


def test_several_outcomes_share_one_solve(akm, workdir):
    """Two dependent variables over the same fixed effects are estimated in one
    pass, and must agree with fitting them separately."""
    together = StreamingHDFE(["log_earn", ("shifted", pl.col("log_earn") * 2 + 1)],
                             ["age_squared"], FE, workdir=workdir / "both",
                             verbose=False).fit(akm.src)
    assert len(together) == 2

    alone = StreamingHDFE("log_earn", ["age_squared"], FE,
                          workdir=workdir / "one", verbose=False).fit(akm.src)
    first = together.fetch_model(0)
    assert rel(first.beta, alone.beta) < 1e-12
    # y -> 2y + 1 doubles the slope, and the intercept is absorbed by the FE
    assert rel(together.fetch_model(1).beta, 2 * alone.beta) < 1e-9


def test_fe_only_model_has_no_covariates(akm, workdir):
    res = StreamingHDFE("log_earn", [], FE, workdir=workdir, verbose=False).fit(akm.src)
    assert res.coefnames == []
    assert res.beta.size == 0
    assert np.isfinite(res.rss)


def test_keep_carries_extra_columns_into_the_residual_file(akm, workdir):
    res = StreamingHDFE("log_earn", ["age_squared"], ["worker_id", "firm_id"],
                        workdir=workdir, keep=["year", "age"], verbose=False).fit(akm.src)
    columns = res.resid().collect_schema().names()
    assert {"year", "age", "resid"} <= set(columns)


def test_keep_ignores_columns_that_are_already_fixed_effects(akm, workdir):
    """`year` is a fixed effect here, so it is already present and must not be
    duplicated."""
    res = StreamingHDFE("log_earn", ["age_squared"], FE, workdir=workdir,
                        keep=["year"], verbose=False).fit(akm.src)
    columns = res.resid().collect_schema().names()
    assert columns.count("year") == 1


def test_interacted_fixed_effects(akm, workdir):
    res = StreamingHDFE("log_earn", ["age_squared"], ["worker_id", "firm_id^year"],
                        workdir=workdir, verbose=False).fit(akm.src)
    assert set(res.n_levels) == {"worker_id", "firm_id^year"}
    assert res.n_levels["firm_id^year"] > res.n_levels["worker_id"] / 100


# --------------------------------------------------------------------------
# option validation
# --------------------------------------------------------------------------

@pytest.mark.parametrize("fe,message", [
    (["worker_id"], "at least two"),
    (["worker_id", "worker_id"], "duplicate"),
])
def test_fixed_effect_sets_are_validated(akm, workdir, fe, message):
    with pytest.raises(ValueError, match=message):
        StreamingHDFE("log_earn", ["age_squared"], fe, workdir=workdir)


@pytest.mark.parametrize("option,value,message", [
    ("solver", "magic", "solver must be"),
    ("precond", "magic", "precond must be"),
    ("assembly", "magic", "assembly must be"),
    ("outputs", "magic", "outputs must be"),
    ("weights_type", "magic", "weights_type must be"),
])
def test_options_are_validated(akm, workdir, option, value, message):
    kwargs = {option: value}
    if option == "weights_type":
        kwargs["weights"] = "age"
    with pytest.raises(ValueError, match=message):
        StreamingHDFE("log_earn", ["age_squared"], FE, workdir=workdir, **kwargs)


def test_stream_must_name_a_fixed_effect(akm, workdir):
    with pytest.raises(ValueError, match="is not one of the fixed effects"):
        StreamingHDFE("log_earn", ["age_squared"], FE, workdir=workdir, stream="nosuch")


def test_missing_fixed_effect_column_is_reported_by_name(akm, workdir):
    est = StreamingHDFE("log_earn", ["age_squared"], ["worker_id", "nosuch"],
                        workdir=workdir, verbose=False)
    with pytest.raises(ValueError, match="columns not found in data.*nosuch"):
        est.fit(akm.src)


def test_missing_covariate_column_surfaces_the_polars_error(akm, workdir):
    """Fixed-effect columns are checked up front, but a missing covariate is
    only discovered when Polars resolves the query. The error names the column
    and lists what is available, so it is still actionable -- just not raised
    by this library.
    """
    est = StreamingHDFE("log_earn", ["nosuch"], FE, workdir=workdir, verbose=False)
    with pytest.raises(pl.exceptions.ColumnNotFoundError, match="nosuch"):
        est.fit(akm.src)


def test_explicit_solver_refuses_to_exceed_its_memory_budget(akm, workdir):
    """With `solver="explicit"` and a budget too small for the reduced matrix,
    the fit must fail with an actionable message rather than allocating."""
    est = StreamingHDFE("log_earn", ["age_squared"], FE, workdir=workdir,
                        solver="explicit", max_s_gb=1e-7, verbose=False)
    with pytest.raises(MemoryError, match="max_s_gb"):
        est.fit(akm.src)


def test_auto_solver_falls_back_when_the_matrix_is_too_large(akm, workdir):
    """The same budget under `solver="auto"` must fall back to the streaming
    solver instead of failing."""
    est = StreamingHDFE("log_earn", ["age_squared"], FE, workdir=workdir,
                        solver="auto", max_s_gb=1e-7, verbose=False)
    res = est.fit(akm.src)
    assert res.solver_info["solver"] == "stream_cg"
    assert "fallback" in res.solver_info

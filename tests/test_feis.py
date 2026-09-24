"""Varying slopes on the streamed dimension (FEIS), and the choice of which
dimension to stream.

pyfixest has no varying-slopes estimator, so the reference here is a
brute-force OLS with one explicit dummy per worker per slope --
`y ~ x + C(worker_id):tc | worker_id + firm_id` in place of
`y ~ x | worker_id[t] + firm_id`. That is why this module uses the small
`trends` panel: the brute-force design matrix has a column per worker.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from conftest import FE_DOF_PF, TOL_BETA, TOL_RESID, TOL_SE, rel, resid_sorted, scaled

pytest.importorskip("pyfixest")

from hdfe_stream import StreamingHDFE, feols_stream  # noqa: E402

# (streaming formula, equivalent brute-force formula, extra options).
# `tc` is the centered year, so the worker-specific dummy slopes span the same
# space as the streamed dimension's slopes on `t`.
CASES = [
    ("y ~ x | worker_id[t] + firm_id",
     "y ~ x + C(worker_id):tc | worker_id + firm_id", {}),
    # slopes on a dimension that is not listed first
    ("y ~ x | firm_id + worker_id[t] + year",
     "y ~ x + C(worker_id):tc | firm_id + worker_id + year", {}),
    # two slopes, weighted
    ("y ~ x | worker_id[t, t2] + firm_id",
     "y ~ x + C(worker_id):tc + C(worker_id):t2 | worker_id + firm_id",
     {"weights": "wa"}),
    # varying slopes together with 2SLS
    ("y ~ x | worker_id[t] + firm_id | xe ~ z",
     "y ~ x + C(worker_id):tc | worker_id + firm_id | xe ~ z", {}),
]
IDS = ["slopes", "slopes-middle", "two-slopes-weighted", "slopes-iv"]
SOLVERS = ["explicit", "stream_cg"]      # `within` does not support slopes


@pytest.fixture(scope="module")
def centered(trends):
    """The brute-force reference needs the year centered, so that the dummy
    slopes are not collinear with the worker intercepts."""
    mid = float(np.mean(trends.frame["year"].unique().to_numpy()))
    frame = trends.frame.with_columns(tc=pl.col("t") - mid)

    class Centered:
        pass

    panel = Centered()
    panel.frame = frame
    panel.pandas = frame.to_pandas()
    panel.src = trends.src
    return panel


@pytest.fixture(scope="module")
def fitted(trends, centered, tmp_path_factory):
    """Streaming fit and brute-force reference for each case and solver."""
    from conftest import pf_feols

    base = tmp_path_factory.mktemp("feis")
    out = {}
    refs = {}
    for index, (fml, brute, options) in enumerate(CASES):
        for solver in SOLVERS:
            out[(index, solver)] = feols_stream(
                fml, trends.src, workdir=base / f"{index}{solver}", vcov="iid",
                cluster=["firm_id", "worker_id"], fe_dof=FE_DOF_PF, verbose=False,
                tol=1e-12, solver=solver, outputs="keep", **options)
        refs[index] = pf_feols(brute, centered, tol=1e-13, vcov="iid",
                               weights=options.get("weights"), collin_tol=1e-7)
    yield out, refs
    for r in out.values():
        r.cleanup()


@pytest.mark.parametrize("index", range(len(CASES)), ids=IDS)
@pytest.mark.parametrize("solver", SOLVERS)
def test_coefficients_match_brute_force(fitted, index, solver):
    fits, refs = fitted
    res = fits[(index, solver)]
    ref_coef = refs[index].coef()[res.coefnames].to_numpy()
    assert rel(res.beta, ref_coef) < TOL_BETA


@pytest.mark.parametrize("index", range(len(CASES)), ids=IDS)
@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("key,spec", [("iid", "iid"), ("hetero", "hetero"),
                                      ("CRV1:firm_id", {"CRV1": "firm_id"})])
def test_standard_errors_match_brute_force(fitted, index, solver, key, spec):
    """iid, heteroskedasticity-robust, and clustered on the non-streamed
    dimension. Clustering on the *streamed* dimension differs by a known
    degrees-of-freedom convention -- see the next test."""
    fits, refs = fitted
    res = fits[(index, solver)]
    ref = refs[index]
    ref.vcov(spec)
    idx = [list(ref.coef().index).index(c) for c in res.coefnames]
    assert rel(res.with_vcov(key).se, ref.se().to_numpy()[idx]) < TOL_SE


@pytest.mark.parametrize("index", range(len(CASES)), ids=IDS)
def test_clustering_on_streamed_dimension_differs_only_by_dof(fitted, index):
    """Clustering on the dimension that carries the slopes.

    fixest (and hdfe_stream) treat the worker slopes as fixed effects nested in
    the worker cluster, so they drop out of the parameter count K. The
    brute-force regression has them as ordinary covariates, so they stay in K.
    The standard errors must therefore differ by exactly the ratio of the two
    small-sample corrections, and by nothing else.
    """
    fits, refs = fitted
    res = fits[(index, "explicit")]
    ref = refs[index]
    ref.vcov({"CRV1": "worker_id"})
    idx = [list(ref.coef().index).index(c) for c in res.coefnames]

    n_slopes = res.diagnostics["fe_params"]["worker_id"] - res.n_levels["worker_id"]
    k_mine = len(res.coefnames) + res.k_fe - res.diagnostics["fe_params"]["worker_id"] + 1
    dof_ratio = np.sqrt((res.n_obs - k_mine) / (res.n_obs - (k_mine + n_slopes)))

    adjusted = res.with_vcov("CRV1:worker_id").se * dof_ratio
    assert rel(adjusted, ref.se().to_numpy()[idx]) < TOL_SE


@pytest.mark.parametrize("index", range(len(CASES)), ids=IDS)
@pytest.mark.parametrize("solver", SOLVERS)
def test_residuals_match_brute_force(fitted, index, solver):
    fits, refs = fitted
    res = fits[(index, solver)]
    theirs = np.sort(np.asarray(refs[index].resid()))
    assert scaled(resid_sorted(res), theirs) < TOL_RESID


def test_slope_fixed_effects_are_reported(fitted):
    """`fixef()` on the streamed dimension returns an intercept and one column
    per slope, plus the group size."""
    fits, _ = fitted
    res = fits[(0, "explicit")]
    fe = res.fixef("worker_id").collect()
    assert set(fe.columns) == {"worker_id", "fe_worker_id", "fe_worker_id[t]", "n_obs"}
    assert fe.height == res.n_levels["worker_id"]
    assert fe["n_obs"].sum() == pytest.approx(res.n_obs)


def test_covariate_spanned_by_worker_trends_is_dropped(trends, workdir):
    """age is birth year plus t, so age^2 lies in the span of the worker
    intercepts and their slopes on t and t2: it must be detected as collinear.
    """
    res = feols_stream("y ~ x + age_squared | worker_id[t, t2] + firm_id",
                       trends.src, workdir=workdir, verbose=False)
    assert "age_squared" in res.collin_vars
    assert res.coefnames == ["x"]


# --------------------------------------------------------------------------
# which dimension gets streamed
# --------------------------------------------------------------------------

@pytest.mark.parametrize("fe,stream,expected,reason", [
    # no slopes, no explicit choice: the highest-cardinality dimension
    (["firm_id", "worker_id"], None, "worker_id", "cardinality"),
    # slopes win over cardinality
    (["firm_id", "worker_id[t]"], None, "worker_id", "slopes"),
    # an explicit stream= wins over everything
    (["worker_id", "firm_id"], "firm_id", "firm_id", "option"),
    # cardinality again, with the larger dimension listed second
    (["year", "firm_id"], None, "firm_id", "cardinality"),
])
def test_streamed_dimension_choice(trends, workdir, fe, stream, expected, reason):
    est = StreamingHDFE("y", ["x"], fe, workdir=workdir, stream=stream, verbose=False)
    res = est.fit(trends.src)
    assert res.diagnostics["stream"]["dim"] == expected, res.diagnostics["stream"]["reason"]


def test_slopes_must_be_on_the_streamed_dimension(trends, workdir):
    with pytest.raises(ValueError, match="only supported on the streamed dimension"):
        StreamingHDFE("y", ["x"], ["worker_id[t]", "firm_id"], workdir=workdir,
                      stream="firm_id")


def test_slopes_on_two_dimensions_rejected(trends, workdir):
    with pytest.raises(ValueError, match="one fixed-effect dimension only"):
        StreamingHDFE("y", ["x"], ["worker_id[t]", "firm_id[t]"], workdir=workdir)


def test_within_solver_rejects_slopes(trends, workdir):
    with pytest.raises(ValueError, match="does not support varying slopes"):
        StreamingHDFE("y", ["x"], ["worker_id[t]", "firm_id"], workdir=workdir,
                      solver="within")


def test_slopes_without_intercept_not_supported(trends, workdir):
    """fixest's `[[...]]` (slopes without the fixed effect) is rejected
    explicitly rather than silently treated as `[...]`."""
    with pytest.raises(NotImplementedError, match=r"\[\[...\]\]"):
        StreamingHDFE("y", ["x"], ["worker_id[[t]]", "firm_id"], workdir=workdir)

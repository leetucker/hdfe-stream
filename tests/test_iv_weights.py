"""Two-stage least squares, weighted least squares, and fit statistics.

Weights come in both flavours: `aweights` (analytic, the default) and
`fweights` (frequency, where N is the sum of the weights rather than the row
count, which changes every degrees-of-freedom correction).
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import (FE_DOF_PF, TOL_BETA, TOL_RESID, TOL_SE, TOL_STAT,
                      pf_feols, rel, resid_sorted, scaled)

pytest.importorskip("pyfixest")

from hdfe_stream import feols_stream  # noqa: E402

# (formula, weight options)
CASES = [
    ("log_earn ~ age_squared + age_cubed | worker_id + firm_id + year",
     {"weights": "wa"}),
    ("log_earn ~ age_squared + age_cubed | worker_id + firm_id + year",
     {"weights": "wf", "weights_type": "fweights"}),
    ("y_iv ~ age_squared | worker_id + firm_id | x2 ~ z1 + z2", {}),
    ("y_iv ~ age_squared | worker_id + firm_id | x2 ~ z1 + z2", {"weights": "wa"}),
    ("y_iv ~ age_squared + i(year, treat, ref=2009) | worker_id + firm_id + year | x2 ~ z1",
     {}),
    ("y_iv ~ 1 | worker_id + firm_id^year | x2 ~ z1 + z2",
     {"weights": "wf", "weights_type": "fweights"}),
]
IDS = ["aweights", "fweights", "iv", "iv-aweights", "iv-interactions", "iv-fweights-nocovar"]
# an IV formula has two pipes: y ~ exog | fixed effects | endog ~ instruments
IV_CASES = [i for i, (fml, _) in enumerate(CASES) if fml.count("|") == 2]

VCOVS = [("iid", "iid"), ("hetero", "hetero"),
         ("CRV1:worker_id", {"CRV1": "worker_id"}), ("CRV1:state", {"CRV1": "state"})]


def reference_kwargs(options):
    return {"weights": options.get("weights"),
            "weights_type": options.get("weights_type", "aweights")}


@pytest.fixture(scope="module")
def fitted(rich, tmp_path_factory):
    """One streaming fit and one pyfixest reference per case.

    The streaming fit computes every vcov flavour in a single pass (`cluster=`
    asks for the extra ones), so the tests below switch between them with
    `with_vcov` instead of refitting.
    """
    base = tmp_path_factory.mktemp("iv")
    fits, refs = {}, {}
    for index, (fml, options) in enumerate(CASES):
        fits[index] = feols_stream(
            fml, rich.src, workdir=base / str(index), vcov={"CRV1": "worker_id"},
            cluster=["state"], fe_dof=FE_DOF_PF, verbose=False, n_buckets=3,
            batch_rows=5_000, tol=1e-11, outputs="keep", **options)
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
    """Including under fweights, where N is the sum of the weights and so
    every statistic that divides by N changes."""
    fits, refs = fitted
    res, ref = fits[index].with_vcov("iid"), refs[index]
    ref.vcov("iid")
    assert res.n_obs == ref._N
    for attr, ref_attr in [("r2", "_r2"), ("adj_r2", "_adj_r2"),
                           ("r2_within", "_r2_within"),
                           ("adj_r2_within", "_adj_r2_within"), ("rmse", "_rmse")]:
        mine, theirs = getattr(res, attr), getattr(ref, ref_attr)
        # pyfixest reports NaN for some statistics under IV; we must agree on
        # which ones are undefined, not just on the finite values
        assert np.isnan(mine) == np.isnan(theirs), attr
        if np.isfinite(theirs):
            assert rel(mine, theirs) < TOL_STAT, attr


@pytest.mark.parametrize("index", range(len(CASES)), ids=IDS)
def test_residuals_match_pyfixest(fitted, index):
    fits, refs = fitted
    theirs = np.sort(np.asarray(refs[index].resid()))
    assert scaled(resid_sorted(fits[index]), theirs) < TOL_RESID


@pytest.mark.parametrize("index", IV_CASES, ids=[IDS[i] for i in IV_CASES])
def test_iv_flagged_and_first_stage_available(fitted, index):
    fits, _ = fitted
    res = fits[index]
    assert res.is_iv
    assert len(res.first_stage) == 1
    assert len(res.f_stat_1st_stage) == 1
    first = res.first_stage[0]
    assert "x2" in first.depvar
    assert first.tidy().height == len(first.coefnames)


@pytest.mark.parametrize("index", IV_CASES, ids=[IDS[i] for i in IV_CASES])
@pytest.mark.parametrize("key,spec", [("iid", "iid"),
                                      ("CRV1:worker_id", {"CRV1": "worker_id"})],
                         ids=["iid", "CRV1"])
def test_first_stage_f_matches_wald_statistic(rich, tmp_path_factory, index, key, spec):
    """The reported first-stage F is the Wald statistic on the excluded
    instruments in the first-stage regression, under the model's own vcov.

    The reference is computed by hand from pyfixest's first-stage OLS rather
    than from its `IV_Diag`, because the two disagree under fweights -- and it
    is pyfixest's own first stage, so this pins the definition rather than
    copying a number.

    The statistic depends on the vcov, so each flavour needs its own fit.
    """
    from pyfixest.estimation.formula.parse import Formula

    fml, options = CASES[index]
    kwargs = reference_kwargs(options)
    base = tmp_path_factory.mktemp("ivf")

    res = feols_stream(fml, rich.src, workdir=base / key.replace(":", "_"), vcov=spec,
                       fe_dof=FE_DOF_PF, verbose=False, tol=1e-11, **options)

    spec_parsed = Formula.parse(fml)[0]
    model = pf_feols(fml, rich, vcov=spec, **kwargs)
    first = pf_feols(f"{spec_parsed.first_stage} | {spec_parsed.fixed_effects}",
                     rich, vcov=spec, **kwargs)

    excluded = [c for c in first.coef().index if c not in model.coef().index]
    idx = [list(first.coef().index).index(c) for c in excluded]
    coefs = first.coef().to_numpy()[idx]
    wald = coefs @ np.linalg.solve(first._vcov[np.ix_(idx, idx)], coefs) / len(idx)

    assert rel(res.f_stat_1st_stage[0], wald) < 1e-7
    res.cleanup()


def test_fweights_count_observations_as_the_weight_sum(fitted, rich):
    """Under fweights, N is the sum of the weights, not the number of rows."""
    fits, _ = fitted
    res = fits[1]
    assert res.weights_type == "fweights"
    assert res.n_obs == pytest.approx(rich.frame["wf"].sum())
    assert res.n_obs > rich.frame.height


def test_aweights_keep_the_row_count(fitted, rich):
    fits, _ = fitted
    res = fits[0]
    assert res.weights_type == "aweights"
    assert res.n_obs == rich.frame.height


def test_unknown_weights_type_rejected(rich, workdir):
    with pytest.raises(ValueError, match="weights_type must be"):
        feols_stream("log_earn ~ age_squared | worker_id + firm_id", rich.src,
                     workdir=workdir, weights="wa", weights_type="pweights")

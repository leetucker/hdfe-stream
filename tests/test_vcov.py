"""Multi-way clustered variance matrices (Cameron-Gelbach-Miller).

Two-way vcovs are checked against pyfixest. Three-way is checked against an
inclusion-exclusion sum computed directly from pyfixest's own score matrix,
because pyfixest itself does not do three-way clustering -- so the reference is
built from its primitives rather than copied from its output.

The comparison is on the variance matrix rather than the standard errors: a
multi-way vcov with few clusters can have negative diagonal entries, giving NaN
standard errors, and the two implementations must agree on those too.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pytest

from conftest import FE_DOF_PF, TOL_BETA, TOL_SE, TOL_VCOV, pf_feols, rel

pytest.importorskip("pyfixest")

from hdfe_stream import feols_stream  # noqa: E402

# (formula, two-way cluster spec, extra options)
CASES = [
    ("log_earn ~ age_squared + age_cubed | worker_id + firm_id + year",
     "worker_id+firm_id", {}),
    ("log_earn ~ age_squared + age_cubed | worker_id + firm_id + year",
     "firm_id+year", {}),
    # one cluster dimension is a fixed effect, the other is not
    ("log_earn ~ age_squared + age_cubed | worker_id + firm_id", "worker_id+state", {}),
    ("log_earn ~ age_squared + age_cubed | worker_id + firm_id", "firm_id+state", {}),
    # neither cluster dimension is a fixed effect
    ("log_earn ~ age_squared + i(year, treat, ref=2009) | worker_id + firm_id",
     "state+year", {}),
    # interacted fixed effects, weighted
    ("log_earn ~ age_squared + age_cubed | worker_id + firm_id^year",
     "firm_id+cohort", {"weights": "wa"}),
    # two-way clustering on an IV model
    ("y_iv ~ age_squared | worker_id + firm_id | x2 ~ z1 + z2", "worker_id+firm_id", {}),
    # frequency weights change every small-sample correction
    ("log_earn ~ age_squared + age_cubed | worker_id + firm_id", "worker_id+firm_id",
     {"weights": "wf", "weights_type": "fweights"}),
]
IDS = ["fe+fe", "fe+fe-year", "fe+nonfe", "fe2+nonfe", "nonfe+nonfe",
       "interacted-fe-weighted", "iv", "fweights"]


@pytest.fixture(scope="module")
def fitted(rich, tmp_path_factory):
    base = tmp_path_factory.mktemp("vcov")
    fits, refs = {}, {}
    for index, (fml, cluster, options) in enumerate(CASES):
        fits[index] = feols_stream(
            fml, rich.src, workdir=base / str(index), vcov={"CRV1": cluster},
            fe_dof=FE_DOF_PF, verbose=False, n_buckets=3, batch_rows=5_000,
            tol=1e-11, **options)
        refs[index] = pf_feols(fml, rich, vcov={"CRV1": cluster},
                               weights=options.get("weights"),
                               weights_type=options.get("weights_type", "aweights"))
    yield fits, refs
    for r in fits.values():
        r.cleanup()


@pytest.mark.parametrize("index", range(len(CASES)), ids=IDS)
def test_two_way_vcov_matches_pyfixest(fitted, index):
    fits, refs = fitted
    res, ref = fits[index], refs[index]
    scale = np.max(np.abs(ref._vcov))
    assert np.max(np.abs(res.vcov - ref._vcov)) / scale < TOL_VCOV


@pytest.mark.parametrize("index", range(len(CASES)), ids=IDS)
def test_two_way_coefficients_match_pyfixest(fitted, index):
    fits, refs = fitted
    assert rel(fits[index].beta, refs[index].coef().to_numpy()) < TOL_BETA


@pytest.mark.parametrize("index", range(len(CASES)), ids=IDS)
def test_two_way_degrees_of_freedom_match_pyfixest(fitted, index):
    """pyfixest uses min(G) over the clustering dimensions for the t
    distribution; hdfe_stream must land on the same number."""
    fits, refs = fitted
    res, ref = fits[index], refs[index]
    cluster = CASES[index][1]
    assert res.df_t == pytest.approx(ref._df_t, abs=0.5)
    assert min(res.n_clusters[cluster]) == min(ref._G[:2])


@pytest.mark.parametrize("index", range(len(CASES)), ids=IDS)
def test_nan_standard_errors_agree(fitted, index):
    """A two-way vcov can have negative variances when clusters are few. Where
    that happens the standard error is NaN, and it must be NaN in both."""
    fits, refs = fitted
    mine, theirs = fits[index].se, refs[index].se().to_numpy()
    assert np.array_equal(np.isnan(mine), np.isnan(theirs))
    finite = ~np.isnan(theirs)
    if finite.any():
        assert rel(mine[finite], theirs[finite]) < TOL_SE


def test_three_way_matches_inclusion_exclusion(rich, workdir):
    """Three-way clustering, against inclusion-exclusion over the seven
    non-empty subsets of the clustering dimensions.

    The reference is assembled from pyfixest's score matrix and bread, with
    pyfixest's own small-sample correction: G_min/(G_min - 1) * (N-1)/(N-K),
    where K drops the parameters of fixed effects that are themselves cluster
    variables (they are nested in the cluster).
    """
    fml = "log_earn ~ age_squared + age_cubed | worker_id + firm_id"
    ways = ["worker_id", "firm_id", "year"]
    frame = rich.pandas

    ref = pf_feols(fml, rich, vcov="hetero")
    scores, bread = ref._scores, np.linalg.inv(ref._tZX)
    n_obs = ref._N
    n_params = len(ref.coef()) + sum(ref._k_fe) - 1

    group_counts = [frame[w].nunique() for w in ways]
    g_min = min(group_counts)
    nested = [d for d in ("worker_id", "firm_id") if d in ways]
    k_cluster = n_params - sum(frame[d].nunique() for d in nested) + len(nested)

    expected = np.zeros_like(bread)
    for size in range(1, len(ways) + 1):
        for combo in combinations(ways, size):
            if size > 1:
                key = frame[list(combo)].astype(str).agg("-".join, axis=1)
            else:
                key = frame[combo[0]]
            codes = key.factorize()[0]
            summed = np.column_stack([np.bincount(codes, weights=scores[:, j])
                                      for j in range(scores.shape[1])])
            correction = g_min / (g_min - 1) * (n_obs - 1) / (n_obs - k_cluster)
            expected += (-1) ** (size + 1) * correction * bread @ (summed.T @ summed) @ bread

    res = feols_stream(fml, rich.src, workdir=workdir,
                       vcov={"CRV1": "+".join(ways)}, fe_dof=FE_DOF_PF,
                       verbose=False, tol=1e-11)
    assert res.n_clusters["+".join(ways)] == group_counts
    assert res.df_t == g_min - 1
    assert rel(res.se, np.sqrt(np.diag(expected))) < TOL_SE


def test_several_vcovs_from_one_pass(rich, workdir):
    """`cluster=` asks for extra vcovs during the single residual pass, and
    `with_vcov` switches between them without refitting."""
    res = feols_stream("log_earn ~ age_squared + age_cubed | worker_id + firm_id",
                       rich.src, workdir=workdir, vcov={"CRV1": "worker_id"},
                       cluster=["firm_id", "worker_id+firm_id", "state"],
                       fe_dof=FE_DOF_PF, verbose=False, tol=1e-11)
    assert set(res.all_vcovs) >= {"iid", "hetero", "CRV1:worker_id", "CRV1:firm_id",
                                  "CRV1:worker_id+firm_id", "CRV1:state"}
    for key in res.all_vcovs:
        switched = res.with_vcov(key)
        assert switched.vcov_type == key
        assert rel(switched.beta, res.beta) < 1e-15      # same fit, different vcov


def test_unrequested_vcov_raises(rich, workdir):
    """A vcov that was not computed during the pass cannot be recovered later,
    and says so rather than returning something wrong."""
    res = feols_stream("log_earn ~ age_squared | worker_id + firm_id", rich.src,
                       workdir=workdir, vcov="iid", verbose=False)
    with pytest.raises(KeyError, match="was not computed"):
        res.with_vcov({"CRV1": "region"})


@pytest.mark.parametrize("spec,message", [
    ("bootstrap", "unsupported vcov"),
    ({"CRV3": "worker_id"}, "only CRV1"),
])
def test_unsupported_vcov_rejected(rich, workdir, spec, message):
    with pytest.raises(ValueError, match=message):
        feols_stream("log_earn ~ age_squared | worker_id + firm_id", rich.src,
                     workdir=workdir, vcov=spec, verbose=False)


def test_cluster_spelling_is_normalized(rich, workdir):
    """'worker_id + firm_id' and 'worker_id+firm_id' are the same request."""
    res = feols_stream("log_earn ~ age_squared | worker_id + firm_id", rich.src,
                       workdir=workdir, vcov={"CRV1": "worker_id + firm_id"},
                       verbose=False)
    assert res.vcov_type == "CRV1:worker_id+firm_id"

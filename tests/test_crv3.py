"""CRV3, the cluster jackknife, against pyfixest.

hdfe_stream computes it by downdating, which is exact when every fixed effect
is nested within the clusters or there are none; anything else is refused.
pyfixest downdates without fixed effects and refits the model once per cluster
with them, so the cases with fixed effects check the downdate against real
refits. That is why this module has its own small panel: the refits cost one
pyfixest fit per cluster.
"""

from __future__ import annotations

import pytest

from conftest import FE_DOF_PF, TOL_BETA, TOL_SE, rel

pytest.importorskip("pyfixest")

from hdfe_stream import feols_stream  # noqa: E402
from hdfe_stream.simulate import simulate_rich  # noqa: E402

X = "age_squared + age_cubed"
CASES = [
    # no fixed effects: clusters that are no dimension at all
    (f"log_earn ~ {X}", "state", {}),
    (f"log_earn ~ {X} + i(occ)", "worker_id", {}),
    (f"log_earn ~ {X}", "state", {"weights": "wa"}),
    (f"log_earn ~ {X}", "state", {"weights": "wf", "weights_type": "fweights"}),
    # the streamed dimension, clustered by itself (jackknifed chunk by chunk)
    (f"log_earn ~ {X} | worker_id", "worker_id", {}),
    (f"log_earn ~ {X} | firm_id", "firm_id", {"weights": "wa"}),
    # a dimension nested in a coarser cluster (per-cluster sums)
    (f"log_earn ~ {X} | firm_id^year", "firm_id", {}),
    (f"log_earn ~ {X} | firm_id + firm_id^year", "firm_id", {}),
]
IDS = ["ols-state", "ols-worker", "ols-aweights", "ols-fweights", "worker",
       "firm-aweights", "firmyear-in-firm", "two-nested"]


@pytest.fixture(scope="module")
def small(tmp_path_factory):
    frame = simulate_rich(n_workers=400, n_firms=25, seed=4)
    path = tmp_path_factory.mktemp("crv3") / "small.parquet"
    frame.write_parquet(path)
    return path, frame.to_pandas()


@pytest.mark.parametrize("fml,cluster,options", CASES, ids=IDS)
def test_crv3_matches_pyfixest(small, tmp_path, fml, cluster, options):
    import pyfixest as pf

    path, pdf = small
    # small batches and several buckets, so clusters straddle chunks
    res = feols_stream(fml, str(path), workdir=tmp_path, vcov={"CRV3": cluster},
                       fe_dof=FE_DOF_PF, verbose=False, batch_rows=900, n_buckets=2,
                       **options)
    with res:
        ref = pf.feols(fml, pdf, vcov={"CRV3": cluster},
                       weights=options.get("weights"),
                       weights_type=options.get("weights_type", "aweights"))
        assert res.vcov_type == f"CRV3:{cluster}"
        assert res.coefnames == list(ref.coef().index)
        assert rel(res.beta, ref.coef().to_numpy()) < TOL_BETA
        assert rel(res.se, ref.se().to_numpy()) < TOL_SE
        assert res.df_t == ref._df_t
        # CRV1 by the same variable comes along, and can be switched to
        crv1 = res.with_vcov({"CRV1": cluster})
        ref.vcov({"CRV1": cluster})
        assert rel(crv1.se, ref.se().to_numpy()) < TOL_SE


def test_string_form_and_reporting(small, tmp_path):
    path, _ = small
    with feols_stream(f"log_earn ~ {X} | worker_id", str(path), workdir=tmp_path,
                      vcov="CRV3:worker_id", verbose=False) as res:
        assert res.vcov_type == "CRV3:worker_id"
        view = res.to_pyfixest()
        assert view._vcov_type_detail == "CRV3" and view._clustervar == ["worker_id"]


def test_fixed_effects_not_nested_are_refused(small, tmp_path):
    """Worker and firm effects clustered by worker: dropping a worker moves
    the firm effects of everyone else, so the downdate is not the jackknife."""
    path, _ = small
    with pytest.raises(ValueError, match=r"'firm_id' is not nested within 'worker_id'"
                                         r".*only CRV1 is supported.*pyfixest"):
        feols_stream(f"log_earn ~ {X} | worker_id + firm_id", str(path),
                     workdir=tmp_path, vcov={"CRV3": "worker_id"}, verbose=False)


def test_iv_is_refused(small, tmp_path):
    path, _ = small
    with pytest.raises(ValueError, match="not available for IV"):
        feols_stream("y_iv ~ age_squared | worker_id | x2 ~ z1 + z2", str(path),
                     workdir=tmp_path, vcov={"CRV3": "worker_id"}, verbose=False)


@pytest.mark.parametrize("vcov,message", [
    ({"CRV3": "worker_id+firm_id"}, "one-way only"),
    ({"CRV3": ""}, "needs a cluster variable"),
    ({"CRV2": "worker_id"}, "unsupported vcov"),
])
def test_vcov_requests_are_validated(small, tmp_path, vcov, message):
    path, _ = small
    with pytest.raises(ValueError, match=message):
        feols_stream(f"log_earn ~ {X}", str(path), workdir=tmp_path, vcov=vcov,
                     verbose=False)

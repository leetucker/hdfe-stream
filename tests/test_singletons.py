"""`fixef_rm`: dropping singleton observations, as pyfixest does.

The data are small and hand-built so that every kind of singleton is present:
workers and firms seen once, and a chain in which dropping one row leaves
another alone in a different dimension (found only in a second round).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import polars as pl
import pytest

from conftest import FE_DOF_PF, TOL_BETA, TOL_SE, rel

pf = pytest.importorskip("pyfixest")

from hdfe_stream import StreamingHDFE, feols_stream, fepois_stream  # noqa: E402


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    rng = np.random.default_rng(11)
    n = 3000
    w = rng.integers(0, 400, n)
    f = rng.integers(0, 40, n)
    t = rng.integers(0, 6, n)
    w[:30] = 10_000 + np.arange(30)           # singleton workers
    f[30:33] = 1000 + np.arange(3)            # singleton firms
    w[33], f[33] = 20_000, 2000               # worker 20000 has two rows, one on
    w[34], f[34] = 20_000, 2001               # a singleton firm and one on another:
    f[35] = 2001                              # dropping row 33 strands row 34 ... if
    w[35] = 20_001                            # firm 2001 then has only row 35 left
    x = rng.normal(size=n)
    frame = pd.DataFrame({"y": x + rng.normal(size=n) + 0.1 * (w % 7), "x": x,
                          "w": w, "f": f, "t": t,
                          "cnt": rng.poisson(np.exp(0.2 * x))})
    path = tmp_path_factory.mktemp("singletons") / "d.parquet"
    frame.to_parquet(path)
    return path, frame


def fit(path, fml="y ~ x | w + f", **kwargs):
    kwargs = {"fe_dof": FE_DOF_PF, "verbose": False, "n_buckets": 2, "batch_rows": 700,
              **kwargs}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return feols_stream(fml, path, **kwargs)


def pyfixest_fit(frame, fml, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pf.feols(fml, frame, **kwargs)


@pytest.mark.parametrize("fixef_rm", ["singleton", "none"])
@pytest.mark.parametrize("vcov", ["iid", "hetero", {"CRV1": "w"}], ids=["iid", "hetero", "crv1"])
@pytest.mark.parametrize("fml", ["y ~ x | w", "y ~ x | w + f", "y ~ x | w + f + t",
                                 "y ~ x | w^t + f"])
def test_matches_pyfixest(data, fml, vcov, fixef_rm):
    path, frame = data
    with fit(path, fml, vcov=vcov, fixef_rm=fixef_rm) as res:
        ref = pyfixest_fit(frame, fml, vcov=vcov, fixef_rm=fixef_rm)
        assert res.n_obs == ref._N
        assert rel(res.beta, ref.coef().to_numpy()) < TOL_BETA
        assert rel(res.se, ref.se().to_numpy()) < TOL_SE


def test_default_drops_singletons(data):
    path, frame = data
    with fit(path, fe_dof="exact") as default, \
            fit(path, fixef_rm="singleton") as explicit, \
            fit(path, fixef_rm="none") as kept:
        assert default.n_obs == explicit.n_obs < kept.n_obs == len(frame)
        assert rel(default.se, explicit.se) < TOL_SE
        assert kept.diagnostics["singletons"]["observations"] == 0


def test_chain_of_singletons_takes_a_second_round(data):
    path, _ = data
    with fit(path, "y ~ x | w^t + f") as res:
        found = res.diagnostics["singletons"]
        assert found["rounds"] >= 2
        assert found["observations"] == sum(found["levels"].values())
        assert set(found["levels"]) <= {"w^t", "f"}


def test_reports_what_it_dropped(data):
    path, frame = data
    with pytest.warns(UserWarning, match="singleton observations dropped"):
        res = feols_stream("y ~ x | w + f", path, verbose=False)
    with res:
        dropped = res.diagnostics["singletons"]["observations"]
        assert dropped == len(frame) - res.n_obs > 0
        assert f"dropped as singletons: {dropped:,} observations" in res.summary_text()


def test_residuals_exclude_dropped_rows(data):
    path, _ = data
    with fit(path, outputs="keep") as res:
        assert res.resid().collect().height == res.n_obs


def test_nothing_dropped_without_fixed_effects(data):
    path, frame = data
    with fit(path, "y ~ x") as res:
        assert res.n_obs == len(frame)
        assert res.diagnostics["singletons"]["observations"] == 0


def test_poisson_matches_pyfixest(data):
    path, frame = data
    fml = "cnt ~ x | w + f"
    for fixef_rm in ("singleton", "none"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = fepois_stream(fml, path, fe_dof=FE_DOF_PF, verbose=False,
                                fixef_rm=fixef_rm, tol=1e-11)
            ref = pf.fepois(fml, frame, fixef_rm=fixef_rm, iwls_tol=1e-12)
        with res:
            assert res.n_obs == ref._N
            assert rel(res.beta, ref.coef().to_numpy()) < 1e-6


def test_low_level_interface_takes_the_option(data):
    path, frame = data
    for fixef_rm, expected in (("singleton", None), ("none", len(frame))):
        est = StreamingHDFE("y", ["x"], ["w", "f"], fixef_rm=fixef_rm, verbose=False)
        with est.fit(pl.scan_parquet(path)) as res:
            assert res.n_obs < len(frame) if expected is None else res.n_obs == expected


def test_invalid_option_is_rejected(data):
    path, _ = data
    with pytest.raises(ValueError, match="fixef_rm"):
        feols_stream("y ~ x | w", path, fixef_rm="drop", verbose=False)

"""The estimator's own columns cannot collide with the user's.

Everything the estimator adds to the rows starts with `__hdfe_`, and a source
with a column of that prefix is refused. Before that, a column named `w`
(with weights), `v0`, `t1`, `c1`, `k0`, `gcode` or `_bucket` broke a fit.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from conftest import FE_DOF_PF, TOL_BETA, TOL_SE, rel

from hdfe_stream import StreamingHDFE, feols_stream, fepois_stream  # noqa: E402

# the names the estimator used to give its own columns, as the user's
OLD_INTERNAL = {"worker": "w", "firm": "c1", "year": "gcode", "wt": "v0", "cl": "k0",
                "x": "t1", "z": "_bucket", "extra": "v1"}


@pytest.fixture(scope="module")
def frame():
    rng = np.random.default_rng(31)
    n = 4000
    d = pd.DataFrame({"worker": rng.integers(0, 300, n), "firm": rng.integers(0, 30, n),
                      "year": rng.integers(0, 6, n), "wt": rng.uniform(0.5, 2, n),
                      "cl": rng.integers(0, 80, n), "x": rng.normal(size=n),
                      "z": rng.normal(size=n), "extra": rng.normal(size=n)})
    d["y"] = d["x"] - 0.5 * d["z"] + rng.normal(size=n) + 0.05 * (d["worker"] % 9)
    d["cnt"] = rng.poisson(np.exp(0.2 * d["x"]))
    return d


def write(frame, tmp_path, name, rename=None):
    path = tmp_path / f"{name}.parquet"
    frame.rename(columns=rename or {}).to_parquet(path)
    return path


def fit(path, fml, **kwargs):
    kwargs = {"fe_dof": FE_DOF_PF, "verbose": False, "n_buckets": 2, "batch_rows": 900,
              **kwargs}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return feols_stream(fml, path, **kwargs)


def test_columns_named_like_the_old_internal_ones_work(frame, tmp_path):
    new = OLD_INTERNAL
    plain = fit(write(frame, tmp_path, "plain"), "y ~ x + z | worker + firm + year",
                weights="wt", vcov={"CRV1": "cl"}, cluster=["worker"], keep=["extra"])
    clashing = fit(write(frame, tmp_path, "clash", new),
                   f"y ~ {new['x']} + {new['z']} | {new['worker']} + {new['firm']} + {new['year']}",
                   weights=new["wt"], vcov={"CRV1": new["cl"]}, cluster=[new["worker"]],
                   keep=[new["extra"]])
    with plain, clashing:
        assert clashing.n_obs == plain.n_obs
        assert rel(clashing.beta, plain.beta) < TOL_BETA
        assert rel(clashing.se, plain.se) < TOL_SE
        assert rel(clashing.with_vcov(f"CRV1:{new['worker']}").se,
                   plain.with_vcov("CRV1:worker").se) < TOL_SE
        cols = clashing.resid().collect_schema().names()
        assert new["extra"] in cols and new["worker"] in cols


def test_a_column_named_w_with_weights_and_a_glm(frame, tmp_path):
    path = write(frame, tmp_path, "w", {"worker": "w"})
    with fit(path, "y ~ x | w + firm", weights="wt") as res:
        assert res.n_obs == len(frame)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with fepois_stream("cnt ~ x | w + firm", path, weights="wt", fe_dof=FE_DOF_PF,
                           verbose=False) as pois:
            assert pois.n_obs == len(frame)


def test_slope_variables_named_like_the_internal_ones(tmp_path):
    rng = np.random.default_rng(5)
    n = 3000
    d = pd.DataFrame({"worker": rng.integers(0, 200, n), "firm": rng.integers(0, 20, n),
                      "t1": rng.normal(size=n), "x": rng.normal(size=n)})
    d["y"] = d["x"] + 0.3 * d["t1"] * (d["worker"] % 5) + rng.normal(size=n)
    d.to_parquet(tmp_path / "a.parquet")
    d.rename(columns={"t1": "slope", "firm": "c1"}).to_parquet(tmp_path / "b.parquet")
    with fit(tmp_path / "a.parquet", "y ~ x | worker[t1] + firm") as a, \
            fit(tmp_path / "b.parquet", "y ~ x | worker[slope] + c1") as b:
        assert rel(a.beta, b.beta) < TOL_BETA


@pytest.mark.parametrize("name", ["__hdfe_w", "__hdfe_anything"])
def test_a_source_column_with_the_reserved_prefix_is_refused(frame, tmp_path, name):
    path = write(frame, tmp_path, "pre", {"extra": name})
    with pytest.raises(ValueError, match="__hdfe_"):
        fit(path, "y ~ x | worker")


def test_sample_refuses_the_reserved_prefix_too(frame, tmp_path):
    path = write(frame, tmp_path, "ok")
    with fit(path, "y ~ x | worker") as res:
        res._sample["source"] = res._sample["source"].rename({"extra": "__hdfe_extra"})
        with pytest.raises(ValueError, match="__hdfe_"):
            res.sample()


def test_row_id_may_not_use_the_reserved_prefix():
    with pytest.raises(ValueError, match="reserved"):
        StreamingHDFE("y", ["x"], ["worker"], row_id="__hdfe_pos")

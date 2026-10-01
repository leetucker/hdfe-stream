"""`row_id` in `resid()`, and `sample()`: which rows of the source a fit used.

The point of both is that the residuals can be put back on the source rows
without the estimator knowing any key, and that the rows the fit dropped, and
why, are visible. The checks are therefore end to end: join on `row_id` and
see that the source's own values add up.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import polars as pl
import pytest

from conftest import FE_DOF_PF

pf = pytest.importorskip("pyfixest")

from hdfe_stream import StreamingHDFE, feols_stream, fepois_stream  # noqa: E402


@pytest.fixture(scope="module")
def frame():
    rng = np.random.default_rng(23)
    n = 3000
    w = rng.integers(0, 400, n)
    f = rng.integers(0, 40, n)
    w[:25] = 10_000 + np.arange(25)             # singleton workers
    x = rng.normal(size=n)
    x2 = rng.normal(size=n)
    d = pd.DataFrame({"y": x + rng.normal(size=n) + 0.1 * (w % 7), "x": x, "x2": x2,
                      "w": w, "f": f, "wt": rng.uniform(0.5, 2, n),
                      "cnt": rng.poisson(np.exp(0.2 * x))})
    d.loc[[100, 101], "x"] = np.nan             # missing covariate
    d.loc[[200], "f"] = np.nan                  # missing fixed effect
    d.loc[[300], "wt"] = np.nan                 # missing weight
    d.loc[[400], "x2"] = -1.0                   # log(x2) is not finite
    return d


@pytest.fixture(scope="module")
def path(frame, tmp_path_factory):
    p = tmp_path_factory.mktemp("sample") / "d.parquet"
    frame.to_parquet(p)
    return p


def fit(fml, source, **kwargs):
    kwargs = {"fe_dof": FE_DOF_PF, "verbose": False, "n_buckets": 2, "batch_rows": 600,
              **kwargs}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return feols_stream(fml, source, **kwargs)


def test_residuals_join_back_on_row_id(path, frame):
    with fit("y ~ x | w + f", path) as res:
        resid = res.resid().collect()
        assert "row_id" in resid.columns
        source = pl.read_parquet(path).with_row_index("row_id")
        joined = source.join(resid.select("row_id", "fe_w", "fe_f", "xb", "resid"),
                             on="row_id")
        assert joined.height == res.n_obs
        # the outcome, from the source's own column, is what the pieces add up to
        gap = (joined["y"] - joined["fe_w"] - joined["fe_f"] - joined["xb"]
               - joined["resid"]).abs().max()
        assert gap < 1e-8


def test_sample_says_which_rows_were_used_and_why_not(path, frame):
    with fit("y ~ x | w + f", path) as res:
        s = res.sample().collect()
        assert s.height == len(frame)
        assert s["row_id"].to_list() == list(range(len(frame)))
        assert s.columns[0] == "row_id" and s.columns[-2:] == ["in_sample", "dropped_because"]
        reasons = dict(s["dropped_because"].value_counts().iter_rows())
        assert reasons[None] == s["in_sample"].sum() == res.n_obs
        assert reasons["missing"] == 3          # two covariates and one fixed effect
        assert reasons["singleton"] > 0
        assert set(s.filter(pl.col("in_sample"))["row_id"]) == set(res.resid().collect()["row_id"])
        assert res.diagnostics["singletons"]["observations"] == reasons["singleton"]


def test_sample_is_the_rows_pyfixest_used(path, frame):
    with fit("y ~ x | w + f", path) as res:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ref = pf.feols("y ~ x | w + f", frame)
        used = res.sample().filter(pl.col("in_sample")).collect()["row_id"].to_list()
        assert used == sorted(ref._data.index.tolist())


def test_nothing_is_dropped_as_a_singleton_when_asked_not_to(path):
    with fit("y ~ x | w + f", path, fixef_rm="none") as res:
        reasons = set(res.sample().collect()["dropped_because"].drop_nulls())
        assert reasons == {"missing"}


def test_transformed_covariates_and_weights_count_as_missing(path):
    with fit("y ~ log(x2) | f", path, weights="wt", fixef_rm="none") as res:
        s = res.sample().collect()
        assert s.filter(pl.col("row_id").is_in([300, 400]))["dropped_because"].to_list() \
            == ["missing", "missing"]
        assert s["in_sample"].sum() == res.n_obs


def test_without_fixed_effects(path):
    with fit("y ~ x", path) as res:
        s = res.sample().collect()
        assert s["in_sample"].sum() == res.n_obs
        assert set(s["dropped_because"].drop_nulls()) == {"missing"}
        assert set(res.resid().collect()["row_id"]) == set(
            s.filter(pl.col("in_sample"))["row_id"])


def test_row_id_is_the_position_in_a_multi_file_source(frame, tmp_path):
    parts = np.array_split(np.arange(len(frame)), 4)
    for i, ix in enumerate(parts):
        frame.iloc[ix].to_parquet(tmp_path / f"part{i}.parquet", row_group_size=250)
    with fit("y ~ x | w + f", str(tmp_path / "part*.parquet"), batch_rows=300) as res:
        resid = res.resid().collect()
        source = pl.concat([pl.read_parquet(tmp_path / f"part{i}.parquet")
                            for i in range(4)]).with_row_index("row_id")
        joined = source.join(resid, on="row_id", suffix="_r")
        gap = (joined["y"] - joined["fe_w"] - joined["fe_f"] - joined["xb"]
               - joined["resid"]).abs().max()
        assert gap < 1e-8
        assert res.sample().collect()["in_sample"].sum() == res.n_obs


def test_a_lazyframe_source_works_like_a_path(path):
    lf = pl.scan_parquet(path).filter(pl.col("w") < 300)
    with fit("y ~ x | w + f", lf) as res:
        s = res.sample().collect()
        assert s.height == lf.select(pl.len()).collect().item()
        assert s["in_sample"].sum() == res.n_obs


def test_sample_refuses_a_source_whose_row_count_changed(path, frame):
    # a plan that returns a different number of rows each time it runs
    state = {"keep": len(frame)}
    lf = pl.scan_parquet(path).filter(
        pl.int_range(pl.len()).map_batches(lambda s: s < state["keep"], return_dtype=pl.Boolean))
    with fit("y ~ x | w", lf) as res:
        state["keep"] = len(frame) - 10
        with pytest.raises(ValueError, match="rows"):
            res.sample()
        assert res.sample(check=False).collect().height == len(frame) - 10


def test_multi_model_sample_is_a_dict_by_formula(path):
    with fit("y + x2 ~ csw(x, wt) | sw(w, w + f)", path) as multi:
        frames = multi.sample()
        assert list(frames) == [r.fml for r in multi]
        counts = {fml: f.collect()["in_sample"].sum() for fml, f in frames.items()}
        for r in multi:
            assert counts[r.fml] == r.n_obs
        # the same fixed effects share a sample; a different set has its own
        by_fe = {}
        for r in multi:
            by_fe.setdefault(tuple(r.fe_names), set()).add(r.n_obs)
        assert all(len(v) == 1 for v in by_fe.values())
        assert len(by_fe) == 2 and len({n for v in by_fe.values() for n in v}) == 2
        ids = {fml: tuple(f.collect()["row_id"]) for fml, f in frames.items()}
        assert len(set(ids.values())) == 1


def test_poisson_separation_is_a_reason(tmp_path):
    rng = np.random.default_rng(3)
    n = 2000
    d = pd.DataFrame({"w": rng.integers(0, 150, n), "f": rng.integers(0, 15, n),
                      "x": rng.normal(size=n)})
    d["cnt"] = rng.poisson(np.exp(0.3 * d["x"]))
    d.loc[d["f"] == 3, "cnt"] = 0               # firm 3 never has a positive count
    p = tmp_path / "sep.parquet"
    d.to_parquet(p)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = fepois_stream("cnt ~ x | w + f", p, fe_dof=FE_DOF_PF, verbose=False)
    with res:
        s = res.sample().collect()
        reasons = dict(s["dropped_because"].value_counts().iter_rows())
        assert reasons["separation"] == res.diagnostics["separation"]["observations"] > 0
        assert reasons[None] == res.n_obs
        assert set(s.filter(pl.col("dropped_because") == "separation")["f"]) == {3}


def test_a_source_column_named_row_id_is_refused_unless_renamed(frame, tmp_path):
    p = tmp_path / "clash.parquet"
    frame.assign(row_id=np.arange(len(frame))).to_parquet(p)
    with pytest.raises(ValueError, match="row_id"):
        fit("y ~ x | w", p)
    with fit("y ~ x | w", p, row_id="__pos") as res:
        assert "__pos" in res.resid().collect().columns
        assert res.sample().collect().columns[0] == "__pos"


def test_keep_columns_still_ride_along(path):
    with fit("y ~ x | w + f", path, keep=["x2"]) as res:
        cols = res.resid().collect_schema().names()
        assert "x2" in cols and "row_id" in cols


def test_low_level_interface(path):
    est = StreamingHDFE("y", ["x"], ["w", "f"], verbose=False, row_id="pos")
    with est.fit(pl.scan_parquet(path)) as res:
        assert "pos" in res.resid().collect().columns
        assert res.sample().collect()["in_sample"].sum() == res.n_obs

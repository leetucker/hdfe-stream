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


def test_singletons_of_every_dimension_are_reasons():
    """With three fixed effects, rows that are singletons only in the third
    are dropped too: sample() agrees with the fit's counts and with the rows
    left by dropping singletons to a fixed point."""
    rng = np.random.default_rng(5)
    n = 2000
    df = pl.DataFrame({"y": rng.normal(size=n), "x": rng.normal(size=n),
                       "a": np.r_[1000 + np.arange(5), rng.integers(0, 50, n - 5)],
                       "b": np.r_[rng.integers(0, 40, 5), 1000 + np.arange(5),
                                  rng.integers(0, 40, n - 10)],
                       "c": np.r_[rng.integers(0, 30, n - 10), 1000 + np.arange(10)]})
    keep = df.with_row_index("row_id")
    while True:
        before = keep.height
        for d in ("a", "b", "c"):
            keep = keep.filter(pl.len().over(d) > 1)
        if keep.height == before:
            break
    with feols_stream("y ~ x | a + b + c", df, verbose=False) as res:
        s = res.sample().collect()
        assert res.diagnostics["singletons"]["levels"] == {"a": 5, "b": 5, "c": 10}
        reasons = dict(s["dropped_because"].value_counts().iter_rows())
        assert reasons["singleton"] == res.diagnostics["singletons"]["observations"]
        assert s["in_sample"].sum() == res.n_obs
        assert s.filter("in_sample")["row_id"].to_list() == keep["row_id"].to_list()


def test_nan_in_a_fixed_effect_or_cluster_is_missing():
    """A NaN in a floating-point fixed effect or cluster drops the row, as a
    null does, rather than forming a level of its own. pyfixest drops the
    fixed effects the same way."""
    rng = np.random.default_rng(11)
    n = 1000
    df = pl.DataFrame({"y": rng.normal(size=n), "x": rng.normal(size=n),
                       "a": rng.integers(0, 20, n).astype(float),
                       "b": rng.integers(0, 10, n).astype(float),
                       "g": rng.integers(0, 30, n).astype(float)})
    df = df.with_columns(
        pl.when(pl.int_range(n) < 20).then(float("nan")).otherwise(pl.col("a")).alias("a"),
        pl.when(pl.int_range(n).is_between(20, 29)).then(float("nan"))
          .otherwise(pl.col("g")).alias("g"))
    with feols_stream("y ~ x | a + b", df, vcov={"CRV1": "g"}, verbose=False) as res:
        s = res.sample().collect()
        assert res.n_obs == n - 30 == s["in_sample"].sum()
        assert s.filter(pl.col("dropped_because") == "missing")["row_id"].to_list() \
            == list(range(30))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # pyfixest refuses a missing cluster rather than dropping the
            # row, so it gets the data without those rows; the NaN fixed
            # effects it drops itself
            ref = pf.feols("y ~ x | a + b", df.filter(pl.col("g").is_not_nan()).to_pandas(),
                           vcov={"CRV1": "g"})
        np.testing.assert_allclose(res.beta, ref.coef().to_numpy(), rtol=1e-8)
        np.testing.assert_allclose(res.se, ref.se().to_numpy(), rtol=1e-6)


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


# ------------------------------------------------------------- DataFrame source

def test_a_dataframe_source_gives_the_same_fit_as_a_path(path):
    frame = pl.read_parquet(path)
    with fit("y ~ x | w + f", path) as from_path, fit("y ~ x | w + f", frame) as from_frame:
        assert from_frame.n_obs == from_path.n_obs
        np.testing.assert_allclose(from_frame.beta, from_path.beta, rtol=1e-10)
        np.testing.assert_allclose(from_frame.se, from_path.se, rtol=1e-10)
        sample = from_frame.sample().collect()
        assert sample.height == frame.height
        assert sample["in_sample"].sum() == from_frame.n_obs
        assert set(from_frame.resid().collect()["row_id"]) == set(
            sample.filter(pl.col("in_sample"))["row_id"])


def test_a_dataframe_source_works_for_the_other_entry_points(path):
    frame = pl.read_parquet(path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with fepois_stream("cnt ~ x | w + f", frame, fe_dof=FE_DOF_PF, verbose=False) as pois:
            assert pois.n_obs > 0
    est = StreamingHDFE("y", ["x"], ["w", "f"], verbose=False)
    with est.fit(frame) as res:
        assert res.n_obs > 0


def test_a_dataframe_source_works_for_leave_out(tmp_path):
    from hdfe_stream import leave_one_out_connected, leave_out_kss
    from hdfe_stream.simulate import simulate_akm

    panel = simulate_akm(n_workers=300, n_firms=30, seed=13)
    pruned = leave_one_out_connected(panel)
    assert pruned.diagnostics["workers_before"] == 300
    kwargs = dict(n_draws=16, block=32, seed=4, verbose=False)
    fml = "log_earn ~ age_squared | worker_id + firm_id"
    path = tmp_path / "p.parquet"
    panel.write_parquet(path)
    from_frame = leave_out_kss(fml, panel, workdir=tmp_path / "a", **kwargs)
    from_path = leave_out_kss(fml, str(path), workdir=tmp_path / "b", **kwargs)
    assert from_frame.n_obs == from_path.n_obs
    for name, value in from_path.leave_out.items():
        assert from_frame.leave_out[name] == pytest.approx(value, rel=1e-8)

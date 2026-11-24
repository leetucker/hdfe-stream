"""core.distinct: approximate and exact distinct counts, in one pass or in
hash-partitioned passes."""

import numpy as np
import polars as pl
import pytest

from hdfe_stream.core.distinct import count_distinct, distinct_bytes
from hdfe_stream.core.profile import scan_source
from hdfe_stream.core.resources import MemoryBudget


@pytest.fixture(scope="module")
def frame():
    rng = np.random.default_rng(3)
    n = 200_000
    x = rng.standard_normal(n)
    x[:50] = 0.0
    x[50:100] = -0.0                      # the same value as 0.0
    x[100:110] = np.nan                   # one value
    g = rng.integers(0, 30_000, n)
    s = rng.choice(np.array(["a", "b", "c", None], dtype=object), n)
    return pl.DataFrame({"x": x, "g": g, "h": rng.integers(0, 7, n), "s": s})


def _truth(df):
    xs = df["x"].to_numpy()
    fin = xs[~np.isnan(xs)]
    return {
        "x": len(np.unique(fin + 0.0)) + 1,                   # every NaN is one value
        "g": df["g"].n_unique(),
        "gh": df.select(pl.struct("g", "h")).to_series().n_unique(),
        "s": df["s"].n_unique(),                              # null counts as a value
        "neg": len(np.unique(fin[fin < 0])),
    }


@pytest.mark.parametrize("partitions", [None, 1, 2, 7])
def test_exact_counts_agree_with_numpy(frame, partitions):
    truth = _truth(frame)
    got = count_distinct(frame.lazy(), {"x": "x", "g": "g", "gh": ["g", "h"], "s": "s",
                                        "neg": "x"},
                         exact=True, where={"neg": pl.col("x") < 0}, partitions=partitions)
    assert got == truth


def test_partitions_follow_the_budget(frame, monkeypatch):
    # a budget too small for one table splits the count into several passes, exactly
    from hdfe_stream.core import distinct as mod

    calls = []
    real = pl.LazyFrame.collect

    def spy(self, *a, **k):
        calls.append(1)
        return real(self, *a, **k)

    monkeypatch.setattr(pl.LazyFrame, "collect", spy)
    # pretend each value takes 1 MB, with 4 GB of headroom whatever this process holds
    monkeypatch.setattr(mod, "_BYTES_ONE", 1_000_000)
    monkeypatch.setattr(MemoryBudget, "headroom_gb", lambda self: 4.0)
    got = count_distinct(frame, {"g": "g"}, exact=True, max_memory_gb=4.0, memory_reserve=0.0)
    assert got["g"] == frame["g"].n_unique()
    assert len(calls) > 2                                  # the estimate, then several parts


def test_a_budget_too_small_for_the_count(frame, monkeypatch):
    from hdfe_stream.core import distinct as mod

    monkeypatch.setattr(mod, "_BYTES_ONE", 10**12)
    with pytest.raises(MemoryError, match="passes"):
        count_distinct(frame, {"g": "g"}, exact=True, max_memory_gb=4.0, memory_reserve=0.0)


def test_approximate_is_the_default_and_close(frame):
    got = count_distinct(frame, {"g": "g", "x": "x"})
    truth = _truth(frame)
    assert abs(got["g"] / truth["g"] - 1) < 0.05
    assert abs(got["x"] / truth["x"] - 1) < 0.05


def test_where_and_errors(frame):
    got = count_distinct(frame, {"h": "h"}, exact=True, where=pl.col("h") < 3)
    assert got == {"h": 3}
    assert count_distinct(frame, {}) == {}
    with pytest.raises(ValueError, match="not in the data"):
        count_distinct(frame, {"z": "zz"}, exact=True)
    with pytest.raises(ValueError, match="not counted"):
        count_distinct(frame, {"h": "h"}, exact=True, where={"q": pl.col("h") > 0})
    with pytest.raises(ValueError, match="at least 1"):
        count_distinct(frame, {"h": "h"}, exact=True, partitions=0)


def test_parquet_source(frame, tmp_path):
    path = tmp_path / "d.parquet"
    frame.write_parquet(path, row_group_size=20_000)
    assert count_distinct(path, {"g": "g"}, exact=True, partitions=3) == {
        "g": frame["g"].n_unique()}


def test_scan_source_exact(frame):
    lf = frame.lazy()
    approx = scan_source(lf, present=[pl.col("g")], distinct={"gh": ["g", "h"]})
    exact = scan_source(lf, present=[pl.col("g")], distinct={"gh": ["g", "h"]},
                        exact_distinct=True)
    truth = _truth(frame)["gh"]
    assert not approx.distinct_exact and exact.distinct_exact
    assert exact.distinct["gh"] == truth
    assert abs(approx.distinct["gh"] / truth - 1) < 0.05
    assert exact.to_dict()["distinct_exact"] is True


def test_distinct_bytes():
    assert distinct_bytes(1000) < distinct_bytes(1000, 2)
    assert distinct_bytes(0) == 0

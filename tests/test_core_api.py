"""The stable names of `hdfe_stream.core` (its docstring lists them), which
packages built on the layer import, and the block store."""

from __future__ import annotations

import importlib
import re

import numpy as np
import polars as pl
import pytest

import hdfe_stream.core
from hdfe_stream.core.store import (
    BlockStore,
    PolarsEngine,
    factorize,
    iter_batches,
    iter_groups,
    write_sample,
)


def _stable():
    doc = hdfe_stream.core.__doc__
    block = doc.split("Stable names", 1)[1].split("Everything else", 1)[0]
    out, module = [], None
    for line in block.splitlines():
        m = re.match(r"\s+core\.(\w+)\s+(.*)", line)
        if m:
            module, names = m.groups()
        elif module and line.strip() and line.startswith(" " * 23):
            names = line.strip()
        else:
            continue
        for n in re.findall(r"\b([A-Za-z]\w*)\b", names.split("(")[0]):
            out.append((module, n))
    return out


STABLE = _stable()


def test_the_list_is_read():
    assert ("formula", "compile_design") in STABLE and ("store", "iter_groups") in STABLE
    assert ("results_base", "vcov_key") in STABLE and len(STABLE) > 25


@pytest.mark.parametrize("module,name", STABLE, ids=[f"{m}.{n}" for m, n in STABLE])
def test_stable_name_exists_and_is_documented(module, name):
    obj = getattr(importlib.import_module(f"hdfe_stream.core.{module}"), name)
    if callable(obj) and module != "typing":
        assert obj.__doc__, f"core.{module}.{name} has no docstring"


@pytest.mark.parametrize("in_memory", [True, False])
def test_block_store_round_trip(tmp_path, in_memory):
    store = BlockStore(tmp_path / "blocks", in_memory, dtype=np.float32)
    rng = np.random.default_rng(0)
    blocks = [{"X": rng.normal(size=(n, 3)), "y": np.arange(n, dtype=np.int32)}
              for n in (5, 0, 7, 4)]
    for b in blocks:
        store.add({**b, "w": None})
    assert store.n_rows == 16 and len(store.blocks) == 3
    got = list(store)
    assert all(g["X"].dtype == np.float64 and g["X"].flags.c_contiguous for g in got)
    assert all("w" not in g for g in got)
    np.testing.assert_allclose(np.vstack([g["X"] for g in got]),
                               np.vstack([b["X"] for b in blocks]), rtol=1e-6)
    assert [list(g) for g in store.iterate(["y"])] == [["y"]] * 3
    store.delete()
    assert store.n_rows == 0 and (in_memory or not (tmp_path / "blocks").exists())


@pytest.mark.parametrize("in_memory", [True, False])
def test_block_store_binary_columns(tmp_path, in_memory):
    """Columns marked binary are stored as one byte each and X comes back
    whole, float64, its columns in their places."""
    rng = np.random.default_rng(1)
    mask = [False, True, False, True, True]
    store = BlockStore(tmp_path / "blocks", in_memory, binary=mask)
    blocks = []
    for n in (600, 900):
        X = rng.normal(size=(n, 5))
        X[:, mask] = rng.integers(0, 2, size=(n, 3))
        blocks.append(X)
        store.add({"X": X, "y": np.zeros(n, dtype=np.int32)})
    got = list(store)
    assert all(g["X"].dtype == np.float64 and g["X"].flags.c_contiguous for g in got)
    assert all("X~b" not in g for g in got)
    np.testing.assert_array_equal(np.vstack([g["X"] for g in got]), np.vstack(blocks))
    assert [sorted(g) for g in store.iterate(["X"])] == [["X"]] * 2
    raw = list(store.iterate(["X"], raw=True))
    assert [sorted(g) for g in raw] == [["X", "X~b"]] * 2
    assert raw[0]["X~b"].dtype == np.uint8 and raw[0]["X"].shape[1] == 2
    assert [list(g) for g in store.iterate(["y"])] == [["y"]] * 2
    plain = BlockStore(tmp_path / "plain", in_memory)
    for X in blocks:
        plain.add({"X": X})
    assert store.bytes < plain.bytes
    with pytest.raises(ValueError, match="columns"):
        store.add({"X": np.zeros((2, 4))})


def test_sample_and_factorize(tmp_path):
    eng = PolarsEngine()
    lf = pl.LazyFrame({"a": [3, 1, None, 3, 2], "b": ["x", "y", "z", "x", "y"],
                       "v": [1.0, float("nan"), 2.0, 3.0, 4.0]})
    write_sample(eng, lf, pl.col("a").is_not_null() & pl.col("v").is_not_nan(), ["a", "b"],
                 tmp_path / "s.parquet", rows=2, row_col="row")
    s = pl.concat(list(iter_batches(tmp_path / "s.parquet", 2)))
    assert s["row"].to_list() == [0, 3, 4] and s.columns == ["row", "a", "b"]
    t = factorize(eng, s.lazy(), ["a"], "a")
    assert t["a"].to_list() == [2, 3] and t["code"].to_list() == [0, 1]


@pytest.mark.parametrize("shuffled", [False, True])
def test_iter_groups_yields_whole_groups(tmp_path, shuffled):
    rng = np.random.default_rng(1)
    g = np.repeat(np.arange(300), rng.integers(1, 9, 300))
    df = pl.DataFrame({"g": g, "row": np.arange(len(g))})
    if shuffled:
        df = df.sample(fraction=1.0, shuffle=True, seed=2)
    df.write_parquet(tmp_path / "s.parquet", row_group_size=97)
    frames = list(iter_groups(PolarsEngine(), tmp_path / "s.parquet", ["g"], rows=50,
                              workdir=tmp_path, n_groups=300, sort_by=["g", "row"]))
    seen = pl.concat(frames)
    assert seen.height == len(g) and seen["g"].n_unique() == 300
    # every group in exactly one frame, and its rows consecutive there
    owner = {}
    for i, f in enumerate(frames):
        gs = f["g"].to_numpy()
        assert (np.diff(gs) != 0).sum() + 1 == len(np.unique(gs))
        for v in np.unique(gs):
            assert owner.setdefault(v, i) == i
    if shuffled:
        assert all(np.all(np.diff(f["row"].to_numpy()[f["g"].to_numpy() == v]) > 0)
                   for f in frames for v in np.unique(f["g"].to_numpy()))
    assert not (tmp_path / "group_buckets").exists()


def test_iter_groups_checks_nested_groupings(tmp_path):
    """Panels in consecutive rows but their sets interleaved: sorted, not
    taken in order."""
    df = pl.DataFrame({"p": [0, 0, 0, 0, 1, 1], "s": [0, 1, 0, 1, 2, 2],
                       "row": range(6)})
    df.write_parquet(tmp_path / "s.parquet")
    frames = list(iter_groups(PolarsEngine(), tmp_path / "s.parquet", ["p"], rows=10,
                              workdir=tmp_path, n_groups=2, sort_by=["p", "s", "row"],
                              also=[(["p", "s"], 3)]))
    out = pl.concat(frames).sort("p")
    assert out.filter(pl.col("p") == 0)["s"].to_list() == [0, 0, 1, 1]

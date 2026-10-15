"""`connected_components` and the estimator's `connected="largest"`.

The components are checked against scipy's connected_components on the same
graph, the pruning against a brute-force peel over Python sets, and the
estimator option against fitting on the set taken beforehand.
"""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl
import pytest
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components as scipy_components

from hdfe_stream import ConnectedComponents, connected_components, feols_stream


def _panel(seed=0, n=3000, n_w=900, n_f=120, islands=4):
    """A sparse worker-firm panel: a random graph that falls apart into a
    large component and some small ones, plus a few disjoint islands, so
    there are components of several sizes, some of them tied."""
    rng = np.random.default_rng(seed)
    w = rng.integers(0, n_w, n)
    f = rng.integers(0, n_f, n)
    ws, fs = [w], [f]
    for i in range(islands):
        size = 2 + i % 2                            # islands of two sizes: ties
        ws.append(10_000 + 10 * i + rng.integers(0, size, 3 * size))
        fs.append(1_000 + 10 * i + rng.integers(0, size, 3 * size))
    w, f = np.concatenate(ws), np.concatenate(fs)
    return pl.DataFrame({"w": w, "f": f, "yr": rng.integers(2000, 2004, len(w)),
                         "x": rng.normal(size=len(w))})


def _scipy_labels(a, b):
    """Component of each first-dimension level (by sorted value) and each
    second-dimension level, from scipy, plus the codes."""
    ua, ia = np.unique(a, return_inverse=True)
    ub, ib = np.unique(b, return_inverse=True)
    n0, n1 = len(ua), len(ub)
    g = sp.coo_matrix((np.ones(len(ia)), (ia, n0 + ib)), shape=(n0 + n1, n0 + n1))
    _, lab = scipy_components(g, directed=False)
    return ua, ub, lab[:n0], lab[n0:]


def _same_partition(x, y):
    """Whether two labelings of the same items define the same partition."""
    pairs = set(zip(x.tolist(), y.tolist()))
    return len(pairs) == len(set(x.tolist())) == len(set(y.tolist()))


def _quiet(**kw):
    return {"verbose": False, **kw}


def test_components_match_scipy():
    df = _panel()
    cc = connected_components(df, "w + f", **_quiet(batch_rows=500))
    ua, ub, la, lb = _scipy_labels(df["w"].to_numpy(), df["f"].to_numpy())
    lw = cc.levels["w"].sort("w")
    lf = cc.levels["f"].sort("f")
    assert np.array_equal(lw["w"].to_numpy(), ua)
    assert np.array_equal(lf["f"].to_numpy(), ub)
    ours = np.concatenate([lw["component"].to_numpy(), lf["component"].to_numpy()])
    assert _same_partition(ours, np.concatenate([la, lb]))
    assert cc.n_components == len(set(la.tolist()) | set(lb.tolist()))
    assert cc.n_components > 3


def test_sizes_ranking_and_tie_break():
    df = _panel()
    cc = connected_components(df, ("w", "f"), **_quiet())
    s = cc.sizes
    assert s["component"].to_list() == list(range(cc.n_components))
    rows = cc.label(df).group_by("component").len().sort("component").collect()
    assert rows["len"].to_list() == s["rows"].to_list()
    assert s["rows"].sum() == df.height == cc.diagnostics["rows"]
    for d in ("w", "f"):
        n = cc.levels[d].group_by("component").len().sort("component")["len"].to_list()
        assert n == s[d].to_list()
    # largest first; equal sizes ordered by their smallest worker
    smallest = (cc.levels["w"].group_by("component").agg(pl.col("w").min())
                  .sort("component"))
    key = list(zip((-s["rows"]).to_list(), smallest["w"].to_list()))
    assert key == sorted(key)
    assert len(set(s["rows"].to_list())) < cc.n_components     # there were ties


def test_rank_by_levels():
    df = pl.DataFrame({"w": [1, 1, 1, 1, 1, 2, 3, 4], "f": [1, 1, 1, 1, 2, 3, 3, 3]})
    by_rows = connected_components(df, "w + f", **_quiet())
    by_w = connected_components(df, "w + f", rank_by="w", **_quiet())
    assert by_rows.sizes["rows"].to_list() == [5, 3]
    assert by_w.sizes["rows"].to_list() == [3, 5]
    assert by_w.sizes["w"].to_list() == [3, 1]


def test_labels_do_not_depend_on_row_order():
    df = _panel(seed=3)
    a = connected_components(df, "w + f", **_quiet(batch_rows=400))
    b = connected_components(df.sample(fraction=1.0, shuffle=True, seed=9), "w + f",
                             **_quiet(batch_rows=700))
    for d in ("w", "f"):
        assert a.levels[d].sort(d).equals(b.levels[d].sort(d))
    assert a.sizes.equals(b.sizes)


def test_interaction():
    df = _panel(seed=4).with_columns(fy=pl.col("f") * 10_000 + pl.col("yr"))
    a = connected_components(df, "w + f^yr", **_quiet())
    b = connected_components(df, "w + fy", **_quiet())
    assert a.fe == ("w", "f^yr")
    assert a.levels["f^yr"].columns == ["f", "yr", "component"]
    assert a.sizes.rename({"f^yr": "fy"}).equals(b.sizes)
    assert a.levels["w"].equals(b.levels["w"])
    assert a.largest(df).collect().equals(b.largest(df).collect())


def test_missing_levels_are_in_no_component():
    # a null firm must not act as one giant firm joining its workers
    df = pl.DataFrame({"w": [1, 2, 3, 3, 4, 5],
                       "f": [None, None, 10.0, 11.0, float("nan"), 12.0]})
    cc = connected_components(df, "w + f", **_quiet())
    assert cc.diagnostics["rows"] == 3
    assert cc.levels["w"]["w"].to_list() == [3, 5]
    assert cc.sizes["rows"].to_list() == [2, 1]
    lab = cc.label(df).collect()
    assert lab["component"].to_list() == [None, None, 0, 0, None, 1]
    assert lab["in_largest"].to_list() == [False, False, True, True, False, False]
    assert cc.largest(df).collect()["w"].to_list() == [3, 3]


def test_largest_and_label_agree_and_keep_order():
    df = _panel(seed=5).with_row_index("i")
    cc = connected_components(df, "w + f", **_quiet())
    lab = cc.label(df).collect()
    assert lab["i"].to_list() == df["i"].to_list()
    big = cc.largest(df).collect()
    assert big.equals(lab.filter("in_largest").drop("component", "in_largest"))
    assert big.height == cc.sizes["rows"][0]


def test_sources(tmp_path):
    df = _panel(seed=6)
    path = tmp_path / "p.parquet"
    df.write_parquet(path)
    a = connected_components(df, "w + f", **_quiet())
    for src in (str(path), df.lazy()):
        b = connected_components(src, "w + f", **_quiet())
        assert a.sizes.equals(b.sizes)
        assert a.largest(src).collect().equals(a.largest(df).collect())


def _peel(pairs, k0, k1):
    """The (k0, k1)-core of a set of distinct pairs, by brute force."""
    pairs = set(pairs)
    while True:
        deg0, deg1 = {}, {}
        for a, b in pairs:
            deg0[a] = deg0.get(a, 0) + 1
            deg1[b] = deg1.get(b, 0) + 1
        keep = {(a, b) for a, b in pairs if deg0[a] >= k0 and deg1[b] >= k1}
        if keep == pairs:
            return pairs
        pairs = keep


@pytest.mark.parametrize("k", [(2, 1), (1, 2), 2, 3, (1, 5), (3, 2)])
def test_min_degree_matches_brute_force(k):
    df = _panel(seed=7, n=4000, n_w=700, n_f=150)
    cc = connected_components(df, "w + f", min_degree=k, **_quiet(batch_rows=300))
    k0, k1 = (k, k) if isinstance(k, int) else k
    core = _peel(zip(df["w"].to_list(), df["f"].to_list()), k0, k1)
    alive_w = {a for a, _ in core}
    alive_f = {b for _, b in core}
    lw = cc.levels["w"].filter(pl.col("component").is_not_null())
    lf = cc.levels["f"].filter(pl.col("component").is_not_null())
    assert set(lw["w"].to_list()) == alive_w
    assert set(lf["f"].to_list()) == alive_f
    assert cc.diagnostics["pruned_levels"]["w"] == df["w"].n_unique() - len(alive_w)
    # components of what is left, compared with scipy on the surviving rows
    kept = df.filter(pl.col("w").is_in(list(alive_w)) & pl.col("f").is_in(list(alive_f)))
    if kept.height:
        _, _, la, lb = _scipy_labels(kept["w"].to_numpy(), kept["f"].to_numpy())
        ours = np.concatenate([lw.sort("w")["component"].to_numpy(),
                               lf.sort("f")["component"].to_numpy()])
        assert _same_partition(ours, np.concatenate([la, lb]))
    # every row of a pruned level is in no component, and the counts add up
    lab = cc.label(df).collect()
    assert lab.filter(pl.col("component").is_not_null()).height == kept.height
    assert cc.sizes["rows"].sum() == kept.height
    assert cc.diagnostics["pruned_rows"] == df.height - kept.height
    assert cc.largest(df).collect().equals(lab.filter("in_largest").drop("component",
                                                                         "in_largest"))
    # the largest component is a fixed point: connected, and the core of itself
    big = cc.largest(df).collect()
    if big.height:
        again = connected_components(big, "w + f", min_degree=k, **_quiet())
        assert again.n_components == 1 and again.diagnostics["pruned_rows"] == 0


def test_movers_only():
    df = pl.DataFrame({"w": [1, 1, 2, 2, 3, 3, 4], "f": [1, 2, 2, 2, 2, 3, 4]})
    cc = connected_components(df, "w + f", min_degree=(2, 1), **_quiet())
    # worker 2 has two rows at one firm: one partner, so a stayer
    assert cc.levels["w"].filter(pl.col("component").is_not_null())["w"].to_list() == [1, 3]
    assert cc.largest(df).collect()["w"].to_list() == [1, 1, 3, 3]
    assert cc.diagnostics["distinct_pairs"] == 6


@pytest.mark.parametrize("k", [1, 2])
def test_join_fallback(monkeypatch, k):
    """Coding by join, used if two levels share a 64-bit hash or a row's
    hash is not found, gives the same answer as coding by hash."""
    import hdfe_stream.connected as connected

    df = _panel(seed=8).with_columns(pl.col("f").cast(pl.String))
    a = connected_components(df, "w + f^yr", min_degree=k, **_quiet(batch_rows=500))
    monkeypatch.setattr(connected, "_hash_index", lambda m, cols: None)
    b = connected_components(df, "w + f^yr", min_degree=k, **_quiet(batch_rows=500))
    monkeypatch.undo()
    calls = []

    def lookup_misses(*args):
        calls.append(1)
        return 1
    monkeypatch.setattr(connected, "_nb_hash_lookup", lookup_misses)
    c = connected_components(df, "w + f^yr", min_degree=k, **_quiet(batch_rows=500))
    assert calls
    for other in (b, c):
        assert a.sizes.equals(other.sizes)
        for d in a.fe:
            assert a.levels[d].equals(other.levels[d])


def test_empty():
    df = pl.DataFrame({"w": [None, 1], "f": [1, None]}, schema={"w": pl.Int64, "f": pl.Int64})
    for k in (1, 2):
        cc = connected_components(df, "w + f", min_degree=k, **_quiet())
        assert cc.n_components == 0 and cc.diagnostics["rows"] == 0
        assert cc.largest(df).collect().height == 0
        assert "components: 0" in cc.summary()


def test_errors():
    df = _panel()
    with pytest.raises(ValueError, match="exactly two"):
        connected_components(df, "w + f + yr", **_quiet())
    with pytest.raises(ValueError, match="slopes"):
        connected_components(df, "w[x] + f", **_quiet())
    with pytest.raises(ValueError, match="same"):
        connected_components(df, ("w", "w"), **_quiet())
    with pytest.raises(ValueError, match="only the fixed effects"):
        connected_components(df, "y ~ x | w + f", **_quiet())
    with pytest.raises(ValueError, match="min_degree"):
        connected_components(df, "w + f", min_degree=0, **_quiet())
    with pytest.raises(ValueError, match="min_degree"):
        connected_components(df, "w + f", min_degree=(2, 2, 2), **_quiet())
    with pytest.raises(ValueError, match="rank_by"):
        connected_components(df, "w + f", rank_by="yr", **_quiet())
    with pytest.raises(ValueError, match="not found"):
        connected_components(df, "w + g", **_quiet())
    cc = connected_components(df, "w + f", **_quiet())
    with pytest.raises(ValueError, match="label"):
        cc.label(df.with_columns(component=1))


def test_summary():
    cc = connected_components(_panel(), "w + f", min_degree=(2, 1), **_quiet())
    assert isinstance(cc, ConnectedComponents)
    text = cc.summary()
    assert "connected components of w x f" in text and "min_degree=(2, 1)" in text


# --------------------------------------------------------------------------
# the estimator's connected="largest"
# --------------------------------------------------------------------------

def _fit(fml, data, **kw):
    kw = {"verbose": False, "n_buckets": 2, "batch_rows": 700, **kw}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return feols_stream(fml, data, **kw)


@pytest.fixture(scope="module")
def outcome_panel():
    df = _panel(seed=11, n=5000, n_w=800, n_f=80)
    rng = np.random.default_rng(2)
    y = (df["x"].to_numpy() + 0.1 * (df["w"].to_numpy() % 7)
         + 0.3 * (df["f"].to_numpy() % 5) + rng.normal(size=df.height))
    y[[3, 50]] = np.nan                             # missing outcomes
    return df.with_columns(y=pl.Series(y))


@pytest.mark.parametrize("fe", ["w + f", "w + f + yr", "f + w"])
def test_fit_matches_set_taken_beforehand(outcome_panel, fe):
    df = outcome_panel
    a = _fit(f"y ~ x | {fe}", df, connected="largest")
    complete = df.filter(pl.col("y").is_not_nan())
    first_two = " + ".join(fe.split(" + ")[:2])
    b = _fit(f"y ~ x | {fe}", connected_components(complete, first_two, verbose=False)
             .largest(complete).collect())
    assert a.n_obs == b.n_obs
    np.testing.assert_allclose(a.beta, b.beta, rtol=1e-10)
    np.testing.assert_allclose(a.se, b.se, rtol=1e-8)
    assert a.n_components == 1
    conn = a.diagnostics["connected"]
    assert conn["fe"] == first_two.split(" + ") and conn["components"] > 1
    assert "largest connected set" in a.summary_text()


def test_fit_sample_reports_disconnected(outcome_panel):
    df = outcome_panel
    res = _fit("y ~ x | w + f", df, connected="largest")
    s = res.sample().collect()
    counts = dict(s.group_by("dropped_because").len().iter_rows())
    assert counts.get("missing") == 2
    assert counts.get("disconnected") == res.diagnostics["connected"]["observations"]
    assert counts.get(None) == res.n_obs
    # the rows called disconnected are exactly those outside the largest set
    complete = df.filter(pl.col("y").is_not_nan())
    lab = connected_components(complete, "w + f", verbose=False).label(df).collect()
    out = s["dropped_because"] == "disconnected"
    usable = df["y"].is_not_nan()
    assert (out == (~lab["in_largest"] & usable)).all()


def test_fit_without_connected_keeps_every_component(outcome_panel):
    res = _fit("y ~ x | w + f", outcome_panel)
    assert res.n_components > 1 and res.diagnostics["connected"] == {}


def test_fit_connected_needs_two_fixed_effects(outcome_panel):
    with pytest.raises(ValueError, match="two fixed effects"):
        _fit("y ~ x | w", outcome_panel, connected="largest")
    with pytest.raises(ValueError, match="connected must be"):
        _fit("y ~ x | w + f", outcome_panel, connected="all")


def test_fit_glm_connected(outcome_panel):
    from hdfe_stream import fepois_stream
    rng = np.random.default_rng(5)
    df = outcome_panel.with_columns(cnt=pl.Series(rng.poisson(2.0, outcome_panel.height)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = fepois_stream("cnt ~ x | w + f", df, connected="largest", verbose=False,
                          n_buckets=2)
        sub = connected_components(df, "w + f", verbose=False).largest(df).collect()
        b = fepois_stream("cnt ~ x | w + f", sub, verbose=False, n_buckets=2)
    assert a.n_obs == b.n_obs
    np.testing.assert_allclose(a.beta, b.beta, rtol=1e-8)


def test_fixef_component_ranked_like_connected_components(outcome_panel):
    """The fit's `component` column on the firm effects is numbered by rows,
    largest first, as connected_components numbers the same rows with the
    firms as first dimension (whose smallest level breaks ties)."""
    df = outcome_panel
    res = _fit("y ~ x | w + f", df, fixef_rm="none")
    assert res.diagnostics["stream"]["dim"] == "w"
    fx = res.fixef("f").select("f", "component").sort("f").collect()
    used = res.sample().filter("in_sample").collect()
    cc = connected_components(used, ("f", "w"), verbose=False)
    assert res.n_components == cc.n_components > 1
    assert fx["component"].to_list() == cc.levels["f"].sort("f")["component"].to_list()
    rows = (used.join(fx, on="f").group_by("component").len().sort("component"))
    assert rows["len"].to_list() == cc.sizes["rows"].to_list()


@pytest.mark.parametrize("k,conn", [((2, 1), None), ((2, 1), "largest"), (2, "largest")])
def test_fit_min_degree_matches_set_taken_beforehand(outcome_panel, k, conn):
    df = outcome_panel
    a = _fit("y ~ x | w + f + yr", df, min_degree=k, connected=conn)
    complete = df.filter(pl.col("y").is_not_nan())
    cc = connected_components(complete, "w + f", min_degree=k, verbose=False)
    sub = (cc.largest(complete) if conn else
           cc.label(complete).filter(pl.col("component").is_not_null())
             .drop("component", "in_largest"))
    b = _fit("y ~ x | w + f + yr", sub.collect())
    assert a.n_obs == b.n_obs
    np.testing.assert_allclose(a.beta, b.beta, rtol=1e-10)
    np.testing.assert_allclose(a.se, b.se, rtol=1e-8)
    pruned = a.diagnostics["connected"]["min_degree"]["observations"]
    assert pruned == cc.diagnostics["pruned_rows"] > 0
    counts = dict(a.sample().collect().group_by("dropped_because").len().iter_rows())
    assert counts.get("min_degree") == pruned
    assert counts.get("disconnected", 0) == a.diagnostics["connected"].get("observations", 0)
    assert counts.get(None) == a.n_obs
    assert "dropped by min_degree" in a.summary_text()


def test_fit_min_degree_needs_two_fixed_effects(outcome_panel):
    with pytest.raises(ValueError, match="min_degree needs"):
        _fit("y ~ x | w", outcome_panel, min_degree=2)
    with pytest.raises(ValueError, match="min_degree must"):
        _fit("y ~ x | w + f", outcome_panel, min_degree=0)


@pytest.mark.parametrize("k", [1, (2, 1)])
def test_fit_leave_one_out_matches_set_taken_beforehand(outcome_panel, k):
    df = outcome_panel
    a = _fit("y ~ x | w + f", df, connected="leave_one_out", min_degree=k)
    complete = df.filter(pl.col("y").is_not_nan())
    cc = connected_components(complete, "w + f", leave_one_out=True, min_degree=k,
                              verbose=False)
    b = _fit("y ~ x | w + f", cc.largest(complete).collect())
    assert a.n_obs == b.n_obs == cc.sizes["rows"][0]
    np.testing.assert_allclose(a.beta, b.beta, rtol=1e-10)
    lo = a.diagnostics["connected"]["leave_one_out"]
    assert lo["observations"] == cc.diagnostics["leave_one_out"]["rows"] > 0
    assert lo["levels_after"]["w"] == cc.sizes["w"][0]
    counts = dict(a.sample().collect().group_by("dropped_because").len().iter_rows())
    assert counts.get("leave_one_out") == lo["observations"]
    assert counts.get("min_degree", 0) == cc.diagnostics.get("pruned_rows", 0)
    assert "leave-one-out connected set" in a.summary_text()


def test_fit_connected_fe(outcome_panel):
    """connected_fe picks the pair: here worker x firm although year is
    listed second."""
    df = outcome_panel
    a = _fit("y ~ x | w + yr + f", df, connected="largest", connected_fe="w + f")
    b = _fit("y ~ x | w + f + yr", df, connected="largest")
    assert a.diagnostics["connected"]["fe"] == ["w", "f"]
    assert a.n_obs == b.n_obs
    np.testing.assert_allclose(a.beta, b.beta, rtol=1e-10)
    with pytest.raises(ValueError, match="not fixed effects"):
        _fit("y ~ x | w + f", df, connected="largest", connected_fe=("w", "yr"))


def test_largest_and_label_default_to_the_source(tmp_path):
    """Without data, largest() and label() use the data the components were
    found in, with all of its columns."""
    import dataclasses

    df = _panel(seed=12)
    path = tmp_path / "p.parquet"
    df.write_parquet(path)
    for src in (df, df.lazy(), str(path)):
        cc = connected_components(src, "w + f", **_quiet())
        assert cc.largest().collect().equals(cc.largest(df).collect())
        assert cc.label().collect().equals(cc.label(df).collect())
        assert cc.largest().collect_schema().names() == df.columns
    with pytest.raises(ValueError, match="pass the data"):
        dataclasses.replace(cc, _source=None).largest()

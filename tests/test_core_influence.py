"""hdfe_stream.core.influence (first written in match-stream):
the weights-fixed weighted mean difference and its clustered variance,
against hdfe-stream's weighted least squares (itself checked against
pyfixest/fixest) and a direct numpy computation."""

import numpy as np
import polars as pl
import pytest

from hdfe_stream import feols_stream
from hdfe_stream.core.influence import cluster_dims, if_vcov, weighted_mean_difference


@pytest.fixture(scope="module")
def df():
    rng = np.random.default_rng(7)
    n = 3000
    a = rng.integers(0, 40, n)
    b = rng.integers(0, 25, n)
    d = rng.random(n) < 0.3
    y = 1 + 0.5 * d + 0.3 * rng.standard_normal(40)[a] + 0.2 * rng.standard_normal(25)[b] \
        + rng.standard_normal(n)
    w = rng.exponential(1.0, n)
    return pl.DataFrame({"y": y, "d": d.astype(np.int64), "w": w, "a": a, "b": b})


@pytest.mark.parametrize("vcov,cluster", [("hetero", None), ({"CRV1": "a"}, "a"),
                                          ({"CRV1": "a+b"}, "a+b")])
def test_equals_weighted_least_squares(df, tmp_path, vcov, cluster):
    fit = feols_stream("y ~ d", df, workdir=tmp_path, weights="w", vcov=vcov)
    r = weighted_mean_difference(df, "y", "w", "d", cluster)
    i = fit.coefnames.index("d")
    np.testing.assert_allclose(r.estimate, fit.beta[i], rtol=1e-10)
    np.testing.assert_allclose(r.variance.vcov[0, 0], fit.vcov[i, i], rtol=1e-8)


def test_matches_a_direct_computation(df):
    y, d, w, a = (df[c].to_numpy() for c in ("y", "d", "w", "a"))
    d = d.astype(bool)
    m1, m0 = np.average(y[d], weights=w[d]), np.average(y[~d], weights=w[~d])
    phi = np.where(d, w * (y - m1) / w[d].sum(), -w * (y - m0) / w[~d].sum())
    s = np.bincount(a, weights=phi)
    G = len(s)
    r = weighted_mean_difference(df, "y", "w", "d", "a", df_k=None)
    np.testing.assert_allclose(r.estimate, m1 - m0, rtol=1e-12)
    np.testing.assert_allclose(r.variance.vcov[0, 0], G / (G - 1) * np.sum(s**2), rtol=1e-10)
    assert r.variance.n_clusters == {"a": G} and r.variance.df == G - 1


def test_two_way_is_inclusion_exclusion(df):
    d = df.with_columns(pl.col("y").alias("p"))
    va = if_vcov(d, ["p"], "a", df_k=None).vcov / (40 / 39)
    vb = if_vcov(d, ["p"], "b", df_k=None).vcov / (25 / 24)
    ab = d.with_columns((pl.col("a") * 100 + pl.col("b")).alias("ab"))
    vab = if_vcov(ab, ["p"], "ab", df_k=None)
    vab = vab.vcov / vab.adj
    two = if_vcov(d, ["p"], ["a", "b"], df_k=None)
    np.testing.assert_allclose(two.vcov, (25 / 24) * (va + vb - vab), rtol=1e-10)  # G_min = 25


def test_several_columns_and_no_clustering(df):
    d = df.with_columns((pl.col("y") * 2).alias("q"))
    v = if_vcov(d, ["y", "q"])
    Y = df["y"].to_numpy()
    n = len(Y)
    want = n / (n - 1) * np.array([[1, 2], [2, 4]]) * np.sum(Y**2)
    np.testing.assert_allclose(v.vcov, want, rtol=1e-10)


def test_checks(df):
    assert cluster_dims(" a + b ") == ["a", "b"] and cluster_dims(None) == []
    with pytest.raises(ValueError, match="repeated"):
        cluster_dims("a+a")
    with pytest.raises(ValueError, match="missing values"):
        if_vcov(df.with_columns(pl.lit(None).alias("c")), ["y"], "c")
    neg = df.with_columns(pl.when(pl.col("a") == 0).then(-0.1).otherwise(pl.col("w")).alias("w"))
    with pytest.raises(ValueError, match="negative weight"):
        weighted_mean_difference(neg, "y", "w", "d")
    r = weighted_mean_difference(neg, "y", "w", "d", allow_negative=True)   # ridge weights
    yv, dv, wv = (neg[c].to_numpy() for c in ("y", "d", "w"))
    dv = dv.astype(bool)
    want = (wv[dv] * yv[dv]).sum() / wv[dv].sum() - (wv[~dv] * yv[~dv]).sum() / wv[~dv].sum()
    np.testing.assert_allclose(r.estimate, want, rtol=1e-12)
    with pytest.raises(ValueError, match="positive total weight"):
        weighted_mean_difference(df.with_columns(pl.lit(1).alias("d")), "y", "w", "d")


def test_group_mean_phis(df):
    from hdfe_stream.core.influence import group_mean_phis
    wmd = weighted_mean_difference(df, "y", "w", "d", "a", df_k=None)
    phis = group_mean_phis(df, "y", "w", "d", wmd)
    assert phis.height == df.height
    # each column sums to zero (the means are at their estimates), and the
    # difference's contributions give back the difference's variance
    np.testing.assert_allclose(phis.sum().row(0), 0, atol=1e-12)
    t = df.with_columns(phis).with_columns((pl.col("phi1") - pl.col("phi0")).alias("phi"))
    np.testing.assert_allclose(if_vcov(t, ["phi"], "a").vcov, wmd.variance.vcov, rtol=1e-12)


@pytest.mark.parametrize("cluster", [None, "a", "a+b"])
def test_joint_if_vcov(df, cluster):
    """Two estimators on two tables (the even and the odd rows, the second
    with a column of its own) whose clusters overlap: the diagonal is each
    estimator's own variance, the covariance the stacked per-cluster sums,
    scaled by the root of the two small-sample factors."""
    from hdfe_stream.core.influence import joint_if_vcov
    rows = df.with_row_index("r").with_columns(
        (pl.col("w") * (pl.col("y") - (pl.col("w") * pl.col("y")).sum() / pl.col("w").sum())
         / pl.col("w").sum()).alias("phi"))
    if cluster is None:
        rows = rows.with_columns(pl.col("r").alias("__unit"))
    t0 = rows.filter(pl.col("r") % 2 == 0)
    t1 = rows.filter(pl.col("r") % 2 == 1).with_columns((2 * pl.col("phi")).alias("phi_b"))
    dims = ["__unit"] if cluster is None else cluster
    own = [if_vcov(t0, ["phi"], dims, df_k=2), if_vcov(t1, ["phi_b"], dims, df_k=2)]
    factors = np.array([v.adj for v in own])
    got = joint_if_vcov([(t0, {0: "phi"}), (t1, {1: "phi_b"})], 2, dims, factors)
    np.testing.assert_allclose(np.diag(got.vcov), [v.vcov[0, 0] for v in own], rtol=1e-10)
    stack = pl.concat([t0.select(*cluster_dims(dims), pl.col("phi").alias("p0"),
                                 pl.lit(0.0).alias("p1")),
                       t1.select(*cluster_dims(dims), pl.lit(0.0).alias("p0"),
                                 pl.col("phi_b").alias("p1"))])
    raw = if_vcov(stack, ["p0", "p1"], dims, df_k=None)
    want = raw.vcov[0, 1] / raw.adj * np.sqrt(factors.prod())
    np.testing.assert_allclose(got.vcov[0, 1], want, rtol=1e-10)
    assert got.adj == 1.0 and got.vcov[0, 1] == got.vcov[1, 0]


@pytest.fixture(scope="module")
def scored(df):
    """`df` with two influence columns and the rows in cluster (a) order."""
    rng = np.random.default_rng(3)
    return (df.with_columns(pl.Series("phi0", rng.standard_normal(df.height) / df.height),
                            pl.Series("phi1", rng.standard_normal(df.height) / df.height))
            .sort("a", maintain_order=True))


@pytest.mark.parametrize("df_k", [None, 2])
def test_accumulator_equals_if_vcov_from_whole_group_batches(scored, tmp_path, df_k):
    from hdfe_stream.core.influence import ClusterScoreAccumulator
    from hdfe_stream.core.store import PolarsEngine, iter_groups
    path = tmp_path / "rows.parquet"
    scored.sample(fraction=1.0, shuffle=True, seed=1).write_parquet(path)   # groups mixed
    acc = ClusterScoreAccumulator("a")
    n_frames = 0
    for f in iter_groups(PolarsEngine(), path, ["a"], 300, tmp_path, scored["a"].n_unique()):
        # two blocks: their cross entries come with the variance
        acc.add_rows(f["a"], f["phi0"].to_numpy(), f.select("phi1").to_numpy())
        n_frames += 1
    assert n_frames > 1
    want = if_vcov(scored, ["phi0", "phi1"], "a", df_k=df_k)
    got = acc.variance(df_k=df_k)
    np.testing.assert_allclose(got.vcov, want.vcov, rtol=1e-12)
    assert (got.n_obs, got.n_clusters, got.df) == (want.n_obs, want.n_clusters, want.df)
    np.testing.assert_allclose(acc.sums * got.adj, got.vcov, rtol=1e-15)


def test_accumulator_from_cluster_sums_and_without_clustering(scored):
    from hdfe_stream.core.influence import ClusterScoreAccumulator
    sums = scored.group_by("a").agg(pl.col("phi0", "phi1").sum())
    acc = ClusterScoreAccumulator("a")
    for part in (sums[:15], sums[15:]):
        acc.add_clusters(part.select("phi0", "phi1").to_numpy(), n_obs=0)
    acc.n_obs = scored.height
    np.testing.assert_allclose(acc.variance(df_k=2).vcov,
                               if_vcov(scored, ["phi0", "phi1"], "a", df_k=2).vcov, rtol=1e-12)
    # each row its own cluster
    row = ClusterScoreAccumulator()
    for part in (scored[:1000], scored[1000:]):
        row.add_clusters(part.select("phi0", "phi1").to_numpy())
    want = if_vcov(scored, ["phi0", "phi1"], None, df_k=2)
    got = row.variance(df_k=2)
    np.testing.assert_allclose(got.vcov, want.vcov, rtol=1e-12)
    assert got.n_clusters == want.n_clusters == {}


def test_accumulator_checks(scored):
    from hdfe_stream.core.influence import ClusterScoreAccumulator
    acc = ClusterScoreAccumulator()
    with pytest.raises(ValueError, match="no scores"):
        acc.variance()
    with pytest.raises(ValueError, match="not consecutive"):
        acc.add_rows(np.array([1, 1, 2, 1]), np.zeros(4))
    acc = ClusterScoreAccumulator()
    acc.add_rows(np.array([1, 2, 2]), np.zeros(3))
    with pytest.raises(ValueError, match="continues from the previous batch"):
        acc.add_rows(np.array([2, 3]), np.zeros(2))
    with pytest.raises(ValueError, match="total width"):
        acc.add_rows(np.array([4]), np.zeros((1, 2)))
    with pytest.raises(ValueError, match="numbers of rows"):
        acc.add_rows(np.array([5, 6]), np.zeros(3))
    with pytest.raises(ValueError, match="missing"):
        acc.add_rows(pl.Series([7, None]), np.zeros(2))

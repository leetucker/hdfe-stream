"""core.crossprod: within cross-products split by an integer column."""

import numpy as np
import polars as pl
import pytest

from hdfe_stream import feols_stream
from hdfe_stream.core.crossprod import CrossProducts, cross_products


def _panel(n=3000, seed=0, missing=False):
    rng = np.random.default_rng(seed)
    worker = rng.integers(0, 150, n)
    firm = rng.integers(0, 25, n)
    year = rng.integers(0, 6, n)
    x1 = rng.standard_normal(n) + 0.3 * firm
    x2 = rng.standard_normal(n) + 0.1 * worker % 3
    y = 1.0 + 0.5 * x1 - 0.25 * x2 + 0.01 * worker + 0.05 * firm + rng.standard_normal(n)
    w = rng.uniform(0.5, 2.0, n)
    fold = rng.integers(0, 4, n)
    df = pl.DataFrame({"worker": worker, "firm": firm, "year": year, "x1": x1, "x2": x2,
                       "y": y, "w": w, "fold": fold})
    if missing:
        df = df.with_columns(pl.when(pl.int_range(pl.len()) % 97 == 5).then(None)
                             .otherwise(pl.col("x2")).alias("x2"))
    return df


def _dummies(codes):
    levels, inv = np.unique(codes, return_inverse=True)
    D = np.zeros((len(codes), len(levels)))
    D[np.arange(len(codes)), inv] = 1.0
    return D


def _residualize(V, fes, w):
    """V less its weighted projection on the fixed-effect dummies."""
    D = np.column_stack([_dummies(f) for f in fes])
    sw = np.sqrt(w)[:, None]
    coef = np.linalg.lstsq(D * sw, V * sw, rcond=None)[0]
    return V - D @ coef


@pytest.mark.parametrize("weighted", [False, True])
def test_no_fixed_effects(weighted, tmp_path):
    df = _panel()
    cp = cross_products(df, ["y", "x1", "x2"], weights="w" if weighted else None,
                        split_by="fold", n_splits=4, workdir=tmp_path, verbose=False)
    assert isinstance(cp, CrossProducts) and cp.names == ["y", "x1", "x2"]
    V = df.select("y", "x1", "x2").to_numpy()
    w = df["w"].to_numpy() if weighted else np.ones(len(df))
    s = df["fold"].to_numpy()
    for j in range(4):
        r = s == j
        np.testing.assert_allclose(cp.gram[j], (V[r] * w[r, None]).T @ V[r], rtol=1e-12)
        np.testing.assert_allclose(cp.sums[j], w[r] @ V[r], rtol=1e-12)
        assert cp.counts[j] == r.sum()
        np.testing.assert_allclose(cp.weight_sums[j], w[r].sum(), rtol=1e-12)
    np.testing.assert_allclose(cp.total(), (V * w[:, None]).T @ V, rtol=1e-12)
    np.testing.assert_allclose(cp.excluding(1), cp.gram[[0, 2, 3]].sum(axis=0))
    assert cp.fe_names == [] and cp.k_fe == 0 and cp.n_obs == len(df)


@pytest.mark.parametrize("weighted", [False, True])
def test_two_fixed_effects_match_dummy_projection(weighted, tmp_path):
    df = _panel()
    cp = cross_products(df, ["y", "x1", "x2"], fe=["worker", "firm"],
                        weights="w" if weighted else None, split_by="fold", n_splits=4,
                        fixef_rm="none", workdir=tmp_path, verbose=False, tol=1e-13)
    V = df.select("y", "x1", "x2").to_numpy()
    w = df["w"].to_numpy() if weighted else np.ones(len(df))
    R = _residualize(V, [df["worker"].to_numpy(), df["firm"].to_numpy()], w)
    s = df["fold"].to_numpy()
    scale = np.abs((R * w[:, None]).T @ R).max()
    for j in range(4):
        r = s == j
        np.testing.assert_allclose(cp.gram[j], (R[r] * w[r, None]).T @ R[r],
                                   atol=1e-8 * scale)
        np.testing.assert_allclose(cp.sums[j], w[r] @ R[r], atol=1e-8 * scale)
    np.testing.assert_allclose(cp.sums.sum(axis=0), 0.0, atol=1e-8 * scale)
    assert cp.fe_names == ["worker", "firm"] and cp.n_obs == len(df)


def test_coefficients_equal_feols(tmp_path):
    """Frisch-Waugh-Lovell: the coefficients solved from the total equal a fit's."""
    df = _panel(seed=3)
    cp = cross_products(df, ["y", "x1", "x2"], fe=["worker", "firm", "year"],
                        weights="w", workdir=tmp_path, verbose=False, tol=1e-13)
    G = cp.total()
    beta = np.linalg.solve(G[1:, 1:], G[1:, 0])
    fit = feols_stream("y ~ x1 + x2 | worker + firm + year", df, weights="w",
                       workdir=tmp_path, verbose=False, tol=1e-13)
    coef = fit.coef()
    np.testing.assert_allclose(beta, [coef["x1"], coef["x2"]], rtol=1e-8)
    assert cp.n_obs == fit.n_obs and cp.k_fe == fit.k_fe


def test_sample_and_dropped_rows(tmp_path):
    df = _panel(missing=True)
    cp = cross_products(df, ["y", "x1", "x2"], fe=["worker"], split_by="fold",
                        n_splits=4, workdir=tmp_path, verbose=False)
    sample = cp.sample().collect()
    assert sample["in_sample"].sum() == cp.n_obs == cp.counts.sum()
    assert (sample.filter(pl.col("x2").is_null())["dropped_because"] == "missing").all()


def test_one_fixed_effect_group_folds_are_within_group(tmp_path):
    """Folds made of whole groups: each fold's sums are its own groups' demeaned rows."""
    df = _panel().with_columns((pl.col("worker") % 3).alias("fold"))
    cp = cross_products(df, ["y", "x1"], fe=["worker"], split_by="fold", n_splits=3,
                        fixef_rm="none", workdir=tmp_path, verbose=False)
    for j in range(3):
        part = df.filter(pl.col("fold") == j)
        R = _residualize(part.select("y", "x1").to_numpy(), [part["worker"].to_numpy()],
                         np.ones(len(part)))
        np.testing.assert_allclose(cp.gram[j], R.T @ R, rtol=1e-9)


def test_bad_splits(tmp_path):
    df = _panel()
    with pytest.raises(ValueError, match="outside"):
        cross_products(df, ["y", "x1"], split_by="fold", n_splits=3, workdir=tmp_path,
                       verbose=False)
    with pytest.raises(ValueError, match="needs n_splits"):
        cross_products(df, ["y", "x1"], split_by="fold", workdir=tmp_path, verbose=False)
    with pytest.raises(ValueError, match="integer"):
        cross_products(df, ["y", "x1"], split_by="w", n_splits=3, workdir=tmp_path,
                       verbose=False)
    with pytest.raises(TypeError, match="sets 'keep'"):
        cross_products(df, ["y", "x1"], keep=["w"], workdir=tmp_path, verbose=False)


def test_saved_rows_are_the_residualized_rows(tmp_path):
    df = _panel()
    cp = cross_products(df, ["y", "x1", "x2"], fe=["worker", "firm"], weights="w",
                        split_by="fold", n_splits=4, save_rows=True, carry=["firm"],
                        fixef_rm="none", workdir=tmp_path, verbose=False, tol=1e-13)
    rows = cp.rows().collect().sort(cp.row_id)
    assert rows.columns == ["row_id", "firm", "fold", cp.weight_column, "y", "x1", "x2"]
    V = df.select("y", "x1", "x2").to_numpy()
    w = df["w"].to_numpy()
    R = _residualize(V, [df["worker"].to_numpy(), df["firm"].to_numpy()], w)
    np.testing.assert_allclose(rows.select("y", "x1", "x2").to_numpy(), R,
                               atol=1e-8 * np.abs(R).max())
    assert (rows["firm"].to_numpy() == df["firm"].to_numpy()).all()
    np.testing.assert_allclose(rows[cp.weight_column].to_numpy(), w)
    Rw = rows.select("y", "x1", "x2").to_numpy()
    np.testing.assert_allclose((Rw * w[:, None]).T @ Rw, cp.total(), rtol=1e-10)
    cp.cleanup()
    with pytest.raises(FileNotFoundError):
        cp.rows()


def test_rows_not_saved_by_default(tmp_path):
    cp = cross_products(_panel(), ["y", "x1"], workdir=tmp_path, verbose=False)
    with pytest.raises(ValueError, match="save_rows"):
        cp.rows()
    assert not any(tmp_path.glob("hdfe_run_*/models"))
    with pytest.raises(ValueError, match="carry needs"):
        cross_products(_panel(), ["y", "x1"], carry=["firm"], workdir=tmp_path,
                       verbose=False)

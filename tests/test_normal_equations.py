"""HDFEResult.solve_normal_equations against explicit dummies."""

import numpy as np
import polars as pl
import pytest
from numpy.testing import assert_allclose

from hdfe_stream import NormalEquationsSolution, feols_stream


def panel(seed=0, n=400):
    rng = np.random.default_rng(seed)
    d = pl.DataFrame({
        "worker": rng.integers(0, 60, n),
        "firm": rng.integers(0, 15, n),
        "year": rng.integers(0, 6, n),
        "x": rng.normal(size=n),
        "z": rng.normal(size=n),
        "w": rng.uniform(0.5, 2, n),
    })
    return d.with_columns(
        (pl.col("x") + 0.1 * pl.col("worker") + 0.3 * pl.col("firm") + rng.normal(size=n))
        .alias("y"))


def design(d, fes, covariates):
    blocks = [d.select(covariates).to_numpy()]
    for f in fes:
        blocks.append((d[f].to_numpy()[:, None] == np.unique(d[f].to_numpy())).astype(float))
    return np.hstack(blocks)


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("fes", [["worker", "year"], ["worker", "firm", "year"]])
def test_matches_dense(weighted, fes):
    d = panel()
    fit, target = d[:300], d[300:]
    rng = np.random.default_rng(1)
    # keep only target rows whose levels are in the estimation sample
    for f in fes:
        target = target.filter(pl.col(f).is_in(fit[f].unique().to_list()))
    target = target.with_columns(pl.Series("v0", rng.normal(size=target.height)),
                                 pl.Series("v1", rng.uniform(size=target.height)))
    options = {"weights": "w"} if weighted else {}
    r = feols_stream("y ~ x + z | " + " + ".join(fes), fit, keep_intermediates=True,
                     fixef_rm="none", verbose=False, **options)
    solution = r.solve_normal_equations(target, ["v0", "v1"], block=1)
    assert isinstance(solution, NormalEquationsSolution)
    A = design(fit, fes, ["x", "z"])
    # the target's columns in the fit's level order
    T = np.hstack([target.select("x", "z").to_numpy()] + [
        (target[f].to_numpy()[:, None] == np.unique(fit[f].to_numpy())).astype(float)
        for f in fes])
    W = fit["w"].to_numpy() if weighted else np.ones(fit.height)
    S = A.T @ (W[:, None] * A)
    V = target.select("v0", "v1").to_numpy()
    U = np.linalg.pinv(S, rcond=1e-10) @ (T.T @ V)
    # identified: predictions on the estimation rows and on the target rows
    for rows, Z in [(fit, A), (target.drop("v0", "v1"), T)]:
        got = solution.predict(rows).select("v0", "v1").collect().to_numpy()
        assert_allclose(got, Z @ U, atol=1e-7)
    # the covariate block is identified on its own
    assert_allclose(solution.coef.select("v0", "v1").to_numpy(), U[:2], atol=1e-7)
    assert solution.fixef(fes[0]).collect().height == fit[fes[0]].n_unique()
    solution.cleanup()
    with pytest.raises(FileNotFoundError):
        solution.fixef(fes[0])


def test_requirements():
    d = panel()
    r = feols_stream("y ~ x | worker + year", d, verbose=False)
    with pytest.raises(RuntimeError, match="keep_intermediates"):
        r.solve_normal_equations(d.with_columns(v=1.0), "v")
    r = feols_stream("y ~ x | worker + year", d[:300], keep_intermediates=True,
                     fixef_rm="none", verbose=False)
    new = pl.DataFrame({"worker": [10_000], "year": [0], "x": [0.0], "v": [1.0]})
    with pytest.raises(ValueError, match="not in the estimation sample"):
        r.solve_normal_equations(new, "v")
    with pytest.raises(ValueError, match="lacks"):
        r.solve_normal_equations(d, "v")

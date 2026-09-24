"""The simulators themselves.

Every other test trusts these to produce a panel with identified fixed
effects, so it is worth checking the structure directly: the right columns,
reproducibility from the seed, a connected worker-firm graph, and effects the
estimator can actually recover.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from hdfe_stream.simulate import simulate_akm, simulate_rich, simulate_trends

BASE_COLUMNS = {"worker_id", "firm_id", "year", "age", "age_squared", "age_cubed",
                "log_earn"}


def test_base_panel_has_the_documented_columns():
    df = simulate_akm(n_workers=500)
    assert set(df.columns) == BASE_COLUMNS
    assert df["worker_id"].dtype == pl.Int64
    assert df["firm_id"].dtype == pl.Utf8


def test_panel_is_unbalanced_and_shuffled():
    """Rows are not sorted on disk, and not every worker is seen every year --
    both of which the streaming path has to cope with."""
    df = simulate_akm(n_workers=500, seed=1)
    per_worker = df.group_by("worker_id").len()["len"]
    assert per_worker.min() < per_worker.max()
    assert not df["worker_id"].is_sorted()


def test_seed_makes_the_panel_reproducible():
    assert simulate_akm(n_workers=200, seed=7).equals(simulate_akm(n_workers=200, seed=7))
    assert not simulate_akm(n_workers=200, seed=7).equals(
        simulate_akm(n_workers=200, seed=8))


def test_workers_move_between_firms():
    """Mobility is what identifies firm effects relative to worker effects."""
    df = simulate_akm(n_workers=500, seed=2)
    firms_per_worker = df.group_by("worker_id").agg(pl.col("firm_id").n_unique())
    movers = (firms_per_worker["firm_id"] > 1).sum()
    assert movers > 0.3 * df["worker_id"].n_unique()


def test_effects_are_recoverable(workdir):
    """The age profile is generated with known coefficients; a fit on the
    simulated data should recover them to within sampling error."""
    pytest.importorskip("pyfixest")
    from hdfe_stream import feols_stream

    df = simulate_akm(n_workers=4_000, seed=5)
    path = workdir.parent / "recover.parquet"
    workdir.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)

    res = feols_stream("log_earn ~ age_squared + age_cubed | worker_id + firm_id",
                       str(path), workdir=workdir, verbose=False)
    estimates = dict(zip(res.coefnames, res.beta))
    # the data-generating values, from simulate_akm
    assert estimates["age_squared"] == pytest.approx(0.08, abs=0.02)
    assert estimates["age_cubed"] == pytest.approx(-0.012, abs=0.004)
    assert res.n_components == 1


def test_rich_panel_adds_the_documented_columns():
    df = simulate_rich(n_workers=500)
    extra = {"state", "region", "cohort", "treat", "occ", "cat", "y2", "x_na",
             "wa", "wf", "z1", "z2", "x2", "y_iv"}
    assert BASE_COLUMNS | extra == set(df.columns)


def test_rich_panel_has_missing_values_in_exactly_one_column():
    df = simulate_rich(n_workers=1_000)
    nulls = {c: df[c].null_count() for c in df.columns}
    assert nulls.pop("x_na") > 0
    assert set(nulls.values()) == {0}


def test_rich_weights_are_usable():
    df = simulate_rich(n_workers=500)
    assert df["wa"].min() > 0                      # analytic weights are positive
    assert df["wf"].min() >= 1
    assert (df["wf"] == df["wf"].round()).all()    # frequency weights are integers


def test_instruments_are_relevant():
    """z1 and z2 have to actually predict x2, or the IV tests would be testing
    a weak-instrument corner rather than the estimator."""
    df = simulate_rich(n_workers=1_000)
    for instrument in ("z1", "z2"):
        assert abs(np.corrcoef(df[instrument], df["x2"])[0, 1]) > 0.1


def test_treat_is_constant_within_worker():
    """Several tests rely on this: it makes C(treat) collinear with the worker
    fixed effect."""
    df = simulate_rich(n_workers=500)
    assert (df.group_by("worker_id").agg(pl.col("treat").n_unique())["treat"] == 1).all()


def test_trends_panel_adds_worker_specific_slopes():
    df = simulate_trends(n_workers=100)
    assert {"t", "t2", "x", "z", "v", "xe", "wa", "y"} <= set(df.columns)
    assert df["t"].dtype == pl.Float64


def test_trends_are_worker_specific(workdir):
    """A model with worker intercepts only should leave a trend in the
    residuals, which the varying-slopes model removes."""
    pytest.importorskip("pyfixest")
    from hdfe_stream import feols_stream

    df = simulate_trends(n_workers=300, seed=4)
    path = workdir.parent / "trends.parquet"
    workdir.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)

    intercepts = feols_stream("y ~ x | worker_id + firm_id", str(path),
                              workdir=workdir / "a", verbose=False)
    slopes = feols_stream("y ~ x | worker_id[t] + firm_id", str(path),
                          workdir=workdir / "b", verbose=False)
    assert slopes.rss < intercepts.rss

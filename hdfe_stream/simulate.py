"""Simulated matched employer-employee panels with AKM structure.

Used by the test suite and the examples, and useful for trying the library out
before pointing it at real data:

    from hdfe_stream import feols_stream
    from hdfe_stream.simulate import simulate_akm

    simulate_akm(n_workers=50_000).write_parquet("sim.parquet")
    feols_stream("log_earn ~ age_squared | worker_id + firm_id + year",
                 "sim.parquet").summary()

`simulate_akm` is the base panel: workers move between firms over a run of
years, and log earnings are a worker effect plus a firm effect plus an age
profile plus noise. Because the worker and firm effects are drawn explicitly,
the true variance decomposition is known, and there is enough worker mobility
to connect the firms into one component.

`simulate_rich` adds the columns the wider feature set needs -- categoricals,
coarser cluster variables, weights, instruments, a second outcome, a column
with missing values -- and `simulate_trends` adds worker-specific time trends
for varying-slopes (FEIS) models. All three take a `seed` and are
deterministic.

Only numpy and Polars are used, so nothing here needs the optional
dependencies.
"""

from __future__ import annotations

import numpy as np
import polars as pl

__all__ = ["simulate_akm", "simulate_rich", "simulate_trends"]

DEFAULT_YEARS = range(2005, 2015)


def simulate_akm(n_workers=50_000, n_firms=None, years=DEFAULT_YEARS,
                 p_move=0.08, p_obs=0.85, sd_worker=0.4, sd_firm=0.25,
                 sd_noise=0.3, seed=0):
    """An AKM panel: one row per worker-year actually observed.

    Parameters
    ----------
    n_workers, n_firms : panel size. `n_firms` defaults to n_workers / 15,
        which keeps firms large enough to be identified.
    years : the years observed (default 2005-2014).
    p_move : probability a worker changes firm between consecutive years.
        Mobility is what connects firms, so lowering it a lot will split the
        data into several components.
    p_obs : probability a worker is observed in a given year, so the panel is
        unbalanced.
    sd_worker, sd_firm, sd_noise : standard deviations of the worker effect,
        the firm effect and the residual.
    seed : anything accepted by `np.random.default_rng`.

    Returns
    -------
    A Polars DataFrame, shuffled so it is not sorted on disk, with columns
    worker_id (int64, deliberately not dense), firm_id (string), year (int32),
    age (float), age_squared, age_cubed (age polynomials scaled into a sane
    range) and log_earn (the outcome).

    Worker and firm effects are positively assorted -- better workers tend to
    be at better firms -- so the firm effect is not orthogonal to the worker
    effect and the fixed effects genuinely have to be estimated jointly.
    """
    if n_firms is None:
        n_firms = max(n_workers // 15, 50)
    rng = np.random.default_rng(seed)
    years = list(years)
    worker_effect = rng.normal(0, sd_worker, n_workers)
    firm_effect = rng.normal(0, sd_firm, n_firms)
    order = np.argsort(firm_effect)             # firms ranked worst to best
    size = rng.pareto(1.2, n_firms) + 1         # a few big firms, many small
    birth = rng.integers(22, 55, n_workers)

    def draw_firm(a):
        """Sorting: a worker's rank in the firm-quality ranking tracks their
        own effect, mixed with size-weighted draws so big firms stay big."""
        rank = np.clip(((a / max(sd_worker, 1e-12) + rng.normal(0, 1.5, a.size)) / 6 + 0.5),
                       0, 0.999)
        base = order[(rank * n_firms).astype(int)]
        alt = rng.choice(n_firms, a.size, p=size / size.sum())
        return np.where(rng.random(a.size) < 0.5, base, alt)

    firm = draw_firm(worker_effect)
    rows = []
    for t, yr in enumerate(years):
        if t:
            moves = rng.random(n_workers) < p_move
            firm = np.where(moves, draw_firm(worker_effect), firm)
        seen = rng.random(n_workers) < p_obs
        age = birth + t
        rows.append(pl.DataFrame({
            # ids are spread out rather than 0..n-1, so that any code path
            # assuming dense ids would fail loudly
            "worker_id": np.flatnonzero(seen).astype(np.int64) * 7 + 1_000_000,
            "firm_id": "F" + pl.Series(firm[seen]).cast(pl.Utf8),
            "year": np.full(int(seen.sum()), yr, np.int32),
            "age": age[seen].astype(float),
            "_worker": worker_effect[seen],
            "_firm": firm_effect[firm[seen]],
        }))
    df = pl.concat(rows)
    noise = np.random.default_rng(seed + 1).normal(0, sd_noise, df.height)
    df = df.with_columns(
        age_squared=(pl.col("age") ** 2) / 100,
        age_cubed=(pl.col("age") ** 3) / 1000,
    ).with_columns(
        log_earn=10 + pl.col("_worker") + pl.col("_firm")
        + 0.08 * pl.col("age_squared") - 0.012 * pl.col("age_cubed")
        + pl.Series(noise),
    ).drop("_worker", "_firm")
    return df.sample(fraction=1.0, shuffle=True, seed=seed)


def simulate_rich(n_workers=50_000, n_firms=None, seed=0, extra_seed=9, **kwargs):
    """`simulate_akm` plus the columns that exercise the wider feature set.

    Adds, on top of the base panel:

    state, region   coarse firm-level groupings, for clustering on something
                    that is not a fixed effect (`region` is coarser still)
    cohort          year-of-birth decade, another non-FE cluster variable
    treat, occ, cat a binary, an integer-coded and a string categorical, for
                    C(), i() and interaction terms
    y2              a second outcome, for multiple-estimation formulas
    x_na            a covariate that is null in ~5% of rows
    wa, wf          analytic and frequency weights
    z1, z2, x2      two instruments and the endogenous regressor they predict
    y_iv            an outcome that depends on x2, for 2SLS

    `log_earn` also picks up an occupation effect and a post-2010 treatment
    effect, so `i(year, treat)` event-study terms have something to find.
    """
    df = simulate_akm(n_workers=n_workers, n_firms=n_firms, seed=seed, **kwargs)
    rng = np.random.default_rng(extra_seed)
    n = df.height

    firms = df["firm_id"].unique().sort()
    state = pl.DataFrame({"firm_id": firms,
                          "state": rng.integers(0, 12, len(firms))})
    df = df.join(state, on="firm_id").with_columns(
        region=pl.col("state") % 4,
        cohort=(pl.col("year") - pl.col("age")) // 10,
        treat=(pl.col("worker_id") % 3 == 0).cast(pl.Int64),
        occ=pl.Series(rng.integers(0, 6, n)),
        cat=pl.lit("k") + pl.Series(rng.integers(0, 3, n)).cast(pl.Utf8),
        y2=pl.col("log_earn") * 0.5 + pl.Series(rng.normal(0, 0.2, n)),
        x_na=pl.when(pl.Series(rng.random(n)) < 0.05).then(None)
              .otherwise(pl.Series(rng.normal(0, 1, n))),
        wa=pl.Series(rng.uniform(0.5, 2.0, n)),
        wf=pl.Series(rng.integers(1, 5, n)).cast(pl.Float64),
    ).with_columns(
        log_earn=pl.col("log_earn") + 0.03 * pl.col("occ")
        + pl.when(pl.col("year") >= 2010).then(0.1 * pl.col("treat")).otherwise(0),
    )

    # instruments: x2 is endogenous through v, which also enters y_iv
    z1, z2, v = (pl.Series(rng.normal(0, 1, n)) for _ in range(3))
    return df.with_columns(
        z1=z1, z2=z2,
        x2=0.6 * z1 + 0.3 * z2 + v + 0.1 * pl.col("age_squared"),
    ).with_columns(
        y_iv=pl.col("log_earn") + 0.2 * pl.col("x2") + 0.5 * v,
    )


def simulate_trends(n_workers=600, n_firms=60, seed=3, **kwargs):
    """`simulate_akm` plus worker-specific time trends, for FEIS models.

    Each worker gets their own linear and quadratic trend in `t` (the year as
    a float), so `y ~ x | worker_id[t] + firm_id` is the correctly specified
    model and a plain worker intercept is not. Adds:

    t, t2      the year, and its centered square
    x, v       an exogenous covariate and an unobservable
    xe, z      an endogenous regressor and its instrument
    wa         analytic weights
    y          the outcome, generated with the worker trends in it

    Defaults are small because the brute-force comparison for varying slopes
    needs one dummy per worker per slope.
    """
    df = simulate_akm(n_workers=n_workers, n_firms=n_firms, seed=seed, **kwargs)
    rng = np.random.default_rng(seed)
    n = df.height

    workers = df["worker_id"].unique().sort()
    trend = pl.DataFrame({"worker_id": workers,
                          "_g": rng.normal(0.02, 0.02, len(workers)),
                          "_g2": rng.normal(0, 0.002, len(workers))})
    mid = float(np.mean(df["year"].unique().to_numpy()))
    return (df.join(trend, on="worker_id")
              .with_columns(t=pl.col("year").cast(pl.Float64),
                            x=pl.Series(rng.normal(0, 1, n)),
                            z=pl.Series(rng.normal(0, 1, n)),
                            wa=pl.Series(rng.uniform(0.5, 2, n)),
                            v=pl.Series(rng.normal(0, 1, n)))
              .with_columns(t2=(pl.col("t") - mid) ** 2,
                            xe=pl.col("z") + 0.5 * pl.col("v") + 0.3 * pl.col("x"))
              .with_columns(y=pl.col("log_earn")
                            + pl.col("_g") * (pl.col("t") - pl.col("t").min())
                            + pl.col("_g2") * pl.col("t2")
                            + 0.3 * pl.col("x") + 0.2 * pl.col("xe")
                            + 0.4 * pl.col("v"))
              .drop("_g", "_g2"))


if __name__ == "__main__":
    import sys

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50_000
    out = sys.argv[2] if len(sys.argv) > 2 else "sim.parquet"
    df = simulate_rich(n_workers=n)
    df.write_parquet(out)
    print(f"{df.height:,} rows x {df.width} columns -> {out}")
    print(f"{df['worker_id'].n_unique():,} workers, {df['firm_id'].n_unique():,} firms, "
          f"{df['year'].n_unique()} years")

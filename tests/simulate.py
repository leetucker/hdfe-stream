"""Simulate a matched employer-employee panel with AKM structure."""
import numpy as np
import polars as pl


def simulate(n_workers=50_000, n_firms=3_000, years=range(2005, 2015),
             p_move=0.08, p_obs=0.85, seed=0):
    rng = np.random.default_rng(seed)
    T = len(years)
    alpha = rng.normal(0, 0.4, n_workers)
    # sorting: better workers draw from better firms
    firm_q = rng.normal(0, 0.25, n_firms)
    order = np.argsort(firm_q)
    size = rng.pareto(1.2, n_firms) + 1
    birth = rng.integers(22, 55, n_workers)

    def draw_firm(a):
        rank = np.clip(((a / 0.4 + rng.normal(0, 1.5, a.size)) / 6 + 0.5), 0, 0.999)
        base = order[(rank * n_firms).astype(int)]
        # mix with size-weighted draws so big firms are big
        alt = rng.choice(n_firms, a.size, p=size / size.sum())
        return np.where(rng.random(a.size) < 0.5, base, alt)

    firm = draw_firm(alpha)
    rows = []
    for t, yr in enumerate(years):
        if t:
            mv = rng.random(n_workers) < p_move
            firm = np.where(mv, draw_firm(alpha), firm)
        seen = rng.random(n_workers) < p_obs
        age = birth + t
        rows.append(pl.DataFrame({
            "pik": np.flatnonzero(seen).astype(np.int64) * 7 + 1_000_000,  # non-dense ids
            "sein": ("E" + pl.Series(firm[seen]).cast(pl.Utf8)),
            "year": np.full(seen.sum(), yr, np.int32),
            "age": age[seen].astype(float),
            "_a": alpha[seen], "_f": firm_q[firm[seen]],
        }))
    df = pl.concat(rows)
    rng2 = np.random.default_rng(seed + 1)
    df = df.with_columns(
        age_squared=(pl.col("age") ** 2) / 100,
        age_cubed=(pl.col("age") ** 3) / 1000,
    ).with_columns(
        log_earn=10 + pl.col("_a") + pl.col("_f")
        + 0.08 * pl.col("age_squared") - 0.012 * pl.col("age_cubed")
        + pl.Series(rng2.normal(0, 0.3, df.height)),
    ).drop("_a", "_f")
    return df.sample(fraction=1.0, shuffle=True, seed=seed)  # unsorted on disk


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50_000
    df = simulate(n_workers=n, n_firms=max(n // 15, 50))
    df.write_parquet("sim.parquet")
    print(df.shape)
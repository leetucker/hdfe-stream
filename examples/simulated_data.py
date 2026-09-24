"""Simulated Parquet files for the examples to read.

The examples all read from disk rather than from a DataFrame in memory,
because that is how hdfe_stream is meant to be used: `feols_stream` accepts a
Parquet path or glob and streams it.

Each file is written once, into `examples/output/`, and reused afterwards.
Delete that directory to start over.
"""

from __future__ import annotations

from pathlib import Path

from hdfe_stream.simulate import simulate_rich, simulate_trends

OUTPUT = Path(__file__).parent / "output"

# 20,000 workers over 10 years is roughly 170,000 rows: large enough to be
# interesting, small enough that every example finishes in seconds. Raise it to
# see the streaming machinery work harder -- nothing else needs to change.
N_WORKERS = 20_000


def _ensure(name, build):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / name
    if not path.exists():
        frame = build()
        frame.write_parquet(path)
        print(f"wrote {path.relative_to(Path.cwd()) if path.is_relative_to(Path.cwd()) else path}"
              f"  ({frame.height:,} rows x {frame.width} columns)")
    return str(path)


def panel_path(n_workers=N_WORKERS):
    """A matched worker-firm panel with everything the examples need.

    Columns: worker_id, firm_id, year, age (+ age_squared, age_cubed),
    log_earn, a second outcome y2, categoricals (treat, occ, cat), cluster
    variables that are not fixed effects (state, region, cohort), weights
    (wa, wf), and an IV block (x2 endogenous, z1/z2 instruments, y_iv).

    See `hdfe_stream.simulate.simulate_rich` for how it is generated.
    """
    return _ensure("panel.parquet", lambda: simulate_rich(n_workers=n_workers))


def trends_path(n_workers=2_000):
    """A panel where each worker has their own time trend, for varying slopes.

    Adds t (the year as a float), t2, an exogenous x, and an outcome y that
    genuinely contains worker-specific trends.
    """
    return _ensure("trends.parquet", lambda: simulate_trends(n_workers=n_workers))


def workdir(name):
    """Scratch space for one example's intermediates and result files."""
    path = OUTPUT / "work" / name
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


if __name__ == "__main__":
    panel_path()
    trends_path()
    print(f"\ndata ready in {OUTPUT}")

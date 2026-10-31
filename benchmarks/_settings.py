"""The memory settings the benchmarks compare, in one place.

The benchmark scripts run them, the figure script (plot_benchmarks.py) prints
their definitions under the figures that draw them, and the documentation
quotes them. None of those imports the others' heavy dependencies, so this file
imports nothing.

A configuration's *variant* says which settings it runs with:

    "<solver>"                  hdfe_stream's defaults, with that solver
    "<solver>+lowmem"           the low-memory setting
    "<solver>+limit:<GB>"       told it may use <GB> of memory (`max_memory_gb`)
    "<solver>+sweep:<R>:<B>"    `rows_per_bucket=R`, `batch_rows=B` (the memory sweep)
"""

# "low memory": smaller read batches and more buckets, so less of the data is
# in flight at once. These are explicit settings of the two options; the
# defaults are buckets of about 384 MB and batches of 2,000,000 rows.
LOW_MEMORY = {"rows_per_bucket": 250_000, "batch_rows": 200_000}

# "sized to the design": rows held at once scale down with the width of the
# design, so that a bucket (sorted in memory) and a batch or row group (read in
# memory) stay near these many bytes of design however wide it is. Never above
# the defaults, so a narrow design runs exactly as the default configuration.
SIZED_BUCKET_BYTES = 1e9
SIZED_BATCH_BYTES = 250e6

LOW_MEMORY_NOTE = (
    f"Low memory: rows_per_bucket={LOW_MEMORY['rows_per_bucket']:,} and "
    f"batch_rows={LOW_MEMORY['batch_rows']:,} (the defaults are buckets of about "
    "384 MB and batches of 2,000,000 rows).")

SIZED_NOTE = (
    "Sized to the design: rows per bucket and per batch scaled down with the "
    f"width of the design, to about {SIZED_BUCKET_BYTES / 1e9:g} GB and "
    f"{SIZED_BATCH_BYTES / 1e6:g} MB of rows (never above the defaults).")


def limit_note(gb):
    """What a ``max_memory_gb=<gb>`` series is."""
    return (f"max_memory_gb={gb}: the fit is told it may use {gb} GB, and sizes its "
            "buckets, solver and accumulators to fit what is left of that; the "
            "other settings are the defaults. The limit is a target, not a "
            "guarantee, so the peak is measured, not assumed to stay under it.")


def hdfe_options(variant):
    """``(solver, options)`` for a variant (see the module docstring)."""
    solver, _, extra = variant.partition("+")
    if not extra:
        return solver, {}
    kind, *args = extra.split(":")
    if kind == "lowmem" and not args:
        return solver, dict(LOW_MEMORY)
    if kind == "limit" and len(args) == 1:
        return solver, {"max_memory_gb": float(args[0])}
    if kind == "sweep" and len(args) == 2:
        return solver, {"rows_per_bucket": int(args[0]), "batch_rows": int(args[1])}
    raise ValueError(f"unknown variant {variant!r}")

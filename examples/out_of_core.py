"""Running on data that does not fit in memory: disk, knobs, and logging.

This is the reason the library exists, so it is worth being concrete about what
is actually held in memory and what is not.

Never in memory
    the rows. They are read in batches, hash-partitioned into buckets by the
    streamed dimension, sorted one bucket at a time, and reduced to a table of
    cells. Row-level output (residuals, per-row fixed effects) is written
    straight to Parquet.

In memory
    one vector per level of each *non-streamed* dimension, times a block of at
    most `rhs_block` variables -- so firms x years, not workers x rows. Plus,
    for the explicit solver, the reduced matrix S, whose size depends on how
    levels co-occur rather than on the number of rows.

The streamed dimension -- normally the biggest one, workers -- costs no memory
at all. That asymmetry is the whole design.

    python examples/out_of_core.py
"""

import logging
from pathlib import Path

import hdfe_stream
from hdfe_stream import feols_stream
from simulated_data import OUTPUT, panel_path, workdir

data = panel_path()
FML = "log_earn ~ age_squared + age_cubed | worker_id + firm_id + year"

# --------------------------------------------------------------- what it costs
# Every fit reports what it used. `disk_peak_gb` is the number to watch when
# sizing a scratch volume: intermediates are deleted as soon as they are no
# longer needed, so the peak is well below the sum of everything written.
fit = feols_stream(FML, data, workdir=workdir("cost"), verbose=False)
print("resource use:")
for key in ["disk_peak_gb", "disk_results_gb", "seconds_total", "cells_per_obs",
            "assembly"]:
    print(f"  {key:<18}{fit.diagnostics[key]}")
print(f"  {'solver':<18}{fit.solver_info['solver']}"
      f" ({fit.solver_info['iterations']} iterations)")

# `cells_per_obs` is worth understanding: rows are reduced to cells, one per
# combination of fixed-effect levels. At 1.0 every row is its own cell and the
# reduction buys nothing; well below 1.0 the later passes get much cheaper.

# ----------------------------------------------------------------- the solvers
# "explicit"   build the reduced matrix S once and solve in memory. Fastest
#              when S fits -- and S is sized by how levels co-occur, not by rows.
# "stream_cg"  never build S; apply it by streaming the cells each iteration.
#              Slower per iteration, but its memory does not depend on S.
# "auto"       explicit, falling back to stream_cg if S would exceed max_s_gb.
print("\nsolvers:")
for solver, options in [("explicit", {}), ("stream_cg", {}),
                        ("auto (tiny budget, falls back)", {"solver": "auto",
                                                            "max_s_gb": 1e-7})]:
    name = options.pop("solver", solver)
    run = feols_stream(FML, data, workdir=workdir(f"solver_{solver[:9]}"),
                       solver=name, verbose=False, **options)
    info = run.solver_info
    print(f"  {solver:<32}-> {info['solver']:<10}"
          f"{info['seconds']:>7.2f}s  {info['iterations']} iterations")
    if "fallback" in info:
        print(f"      fell back because: {info['fallback']}")

# -------------------------------------------------------------- memory budgets
# batch_rows       rows read per batch
# rows_per_bucket  target rows per fe[0] bucket; more buckets, less peak memory
#                  in the sort
# rhs_block        variables solved for at a time. This bounds the biggest
#                  in-memory array: levels x rhs_block
# max_s_gb         budget for the explicit S (default: a quarter of RAM)
tight = feols_stream(FML, data, workdir=workdir("tight"),
                     batch_rows=20_000,        # small batches
                     rows_per_bucket=50_000,   # many buckets
                     rhs_block=2,              # narrow solve blocks
                     verbose=False)
print(f"\nwith tight memory settings: beta unchanged to "
      f"{abs(tight.beta - fit.beta).max():.1e}, "
      f"{tight.solver_info['blocks']} solve block(s)")

# -------------------------------------------------------------------- the disk
# Each fit works in its own run directory under `workdir`, so concurrent fits
# never collide. Result files live as long as the result object; everything else
# is deleted as soon as the fit is done with it.
run_dir = Path(fit.files_dir).parents[1]
print(f"\nfiles kept after the fit, under {run_dir.name}:")
for path in sorted(run_dir.rglob("*")):
    if path.is_file():
        print(f"  {path.relative_to(run_dir)}  ({path.stat().st_size / 1e6:.2f} MB)")

# `outputs="keep"` survives the result object, for reading in another process.
# `save_resid=False` skips the row-level file, which is the big one.
lean = feols_stream(FML, data, workdir=workdir("lean"), save_resid=False,
                    verbose=False)
print(f"\nwith save_resid=False: results take "
      f"{lean.diagnostics['disk_results_gb'] * 1000:.1f} MB instead of "
      f"{fit.diagnostics['disk_results_gb'] * 1000:.1f} MB "
      "(the row-level file is the big one)")

# A context manager releases the files at the end of the block.
with feols_stream(FML, data, workdir=workdir("scoped"), verbose=False) as scoped:
    firm_effects = scoped.fixef("firm_id").collect()      # collect what you need
print(f"collected {firm_effects.height:,} firm effects; files released on exit")

# Leftovers from a killed job can be swept up. Only directories this library
# created are touched, and by default only ones whose process is gone.
leftover = hdfe_stream.cleanup(OUTPUT / "work", dry_run=True)
print(f"\ncleanup(dry_run=True) would remove {len(leftover)} leftover run(s), "
      f"{sum(size for _, size in leftover) / 1e6:.1f} MB")

# ------------------------------------------------------------------ logging
# A long out-of-core fit is not interactive. Pass a logger and progress,
# warnings and summaries all go to it as they happen, with nothing on stdout.
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s %(message)s",
                    force=True)      # force: something may already have configured it
log = logging.getLogger("akm")

print("\nwith logger= (progress goes to the log, not stdout):")
logged = feols_stream(FML, data, workdir=workdir("logged"), logger=log)
logged.summary(logger=log)           # the whole summary as one record

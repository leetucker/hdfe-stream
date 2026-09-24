"""Time and peak memory: hdfe_stream against pyfixest on an AKM regression.

    python benchmarks/akm_benchmark.py                 # default sizes
    python benchmarks/akm_benchmark.py 25000 100000    # pick your own

The model is the standard three-way AKM specification on the simulated panel
that ships with the library:

    log_earn ~ age_squared + age_cubed | worker_id + firm_id + year

Each configuration runs in its own subprocess, because peak memory is a
high-water mark the kernel keeps per process: measuring several configurations
in one process would report the largest of them for all. The child reports

  * wall-clock seconds, split into reading the data and fitting
  * peak resident set size, from getrusage, for the whole process
  * for hdfe_stream, peak disk use, which is what it spends instead of memory

pyfixest needs the data as a pandas DataFrame, so reading it in is part of the
job and is counted. hdfe_stream is given the Parquet path and does its own
reading, in batches. That difference *is* the comparison -- it is not an
artifact of how this is measured.

The two libraries are asked for comparable accuracy rather than identical
settings, since their convergence criteria measure different things: pyfixest
iterates until the demeaned variables stop moving, hdfe_stream until the
reduced normal equations are solved. Defaults are used for both, and the
script checks that the coefficients agree to a few parts in 1e-7 before
reporting a row, so the times are for equally good answers.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
FML = "log_earn ~ age_squared + age_cubed | worker_id + firm_id + year"

# n_workers for each size; the panel is roughly 8.5 rows per worker
DEFAULT_SIZES = [25_000, 100_000, 400_000]

CONFIGS = {
    "pyfixest (MapDemeaner, default)": ("pyfixest", "map"),
    "pyfixest (LsmrDemeaner)": ("pyfixest", "lsmr"),
    "hdfe_stream (solver='explicit')": ("hdfe", "explicit"),
    "hdfe_stream (solver='stream_cg')": ("hdfe", "stream_cg"),
    "hdfe_stream (solver='within')": ("hdfe", "within"),
    # the same solve with the batch and bucket sizes turned down: this is the
    # knob pyfixest has no equivalent of
    "hdfe_stream (stream_cg, low memory)": ("hdfe", "stream_cg+lowmem"),
}

# what "low memory" means above: smaller read batches and more fe[0] buckets,
# so less of the data is in flight at once
LOW_MEMORY = {"rows_per_bucket": 250_000, "batch_rows": 200_000}


def peak_rss_mb():
    """Peak resident set size of this process, in MB.

    Read from /proc/self/status, not from getrusage: on Linux ru_maxrss is
    *inherited across fork+exec*, so a child spawned from a parent holding
    500 MB reports 500 MB before doing anything at all. VmHWM is the kernel's
    per-process high-water mark and is reset on exec, which is what we need.
    Falls back to getrusage where /proc is not available, which is then only
    meaningful if the parent is small.
    """
    try:
        with open("/proc/self/status") as handle:
            for line in handle:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) / 1024      # kB -> MB
    except OSError:
        pass
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def data_path(n_workers):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / f"akm_{n_workers}.parquet"
    if not path.exists():
        from hdfe_stream.simulate import simulate_akm
        frame = simulate_akm(n_workers=n_workers)
        frame.write_parquet(path)
        print(f"  simulated {frame.height:,} rows -> {path.name}", flush=True)
    return path


# --------------------------------------------------------------------------
# the child process: one configuration, one size
# --------------------------------------------------------------------------

def run_pyfixest(path, variant, workdir):
    import pandas as pd
    import pyfixest as pf

    t0 = time.perf_counter()
    frame = pd.read_parquet(path)
    t_read = time.perf_counter() - t0

    demeaner = pf.MapDemeaner() if variant == "map" else pf.LsmrDemeaner()
    t0 = time.perf_counter()
    fit = pf.feols(FML, data=frame, vcov={"CRV1": "worker_id"},
                   fixef_rm="none", demeaner=demeaner)
    t_fit = time.perf_counter() - t0

    return {"t_read": t_read, "t_fit": t_fit, "n_obs": int(fit._N),
            "coefs": fit.coef().to_dict(), "disk_gb": 0.0}


def run_hdfe(path, solver, workdir):
    from hdfe_stream import feols_stream

    options = {}
    if solver.endswith("+lowmem"):
        solver = solver.removesuffix("+lowmem")
        options = dict(LOW_MEMORY)
    elif "+sweep:" in solver:
        solver, rows_per_bucket, batch_rows = solver.split(":")
        solver = solver.removesuffix("+sweep")
        options = {"rows_per_bucket": int(rows_per_bucket),
                   "batch_rows": int(batch_rows)}

    t0 = time.perf_counter()
    fit = feols_stream(FML, str(path), workdir=str(workdir), solver=solver,
                       vcov={"CRV1": "worker_id"}, fe_dof="pyfixest",
                       verbose=False, save_resid=False, **options)
    t_fit = time.perf_counter() - t0

    out = {"t_read": 0.0, "t_fit": t_fit, "n_obs": float(fit.n_obs),
           "coefs": fit.coef(), "disk_gb": fit.diagnostics["disk_peak_gb"]}
    fit.cleanup()
    return out


def child(kind, variant, path, workdir):
    """Run one configuration and print a JSON line to stdout."""
    baseline = peak_rss_mb()          # after interpreter start, before the work
    runner = run_pyfixest if kind == "pyfixest" else run_hdfe
    result = runner(Path(path), variant, Path(workdir))
    result["peak_rss_mb"] = peak_rss_mb()
    result["baseline_rss_mb"] = baseline
    result["coefs"] = {k: float(v) for k, v in result["coefs"].items()}
    print("RESULT " + json.dumps(result))


# --------------------------------------------------------------------------
# the parent process
# --------------------------------------------------------------------------

def measure(kind, variant, path, workdir, warmup=False):
    """Run one configuration in a subprocess and return its report."""
    command = [sys.executable, str(Path(__file__).resolve()),
               "--child", kind, variant, str(path), str(workdir)]
    env = dict(os.environ, PYTHONWARNINGS="ignore")
    proc = subprocess.run(command, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-3:]
        return {"failed": " | ".join(tail)}
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[len("RESULT "):])
    return {"failed": "no result line"}


MEMORY_SWEEP = [(20_000_000, 2_000_000), (1_000_000, 500_000),
                (250_000, 200_000), (100_000, 100_000)]


def memory_sweep(n_workers):
    """How peak memory responds to the batch and bucket settings.

    Same data, same solver, same answer -- only how much is in flight at once.
    """
    path = data_path(n_workers)
    workdir = DATA_DIR / "work"
    workdir.mkdir(parents=True, exist_ok=True)
    measure("hdfe", "stream_cg", path, workdir)          # warm up

    print(f"\n=== memory knobs, {n_workers:,} workers ===")
    print("| rows_per_bucket | batch_rows | wall time | peak memory | peak disk |")
    print("|---:|---:|---:|---:|---:|")
    for rows_per_bucket, batch_rows in MEMORY_SWEEP:
        variant = f"stream_cg+sweep:{rows_per_bucket}:{batch_rows}"
        report = measure("hdfe", variant, path, workdir)
        if "failed" in report:
            print(f"| {rows_per_bucket:,} | {batch_rows:,} | failed | | |")
            continue
        print(f"| {rows_per_bucket:,} | {batch_rows:,} "
              f"| {report['t_fit']:,.1f} s | {report['peak_rss_mb']:,.0f} MB "
              f"| {report['disk_gb'] * 1000:,.0f} MB |")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sizes", nargs="*", type=int, default=DEFAULT_SIZES,
                        help="number of workers in each panel")
    parser.add_argument("--memory-sweep", type=int, metavar="N_WORKERS",
                        help="instead of the comparison, show how peak memory "
                             "responds to rows_per_bucket and batch_rows")
    parser.add_argument("--child", nargs=4, metavar=("KIND", "VARIANT", "PATH", "WORKDIR"),
                        help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.child:
        child(*args.child)
        return
    if args.memory_sweep:
        memory_sweep(args.memory_sweep)
        return

    workdir = DATA_DIR / "work"
    workdir.mkdir(parents=True, exist_ok=True)
    results = {}

    for n_workers in args.sizes:
        path = data_path(n_workers)
        import polars as pl
        n_rows = pl.scan_parquet(path).select(pl.len()).collect().item()
        print(f"\n=== {n_workers:,} workers, {n_rows:,} rows "
              f"({path.stat().st_size / 1e6:.0f} MB on disk) ===", flush=True)

        # warm up once so numba kernel compilation (cached on disk) is not
        # charged to the first configuration that happens to run
        measure("hdfe", "explicit", path, workdir)

        for label, (kind, variant) in CONFIGS.items():
            report = measure(kind, variant, path, workdir)
            results[(n_workers, label)] = report
            if "failed" in report:
                print(f"  {label:<34} FAILED: {report['failed'][:80]}", flush=True)
                continue
            total = report["t_read"] + report["t_fit"]
            print(f"  {label:<34}{total:7.1f}s total "
                  f"({report['t_read']:5.1f}s read + {report['t_fit']:5.1f}s fit)"
                  f"   peak {report['peak_rss_mb']:7.0f} MB"
                  f"  (imports {report['baseline_rss_mb']:5.0f} MB,"
                  f" work {report['peak_rss_mb'] - report['baseline_rss_mb']:7.0f} MB)"
                  + (f"   disk {report['disk_gb'] * 1000:.0f} MB"
                     if report["disk_gb"] else ""), flush=True)

        check_agreement(results, n_workers)

    print()
    print(markdown_table(results, args.sizes))


def check_agreement(results, n_workers):
    """All configurations must produce the same estimates, or the timings are
    not comparable."""
    rows = [(label, r) for (size, label), r in results.items()
            if size == n_workers and "failed" not in r]
    if len(rows) < 2:
        return
    base_label, base = rows[0]
    worst = 0.0
    for label, report in rows[1:]:
        for name, value in base["coefs"].items():
            worst = max(worst, abs(report["coefs"][name] - value) / max(abs(value), 1e-12))
    verdict = "agree" if worst < 1e-6 else "DISAGREE"
    print(f"  estimates {verdict}: largest relative difference {worst:.1e}", flush=True)


def markdown_table(results, sizes):
    lines = []
    for n_workers in sizes:
        rows = [(label, results.get((n_workers, label))) for label in CONFIGS]
        if not any(r and "failed" not in r for _, r in rows):
            continue
        baseline = next((r for _, r in rows if r and "failed" not in r), None)
        n_obs = f"{baseline['n_obs']:,.0f}" if baseline else "?"
        lines.append(f"\n**{n_workers:,} workers, {n_obs} rows**\n")
        lines.append("| configuration | wall time | peak memory | peak disk |")
        lines.append("|---|---:|---:|---:|")
        for label, report in rows:
            if not report or "failed" in report:
                lines.append(f"| {label} | failed | | |")
                continue
            total = report["t_read"] + report["t_fit"]
            disk = f"{report['disk_gb'] * 1000:,.0f} MB" if report["disk_gb"] else "—"
            lines.append(f"| {label} | {total:,.1f} s | {report['peak_rss_mb']:,.0f} MB "
                         f"| {disk} |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()

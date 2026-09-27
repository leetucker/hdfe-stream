"""Shared machinery for the benchmarks: one configuration per subprocess, peak
memory and disk measured the same way for every library, results to CSV.

Each configuration runs in its own subprocess because peak memory is a
high-water mark the kernel keeps per process: measuring several configurations
in one process would report the largest of them for all.

The child measures

  * wall-clock seconds, split into reading the data and the estimation itself;
  * peak resident set size (`VmHWM`) for the whole process;
  * peak disk, by polling the allocated size of the configuration's own work
    directory. Libraries that do not write to disk report zero.

The parent writes one CSV row per (size, configuration) as soon as it has it,
so a run that dies part-way keeps everything measured so far, and re-running a
size replaces that size's rows rather than duplicating them.
"""

from __future__ import annotations

import csv
import json
import os
import platform
import resource
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
RESULTS_DIR = HERE / "results"

# the simulated panel has firms = workers / 15 at every size (simulate_akm's
# default), so the x axis of every figure is a single, interpretable dimension
FIRMS_PER_WORKER = 1 / 15


def peak_rss_mb():
    """Peak resident set size of this process, in MB.

    Read from /proc/self/status, not from getrusage: on Linux ru_maxrss is
    *inherited across fork+exec*, so a child spawned from a parent holding
    500 MB reports 500 MB before doing anything at all. VmHWM is the kernel's
    per-process high-water mark and is reset on exec, which is what we need.
    """
    try:
        with open("/proc/self/status") as handle:
            for line in handle:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) / 1024      # kB -> MB
    except OSError:
        pass
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def allocated_bytes(root):
    """Bytes actually allocated under `root` (blocks, not apparent size, so
    sparse files count for what they occupy)."""
    total = 0
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            try:
                total += os.lstat(os.path.join(dirpath, name)).st_blocks * 512
            except OSError:
                pass            # removed between listing and stat
    return total


class DiskPoller:
    """Samples the allocated size of a directory in a background thread and
    keeps the maximum. A peak shorter than the interval can be missed, which is
    why hdfe_stream's own bookkeeping is also used where it has one."""

    def __init__(self, root, interval=0.2):
        self.root = Path(root)
        self.interval = interval
        self.peak = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, allocated_bytes(self.root))
            self._stop.wait(self.interval)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        self.peak = max(self.peak, allocated_bytes(self.root))


def data_path(n_workers):
    """The simulated panel for `n_workers`, generated once and cached."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / f"akm_{n_workers}.parquet"
    if not path.exists():
        from hdfe_stream.simulate import simulate_akm
        frame = simulate_akm(n_workers=n_workers)
        frame.write_parquet(path)
        print(f"  simulated {frame.height:,} rows -> {path.name}", flush=True)
    return path


def panel_shape(path):
    import polars as pl

    lazy = pl.scan_parquet(path)
    row = lazy.select(pl.len().alias("rows"),
                      pl.col("worker_id").n_unique().alias("workers"),
                      pl.col("firm_id").n_unique().alias("firms")).collect()
    return {k: int(row[k].item()) for k in ("rows", "workers", "firms")}


def run_child(runner, kind, variant, path, workdir):
    """In the child: run one configuration and print its report as JSON."""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    baseline = peak_rss_mb()          # after interpreter start, before the work
    with DiskPoller(workdir) as disk:
        result = runner(kind, variant, Path(path), workdir)
    result["peak_rss_mb"] = peak_rss_mb()
    result["baseline_rss_mb"] = baseline
    result["peak_disk_mb"] = max(disk.peak / 1e6,
                                 result.pop("reported_disk_mb", 0.0))
    print("RESULT " + json.dumps(result, default=float))


def measure(script, kind, variant, path, workdir, timeout=None):
    """Run one configuration in a subprocess and return its report."""
    workdir = Path(workdir)
    if workdir.exists():
        shutil.rmtree(workdir)
    command = [sys.executable, str(script), "--child", kind, variant,
               str(path), str(workdir)]
    env = dict(os.environ, PYTHONWARNINGS="ignore")
    try:
        proc = subprocess.run(command, capture_output=True, text=True, env=env,
                              timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"status": f"timeout after {timeout:.0f} s"}
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    if proc.returncode == -9:
        return {"status": "killed (out of memory)"}
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-3:]
        return {"status": "failed: " + " | ".join(tail)[:300]}
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT "):
            report = json.loads(line[len("RESULT "):])
            report["status"] = "ok"
            return report
    return {"status": "failed: no result line"}


def upsert_csv(path, rows, key=("n_workers", "configuration")):
    """Write `rows` into the CSV at `path`, replacing rows with the same key."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if path.exists():
        with open(path, newline="") as handle:
            existing = list(csv.DictReader(handle))
    new_keys = {tuple(str(r[k]) for k in key) for r in rows}
    kept = [r for r in existing if tuple(str(r[k]) for k in key) not in new_keys]
    merged = kept + [{k: v for k, v in r.items()} for r in rows]
    fields = list(dict.fromkeys(f for r in merged for f in r))
    merged.sort(key=lambda r: (int(r["n_workers"]), str(r.get("order", ""))))
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(merged)


def machine_info():
    """What the numbers were measured on, for the documentation."""
    import importlib.metadata as md

    def version(name):
        try:
            return md.version(name)
        except md.PackageNotFoundError:
            return None

    cpu = platform.processor()
    try:
        with open("/proc/cpuinfo") as handle:
            for line in handle:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    mem_gb = None
    try:
        with open("/proc/meminfo") as handle:
            for line in handle:
                if line.startswith("MemTotal:"):
                    mem_gb = round(int(line.split()[1]) / 1024 ** 2, 1)
    except OSError:
        pass
    return {"cpu": cpu, "logical_cpus": os.cpu_count(), "memory_gb": mem_gb,
            "platform": platform.platform(), "python": platform.python_version(),
            **{name: version(name) for name in
               ("hdfe-stream", "pyfixest", "xhdfe", "polars", "numpy", "numba",
                "scipy", "pandas")},
            "measured": time.strftime("%Y-%m-%d")}


def write_machine_info(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(machine_info(), indent=1) + "\n")

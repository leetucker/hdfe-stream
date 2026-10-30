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

Two libraries in other languages, R's fixest and Julia's FixedEffectModels.jl,
run the same way from benchmarks/external/: a script per language, given what
to fit as JSON, measures itself as the Python child does (VmHWM for peak
memory) and prints the same RESULT line. Neither language is a dependency:
where one is not installed, its configurations are skipped, and nothing is
written for them (benchmarks/README.md says how to install them).

A run that is restarted can skip what it already measured. When the environment
variable HDFE_BENCH_LEDGER names a file, every row written is also recorded
there with a digest, and `is_done` reports a configuration as finished only
if its row is still in the CSV exactly as the ledger recorded it. Without the
variable, which is the normal case, nothing is skipped and nothing extra is
written.
"""

from __future__ import annotations

import csv
import hashlib
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
# HDFE_BENCH_RESULTS redirects the CSVs and machine.json, for trial runs that
# should not touch the published results
RESULTS_DIR = Path(os.environ.get("HDFE_BENCH_RESULTS") or HERE / "results")
EXTERNAL_DIR = HERE / "external"

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


def glm_data_path(n_workers):
    """The simulated panel for `n_workers` with a count and a binary outcome,
    generated once and cached.

    The rows are those of `data_path(n_workers)` (the same simulation and seed),
    and both outcomes follow the AKM index of log_earn, less its noise and
    shifted to a sensible scale:

        index   = worker effect + firm effect + 0.08 age_squared - 0.012 age_cubed - 0.4
        y_count ~ Poisson(exp(index - 0.5))
        y_binary ~ Bernoulli(1 / (1 + exp(-index)))

    so the true coefficients on age_squared and age_cubed are 0.08 and -0.012
    for both.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / f"glm_{n_workers}.parquet"
    if not path.exists():
        import numpy as np
        import polars as pl
        from hdfe_stream.simulate import simulate_akm

        frame = simulate_akm(n_workers=n_workers, keep_effects=True)
        index = (frame["true_worker_effect"] + frame["true_firm_effect"]
                 + 0.08 * frame["age_squared"] - 0.012 * frame["age_cubed"]
                 - 0.4).to_numpy()
        rng = np.random.default_rng(20260927)
        frame = frame.drop("true_worker_effect", "true_firm_effect").with_columns(
            y_count=pl.Series(rng.poisson(np.exp(index - 0.5)).astype(float)),
            y_binary=pl.Series((rng.random(index.size)
                                < 1 / (1 + np.exp(-index))).astype(float)))
        frame.write_parquet(path)
        print(f"  simulated {frame.height:,} rows -> {path.name}", flush=True)
    return path


GENERAL_CHUNK = 1_000_000


def general_data_path(n_rows, n_covariates=10):
    """The simulated data of the OLS, IV and binary-outcome benchmarks, for
    `n_rows` rows and `n_covariates` covariates, generated once and cached.

    No panel structure, unlike the AKM data: independent rows, written a
    million at a time, so that the generator never holds more than that.

        x0 .. x{k-1}  standard normal; x0 also loads on the group effect
        g             one of n_rows / 20 groups (the fixed effect), effect a_g ~ N(0, 1)
        z1, z2        instruments, standard normal
        d             endogenous: 0.6 z1 + 0.4 z2 + 0.5 u + v + 0.3 a_g
        y             sum_j b_j x_j + a_g + e                       (OLS)
        yiv           d + sum_j b_j x_j + a_g + u                   (IV; d is correlated with u)
        yb            Bernoulli(logistic(0.4 sum_j b_j x_j - 0.2))  (logit and probit)

    with b_j running from 0.5 to -0.5 over the covariates.
    """
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / f"general_{n_rows}_{n_covariates}.parquet"
    if path.exists():
        return path
    k = n_covariates
    groups = max(n_rows // 20, 1)
    beta = np.linspace(0.5, -0.5, k)
    effects = np.random.default_rng(20261030).standard_normal(groups)
    partial = path.with_name(path.name + ".part")
    writer = None
    for chunk, start in enumerate(range(0, n_rows, GENERAL_CHUNK)):
        n = min(GENERAL_CHUNK, n_rows - start)
        rng = np.random.default_rng([20261030, chunk])
        g = rng.integers(0, groups, n)
        X = rng.standard_normal((n, k))
        X[:, 0] += 0.5 * effects[g]
        z = rng.standard_normal((n, 2))
        u, v, e = rng.standard_normal((3, n))
        d = 0.6 * z[:, 0] + 0.4 * z[:, 1] + 0.5 * u + v + 0.3 * effects[g]
        index = X @ beta
        columns = {f"x{j}": X[:, j] for j in range(k)}
        columns.update(g=g, z1=z[:, 0], z2=z[:, 1], d=d,
                       y=index + effects[g] + e,
                       yiv=d + index + effects[g] + u,
                       yb=(rng.random(n) < 1 / (1 + np.exp(-(0.4 * index - 0.2)))
                           ).astype(float))
        table = pa.table(columns)
        if writer is None:
            writer = pq.ParquetWriter(partial, table.schema)
        writer.write_table(table)
    writer.close()
    partial.rename(path)
    print(f"  simulated {n_rows:,} rows -> {path.name}", flush=True)
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


# how R, Julia and Python say that an allocation was refused
OUT_OF_MEMORY = ("cannot allocate vector of size", "OutOfMemoryError", "MemoryError")


def measure(script, kind, variant, path, workdir, timeout=None, command=None):
    """Run one configuration in a subprocess and return its report: the
    script's own child, or `command` (from `external_command`) instead."""
    workdir = Path(workdir)
    if workdir.exists():
        shutil.rmtree(workdir)
    command = command or [sys.executable, str(script), "--child", kind, variant,
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
        output = (proc.stderr or proc.stdout).strip().splitlines()
        # an allocation larger than the machine can back is refused outright,
        # before the kernel would kill the process, and each language reports
        # it in its own words
        refused = next((line.strip() for line in output if any(
            sign in line for sign in OUT_OF_MEMORY)), None)
        if refused:
            return {"status": "out of memory: " + refused[:200]}
        return {"status": "failed: " + " | ".join(output[-3:])[:300]}
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT "):
            report = json.loads(line[len("RESULT "):])
            report["status"] = "ok"
            return report
    return {"status": "failed: no result line"}


# --------------------------------------------------------------------------
# R and Julia
# --------------------------------------------------------------------------

EXTERNAL = {
    # kind -> (language, executable, its environment variable, script)
    "fixest": ("R", "Rscript", "HDFE_BENCH_RSCRIPT", "fixest.R"),
    "fem": ("Julia", "julia", "HDFE_BENCH_JULIA", "fixedeffectmodels.jl"),
    # the OLS, IV and binary-outcome benchmarks: fixest, and base R's lm and glm
    "regr": ("R", "Rscript", "HDFE_BENCH_RSCRIPT", "regression.R"),
}


def external_executable(kind):
    """Rscript or julia: from its environment variable if set, else the R
    environment of benchmarks/README.md (benchmarks/external/.r), else PATH.
    None if not installed."""
    _language, name, variable, _script = EXTERNAL[kind]
    if os.environ.get(variable):
        return os.environ[variable]
    local = EXTERNAL_DIR / ".r" / "bin" / name
    if local.exists():
        return str(local)
    return shutil.which(name)


def external_command(kind, **spec):
    """The command that runs one fit with fixest ("fixest") or
    FixedEffectModels.jl ("fem"), or None if that language is not installed.

    The fit is described by `spec`: path, outcome, covariates (numeric
    columns), categorical ({"column", "ref"} for a set of indicators, or None),
    fixed_effects, cluster, drop_singletons, columns (to read, or None for all)
    and, for Julia, warmup_path (a small panel of the same columns, fitted
    first so that compilation is not timed). Both use every logical CPU, as
    the Python libraries do: fixest's own default is half of them, and
    Julia's is one thread.

    FixedEffectModels.jl is also given a tolerance of 1e-8. At its default,
    1e-6, its coefficients on the covariates benchmark are up to 1e-4 (relative)
    from every other library's, which agree to about 1e-8 among themselves; at
    1e-8 it agrees with them as closely, so its times are for equally good
    answers, at about 1.5 times its default time."""
    executable = external_executable(kind)
    if executable is None:
        return None
    spec = {"threads": os.cpu_count(), **({"tol": 1e-8} if kind == "fem" else {}),
            **spec}
    for key in ("path", "warmup_path"):
        if key in spec:
            spec[key] = str(spec[key])
    script = str(EXTERNAL_DIR / EXTERNAL[kind][3])
    if kind == "fem":
        return [executable, f"--threads={spec['threads']}",
                f"--project={EXTERNAL_DIR / 'julia'}", script, json.dumps(spec)]
    return [executable, script, json.dumps(spec)]


def external_missing(kind):
    language, name, variable, _script = EXTERNAL[kind]
    return (f"skipped: {language} is not installed (no {name} on PATH; or set "
            f"{variable})")


def external_versions():
    """R, fixest, Julia and FixedEffectModels.jl, where installed."""
    out = {}
    commands = {
        "fixest": ['cat(paste(R.version$major, R.version$minor, sep = "."), '
                   'as.character(packageVersion("fixest")))'],
        "fem": ['import Pkg; print(VERSION, " ", Pkg.dependencies()[Base.UUID('
                '"9d5cd8c9-2029-5cab-9928-427838db53e3")].version)'],
    }
    names = {"fixest": ("R", "fixest"), "fem": ("julia", "FixedEffectModels.jl")}
    for kind, (code,) in commands.items():
        executable = external_executable(kind)
        if executable is None:
            continue
        command = ([executable, "-e", code] if kind == "fixest" else
                   [executable, "--startup-file=no",
                    f"--project={EXTERNAL_DIR / 'julia'}", "-e", code])
        try:
            words = subprocess.run(command, capture_output=True, text=True,
                                   timeout=300).stdout.split()
        except (OSError, subprocess.SubprocessError):
            continue
        if len(words) == 2:
            out.update(zip(names[kind], words))
    return out


LEDGER_ENV = "HDFE_BENCH_LEDGER"

# a row with one of these statuses is a result, to be kept rather than
# measured again; a failure may be the process being stopped, so it is not
_FINAL = ("ok", "timeout", "killed (out of memory)", "out of memory")


def _digest(row):
    """Identify a row by its non-empty values, as text, so that the row as
    written and as read back from the CSV give the same digest."""
    items = sorted((k, str(v)) for k, v in row.items() if v not in (None, ""))
    return hashlib.sha1(json.dumps(items).encode()).hexdigest()


def _ledger():
    name = os.environ.get(LEDGER_ENV)
    if not name:
        return None, {}
    path = Path(name)
    try:
        return path, json.loads(path.read_text())
    except (OSError, ValueError):
        return path, {}


def _entry(path, row, key):
    return Path(path).name, "|".join(str(row[k]) for k in key)


def _atomic_write(path, write):
    """Write through `write(handle)` to a temporary file, then rename it, so
    that a process stopped part-way never leaves half a file."""
    path = Path(path)
    temporary = path.with_name(path.name + ".part")
    with open(temporary, "w", newline="") as handle:
        write(handle)
    os.replace(temporary, path)


def is_done(path, values, key=("n_workers", "configuration")):
    """Whether the configuration with these key `values` (a dict) was already
    measured by this run: its row is in the CSV at `path`, unchanged since it
    was written, with a final status. Always False unless the ledger is on."""
    ledger_path, ledger = _ledger()
    if ledger_path is None or not Path(path).exists():
        return False
    name, entry = _entry(path, values, key)
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            if "|".join(str(row[k]) for k in key) == entry:
                return (row["status"].startswith(_FINAL)
                        and ledger.get(name, {}).get(entry) == _digest(row))
    return False


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
    merged.sort(key=lambda r: (int(r[key[0]]), int(r.get("order") or 0)))

    def write(handle):
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(merged)

    _atomic_write(path, write)
    ledger_path, ledger = _ledger()
    if ledger_path is not None:
        for r in rows:
            name, entry = _entry(path, r, key)
            ledger.setdefault(name, {})[entry] = _digest(r)
        _atomic_write(ledger_path, lambda handle: handle.write(json.dumps(ledger)))


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
                "scikit-learn", "polars-ols", "polars-ds", "statsmodels", "linearmodels",
                "glum",
                "scipy", "pandas")},
            **external_versions(),
            "measured": time.strftime("%Y-%m-%d")}


def write_machine_info(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(machine_info(), indent=1) + "\n")

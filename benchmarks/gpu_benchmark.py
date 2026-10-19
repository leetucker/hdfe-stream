"""Experimental: the AKM regression with and without a GPU, on one machine.

    python benchmarks/gpu_benchmark.py                  # the default sizes
    python benchmarks/gpu_benchmark.py 100000 1000000   # pick your own

The job is akm_benchmark.py's,

    log_earn ~ age_squared + age_cubed | worker_id + firm_id + year

clustered by worker, on the same simulated panels. Seven configurations:

  * hdfe_stream on the CPU, as everywhere else (its default solver);
  * hdfe_stream with the solve on the GPU (``device="cuda"``, through CuPy);
  * the same, with the fit's Polars queries on the GPU as well
    (``polars_engine="gpu"``, cudf-polars);
  * pyfixest with its LSMR demeaner on its default backend (``within``, on
    the CPU), as "pyfixest (LSMR)" in akm_benchmark.py;
  * pyfixest with its LSMR demeaner on PyTorch on the GPU
    (``LsmrDemeaner(backend="torch", device="cuda")``, in float64, its
    default precision). pyfixest falls back to a CPU demeaner without a word
    when torch cannot be imported, so the result is kept only if torch sees
    the GPU and the fit allocated memory on it;
  * xhdfe on the CPU, its default;
  * xhdfe with its CUDA backend (``XHDFE_GPU_BACKEND=cuda``), which needs a
    CUDA build of xhdfe. Its result is kept only if xhdfe reports that the
    GPU was used (``gpu_used_``), since it returns a CPU result otherwise.

On a machine without an NVIDIA GPU the script says so and exits without
measuring anything. The other benchmarks are not repeated: each library on
the CPU is the reference point for itself on the GPU, on this machine, and
pyfixest's MAP demeaner, which has no GPU version, is in akm_benchmark.py.

Measured as in the other benchmarks (benchmarks/_harness.py), plus the peak
GPU memory of the configuration's process (polled from nvidia-smi) and, for
hdfe_stream, the seconds spent in the solve and any Polars query that the GPU
engine could not run and ran on the CPU instead. Results go to
benchmarks/results/gpu.csv, the machine to benchmarks/results/gpu_machine.json.
Every configuration starts a fresh process, so the GPU ones pay for creating
a CUDA context (well under a second) each time, as a user's script would.
What is compiled once and cached on disk is not charged to them: numba's
kernels, CuPy's and cudf's, and pyfixest's torch.compile of LSMR are warmed
up on the smallest panel first.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import threading
import time
import warnings
from pathlib import Path

from _harness import (
    DATA_DIR,
    RESULTS_DIR,
    data_path,
    is_done,
    machine_info,
    measure,
    panel_shape,
    run_child,
    upsert_csv,
)

SCRIPT = Path(__file__).resolve()
FML = "log_earn ~ age_squared + age_cubed | worker_id + firm_id + year"
COEFS = ("age_squared", "age_cubed")
CSV = RESULTS_DIR / "gpu.csv"

DEFAULT_SIZES = [25_000, 100_000, 400_000, 1_000_000, 5_000_000]

CONFIGS = {
    "hdfe_stream (CPU)": ("hdfe", "cpu"),
    "hdfe_stream (GPU solve)": ("hdfe", "cuda"),
    "hdfe_stream (GPU solve + GPU Polars)": ("hdfe", "cuda+polars"),
    "pyfixest (LSMR, CPU)": ("pyfixest", "cpu"),
    "pyfixest (LSMR, GPU)": ("pyfixest", "cuda"),
    "xhdfe (CPU)": ("xhdfe", "cpu"),
    "xhdfe (GPU)": ("xhdfe", "cuda"),
}

# a GPU instance is dear: anything slower than this is recorded as a timeout
TIMEOUT_S = 30 * 60


# --------------------------------------------------------------------------
# GPU detection and memory
# --------------------------------------------------------------------------

def nvidia_smi(*query):
    """nvidia-smi's CSV output as rows of fields, or None without a GPU."""
    if shutil.which("nvidia-smi") is None:
        return None
    proc = subprocess.run(["nvidia-smi", *query, "--format=csv,noheader,nounits"],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        return None
    return [[f.strip() for f in line.split(",")] for line in proc.stdout.splitlines()
            if line.strip()]


def gpus():
    return nvidia_smi("--query-gpu=name,memory.total,driver_version,compute_cap") or []


class GpuMemoryPoller:
    """The most GPU memory this process held, sampled from nvidia-smi."""

    def __init__(self, interval=0.25):
        self.interval, self.peak_mb = interval, 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _sample(self):
        pid = str(os.getpid())
        for row in nvidia_smi("--query-compute-apps=pid,used_memory") or []:
            if row[0] == pid and row[1].replace(".", "", 1).isdigit():
                self.peak_mb = max(self.peak_mb, float(row[1]))

    def _run(self):
        while not self._stop.is_set():
            self._sample()
            self._stop.wait(self.interval)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()


# --------------------------------------------------------------------------
# the child process: one configuration, one size
# --------------------------------------------------------------------------

def run_pyfixest(variant, path):
    import pandas as pd
    import pyfixest as pf

    if variant == "cuda":
        try:
            import torch
        except ImportError as err:
            raise RuntimeError("torch is not installed, and pyfixest would quietly "
                               "demean on the CPU instead") from err
        if not torch.cuda.is_available():
            raise RuntimeError(f"torch {torch.__version__} sees no CUDA device "
                               "(a CPU-only build?)")
        demeaner = pf.LsmrDemeaner(backend="torch", device="cuda")
    else:
        demeaner = pf.LsmrDemeaner()
    t0 = time.perf_counter()
    frame = pd.read_parquet(path)
    t_read = time.perf_counter() - t0
    # pyfixest reports what it changed by itself (torch.compile falling back
    # to eager execution, a CPU fallback) only as warnings: keep them
    with GpuMemoryPoller() as gpu, warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        t0 = time.perf_counter()
        fit = pf.feols(FML, data=frame, vcov={"CRV1": "worker_id"},
                       fixef_rm="none", demeaner=demeaner)
        t_fit = time.perf_counter() - t0
    if variant == "cuda" and torch.cuda.max_memory_allocated() == 0:
        raise RuntimeError("pyfixest allocated nothing on the GPU")
    coefs = fit.coef().to_dict()
    notes = sorted({str(w.message).strip().splitlines()[0][:160] for w in caught
                    if not issubclass(w.category, (DeprecationWarning, FutureWarning))})
    return {"t_read": t_read, "t_fit": t_fit, "n_obs": int(fit._N),
            "coefs": {k: float(coefs[k]) for k in COEFS}, "gpu_peak_mb": gpu.peak_mb,
            "notes": "; ".join(notes)}


def run_xhdfe(variant, path):
    if variant == "cuda":
        os.environ["XHDFE_GPU_BACKEND"] = "cuda"
    import pandas as pd
    import xhdfe

    t0 = time.perf_counter()
    frame = pd.read_parquet(path)
    t_read = time.perf_counter() - t0
    with GpuMemoryPoller() as gpu:
        t0 = time.perf_counter()
        fit = xhdfe.feols(FML, data=frame, se_type="cluster", clusters="worker_id")
        t_fit = time.perf_counter() - t0
    used = int(getattr(fit, "gpu_used_", 0) or 0)
    if variant == "cuda" and used != 1:
        raise RuntimeError(f"xhdfe did not use the GPU (gpu_used_={used}, "
                           f"gpu_status_code_={getattr(fit, 'gpu_status_code_', '?')}); "
                           "is it a CUDA build?")
    coefs = dict(zip(fit.coef_names_, fit.coef_))
    return {"t_read": t_read, "t_fit": t_fit, "n_obs": int(len(frame)),
            "coefs": {k: float(coefs[k]) for k in COEFS},
            "gpu_peak_mb": gpu.peak_mb, "gpu_used": used}


def run_hdfe(variant, path, workdir):
    from hdfe_stream import feols_stream

    options = {}
    if variant.startswith("cuda"):
        options["device"] = "cuda"
    if variant.endswith("+polars"):
        options["polars_engine"] = "gpu"
    with GpuMemoryPoller() as gpu:
        t0 = time.perf_counter()
        fit = feols_stream(FML, str(path), workdir=str(workdir),
                           vcov={"CRV1": "worker_id"}, fe_dof="pyfixest",
                           verbose=False, save_resid=False, **options)
        t_fit = time.perf_counter() - t0
    coefs = fit.coef()
    fallbacks = fit.diagnostics.get("polars_gpu_fallbacks", [])
    out = {"t_read": 0.0, "t_fit": t_fit, "n_obs": int(fit.n_obs),
           "coefs": {k: float(coefs[k]) for k in COEFS},
           "reported_disk_mb": fit.diagnostics["disk_peak_gb"] * 1000,
           "gpu_peak_mb": gpu.peak_mb,
           "solve_s": fit.solver_info.get("seconds"),
           "solver": fit.solver_info.get("solver"),
           "device": fit.solver_info.get("device", "cpu"),
           "polars_cpu_fallbacks": ";".join(sorted({f["query"] for f in fallbacks})),
           "polars_cpu_fallback_reasons": ";".join(sorted({f["reason"] for f in fallbacks}))}
    fit.cleanup()
    return out


def runner(kind, variant, path, workdir):
    if kind == "xhdfe":
        return run_xhdfe(variant, path)
    if kind == "pyfixest":
        return run_pyfixest(variant, path)
    return run_hdfe(variant, path, workdir)


# --------------------------------------------------------------------------
# the parent process
# --------------------------------------------------------------------------

def gpu_machine_info(found):
    import importlib.metadata as md

    def version(name):
        try:
            return md.version(name)
        except md.PackageNotFoundError:
            return None

    info = machine_info()
    info["gpus"] = [dict(zip(("name", "memory_mib", "driver", "compute_capability"), g))
                    for g in found]
    info.update({name: version(name) for name in
                 ("cupy-cuda13x", "cupy-cuda12x", "cudf-polars-cu13", "cudf-polars-cu12",
                  "torch")})
    return info


def check_agreement(reports):
    ok = [r for r in reports.values() if r["status"] == "ok"]
    if len(ok) < 2:
        return 0.0
    base = ok[0]["coefs"]
    return max(abs(r["coefs"][k] - base[k]) / max(abs(base[k]), 1e-12)
               for r in ok[1:] for k in COEFS)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sizes", nargs="*", type=int, default=DEFAULT_SIZES,
                        help="number of workers in each panel")
    parser.add_argument("--only", nargs="+", metavar="CONFIG",
                        help="run only these configurations (by label)")
    parser.add_argument("--child", nargs=4,
                        metavar=("KIND", "VARIANT", "PATH", "WORKDIR"),
                        help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.child:
        run_child(runner, *args.child)
        return

    found = gpus()
    if not found:
        print("no NVIDIA GPU found (nvidia-smi is missing or lists none); "
              "the GPU benchmark is skipped")
        return
    print("GPU: " + "; ".join(f"{g[0]}, {int(float(g[1])) / 1024:.0f} GiB, driver {g[2]}"
                              for g in found), flush=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "gpu_machine.json").write_text(
        json.dumps(gpu_machine_info(found), indent=1) + "\n")

    configs = {k: v for k, v in CONFIGS.items() if not args.only or k in args.only}
    workdir = DATA_DIR / "work"
    warmed = set()
    for n_workers in args.sizes:
        path = data_path(n_workers)
        shape = panel_shape(path)
        print(f"\n=== {n_workers:,} workers, {shape['firms']:,} firms, "
              f"{shape['rows']:,} rows ===", flush=True)
        todo = [label for label in configs
                if not is_done(CSV, {"n_workers": n_workers, "configuration": label})]
        # warm up, once, what each pending configuration compiles and caches
        # on disk (numba's kernels; CuPy's and cudf's; pyfixest's
        # torch.compile), so that it is not charged to whichever size runs
        # first; xhdfe is compiled ahead of time
        for label in todo:
            kind, variant = configs[label]
            if kind != "xhdfe" and (kind, variant) not in warmed:
                measure(SCRIPT, kind, variant, data_path(25_000), workdir)
                warmed.add((kind, variant))

        reports = {}
        for label, (kind, variant) in configs.items():
            order = list(CONFIGS).index(label)
            if label not in todo:
                print(f"  {label:<40}measured before, kept", flush=True)
                continue
            report = measure(SCRIPT, kind, variant, path, workdir, timeout=TIMEOUT_S)
            reports[label] = report
            row = {"n_workers": n_workers, "n_firms": shape["firms"],
                   "n_rows": shape["rows"], "configuration": label,
                   "order": order, "status": report["status"]}
            if report["status"] == "ok":
                row.update(
                    wall_s=round(report["t_read"] + report["t_fit"], 3),
                    read_s=round(report["t_read"], 3),
                    fit_s=round(report["t_fit"], 3),
                    solve_s=report.get("solve_s"),
                    peak_memory_mb=round(report["peak_rss_mb"], 1),
                    import_memory_mb=round(report["baseline_rss_mb"], 1),
                    peak_gpu_memory_mb=round(report["gpu_peak_mb"], 1),
                    peak_disk_mb=round(report["peak_disk_mb"], 1),
                    solver=report.get("solver"), device=report.get("device"),
                    polars_cpu_fallbacks=report.get("polars_cpu_fallbacks"),
                    polars_cpu_fallback_reasons=report.get("polars_cpu_fallback_reasons"),
                    notes=report.get("notes") or None,
                    **{f"coef_{k}": repr(report["coefs"][k]) for k in COEFS})
                solve = (f"   solve {row['solve_s']:6.2f} s"
                         if row["solve_s"] is not None else "")
                print(f"  {label:<40}{row['wall_s']:8.1f} s   peak "
                      f"{row['peak_memory_mb']:8,.0f} MB   GPU "
                      f"{row['peak_gpu_memory_mb']:7,.0f} MB{solve}", flush=True)
                if row["polars_cpu_fallbacks"]:
                    print(f"  {'':<40}on the CPU: {row['polars_cpu_fallbacks']}", flush=True)
                if row["notes"]:
                    print(f"  {'':<40}warned: {row['notes']}", flush=True)
            else:
                print(f"  {label:<40}{report['status']}", flush=True)
            upsert_csv(CSV, [row])

        worst = check_agreement(reports)
        verdict = "agree" if worst < 1e-6 else "DISAGREE"
        print(f"  estimates {verdict}: largest relative difference {worst:.1e}",
              flush=True)


if __name__ == "__main__":
    main()

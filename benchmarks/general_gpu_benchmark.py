"""Experimental: OLS, IV and logit regressions with and without a GPU, on one
machine. The GPU counterpart of ols_benchmark.py, iv_benchmark.py and
binary_benchmark.py, as gpu_benchmark.py is of akm_benchmark.py.

    python benchmarks/general_gpu_benchmark.py                   # the default sizes
    python benchmarks/general_gpu_benchmark.py 100000 1000000    # pick your own

The data and the models, with no fixed effects, are those of the CPU benchmarks
(`_harness.general_data_path`):

    y   ~ x0 + ... + x9                  OLS
    yiv ~ x0 + ... + x9 | d ~ z1 + z2    two-stage least squares
    yb  ~ x0 + ... + x9                  logit

Per model:

  * hdfe_stream on the CPU, as in the other benchmarks;
  * hdfe_stream with the fit's Polars queries on the GPU
    (``polars_engine="gpu"``, cudf-polars). Linear models have nothing else
    to move: their only solve is a k x k system, and the time is in reading
    and writing the rows, which is what the Polars engine does;
  * for logit, hdfe_stream with every IRLS step on the GPU (``device="cuda"``:
    the rows are uploaded once and stay there) and with both;
  * the fastest CPU implementation of the CPU benchmarks, on this machine's
    (slower) CPU, as the reference: polars-ols for OLS and IV, polars-ds for
    logit;
  * cuML's LinearRegression and LogisticRegression, which fit on the GPU
    (OLS and logit; cuML has no 2SLS). They run in a virtual environment of
    their own, since cuML pins a stack that would disturb the others
    (`cuml_python`);
  * the normal equations (OLS, 2SLS) and Newton's method (logit), written in
    CuPy: about the least work the GPU can be given.

A GPU configuration is kept only if the process held GPU memory, since each of
these libraries can quietly fall back to the CPU. Measured as in the other
benchmarks, from the Parquet file to the estimates (the GPU libraries' time
includes copying the data to the GPU), plus the peak GPU memory of the
process. Results go to benchmarks/results/general_gpu.csv, the machine to
benchmarks/results/gpu_machine.json. On a machine without an NVIDIA GPU the
script says so and exits without measuring anything.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from _harness import (DATA_DIR, RESULTS_DIR, general_data_path, is_done, measure,
                      run_child, upsert_csv)
from _regression import (KEYS, OUTCOME, RUNNERS, TOLERANCE, agreement, covariates,
                         packed, stored_coefs, variables)
from gpu_benchmark import GpuMemoryPoller, gpu_machine_info, gpus

SCRIPT = Path(__file__).resolve()
CSV = RESULTS_DIR / "general_gpu.csv"

DEFAULT_SIZES = [100_000, 1_000_000, 5_000_000, 20_000_000]

# label -> (model, library, variant)
CONFIGS = {
    "hdfe_stream OLS (CPU)": ("ols", "hdfe", "cpu"),
    "hdfe_stream OLS (GPU Polars)": ("ols", "hdfe", "polars"),
    "polars-ols (CPU)": ("ols", "polars_ols", "cpu"),
    "cuML OLS": ("ols", "cuml", "gpu"),
    "CuPy normal equations (OLS)": ("ols", "cupy", "gpu"),
    "hdfe_stream IV (CPU)": ("iv", "hdfe", "cpu"),
    "hdfe_stream IV (GPU Polars)": ("iv", "hdfe", "polars"),
    "polars-ols two stages (CPU)": ("iv", "polars_ols", "cpu"),
    "CuPy 2SLS": ("iv", "cupy", "gpu"),
    "hdfe_stream logit (CPU)": ("logit", "hdfe", "cpu"),
    "hdfe_stream logit (GPU IRLS)": ("logit", "hdfe", "cuda"),
    "hdfe_stream logit (GPU Polars)": ("logit", "hdfe", "polars"),
    "hdfe_stream logit (GPU IRLS + GPU Polars)": ("logit", "hdfe", "cuda+polars"),
    "polars-ds logit (CPU)": ("logit", "polars_ds", "cpu"),
    "cuML logit": ("logit", "cuml", "gpu"),
    "CuPy Newton logit": ("logit", "cupy", "gpu"),
}

# a GPU instance is dear: anything slower than this is recorded as a timeout
TIMEOUT_S = 30 * 60


def cuml_python():
    """The interpreter of the virtual environment with cuML (remote/setup_gpu.sh
    builds it), from HDFE_BENCH_CUML_PYTHON or its usual place; None if absent."""
    for candidate in (os.environ.get("HDFE_BENCH_CUML_PYTHON"),
                      "/opt/bench/cuml-venv/bin/python"):
        if candidate and Path(candidate).exists():
            return candidate
    return None


# --------------------------------------------------------------------------
# the child process: one configuration, one size
# --------------------------------------------------------------------------

def run_hdfe(task, variant, path, workdir):
    from hdfe_stream import feglm_stream, feols_stream

    xs = covariates(path)
    rhs = " + ".join(xs)
    options = dict(workdir=str(workdir), vcov="iid", verbose=False, save_resid=False)
    if variant.startswith("cuda"):
        options["device"] = "cuda"
    if variant.endswith("polars"):
        options["polars_engine"] = "gpu"
    with GpuMemoryPoller() as gpu:
        t0 = time.perf_counter()
        if task == "ols":
            fit = feols_stream(f"y ~ {rhs}", str(path), **options)
        elif task == "iv":
            fit = feols_stream(f"yiv ~ {rhs} | d ~ z1 + z2", str(path), **options)
        else:
            fit = feglm_stream(f"yb ~ {rhs}", str(path), task, **options)
        t_fit = time.perf_counter() - t0
    if variant != "cpu" and gpu.peak_mb == 0:
        raise RuntimeError("the fit held no GPU memory")
    coefs = fit.coef()
    fallbacks = fit.diagnostics.get("polars_gpu_fallbacks", [])
    out = packed(0.0, t_fit, fit.n_obs, list(coefs), list(coefs.values()), list(fit.se))
    out.update(gpu_peak_mb=gpu.peak_mb,
               polars_cpu_fallbacks=";".join(sorted({f["query"] for f in fallbacks})))
    fit.cleanup()
    return out


def load_gpu(task, path):
    """The data as CuPy arrays: X (with a constant), y, and for IV the
    endogenous column and the instruments."""
    import cupy as cp
    import pandas as pd

    xs = covariates(path)
    frame = pd.read_parquet(path, columns=variables(task, 0, xs))
    X = cp.asarray(frame[xs].to_numpy())
    y = cp.asarray(frame[OUTCOME[task]].to_numpy())
    extra = (cp.asarray(frame[["d", "z1", "z2"]].to_numpy()),) if task == "iv" else ()
    cp.cuda.Stream.null.synchronize()
    return xs, X, y, extra


def run_cuml(task, path):
    import cupy as cp

    t0 = time.perf_counter()
    xs, X, y, _ = load_gpu(task, path)
    t_read = time.perf_counter() - t0
    with GpuMemoryPoller() as gpu:
        t0 = time.perf_counter()
        if task == "ols":
            from cuml.linear_model import LinearRegression
            fit = LinearRegression(algorithm="eig").fit(X, y)
        else:
            from cuml.linear_model import LogisticRegression
            fit = LogisticRegression(penalty=None, tol=1e-10, max_iter=1000).fit(X, y)
        coef = cp.asnumpy(cp.asarray(fit.coef_)).ravel()
        intercept = float(cp.asnumpy(cp.asarray(fit.intercept_)).ravel()[0])
        t_fit = time.perf_counter() - t0
    if gpu.peak_mb == 0:
        raise RuntimeError("cuML held no GPU memory")
    out = packed(t_read, t_fit, len(y), [*xs, "Intercept"], [*coef, intercept])
    out["gpu_peak_mb"] = gpu.peak_mb
    return out


def run_cupy(task, path):
    """The normal equations, 2SLS and Newton's method for logit, in CuPy."""
    import cupy as cp

    t0 = time.perf_counter()
    xs, X, y, extra = load_gpu(task, path)
    t_read = time.perf_counter() - t0
    with GpuMemoryPoller() as gpu:
        t0 = time.perf_counter()
        X1 = cp.concatenate([cp.ones((len(y), 1)), X], axis=1)
        if task == "ols":
            bread = cp.linalg.inv(X1.T @ X1)
            beta = bread @ (X1.T @ y)
            resid = y - X1 @ beta
            names = ["Intercept", *xs]
        elif task == "iv":
            W = cp.concatenate([X1, extra[0][:, :1]], axis=1)
            Z = cp.concatenate([X1, extra[0][:, 1:]], axis=1)
            fitted = Z @ cp.linalg.solve(Z.T @ Z, Z.T @ W)
            bread = cp.linalg.inv(fitted.T @ fitted)
            beta = bread @ (fitted.T @ y)
            resid = y - W @ beta
            names = ["Intercept", *xs, "d"]
        else:
            beta = cp.zeros(X1.shape[1])
            for _ in range(50):
                p = 1 / (1 + cp.exp(-(X1 @ beta)))
                w = p * (1 - p)
                info = X1.T @ (X1 * w[:, None])
                step = cp.linalg.solve(info, X1.T @ (y - p))
                beta = beta + step
                if float(cp.abs(step).max()) < 1e-10:
                    break
            bread = cp.linalg.inv(info)
            names = ["Intercept", *xs]
        if task == "logit":
            ses = cp.sqrt(cp.diag(bread))
        else:
            ses = cp.sqrt(cp.diag(bread) * (resid @ resid) / (len(y) - len(beta)))
        beta, ses = cp.asnumpy(beta), cp.asnumpy(ses)
        t_fit = time.perf_counter() - t0
    out = packed(t_read, t_fit, len(y), names, beta, ses)
    out["gpu_peak_mb"] = gpu.peak_mb
    return out


def runner(kind, variant, path, workdir):
    task, library = kind.split(":")
    if library == "hdfe":
        return run_hdfe(task, variant, path, workdir)
    if library == "cuml":
        return run_cuml(task, path)
    if library == "cupy":
        return run_cupy(task, path)
    out = RUNNERS[library](task, 0, path)         # the CPU reference
    out["gpu_peak_mb"] = 0.0
    return out


# --------------------------------------------------------------------------
# the parent process
# --------------------------------------------------------------------------

def cuml_version(python):
    if python is None:
        return None
    proc = subprocess.run([python, "-c", "import cuml; print(cuml.__version__)"],
                          capture_output=True, text=True)
    return proc.stdout.strip() or None


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sizes", nargs="*", type=int, default=DEFAULT_SIZES,
                        help="number of rows in each data set")
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
    python = cuml_python()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    info = gpu_machine_info(found)
    info["cuml"] = cuml_version(python)
    (RESULTS_DIR / "gpu_machine.json").write_text(json.dumps(info, indent=1) + "\n")

    key = ("n_rows", "configuration")
    configs = {k: v for k, v in CONFIGS.items() if not args.only or k in args.only}
    workdir = DATA_DIR / "work_general_gpu"

    def command(library, kind, variant, path):
        """cuML runs in its own environment; everything else in this one."""
        if library != "cuml":
            return None
        return [python, str(SCRIPT), "--child", kind, variant, str(path), str(workdir)]

    warmed = set()
    for n_rows in args.sizes:
        path = general_data_path(n_rows)
        small = general_data_path(100_000)
        print(f"\n=== {n_rows:,} rows, 10 covariates ===", flush=True)
        todo = [label for label in configs
                if not is_done(CSV, {"n_rows": n_rows, "configuration": label}, key=key)]
        # warm up, once, what each pending configuration compiles and caches on
        # disk (numba's kernels; CuPy's and cudf's), so that it is not charged
        # to whichever size runs first
        for label in todo:
            task, library, variant = configs[label]
            if (task, library, variant) in warmed or (library == "cuml" and not python):
                continue
            measure(SCRIPT, f"{task}:{library}", variant, small, workdir,
                    command=command(library, f"{task}:{library}", variant, small))
            warmed.add((task, library, variant))

        reports = {}
        for order, (label, (task, library, variant)) in enumerate(CONFIGS.items()):
            if label not in configs:
                continue
            if label not in todo:
                print(f"  {label:<44}measured before, kept", flush=True)
                continue
            if library == "cuml" and not python:
                print(f"  {label:<44}skipped: no cuML environment "
                      "(HDFE_BENCH_CUML_PYTHON)", flush=True)
                continue
            kind = f"{task}:{library}"
            report = measure(SCRIPT, kind, variant, path, workdir, timeout=TIMEOUT_S,
                             command=command(library, kind, variant, path))
            reports[label] = report
            row = {"n_rows": n_rows, "n_covariates": 10, "task": task,
                   "configuration": label, "order": order, "status": report["status"]}
            if report["status"] == "ok":
                row.update(
                    wall_s=round(report["t_read"] + report["t_fit"], 3),
                    read_s=round(report["t_read"], 3),
                    fit_s=round(report["t_fit"], 3),
                    peak_memory_mb=round(report["peak_rss_mb"], 1),
                    import_memory_mb=round(report["baseline_rss_mb"], 1),
                    peak_gpu_memory_mb=round(report["gpu_peak_mb"], 1),
                    polars_cpu_fallbacks=report.get("polars_cpu_fallbacks") or None,
                    n_obs=report["n_obs"],
                    **{f"coef_{k}": repr(report["coefs"][k]) for k in KEYS[task]})
                print(f"  {label:<44}{row['wall_s']:8.2f} s   peak "
                      f"{row['peak_memory_mb']:8,.0f} MB   GPU "
                      f"{row['peak_gpu_memory_mb']:7,.0f} MB", flush=True)
                if row["polars_cpu_fallbacks"]:
                    print(f"  {'':<44}on the CPU: {row['polars_cpu_fallbacks']}", flush=True)
            else:
                print(f"  {label:<44}{report['status']}", flush=True)
            upsert_csv(CSV, [row], key=key)

        for task in sorted({c[0] for c in configs.values()}):
            same = {label for label, c in CONFIGS.items() if c[0] == task}
            group = {label: r for label, r in reports.items() if label in same}
            worst = agreement(task, group, stored_coefs(
                CSV, task, n_rows, set(group) | (set(CONFIGS) - same)))
            verdict = "agree" if worst < TOLERANCE[task] else "DISAGREE"
            print(f"  {task}: estimates {verdict}: largest relative difference "
                  f"{worst:.1e}", flush=True)


if __name__ == "__main__":
    main()

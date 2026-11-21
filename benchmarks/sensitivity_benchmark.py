"""Time and memory of a sensitivity analysis for parallel trends
(`core.sensitivity.trend_sensitivity`) against R's HonestDiD 0.2.8 and the Python
package diff-diff 4.0.0, on simulated event studies with 4 to 32 pre-treatment and 4
post-treatment periods, for one value of M.

    python benchmarks/sensitivity_benchmark.py

Each run is a fresh process; the time excludes loading the package (and, here, a
warm-up call that loads the compiled kernels from numba's cache), and memory is the
process's peak resident set size, with its value after loading as the baseline.

R is `HDFE_BENCH_RSCRIPT` (or Rscript on PATH) with HonestDiD installed; diff-diff is
the Python of `HDFE_BENCH_DIFF_DIFF_PYTHON`. diff-diff's relative-magnitudes interval is
not the method's: it widens the identified set by z standard errors (its conditional
test is disabled), so its time is not comparable for that restriction, and it is shown
for the FLCI only.

Writes benchmarks/results/sensitivity.csv and docs/benchmarks/sensitivity_table.md.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SHAPES = [(4, 4), (8, 4), (16, 4), (32, 4)]
JOBS = {
    "relative_magnitudes": "relative magnitudes, C-LF",
    "smoothness_relative_magnitudes": "smoothness and relative magnitudes, C-LF",
    "smoothness": "smoothness, FLCI",
    "smoothness_hybrid": "smoothness, C-F",
}

WORKER = r"""
import json, resource, sys, time
import numpy as np
from hdfe_stream.core.sensitivity import trend_sensitivity
es = json.load(open(sys.argv[1])); job = sys.argv[2]; M = float(sys.argv[3]); threads = int(sys.argv[4])
b, V, n_pre = np.array(es["beta"]), np.array(es["sigma"]), es["n_pre"]
kw = {"relative_magnitudes": dict(restriction="relative_magnitudes"),
      "smoothness_relative_magnitudes": dict(restriction="smoothness_relative_magnitudes"),
      "smoothness": dict(restriction="smoothness"),
      "smoothness_hybrid": dict(restriction="smoothness", ci_method="hybrid_fixed_length")}[job]
trend_sensitivity(b[2:6], V[2:6, 2:6], 2, M=[M], n_threads=threads, **kw)  # warm-up
base = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
t = time.perf_counter()
r = trend_sensitivity(b, V, n_pre, M=[M], n_threads=threads, **kw)
s = time.perf_counter() - t
lo, hi = r.table.row(0)[1:]
print("RESULT", json.dumps(dict(seconds=s, peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                                baseline_rss_mb=base, lb=lo, ub=hi)))
"""

DIFF_DIFF = r"""
import json, resource, sys, time
import numpy as np
from diff_diff.honest_did import HonestDiD
es = json.load(open(sys.argv[1])); job = sys.argv[2]; M = float(sys.argv[3])
b, V, n_pre = np.array(es["beta"]), np.array(es["sigma"]), es["n_pre"]
n_post = len(b) - n_pre
l = np.zeros(n_post); l[0] = 1.0
h = HonestDiD(method="smoothness", M=M)
base = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
t = time.perf_counter()
lb, ub, lo, hi = h._compute_smoothness_bounds(b[:n_pre], b[n_pre:], V, V[n_pre:, n_pre:], l, n_pre, n_post, M)
s = time.perf_counter() - t
print("RESULT", json.dumps(dict(seconds=s, peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                                baseline_rss_mb=base, lb=lo, ub=hi)))
"""


def event_study(n_pre, n_post, seed):
    """A simulated event study like the R reference's: correlated coefficients
    with a small pre-trend and an effect of about three standard errors."""
    rng = np.random.default_rng(seed)
    n = n_pre + n_post
    F = rng.normal(size=(n, 2))
    C = F @ F.T + np.eye(n)
    d = np.sqrt(np.diag(C))
    se = 0.05 * np.exp(rng.normal(scale=0.3, size=n))
    sigma = (C / np.outer(d, d)) * np.outer(se, se)
    times = np.concatenate([-np.arange(n_pre, 0, -1), np.arange(1, n_post + 1)])
    beta = 0.005 * times + np.concatenate([np.zeros(n_pre), 0.15 + 0.02 * np.arange(1, n_post + 1)])
    beta = beta + np.linalg.cholesky(sigma) @ rng.normal(size=n)
    return {"beta": beta.tolist(), "sigma": sigma.tolist(), "n_pre": n_pre}


def run(cmd):
    out = subprocess.run(cmd, capture_output=True, text=True, env=dict(os.environ, PYTHONWARNINGS="ignore"))
    for line in out.stdout.splitlines():
        if line.startswith("RESULT"):
            return json.loads(line[6:])
    raise RuntimeError(f"{cmd[0]} failed:\n{out.stdout[-2000:]}\n{out.stderr[-2000:]}")


def main():
    from hdfe_stream.core.sensitivity import default_M
    rscript = os.environ.get("HDFE_BENCH_RSCRIPT", "Rscript")
    ddpy = os.environ.get("HDFE_BENCH_DIFF_DIFF_PYTHON")
    threads = os.cpu_count() or 1
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for n_pre, n_post in SHAPES:
            es = event_study(n_pre, n_post, 1000 + n_pre)
            path = Path(tmp) / f"es_{n_pre}_{n_post}.json"
            path.write_text(json.dumps(es))
            m_sd = float(default_M(np.array(es["beta"]), np.array(es["sigma"]), n_pre, "smoothness")[-1]) / 2
            for job in JOBS:
                M = 1.0 if "relative" in job else m_sd
                row = dict(n_pre=n_pre, n_post=n_post, job=job, M=M)
                row["r"] = run([rscript, str(HERE / "external" / "honestdid_benchmark.R"), str(path), job, str(M)])
                row["ours_1"] = run([sys.executable, "-c", WORKER, str(path), job, str(M), "1"])
                row[f"ours_{threads}"] = run([sys.executable, "-c", WORKER, str(path), job, str(M), str(threads)])
                if ddpy and job == "smoothness":
                    row["diff_diff"] = run([ddpy, "-c", DIFF_DIFF, str(path), job, str(M)])
                rows.append(row)
                print(n_pre, n_post, job, {k: round(v["seconds"], 3) for k, v in row.items() if isinstance(v, dict)},
                      flush=True)
    out = HERE / "results" / "sensitivity.csv"
    keys = ["r", "ours_1", f"ours_{threads}", "diff_diff"]
    with open(out, "w") as f:
        f.write("n_pre,n_post,job,M,implementation,seconds,peak_rss_mb,baseline_rss_mb,lb,ub\n")
        for row in rows:
            for k in keys:
                if k in row:
                    v = row[k]
                    f.write(f"{row['n_pre']},{row['n_post']},{row['job']},{row['M']},{k},{v['seconds']},"
                            f"{v['peak_rss_mb']},{v['baseline_rss_mb']},{v['lb']},{v['ub']}\n")
    lines = [f"| Pre-treatment periods | Restriction, interval | R HonestDiD | hdfe-stream, 1 thread | hdfe-stream, {threads} threads | diff-diff |",
             "|---:|---|---:|---:|---:|---:|"]
    for row in rows:
        dd = f"{row['diff_diff']['seconds']:.3g}" if "diff_diff" in row else "–"
        lines.append(f"| {row['n_pre']} | {JOBS[row['job']]} | {row['r']['seconds']:.3g} | "
                     f"{row['ours_1']['seconds']:.3g} | {row[f'ours_{threads}']['seconds']:.3g} | {dd} |")
    names = {"r": "R HonestDiD", "ours_1": "hdfe-stream (1 thread)",
             f"ours_{threads}": f"hdfe-stream ({threads} threads)", "diff_diff": "diff-diff"}
    lines += ["", "Largest increase in peak memory over the process's value after loading the package:", ""]
    for k in keys:
        extra = [row[k]["peak_rss_mb"] - row[k]["baseline_rss_mb"] for row in rows if k in row]
        if extra:
            lines.append(f"- {names[k]}: {max(extra):.0f} MB")
    (ROOT / "docs" / "benchmarks" / "sensitivity_table.md").write_text("\n".join(lines) + "\n")
    print(out)


if __name__ == "__main__":
    main()

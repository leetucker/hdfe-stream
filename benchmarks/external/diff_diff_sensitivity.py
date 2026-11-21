"""One sensitivity analysis with diff-diff, for benchmarks/sensitivity_benchmark.py:
the smoothness restriction's FLCI for the first post-treatment effect, at each value
of M, from the event study's coefficients and covariance. Prints one line,

    RESULT {"seconds": ..., "peak_rss_mb": ..., "baseline_rss_mb": ..., "lower": [...], "upper": [...]}

    python diff_diff_sensitivity.py '<spec as JSON>'      (path, M)

diff-diff's public HonestDiD takes a fitted results object; the computation it runs
for the smoothness restriction is `_compute_smoothness_bounds`, called here on the
arrays, once per value of M, as its `sensitivity_analysis` does. Its
relative-magnitudes interval is not the method's (it widens the estimated identified
set by z standard errors), so it is not timed.
"""

import json
import sys
import time

import numpy as np


def hwm_mb():
    with open("/proc/self/status") as handle:
        for line in handle:
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) / 1024
    return float("nan")


def main():
    spec = json.loads(sys.argv[1])
    from diff_diff.honest_did import HonestDiD

    es = json.load(open(spec["path"]))
    b, V, n_pre = np.array(es["beta"]), np.array(es["sigma"]), es["n_pre"]
    n_post = len(b) - n_pre
    l_vec = np.zeros(n_post)
    l_vec[0] = 1.0
    lower, upper = [], []
    baseline = hwm_mb()
    t0 = time.perf_counter()
    for M in spec["M"]:
        h = HonestDiD(method="smoothness", M=M, l_vec=l_vec)
        _lb, _ub, lo, hi = h._compute_smoothness_bounds(
            b[:n_pre], b[n_pre:], V, V[n_pre:, n_pre:], l_vec, n_pre, n_post, M)
        lower.append(lo)
        upper.append(hi)
    seconds = time.perf_counter() - t0
    print("RESULT " + json.dumps({"seconds": seconds, "peak_rss_mb": hwm_mb(),
                                  "baseline_rss_mb": baseline, "lower": lower,
                                  "upper": upper}))


if __name__ == "__main__":
    main()

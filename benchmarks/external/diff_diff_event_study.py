"""An event study and its sensitivity analysis with diff-diff, for
benchmarks/event_study_benchmark.py: read the panel with pandas, fit the two-way fixed
effects event study (TwoWayFixedEffects with event_study=True, clustered by unit, its
default), then HonestDiD's smoothness FLCI for the first post-treatment effect at the
values of M given. Prints one line,

    RESULT {"t_read": ..., "t_fit": ..., "t_smoothness": ..., "peak_rss_mb": ..., ...}

    python diff_diff_event_study.py '<spec as JSON>'      (path, n_pre, n_post, m)

diff-diff's relative-magnitudes interval is not the method's (it widens the estimated
identified set by z standard errors), so it is not timed. diff-diff runs on pandas,
in memory.
"""

import json
import sys
import time
import warnings

import numpy as np


def hwm_mb():
    with open("/proc/self/status") as handle:
        for line in handle:
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) / 1024
    return float("nan")


def main():
    spec = json.loads(sys.argv[1])
    warnings.simplefilter("ignore")
    import diff_diff as dd
    import pandas as pd

    baseline = hwm_mb()
    n_pre, n_post = spec["n_pre"], spec["n_post"]
    t0 = time.perf_counter()
    df = pd.read_parquet(spec["path"], columns=["unit", "year", "treated", "y"])
    t_read = time.perf_counter() - t0
    first = n_pre + 1                      # the first treated year
    t0 = time.perf_counter()
    fit = dd.TwoWayFixedEffects().fit(
        df, outcome="y", treatment="treated", unit="unit", time="year", event_study=True,
        post_periods=list(range(first, first + n_post)), reference_period=first - 1)
    t_fit = time.perf_counter() - t0
    l_vec = np.zeros(n_post)
    l_vec[0] = 1.0
    t0 = time.perf_counter()
    sens = dd.HonestDiD(method="smoothness", l_vec=l_vec).sensitivity_analysis(
        fit, M_grid=list(spec["m"]))
    t_sd = time.perf_counter() - t0
    lower = [float(lo) for lo, _hi in sens.robust_cis]
    upper = [float(hi) for _lo, hi in sens.robust_cis]
    print("RESULT " + json.dumps({
        "t_read": t_read, "t_fit": t_fit, "t_smoothness": t_sd, "n_obs": int(len(df)),
        "peak_rss_mb": hwm_mb(), "baseline_rss_mb": baseline, "peak_disk_mb": 0,
        "smoothness": {"lower": lower, "upper": upper}}))


if __name__ == "__main__":
    main()

"""Time and peak memory of a wide fit with fixed effects against the size of its
chunks and buckets: the trade-off behind the defaults for wide designs.

    python benchmarks/chunk_size_benchmark.py            # 503 indicators
    python benchmarks/chunk_size_benchmark.py 1 3        # pick bin widths

The model and data are those of the covariates benchmark
(`covariates_benchmark.py`): the 1,000,000-worker panel (8.5 million rows,
66,665 firms), ``log_earn ~ i(age_bin) | worker_id + firm_id``, clustered by
worker, here at its widest (bins one, three and six months wide: 503, 167 and 83 indicators).
Each configuration sets `batch_rows` (the rows of a chunk in the passes after
pass 0) and `rows_per_bucket` (the rows of a bucket in pass 0's sort), or
leaves them at their defaults; "direct restrictions" also has pass 0's row
restrictions query the evaluated design rather than a narrow copy of the
rows. Every configuration is fitted `ROUNDS` times, in a different order each
round, so that a drift in the machine does not favor one. Results go to
benchmarks/results/chunk_size.csv, one row per configuration and round.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from _harness import DATA_DIR, RESULTS_DIR, is_done, measure, run_child, upsert_csv, \
    write_machine_info
from covariates_benchmark import covariate_path, formulas, level_of

SCRIPT = Path(__file__).resolve()
CSV = RESULTS_DIR / "chunk_size.csv"
KEY = ("n_covariates", "configuration", "round")

DEFAULT_WIDTHS = [1, 3, 6]
ROUNDS = 3
TIMEOUT_S = 3600

# label -> (batch_rows, rows_per_bucket, how the restrictions are queried); None is
# the default; "256MB" and "384MB" about that much of the design (a batch) or of
# the rows at their width (a bucket), the sizes before chunks were aligned with buckets
CONFIGS = {
    "defaults": ("hdfe", (None, None, "default")),
    "unaligned (batch 256 MB, buckets 384 MB)": ("hdfe", ("256MB", "384MB", "default")),
    "0.13.1 sizes (batch 2,000,000, buckets 500,000)": ("hdfe", (2_000_000, 500_000,
                                                                 "default")),
    "0.13.1 sizes, direct restrictions": ("hdfe", (2_000_000, 500_000, "direct")),
}


def run_hdfe(label, width, path, workdir):
    from hdfe_stream import feols_stream
    from hdfe_stream.core import passes

    batch, bucket, restrictions = CONFIGS[label][1]
    fml, _, k = formulas(width, path)
    options = {}
    m = k + 1                                 # the design's columns: outcome and k
    if batch == "256MB":
        options["batch_rows"] = min(2_000_000, max(20_000, int(256e6 / (8 * (m + 4)))))
    elif batch is not None:
        options["batch_rows"] = batch
    if bucket == "384MB":                     # pass 0's width: design, sources, row id
        options["rows_per_bucket"] = min(20_000_000, max(50_000, int(384e6 / (8 * (m + 5)))))
    elif bucket is not None:
        options["rows_per_bucket"] = bucket
    if restrictions == "direct":
        passes._MORSEL_ROWS = 1          # the morsels always "fit": the direct query
    t0 = time.perf_counter()
    fit = feols_stream(fml, str(path), workdir=str(workdir), vcov={"CRV1": "worker_id"},
                       verbose=False, save_resid=False, **options)
    t_fit = time.perf_counter() - t0
    out = {"t_read": 0.0, "t_fit": t_fit,
           "coef": {str(level_of(c)): float(v) for c, v in fit.coef().items()},
           "n_coef": len(fit.coefnames),
           "batch_rows": fit.diagnostics.get("batch_rows"),
           "n_buckets": fit.diagnostics.get("n_buckets")}
    fit.cleanup()
    return out


def runner(kind, label, path, workdir):
    _, width = kind.split(":")
    return run_hdfe(label, int(width), path, workdir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("widths", nargs="*", type=int, default=DEFAULT_WIDTHS,
                        help="age bin widths in months")
    parser.add_argument("--only", nargs="+", metavar="CONFIG",
                        help="run only these configurations (by label)")
    parser.add_argument("--rounds", type=int, default=ROUNDS)
    parser.add_argument("--child", nargs=4, metavar=("KIND", "LABEL", "PATH", "WORKDIR"),
                        help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child:
        run_child(runner, *args.child)
        return
    labels = [c for c in CONFIGS if not args.only or c in args.only]
    write_machine_info(RESULTS_DIR / "machine.json")
    path = covariate_path()
    workdir = DATA_DIR / "work_chunk_size"
    measure(SCRIPT, "hdfe:60", "defaults", path, workdir)      # compile numba's kernels
    for width in args.widths:
        _, _, k = formulas(width, path)
        print(f"\n=== age bins {width} months wide: {k} indicators ===", flush=True)
        base = None
        for rnd in range(1, args.rounds + 1):
            order = labels if rnd % 2 else labels[::-1]
            for label in order:
                key = {"n_covariates": k, "configuration": label, "round": rnd}
                if is_done(CSV, key, key=KEY):
                    print(f"  round {rnd} {label:<48}measured before, kept", flush=True)
                    continue
                report = measure(SCRIPT, f"hdfe:{width}", label, path, workdir,
                                 timeout=TIMEOUT_S)
                row = {**key, "bin_months": width, "order": list(CONFIGS).index(label),
                       "status": report["status"]}
                if report["status"] == "ok":
                    if base is None:
                        base = report["coef"]
                    gap = max(abs(report["coef"].get(lv, float("nan")) - v)
                              / max(abs(v), 1e-3) for lv, v in base.items())
                    row.update(wall_s=round(report["t_fit"], 3),
                               peak_memory_mb=round(report["peak_rss_mb"], 1),
                               import_memory_mb=round(report["baseline_rss_mb"], 1),
                               peak_disk_mb=round(report["peak_disk_mb"], 1),
                               n_coef=report["n_coef"], max_rel_diff=gap)
                    print(f"  round {rnd} {label:<48}{row['wall_s']:8.1f} s   peak "
                          f"{row['peak_memory_mb']:8,.0f} MB   diff {gap:.1e}", flush=True)
                else:
                    print(f"  round {rnd} {label:<48}{report['status']}", flush=True)
                upsert_csv(CSV, [row], key=KEY)


if __name__ == "__main__":
    main()

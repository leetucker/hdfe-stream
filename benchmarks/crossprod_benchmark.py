"""Time candidate kernels for `core.crossprod`'s rows pass: adding a chunk of
residualized rows to each split's cross-products.

    python benchmarks/crossprod_benchmark.py                 # widths 3 to 1000
    python benchmarks/crossprod_benchmark.py 3 50 200        # pick your own
    python benchmarks/crossprod_benchmark.py --quick         # a smoke test

The pass today gathers each split's rows with NumPy indexing and adds
``B.T @ B``, which NumPy hands to BLAS's dsyrk. The candidates change how the
rows are gathered (NumPy indexing, or a parallel numba copy that also applies
the weights) and how the product is formed (dsyrk, or dgemm on panels of at
most 256 columns, which OpenBLAS runs faster than dsyrk on wide matrices):

    current          NumPy gather, B.T @ B (dsyrk through NumPy): the pass today
    gather + dsyrk   NumPy gather, scipy's dsyrk into a Fortran-order accumulator
    gather + panels  NumPy gather, dgemm on 256-column panels of the upper triangle
    pgather + dsyrk  parallel numba gather (and weights), dsyrk
    pgather + panels parallel numba gather (and weights), 256-column panels

Two kinds of measurement, for each width (number of variables m):

kernel
    Chunks in memory, as the pass sees them: without fixed effects, batches of
    the estimator's default size for that width; with them, chunks of whole
    groups (rows in runs of one split, as folds of a fixed effect's groups
    make them). Splits 1, 5, 20 and 100, rows assigned at random; and with 5
    splits, rows in group runs, a skewed assignment (one split with 80% of the
    rows) and weights. Every candidate's sums are checked against the current
    pass's before it is timed.
end_to_end
    `cross_products` on a simulated Parquet panel (about 1.5 GB of design),
    without fixed effects and with one (folds of its groups), 5 splits, with
    the candidate in place of the pass's kernel: the reader, the fixed-effect
    machinery and the thread pools compete as they do in a real fit.

Each configuration runs in its own process on all the machine's threads (numba
and BLAS at their defaults). Results go to benchmarks/results/crossprod.csv:
one row per (width, configuration, scenario), with the median seconds over
the repetitions and the largest relative difference from the current pass.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from _harness import (
    DATA_DIR,
    RESULTS_DIR,
    is_done,
    measure,
    peak_rss_mb,
    upsert_csv,
    write_machine_info,
)

SCRIPT = Path(__file__).resolve()
CSV = RESULTS_DIR / "crossprod.csv"
KEY = ("width", "configuration", "scenario")

DEFAULT_WIDTHS = [3, 10, 50, 200, 500, 1000]
CONFIGS = {
    "current": ("hdfe", "current"),
    "gather + dsyrk": ("hdfe", "gather_dsyrk"),
    "gather + panels": ("hdfe", "gather_panels"),
    "pgather + dsyrk": ("hdfe", "pgather_dsyrk"),
    "pgather + panels": ("hdfe", "pgather_panels"),
}
PANEL = 256
TIMEOUT_S = 3 * 3600
END_TO_END_MAX_WIDTH = 500        # 1.5 GB of design at most; wider is slow to simulate


# ----------------------------------------------------------------- candidates

_PGATHER = None


def _kernels():
    """The parallel gather, compiled once per process (numba caches it on disk)."""
    global _PGATHER
    if _PGATHER is not None:
        return _PGATHER
    import numba as nb

    @nb.njit(parallel=True, cache=True, nogil=True)
    def pgather(V, order, w, weighted, out):
        """Rows of V in split order, into `out` (C order), times sqrt(w)."""
        n, m = V.shape
        n_threads = nb.get_num_threads()
        for t in nb.prange(n_threads):
            lo = t * n // n_threads
            hi = (t + 1) * n // n_threads
            for i in range(lo, hi):
                r = order[i]
                if weighted:
                    sw = np.sqrt(w[r])
                    for j in range(m):
                        out[i, j] = V[r, j] * sw
                else:
                    for j in range(m):
                        out[i, j] = V[r, j]

    _PGATHER = pgather
    return pgather


class Candidate:
    """Accumulates Sum w v v', Sum w v, Sum w and the counts by split."""

    def __init__(self, variant, m, n_splits):
        from scipy.linalg.blas import dsyrk
        self.variant, self.m, self.S = variant, m, n_splits
        self.dsyrk = dsyrk
        f_order = variant != "current"
        self.gram = [np.zeros((m, m), order="F" if f_order else "C") for _ in range(n_splits)]
        self.sums = np.zeros((n_splits, m))
        self.wsum = np.zeros(n_splits)
        self.count = np.zeros(n_splits, np.int64)
        self.buf = np.empty((0, m))
        self.pgather = _kernels() if variant.startswith("pgather") else None

    def _product(self, j, B):
        """gram[j] += B'B for a C-order block B (rows x m)."""
        G = self.gram[j]
        if self.variant == "current":
            G += B.T @ B
            return
        if self.variant.endswith("dsyrk") or self.m <= PANEL:
            # B.T is Fortran order (m x rows): dsyrk(a, trans=0) is a a' = B'B
            self.dsyrk(1.0, B.T, beta=1.0, c=G, trans=0, overwrite_c=1)
            return
        BT = B.T
        edges = list(range(0, self.m, PANEL)) + [self.m]
        for a in range(len(edges) - 1):
            I = slice(edges[a], edges[a + 1])
            for b in range(a, len(edges) - 1):
                J = slice(edges[b], edges[b + 1])
                G[I, J] += BT[I] @ BT[J].T

    def add(self, V, w, s, weighted):
        n = V.shape[0]
        if self.S == 1:
            order = None
            bounds = np.array([0, n])
        else:
            order = np.argsort(s, kind="stable")
            bounds = np.concatenate([[0], np.cumsum(np.bincount(s, minlength=self.S))])
        if self.pgather is not None:
            if self.buf.shape[0] < n:
                self.buf = np.empty((n, self.m))
            out = self.buf[:n]
            idx = np.arange(n) if order is None else order
            self.pgather(V, idx, w, weighted, out)
            for j in range(self.S):
                lo, hi = bounds[j], bounds[j + 1]
                if hi == lo:
                    continue
                rows = idx[lo:hi]
                wj = w[rows]
                self.count[j] += hi - lo
                self.wsum[j] += wj.sum() if weighted else hi - lo
                # the block is already scaled by sqrt(w): its sums need sqrt(w) once more
                B = out[lo:hi]
                self.sums[j] += (np.sqrt(wj) @ B) if weighted else B.sum(axis=0)
                self._product(j, B)
            return
        for j in range(self.S):
            lo, hi = bounds[j], bounds[j + 1]
            if hi == lo:
                continue
            B = V if order is None else V[order[lo:hi]]
            wj = w if order is None else w[order[lo:hi]]
            self.count[j] += B.shape[0]
            if weighted:
                self.wsum[j] += wj.sum()
                self.sums[j] += wj @ B
                B = B * np.sqrt(wj)[:, None]
            else:
                self.wsum[j] += B.shape[0]
                self.sums[j] += B.sum(axis=0)
            self._product(j, B)

    def result(self):
        out = np.empty((self.S, self.m, self.m))
        for j, G in enumerate(self.gram):
            if self.variant == "current":
                out[j] = (G + G.T) / 2
            else:
                u = np.triu(G)
                out[j] = u + np.triu(u, 1).T
        return out, self.sums, self.wsum, self.count


def as_crossprod_kernel(variant):
    """A drop-in for core.crossprod._accumulate that uses the candidate: each
    chunk's sums are formed by the candidate and added to the pass's."""

    def _accumulate(V, w, s, n_splits, gram, sums, wsum, count, weighted):
        cand = Candidate(variant, V.shape[1], n_splits)
        cand.add(np.ascontiguousarray(V), w, s.astype(np.int64, copy=False), weighted)
        g, su, ws, co = cand.result()
        gram += g
        sums += su
        wsum += ws
        count += co

    return _accumulate


# ------------------------------------------------------------------ scenarios

def chunk_rows(m, grouped):
    """The pass's chunk size: without fixed effects the estimator's default
    batch; with them chunks of whole groups, here capped at 256 MB of design."""
    if not grouped:
        return min(2_000_000, max(100_000, int(64e6 / (8 * (m + 4)))))
    return min(2_000_000, max(20_000, int(256e6 / (8 * m))))


def kernel_scenarios():
    out = [{"name": f"rows, {S} splits", "splits": S, "layout": "rows", "weighted": False}
           for S in (1, 5, 20, 100)]
    out += [{"name": "group runs, 5 splits", "splits": 5, "layout": "groups",
             "weighted": False},
            {"name": "skewed, 5 splits", "splits": 5, "layout": "skewed", "weighted": False},
            {"name": "rows, 5 splits, weighted", "splits": 5, "layout": "rows",
             "weighted": True}]
    return out


def make_chunk(m, sc, rng):
    grouped = sc["layout"] == "groups"
    n = chunk_rows(m, grouped)
    V = rng.standard_normal((n, m))
    w = rng.uniform(0.5, 2.0, n)
    S = sc["splits"]
    if sc["layout"] == "rows":
        s = rng.integers(0, S, n)
    elif sc["layout"] == "groups":
        sizes = rng.geometric(1 / 20, n // 5)
        sizes = sizes[np.cumsum(sizes) <= n]
        g = np.repeat(np.arange(len(sizes)), sizes)
        g = np.concatenate([g, np.full(n - len(g), len(sizes))])
        s = rng.integers(0, S, len(sizes) + 1)[g]
    else:
        s = np.where(rng.random(n) < 0.8, 0, rng.integers(1, S, n))
    return V, w, s.astype(np.int64)


def time_kernel(variant, m, sc, reps):
    rng = np.random.default_rng(m * 1000 + sc["splits"])
    V, w, s = make_chunk(m, sc, rng)
    ref = Candidate("current", m, sc["splits"])
    ref.add(V, w, s, sc["weighted"])
    rg = ref.result()[0]
    cand = Candidate(variant, m, sc["splits"])
    cand.add(V, w, s, sc["weighted"])             # also compiles numba's kernel
    cg = cand.result()[0]
    diff = float(np.abs(cg - rg).max() / max(np.abs(rg).max(), 1e-300))
    times = []
    for _ in range(reps):
        cand = Candidate(variant, m, sc["splits"])
        t = time.perf_counter()
        cand.add(V, w, s, sc["weighted"])
        times.append(time.perf_counter() - t)
    sec = statistics.median(times)
    return {"scenario": f"kernel: {sc['name']}", "seconds": sec, "rows": len(V),
            "ns_per_row": sec / len(V) * 1e9, "max_rel_diff": diff}


def panel_path(m, fe):
    folder = DATA_DIR / "crossprod"
    folder.mkdir(parents=True, exist_ok=True)
    n = int(min(5_000_000, 1.5e9 / (8 * (m + 3))))
    path = folder / f"panel_m{m}_{'fe' if fe else 'nofe'}.parquet"
    if path.exists():
        return path, n
    import pyarrow as pa
    import pyarrow.parquet as pq
    rng = np.random.default_rng(m)
    names = [f"v{j}" for j in range(m)]
    schema = pa.schema([(nm, pa.float64()) for nm in names]
                       + [("g", pa.int64()), ("fold", pa.int64())])
    with pq.ParquetWriter(path, schema) as wr:
        done = 0
        while done < n:
            k = min(500_000, n - done)
            g = (np.arange(done, done + k) // 20) if fe else rng.integers(0, 10**6, k)
            from_g = (g * 2654435761) % 5
            arrays = [pa.array(rng.standard_normal(k) + 0.01 * (g % 7)) for _ in names]
            arrays += [pa.array(g), pa.array(from_g if fe else rng.integers(0, 5, k))]
            wr.write_table(pa.Table.from_arrays(arrays, schema=schema))
            done += k
    return path, n


def time_end_to_end(variant, m, fe, reps):
    from hdfe_stream.core import crossprod
    path, n = panel_path(m, fe)
    names = [f"v{j}" for j in range(m)]
    original = crossprod._accumulate
    times = []
    try:
        if variant != "current":
            crossprod._accumulate = as_crossprod_kernel(variant)
        for _ in range(reps + 1):                          # the first compiles
            t = time.perf_counter()
            cp = crossprod.cross_products(str(path), names, ["g"] if fe else [],
                                          split_by="fold", n_splits=5,
                                          workdir=DATA_DIR / "work", verbose=False)
            times.append(time.perf_counter() - t)
    finally:
        crossprod._accumulate = original
    ref = np.load(path.with_suffix(".ref.npy")) if path.with_suffix(".ref.npy").exists() \
        else None
    if variant == "current":
        np.save(path.with_suffix(".ref.npy"), cp.gram)
        diff = 0.0
    else:
        diff = (None if ref is None else
                float(np.abs(cp.gram - ref).max() / max(np.abs(ref).max(), 1e-300)))
    sec = statistics.median(times[1:])
    return {"scenario": f"end to end: {'one fixed effect' if fe else 'no fixed effects'}, "
                        "5 splits",
            "seconds": sec, "rows": n, "ns_per_row": sec / n * 1e9, "max_rel_diff": diff}


# --------------------------------------------------------------------- driver

def child(spec):
    variant, m, reps = spec["variant"], spec["width"], spec["reps"]
    t0 = time.perf_counter()
    rows = [time_kernel(variant, m, sc, reps) for sc in kernel_scenarios()]
    if spec["end_to_end"] and m <= END_TO_END_MAX_WIDTH:
        rows += [time_end_to_end(variant, m, fe, max(1, reps // 2)) for fe in (False, True)]
    print("RESULT " + json.dumps({"rows": rows, "seconds": time.perf_counter() - t0,
                                  "peak_rss_mb": peak_rss_mb()}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("widths", nargs="*", type=int, default=DEFAULT_WIDTHS,
                        help="numbers of variables")
    parser.add_argument("--only", nargs="+", metavar="CONFIG",
                        help="run only these configurations (by label)")
    parser.add_argument("--reps", type=int, default=7, help="repetitions per scenario")
    parser.add_argument("--no-end-to-end", action="store_true",
                        help="only the in-memory kernel scenarios")
    parser.add_argument("--quick", action="store_true",
                        help="widths 3, 50 and 200, 3 repetitions: a smoke test")
    parser.add_argument("--child", metavar="SPEC", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child:
        child(json.loads(args.child))
        return
    widths, reps = (([3, 50, 200], 3) if args.quick else (args.widths, args.reps))
    labels = [label for label in CONFIGS if not args.only or label in args.only]
    if "current" not in labels:
        labels = ["current", *labels]       # the end-to-end check compares with it
    write_machine_info(RESULTS_DIR / "crossprod_machine.json")
    print(f"threads: {os.cpu_count()} logical CPUs", flush=True)
    for m in widths:
        print(f"\n=== {m} variables ===", flush=True)
        base = {}
        for c, label in enumerate(labels):
            values = {"width": m, "configuration": label}
            first = f"kernel: {kernel_scenarios()[0]['name']}"
            if is_done(CSV, {**values, "scenario": first}, KEY):
                print(f"  {label:<20}measured before, kept", flush=True)
                continue
            spec = {"variant": CONFIGS[label][1], "width": m, "reps": reps,
                    "end_to_end": not args.no_end_to_end}
            cmd = [sys.executable, str(SCRIPT), "--child", json.dumps(spec)]
            report = measure(SCRIPT, "", "", "", DATA_DIR / "work_driver",
                             timeout=TIMEOUT_S, command=cmd)
            if report["status"] != "ok":
                print(f"  {label:<20}{report['status']}", flush=True)
                upsert_csv(CSV, [{**values, "order": c, "scenario": first,
                                  "status": report["status"]}], KEY)
                continue
            rows = []
            for r in report["rows"]:
                if label == "current":
                    base[r["scenario"]] = r["seconds"]
                ref = base.get(r["scenario"])
                rows.append({**values, "order": c, "scenario": r["scenario"],
                             "status": "ok", "rows": r["rows"],
                             "seconds": round(r["seconds"], 6),
                             "ns_per_row": round(r["ns_per_row"], 3),
                             "speedup_vs_current": (round(ref / r["seconds"], 3)
                                                    if ref else None),
                             "max_rel_diff": r["max_rel_diff"],
                             "peak_memory_mb": round(report["peak_rss_mb"], 1)})
                print(f"  {label:<20}{r['scenario']:<52}{r['seconds']:9.4f} s"
                      + (f"   x{ref / r['seconds']:5.2f}" if ref else "")
                      + (f"   diff {r['max_rel_diff']:.1e}"
                         if r["max_rel_diff"] is not None else ""), flush=True)
            upsert_csv(CSV, rows, KEY)


if __name__ == "__main__":
    main()

"""Small helpers shared across the package: machine facts, file naming,
Parquet iteration in whole fe[0] groups, and segment/scatter primitives.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq


def _phys_mem_gb():
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    except (ValueError, OSError, AttributeError):
        return 16.0


def _safe(name):
    """File-name-safe version of a dimension name like 'sein^year'."""
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name.replace("^", "_x_"))


def iter_group_chunks(paths, columns, batch_rows=2_000_000):
    """Yield dicts of numpy arrays from Parquet file(s) sorted by `gcode`.

    Every yielded chunk contains *complete* fe[0] groups: rows of the last
    group in a batch are carried into the next one. With several files,
    `gcode` must be increasing across files (as the bucket files are).
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]
    carry = None
    for batch in (b for p in paths
                  for b in pq.ParquetFile(p).iter_batches(batch_size=batch_rows,
                                                          columns=columns)):
        tbl = pa.Table.from_batches([batch])
        if carry is not None:
            tbl = pa.concat_tables([carry, tbl])
        g = tbl.column("gcode").to_numpy()
        cut = int(np.searchsorted(g, g[-1], side="left"))
        if cut == 0:  # the whole buffer is one group; keep accumulating
            carry = tbl
            continue
        yield _tbl_to_np(tbl.slice(0, cut))
        carry = tbl.slice(cut)
    if carry is not None and carry.num_rows:
        yield _tbl_to_np(carry)


def _tbl_to_np(tbl):
    return {c: tbl.column(c).to_numpy() for c in tbl.column_names}


def _segments(g):
    """Segment starts and local segment index for a sorted code vector."""
    brk = np.flatnonzero(g[1:] != g[:-1]) + 1
    starts = np.concatenate(([0], brk))
    local = np.zeros(len(g), dtype=np.int64)
    local[brk] = 1
    np.cumsum(local, out=local)
    return starts, local


def _scatter(idx, vals, n):
    """Column-wise bincount: out[j] = sum of the rows of vals with idx == j."""
    vals = np.asarray(vals)
    if vals.ndim == 1:
        return np.bincount(idx, weights=vals, minlength=n)
    return np.column_stack(
        [np.bincount(idx, weights=vals[:, j], minlength=n) for j in range(vals.shape[1])]
    )


def _norm_vars(items):
    """Normalize variables to an ordered {name: Float64 Polars expression}."""
    if isinstance(items, dict):
        return {k: v.cast(pl.Float64) for k, v in items.items()}
    if isinstance(items, (str, tuple)):
        items = [items]
    out = {}
    for it in items:
        if isinstance(it, str):
            out[it] = pl.col(it).cast(pl.Float64)
        else:
            name, expr = it
            out[name] = expr.cast(pl.Float64)
    return out

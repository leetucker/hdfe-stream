"""Small helpers shared across the package: machine facts, file naming,
Parquet iteration in whole fe[0] groups, and segment/scatter primitives.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from ._columns import GCODE
from ._types import PathLike, Source


def _phys_mem_gb():
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    except (ValueError, OSError, AttributeError):
        return 16.0


def _safe(name):
    """File-name-safe version of a dimension name like 'firm_id^year'."""
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name.replace("^", "_x_"))


def iter_group_chunks(paths: PathLike | Sequence[PathLike], columns: Sequence[str],
                      batch_rows: int = 2_000_000) -> Iterator[dict[str, np.ndarray]]:
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
        g = tbl.column(GCODE).to_numpy()
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


# --------------------------------------------------------------------------
# row-wise expressions
#
# Pass 0 can evaluate the design either before the rows are partitioned into
# buckets or afterwards, one bucket at a time. Afterwards is far cheaper for a
# wide design (a categorical expanded into hundreds of indicators is carried
# through the partition as one column), but it is only correct for expressions
# whose value on a row depends on that row alone: evaluated per bucket,
# `pl.col("x") - pl.col("x").mean()` would subtract each bucket's mean.
#
# Polars 1.x has no public test for that, so `_row_wise` walks the serialized
# expression tree against an allowlist. Anything it does not recognize counts
# as not row-wise, which is the safe direction: the caller then evaluates the
# design before partitioning, exactly as it did before, and pays only memory.
# A change in Polars' serialization can therefore make the test more cautious
# but never wrong.
# --------------------------------------------------------------------------

# expression variants that are row-wise provided their inputs are
_ROW_NODES = {"Column", "Cast", "Alias", "BinaryExpr", "Ternary", "Function", "Literal"}

# Function names (with their category, where Polars nests them). Aggregations,
# windows, sorts, slices, filters, shift/diff/cum_*, rank and fill_null by
# strategy are all missing on purpose.
_ROW_FUNCTIONS = {
    "Log", "Log1p", "Exp", "Negate", "Abs", "Floor", "Ceil", "Round", "Clip", "Sign",
    "FillNull", "AsStruct", "Hash",
    "Pow.Generic", "Pow.Sqrt", "Pow.Cbrt",
    "Boolean.IsNull", "Boolean.IsNotNull", "Boolean.IsFinite", "Boolean.IsInfinite",
    "Boolean.IsNan", "Boolean.IsNotNan", "Boolean.Not", "Boolean.IsIn",
    "Trigonometry.Sin", "Trigonometry.Cos", "Trigonometry.Tan", "Trigonometry.ArcSin",
    "Trigonometry.ArcCos", "Trigonometry.ArcTan", "Trigonometry.Sinh",
    "Trigonometry.Cosh", "Trigonometry.Tanh",
    "StringExpr.LenChars", "StringExpr.LenBytes", "StringExpr.Contains",
    "StringExpr.StartsWith", "StringExpr.EndsWith", "StringExpr.Lowercase",
    "StringExpr.Uppercase",
    "TemporalExpr.Year", "TemporalExpr.Month", "TemporalExpr.Day",
    "TemporalExpr.Quarter", "TemporalExpr.Week", "TemporalExpr.WeekDay",
    "TemporalExpr.OrdinalDay", "TemporalExpr.Hour", "TemporalExpr.Minute",
    "TemporalExpr.Second",
}


def _function_name(spec):
    """'Log' from "Log"; 'Pow.Sqrt' from {"Pow": "Sqrt"}; 'Clip' from
    {"Clip": {...options}}; 'Boolean.IsIn' from {"Boolean": {"IsIn": ...}}."""
    if isinstance(spec, str):
        return spec
    if isinstance(spec, dict) and len(spec) == 1:
        (key, value), = spec.items()
        if isinstance(value, str):
            return f"{key}.{value}"
        if isinstance(value, dict) and len(value) == 1 and key in (
                "Boolean", "StringExpr", "TemporalExpr", "Trigonometry", "Pow"):
            return f"{key}.{next(iter(value))}"
        return key
    return None


def _as_lazy(source: Source) -> pl.LazyFrame:
    """The source as a LazyFrame: a LazyFrame as is, a DataFrame as its lazy
    view (no copy), anything else as a Parquet path or glob to scan."""
    if isinstance(source, pl.LazyFrame):
        return source
    if isinstance(source, pl.DataFrame):
        return source.lazy()
    return pl.scan_parquet(source)


def _row_wise(expr):
    """(True, None) if `expr` is known to be row-wise, else (False, what was
    not recognized)."""
    import json

    try:
        tree = json.loads(expr.meta.serialize(format="json"))
    except Exception as err:          # noqa: BLE001 -- anything: be cautious
        return False, f"cannot inspect it ({type(err).__name__})"

    def walk(node):
        if not isinstance(node, dict) or len(node) != 1:
            return str(node)[:40]
        (kind, body), = node.items()
        if kind not in _ROW_NODES:
            return kind
        if kind == "Column":
            return None
        if kind == "Literal":
            # a scalar broadcasts; a Series literal is positional
            return None if isinstance(body, dict) and set(body) <= {"Dyn", "Scalar"} else "Series literal"
        if kind == "Cast":
            return walk(body["expr"])
        if kind == "Alias":
            return walk(body[0])
        if kind == "BinaryExpr":
            return walk(body["left"]) or walk(body["right"])
        if kind == "Ternary":
            return walk(body["predicate"]) or walk(body["truthy"]) or walk(body["falsy"])
        name = _function_name(body.get("function"))
        if name not in _ROW_FUNCTIONS:
            return f"function {name}"
        inputs = body.get("input", [])
        if name == "Boolean.IsIn" and any("Literal" not in i for i in inputs[1:]):
            return "is_in against a column"
        for sub in inputs:
            bad = walk(sub)
            if bad:
                return bad
        return None

    bad = walk(tree)
    return (bad is None), bad

"""Distinct counts of columns or combinations of columns, approximate or exact,
in streaming passes over the source.

`count_distinct` counts, for each named key, the distinct values among the
rows its filter keeps. By default the count is approximate: Polars'
HyperLogLog (``approx_n_unique``, about 2% off) in one streaming pass, the
same count the fits use to choose how to stream a fixed effect, and the
fastest. With ``exact=True`` the count is exact: Polars' ``n_unique`` holds a
hash table of the distinct values, about 50 bytes a value for a single
column and about 120 for a pair of columns, so its memory grows with the
count. When the tables would not fit the memory budget, the values are split
by a hash of the key into k parts, each counted in its own streaming pass, so
that each pass holds a k-th of the values. k comes from an approximate count
first (one more pass). A value's parts never overlap, so the parts' counts
add up to the exact count.

Floats are counted by value: -0.0 and 0.0 are the same value, and so are
all NaNs. A null counts as one value, as in Polars' ``n_unique``; filter the
rows to leave nulls out.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import polars as pl

from .utils import as_lazy

__all__ = ["count_distinct", "distinct_bytes"]

Key = str | Sequence[str] | pl.Expr

_BYTES_ONE = 56           # per distinct value of one column in n_unique's table (measured ~45)
_BYTES_EXTRA = 64         # per further column of a combination (row-encoded; measured ~60)
_SEED = 0x5EED            # the partitioning hash: any seed, fixed so runs repeat
_UPPER = 1.25
_MAX_PARTITIONS = 256     # more passes than this: the budget is too small for the count


def distinct_bytes(n_values: int, n_columns: int = 1) -> float:
    """Expected bytes of Polars' hash table for ``n_values`` distinct values
    of a key of ``n_columns`` columns (an exact count's memory per pass)."""
    return float(n_values) * (_BYTES_ONE + _BYTES_EXTRA * max(n_columns - 1, 0))


def _key_expr(key: Key, schema) -> tuple[pl.Expr, int]:
    """The key as one expression whose values compare as the key's do, and
    its number of columns."""
    if isinstance(key, pl.Expr):
        return key, 1
    cols = [key] if isinstance(key, str) else list(key)
    if not cols:
        raise ValueError("a distinct-count key needs at least one column")
    for c in cols:
        if c not in schema:
            raise ValueError(f"column {c!r} is not in the data")
    # 0.0 + -0.0 is 0.0: one value for the two zeros, under hashing as under equality
    exprs = [pl.col(c) + 0.0 if schema[c].is_float() else pl.col(c) for c in cols]
    if len(exprs) == 1:
        return exprs[0].alias(cols[0]), 1
    return pl.struct(exprs), len(exprs)


def count_distinct(source, keys: Mapping[str, Key], *, exact: bool = False,
                   where: pl.Expr | Mapping[str, pl.Expr] | None = None,
                   partitions: int | None = None, max_memory_gb: float | None = None,
                   memory_reserve: float | None = None) -> dict[str, int]:
    """Distinct values of each key among the rows its filter keeps.

    Parameters
    ----------
    source : Parquet path or glob, polars.DataFrame or polars.LazyFrame
    keys : name -> a column, a list of columns (their combinations are
        counted) or an expression.
    exact : bool, default False
        False: HyperLogLog, one streaming pass, about 2% off. True: exact,
        in one streaming pass when the hash tables fit the memory budget and
        in k hash-partitioned passes when they do not.
    where : a boolean expression for every key, or name -> expression; rows
        where it is false (or null) are not counted.
    partitions : int, optional
        The number of hash partitions of an exact count (one pass each);
        default the fewest whose tables fit half the budget's headroom. When
        that would take more than 256 passes, the count raises MemoryError.
    max_memory_gb, memory_reserve : the memory budget (stream-limits'
        `MemoryBudget`) an exact count's tables must fit.

    Returns
    -------
    dict
        name -> count.
    """
    lf = as_lazy(source)
    if not keys:
        return {}
    schema = lf.collect_schema()
    names = list(keys)
    built = {n: _key_expr(keys[n], schema) for n in names}
    if isinstance(where, Mapping):
        unknown = set(where) - set(names)
        if unknown:
            raise ValueError(f"where names keys that are not counted: {sorted(unknown)}")
        filt = {n: where.get(n) for n in names}
    else:
        filt = {n: where for n in names}

    def kept(n, extra=None):
        e = built[n][0]
        conds = [c for c in (filt[n], extra) if c is not None]
        if not conds:
            return e
        return e.filter(pl.all_horizontal(conds).fill_null(False))

    def run(aggs):
        row = lf.select(aggs).collect(engine="streaming").row(0, named=True)
        return {n: int(row[n] or 0) for n in names}

    if not exact:
        return run([kept(n).hash(seed=_SEED).approx_n_unique().alias(n) for n in names])
    if partitions is None:
        approx = run([kept(n).hash(seed=_SEED).approx_n_unique().alias(n) for n in names])
        need = sum(distinct_bytes(approx[n], built[n][1]) for n in names) * _UPPER
        from .resources import MemoryBudget
        room = MemoryBudget(limit_gb=max_memory_gb, reserve=memory_reserve).headroom_gb() * 1e9
        partitions = max(1, math.ceil(need / max(room / 2, 1.0)))
        if partitions > _MAX_PARTITIONS:
            raise MemoryError(
                f"an exact count of about {sum(approx.values()):,} distinct values needs about "
                f"{need / 1e9:.2f} GB, and the memory budget leaves {max(room, 0) / 1e9:.2f} GB "
                f"(max_memory_gb, less memory_reserve, less what this process holds): more "
                f"than {_MAX_PARTITIONS} passes; give more memory, or partitions=")
    k = int(partitions)
    if k < 1:
        raise ValueError("partitions must be at least 1")
    if k == 1:
        return run([kept(n).n_unique().alias(n) for n in names])
    total = dict.fromkeys(names, 0)
    for i in range(k):
        part = run([kept(n, (built[n][0].hash(seed=_SEED) % k) == i).n_unique().alias(n)
                    for n in names])
        for n in names:
            total[n] += part[n]
    return total

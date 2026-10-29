"""A row store for estimators that read every row in each iteration.

Pass 0 of such an estimator reads the source once and writes what each
iteration needs -- an evaluated design and a few per-row arrays -- as
fixed-size blocks of plain numpy arrays (`BlockStore`); every iteration then
reads the blocks in order, the next one in a background thread while the
current one is worked on. Nothing row-sized is held in memory beyond the
block being worked on. logit-stream's Newton steps are built on it.

Why blocks of ``.npy`` files rather than Parquet: a block on disk is the
matrix as BLAS wants it (row-major), read with one ``readinto``, where a
Parquet batch must be decoded and its columns stacked. On a 3-million-row,
39-column indicator design that saves about a third of a pass, at about 35
times the disk (Parquet compresses indicators very well): worth it for many
passes over a design that fits on disk, not for one or two.

The way there (`write_sample`, `iter_batches`): Polars' streaming sink
writes the *sample* -- the source columns the model uses, for the rows it
keeps -- to a local Parquet file; that copy is read back a batch at a time,
the design evaluated on each batch (formula expressions are row-wise), and
the batch written as a block; then the copy is deleted. Writing the source
columns rather than the design keeps the copy narrow, which matters because
the sink holds about a row group per thread of what it writes: with the
design written, a 41-column design at 3 million rows peaked at 1.2 GB
there, against about 0.25 GB.

When blocks must hold whole groups of rows (a choice set, a panel),
`iter_groups` yields the sample in frames of whole groups: in the order the
rows arrive when each group's rows are already consecutive (one scan
checks), else hash-partitioned into buckets small enough to sort in memory,
as the HDFE estimators' pass 0 does with its streamed dimension.

All queries go through a `PolarsEngine`: the CPU streaming engine, or with
``polars_engine="gpu"`` cudf-polars, falling back to the CPU for any query
the GPU engine cannot run.
"""

from __future__ import annotations

import concurrent.futures
import shutil
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq

__all__ = ["Block", "BlockStore", "PolarsEngine", "block_rows", "factorize", "iter_batches",
           "iter_groups", "not_missing", "write_sample"]


# --------------------------------------------------------------------------
# Polars queries, on the CPU or the GPU engine
# --------------------------------------------------------------------------

class PolarsEngine:
    """Runs an estimator's Polars queries: on the CPU streaming engine, or
    with ``polars_engine="gpu"`` on cudf-polars, falling back to the CPU
    (and recording the query and the reason in `fallbacks`) for any query
    the GPU engine cannot run."""

    def __init__(self, polars_engine: str = "cpu"):
        if polars_engine not in ("cpu", "gpu"):
            raise ValueError("polars_engine must be 'cpu' or 'gpu'")
        if polars_engine == "gpu":
            from .gpu import check_polars_gpu
            check_polars_gpu()
        self.name = polars_engine
        self.fallbacks: list[dict] = []
        self._gpu = None

    def _gpu_engine(self):
        if self._gpu is None:
            self._gpu = pl.GPUEngine(raise_on_fail=True)
        return self._gpu

    def _failed(self, what, err):
        reason = (str(err).strip().splitlines() or [type(err).__name__])[0][:200]
        self.fallbacks.append({"query": what, "reason": reason})

    def collect(self, lf: pl.LazyFrame, what: str = "query",
                engine: str = "streaming") -> pl.DataFrame:
        """`lf` collected; `what` names it in `fallbacks`, and `engine` is
        the CPU engine to use."""
        if self.name == "gpu":
            try:
                return lf.collect(engine=self._gpu_engine())
            except Exception as err:            # noqa: BLE001 -- any: run on the CPU
                self._failed(what, err)
        return lf.collect(engine=engine)

    def sink(self, lf: pl.LazyFrame, target, what: str = "sink", **kw) -> None:
        """`lf` sunk to Parquet at `target` (a path, or a
        ``pl.PartitionBy``), with `kw` passed to ``sink_parquet``."""
        if self.name == "gpu":
            try:
                lf.sink_parquet(target, engine=self._gpu_engine(), **kw)
                return
            except Exception as err:            # noqa: BLE001
                self._failed(what, err)
                if isinstance(target, (str, Path)):
                    Path(target).unlink(missing_ok=True)
                else:
                    shutil.rmtree(target.base_path if hasattr(target, "base_path")
                                  else str(target), ignore_errors=True)
        lf.sink_parquet(target, engine="streaming", **kw)


# --------------------------------------------------------------------------
# the store
# --------------------------------------------------------------------------

@dataclass
class Block:
    """One block: its row count, and its arrays' files (on disk) or the
    arrays themselves (in memory), by field name."""

    n: int
    files: dict = field(default_factory=dict)
    arrays: dict = field(default_factory=dict)


class BlockStore:
    """Blocks of rows, on disk under `directory` or in memory.

    A block is a dict of arrays, one row per row of the block, by field name;
    which fields there are is the estimator's business. The field `X` (the
    design) is special in two ways: it is stored C-ordered in `dtype`
    (float32 halves the disk), and read back as float64. With `binary`, a
    mask of X's columns that hold only 0s and 1s, those columns are stored
    apart as one byte each (field ``"X~b"``) and X is put back together as
    it is read.
    """

    def __init__(self, directory: Path, in_memory: bool, dtype=np.float64,
                 prefetch: bool = True, binary: Sequence[bool] | None = None):
        self.dir = Path(directory)
        self.in_memory = bool(in_memory)
        self.dtype = np.dtype(dtype)
        self.prefetch = prefetch
        self.blocks: list[Block] = []
        self.bytes = 0
        b = None if binary is None else np.asarray(binary, dtype=bool)
        self.binary = b if b is not None and b.any() else None
        if self.binary is not None:
            self._dcols = np.flatnonzero(~self.binary)
            self._bcols = np.flatnonzero(self.binary)
        if not self.in_memory:
            self.dir.mkdir(parents=True, exist_ok=True)

    @property
    def n_rows(self) -> int:
        return sum(b.n for b in self.blocks)

    def add(self, arrays: dict) -> None:
        """Append a block. Fields that are None are left out; a block of no
        rows is not stored."""
        arrays = {k: v for k, v in arrays.items() if v is not None}
        if not arrays:
            return
        n = len(arrays["X"] if "X" in arrays else next(iter(arrays.values())))
        if "X" in arrays:
            X = arrays["X"]
            if self.binary is not None:
                if X.shape[1] != len(self.binary):
                    raise ValueError(f"X has {X.shape[1]} columns, binary= {len(self.binary)}")
                arrays["X~b"] = np.ascontiguousarray(X[:, self.binary], dtype=np.uint8)
                X = X[:, ~self.binary]
            arrays["X"] = np.ascontiguousarray(X, dtype=self.dtype)
        if n == 0:
            return
        blk = Block(n=n)
        if self.in_memory:
            blk.arrays = arrays
            self.bytes += sum(a.nbytes for a in arrays.values())
        else:
            i = len(self.blocks)
            for k, a in arrays.items():
                path = self.dir / f"b{i:06d}_{k}.npy"
                np.save(path, a)
                blk.files[k] = path
                self.bytes += path.stat().st_size
        self.blocks.append(blk)

    def _load(self, blk: Block, fields=None) -> dict:
        src = blk.arrays if self.in_memory else blk.files
        if fields is not None and "X" in fields:
            fields = [*fields, "X~b"]
        keys = src.keys() if fields is None else [k for k in fields if k in src]
        out = {k: (src[k] if self.in_memory else np.load(src[k])) for k in keys}
        if "X" in out and "X~b" not in out and out["X"].dtype != np.float64:
            out["X"] = out["X"].astype(np.float64)
        return out

    def _whole(self, out: dict) -> dict:
        """X put back together from its stored parts (in the consuming
        thread: numba's parallel kernels are not to be launched from two
        threads at once)."""
        if "X~b" in out:
            from .kernels_base import _nb_unpack
            Xb = out.pop("X~b")
            X = np.empty((len(Xb), len(self.binary)))
            _nb_unpack(out["X"], self._dcols, Xb, self._bcols, X)
            out["X"] = X
        return out

    def __iter__(self):
        return self.iterate()

    def iterate(self, fields: Sequence[str] | None = None, raw: bool = False) -> Iterator[dict]:
        """The blocks in order (only `fields` of each, if given), each read
        while the previous one is worked on (file reads release the GIL, and
        so do the BLAS products). With `raw`, X as stored: with `binary`,
        its other columns in `X` and the 0/1 ones in ``"X~b"`` (for a
        consumer that puts it together elsewhere, as on a GPU)."""
        whole = (lambda b: b) if raw else self._whole
        if self.in_memory or not self.prefetch or len(self.blocks) < 2:
            for blk in self.blocks:
                yield whole(self._load(blk, fields))
            return
        with concurrent.futures.ThreadPoolExecutor(1) as pool:
            nxt = pool.submit(self._load, self.blocks[0], fields)
            for i in range(len(self.blocks)):
                cur = nxt.result()
                if i + 1 < len(self.blocks):
                    nxt = pool.submit(self._load, self.blocks[i + 1], fields)
                yield whole(cur)
                del cur

    def delete(self) -> None:
        """Forget the blocks and remove their files."""
        self.blocks = []
        if not self.in_memory:
            shutil.rmtree(self.dir, ignore_errors=True)


# --------------------------------------------------------------------------
# pass 0 helpers
# --------------------------------------------------------------------------

def block_rows(block_mb: float, width: int) -> int:
    """Rows per block for blocks of about `block_mb` MB of float64 at
    `width` columns (plus a few per-row arrays); at least 1,024."""
    return max(1024, int(block_mb * 1e6 / (8 * (width + 4))))


def not_missing(schema, cols: Sequence[str]) -> list[pl.Expr]:
    """For each column, an expression true where it is neither null nor,
    for a float column, NaN."""
    out = []
    for c in cols:
        e = pl.col(c).is_not_null()
        if schema[c].is_float():
            e = e & pl.col(c).is_not_nan()
        out.append(e)
    return out


def factorize(engine: PolarsEngine, frame: pl.LazyFrame, cols: Sequence[str], what: str,
              code: str = "code") -> pl.DataFrame:
    """The distinct values of `cols` in `frame`, sorted, with codes 0, 1, ...
    (Int64) in a column `code`: a table of levels, held in memory. Join it
    back on `cols` to code rows."""
    return engine.collect(frame.select(cols).unique().sort(cols)
                          .with_row_index(code).with_columns(pl.col(code).cast(pl.Int64)),
                          f"levels of {what}")


def write_sample(engine: PolarsEngine, source: pl.LazyFrame, keep: pl.Expr,
                 cols: Sequence[str], path, rows: int, row_col: str,
                 drop: tuple[pl.DataFrame, Sequence[str]] | None = None) -> None:
    """Sink the sample -- `cols` of the rows of `source` where `keep` holds,
    with each row's position in the source as an Int64 column `row_col`
    (first) -- to a Parquet file at `path`, in row groups of at most `rows`.
    With `drop`, ``(table, columns)``, the rows matching the table's rows on
    those columns are left out too (an anti-join, after the positions are
    taken, so they stay the source's)."""
    src = source.with_row_index(row_col).with_columns(pl.col(row_col).cast(pl.Int64))
    src = src.filter(keep)
    if drop is not None and drop[0].height:
        src = src.join(drop[0].lazy(), on=list(drop[1]), how="anti", maintain_order="left")
    engine.sink(src.select(pl.col(row_col), *[pl.col(c) for c in cols]),
                str(path), "pass 0 sample", row_group_size=min(rows, 250_000))


def iter_batches(path, rows: int) -> Iterator[pl.DataFrame]:
    """A Parquet file's rows, `rows` at a time, as DataFrames."""
    pf = pq.ParquetFile(path)
    try:
        for batch in pf.iter_batches(batch_size=rows):
            yield pl.from_arrow(batch)
    finally:
        pf.close()


def _runs(cols: Sequence[str]) -> pl.Expr:
    """True on each row that starts a run of equal values of `cols`."""
    return pl.any_horizontal([(pl.col(c) != pl.col(c).shift()).fill_null(True) for c in cols])


def iter_groups(engine: PolarsEngine, path, unit: Sequence[str], rows: int, workdir,
                n_groups: int, sort_by: Sequence[str] | None = None, *,
                also: Sequence[tuple[Sequence[str], int]] = (), seed: int = 20261027,
                log=None) -> Iterator[pl.DataFrame]:
    """The rows of the Parquet file at `path` in frames of whole groups --
    a group being the rows with one value of the `unit` columns, which `path`
    has `n_groups` of -- each group's rows consecutive and, within a frame,
    in the order of `sort_by` (default `unit`).

    When each group's rows are consecutive in the file already (one scan
    checks: as many runs of `unit` as groups -- and of each further grouping
    in `also`, ``(columns, number of groups)`` pairs, as of choice sets
    within panels), the frames are its rows in order, about `rows` at a time (more when a group is larger). Otherwise
    the rows are hash-partitioned by group into buckets of at most about
    ``8 * rows`` rows (and at least a million) under `workdir`, and each
    bucket is sorted by `sort_by` and yielded in frames of about `rows`;
    the buckets are removed afterwards. `log`, if given, is told which.
    """
    sort_by = list(sort_by or unit)
    starts = _runs(unit)
    checks = [(list(unit), n_groups), *[(list(c), n) for c, n in also]]
    runs = engine.collect(pl.scan_parquet(path).select(
        [_runs(c).sum().alias(f"r{i}") for i, (c, _) in enumerate(checks)]), "group runs")
    in_order = all(int(runs[f"r{i}"].item()) == n for i, (_, n) in enumerate(checks))

    def frames(source):
        """Frames of whole groups from `source` (frames in which groups are
        consecutive): each frame's last group is held back until the next
        frame shows where it ends."""
        carry = None
        for frame in source:
            if carry is not None and carry.height:
                frame = pl.concat([carry, frame], how="vertical_relaxed")
            last = int(np.flatnonzero(frame.select(starts.alias("b"))["b"].to_numpy())[-1])
            if last == 0:
                carry = frame               # one group so far: keep accumulating
                continue
            yield frame.slice(0, last)
            carry = frame.slice(last)
        if carry is not None and carry.height:
            yield carry

    if in_order:
        # each group's rows are consecutive; put them in `sort_by`'s order
        # within the group (the groups stay in the file's order)
        rest = [c for c in sort_by if c not in unit]
        if not rest:
            yield from frames(iter_batches(path, rows))
            return
        run = "__store_run"
        for f in frames(iter_batches(path, rows)):
            yield (f.with_columns(starts.cum_sum().alias(run))
                   .sort([run, *rest], maintain_order=True).drop(run))
        return
    n_obs = pq.ParquetFile(path).metadata.num_rows
    P = max(1, -(-n_obs // max(rows * 8, 1_000_000)))
    if log is not None:
        log(f"pass 0: the groups are not in consecutive rows; partitioning them into {P} "
            "buckets to sort")
    bdir = Path(workdir) / "group_buckets"
    shutil.rmtree(bdir, ignore_errors=True)
    key = pl.col(unit[0]) if len(unit) == 1 else pl.struct(list(unit))
    bucket = "__store_bucket"
    engine.sink(pl.scan_parquet(path).with_columns((key.hash(seed=seed) % P).cast(pl.UInt32)
                                                   .alias(bucket)),
                pl.PartitionBy(str(bdir), key=bucket, include_key=False), "pass 0 partition")
    try:
        for b in range(P):
            d = bdir / f"{bucket}={b}"
            if not d.exists():
                continue
            frame = pl.scan_parquet(str(d / "*.parquet")).sort(sort_by).collect(engine="in-memory")
            yield from frames(frame.slice(off, rows) for off in range(0, frame.height, rows))
            del frame
    finally:
        shutil.rmtree(bdir, ignore_errors=True)

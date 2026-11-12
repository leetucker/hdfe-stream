"""Bounded, disk-backed resampling of whole groups with replacement.

`DiskGroupResampler` stages a Parquet file by group (with `core.partition`), then
writes resamples of N groups drawn uniformly with replacement, each copy of a
group with its own ID, in bounded memory. The groups are units, or clusters of
units (``within``), whose units are then given IDs of their own in each copy.
Consumers own the sample selection and the model; the refit bootstrap
(`core.refit`) is built on it.
"""

import hashlib
import itertools
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq

from .partition import GroupPartitions, Part


@dataclass
class ResampleLeaf:
    """A staged piece of the source: its path and the range of its group indices."""

    path: Path
    start: int
    stop: int


class DiskGroupResampler(GroupPartitions):
    """Stage ``snapshot`` by ``key`` and draw whole-group resamples from it.

    The source is partitioned in bounded memory (``budget``, ``preprocess_mb``,
    scratch in ``root``); each piece is sorted by ``sort_columns``, filtered by
    ``select(frame)`` (the consumer's sample rules, applied a piece at a time),
    and its groups are numbered in ``index_column``. ``n_units`` is the number of
    groups and ``n_rows`` of rows; ``identity`` hashes the group order, which with
    a seed fixes the draws.

    With ``within`` (a unit column nested in the ``key`` clusters), clusters are
    drawn, and in each copy every unit gets an ID of its own: ``draw_column`` is
    then the copy times the largest number of units in a cluster plus the unit's
    rank within its cluster, so that it can serve as the unit ID of a refit.
    ``n_within`` counts the units.
    """

    def __init__(
        self,
        snapshot,
        root,
        budget,
        preprocess_mb,
        *,
        key,
        sort_columns,
        select,
        index_column,
        draw_column,
        within=None,
    ):
        super().__init__(root, budget, preprocess_mb, key=key)
        self.sort_columns, self.select = sort_columns, select
        self.index_column, self.draw_column = index_column, draw_column
        self.within = within
        self.ordinal_column = index_column + "_within"
        self.max_within = 1
        self.n_within = 0
        self.leaves = []
        self.n_units = self.n_rows = 0
        self.identity = hashlib.sha256()
        size = 0
        with pq.ParquetFile(snapshot, pre_buffer=False) as pf:
            rows = pf.metadata.num_rows
            for batch in pf.iter_batches(batch_size=self.read_rows):
                self._guard(32e6 + batch.nbytes * 3)
                size += batch.nbytes
        self._prepare(Part(Path(snapshot), rows, size), 0)
        self.ends = np.array([leaf.stop for leaf in self.leaves], dtype=np.int64)

    def _leaf(self, part):
        self._guard(part.size * 4 + 32e6)
        with pq.ParquetFile(part.path, pre_buffer=False) as pf:
            data = pl.from_arrow(pf.read()).sort(self.sort_columns)
        data = self.select(data)
        if len(data):
            ids = data[self.key].to_numpy()
            starts = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1]
            lengths = np.diff(np.r_[starts, len(data)])
            for value in ids[starts]:
                encoded = repr(
                    value.item() if isinstance(value, np.generic) else value
                ).encode()
                self.identity.update(len(encoded).to_bytes(8, "little"))
                self.identity.update(encoded)
            data = data.with_columns(
                pl.Series(
                    self.index_column,
                    np.repeat(
                        np.arange(self.n_units, self.n_units + len(starts)), lengths
                    ),
                )
            )
            if self.within is not None:
                # each unit's rank within its cluster (rows are sorted by cluster, unit)
                data = data.with_columns(
                    (pl.col(self.within).rank("dense").over(self.key) - 1)
                    .cast(pl.Int64)
                    .alias(self.ordinal_column)
                )
                counts = data.group_by(self.key).agg(pl.col(self.within).n_unique())
                self.max_within = max(self.max_within, int(counts[self.within].max()))
                self.n_within += int(counts[self.within].sum())
            path = self._name("resample-source")
            data.write_parquet(path, row_group_size=32768)
            self.leaves.append(
                ResampleLeaf(path, self.n_units, self.n_units + len(starts))
            )
            self.n_units += len(starts)
            self.n_rows += len(data)
            self.partition_count += 1
            self.max_leaf_bytes = max(self.max_leaf_bytes, part.size)
        part.path.unlink()

    def materialize(self, rng, root, *, time_column, chunk_rows=32768):
        """Write one exact N-group resample. Return path, dates, rows, plan hash.

        Integer plans are routed to leaf-specific binary files, never collected
        into an N-row in-memory manifest. Output buffers and join multiplicity
        are bounded separately from the number of groups in the population.
        ``time_column`` (or None) names the column whose distinct values are
        returned as the dates the resample covers.
        """
        root = Path(root)
        root.mkdir()
        digest = hashlib.sha256()
        for offset in range(0, self.n_units, chunk_rows):
            count = min(chunk_rows, self.n_units - offset)
            indices = rng.integers(0, self.n_units, size=count, dtype=np.int64)
            digest.update(indices.astype("<i8", copy=False).tobytes())
            buckets = np.searchsorted(self.ends, indices, side="right")
            order = np.argsort(buckets, kind="stable")
            ordered = buckets[order]
            starts = np.r_[0, np.flatnonzero(ordered[1:] != ordered[:-1]) + 1, count]
            for a, b in itertools.pairwise(starts):
                chosen = order[a:b]
                pairs = np.column_stack([indices[chosen], offset + chosen]).astype(
                    "<i8"
                )
                with (root / f"plan-{ordered[a]}.bin").open("ab") as f:
                    pairs.tofile(f)
        output = root / "resample.parquet"
        writer, pending, pending_rows, pending_bytes = None, [], 0, 0
        dates, rows = set(), 0
        try:
            for j, leaf in enumerate(self.leaves):
                plan = root / f"plan-{j}.bin"
                if not plan.exists():
                    continue
                with pq.ParquetFile(leaf.path, pre_buffer=False) as pf:
                    data = pl.from_arrow(pf.read())
                ids = data[self.index_column].to_numpy()
                starts = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1]
                lengths = np.diff(np.r_[starts, len(data)])
                leaf_bytes = data.estimated_size()

                def blocks(
                    pairs, leaf=leaf, lengths=lengths, leaf_bytes=leaf_bytes, data=data
                ):
                    counts = np.bincount(
                        pairs[:, 0] - leaf.start, minlength=leaf.stop - leaf.start
                    )
                    nrows = int(counts @ lengths)
                    upper = leaf_bytes * int(counts.max()) + 16 * nrows
                    if upper > self.leaf_bytes * 2 and len(pairs) > 1:
                        mid = len(pairs) // 2
                        yield from blocks(pairs[:mid])
                        yield from blocks(pairs[mid:])
                        return
                    self._guard(4 * upper + 2 * leaf_bytes + 32e6)
                    selection = pl.DataFrame(
                        {self.index_column: pairs[:, 0], self.draw_column: pairs[:, 1]}
                    )
                    frame = data.join(
                        selection, on=self.index_column, how="inner"
                    ).drop(self.index_column)
                    if self.within is not None:
                        frame = frame.with_columns(
                            (
                                pl.col(self.draw_column) * self.max_within
                                + pl.col(self.ordinal_column)
                            ).alias(self.draw_column)
                        ).drop(self.ordinal_column)
                    yield frame

                with plan.open("rb") as f:
                    while True:
                        pairs = np.fromfile(f, dtype="<i8", count=512).reshape(-1, 2)
                        if not len(pairs):
                            break
                        for frame in blocks(pairs):
                            if time_column is not None:
                                dates.update(frame[time_column].unique().to_list())
                            rows += len(frame)
                            if writer is None:
                                writer = pq.ParquetWriter(
                                    output,
                                    frame.to_arrow().schema,
                                    compression="zstd",
                                    use_dictionary=False,
                                )
                            pending.append(frame)
                            pending_rows += len(frame)
                            pending_bytes += frame.estimated_size()
                            if (
                                pending_rows >= 32768
                                or pending_bytes >= self.leaf_bytes * 2
                                or len(pending) >= 128
                            ):
                                writer.write_table(
                                    pl.concat(pending).to_arrow(), row_group_size=32768
                                )
                                pending, pending_rows, pending_bytes = [], 0, 0
                plan.unlink()
            if pending:
                writer.write_table(pl.concat(pending).to_arrow(), row_group_size=32768)
        finally:
            if writer is not None:
                writer.close()
        return output, sorted(dates), rows, digest.hexdigest()

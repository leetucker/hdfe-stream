"""Connected components of a pair of fixed-effect dimensions:
`connected_components`, and the estimator's `connected="largest"`.

The graph has one vertex per level of each of the two dimensions, and each
row joins its two levels. Its components are found by union-find over the
rows as they stream (the kernels in kernels_base and kernels_graph), so
without pruning only level-sized arrays are held: the level tables, a hash
and a code per level to look rows up by, and a parent and a row count per
level; never the rows, nor even the distinct pairs. Pruning by degree
(`min_degree`) needs the distinct pairs, which are then held in memory as one
sorted 64-bit key each, with a 32-bit row count.

Levels are coded in sorted order, which is what makes the labels a function of
the data alone: union-find links toward the smaller code, so a component's
root is its smallest level of the first dimension whatever the order of the
rows, and that is the tie-break when components are ranked by size."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import polars as pl

from ._columns import PREFIX, check_source
from ._types import Source
from .feterms import _parse_fe, _parse_fe_term
from .kernels_base import _nb_union_pairs
from .kernels_graph import (_nb_add_into, _nb_articulation_points, _nb_bfs_components,
                            _nb_count_into, _nb_dense_labels, _nb_hash_lookup,
                            _nb_level_rows, _nb_merge_pairs, _nb_pair_rows,
                            _nb_pairs_degree, _nb_pairs_fill, _nb_peel, _nb_relabel,
                            _nb_union_keys)
from .report import _log
from .utils import _as_lazy, _not_nan

_CODE = (f"{PREFIX}cc0", f"{PREFIX}cc1")     # level codes (or hashes) of the two dimensions
COMPONENT, IN_LARGEST = "component", "in_largest"


@dataclass
class ConnectedComponents:
    """The connected components of the graph of two fixed-effect dimensions,
    as returned by `connected_components`: small level tables, not rows.

    Attributes
    ----------
    fe : (str, str)
        The two dimensions, e.g. ``("worker_id", "firm_id^year")``.
    levels : dict of str to polars.DataFrame
        ``{dimension: DataFrame}``: every level of the dimension found in a
        complete row (its column, or columns for an interaction), and its
        ``component``. Components are numbered by size, so
        ``component == 0`` is the largest; levels removed by ``min_degree`` or
        ``leave_one_out`` have a null component.
    sizes : polars.DataFrame
        One row per component, in order: ``component``, ``rows``, and the
        number of levels of each dimension (a column named after it).
    min_degree : (int, int)
        (first dimension, second dimension), as used.
    leave_one_out : bool
        Whether this is the leave-one-out connected set.
    diagnostics : dict
        Counts and timings (see `summary`).
    """

    fe: tuple[str, str]
    levels: dict[str, pl.DataFrame]
    sizes: pl.DataFrame
    min_degree: tuple[int, int] = (1, 1)
    leave_one_out: bool = False
    diagnostics: dict = field(default_factory=dict)
    # with both min_degree and leave_one_out: which levels min_degree kept,
    # so a fit can say which of the two dropped a row
    _after_min_degree: dict | None = field(default=None, repr=False)
    # the data the components were found in, as given (a LazyFrame: a plan,
    # not rows), which largest() and label() use when given no data
    _source: pl.LazyFrame | None = field(default=None, repr=False)

    @property
    def _pruned(self):
        """Whether levels were removed, so that a row is in the set only if
        both of its levels are."""
        return self.min_degree != (1, 1) or self.leave_one_out

    @property
    def n_components(self) -> int:
        """The number of components."""
        return self.sizes.height

    def _cols(self, d):
        return [c.strip() for c in d.split("^")]

    def _data(self, data):
        """`data` as a LazyFrame, or the data the components were found in."""
        if data is not None:
            return _as_lazy(data)
        if self._source is None:
            raise ValueError("these components hold no reference to the data they were "
                             "found in; pass the data")
        return self._source

    def _complete(self, data):
        """`data` without the rows that have a null or NaN in a column of
        either dimension (they are in no component)."""
        lf = self._data(data)
        cols = list(dict.fromkeys(c for d in self.fe for c in self._cols(d)))
        missing = [c for c in cols if c not in lf.collect_schema()]
        if missing:
            raise ValueError(f"columns not found in data: {missing}")
        nan = _not_nan(lf.collect_schema(), cols)
        lf = lf.drop_nulls(cols)
        return lf.filter(pl.all_horizontal(nan)) if nan else lf

    def _join(self, lf, table, d, how):
        """`lf` joined with `table`, which holds levels of `d` (and perhaps
        more columns), on those levels, keeping `lf`'s order."""
        return lf.join(table.lazy(), on=self._cols(d), how=how, maintain_order="left")

    def largest(self, data: Source | None = None) -> pl.LazyFrame:
        """The rows of the data in the largest component, lazily and in their
        original order: the largest connected set.

        By default the data is the data the components were found in, with
        all its columns. Other `data` must be a subset of it, or at least
        have the same levels: each dimension is filtered with whichever is smaller, its levels
        in the largest component (a semi-join) or the rest (an anti-join), so
        a level that was never seen may be let through. Without
        `min_degree`, one dimension's table settles it, since both levels of
        a row are in the same component."""
        lf = self._complete(data)
        choices = []
        for d in self.fe:
            t = self.levels[d]
            inside = t.filter(pl.col(COMPONENT) == 0).drop(COMPONENT)
            outside = t.filter(pl.col(COMPONENT).fill_null(1) != 0).drop(COMPONENT)
            choices.append(min((inside, "semi", d), (outside, "anti", d),
                               key=lambda c: c[0].height))
        if not self._pruned:
            choices = [min(choices, key=lambda c: c[0].height)]
        for table, how, d in choices:
            lf = self._join(lf, table, d, how)
        return lf

    def label(self, data: Source | None = None) -> pl.LazyFrame:
        """The data (by default the data the components were found in, as
        for `largest`), lazily and in its original order, with two columns
        added: ``component`` (UInt32), the row's component, null when the row
        is in none (a null or NaN level, or a level removed by ``min_degree``
        or ``leave_one_out``); and ``in_largest``, whether the row is in the
        largest component.

        It joins the full level table of the dimension with fewer levels
        (and, after pruning, the smaller list of the other's removed or
        surviving levels), so it holds that table, not the rows."""
        lf = self._data(data)
        clash = [c for c in (COMPONENT, IN_LARGEST) if c in lf.collect_schema()]
        if clash:
            raise ValueError(f"the data has columns named {clash}, which label() adds; "
                             "rename them")
        small, other = sorted(self.fe, key=lambda d: self.levels[d].height)
        lf = self._join(lf, self.levels[small], small, "left")
        if self._pruned:
            t = self.levels[other]
            dead = t.filter(pl.col(COMPONENT).is_null()).select(self._cols(other))
            alive = t.filter(pl.col(COMPONENT).is_not_null()).select(self._cols(other))
            flag = f"{PREFIX}alive"
            listed, is_alive = (dead, False) if dead.height <= alive.height else (alive, True)
            lf = self._join(lf, listed.with_columns(pl.lit(is_alive).alias(flag)), other,
                            "left")
            lf = (lf.with_columns(pl.when(pl.col(flag).fill_null(not is_alive))
                                    .then(pl.col(COMPONENT)).alias(COMPONENT))
                    .drop(flag))
        return lf.with_columns((pl.col(COMPONENT) == 0).fill_null(False).alias(IN_LARGEST))

    def summary(self) -> str:
        """A short report: the components, the largest, and what pruning
        removed."""
        d, s = self.diagnostics, self.sizes
        d0, d1 = self.fe
        lines = [f"connected components of {d0} x {d1}"]
        if self.min_degree != (1, 1):
            lines.append(f"  min_degree={self.min_degree} removed "
                         f"{d['pruned_levels'][d0]:,} {d0}, {d['pruned_levels'][d1]:,} {d1} "
                         f"and {d['pruned_rows']:,} rows in {d['pruning_rounds']} round(s)")
        if self.leave_one_out:
            lo = d["leave_one_out"]
            lines.append(f"  leave-one-out connected set: from {lo['components_before']:,} "
                         f"components, kept the largest; removed "
                         f"{lo['single_observation']:,} {d0} observed once and "
                         f"{lo['cut']:,} {d0} whose removal would disconnect it, in "
                         f"{lo['rounds']} round(s); {lo['levels'][d0]:,} {d0}, "
                         f"{lo['levels'][d1]:,} {d1} and {lo['rows']:,} rows removed")
        lines.append(f"  components: {self.n_components:,}")
        rows = d["rows"]
        if self.n_components:
            top = s.row(0, named=True)
            lines.append(f"  largest: {top['rows']:,} of {rows:,} rows "
                         f"({top['rows'] / max(rows, 1):.1%}), "
                         f"{top[d0]:,} of {d['levels'][d0]:,} {d0}, "
                         f"{top[d1]:,} of {d['levels'][d1]:,} {d1}")
        if self.n_components > 1:
            nxt = s.row(1, named=True)
            lines.append(f"  next largest: {nxt['rows']:,} rows")
        return "\n".join(lines)


def connected_components(data: Source, fe: str | Sequence[str], *,
                         min_degree: int | tuple[int, int] = 1, leave_one_out: bool = False,
                         rank_by: str = "rows", batch_rows: int = 2_000_000,
                         verbose: bool = True,
                         logger: logging.Logger | None = None,
                         log_level: int = logging.INFO) -> ConnectedComponents:
    """Connected components of the graph of two fixed-effect dimensions, and
    so the largest connected set.

    The graph has one vertex per level of each dimension, and each row joins
    its two levels. Its components are found by union-find over the rows as
    they stream.

    Parameters
    ----------
    data : str, Path, polars.LazyFrame or polars.DataFrame
        Parquet path or glob, Polars LazyFrame, or DataFrame. It is read in
        batches and never held in memory.
    fe : str or (str, str)
        The two dimensions, as a formula's fixed-effect part
        (``"worker_id + firm_id"``) or a pair of names
        (``("worker_id", "firm_id")``). Either may be an interaction
        (``"firm_id^year"``). Rows with a null or NaN in any of their columns
        are in no component.
    min_degree : int or (int, int), default 1
        Remove every level with fewer distinct partners in the other dimension
        (distinct firms for a worker, distinct workers for a firm) than this,
        repeating until none is left, before finding the components. An int
        applies to both dimensions; a pair is (first, second). ``(2, 1)``
        keeps only movers: workers seen at two or more firms, and the firms
        they were seen at. ``2`` also drops firms with a single worker, and so
        on until every worker and firm left has two partners. ``(1, k)`` keeps
        firms with at least k distinct workers. Partners are distinct levels,
        so a worker with many rows at one firm has one; with
        ``firm_id^year``, a worker who stays at a firm for two years has two.
        The default, 1, removes nothing. ``min_degree=2`` is not the
        leave-one-out connected set: see below.
    leave_one_out : bool, default False
        Prune to the leave-one-out connected set that Kline-Saggio-Solvsten
        estimation needs (`leave_out_kss` applies it), in which every effect
        stays estimable when any one worker is dropped. The first dimension
        plays the workers. Following LeaveOutTwoWay's ``pruning_unbal_v3.m``
        and VarianceComponentsHDFE.jl:

        1. take the largest connected component;
        2. drop workers observed only once (their leverage is exactly 1);
        3. remove every worker whose removal would split the network (an
           articulation point), and take the largest piece left;
        4. repeat 3 until no such worker is left.

        Step 3 repeats because removing one such worker can make another. The
        result is one component, 0; every other level's component is null.
        Unlike ``min_degree=2`` it keeps stayers seen at least twice, and it
        removes movers that ``min_degree=2`` keeps, so neither set contains
        the other, although every mover and firm it keeps ``min_degree=2``
        keeps too. With ``min_degree``, the pruning by degree comes first.
        Expect it to remove a lot: in KSS's own application, about half the
        firms, mostly ones attached to a single mover.
    rank_by : str, default "rows"
        What "largest" means: ``"rows"``, or the name of either dimension to
        count its levels. Ties go to the component whose smallest level of
        the first dimension sorts first.
    batch_rows : int, default 2_000_000
        Rows per batch as the data streams.
    verbose, logger, log_level
        Progress messages, as for `feols_stream`.

    Returns
    -------
    ConnectedComponents
        Small level tables, not rows. Its `ConnectedComponents.largest` and
        `ConnectedComponents.label` apply them to the data lazily; it keeps a
        reference to `data` for that (a DataFrame is then kept alive with it),
        and either takes other data with the same levels instead.

    Notes
    -----
    Pruning comes first, then the components: the components are those of
    what is left, so the largest one is connected and every level in it has
    at least ``min_degree`` partners in it. (Taking the largest component
    first and pruning it can disconnect it, once ``min_degree`` exceeds 2.)

    **Memory.** Without pruning: the level tables themselves, about 20 bytes
    more per level, and a batch of rows. Pruning also holds the distinct
    pairs, at 12 bytes each, and briefly about twice that while they are
    merged; ``leave_one_out`` adds the network's adjacency, about 8 bytes per
    pair more, and 50 per level. In practice the peak is set by Polars
    finding the distinct levels and reading the Parquet: on 150 million rows
    with 25 million 9-character worker ids, about 1.3 GB above what a bare
    scan of the two columns takes.

    **Three or more dimensions.** Connectedness of one pair is necessary for
    the effects to be identified but not sufficient: a further dimension can
    make levels of the pair redundant that no graph of the pair shows (year
    effects, say, when every move between firms coincides with a change of
    year). The components here are those of the pair only; they are not
    combined with any other dimension, which would be wrong rather than
    conservative, since two firms sharing years are not comparable unless a
    worker moves between them.

    **In a fit.** The same set is what ``feols_stream(...,
    connected="largest")`` keeps, without pruning, of the first two fixed
    effects listed in the formula, but found among the rows the model can use
    (rows with a missing outcome or covariate do not connect anything there).
    Filter those out before calling this to get the same rows.

    Examples
    --------
    >>> cc = connected_components("data/*.parquet", "worker_id + firm_id")
    >>> cc.sizes                       # rows and levels per component
    >>> akm = cc.largest()             # the largest connected set, lazily
    >>> rows = cc.label()              # every row, with its component
    """
    dims, fe_cols = _parse_pair(fe)
    k = _min_degree(min_degree)
    if rank_by != "rows" and rank_by not in dims:
        raise ValueError(f"rank_by must be 'rows' or one of {list(dims)}, got {rank_by!r}")
    lf = _as_lazy(data)
    schema = lf.collect_schema()
    check_source(schema.names())
    missing = [c for d in dims for c in fe_cols[d] if c not in schema]
    if missing:
        raise ValueError(f"columns not found in data: {missing}")
    cc = _find_components(lf, dims, fe_cols, k, rank_by, batch_rows,
                          lambda msg: _log(verbose, msg, logger, log_level),
                          leave_one_out=bool(leave_one_out))
    cc._source = lf
    return cc


def _parse_pair(fe):
    """("worker_id", "firm_id^year") and the columns of each, from a string
    "worker_id + firm_id^year" or a sequence of two terms."""
    if isinstance(fe, str):
        if "~" in fe or "|" in fe:
            raise ValueError(f"fe={fe!r}: pass only the fixed effects, e.g. "
                             "'worker_id + firm_id'")
        terms = _parse_fe(fe)
    else:
        terms = list(fe)
    if len(terms) != 2:
        raise ValueError(f"fe must name exactly two fixed-effect dimensions, got {terms}. "
                         "Components are those of a pair; with three or more "
                         "dimensions, pass the pair whose connectedness matters "
                         "(e.g. workers and firms)")
    dims = []
    for t in terms:
        name, slopes = _parse_fe_term(t)
        if slopes:
            raise ValueError(f"{t!r}: varying slopes have no bearing on which levels are "
                             f"connected; pass {name!r}")
        dims.append(name)
    if dims[0] == dims[1]:
        raise ValueError(f"the two dimensions are the same: {dims[0]!r}")
    return tuple(dims), {d: [c.strip() for c in d.split("^")] for d in dims}


def _min_degree(min_degree):
    k = ((min_degree, min_degree) if isinstance(min_degree, (int, np.integer))
         else tuple(min_degree))
    if (len(k) != 2 or not all(isinstance(v, (int, np.integer)) and not isinstance(v, bool)
                               and v >= 1 for v in k)):
        raise ValueError("min_degree must be a positive integer or a pair of them, "
                         f"got {min_degree!r}")
    return int(k[0]), int(k[1])


def _find_components(lf: pl.LazyFrame, dims: tuple[str, str], fe_cols: dict,
                     min_degree: tuple[int, int] = (1, 1), rank_by: str = "rows",
                     batch_rows: int = 2_000_000,
                     log: Callable[[str], None] = lambda msg: None,
                     leave_one_out: bool = False) -> ConnectedComponents:
    """`connected_components` on a LazyFrame whose columns are checked."""
    t0 = time.time()
    d0, d1 = dims
    cols = list(dict.fromkeys(fe_cols[d0] + fe_cols[d1]))
    nan = _not_nan(lf.collect_schema(), cols)
    complete = lf.select(cols).drop_nulls(cols)
    if nan:
        complete = complete.filter(pl.all_horizontal(nan))

    # Codes in sorted order of the levels: what makes the labels, and the
    # tie-break between components of the same size, a function of the data
    # rather than of the order of the rows. One dimension at a time: Polars
    # finds the distinct levels with a hash table per thread, and two at once
    # peaked at twice the memory.
    log(f"connected components: factorizing {d0} and {d1}")
    maps = {d: complete.select(fe_cols[d]).unique().sort(fe_cols[d])
                       .collect(engine="streaming")
            for d in dims}
    n0, n1 = maps[d0].height, maps[d1].height
    index = [_hash_index(maps[d], fe_cols[d]) for d in dims]

    alive = np.ones(n0 + n1, bool)
    diag = {"levels": {d0: n0, d1: n1}, "min_degree": list(min_degree)}
    after_min_degree = None
    if min_degree == (1, 1) and not leave_one_out:
        log(f"connected components: union-find over the rows of {d0} x {d1}")
        acc = _coded_pass(complete, dims, fe_cols, maps, index, batch_rows,
                          lambda: _UnionRows(n0, n1), log)
        parent, small, rows_small = acc.parent, acc.small, acc.rows_small
        del acc
        nc = _nb_dense_labels(parent, alive)
        labels_small = parent[n0:] if small else parent[:n0]
        rows = np.zeros(nc, np.int64)
        _nb_add_into(labels_small, rows_small, rows)
        diag["rows"] = int(rows_small.sum())
        del rows_small
    else:
        log(f"connected components: collecting the distinct pairs of {d0} x {d1}")
        acc = _coded_pass(complete, dims, fe_cols, maps, index, batch_rows,
                          lambda: _DistinctPairs(n1), log)
        keys, counts = acc.result()
        del acc
        diag["rows"] = int(counts.sum(dtype=np.int64))
        diag["distinct_pairs"] = len(keys)
        n1u = np.uint64(max(n1, 1))
        labels = np.int32 if n0 + n1 < 2**31 else np.int64
        if min_degree != (1, 1):
            log(f"connected components: removing levels with fewer than {min_degree} "
                f"distinct partners among {len(keys):,} distinct pairs")
            rounds = _nb_peel(keys, n1u, min_degree[0], min_degree[1], alive[:n0],
                              alive[n0:])
            diag["pruning_rounds"] = int(rounds)
            diag["pruned_levels"] = {d0: int(n0 - alive[:n0].sum()),
                                     d1: int(n1 - alive[n0:].sum())}
            diag["pruned_rows"] = diag["rows"] - _rows_alive(keys, counts, n1u, n0, alive)
        if leave_one_out:
            if min_degree != (1, 1):
                after_min_degree = {d0: alive[:n0].copy(), d1: alive[n0:].copy()}
            before = alive.copy()
            log("connected components: pruning to the leave-one-out connected set")
            lo = _leave_one_out(keys, counts, n1u, n0, n1, alive,
                                {"rows": None, d0: 0, d1: 1}[rank_by])
            lo["levels"] = {d0: int(before[:n0].sum() - alive[:n0].sum()),
                            d1: int(before[n0:].sum() - alive[n0:].sum())}
            lo["rows"] = (_rows_alive(keys, counts, n1u, n0, before)
                          - _rows_alive(keys, counts, n1u, n0, alive))
            diag["leave_one_out"] = lo
            del before
            parent = np.where(alive, 0, -1).astype(labels)
            nc = int(alive.any())
        else:
            log("connected components: union-find over the pairs left")
            parent = np.arange(n0 + n1, dtype=labels)
            _nb_union_keys(parent, keys, n1u, alive[:n0], alive[n0:])
            nc = _nb_dense_labels(parent, alive)
        rows = np.zeros(nc, np.int64)
        _nb_pair_rows(keys, counts, n1u, parent, rows)
        del keys, counts
    del index

    # rank by size; components are numbered by their root, the smallest
    # level of the first dimension, so a stable sort breaks ties by it
    sizes = {"rows": rows}
    for d, lab in ((d0, parent[:n0]), (d1, parent[n0:])):
        sizes[d] = np.zeros(nc, np.int64)
        _nb_count_into(lab, sizes[d])
    order = np.argsort(-sizes[rank_by], kind="stable")
    rank = np.empty(nc, parent.dtype)
    rank[order] = np.arange(nc, dtype=parent.dtype)
    _nb_relabel(parent, rank)

    levels = {}
    for d, lab in ((d0, parent[:n0]), (d1, parent[n0:])):
        comp = pl.Series(COMPONENT, lab)
        levels[d] = (maps[d]
                     .with_columns(pl.when(comp >= 0).then(comp).cast(pl.UInt32)
                                     .alias(COMPONENT)))
    del maps, parent
    table = pl.DataFrame({COMPONENT: np.arange(nc, dtype=np.uint32),
                          "rows": sizes["rows"][order],
                          d0: sizes[d0][order], d1: sizes[d1][order]})
    diag["seconds"] = round(time.time() - t0, 2)
    cc = ConnectedComponents(fe=dims, levels=levels, sizes=table, min_degree=min_degree,
                             leave_one_out=leave_one_out, diagnostics=diag,
                             _after_min_degree=after_min_degree)
    if nc:
        top = table.row(0, named=True)
        log(f"connected components: {nc:,}; the largest has {top['rows']:,} of "
            f"{diag['rows']:,} rows ({top['rows'] / max(diag['rows'], 1):.1%}), "
            f"{top[d0]:,} {d0} and {top[d1]:,} {d1} ({diag['seconds']}s)")
    return cc


def _rows_alive(keys, counts, n1u, n0, alive):
    """Rows whose two levels are both `alive`."""
    out = np.zeros(n0, np.int64)
    _nb_level_rows(keys, counts, n1u, alive, out)
    return int(out.sum())


def _leave_one_out(keys, counts, n1u, n0, n1, alive, rank_by):
    """Prune `alive` (both dimensions, in place) to the leave-one-out
    connected set of the pairs `keys` (see `connected_components`); the
    first dimension plays the workers. `rank_by` is None for rows, else the
    dimension (0 or 1) whose levels measure "largest". Returns the counts.

    The network is the distinct pairs between the levels alive on entry, as
    CSR adjacency in both directions; removals after that only clear `alive`,
    which the traversals skip. Components are numbered by their smallest
    vertex, so `argmax` breaks ties in favor of the component with the
    smallest level of the first dimension, as when components are ranked."""
    nv = n0 + n1
    indptr = np.zeros(nv + 1, np.int64)
    _nb_pairs_degree(keys, n1u, n0, alive, indptr)
    indices = np.empty(indptr[-1], np.int32 if nv < 2**31 else np.int64)
    _nb_pairs_fill(keys, n1u, n0, alive, indptr, indices)
    label = np.empty(nv, np.int64)

    def keep_largest():
        nc = _nb_bfs_components(indptr, indices, alive, label)
        if nc == 0:
            return 0
        size = np.zeros(nc, np.int64)
        if rank_by is None:
            _nb_pair_rows(keys, counts, n1u, label, size)
        else:
            _nb_count_into(label[:n0] if rank_by == 0 else label[n0:], size)
        np.logical_and(alive, label == int(np.argmax(size)), out=alive)
        return nc

    # 1. the largest connected component
    out = {"components_before": keep_largest()}
    # 2. workers observed once: a pendant vertex is never a cut vertex, but
    # its leverage is 1. A worker's rows are all at firms in its component,
    # so after step 1 they are all counted here.
    rows0 = np.zeros(n0, np.int64)
    _nb_level_rows(keys, counts, n1u, alive, rows0)
    single = (rows0 == 1) & alive[:n0]
    del rows0
    out["single_observation"] = int(single.sum())
    if out["single_observation"]:
        alive[:n0] &= ~single
        keep_largest()
    del single
    # 3-4. workers that are articulation points, until there are none
    cut_total, rounds = 0, 0
    is_cut = np.zeros(nv, bool)
    while True:
        is_cut[:] = False
        _nb_articulation_points(indptr, indices, is_cut, alive)
        cut = is_cut[:n0] & alive[:n0]
        n_cut = int(cut.sum())
        if not n_cut:
            break
        cut_total += n_cut
        rounds += 1
        alive[:n0] &= ~cut
        keep_largest()
    out.update(cut=cut_total, rounds=rounds)
    return out


_SEED = 0x5EED_C0DE


class _Miss(Exception):
    """A row's level hash was not among the levels' (see `_coded_pass`)."""


def _key(cols):
    return pl.col(cols[0]) if len(cols) == 1 else pl.struct(cols)


def _hash_index(m, cols):
    """Sorted 64-bit hashes of the levels in `m` (in code order), the codes in
    that order, and an index on the hashes' top bits; None if two levels
    share a hash. With no two alike, a row's hash identifies its level
    exactly, since every row's level is one of these."""
    h = m.select(_key(cols).hash(_SEED)).to_series().to_numpy()
    perm = np.argsort(h).astype(np.uint32 if len(h) < 2**32 else np.uint64)
    sh = h[perm]
    del h
    if len(sh) > 1 and bool((sh[1:] == sh[:-1]).any()):
        return None
    bits = max(1, int(np.log2(max(len(sh), 2))) - 2)   # about 4 levels a bucket
    shift = np.uint64(64 - bits)
    starts = np.zeros((1 << bits) + 1, np.int64)
    np.cumsum(np.bincount((sh >> shift).astype(np.int64), minlength=1 << bits),
              out=starts[1:])
    return sh, perm, starts, shift


def _coded_pass(complete, dims, fe_cols, maps, index, batch_rows, make, log):
    """Stream the rows' level codes, a batch at a time, into the accumulator
    `make()` returns (its `consume(a, b)`), and return the accumulator.

    Codes are found by hashing each row's levels and looking the hash up
    among the levels' (`_hash_index`): about 12 bytes per level. Joining the
    level tables onto the rows gives the same codes, but its hash tables of
    the levels themselves peaked about 1.5 GB higher at 25 million string
    levels. The join remains the fallback, should two levels share a hash
    or a row's hash not be found (neither of which has been seen).

    The batches come through `sink_batches`, which runs inside the streaming
    engine; `collect_batches` hands them out as fast as the scan produces
    them and held all of them at once."""
    acc = make()
    if all(ix is not None for ix in index):
        hashed = complete.select([_key(fe_cols[d]).hash(_SEED).alias(c)
                                  for d, c in zip(dims, _CODE)])

        def consume_hashes(b):
            codes = []
            for c, (sh, perm, starts, shift) in zip(_CODE, index):
                h = b[c].to_numpy()
                out = np.empty(len(h), perm.dtype)
                if _nb_hash_lookup(sh, perm, starts, shift, h, out):
                    raise _Miss
                codes.append(out)
            acc.consume(*codes)
        try:
            hashed.sink_batches(consume_hashes, chunk_size=batch_rows, maintain_order=False)
            return acc
        except _Miss:
            pass
    log("connected components: coding the levels by join")
    acc = make()
    coded = complete
    for d, c in zip(dims, _CODE):
        coded = coded.join(maps[d].lazy().with_row_index(c), on=fe_cols[d])
    coded.select(_CODE).sink_batches(
        lambda b: acc.consume(b[_CODE[0]].to_numpy(), b[_CODE[1]].to_numpy()),
        chunk_size=batch_rows, maintain_order=False)
    return acc


class _UnionRows:
    """Union-find over the rows, and the rows per level of the dimension
    with fewer levels (both levels of a row are in the same component, so
    either dimension's counts give the rows per component)."""

    def __init__(self, n0, n1):
        nv = n0 + n1
        self.n0 = n0
        self.parent = np.arange(nv, dtype=np.int32 if nv < 2**31 else np.int64)
        self.small = n1 <= n0
        self.rows_small = np.zeros(min(n0, n1), np.int64)

    def consume(self, a, b):
        _nb_union_pairs(self.parent, a, b, self.n0)
        _nb_count_into(b if self.small else a, self.rows_small)


class _DistinctPairs:
    """The distinct (first, second) code pairs among the rows, as sorted
    uint64 keys first * n1 + second, and the rows behind each (uint32: a
    single pair of levels does not reach four billion rows).

    Each batch is reduced to its distinct pairs and pushed onto a stack of
    sorted runs; a run is merged into the one below while it is at least half
    that one's size, as in a binary counter, so each pair is merged
    O(log batches) times and memory peaks near twice the distinct pairs."""

    def __init__(self, n1):
        self.n1 = np.uint64(max(n1, 1))
        self.runs = []

    def consume(self, a, b):
        key = a.astype(np.uint64) * self.n1 + b.astype(np.uint64)
        k, c = np.unique(key, return_counts=True)
        del key
        self.runs.append((k, c.astype(np.uint32)))
        while len(self.runs) > 1 and 2 * len(self.runs[-1][0]) >= len(self.runs[-2][0]):
            self._merge()

    def _merge(self):
        (ka, ca), (kb, cb) = self.runs.pop(-2), self.runs.pop()
        self.runs.append(_nb_merge_pairs(ka, ca, kb, cb))

    def result(self):
        while len(self.runs) > 1:
            self._merge()
        if not self.runs:
            return np.zeros(0, np.uint64), np.zeros(0, np.uint32)
        return self.runs.pop()

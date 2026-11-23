"""Within cross-products of a set of variables, split by an integer column.

`cross_products` partials the fixed effects out of every variable, as a fit
does (Frisch-Waugh-Lovell: the same passes and the same solve), and then, in
one more pass over the rows, sums the cross-products of the residualized
variables separately for each value of a split column. With the split
column holding fold numbers, that is one Gram matrix per fold, enough for
K-fold cross-validation of any estimator whose loss is quadratic (a lasso,
ridge or elastic net, say) without reading the rows again; with shard
numbers, enough for subsampling.

Every row is residualized with the full sample's fixed effects. Within a fold
the cross-products are therefore those of the full-sample residuals, not of a
fit to the other folds' rows alone. When the folds are whole groups of the
streamed dimension and it is the only fixed effect, the two are the same: a
group's mean comes from its own rows, all in one fold.

Without fixed effects nothing is grouped or partialled out, so the rows are
not copied to the run directory as a fit copies them: the one pass after
pass 0 reads the source itself, through the same filter and the same
evaluated design.

With ``save_rows=True`` the residualized rows are also written to Parquet,
for estimators that need more passes over them than the cross-products
give (penalty loadings computed from residuals, say), without partialling
the fixed effects out again.
"""

from __future__ import annotations

import gc
import sys
import time
import traceback
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numba as nb
import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from ._columns import WCOL, vcol
from .estimator import _StreamingBase
from .inference import _stack
from .kernels_base import _nb_residualize_rows
from .resources import peak_rss_gb, reset_peak_rss, thread_scope
from .results import sample_frame
from .typing import Source, Variables
from .utils import _norm_vars
from .workspace import _ACTIVE_RUNS


@dataclass
class CrossProducts:
    """The result of `cross_products`.

    For split s, ``gram[s]`` is sum_i w_i v_i v_i' over its rows, ``sums[s]``
    is sum_i w_i v_i, ``weight_sums[s]`` is sum_i w_i and ``counts[s]`` the
    number of rows, where v_i holds the residualized variables of row i (in
    ``names`` order) and w_i its weight (1 without weights). Without fixed
    effects nothing is residualized, so ``sums`` and ``weight_sums`` give the
    means needed for an intercept. With them, the full sample's ``sums`` are
    zero up to rounding.

    Attributes
    ----------
    names : list of str
        The variables, in the order given.
    gram : numpy.ndarray, shape (S, m, m)
    sums : numpy.ndarray, shape (S, m)
    weight_sums : numpy.ndarray, shape (S,)
    counts : numpy.ndarray of int64, shape (S,)
    fe_names : list of str
        The fixed effects partialled out.
    n_obs : int
        Rows in the sample.
    k_fe : int
        The fixed effects' parameters net of the redundant ones (as a fit with
        ``fe_dof="exact"`` counts them).
    n_levels : dict
        Levels of each fixed effect in the sample.
    diagnostics : dict
        The solve, the sample restrictions and the time and memory used.
    row_id : str
        The name of the column of `rows` holding each row's position in the
        source.
    weight_column : str or None
        The name of the column of `rows` holding the weights, or None.
    """

    names: list[str]
    gram: np.ndarray
    sums: np.ndarray
    weight_sums: np.ndarray
    counts: np.ndarray
    fe_names: list[str]
    n_obs: int
    k_fe: int
    n_levels: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)
    row_id: str = "row_id"
    weight_column: str | None = None
    _sample: Any = field(default=None, repr=False)
    _rows: str | None = field(default=None, repr=False)
    _run: Any = field(default=None, repr=False)

    @property
    def n_splits(self) -> int:
        return self.gram.shape[0]

    def total(self) -> np.ndarray:
        """The full sample's cross-products: the splits' sum."""
        return self.gram.sum(axis=0)

    def excluding(self, s: int) -> np.ndarray:
        """The cross-products of every split but `s`, summed (never formed
        as the total less split `s`, which would cancel digits)."""
        keep = [g for i, g in enumerate(self.gram) if i != s]
        return np.sum(keep, axis=0) if keep else np.zeros_like(self.gram[0])

    def sample(self, check: bool = True) -> pl.LazyFrame:
        """The source with ``row_id``, ``in_sample`` and ``dropped_because``
        added, as `HDFEResult.sample` gives them."""
        return sample_frame(self._sample, check)

    def rows(self) -> pl.LazyFrame:
        """The residualized rows, lazily (``save_rows=True``): the row's
        position in the source (`row_id`), the `carry` columns, the split
        column, the weights (`weight_column`) and one column per variable.
        They are in the estimator's order (grouped by the streamed fixed
        effect), not the source's. The files last as long as this object, or
        until `cleanup`."""
        if self._rows is None:
            raise ValueError("the rows were not saved; call cross_products with "
                             "save_rows=True")
        if not Path(self._rows).exists():
            raise FileNotFoundError("the saved rows are gone: cleanup() was called")
        return pl.scan_parquet(self._rows)

    def cleanup(self) -> None:
        """Delete the saved rows (if any) now."""
        if self._run is not None:
            self._run.cleanup()


class _Estimator(_StreamingBase):
    """The estimator `cross_products` drives. Without fixed effects it keeps
    pass 0's rows as the query that would have written them (the source,
    filtered and with the design evaluated) instead of writing them, and the
    rows pass reads that query: one pass over the source in place of a
    write and a read of a copy. With fixed effects it is the estimator
    unchanged."""

    _direct = None

    def _sink(self, lf, target, what, before_retry=None, **kw):
        if self.no_fe and what == "pass 0 write rows":
            self._direct = lf
            if getattr(self, "cmaps", None):
                # cluster codes are joined onto these rows; the source's
                # columns alone (_raw_rows) do not have them
                self._raw_rows = None
            return None
        return super()._sink(lf, target, what, before_retry=before_retry, **kw)

    def _row_chunks(self, cols):
        if self._direct is None:
            yield from super()._row_chunks(cols)
            return
        pieces = 8 * nb.get_num_threads()
        raw = getattr(self, "_raw_rows", None)
        if raw is not None:
            # the source's columns a batch at a time, the design evaluated on
            # each (see _pass0_code): bounded by the batch, where a query
            # holding the evaluated design is not
            query, evaluate = raw
            frames = (evaluate(batch.lazy()).select(cols).collect()
                      for batch in query.collect_batches(chunk_size=self.batch_rows,
                                                         engine="streaming"))
        else:
            frames = self._direct.select(cols).collect_batches(chunk_size=self.batch_rows,
                                                               engine="streaming")
        for frame in frames:
            n = frame.height
            if n == 0:
                continue
            ch = {c: frame.get_column(c).to_numpy() for c in cols}
            del frame
            starts = np.unique(np.linspace(0, n, min(n, pieces) + 1).astype(np.int64))
            yield ch, starts, np.zeros((n, 0), np.int64)


@nb.njit(cache=True)
def _split_bounds(s, n_splits):
    counts = np.zeros(n_splits + 1, np.int64)
    for v in s:
        counts[v + 1] += 1
    for j in range(n_splits):
        counts[j + 1] += counts[j]
    return counts


def _accumulate(V, w, s, n_splits, gram, sums, wsum, count, weighted):
    """Add one chunk of residualized rows to each split's sums."""
    if n_splits == 1:
        parts = [(0, None)]
    else:
        order = np.argsort(s, kind="stable")
        bounds = _split_bounds(s, n_splits)
        parts = [(j, order[bounds[j]:bounds[j + 1]]) for j in range(n_splits)
                 if bounds[j + 1] > bounds[j]]
    for j, idx in parts:
        B = V if idx is None else V[idx]
        wj = w if idx is None else w[idx]
        count[j] += B.shape[0]
        if weighted:
            wsum[j] += wj.sum()
            sums[j] += wj @ B
            B = B * np.sqrt(wj)[:, None]
        else:
            wsum[j] += B.shape[0]
            sums[j] += B.sum(axis=0)
        gram[j] += B.T @ B


def _split_values(ch, split_by, n_splits):
    s = ch[split_by]
    if not np.issubdtype(s.dtype, np.integer):
        if np.issubdtype(s.dtype, np.floating) and np.all(np.isfinite(s)) \
                and np.all(s == np.round(s)):
            s = s.astype(np.int64)
        else:
            raise ValueError(f"split_by={split_by!r} must be an integer column with no "
                             "missing values")
    s = s.astype(np.int64, copy=False)
    if len(s) and (s.min() < 0 or s.max() >= n_splits):
        raise ValueError(f"split_by={split_by!r} has values outside 0..{n_splits - 1}")
    return s


def cross_products(source: Source, variables: Variables, fe: Sequence[str] = (), *,
                   weights: str | pl.Expr | None = None, weights_type: str = "aweights",
                   split_by: str | None = None, n_splits: int | None = None,
                   save_rows: bool = False, carry: Sequence[str] = (),
                   **options: Any) -> CrossProducts:
    """Cross-products of `variables` with the fixed effects partialled out,
    one set per value of `split_by`.

    Parameters
    ----------
    source : str, Path, polars.LazyFrame or polars.DataFrame
        A Parquet path or glob, a LazyFrame, or a DataFrame.
    variables : str, tuple or list
        Column names and/or ``(name, polars.Expr)`` pairs, as for the
        estimator's `y` and `x`. A row with a missing or non-finite value in
        any of them is dropped, as a fit drops it.
    fe : sequence of str
        Fixed-effect dimensions, as for the estimator. Empty: nothing is
        partialled out, and the rows are read in batches without grouping.
    weights, weights_type
        As for the estimator.
    split_by : str, optional
        An integer column of the source with values in ``0..n_splits - 1``,
        such as a fold number. Rows with a null in it are an error.
        Default: one split, the whole sample.
    n_splits : int, optional
        The number of splits; required with `split_by`.
    save_rows : bool, default False
        Also write the residualized rows to Parquet, in the run directory
        under ``workdir`` (see `CrossProducts.rows`): about 8 bytes per row
        per variable on disk.
    carry : sequence of str
        Source columns to write with the saved rows, unchanged (a cluster
        id, say). Needs `save_rows`.
    **options
        The estimator's other options (``workdir``, ``n_threads``,
        ``max_memory_gb``, ``fixef_rm``, ``connected``, ``verbose``, ...).
        ``assembly`` and ``keep`` are set here.

    Returns
    -------
    CrossProducts
    """
    if split_by is None:
        if n_splits not in (None, 1):
            raise ValueError("n_splits needs split_by")
        n_splits = 1
    elif not isinstance(n_splits, int) or isinstance(n_splits, bool) or n_splits < 1:
        raise ValueError("split_by needs n_splits, a positive integer")
    carry = [carry] if isinstance(carry, str) else list(carry)
    if carry and not save_rows:
        raise ValueError("carry needs save_rows=True")
    for bad in ("assembly", "keep", "models", "save_resid", "x", "y"):
        if bad in options:
            raise TypeError(f"cross_products sets {bad!r} itself")
    if isinstance(variables, (str, tuple)):
        variables = [variables]
    variables = list(variables)
    if not variables:
        raise ValueError("variables must name at least one variable")
    est = _Estimator(variables[:1], variables[1:], list(fe), weights=weights,
                         weights_type=weights_type, assembly="rows", save_resid=False,
                         keep=list(dict.fromkeys(
                             ([split_by] if split_by is not None else []) + carry)),
                         **options)
    if est.slopes:
        raise ValueError("cross_products does not take fixed effects with varying slopes")
    requested = set(_norm_vars(variables))
    names = [nm for nm in est.var_names if nm in requested]
    clash = [c for c in [est.row_id, *carry, *([split_by] if split_by else [])]
             if c in requested]
    if clash:
        raise ValueError(f"columns {clash} are both variables and carried columns")
    idx = np.array([est.vidx[nm] for nm in names], dtype=np.int64)
    t0 = time.time()
    _, keys, extra = est._requests("iid", (), None)
    est._init_run()
    reset_peak_rss()
    try:
        with thread_scope(est.n_threads):
            est._run_passes(source, keys, extra)
            est._log(f"step 2: solving for {est.m} variables (blocks of "
                     f"{est.rhs_block}), solver={est.solver}")
            gamma, info = est._solve()
            est._log("step 3: cross-products by split")
            rows_path = None
            if save_rows:
                (est.workdir / "models").mkdir(exist_ok=True)
                rows_path = str(est.workdir / "models" / "crossprod_rows.parquet")
            out = _split_pass(est, gamma, idx, names, split_by, n_splits, carry,
                              rows_path)
            k_fe = est._k_fe("exact") if est.fe else 0
    except BaseException:
        traceback.clear_frames(sys.exc_info()[2])
        gc.collect()
        est._release_intermediates()
        if not est.keep_intermediates:
            est._run.cleanup()
        raise
    est._release_intermediates()
    if not est.keep_intermediates and not save_rows:
        est._run.cleanup()
    _ACTIVE_RUNS.discard(str(est.workdir))
    gram, sums, wsum, count = out
    rss = peak_rss_gb()
    diag = {"solver_info": dict(info), "stream": est.stream_choice,
            "singletons": dict(est.singletons), "connected": dict(est.connected_info),
            "fe_params": dict(est.fe_params), "n_cells": int(est.n_cells),
            "design_evaluated": getattr(est, "design_evaluated", None),
            "seconds_total": round(time.time() - t0, 2),
            "peak_rss_gb": None if rss is None else round(rss, 3)}
    return CrossProducts(names=names, gram=gram, sums=sums, weight_sums=wsum,
                         counts=count, fe_names=list(est.fe), n_obs=int(est.n_obs),
                         k_fe=int(k_fe), n_levels=dict(est.n_levels), diagnostics=diag,
                         row_id=est.row_id,
                         weight_column=WCOL if est.weights is not None else None,
                         _sample=est._sample_spec, _rows=rows_path,
                         _run=est._run if save_rows else None)


def _split_pass(est, gamma, idx, names, split_by, n_splits, carry, rows_path):
    """The rows pass of `_assemble`, with the sums kept apart by split (and
    the residualized rows written to `rows_path`, if given)."""
    m, nt = est.m, nb.get_num_threads()
    weighted = est.weights is not None
    wcol = [WCOL] if weighted else []
    scol = [split_by] if split_by is not None else []
    extra = list(dict.fromkeys([est.row_id, *carry, *scol])) if rows_path else scol
    cols = [*est.ccols, *wcol, *extra, *[vcol(j) for j in range(m)]]
    writer = None
    q = len(idx)
    gram = np.zeros((n_splits, q, q))
    sums = np.zeros((n_splits, q))
    wsum = np.zeros(n_splits)
    count = np.zeros(n_splits, np.int64)
    zeros = None
    for ch, starts, codes in est._row_chunks(cols):
        V = _stack(ch, [vcol(j) for j in range(m)])
        n = len(V)
        w = (np.ascontiguousarray(ch[WCOL], dtype=np.float64) if weighted else np.ones(n))
        _nb_residualize_rows(starts, codes, est.offs, w, V, gamma, not est.no_fe, nt)
        if q != m:
            V = np.ascontiguousarray(V[:, idx])
        if split_by is None:
            if zeros is None or len(zeros) < n:
                zeros = np.zeros(n, np.int64)
            s = zeros[:n]
        else:
            s = _split_values(ch, split_by, n_splits)
        _accumulate(V, w, s, n_splits, gram, sums, wsum, count, weighted)
        if rows_path:
            arrays = [pa.array(ch[c]) for c in extra]
            arrays += [pa.array(w)] if weighted else []
            arrays += [pa.array(V[:, j]) for j in range(V.shape[1])]
            table = pa.Table.from_arrays(arrays, names=[*extra, *wcol, *names])
            if writer is None:
                writer = pq.ParquetWriter(rows_path, table.schema)
            writer.write_table(table)
    if writer is not None:
        writer.close()
    elif rows_path:          # an empty sample: an empty file with the columns
        fields = ([pa.field(c, pa.int64()) for c in extra]
                  + [pa.field(c, pa.float64()) for c in [*wcol, *names]])
        pq.write_table(pa.schema(fields).empty_table(), rows_path)
    for j in range(n_splits):
        gram[j] = (gram[j] + gram[j].T) / 2
    return gram, sums, wsum, count

"""Building blocks for estimators that run their own Newton iteration on a
logit, probit or Poisson model with no absorbed fixed effect or one: the
stored design (pass 0) and the streamed pass that gives each model's
log-likelihood, gradient and information.

Stable names
    The names below are the supported API for packages built on this one.
    They change only in a minor release (0.x -> 0.y), with the change in the
    changelog; tests/test_core_newton.py pins them::

        glm_design      formula + data -> GLMDesign (pass 0)
        GLMDesign       the stored rows; `newton_pass`, `rows`, `cleanup`
        NewtonMoments   what one pass returns

The model has the linear predictor

    eta_i = off_i + x_i'b + a_g(i)

where a_g is the intercept of the row's group: one group (the intercept)
without a fixed effect, or one per level of the absorbed fixed effect. A
pass returns the log-likelihood, its gradient and the information with
respect to b and a. The information is the expected one: for the logit and
Poisson that is minus the Hessian, and for the probit it is Fisher
scoring's. It has an arrow shape (a diagonal block for a, which each row
touches one entry of), and `NewtonMoments.profiled` eliminates a from it
exactly (the Schur complement), so that a Newton step, penalized or not, is
a k x k problem in b whatever the number of groups; `NewtonMoments.effects_step`
then gives a's part of the step.

One pass can evaluate several models (coefficient vectors) at once, each on
its own set of the rows: with `split_by=` the rows carry an integer split
(a cross-validation fold, say), and each model in a pass names the splits
its sums cover. The rows are read once; the arithmetic is per model, and
each row's nonzeros are found once for all of them. The log-likelihood is
returned for every model on every split, so a model's held-out fit comes
from the same pass.

Pass 0 writes the rows the model uses to hdfe-stream's block store
(`core.store`): the design (its 0/1 columns as one byte each), the outcome,
the weights, the offset and integer codes for the fixed effect and the
splits, grouped by the fixed effect when that fits in memory. The design is
evaluated batch by batch from a narrow copy of the sample (the source
columns the formula uses), so a categorical expanded into many indicators
never passes through Polars as a wide query.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numba as nb
import numpy as np
import polars as pl
import pyarrow.parquet as pq
from scipy import sparse
from scipy.linalg import solve_triangular

from ._columns import PREFIX
from .families import get_family
from .feterms import parse_fe
from .formula import categorical_vars, compile_design, numeric_expr, parse_formula
from .kernels_newton import (LOGIT, POISSON, PROBIT, newton_pass_rows, newton_row_terms,
                             part_starts)
from .report import warn
from .resources import MemoryBudget
from .store import (BlockStore, PolarsEngine, block_rows, factorize, iter_batches,
                    iter_groups, not_missing, write_sample)
from .utils import as_lazy
from .workspace import Run, _base_dir

__all__ = ["glm_design", "GLMDesign", "NewtonMoments"]

_P = f"{PREFIX}nt_"
ROW = f"{_P}row"                # the row's position in the source
Y = f"{_P}y"
W = f"{_P}w"
OFF = f"{_P}off"
CODE = f"{_P}code"              # factorization tables' code column
SPLIT = f"{_P}split"            # the split clustering's name

FAMILIES = {"logit": LOGIT, "probit": PROBIT, "poisson": POISSON}

# The memory the per-part copies of a pass's sums may take (each thread sums
# its part of a chunk into its own copy; fewer parts when the copies are big).
PART_BYTES = 512 * 2**20

# Use the nonzeros-only products when a row has fewer than this share of
# its covariates nonzero, on average.
NNZ_SHARE = 0.5

# Which way a pass forms the information over c columns. The fused row
# kernel (`newton_pass_rows`) costs about (nonzeros per row)^2 / 2 per row,
# BLAS products c^2 / 2 at a much higher rate; the kernel wins up to about
# KERNEL_DENSE_COLUMNS / (the rows' nonzero share) columns -- 96 for dense
# rows, about 300 at a third nonzero, 1,000 at a tenth -- and never beyond
# KERNEL_SPARSE_COLUMNS, where each thread's c x c copy of the sums no
# longer stays in cache.
KERNEL_DENSE_COLUMNS = 96
KERNEL_SPARSE_COLUMNS = 1024

# The design's 0/1 columns (indicators of factor levels, binary covariates)
# are stored as one byte each and put back together as each block is read.
COMPACT_BINARY = True


def _xcol(j: int) -> str:
    return f"{_P}x{j}"


# --------------------------------------------------------------------------
# results of a pass
# --------------------------------------------------------------------------

@dataclass
class NewtonMoments:
    """One pass's sums for M models (the first axis of every array), with
    k covariates (`columns` of them in the information) and G groups.

    Attributes
    ----------
    loglik : (M, S) array
        Each model's log-likelihood on each split's rows (all splits, those
        its sums cover and those they do not).
    gradient : (M, k) array
        d loglik / d b over the model's splits, every covariate.
    effects_gradient : (M, G) array
        d loglik / d a.
    information : (M, c, c) array or None
        The information for b, over the `columns` (c of them).
    effects_information : (M, G) array or None
        The information for each a_g (the diagonal block).
    cross : (M, G, c) array or None
        The information between a_g and b.
    skipped : (M, S) array
        Rows left out of each model because their group's effect is not
        finite (see `GLMDesign.newton_pass`).
    columns : array of int
    """

    loglik: np.ndarray
    gradient: np.ndarray
    effects_gradient: np.ndarray
    information: np.ndarray | None
    effects_information: np.ndarray | None
    cross: np.ndarray | None
    skipped: np.ndarray
    columns: np.ndarray

    def _dinv(self, m):
        D = self.effects_information[m]
        with np.errstate(divide="ignore"):
            return np.where(D > 0, 1.0 / D, 0.0)

    def profiled(self, m: int = 0):
        """(information, gradient) of model m for b (its `columns`) with the
        effects profiled out: I_bb - I_ba I_aa^-1 I_ab and g_b - I_ba
        I_aa^-1 g_a. A group with no information (no rows in the model) is
        left out."""
        if self.information is None:
            raise ValueError("the pass was made without the information")
        Di = self._dinv(m)
        A = self.cross[m]
        AD = A * Di[:, None]
        C = self.information[m] - A.T @ AD
        g = self.gradient[m, self.columns] - AD.T @ self.effects_gradient[m]
        return (C + C.T) / 2, g

    def effects_step(self, dbeta, m: int = 0):
        """The effects' part of the Newton step, given b's part (over
        `columns`): I_aa^-1 (g_a - I_ab db); zero for a group with no
        information."""
        return self._dinv(m) * (self.effects_gradient[m] - self.cross[m] @ np.asarray(dbeta))


def newton_direction(C, g, rcond: float = 1e-10):
    """The Newton step C^-1 g and its decrement g' C^-1 g, solved after
    scaling C to unit diagonal (Jacobi equilibration: indifferent to the
    covariates' scale). A C that is singular, as with collinear covariates,
    gets the minimum-norm step through its eigenvalues above `rcond` times
    the largest."""
    C = np.asarray(C, dtype=np.float64)
    g = np.asarray(g, dtype=np.float64)
    if not len(g):
        return np.zeros(0), 0.0
    d = np.sqrt(np.clip(np.diag(C), 0.0, None))
    d = np.where(d > 0, d, 1.0)
    Cs = C / np.outer(d, d)
    gs = g / d
    try:
        L = np.linalg.cholesky(Cs)
        if np.min(np.diag(L)) ** 2 < rcond * np.max(np.diag(L)) ** 2:
            raise np.linalg.LinAlgError
        # triangular solves: numpy's general `solve` (LU) on a small system
        # is slow on a multithreaded BLAS, tens of ms against tens of us
        x = solve_triangular(L.T, solve_triangular(L, gs, lower=True), lower=False)
    except np.linalg.LinAlgError:
        ev, V = np.linalg.eigh((Cs + Cs.T) / 2)
        keep = ev > rcond * max(float(ev.max()), 0.0)
        x = V[:, keep] @ ((V[:, keep].T @ gs) / ev[keep])
    return x / d, float(gs @ x)


# --------------------------------------------------------------------------
# the stored design
# --------------------------------------------------------------------------

@dataclass
class GLMDesign:
    """The rows of a logit, probit or Poisson model, stored by pass 0
    (`glm_design`).

    Attributes
    ----------
    formula : str
    family : str
        ``"logit"``, ``"probit"`` or ``"poisson"``.
    depvar : str
    names : list of str
        The covariates (the columns of b; no intercept, which is the one
        group's effect without a fixed effect).
    fe : str or None
        The absorbed fixed effect.
    n_groups : int
        G: the groups of `fe` kept (1 without a fixed effect).
    fe_levels : polars.DataFrame or None
        The kept groups' values of the fixed effect's columns, in the order
        of the effects (row g is group g).
    n_splits : int
        S (1 without `split_by`).
    n_obs : int
        Rows used (after dropping the groups whose effect would be infinite).
    weight_sum : float
        Their summed weight (analytic weights are scaled to a mean of 1 over
        the rows kept, as the package's estimators do).
    group_weights : (S, G) array
        The summed weight of each split's rows in each group.
    group_outcomes : (S, G) array
        Their summed weighted outcome.
    outcome_counts : (S, G, 2) array or None
        For the logit and probit, the weighted counts of y = 0 and y = 1 in
        each split and group (group_weights - group_outcomes, and
        group_outcomes); None for Poisson.
    x_sums, x_squares : (S, k) arrays
        Sum w x and sum w x^2 of each covariate in each split (for
        standardizing).
    x_nonzero : (k,) array
        Each covariate's share of nonzero rows.
    separation : dict
        The groups dropped because their effect would be infinite (for the
        logit and probit, an outcome that never varies; for Poisson, one that
        is always zero), and their rows.
    diagnostics : dict
    """

    formula: str
    family: str
    depvar: str
    names: list
    fe: str | None
    n_groups: int
    fe_levels: Any
    n_splits: int
    n_obs: int
    weight_sum: float
    group_weights: np.ndarray
    group_outcomes: np.ndarray
    outcome_counts: np.ndarray | None
    x_sums: np.ndarray
    x_squares: np.ndarray
    x_nonzero: np.ndarray
    separation: dict
    diagnostics: dict = field(default_factory=dict)
    fe_sorted: bool = False
    dropped_levels: dict = field(default_factory=dict)
    usable: Any = None
    fe_table: Any = None
    x_binary: Any = None
    exprs: dict = field(default_factory=dict)       # covariate name -> Polars expression
    y_expr: Any = None
    offset_expr: Any = None
    _store: Any = None
    _run: Any = None
    _keep: Any = None
    _split_of: Any = None
    _remap: Any = None
    _chunk_bytes: float = 64e6
    _has_offset: bool = False

    @property
    def passes(self) -> int:
        return int(self.diagnostics.get("passes", 0))

    @property
    def use_nonzeros(self) -> bool:
        """Whether a pass forms the products from each row's nonzeros only:
        when a row has fewer than `NNZ_SHARE` of its covariates nonzero, on
        average."""
        k = len(self.names)
        return bool(k) and float(np.sum(self.x_nonzero)) < NNZ_SHARE * k

    def _blocks(self):
        k = len(self.names)
        target = max(256, int(self._chunk_bytes // (8 * max(k + 8, 1))))
        for blk in self._store.iterate(["X", "y", "w", "off", "fe", "cl"]):
            n_blk = len(blk["y"])
            rows = -(-n_blk // max(1, round(n_blk / target)))     # chunks of equal size
            for lo in range(0, n_blk, rows):
                hi = min(n_blk, lo + rows)
                ch = {key: (v[lo:hi] if v is not None else None) for key, v in blk.items()}
                n = hi - lo
                f = ch.get("fe")
                f = np.zeros(n, np.int64) if f is None else f
                keep = self._keep[f]
                if not keep.all():
                    ch = {key: (v[keep] if v is not None else None) for key, v in ch.items()}
                    f = f[keep]
                    n = len(f)
                if not n:
                    continue
                if self.fe is not None:
                    f = self._remap[f]
                s = (self._split_of[ch["cl"][:, 0]] if self._split_of is not None
                     else np.zeros(n, np.int64))
                yield (np.ascontiguousarray(ch["X"]), ch["y"], ch.get("w"), ch.get("off"),
                       np.ascontiguousarray(f, dtype=np.int64),
                       np.ascontiguousarray(s, dtype=np.int64))

    def _rows_full(self):
        """The stored rows of the kept groups, a block at a time, as a dict:
        X, y, w, off (None without), f (the effect's index), s (the split),
        cl (the clusterings' codes, or None), row (the position in the
        source) and fe (the group's code in `fe_table`, or None)."""
        for blk in self._store.iterate(["X", "y", "w", "off", "fe", "cl", "row"]):
            n = len(blk["y"])
            fe = blk.get("fe")
            f = np.zeros(n, np.int64) if fe is None else fe
            keep = self._keep[f]
            if not keep.all():
                blk = {key: (v[keep] if v is not None else None) for key, v in blk.items()}
                f = f[keep]
                fe = blk.get("fe")
                n = len(f)
            if not n:
                continue
            idx = self._remap[f] if self.fe is not None else f
            cl = blk.get("cl")
            s = (self._split_of[cl[:, 0]] if self._split_of is not None
                 else np.zeros(n, np.int64))
            yield {"X": np.ascontiguousarray(blk["X"]), "y": blk["y"], "w": blk.get("w"),
                   "off": blk.get("off"), "f": np.ascontiguousarray(idx, dtype=np.int64),
                   "s": s, "cl": cl, "row": blk["row"], "fe": fe}

    def rows(self):
        """The stored rows, in chunks: ``(X, y, w, group, split)`` arrays,
        for sums a pass does not give (X the covariates, y the outcome as
        floats, w the weights or None, group each row's effect index 0..G-1,
        split its split 0..S-1). Rows of dropped groups are left out. The
        offset, if any, is in `rows_with_offset`. Counts as a pass."""
        self.diagnostics["passes"] = self.passes + 1
        for X, y, w, off, f, s in self._blocks():
            yield X, y, w, f, s

    def rows_with_offset(self):
        """As `rows`, with the offset (None without one) after the weights:
        ``(X, y, w, offset, group, split)``. Counts as a pass."""
        self.diagnostics["passes"] = self.passes + 1
        yield from self._blocks()

    def newton_pass(self, beta, effects, *, models=None, columns=None,
                    information: bool = True, observed: bool = False) -> NewtonMoments:
        """One read of the rows: each model's log-likelihood on every split,
        and its gradient and information over its splits.

        Parameters
        ----------
        beta : (M, k) or (k,) array
            Each model's coefficients.
        effects : (M, G) or (G,) array
            Each model's group effects (without a fixed effect, G = 1: the
            intercept). A row whose linear predictor is not finite (an effect
            of +-inf or nan) is left out of that model -- its log-likelihood,
            gradient and information -- and counted in `skipped`: the way to
            hold a group out of a model (one whose outcome does not vary
            within the model's rows, say, whose effect would be infinite).
        models : (M, S) boolean array, optional
            The splits each model's gradient and information cover; default
            all.
        columns : sequence of int, optional
            The covariates the information is formed for (an active set);
            default all. The gradient is always for every covariate.
        information : bool
            Whether to form the information (False: the log-likelihood and
            gradients only, which cost much less).
        observed : bool
            The observed information (minus the Hessian) instead of the
            expected one. The same for the logit and Poisson; for the probit,
            Newton's method converges quadratically with it, Fisher scoring
            (the expected) only linearly.
        """
        sums = self._sums(beta, effects, models=models, columns=columns,
                          information=information, observed=observed)
        for X, y, w, off, f, sp in self._blocks():
            sums.add(X, y, w, off, f, sp)
        self.diagnostics["passes"] = self.passes + 1
        return sums.moments()

    def _sums(self, beta, effects, *, models=None, columns=None, information=True,
              observed=False, meat=0) -> _Sums:
        """The running sums of a pass (`newton_pass`'s arguments; `meat` as
        in `newton_pass_rows`), for a caller that reads the rows itself."""
        return _Sums(self, beta, effects, models, columns, information, observed, meat)

    def cleanup(self) -> None:
        """Delete the stored rows."""
        if self._run is not None:
            self._run.cleanup()
            self._run = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.cleanup()


# --------------------------------------------------------------------------
# pass 0
# --------------------------------------------------------------------------

@dataclass
class _Spec:
    fml: str
    depvar: str
    y: pl.Expr
    x: list                 # [(coefficient name, Float64 expression)]
    fe: tuple


def _plan(fml: str, lf: pl.LazyFrame, engine: PolarsEngine) -> _Spec:
    """The one model of `fml`: pyfixest's coding of the covariates (from
    hdfe-stream's formula compiler), without the constant, which is the
    effects'."""
    schema = lf.collect_schema()
    specs = parse_formula(fml)
    if len(specs) != 1:
        raise ValueError(f"{fml}: one model (no multiple-estimation syntax)")
    s = specs[0]
    if s.iv is not None:
        raise ValueError(f"{fml}: IV formulas are not available here")
    cats = categorical_vars(s.rhs, schema)
    levels = {v: engine.collect(lf.select(pl.col(v).drop_nulls().unique().sort()),
                                f"levels of {v}")[v].to_list() for v in sorted(cats)}
    fe = tuple(parse_fe(s.fe))
    for d in fe:
        if "[" in d:
            raise ValueError(f"{fml}: varying slopes are not available here")
    _, cols = compile_design(s.dep, s.rhs, schema, levels, intercept=False)
    cols = [(nm, ex) for nm, ex in cols if nm != "Intercept"]
    y = pl.col(s.dep) if s.dep in schema else numeric_expr(s.dep)
    return _Spec(fml=s.key, depvar=s.dep, y=y.cast(pl.Float64), x=cols, fe=fe)


_WIDTH = {pl.Int8: 1, pl.UInt8: 1, pl.Int16: 2, pl.UInt16: 2, pl.Int32: 4, pl.UInt32: 4,
          pl.Float32: 4, pl.Boolean: 1}


def _partition_gb(path, rows):
    """The memory `iter_groups` needs to sort one bucket of ungrouped rows: a
    bucket holds up to max(8 * rows, 1M) rows, and collecting, sorting and
    slicing it holds about three copies."""
    width = sum(_WIDTH.get(dt, 8) for dt in pl.scan_parquet(path).collect_schema().values())
    return 3 * max(8 * rows, 1_000_000) * width / 1e9


def _grouped(path, cols, n_groups, rows):
    """Whether each group's rows of the file are consecutive already (then
    `iter_groups` partitions nothing): as many runs of the key columns as
    groups, read `rows` at a time."""
    runs, last = 0, None
    pf = pq.ParquetFile(path, pre_buffer=False)
    try:
        for batch in pf.iter_batches(batch_size=rows, columns=list(cols)):
            key = [batch.column(c).to_numpy(zero_copy_only=False) for c in cols]
            start = np.zeros(len(key[0]), bool)
            for kk in key:
                start[1:] |= kk[1:] != kk[:-1]
            start[0] = last is None or any(kk[0] != v for kk, v in zip(key, last))
            runs += int(start.sum())
            last = [kk[-1] for kk in key]
    finally:
        pf.close()
    return runs == n_groups



class _Sums:
    """A pass's sums, block by block (see `GLMDesign.newton_pass`): by the
    fused row kernel, or by BLAS products for dense or very wide rows (see
    KERNEL_DENSE_COLUMNS)."""

    def __init__(self, ld, beta, effects, models, columns, information, observed, meat):
        beta = np.ascontiguousarray(np.atleast_2d(np.asarray(beta, dtype=np.float64)))
        effects = np.ascontiguousarray(np.atleast_2d(np.asarray(effects, dtype=np.float64)))
        M, k = beta.shape
        G, S = ld.n_groups, ld.n_splits
        if k != len(ld.names):
            raise ValueError(f"beta has {k} columns; the design has {len(ld.names)}")
        if effects.shape != (M, G):
            raise ValueError(f"effects must be ({M}, {G})")
        inc = (np.ones((M, S), np.bool_) if models is None
               else np.ascontiguousarray(np.asarray(models, dtype=np.bool_).reshape(M, S)))
        cols = (np.arange(k, dtype=np.int64) if columns is None
                else np.asarray(columns, dtype=np.int64))
        if len(np.unique(cols)) != len(cols) or (len(cols) and
                                                 (cols.min() < 0 or cols.max() >= k)):
            raise ValueError("columns must be distinct covariate positions")
        information = bool(information or meat)
        c = len(cols) if information else 0
        colpos = np.full(k, -1, np.int64)
        colpos[cols] = np.arange(len(cols))
        self.nnz = ld.use_nonzeros
        self.blas = c > _kernel_columns(ld)
        self.grouped = bool(ld.fe_sorted and G > 1)
        # each part's copies: ll and skipped, the gradient, the information,
        # and (ungrouped) the groups' sums
        per_part = 8 * (2 * M * S + M * k + M * c * c
                        + (0 if self.grouped else M * G * (2 + c)))
        self.max_parts = max(1, int(PART_BYTES // max(per_part, 1)))
        self.beta, self.effects, self.inc, self.cols, self.colpos = beta, effects, inc, cols, colpos
        self.all_cols = len(cols) == k
        self.M, self.k, self.G, self.S, self.c = M, k, G, S, c
        self.information, self.observed, self.meat = information, bool(observed), int(meat)
        self.fam = FAMILIES[ld.family]
        self.ll = np.zeros((M, S))
        self.sk = np.zeros((M, S))
        self.gb = np.zeros((M, k))
        self.H = np.zeros((M, c, c))
        self.ga = np.zeros((M, G))
        self.D = np.zeros((M, G if information else 0))
        self.C = np.zeros((M, G if information else 0, c))

    def add(self, X, y, w, off, f, s):
        """Add a chunk of rows: X (n, k), y, w and off (n,) (w and off may be
        None), f the effects' indices and s the splits (n,)."""
        y = np.ascontiguousarray(y, dtype=np.float64)
        w = _EMPTY if w is None else np.ascontiguousarray(w, np.float64)
        off = _EMPTY if off is None else np.ascontiguousarray(off, np.float64)
        if self.blas:
            self._add_blas(X, y, w, off, f, s)
        else:
            self._add_kernel(X, y, w, off, f, s)

    def _add_kernel(self, X, y, w, off, f, s):
        M, k, G, S, c = self.M, self.k, self.G, self.S, self.c
        n = len(y)
        P = max(1, min(nb.get_num_threads(), self.max_parts, n // 256 or 1))
        starts = (part_starts(f, P) if self.grouped
                  else np.linspace(0, n, P + 1).astype(np.int64))
        P = len(starts) - 1
        info = self.information
        ll_t = np.zeros((P, M, S))
        sk_t = np.zeros((P, M, S))
        gb_t = np.zeros((P, M, k))
        H_t = np.zeros((P, M, c, c))
        gshape = (0, M, 0) if self.grouped else (P, M, G)
        ga_t = np.zeros(gshape)
        D_t = np.zeros(gshape if info else (0, M, 0))
        C_t = np.zeros((gshape[0], M, gshape[2], c) if info else (0, M, 0, c))
        with nb.parallel_chunksize(1):
            newton_pass_rows(self.fam, X, y, w, off, f, s, self.beta, self.effects, self.inc,
                             self.colpos, info, self.nnz, self.grouped, starts, ll_t, sk_t,
                             gb_t, H_t, ga_t, D_t, C_t, self.ga, self.D, self.C,
                             self.observed, self.meat)
        self.ll += ll_t.sum(axis=0)
        self.sk += sk_t.sum(axis=0)
        self.gb += gb_t.sum(axis=0)
        self.H += H_t.sum(axis=0)
        if not self.grouped:
            self.ga += ga_t.sum(axis=0)
            if info:
                self.D += D_t.sum(axis=0)
                self.C += C_t.sum(axis=0)

    def _add_blas(self, X, y, w, off, f, s):
        M, G, S = self.M, self.G, self.S
        n = len(y)
        eta = X @ self.beta.T
        eta += self.effects[:, f].T
        if len(off):
            eta += off[:, None]
        P = max(1, min(nb.get_num_threads(), n // 256 or 1))
        starts = np.linspace(0, n, P + 1).astype(np.int64)
        ll_t, sk_t = np.zeros((P, M, S)), np.zeros((P, M, S))
        R, Hh = np.empty((n, M)), np.empty((n, M))
        newton_row_terms(self.fam, y, w, eta, s, self.inc, starts, ll_t, sk_t, R, Hh,
                         self.observed, self.meat)
        self.ll += ll_t.sum(axis=0)
        self.sk += sk_t.sum(axis=0)
        self.gb += (X.T @ R).T
        for m in range(M):
            self.ga[m] += np.bincount(f, weights=R[:, m], minlength=G)
        if not self.information:
            return
        Xc = X if self.all_cols else X[:, self.cols]
        for m in range(M):
            hm = Hh[:, m]
            self.D[m] += np.bincount(f, weights=hm, minlength=G)
            Xs = Xc * np.sqrt(np.maximum(hm, 0.0))[:, None]
            self.H[m] += Xs.T @ Xs
            if G == 1:
                self.C[m, 0] += hm @ Xc
            else:
                F = sparse.csr_matrix((hm, (f, np.arange(n))), shape=(G, n))
                self.C[m] += F @ Xc

    def moments(self) -> NewtonMoments:
        info = self.information
        Hs = None
        if info:
            H = self.H
            Hs = np.triu(H) + np.transpose(np.triu(H, 1), (0, 2, 1))
        return NewtonMoments(loglik=self.ll, gradient=self.gb, effects_gradient=self.ga,
                             information=Hs, effects_information=self.D if info else None,
                             cross=self.C if info else None,
                             skipped=self.sk.astype(np.int64), columns=self.cols)


def _kernel_columns(ld) -> float:
    """The most information columns a pass forms by the row kernel (see
    KERNEL_DENSE_COLUMNS); BLAS products beyond."""
    if not ld.use_nonzeros:
        return KERNEL_DENSE_COLUMNS
    k = len(ld.names)
    share = float(np.sum(ld.x_nonzero)) / k if k else 1.0
    return min(KERNEL_SPARSE_COLUMNS, KERNEL_DENSE_COLUMNS / max(share, 1e-12))


_EMPTY = np.empty(0)


def glm_design(fml: str, data: Any, *, family: str, weights: str | None = None,
               weights_type: str = "aweights", split_by: str | None = None,
               n_splits: int | None = None, offset: str | pl.Expr | None = None,
               clusters: dict | None = None, drop_singletons: bool = False,
               workdir=None, storage: str = "disk",
               dtype: str = "float64", block_mb: float = 64.0, chunk_mb: float = 64.0,
               polars_engine: str = "cpu", max_memory_gb: float | None = None,
               memory_reserve: float | None = None, verbose: bool = False,
               logger=None) -> GLMDesign:
    """Pass 0 of a logit, probit or Poisson model: the rows stored for
    `GLMDesign.newton_pass`.

    Parameters
    ----------
    fml : str
        One model, ``"y ~ x1 + x2"`` or ``"y ~ x1 + x2 | fe"`` (at most one
        absorbed fixed effect; pyfixest's syntax for the covariates). The
        intercept is the effects' (no ``Intercept`` column).
    data : str, Path, polars.LazyFrame or polars.DataFrame
    family : {"logit", "probit", "poisson"}
        For the logit and probit the outcome is 0 or 1; for Poisson it is
        nonnegative.
    weights, weights_type
        As for `feglm_stream`: analytic weights (scaled to a mean of 1) or
        frequency weights.
    split_by : str, optional
        An integer column with values 0, ..., n_splits - 1: each row's split.
    n_splits : int, optional
        With `split_by`.
    offset : str or polars.Expr, optional
        Added to the linear predictor with coefficient 1 (a Poisson
        exposure's logarithm, say).
    clusters : dict, optional
        Name -> source columns of clusterings to store codes for (the
        variance passes of an estimator built on this one).
    drop_singletons : bool, default False
        Drop the fixed effect's groups of one row first, as hdfe-stream's
        estimators do by default (``fixef_rm="singleton"``).
    workdir : path, optional
        Where the rows are stored (the system's temporary directory by
        default, or ``HDFE_STREAM_WORKDIR``).
    storage : {"disk", "memory"}
    dtype : {"float64", "float32"}
        The stored design's precision (float32 halves the disk; the 0/1
        columns take one byte either way).
    block_mb, chunk_mb : float
        The stored blocks' size, and the chunks a pass works on.
    polars_engine, max_memory_gb, memory_reserve, verbose, logger
        As for the estimators.

    Returns
    -------
    GLMDesign
        Call `cleanup()` (or use it as a context manager) to delete the rows.

    Groups of the fixed effect whose effect would be infinite (logit and
    probit: an outcome that never varies; Poisson: one that is always zero)
    are dropped, with a warning.
    """
    if family not in FAMILIES:
        raise ValueError(f"family must be one of {sorted(FAMILIES)}, not {family!r}")
    if weights_type not in ("aweights", "fweights"):
        raise ValueError("weights_type must be 'aweights' or 'fweights'")
    if storage not in ("disk", "memory"):
        raise ValueError("storage must be 'disk' or 'memory'")
    if dtype not in ("float64", "float32"):
        raise ValueError("dtype must be 'float64' or 'float32'")
    if (split_by is None) != (n_splits is None):
        raise ValueError("give split_by and n_splits together")
    if n_splits is not None and (isinstance(n_splits, bool) or not isinstance(n_splits, int)
                                 or n_splits < 1):
        raise ValueError("n_splits must be a positive integer")
    clusters = dict(clusters or {})
    if SPLIT in clusters:
        raise ValueError(f"{SPLIT!r} is reserved")

    def log(msg):
        if verbose:
            if logger is not None:
                logger.info(msg)
            else:
                print(f"[hdfe {time.strftime('%H:%M:%S')}] {msg}", flush=True)

    t0 = time.time()
    fam = get_family(family)
    lf = as_lazy(data)
    schema = lf.collect_schema()
    clash = [c for c in schema.names() if c.startswith(_P)]
    if clash:
        raise ValueError(f"the data has columns whose names start with {_P!r}: {clash}")
    engine = PolarsEngine(polars_engine)
    spec = _plan(fml, lf, engine)
    if len(spec.fe) > 1:
        raise ValueError(f"{fml}: at most one absorbed fixed effect (got {list(spec.fe)})")
    if split_by is not None and split_by not in schema:
        raise ValueError(f"split_by: column {split_by!r} is not in the data")
    if split_by is not None:
        clusters = {SPLIT: [split_by], **clusters}
    budget = MemoryBudget(max_memory_gb, memory_reserve)
    run = Run(_base_dir(workdir), auto_cleanup=True)
    try:
        return _pass0(spec, lf, schema, fam, engine, run, budget, log, t0,
                      weights=weights, weights_type=weights_type, split_by=split_by,
                      n_splits=n_splits, offset=offset, clusters=clusters,
                      drop_singletons=drop_singletons,
                      in_memory=storage == "memory",
                      dtype=np.float32 if dtype == "float32" else np.float64,
                      block_mb=block_mb, chunk_mb=chunk_mb)
    except BaseException:
        run.cleanup()
        raise


def _pass0(spec, lf, schema, fam, engine, run, budget, log, t0, *, weights, weights_type,
           split_by, n_splits, offset, clusters, drop_singletons, in_memory, dtype, block_mb,
           chunk_mb):
    K = len(spec.x)
    fe_cols = {d: d.split("^") for d in spec.fe}
    raw = list(dict.fromkeys([c for cs in clusters.values() for c in cs]
                             + [c for cs in fe_cols.values() for c in cs]))
    for c in raw + ([weights] if weights else []):
        if c not in schema:
            raise ValueError(f"column {c!r} is not in the data")
    off_expr = None
    if offset is not None:
        off_expr = (pl.col(offset) if isinstance(offset, str) else offset).cast(pl.Float64)
    xexprs = [ex.alias(_xcol(j)) for j, (_, ex) in enumerate(spec.x)]
    wexpr = [pl.col(weights).cast(pl.Float64).alias(W)] if weights else []
    keep = ([spec.y.is_not_null(), spec.y.is_not_nan()]
            + [ex.is_finite() for _, ex in spec.x]
            + ([pl.col(weights).cast(pl.Float64).is_finite()] if weights else [])
            + ([off_expr.is_finite()] if off_expr is not None else [])
            + not_missing(schema, raw))
    usable = pl.all_horizontal(keep).fill_null(False)
    # The sample is written with the source columns the model uses, not the
    # design they expand to: that copy is narrow (a categorical is one column,
    # not one per level), so Polars' streaming sink -- which holds about a row
    # group per thread of what it writes -- stays small, and so does the copy
    # on disk. The design is evaluated from it batch by batch below; the
    # formula's expressions are row-wise, so that gives the same rows.
    roots = [c for e in [spec.y, *[ex for _, ex in spec.x],
                         *([off_expr] if off_expr is not None else [])]
             for c in e.meta.root_names()]
    cols = list(dict.fromkeys([*roots, *([weights] if weights else []), *raw]))
    rows = block_rows(block_mb, K)
    design_path = run.path / "sample.parquet"
    log("pass 0: writing the sample")
    write_sample(engine, lf, usable, cols, design_path, rows, ROW)
    evaluated = [pl.col(ROW), spec.y.alias(Y), *xexprs, *wexpr,
                 *([off_expr.alias(OFF)] if off_expr is not None else []),
                 *[pl.col(c) for c in raw]]
    design = pl.scan_parquet(design_path).select(evaluated)

    # facts about the sample, from the local copy
    stats = engine.collect(design.select(
        pl.len().alias("n"),
        *([pl.col(W).sum().alias("wsum"), pl.col(W).min().alias("wmin")] if weights else []),
        pl.col(Y).min().alias("ylo"), pl.col(Y).max().alias("yhi"),
        ((pl.col(Y) != 0) & (pl.col(Y) != 1)).sum().alias("ynb"),
        *[((pl.col(_xcol(j)) != 0) & (pl.col(_xcol(j)) != 1)).sum().alias(f"b{j}")
          for j in range(K)],
        *[(pl.col(_xcol(j)) != 0).mean().alias(f"z{j}") for j in range(K)]), "pass 0 moments")
    n_obs = int(stats["n"].item())
    n_raw = int(engine.collect(lf.select(pl.len()), "row count").item())
    if n_obs == 0:
        raise ValueError(f"{spec.fml}: no complete observations")
    fam.check(float(stats["ylo"].item()), float(stats["yhi"].item()),
              int(stats["ynb"].item()))
    if weights:
        if stats["wmin"].item() <= 0:
            raise ValueError("weights must be strictly positive")
        wsum = float(stats["wsum"].item())
    else:
        wsum = float(n_obs)
    # Analytic weights are written rescaled to average 1 over the sample, as
    # Stata does for maximum likelihood: the estimates do not change, but the
    # iid vcov and the log-likelihood would otherwise depend on the weights'
    # scale. Frequency weights are counts, and stay.
    wscale = n_obs / wsum if weights and weights_type != "fweights" else 1.0
    xbinary = np.array([stats[f"b{j}"].item() == 0 for j in range(K)], dtype=bool)
    xnonzero = np.array([stats[f"z{j}"].item() or 0.0 for j in range(K)], dtype=float)

    cluster_tables = {name: factorize(engine, design, cc, f"cluster {name}", CODE)
                      for name, cc in clusters.items()}
    fe_tables = {d: factorize(engine, design, cc, f"fixed effect {d}", CODE)
                 for d, cc in fe_cols.items()}
    store = BlockStore(run.path / "blocks", in_memory, dtype,
                       binary=xbinary if COMPACT_BINARY else None)
    cjoins = [(clusters[name], t) for name, t in cluster_tables.items()]
    fjoins = [(fe_cols[d], t) for d, t in fe_tables.items()]

    # the splits, and the groups' and the covariates' sums (by split), formed
    # as the blocks are written
    split_of = None
    S = 1
    if split_by:
        t = cluster_tables[SPLIT]
        vals = t[split_by].to_numpy()
        codes = t[CODE].to_numpy()
        if (not np.issubdtype(vals.dtype, np.integer) or vals.min() < 0
                or vals.max() >= n_splits):
            raise ValueError(f"split_by: {split_by!r} must hold integers 0 to {n_splits - 1}")
        split_of = np.zeros(int(codes.max()) + 1, np.int64)
        split_of[codes] = vals
        S = n_splits
    fe = spec.fe[0] if spec.fe else None
    G0 = fe_tables[fe].height if fe else 1
    gw = np.zeros((S, G0))
    gy = np.zeros((S, G0))
    gn = np.zeros(G0, np.int64)
    gpos = np.zeros(G0, np.int64)
    xs, xq = np.zeros((S, K)), np.zeros((S, K))
    xcont = np.flatnonzero(~xbinary)       # the squares of 0/1 columns are the columns

    def add_sums(X, y, w, f, cl):
        n = len(y)
        f = f if fe else np.zeros(n, np.int64)
        s = split_of[cl[:, 0]] if split_of is not None else None
        wt = np.ones(n) if w is None else w
        idx = f if s is None else s * G0 + f
        gw[...] += np.bincount(idx, weights=wt, minlength=S * G0).reshape(S, G0)
        gy[...] += np.bincount(idx, weights=wt * y, minlength=S * G0).reshape(S, G0)
        gn[...] += np.bincount(f, minlength=G0)
        gpos[...] += np.bincount(f, weights=(y > 0), minlength=G0).astype(np.int64)
        for j in range(S):
            r = None if s is None else s == j
            if r is not None and not r.any():
                continue
            Xj, wj = (X, wt) if r is None else (X[r], wt[r])
            sj = wj @ Xj
            xs[j] += sj
            xq[j] += sj                         # the 0/1 columns' squares
            if len(xcont):
                xq[j, xcont] += wj @ (Xj[:, xcont] ** 2) - sj[xcont]

    def convert(frame: pl.DataFrame):
        frame = frame.select(evaluated)                 # the design, for this batch
        cl = fc = None
        if cjoins:
            cl = np.column_stack([
                frame.select(cc).join(t, on=cc, how="left", maintain_order="left")[CODE]
                .to_numpy() for cc, t in cjoins]).astype(np.int64)
        if fjoins:
            cc, t = fjoins[0]
            fc = (frame.select(cc).join(t, on=cc, how="left", maintain_order="left")[CODE]
                  .to_numpy().astype(np.int64))
        X = (frame.select([_xcol(j) for j in range(K)]).to_numpy(order="c")
             if K else np.zeros((frame.height, 0)))
        X = X.astype(np.float64, copy=False)
        y = frame[Y].to_numpy().astype(np.float64)
        w = frame[W].to_numpy().astype(np.float64) * wscale if weights else None
        add_sums(X, y, w, fc, cl)
        store.add({"X": X, "y": y, "w": w,
                   "off": frame[OFF].to_numpy().astype(np.float64)
                   if off_expr is not None else None,
                   "cl": cl, "fe": fc, "row": frame[ROW].to_numpy().astype(np.int64)})

    log(f"pass 0: writing {n_obs:,} rows as blocks of {rows:,}"
        + (" (in memory)" if in_memory else ""))
    fe_sorted = False
    if len(fe_cols) == 1:
        d = next(iter(fe_cols))
        fcols = fe_cols[d]
        big = engine.collect(design.group_by(fcols).len().select(pl.col("len").max()),
                             "largest fixed-effect level").item()
        fe_sorted = big <= 4 * rows
        if not fe_sorted:
            log(f"pass 0: the largest level of {d} has {big:,} rows, more than a block "
                "holds; the rows stay in their order")
        elif (need := _partition_gb(design_path, rows)) > budget.headroom_gb():
            if not _grouped(design_path, fcols, fe_tables[d].height, rows):
                fe_sorted = False
                log(f"pass 0: sorting the rows by {d} needs about {need:.1f} GB and "
                    f"{budget.headroom_gb():.1f} GB is free ({budget}); the rows stay in "
                    "their order (slower passes)")
    if fe_sorted:
        for frame in iter_groups(engine, design_path, fcols, rows, run.path,
                                 fe_tables[d].height, [*fcols, ROW], log=log):
            convert(frame)
    else:
        for frame in iter_batches(design_path, rows):
            convert(frame)
    design_path.unlink(missing_ok=True)

    single = (gn == 1) if fe and drop_singletons else np.zeros(G0, bool)
    if fam.name == "poisson":
        bad = gpos == 0                         # an outcome that is always zero
    else:
        bad = (gpos == 0) | (gpos == gn)        # an outcome that never varies
    bad &= ~single
    drop = single | bad
    keep = ~drop if fe else np.ones(1, bool)
    separation = {"checked": bool(fe), "levels": 0, "observations": 0}
    dropped_levels = {"singleton": None, "separation": None}
    if fe:
        t = fe_tables[fe]
        codes = t[CODE].to_numpy()
        dropped_levels = {k: t.filter(pl.Series(mask[codes])).drop(CODE)
                          for k, mask in (("singleton", single), ("separation", bad))}
    if fe and bad.any():
        n_drop = int(gn[bad].sum())
        separation.update(levels=int(bad.sum()), observations=n_drop)
        why = ("their outcome is always zero" if fam.name == "poisson"
               else "their outcome never varies")
        warn(f"{n_drop:,} observations in {int(bad.sum()):,} levels of {fe} dropped "
             f"because {why} (their effects would be infinite)", None)
    if fe and drop.any():
        if not keep.any():
            raise ValueError(f"{spec.fml}: no observations left after dropping the levels of "
                             f"{fe} whose effects would be infinite")
        for blk in store.iterate(["X", "w", "fe", "cl"]):
            gone = drop[blk["fe"]]
            if not gone.any():
                continue
            n = len(gone)
            s = (split_of[blk["cl"][:, 0]] if split_of is not None
                 else np.zeros(n, np.int64))[gone]
            wt = (np.ones(n) if blk.get("w") is None else blk["w"])[gone]
            X = blk["X"][gone]
            for j in range(S):
                r = s == j
                if r.any():
                    xs[j] -= wt[r] @ X[r]
                    xq[j] -= wt[r] @ (X[r] ** 2)
    gw, gy = gw[:, keep], gy[:, keep]
    n_kept = int(n_obs - gn[drop].sum()) if fe else int(n_obs)
    if weights and weights_type != "fweights" and fe and drop.any():
        # analytic weights average 1 over the rows kept, as hdfe-stream's
        # estimators scale them: the stored weights and the sums rescaled
        factor = n_kept / float(gw.sum())
        for blk in store.blocks:
            if store.in_memory:
                blk.arrays["w"] = blk.arrays["w"] * factor
            else:
                np.save(blk.files["w"], np.load(blk.files["w"]) * factor)
        gw, gy, xs, xq = gw * factor, gy * factor, xs * factor, xq * factor
        wscale *= factor
    G = int(keep.sum())
    remap = np.cumsum(keep) - 1
    levels = None
    if fe:
        levels = fe_tables[fe].sort(CODE).filter(pl.Series(keep)).drop(CODE)
    counts = None if fam.name == "poisson" else np.stack([gw - gy, gy], axis=-1)
    return GLMDesign(
        formula=spec.fml, family=fam.name, depvar=spec.depvar,
        names=[nm for nm, _ in spec.x], fe=fe, n_groups=G, fe_levels=levels, n_splits=S,
        n_obs=n_kept, weight_sum=float(gw.sum()),
        group_weights=gw, group_outcomes=gy, outcome_counts=counts, x_sums=xs,
        x_squares=xq, x_nonzero=xnonzero, separation=separation,
        diagnostics={"passes": 0, "pass0_seconds": round(time.time() - t0, 2),
                     "n_raw": n_raw, "fe_sorted": fe_sorted,
                     "cluster_tables": cluster_tables,
                     "singletons": int(gn[single].sum()) if fe else 0,
                     "weight_scale": float(wscale)},
        dropped_levels=dropped_levels, usable=usable, exprs=dict(spec.x), y_expr=spec.y,
        offset_expr=off_expr,
        fe_table=fe_tables[fe] if fe else None, x_binary=xbinary,
        fe_sorted=fe_sorted, _store=store, _run=run, _keep=keep, _split_of=split_of,
        _remap=remap, _chunk_bytes=float(chunk_mb) * 1e6, _has_offset=off_expr is not None)

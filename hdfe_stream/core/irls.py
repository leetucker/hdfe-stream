"""IRLS machinery shared by estimators whose every step is a weighted
least-squares fit of a working response on the covariates and the fixed
effects -- the regression `StreamingHDFE` solves -- with weights that change
from step to step: `_IRLSMixin`, the coefficient state `_State`, and the
offset's variable name.

`hdfe_stream.glm` documents the scheme (one row pass and one solve per step,
nothing row-sized written). What is model-specific -- the fused row pass that
evaluates the model's weights and working response, the loop, and the fit
statistics -- stays with the estimator.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from types import SimpleNamespace

import numba as nb
import numpy as np

from ._columns import WCOL, vcol
from .inference import _stack
from .kernels_base import _nb_assemble, _nb_diag
from .kernels_glm import _nb_group_effects

_OFFSET = "__offset__"          # the offset's variable name in the row files


@dataclass
class _State:
    """Coefficients that give every row's linear predictor, or (`init`) the
    starting values, which are a function of the row's outcome."""

    beta: np.ndarray = None
    fe: np.ndarray = None       # non-streamed effects, stacked as in Gamma
    a_id: np.ndarray = None     # fe[0] effects of the identifying groups
    a_s: np.ndarray = None      # fe[0] effects of the single-cell groups
    init: bool = False

    def toward(self, other, t):
        """The coefficients a fraction t of the way to `other`; the linear
        predictor is linear in them."""
        return _State(*(a + t * (b - a) for a, b in (
            (self.beta, other.beta), (self.fe, other.fe),
            (self.a_id, other.a_id), (self.a_s, other.a_s))))

    def zeros_like(self):
        return _State(*(np.zeros_like(a) for a in (self.beta, self.fe, self.a_id, self.a_s)))


class _IRLSMixin:
    """The weighted least-squares steps of an IRLS estimator over the streamed
    passes, independent of the model: the coefficient state and the
    linear predictor it gives each row, the layout of the cells' sums, one
    weighted least-squares fit of a working response from those sums
    (`_wls`), and the coefficients it gives, fe[0] group effects included
    (`_update`).

    Mixed in ahead of `StreamingHDFE` (see `hdfe_stream.glm.StreamingGLM`).
    The subclass provides the rest of IRLS: the row pass that fills the cell
    sums for a step (an object with ``W``, ``raw``, ``n_id``, ``sums_id``,
    ``n_s`` and ``sums_s``), the loop that iterates it, and the model's own
    statistics. It sets ``yname`` (the one outcome), ``offset_name`` (None
    without an offset), ``_raw_w`` and ``_bufs``; `_eta` reads
    ``family.init_eta`` and `_layout` sets ``ybar`` for the starting values.
    """

    # cells are runs of rows, and their sums come from each step's row pass
    # rather than from pass 1 (see passes.py)
    cells_contiguous = True
    cell_sums = False
    # the factor the weights are read with: the estimator may rescale them
    # (StreamingGLM: analytic weights to average 1)
    _wscale = 1.0

    # ------------------------------------------------------------ layout
    def _layout(self):
        """Counts that fix where each cell's sums go: identifying cells as
        pass 1 stored them, then one row per single-cell group."""
        G = self.n_levels.get(self.g_fe, 0) if self.fe else 0
        self.G_id = len(self.starts) - 1
        self.G_s = G - self.G_id
        self.M_id = int(self.starts[-1])
        self.s_starts = np.arange(self.G_s + 1, dtype=np.int64)
        self.codes_s = None             # filled by the first row pass
        self._bufs = {}
        self.ybar = float(self.means[self.vidx[self.yname]])

    @contextmanager
    def _quiet(self, on):
        verbose = self.verbose
        self.verbose = verbose and not on
        try:
            yield
        finally:
            self.verbose = verbose

    # ------------------------------------------------------------ weighted least squares
    def _wls(self, agg, names, x0):
        """One weighted least-squares fit of the working response: Gamma for
        [z, x] and the assembled [z, x]' M W M [z, x]."""
        c = agg.W.shape[0]
        _, L = self._offsets()
        if not L:
            return np.zeros((0, c)), agg.W.copy(), {"solver": "none", "iterations": 0,
                                                   "converged": True}
        self.n = agg.n_id
        self.diag = np.zeros(L)
        _nb_diag(self.starts, self.codes, self.n, self.offs, self.diag)
        gamma, info = self._solve(sums=agg.sums_id, names=names, raw_ss=agg.raw, x0=x0,
                                  in_memory=True)
        # Left to choose, the solver builds S for the first step and streams
        # the cells after that. S has to be rebuilt whenever the weights
        # change, and on the AKM panels rebuilding it cost more than streaming
        # the cells once per CG iteration, the more so as later steps start
        # from the previous solution and need fewer iterations (docs/examples/glm.md).
        # AMG needs S, so with it S is rebuilt every step, and so does the GPU
        # solve (device="cuda"), which runs only on S.
        if self.solver == "auto":
            self.solver = ("explicit" if (self.precond == "amg" or self.device == "cuda")
                           and info["solver"] != "stream_cg" else "stream_cg")
        acc = np.zeros((nb.get_num_threads(), c, c))
        _nb_assemble(self.starts, self.codes, self.n, agg.sums_id, self.offs, gamma, acc)
        return gamma, agg.W + acc.sum(axis=0), info

    def _update(self, agg, gamma, A):
        """The coefficients that solve one weighted least-squares step; with
        ``_fixed_beta`` set, the fixed effects that solve it with those
        slopes held (an offset X beta, as alpaca's `feglmOffset`)."""
        k = A.shape[0] - 1
        fixed = getattr(self, "_fixed_beta", None)
        if fixed is not None:
            beta = np.asarray(fixed, dtype=np.float64)
        else:
            beta = np.linalg.solve(A[1:, 1:], A[1:, 0]) if k else np.zeros(0)
        fe = np.ascontiguousarray(gamma[:, 0] - gamma[:, 1:] @ beta)
        a_id, a_s = np.zeros(self.G_id), np.zeros(self.G_s)
        if self.fe:
            _nb_group_effects(self.starts, self.codes, agg.n_id, agg.sums_id, self.offs,
                              fe, beta, a_id)
            _nb_group_effects(self.s_starts, self.codes_s, agg.n_s, agg.sums_s, self.offs,
                              fe, beta, a_s)
        return _State(beta, fe, a_id, a_s)

    def _select(self, agg, sel, slot):
        """The cell sums of a subset of the working regression's columns."""
        out = SimpleNamespace(**vars(agg))
        c = len(sel)
        out.W, out.raw = agg.W[np.ix_(sel, sel)], agg.raw[sel]
        out.sums_id = self._buffer(("sums_id", c), (self.M_id, c))
        out.sums_id[:] = agg.sums_id[:, sel]
        out.sums_s = self._buffer(("sums_s", c), (self.G_s, c))
        out.sums_s[:] = agg.sums_s[:, sel]
        return out

    def _buffer(self, key, shape):
        """A cell-sized work array, made once per key and then reused:
        memory-mapped in the run directory unless cells_in_memory. Reusing
        (rather than reopening) matters, as reopening a mapped file for
        writing truncates it under any map still open on it."""
        buf = self._bufs.get(key)
        if buf is None:
            if self.cells_in_memory or 0 in shape:
                buf = np.zeros(shape)
            else:
                name = "glm_" + "_".join(map(str, key)) + ".npy"
                buf = np.lib.format.open_memmap(str(self.workdir / name), mode="w+",
                                                dtype=np.float64, shape=shape)
            self._bufs[key] = buf
        return buf

    def _release_intermediates(self):
        self._bufs = {}
        super()._release_intermediates()

    # ------------------------------------------------------------ rows
    def _columns(self, yj, xj):
        cols = [*self.ccols, vcol(yj), *[vcol(j) for j in xj]]
        if self.weights is not None:
            cols.append(WCOL)
        if self.offset_name is not None:
            cols.append(vcol(self.vidx[_OFFSET]))
        return list(dict.fromkeys(cols))

    def _cell_layout(self, starts, codes, cursor):
        """The cells of a chunk of whole fe[0] groups, and where its groups'
        coefficients and cells' sums are; advances `cursor` (identifying
        cells, identifying groups, single-cell groups) past the chunk."""
        if not self.fe:
            return None
        n = len(codes)
        if self.o_fe:
            brk = np.zeros(n, bool)
            brk[starts[:-1]] = True
            brk[1:] |= np.any(codes[1:] != codes[:-1], axis=1)
            cstarts = np.append(np.flatnonzero(brk), n).astype(np.int64)
        else:
            cstarts = starts                        # one dimension: a cell is a group
        gcell = np.searchsorted(cstarts, starts).astype(np.int64)
        ncell = np.diff(gcell)
        is_id = ncell > 1
        cell_id = np.repeat(is_id, ncell)
        lay = SimpleNamespace(starts=starts, cstarts=cstarts, gcell=gcell, is_id=is_id,
                              cell_id=cell_id,
                              Ng=np.diff(starts), c0=cursor[0], g0=cursor[1], s0=cursor[2],
                              nc=int(cell_id.sum()), ng=int(is_id.sum()),
                              ns=int(len(is_id) - is_id.sum()))
        cursor[0] += lay.nc
        cursor[1] += lay.ng
        cursor[2] += lay.ns
        return lay

    def _chunk_data(self, ch, yj, xj):
        y = np.ascontiguousarray(ch[vcol(yj)], dtype=np.float64)
        X = _stack(ch, [vcol(j) for j in xj]) if xj else np.zeros((len(y), 0))
        f = (np.ascontiguousarray(ch[WCOL], dtype=np.float64) * self._wscale
             if self.weights is not None else None)
        off = (np.ascontiguousarray(ch[vcol(self.vidx[_OFFSET])], dtype=np.float64)
               if self.offset_name is not None else None)
        return y, X, f, off

    def _eta(self, s, y, X, off, codes, lay):
        """Each row's linear predictor under coefficients `s`."""
        if s.init:
            return self.family.init_eta(y, self.ybar)
        eta = X @ s.beta if X.shape[1] else np.zeros(len(y))
        if lay is not None:
            a = np.empty(len(lay.is_id))
            a[lay.is_id] = s.a_id[lay.g0:lay.g0 + lay.ng]
            a[~lay.is_id] = s.a_s[lay.s0:lay.s0 + lay.ns]
            eta += np.repeat(a, lay.Ng)
            for d in range(len(self.o_fe)):
                eta += s.fe[self.offs[d] + codes[:, d]]
        if off is not None:
            eta += off
        return eta

    def _group_effects(self, state, lay):
        """Each of the chunk's groups' fe[0] effect under `state` (zeros at
        the start, when there are none yet)."""
        a = np.zeros(len(lay.is_id))
        if not state.init:
            a[lay.is_id] = state.a_id[lay.g0:lay.g0 + lay.ng]
            a[~lay.is_id] = state.a_s[lay.s0:lay.s0 + lay.ns]
        return a

    # ------------------------------------------------------------ inference
    def _collinear(self, Axx, idx=None, raw=None):
        if raw is None and idx is not None and self._raw_w is not None:
            raw = self._raw_w[idx]
        return super()._collinear(Axx, idx, raw)

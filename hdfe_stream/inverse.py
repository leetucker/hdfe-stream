"""Applying S^- and the projection A S^- A' to arbitrary vectors, out of core.

Machinery for leave-out estimation, not yet a public API. The fitted model
solves for its own coefficients; leverages and the Kline-Saggio-Solvsten bias
correction need something more general -- the ability to push an arbitrary
vector through the inverse of the normal-equations matrix. That is what this
provides.

The cost of one call is **two passes over the rows and one reduced solve**, for
a whole block of vectors at once. The reduced operator is built once and cached,
so repeated calls -- which is how the Johnson-Lindenstrauss approximation uses
this, hundreds of times -- pay for it only on the first.

Availability: the row files are deleted when a fit finishes, so this needs a fit
made with `keep_intermediates=True`, and `reload_intermediates()` afterwards to
map the cell arrays back in. `solver="within"` is not supported: its backend
solves for the design's own right-hand sides and does not expose the operator.
"""

from __future__ import annotations

from contextlib import contextmanager

import numba as nb
import numpy as np

from ._columns import GCODE, WCOL, vcol
from .kernels_inverse import (_nb_center_rows, _nb_project_coef, _nb_project_rows,
                              _nb_rademacher, _nb_reduce_rows, _nb_spread_groups)
from .utils import iter_group_chunks

# Shared-state contract with the other mixins
# -------------------------------------------
# Reads, set by _PassesMixin: paths["rows"], n_levels, offs, ccols, and the cell
# arrays the solver needs (starts, codes, n, sums, diag).
# From StreamingHDFE: weights, batch_rows, rhs_block, solver, and the layout.
#
# Sets: _reduced_solver, a cached (solve, info) pair. Nothing else reads it.


class _InverseMixin:

    def reload_intermediates(self):
        """Map the cell arrays back in after a fit.

        `_release_intermediates` drops the references even when the files are
        kept, so anything using the operator after a fit has to ask for them
        again.
        """
        self._load_ident()
        return self

    def _reduced(self):
        """The cached (solve, info) pair for the reduced system.

        `_make_block_solver` builds the explicit matrix S when it can, which is
        the expensive part; caching it is what makes hundreds of right-hand
        sides affordable.
        """
        if getattr(self, "_reduced_solver", None) is None:
            if self.solver == "within":
                raise NotImplementedError(
                    "applying the inverse to arbitrary vectors needs the "
                    "'explicit' or 'stream_cg' solver; 'within' only solves for "
                    "the design's own right-hand sides")
            self._reduced_solver = self._make_block_solver()
        return self._reduced_solver

    def _row_columns(self):
        weighted = self.weights is not None
        cols = [GCODE, *self.ccols] + ([WCOL] if weighted else [])
        return cols, weighted

    def _covariate_columns(self, covariates):
        """Row-file column names for a list of covariate variable names."""
        missing = [c for c in covariates if c not in self.vidx]
        if missing:
            raise ValueError(f"not variables of this fit: {missing}; "
                             f"available: {list(self.vidx)}")
        return [vcol(self.vidx[c]) for c in covariates]

    @staticmethod
    def _covariate_matrix(chunk, columns, n_rows):
        if not columns:
            return np.zeros((n_rows, 0))
        return np.column_stack([chunk[c] for c in columns]).astype(np.float64)

    def _chunk_layout(self, chunk):
        """(starts, codes, w) for a chunk already in hand."""
        gcode = chunk[GCODE]
        n_rows = len(gcode)
        starts = np.concatenate(
            ([0], np.flatnonzero(gcode[1:] != gcode[:-1]) + 1, [n_rows]))
        codes = np.column_stack([chunk[c] for c in self.ccols]).astype(np.int64)
        w = (np.ascontiguousarray(chunk[WCOL], dtype=np.float64)
             if self.weights is not None else np.ones(n_rows))
        return starts, codes, w

    def _chunks(self, extra_columns=()):
        """Iterate the row files in fixed order, yielding whole groups.

        Yields (ordinal, chunk, starts, codes, w). `ordinal` is the running row
        count, which is what makes a regenerated random vector line up across
        two passes.
        """
        cols, weighted = self._row_columns()
        cols = list(dict.fromkeys([*cols, *extra_columns]))
        ordinal = 0
        for chunk in iter_group_chunks(self.paths["rows"], cols, self.batch_rows):
            gcode = chunk[GCODE]
            n_rows = len(gcode)
            starts = np.concatenate(
                ([0], np.flatnonzero(gcode[1:] != gcode[:-1]) + 1, [n_rows]))
            codes = np.column_stack([chunk[c] for c in self.ccols]).astype(np.int64)
            w = (np.ascontiguousarray(chunk[WCOL], dtype=np.float64) if weighted
                 else np.ones(n_rows))
            yield ordinal, chunk, starts, codes, w
            ordinal += n_rows

    def _reduce(self, vectors, n_vectors, covariate_columns, extra_columns,
                total_levels, n_threads, sqrt_weights=False):
        """One pass over the rows, forming both halves of the reduced
        right-hand side: the level block and the covariate block."""
        n_cov = len(covariate_columns)
        acc = np.zeros((n_threads, total_levels, n_vectors))
        acc_x = np.zeros((n_threads, n_cov, n_vectors))
        for ordinal, chunk, starts, codes, w in self._chunks(
                tuple(extra_columns) + tuple(covariate_columns)):
            R = np.ascontiguousarray(vectors(ordinal, len(w), n_vectors),
                                     dtype=np.float64)
            if sqrt_weights:
                R = np.ascontiguousarray(R * self._sqrt_scale(w)[1][:, None])
            X = self._covariate_matrix(chunk, covariate_columns, len(w))
            _nb_center_rows(starts, codes, self.offs, w, X, R, acc, acc_x)
        return acc.sum(axis=0), acc_x.sum(axis=0)

    def _border(self, covariates):
        """Precompute the covariate border of the reduced system, once.

        Partialling out the streamed dimension leaves a system over
        [covariates, other fixed effects] whose bottom-right block is the
        reduced matrix the solver already knows how to apply. The covariate
        block is k x k, so the whole thing is handled by a Schur complement on
        that small corner:

            Z     = T_dd^- T_dx          k solves, done here and cached
            Schur = T_xx - T_dx' Z       k x k, dense

        Both are properties of the design alone, so hundreds of projections
        share them.
        """
        key = tuple(covariates)
        cache = getattr(self, "_border_cache", None)
        if cache is None:
            cache = self._border_cache = {}
        if key in cache:
            return cache[key]

        columns = self._covariate_columns(covariates)
        if not columns:
            cache[key] = (columns, None, None)
            return cache[key]

        solve, _ = self._reduced()
        _, total_levels = self._offsets()
        n_threads = nb.get_num_threads()

        # feeding the covariates themselves through the reduction gives
        # T_dx = D_o'W M_0 X and T_xx = X'W M_0 X in one pass
        T_dx, T_xx = self._reduce_design(columns, total_levels, n_threads)
        Z, _, _, _ = solve(np.ascontiguousarray(T_dx))
        schur = T_xx - T_dx.T @ Z
        cache[key] = (columns, Z, np.linalg.pinv(schur, hermitian=True))
        return cache[key]

    def _reduce_design(self, columns, total_levels, n_threads):
        """T_dx and T_xx: the reduction applied to the covariates themselves."""
        n_cov = len(columns)
        acc = np.zeros((n_threads, total_levels, n_cov))
        acc_x = np.zeros((n_threads, n_cov, n_cov))
        for _, chunk, starts, codes, w in self._chunks(tuple(columns)):
            X = self._covariate_matrix(chunk, columns, len(w))
            _nb_center_rows(starts, codes, self.offs, w, X, X, acc, acc_x)
        return acc.sum(axis=0), acc_x.sum(axis=0)

    @staticmethod
    def _sqrt_scale(w):
        """w^(1/2) and w^(-1/2), for moving in and out of the square-root
        weight metric."""
        root = np.sqrt(w)
        return root, np.divide(1.0, root, out=np.zeros_like(root), where=root > 0)

    def project_rows(self, vectors, n_vectors, sink, extra_columns=(),
                     covariates=(), sqrt_weights=False):
        """Apply the projection A S^- A' to row-level vectors.

        `vectors(ordinal, n_rows, n_vectors) -> (n_rows, n_vectors) float64`
        produces the block of vectors for a chunk of rows. It is called twice per
        chunk, once in each pass, and must return the same thing both times --
        generate from `ordinal`, do not consume a stateful stream.

        `sink(ordinal, chunk, projected)` receives each chunk's results, where
        `projected[i, j] = (A S^- A' r_j)_i`. Nothing is accumulated here: what
        to do with the values is the caller's business, which is what keeps this
        usable both for per-row output and for scalar reductions.

        `covariates` names the variables that belong in the design alongside the
        fixed effects. Leaving it empty projects onto the fixed effects alone,
        which is a *different operator* -- for a model with controls, leverages
        from the fixed effects alone are leverages of the wrong design.

        `sqrt_weights` applies the operator in the square-root weight metric,
        W^(1/2) A S^- A' W^(1/2), instead of A S^- A' W. The two agree without
        weights. With them only the first is symmetric and idempotent, which is
        what a leverage estimate needs: the Rademacher trick estimates the
        diagonal of the square of the operator, and that equals the diagonal
        itself only when the operator is a projection in the usual sense. It is
        the same operator either side of a rescaling, so it costs nothing.

        Returns the solver info dict for the reduced solve.
        """
        solve, info = self._reduced()
        _, total_levels = self._offsets()
        n_threads = nb.get_num_threads()
        columns, Z, schur_inv = self._border(covariates)

        # pass 1: reduce the rows to the right-hand side of the reduced system
        rhs, rhs_x = self._reduce(vectors, n_vectors, columns, extra_columns,
                                  total_levels, n_threads, sqrt_weights)

        # No normalization of the solution. S is singular -- the fixed effects
        # are identified only up to the usual shifts -- so `solve` returns one
        # particular solution out of many. That is harmless here: the streamed
        # block is then derived from this one, and A applied to the pair
        # annihilates exactly the null space, so the row-level projection is the
        # same whichever solution came back. `_normalize` exists to make the
        # *reported* fixed effects comparable, which is not what this is for.
        U, iterations, converged, rel = solve(rhs)

        if columns:
            # Schur complement on the small covariate corner
            C = schur_inv @ (rhs_x - Z.T @ rhs)
            U = U - Z @ C
        else:
            C = np.zeros((0, n_vectors))

        # pass 2: take the solution back to row level
        for ordinal, chunk, starts, codes, w in self._chunks(
                tuple(extra_columns) + tuple(columns)):
            R = np.ascontiguousarray(vectors(ordinal, len(w), n_vectors),
                                     dtype=np.float64)
            X = self._covariate_matrix(chunk, columns, len(w))
            if sqrt_weights:
                root, inverse_root = self._sqrt_scale(w)
                R = R * inverse_root[:, None]
            out = np.empty_like(R)
            _nb_project_rows(starts, codes, self.offs, w, X, C, R, U, out)
            if sqrt_weights:
                out = out * root[:, None]
            sink(ordinal, chunk, out)

        return {**info, "iterations": iterations, "converged": bool(converged),
                "max_rel_residual": float(rel), "n_covariates": len(columns)}

    def apply_inverse(self, levels, covariate_values, streamed, covariates=(),
                      sink=None, extra_columns=()):
        """u = S^- v for a vector given in coefficient space.

        The companion to `project_rows`, which takes its right-hand side from
        the rows. Here the three blocks are supplied directly:

            levels           (L, m)  the non-streamed fixed effects
            covariate_values (k, m)  the covariates, matching `covariates`
            streamed         (G, m)  the streamed dimension, one row per group

        Returns `(u_levels, u_covariates, u_streamed)` with the same shapes. If
        `sink(ordinal, chunk, values)` is given it also receives `(A u)_i` per
        row, which is usually what the caller actually wants.

        `streamed` is the one array here indexed by the streamed dimension. It
        is G x m, so a million groups and a block of 8 vectors is 64 MB --
        smaller than the row-sized arrays the leverage pass already keeps, and
        bounded by the block size. `project_rows` avoids it entirely; a
        coefficient-space right-hand side cannot.
        """
        solve, info = self._reduced()
        _, total_levels = self._offsets()
        n_threads = nb.get_num_threads()
        columns, Z, schur_inv = self._border(covariates)
        n_vectors = levels.shape[1]
        levels, streamed, null_share = self._drop_null_component(
            levels, covariate_values, streamed)

        # pass 1: rho = P^-1 v_0 spread over rows, reduced onto [X, D_o]
        acc = np.zeros((n_threads, total_levels, n_vectors))
        acc_x = np.zeros((n_threads, len(columns), n_vectors))
        for _, chunk, starts, codes, w in self._chunks(
                tuple(extra_columns) + tuple(columns)):
            rho = self._spread(streamed, chunk, starts, w, n_vectors)
            X = self._covariate_matrix(chunk, columns, len(w))
            _nb_reduce_rows(starts, codes, self.offs, w, X, rho, acc, acc_x)

        rhs = levels - acc.sum(axis=0)
        rhs_x = covariate_values - acc_x.sum(axis=0)
        del acc, acc_x

        U, iterations, converged, rel = solve(np.ascontiguousarray(rhs))
        if columns:
            C = schur_inv @ (rhs_x - Z.T @ rhs)
            U = U - Z @ C
        else:
            C = np.zeros((0, n_vectors))

        # pass 2: the streamed block follows from the rest, group by group
        U_streamed = np.empty_like(streamed)
        for ordinal, chunk, starts, codes, w in self._chunks(
                tuple(extra_columns) + tuple(columns)):
            rho = self._spread(streamed, chunk, starts, w, n_vectors)
            X = self._covariate_matrix(chunk, columns, len(w))
            rows = np.empty_like(rho)
            first = int(chunk[GCODE][0])
            groups = U_streamed[first:first + len(starts) - 1]
            _nb_project_coef(starts, codes, self.offs, w, X, C, rho, U, rows, groups)
            if sink is not None:
                sink(ordinal, chunk, rows)

        return U, C, U_streamed

    def _null_space(self):
        """A basis for the null space of S, in (streamed, levels) blocks.

        Keeping every level of every dimension makes S singular: the fitted
        values are unchanged by adding a constant to one dimension and taking it
        off another, and by doing so separately within each connected component.
        Those shifts are exactly the null space, and they are known structurally
        rather than needing to be discovered:

          * one vector per connected component -- plus one on the component's
            streamed groups, minus one on its levels of the first other
            dimension;
          * one vector per further dimension -- plus one across the first other
            dimension, minus one across that one.

        A right-hand side given in coefficient space has to have its component
        along these removed before it can be solved for at all. One built from
        rows never does, which is why `project_rows` needs none of this.
        """
        if getattr(self, "_null_basis", None) is not None:
            return self._null_basis

        off, total_levels = self._offsets()
        n_groups = self.n_levels[self.g_fe]
        first = self.o_fe[0]
        comp = np.asarray(self.comp)
        n_components = int(comp.max()) + 1 if len(comp) else 0

        group_component = self._group_components(off[first])
        vectors = []
        for c in range(n_components):
            streamed = (group_component == c).astype(float)
            levels = np.zeros(total_levels)
            block = slice(off[first], off[first] + self.n_levels[first])
            levels[block] = -(comp == c).astype(float)
            vectors.append((streamed, levels))
        for other in self.o_fe[1:]:
            streamed = np.zeros(n_groups)
            levels = np.zeros(total_levels)
            levels[off[first]:off[first] + self.n_levels[first]] = 1.0
            levels[off[other]:off[other] + self.n_levels[other]] = -1.0
            vectors.append((streamed, levels))

        if not vectors:
            self._null_basis = (np.zeros((n_groups, 0)), np.zeros((total_levels, 0)))
            return self._null_basis

        streamed = np.column_stack([v[0] for v in vectors])
        levels = np.column_stack([v[1] for v in vectors])
        # orthonormalize the stacked basis so the projection is a plain product
        stacked = np.vstack([streamed, levels])
        q, _ = np.linalg.qr(stacked)
        self._null_basis = (q[:n_groups], q[n_groups:])
        return self._null_basis

    def _group_components(self, level_offset):
        """Which connected component each streamed group belongs to."""
        n_groups = self.n_levels[self.g_fe]
        out = np.zeros(n_groups, np.int64)
        comp = np.asarray(self.comp)
        for _, chunk, starts, codes, _w in self._chunks():
            gcodes = chunk[GCODE][starts[:-1]]
            out[gcodes] = comp[codes[starts[:-1], 0]]
        return out

    def _drop_null_component(self, levels, covariate_values, streamed):
        """Project a coefficient-space vector onto the range of S.

        The dropped part is annihilated by A, so nothing observable is lost --
        it is only the arbitrary normalization of the fixed effects. Without
        this the system would simply be inconsistent and the solver would run
        away.
        """
        q_streamed, q_levels = self._null_space()
        if q_levels.shape[1] == 0:
            return levels, streamed, 0.0
        overlap = q_streamed.T @ streamed + q_levels.T @ levels
        size = np.linalg.norm(overlap) / max(
            np.linalg.norm(np.vstack([streamed, levels])), 1e-300)
        return (levels - q_levels @ overlap,
                streamed - q_streamed @ overlap, float(size))

    def _spread(self, streamed, chunk, starts, w, n_vectors):
        """The streamed block as per-row values, divided by the group weight."""
        first = int(chunk[GCODE][0])
        block = np.ascontiguousarray(streamed[first:first + len(starts) - 1])
        out = np.empty((len(w), n_vectors))
        _nb_spread_groups(starts, w, block, out)
        return out

    def rademacher(self, seed=0, first_draw=0):
        """A `vectors` callable producing reproducible Rademacher +/-1 draws.

        `first_draw` picks the stretch of the sequence to use, so a long run can
        be processed in blocks without any block repeating another's vectors.
        """
        def generate(ordinal, n_rows, n_vectors):
            out = np.empty((n_rows, n_vectors))
            _nb_rademacher(ordinal, n_rows, n_vectors, seed, first_draw, out)
            return out
        return generate

    def row_count(self):
        """Rows in the stored row files, in the order `project_rows` sees them."""
        return sum(len(w) for _, _, _, _, w in self._chunks())

    @contextmanager
    def _scratch_capped(self, columns):
        """Temporarily cap the row chunk so `columns` float64 columns fit budget.

        Safe to vary freely: every random vector these passes use is generated
        from the absolute row ordinal, not from a per-chunk counter, so the chunk
        size changes how much memory is held and nothing else. The numbers out
        are identical.
        """
        budget = int(getattr(self, "scratch_mb", 0) or 32) << 20
        want = max(1, budget // (max(int(columns), 1) * 8))
        previous = self.batch_rows
        self.batch_rows = max(1, min(previous, want))
        try:
            yield self.batch_rows
        finally:
            self.batch_rows = previous

    def _store_path(self, tag):
        """A fresh scratch path for a row-sized store.

        Unique per call, deliberately. These stores are now handed back as
        memory-mapped views rather than copied into RAM, so two results that
        shared a file would silently alias: a second `jla_leverages` would
        overwrite the first one's arrays while the caller still held them.
        """
        n = getattr(self, "_store_seq", 0)
        self._store_seq = n + 1
        return self.workdir / f"{tag}_{n:04d}.f64"

    def row_weights(self):
        """The regression weight per row, in that same order; ones if the fit
        was unweighted, so callers can treat both cases alike."""
        out = np.empty(self.row_count())
        for ordinal, _chunk, _starts, _codes, w in self._chunks():
            out[ordinal:ordinal + len(w)] = w
        return out

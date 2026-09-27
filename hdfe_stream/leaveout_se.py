"""Standard errors for the leave-out variance components.

The estimator is a quadratic form in the outcome with a zero diagonal. Writing
everything in the square-root-weight metric -- tilde design X = W^(1/2) A, tilde
outcome y = W^(1/2)(y - ybar), errors of variance sigma2_i = w_i Var(e_i), which
is exactly what `_leave_one_out_sigma2` returns -- the kernel is

    C = B - (1/2)(diag(b) M + M diag(b)),    b_i = B_ii / (1 - P_ii)

with B = X S^- Q S^- X', M = I - P, P = X S^- X'. Two facts drive everything
here:

  * C_ii = B_ii - M_ii b_i = 0 identically. So E[y'Cy] = mu'C mu = theta with no
    trace term, which is a one-line proof that the KSS estimator is unbiased;
    and y'Cy reproduces `plug_in - bias` exactly, which is how the implementation
    below is tested without any Monte Carlo.

  * because the diagonal vanishes, for independent errors

        V[theta-hat] = 4 sum_i sigma2_i (C mu)_i^2 + 2 tr(C Om C Om)

    with Om = diag(sigma2) and no appeal to normality: the third- and
    fourth-moment terms all carry a factor of C_ii. This is the variance in
    Theorem 2 of Kline, Saggio and Solvsten (2020), whose second term they write
    as 2 sum_{i != l} C_il^2 sigma2_i sigma2_l -- equal to the trace precisely
    because the diagonal is zero.

The estimator of V substitutes sigma2-tilde for sigma2 and y for mu:

    V-hat = 4 sum_i sigma2-tilde_i (C y)_i^2 - 2 tr(C Om-tilde C Om-tilde)

The first term overshoots by exactly 4 tr(C Om C Om), because
E[(Cy)_i^2] = (C mu)_i^2 + sum_l C_il^2 sigma2_l; since V carries
+2 tr(C Om C Om), subtracting 2 tr leaves the whole thing unbiased. That also
says the first term *alone* is conservative rather than anticonservative, which
is why it is reported separately.

What is not implemented, and why
-------------------------------
KSS's own sigma2-tilde is a cross-fit estimator built from two independent
leave-out predictions per observation, which in a two-way model means two
edge-disjoint paths through the worker-firm network, found by Dijkstra. Their
V-hat then classifies every (i, l) pair by the sparsity pattern of those
predictions. Neither is available to a streaming implementation -- the first is
a per-observation graph search, the second is O(n^2) -- and neither is in
Saggio's reference implementation either, which instead smooths the ordinary
leave-out sigma2 against (P_ii, B_ii) and estimates the trace by simulation.
That is what is done here. See docs/kss.md for the measured coverage.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numba as nb
import numpy as np

from .kernels_inverse import _nb_group_sums, _nb_reduce_rows
from .kernels_se import _nb_bii_moments, _nb_cov_cross, _nb_readout
from .report import _log

# rows per block when reducing row-sized arrays; bounds the working set
_BLOCK_ROWS = 1_000_000

# rows sampled to place the smoother's bin edges, so no full column is sorted
_QUANTILE_SAMPLE = 2_000_000

COMPONENTS = ("var(psi)", "var(alpha)", "cov(psi, alpha)")


@dataclass
class ComponentSE:
    """Standard errors for the three components, and the parts they came from."""

    se: dict
    variance: dict
    first_term: dict
    trace_term: dict
    se_conservative: dict
    theta: dict
    n_draws: int
    diagnostics: dict = field(default_factory=dict)
    # per weakly identified component: lambda_1, b1, the raw V[b1] theta1 is
    # formed with, and the 2x2 covariance of (b1, theta1)
    weak: dict = None

    def summary(self):
        lines = [f"leave-out standard errors ({self.n_draws} draws)",
                 f"{'component':<20}{'estimate':>12}{'se':>11}"
                 f"{'se (cons.)':>12}{'trace share':>13}",
                 "-" * 68]
        for key in COMPONENTS:
            share = (self.trace_term[key] / self.first_term[key]
                     if self.first_term[key] else float("nan"))
            lines.append(f"{key:<20}{self.theta[key]:>12.6f}{self.se[key]:>11.6f}"
                         f"{self.se_conservative[key]:>12.6f}{share:>12.2%}")
        return "\n".join(lines)


class _StandardErrorMixin:
    # Defined by other mixins: _chunks, _chunk_layout, _covariate_matrix,
    # _border, _model_covariates, _offsets, apply_inverse, row_count,
    # row_weights, rademacher, _quiet_solver, offs, o_fe, g_fe, n_levels,
    # rhs_block, weights, workdir, verbose, logger, log_level.

    # ---------------------------------------------------------------- weights
    def _level_weight_totals(self):
        """(per-level weight totals, per-group weight totals, grand total).

        The centered selectors that define the quadratic forms need the weight
        behind each level, so that a level's contribution can be centered without
        a second pass. Computed once and cached: it does not depend on the draw.
        """
        if getattr(self, "_lw_cache", None) is not None:
            return self._lw_cache

        n_threads = nb.get_num_threads()
        _, total_levels = self._offsets()
        n_groups = self.n_levels[self.g_fe]
        acc = np.zeros((n_threads, total_levels, 1))
        acc_x = np.zeros((n_threads, 0, 1))
        groups = np.zeros((n_groups, 1))
        for _ordinal, chunk, starts, codes, w in self._chunks():
            ones = np.ones((len(w), 1))
            empty = np.zeros((len(w), 0))
            _nb_reduce_rows(starts, codes, self.offs, w, empty, ones, acc, acc_x)
            first = int(chunk["gcode"][0])
            _nb_group_sums(starts, w, ones,
                           groups[first:first + len(starts) - 1])
        levels = acc.sum(axis=0)[:, 0]
        group_totals = groups[:, 0]
        self._lw_cache = (levels, group_totals, float(group_totals.sum()))
        return self._lw_cache

    # ------------------------------------------------------------- the design
    def _se_layout(self, covariates, psi):
        """The indices and sizes every routine below shares."""
        offsets, total_levels = self._offsets()
        psi = psi or self.o_fe[0]
        if psi not in self.o_fe:
            raise ValueError(f"psi must be a non-streamed dimension, one of "
                             f"{self.o_fe}; got {psi!r}")
        columns, _, _ = self._border(covariates)
        level_w, group_w, total = self._level_weight_totals()
        return {
            "psi": psi, "psi_offset": offsets[psi],
            "psi_column": self.o_fe.index(psi), "n_psi": self.n_levels[psi],
            "total_levels": total_levels,
            "n_groups": self.n_levels[self.g_fe], "columns": columns,
            "level_w": level_w, "group_w": group_w, "total": total,
            "norm": self._moment_divisor(total),
        }

    def _chunk_w(self, chunk, n):
        """Row weights for a chunk, or ones when the fit was unweighted.

        The row files carry a `w` column only for a weighted fit, which is why
        this is not simply an indexing expression.
        """
        if self.weights is None:
            return np.ones(n)
        return np.ascontiguousarray(chunk["w"], dtype=np.float64)

    def _outcome_mean(self, depcol):
        """The weight-weighted mean of the outcome, in one bounded pass."""
        total = 0.0
        weight = 0.0
        for _ordinal, chunk, _s, _c, w in self._chunks((depcol,)):
            y = np.asarray(chunk[depcol], dtype=np.float64)
            total += float(w @ y)
            weight += float(w.sum())
        return total / weight if weight else 0.0

    def _moment_divisor(self, total):
        """What the variance components divide by: n - 1, not n.

        LeaveOutTwoWay (`kss_quadratic_form.m`) and VarianceComponentsHDFE.jl
        divide both the plug-in and the bias term by the number of person-year
        observations less one. With weights the same factor applies to the
        weight total, T (R - 1) / R for R person-year rows, which reproduces
        the reference exactly at match level, where the weights are spell
        lengths and T = R. The *centering* of the quadratic forms still uses T:
        only their scale changes.
        """
        rows = self._person_years()
        return total * (rows - 1.0) / rows if rows > 1 else total

    def _person_years(self):
        """Person-year observations behind the fit. At match level a row is a
        match and the count is carried alongside; otherwise it is the rows."""
        stored = getattr(self, "_match_person_years", None)
        return float(stored) if stored is not None else float(self.row_count())

    @staticmethod
    def _inverse_root(w):
        root = np.sqrt(w)
        return root, np.divide(1.0, root, out=np.zeros_like(root), where=root > 0)

    # ------------------------------------------------------------------- B_ii
    def bii_rows(self, n_draws=250, block=None, seed=0, covariates=None,
                 psi=None, in_memory=False):
        """Per-row B_ii for all three components, by random projection.

        Each quadratic form factors as Q = L'L for the variances and
        Q = (L_psi' L_alpha + L_alpha' L_psi)/2 for the covariance, with
        L_d = W^(1/2) Z_d / sqrt(T) built from dimension d's weight-centered
        selector. So

            B_ii = || L S^- x_i ||^2  and  (L_psi S^- x_i)'(L_alpha S^- x_i)

        and for a random q with E[qq'] = I, E[(q'L S^- x_i)^2] is the first while
        E[(q'L_psi S^- x_i)(q'L_alpha S^- x_i)] is the second. Since
        q'L S^- x_i = sqrt(w_i) x_i' S^- (L'q), one pair of solves per draw gives
        all three diagonals at once.

        The point estimate deliberately never forms these: it estimates
        sum_i B_ii sigma2_i directly in coefficient space, which is cheaper and
        less noisy. Inference needs the individual values.

        Returns a dict keyed by component name, each a row-sized array in the
        row order the residual file uses.
        """
        block = self.rhs_block if block is None else int(block)
        if block < 1 or n_draws < 1:
            raise ValueError("n_draws and block must be at least 1")
        covariates = self._model_covariates(covariates)
        L = self._se_layout(covariates, psi)
        n_threads = nb.get_num_threads()
        n_cov = len(L["columns"])
        n_rows = self.row_count()
        root_total = np.sqrt(L["norm"])     # scale; the centering below uses T
        psi_slice = slice(L["psi_offset"], L["psi_offset"] + L["n_psi"])

        store = self._row_store(n_rows, 3, "bii", in_memory)
        scratch = 6 * block   # the draw stack and the two readout blocks
        _log(self.verbose,
             f"leave-out: B_ii diagonals, {n_draws} draws in blocks of {block}",
             self.logger, self.log_level)

        drawn = 0
        with self._quiet_solver(), self._scratch_capped(scratch):
            while drawn < n_draws:
                size = min(block, n_draws - drawn)
                vectors = self.rademacher(seed=seed, first_draw=drawn)

                # pass 1: L'q for both dimensions. Feeding q / sqrt(w) to the
                # kernels, which apply W, yields the W^(1/2) q these want.
                acc = np.zeros((n_threads, L["total_levels"], size))
                acc_x = np.zeros((n_threads, n_cov, size))
                gsum = np.zeros((L["n_groups"], size))
                for ordinal, chunk, starts, codes, w in self._chunks(L["columns"]):
                    _root, inv = self._inverse_root(w)
                    q = np.ascontiguousarray(
                        vectors(ordinal, len(w), size) * inv[:, None])
                    X = self._covariate_matrix(chunk, L["columns"], len(w))
                    _nb_reduce_rows(starts, codes, self.offs, w, X, q, acc, acc_x)
                    first = int(chunk["gcode"][0])
                    _nb_group_sums(starts, w, q,
                                   gsum[first:first + len(starts) - 1])
                levels = acc.sum(axis=0)

                # center: subtract the level's weight share of the grand total.
                # Summing the psi block over levels gives sum_i sqrt(w_i) q_i,
                # since every row sits at exactly one level of that dimension.
                grand = levels[psi_slice].sum(axis=0) / L["total"]
                rhs_levels = np.zeros((L["total_levels"], 2 * size))
                rhs_levels[psi_slice, :size] = (
                    levels[psi_slice] - np.outer(L["level_w"][psi_slice], grand)
                ) / root_total
                rhs_groups = np.zeros((L["n_groups"], 2 * size))
                rhs_groups[:, size:] = (
                    gsum - np.outer(L["group_w"], grand)) / root_total
                rhs_cov = np.zeros((n_cov, 2 * size))

                U_l, U_c, U_g = self.apply_inverse(
                    rhs_levels, rhs_cov, rhs_groups, covariates=covariates)

                # pass 2: read the two solutions back at row level and square
                for ordinal, chunk, starts, codes, w in self._chunks(L["columns"]):
                    X = self._covariate_matrix(chunk, L["columns"], len(w))
                    rows = np.empty((len(w), 2 * size))
                    first = int(chunk["gcode"][0])
                    _nb_readout(starts, codes, self.offs, X, U_l, U_c,
                                U_g[first:first + len(starts) - 1], rows)
                    root, _inv = self._inverse_root(w)
                    span = slice(ordinal, ordinal + len(w))
                    _nb_bii_moments(
                        root, np.ascontiguousarray(rows[:, :size]),
                        np.ascontiguousarray(rows[:, size:]),
                        store[0][span], store[1][span], store[2][span])
                drawn += size

        # divide in place and hand back the mapped slots themselves: a copy
        # here would be three row-sized arrays resident for the whole call
        for slot in store:
            for start in range(0, n_rows, _BLOCK_ROWS):
                stop = min(start + _BLOCK_ROWS, n_rows)
                slot[start:stop] = slot[start:stop] / n_draws
        return {name: store[k] for k, name in enumerate(COMPONENTS)}

    def _row_store(self, n_rows, count, tag, in_memory):
        """`count` row-sized running sums, in RAM or memory-mapped."""
        if in_memory:
            return [np.zeros(n_rows) for _ in range(count)]
        store = np.memmap(self._store_path(tag), dtype=np.float64,
                          mode="w+", shape=(count, n_rows))
        store[:] = 0.0
        return [store[k] for k in range(count)]

    # ---------------------------------------------------------------- Q times
    def _apply_q(self, component, U_levels, U_groups, L, out_levels, out_groups):
        """Write Q_c u into (out_levels, out_groups) for each column of u.

        `component` indexes COMPONENTS. The two variances need only
        coefficient-space arithmetic, because the psi and alpha selectors are
        weight-centered and so Q applied to u is the level's weight times its
        deviation from the weighted mean. The covariance pairs the two
        dimensions, so it needs one sweep over the rows to cross them.

        The outputs are written in place and should arrive zeroed: only the block
        the component touches is written.
        """
        psi_slice = slice(L["psi_offset"], L["psi_offset"] + L["n_psi"])
        total, norm = L["total"], L["norm"]
        level_w = L["level_w"][psi_slice]
        base_l = U_levels[psi_slice]
        psi_mean = (level_w @ base_l) / total
        alpha_mean = (L["group_w"] @ U_groups) / total

        if component == 0:
            out_levels[psi_slice] = level_w[:, None] * (base_l - psi_mean) / norm
            return
        if component == 1:
            out_groups[:] = (L["group_w"][:, None]
                             * (U_groups - alpha_mean)) / norm
            return

        n_threads = nb.get_num_threads()
        size = U_groups.shape[1]
        cross_psi = np.zeros((n_threads, L["total_levels"], size))
        cross_groups = np.zeros((L["n_groups"], size))
        U_levels = np.ascontiguousarray(U_levels)
        U_groups = np.ascontiguousarray(U_groups)
        for _ordinal, chunk, starts, codes, w in self._chunks():
            first = int(chunk["gcode"][0])
            span = slice(first, first + len(starts) - 1)
            _nb_cov_cross(starts, codes, L["psi_column"], L["psi_offset"], w,
                          U_levels, U_groups[span], cross_psi,
                          cross_groups[span])
        cross_level_sum = cross_psi.sum(axis=0)
        # centering: the mean times the level's weight total, exact because each
        # centered selector is weight-orthogonal to the constant
        out_levels[psi_slice] = (cross_level_sum[psi_slice]
                                 - level_w[:, None] * alpha_mean) / (2 * norm)
        out_groups[:] = (cross_groups
                         - L["group_w"][:, None] * psi_mean) / (2 * norm)

    # ---------------------------------------------------------------- M times
    def _m_times(self, b, covariates, psi, tag, in_memory=False):
        """u_k = M b_k for each component's row vector b_k, as a row store.

        One solve for all three columns: reduce, apply S^-, read back. This is
        the vector the rank-two term of the reference-centered kernel needs
        (see `component_variances`).
        """
        L = self._se_layout(covariates, psi)
        n_threads = nb.get_num_threads()
        n_cov = len(L["columns"])
        acc = np.zeros((n_threads, L["total_levels"], 3))
        acc_x = np.zeros((n_threads, n_cov, 3))
        gsum = np.zeros((L["n_groups"], 3))
        for ordinal, chunk, starts, codes, w in self._chunks(L["columns"]):
            _root, inv = self._inverse_root(w)
            span = slice(ordinal, ordinal + len(w))
            stack = np.ascontiguousarray(
                np.stack([np.asarray(b[k][span]) for k in range(3)], axis=1)
                * inv[:, None])
            X = self._covariate_matrix(chunk, L["columns"], len(w))
            _nb_reduce_rows(starts, codes, self.offs, w, X, stack, acc, acc_x)
            first = int(chunk["gcode"][0])
            _nb_group_sums(starts, w, stack, gsum[first:first + len(starts) - 1])
        U_l, U_c, U_g = self.apply_inverse(acc.sum(axis=0), acc_x.sum(axis=0),
                                           gsum, covariates=covariates)
        out = self._row_store(self.row_count(), 3, tag, in_memory)
        for ordinal, chunk, starts, codes, w in self._chunks(L["columns"]):
            X = self._covariate_matrix(chunk, L["columns"], len(w))
            first = int(chunk["gcode"][0])
            base = np.empty((len(w), 3))
            _nb_readout(starts, codes, self.offs, X, U_l, U_c,
                        U_g[first:first + len(starts) - 1], base)
            root, _inv = self._inverse_root(w)
            span = slice(ordinal, ordinal + len(w))
            for k in range(3):
                out[k][span] = np.asarray(b[k][span]) - root * base[:, k]
        return out

    # ---------------------------------------------------------------- C times
    def _apply_c(self, vectors, n_vectors, b, sink, covariates, psi,
                 extra_columns=(), deflate=None, per_component=False,
                 rank_two=None):
        """Apply C to a block of row-level vectors, for all three components.

        C v = B v - (1/2) b .* (M v) - (1/2) M (b .* v), so three ingredients are
        needed per component: the design solve u0 = S^- X'v, which gives both
        B v (after a second solve through Q) and M v for free; and the solve
        u' = S^- X'(b .* v), which gives M (b .* v).

        Two solves in sequence, so three passes over the rows: build the
        right-hand sides, apply each Q in coefficient space, then read all seven
        solutions back and assemble. Nothing row-sized is carried between
        passes -- `vectors` is regenerated from the row ordinal each time, the
        way the leverage pass does it.

        `sink(ordinal, chunk, cv)` receives a (rows, 3, n_vectors) array, and
        `vectors(ordinal, chunk, n_rows, n_vectors)` is asked for the input. Both
        are handed the chunk so that anything row-level they need -- the outcome,
        the weights -- can be read from it instead of from an array held across
        the whole call. `extra_columns` names row-file columns to make available
        that are not covariates of the design.

        `deflate=(lambdas, U)` applies C2 instead: B loses one rank per component,
        B2 = B - lambda_1 x1 x1' with x1 = X~ u1 the leading eigen-direction's
        row values, and `b` should then be built from diag(B2) to match. The
        rank-one term needs x1'v, a scalar per component and column, which the
        first pass accumulates while it is reading v anyway. `U` is a coefficient
        triple with one column per component.

        `per_component=True` gives each component its own input: `vectors`
        returns (rows, 3, n_vectors) and component k's C is applied to slice k.
        One call then does the work of three single-input calls whose other two
        outputs would be thrown away -- the trace needs exactly that, since each
        component's draw is scaled by its own Omega.

        `rank_two=(u, scale)` adds scale (1 u_k' + u_k 1') v to component k's
        output, `u` a row store with one row vector per component: the kernel
        of the reference-centered match-level estimator (`component_variances`).
        Like deflation it needs two scalars per component and column, u_k'v and
        1'v, which the first pass accumulates.
        """
        L = self._se_layout(covariates, psi)
        n_threads = nb.get_num_threads()
        n_cov = len(L["columns"])
        size = n_vectors
        psi_slice = slice(L["psi_offset"], L["psi_offset"] + L["n_psi"])
        # what to read per chunk: the design's covariates plus whatever the
        # caller needs at row level. Only the former enter X.
        wanted = tuple(dict.fromkeys(tuple(L["columns"]) + tuple(extra_columns)))

        # pass 3 is the widest: base (4 or 6 blocks), through_q (3), cv (3),
        # the input (1 or 3) and a few per-component temporaries
        n_in = 3 if per_component else 1
        with self._scratch_capped((n_in + 3 + 3 + 3 + n_in + 3) * size):
            return self._apply_c_passes(vectors, n_vectors, b, sink, covariates,
                                        psi, wanted, L, n_threads, n_cov, size,
                                        psi_slice, deflate, per_component,
                                        rank_two)

    def _eigen_rows(self, U, chunk, starts, codes, w, columns):
        """sqrt(w_i) x_i' u for each column of the coefficient triple U."""
        X = self._covariate_matrix(chunk, columns, len(w))
        first = int(chunk["gcode"][0])
        rows = np.empty((len(w), U[2].shape[1]))
        _nb_readout(starts, codes, self.offs, X, np.ascontiguousarray(U[0]),
                    np.ascontiguousarray(U[1]),
                    np.ascontiguousarray(U[2][first:first + len(starts) - 1]),
                    rows)
        return rows * np.sqrt(w)[:, None]

    def _apply_c_passes(self, vectors, n_vectors, b, sink, covariates, psi,
                        wanted, L, n_threads, n_cov, size, psi_slice,
                        deflate=None, per_component=False, rank_two=None):
        # Column layout of the first solve: the input block(s) -- one shared by
        # all three components, or one each -- then b_k .* v_k for each k.
        n_in = 3 if per_component else 1
        wide = n_in + 3

        def in_cols(k):
            j = k if per_component else 0
            return slice(j * size, (j + 1) * size)

        def b_cols(k):
            return slice((n_in + k) * size, (n_in + k + 1) * size)

        def part(v, k):
            return v[:, k, :] if per_component else v

        # pass 1: X'v and X'(b .* v) in one sweep, with deflation x1'v, and
        # for the rank-two term u_k'v and 1'v
        along = np.zeros((3, size))
        u_dot, one_dot = np.zeros((3, size)), np.zeros((3, size))
        acc = np.zeros((n_threads, L["total_levels"], wide * size))
        acc_x = np.zeros((n_threads, n_cov, wide * size))
        gsum = np.zeros((L["n_groups"], wide * size))
        for ordinal, chunk, starts, codes, w in self._chunks(wanted):
            _root, inv = self._inverse_root(w)
            v = vectors(ordinal, chunk, len(w), size)
            span = slice(ordinal, ordinal + len(w))
            stack = np.empty((len(w), wide * size))
            for k in range(n_in):
                stack[:, in_cols(k)] = part(v, k)
            for k in range(3):
                stack[:, b_cols(k)] = part(v, k) * b[k][span, None]
            stack = np.ascontiguousarray(stack * inv[:, None])
            X = self._covariate_matrix(chunk, L["columns"], len(w))
            if deflate is not None:
                x1 = self._eigen_rows(deflate[1], chunk, starts, codes, w,
                                      L["columns"])
                for k in range(3):
                    along[k] += x1[:, k] @ part(v, k)
            if rank_two is not None:
                for k in range(3):
                    vk = part(v, k)
                    u_dot[k] += np.asarray(rank_two[0][k][span]) @ vk
                    one_dot[k] += vk.sum(axis=0)
            _nb_reduce_rows(starts, codes, self.offs, w, X, stack, acc, acc_x)
            first = int(chunk["gcode"][0])
            _nb_group_sums(starts, w, stack,
                           gsum[first:first + len(starts) - 1])
        levels, cov_block = acc.sum(axis=0), acc_x.sum(axis=0)

        U0_l, U0_c, U0_g = self.apply_inverse(
            levels, cov_block, gsum, covariates=covariates)

        # pass 2: Q_k u0_k for each component
        q_levels = np.zeros((L["total_levels"], 3 * size))
        q_groups = np.zeros((L["n_groups"], 3 * size))
        for k in range(3):
            cols = slice(k * size, (k + 1) * size)
            self._apply_q(k, np.ascontiguousarray(U0_l[:, in_cols(k)]),
                          np.ascontiguousarray(U0_g[:, in_cols(k)]), L,
                          q_levels[:, cols], q_groups[:, cols])

        U1_l, U1_c, U1_g = self.apply_inverse(
            q_levels, np.zeros((n_cov, 3 * size)), q_groups,
            covariates=covariates)

        # pass 3: read everything back and assemble
        for ordinal, chunk, starts, codes, w in self._chunks(wanted):
            X = self._covariate_matrix(chunk, L["columns"], len(w))
            first = int(chunk["gcode"][0])
            n_chunk = len(starts) - 1
            base = np.empty((len(w), wide * size))
            _nb_readout(starts, codes, self.offs, X, U0_l, U0_c,
                        U0_g[first:first + n_chunk], base)
            through_q = np.empty((len(w), 3 * size))
            _nb_readout(starts, codes, self.offs, X, U1_l, U1_c,
                        U1_g[first:first + n_chunk], through_q)

            root, _inv = self._inverse_root(w)
            v = vectors(ordinal, chunk, len(w), size)
            span = slice(ordinal, ordinal + len(w))
            cv = np.empty((len(w), 3, size))
            x1 = (self._eigen_rows(deflate[1], chunk, starts, codes, w,
                                   L["columns"]) if deflate is not None else None)
            for k in range(3):
                vk = part(v, k)
                Mv = vk - root[:, None] * base[:, in_cols(k)]
                Bv = root[:, None] * through_q[:, k * size:(k + 1) * size]
                if deflate is not None:
                    Bv = Bv - deflate[0][k] * np.outer(x1[:, k], along[k])
                M_bv = vk * b[k][span, None] - root[:, None] * base[:, b_cols(k)]
                cv[:, k, :] = Bv - 0.5 * (b[k][span, None] * Mv + M_bv)
                if rank_two is not None:
                    u = np.asarray(rank_two[0][k][span])
                    cv[:, k, :] += rank_two[1] * (u_dot[k][None, :]
                                                  + np.outer(u, one_dot[k]))
            sink(ordinal, chunk, cv)

    # ------------------------------------------------------------- assembling
    def component_variances(self, result, sigma2, leverages, b_ii=None,
                            n_draws=250, block=None, seed=0, covariates=None,
                            psi=None, trace=True, smooth_bins=None,
                            in_memory=False, weak_id=None, weak_components=None,
                            stayer_rows=None, centering="weighted",
                            reported=(0, 1, 2), sigma2_point=None):
        """V-hat and the standard errors, given the pieces the point estimate made.

        `sigma2` and `leverages` come from `leave_out_components`; `b_ii` is the
        output of `bii_rows`, computed here if not supplied.

        `trace=False` skips the Hutchinson term and returns the conservative
        first-term-only standard error, which costs three passes instead of
        three per block of draws. Measured on simulated panels the trace term is
        one to nine percent of the first term, so skipping it moves the standard
        error by under five percent and coverage by well under a point.

        Leaving out a match, the rows are matches and four things change
        (`leave_out_match`, and docs/kss_methodological_differences.md 5.7):

          * `stayer_rows` marks the matches of workers seen at one firm. Their
            leverage is one, and for var(psi) and cov their B_ii is exactly
            zero -- a stayer's outcome does not move psi-hat -- so b is set to
            zero there, as the reference implementation, LeaveOutTwoWay's
            leave_out_COMPLETE with matches (beta there), does. That keeps
            the kernel's diagonal zero.
          * `centering="reference"` describes the estimator whose sigma2 is
            centered at the mean over matches of sqrt(w) ybar. It is exactly
            y_r' K y_r with y_r = sqrt(w) y uncentered and
            K = C + (1 u' + u 1') / (2 R),  u = M b,  R the number of rows:
            a rank-two update of C, applied alongside it.
          * `reported` lists the components with a standard error; the others
            are NaN. leave_out_COMPLETE reports none for var(alpha) at match
            level, which leans on stayers whose variance is not a leave-out
            estimate.
          * `sigma2_point` is the point estimate's own sigma2, when `sigma2`
            (what is smoothed for the variance) differs from it. theta1 of the
            q = 1 interval is formed from the reported estimate, so it needs
            the sigma2 that estimate used.
        """
        covariates = self._model_covariates(covariates)
        block = self.rhs_block if block is None else int(block)
        L = self._se_layout(covariates, psi)
        if b_ii is None:
            b_ii = self.bii_rows(n_draws=n_draws, block=block, seed=seed,
                                 covariates=covariates, psi=L["psi"],
                                 in_memory=in_memory)

        n_rows = self.row_count()
        if centering not in ("reference", "weighted"):
            raise ValueError("centering must be 'reference' or 'weighted'")
        # b_i = B_ii / M_ii, with the leverage pass's own non-linearity
        # correction. Written to its own store rather than over the B_ii slots:
        # the smoother bins on (P_ii, B_ii), as the reference does, and it
        # evaluates those bins per chunk, so B_ii has to survive intact. Both
        # stores are memory-mapped, so this is disk rather than resident memory,
        # and `_apply_c` reads b a chunk at a time.
        b = self._row_store(n_rows, 3, "bpar", in_memory)
        for k, name in enumerate(COMPONENTS):
            for start in range(0, n_rows, _BLOCK_ROWS):
                stop = min(start + _BLOCK_ROWS, n_rows)
                b[k][start:stop] = (b_ii[name][start:stop]
                                    * leverages.factor[start:stop]
                                    / leverages.complement[start:stop])
                if stayer_rows is not None:
                    b[k][start:stop][np.asarray(stayer_rows[start:stop])] = 0.0

        # the rank-two term of the reference-centered kernel
        rank_two = None
        if centering == "reference":
            rank_two = (self._m_times(b, covariates, L["psi"], "mb", in_memory),
                        0.5 / n_rows)

        # sigma2-tilde: smoothed against (P_ii, B_ii). The raw leave-out sigma2
        # is a product of two noisy things and is often negative, which makes the
        # trace term come out negative -- impossible for sum C_il^2 s_i s_l -- and
        # the whole variance estimate incoherent. Smoothing is not a refinement
        # here, it is what makes the estimator well behaved.
        #
        # Only the fitted table is kept; the fitted values are evaluated per
        # chunk. Materializing them would cost three row-sized arrays, and three
        # more for their square roots in the trace draws.
        tables = [self._fit_smoother(sigma2, leverages.leverage, b_ii[name],
                                     smooth_bins) for name in COMPONENTS]

        def sigma_at(k, span):
            return self._smooth_at(tables[k], sigma2, leverages.leverage,
                                   b_ii[COMPONENTS[k]], span)

        # The outcome is already a column of the row files, so it is read per
        # chunk rather than held: one more row-sized array avoided.
        depcol = f"v{self.vidx[result.depvar]}"
        y_bar = self._outcome_mean(depcol)

        first = np.zeros(3)
        theta = np.zeros(3)

        # the outcome the kernel is applied to: centered at the weighted mean
        # for C, uncentered for the reference-centered K (its centering is in
        # the rank-two term)
        shift = y_bar if centering == "weighted" else 0.0

        def shifted_outcome(chunk, n):
            """sqrt(w)(y - shift) for one chunk, built where it is consumed."""
            w = self._chunk_w(chunk, n)
            return np.sqrt(w) * (np.asarray(chunk[depcol], dtype=np.float64)
                                 - shift)

        def y_tilde(_ordinal, chunk, n, _m):
            return shifted_outcome(chunk, n)[:, None]

        def collect_y(ordinal, chunk, cv):
            span = slice(ordinal, ordinal + cv.shape[0])
            shifted = shifted_outcome(chunk, cv.shape[0])
            for k in range(3):
                column = cv[:, k, 0]
                s = self._smooth_at(tables[k], sigma2, leverages.leverage,
                                    b_ii[COMPONENTS[k]], span)
                first[k] += 4.0 * float(s @ (column ** 2))
                theta[k] += float(shifted @ column)

        _log(self.verbose, "leave-out: C y, exact", self.logger, self.log_level)
        with self._quiet_solver():
            self._apply_c(y_tilde, 1, b, collect_y, covariates, L["psi"],
                          extra_columns=(depcol,), rank_two=rank_two)

        trace_total = np.zeros(3)
        blocks = 0
        if trace:
            _log(self.verbose,
                 f"leave-out: trace term, {n_draws} draws in blocks of {block}",
                 self.logger, self.log_level)
            trace_total, blocks = self._cc_trace(
                b, sigma_at, n_draws, block, seed + 7_700_017, covariates,
                L["psi"], rank_two=rank_two)

        # the q = 1 pieces, for the components the diagnostic says need them
        weak = None
        if weak_id is not None:
            if weak_components is None:
                weak_components = [k for k, n in enumerate(COMPONENTS)
                                   if weak_id.q[n] >= 1]
            # at match level the deflated kernel keeps a zero diagonal only
            # where stayers do not load on the weak direction: var(psi)
            weak_components = [k for k in weak_components
                               if k in reported and (stayer_rows is None
                                                     or k == 0)]
            if weak_components:
                _log(self.verbose,
                     "leave-out: weak-identification interval for "
                     + ", ".join(COMPONENTS[k] for k in weak_components),
                     self.logger, self.log_level)
                weak = self._weak_moments(
                    result, sigma2 if sigma2_point is None else sigma2_point,
                    leverages, b_ii, sigma_at, weak_id,
                    weak_components, n_draws, block, seed + 8_800_021,
                    covariates, psi, in_memory, stayer_rows=stayer_rows,
                    centering=centering)

        variance = first - 2.0 * trace_total
        names = list(COMPONENTS)
        for k in range(3):
            if k not in reported:
                variance[k] = first[k] = trace_total[k] = theta[k] = np.nan
        return ComponentSE(
            weak=weak,
            se={n: float(np.sqrt(v)) if v > 0 else float("nan")
                for n, v in zip(names, variance)},
            variance=dict(zip(names, variance.tolist())),
            first_term=dict(zip(names, first.tolist())),
            trace_term=dict(zip(names, (2.0 * trace_total).tolist())),
            se_conservative={n: float(np.sqrt(f)) if f > 0 else float("nan")
                             for n, f in zip(names, first)},
            theta=dict(zip(names, theta.tolist())),
            n_draws=n_draws if trace else 0,
            diagnostics={"psi": L["psi"], "alpha": self.g_fe,
                         "covariates": list(covariates), "blocks": blocks,
                         "trace_included": bool(trace),
                         "smooth_bins": smooth_bins, "centering": centering,
                         "reported": [names[k] for k in reported],
                         "max_b_ii": {n: float(np.max(b_ii[n])) for n in names},
                         "stayer_rows": (0 if stayer_rows is None
                                         else int(np.sum(stayer_rows))),
                         "negative_trace": {n: bool(t < 0) for n, t
                                            in zip(names, trace_total)}},
        )

    def _weak_moments(self, result, sigma2, leverages, b_ii, sigma_at,
                      weak_id, components, n_draws, block, seed, covariates,
                      psi, in_memory, stayer_rows=None, centering="weighted"):
        """(b1, V[b1] raw, 2x2 covariance of (b1, theta1)) per component.

        The q = 1 interval of KSS Section 6 needs the leading direction's
        estimate b1 = sum_i x1_i y~_i (x1 = X~ u1, u1' S u1 = 1) and the joint
        covariance of (b1, theta1). Deflating B by that direction,
        B2 = B - lambda_1 x1 x1', and building C2 from B2 as C is built from B,
        makes theta1 = y~' C2 y~ exactly -- again a zero-diagonal quadratic
        form -- so, with no appeal to normality,

            V[b1]          = sum x1^2 s2
            Cov(b1, th1)   = 2 sum x1 s2 (C2 mu)
            V[th1]         = 4 sum s2 (C2 mu)^2 + 2 tr(C2 Om C2 Om)

        each estimated as for the standard errors, with sigma2 smoothed and
        floored at zero. The floor matters here: with s >= 0, Cauchy-Schwarz
        gives Cov^2 <= V[b1] times the first term of V[th1], so if subtracting
        the trace leaves the matrix indefinite, falling back to the first term
        alone -- upward biased, KSS Remark 9 -- is guaranteed to repair it.

        `stayer_rows` and `centering` are as in `component_variances`: b2 is zero
        on stayers' matches, and the reference centering adds the rank-two term
        built from M b2. b1 needs no change, since x1 is orthogonal to sqrt(w).
        """
        n_rows = self.row_count()
        L = self._se_layout(covariates, psi)
        lams = np.array([float(weak_id.eigenvalues[n][0])
                         if weak_id.eigenvalues[n] else 0.0 for n in COMPONENTS])
        U = weak_id.lead_vectors
        depcol = f"v{self.vidx[result.depvar]}"
        y_bar = self._outcome_mean(depcol)
        wanted = tuple(dict.fromkeys(tuple(L["columns"]) + (depcol,)))

        def floor_at(k, span):
            return np.maximum(sigma_at(k, span), 0.0)

        # pass: b1, the raw V[b1] theta1 is formed with, V[b1] itself, and b2
        b1, v_raw, s11 = np.zeros(3), np.zeros(3), np.zeros(3)
        b2 = self._row_store(n_rows, 3, "bpar_deflated", in_memory)
        for ordinal, chunk, starts, codes, w in self._chunks(wanted):
            span = slice(ordinal, ordinal + len(w))
            x1 = self._eigen_rows(U, chunk, starts, codes, w, L["columns"])
            yt = np.sqrt(w) * (np.asarray(chunk[depcol], dtype=np.float64) - y_bar)
            b1 += x1.T @ yt
            v_raw += (x1 ** 2).T @ np.asarray(sigma2[span], dtype=np.float64)
            inverse_m = (np.asarray(leverages.factor[span])
                         / np.asarray(leverages.complement[span]))
            for k in range(3):
                s11[k] += (x1[:, k] ** 2) @ floor_at(k, span)
                b2[k][span] = ((np.asarray(b_ii[COMPONENTS[k]][span])
                                - lams[k] * x1[:, k] ** 2) * inverse_m)
                if stayer_rows is not None:
                    b2[k][span][np.asarray(stayer_rows[span])] = 0.0
        rank_two = None
        if centering == "reference":
            rank_two = (self._m_times(b2, covariates, L["psi"], "mb2",
                                      in_memory), 0.5 / n_rows)
        shift = y_bar if centering == "weighted" else 0.0

        # C2 y~, exactly
        s12, first, quad = np.zeros(3), np.zeros(3), np.zeros(3)

        def y_tilde(_ordinal, chunk, n, _m):
            w = self._chunk_w(chunk, n)
            return (np.sqrt(w) * (np.asarray(chunk[depcol], dtype=np.float64)
                                  - shift))[:, None]

        def collect(ordinal, chunk, cv):
            span = slice(ordinal, ordinal + cv.shape[0])
            starts, codes, w = self._chunk_layout(chunk)
            x1 = self._eigen_rows(U, chunk, starts, codes, w, L["columns"])
            yt = y_tilde(ordinal, chunk, cv.shape[0], 1)[:, 0]
            for k in range(3):
                s = floor_at(k, span)
                column = cv[:, k, 0]
                s12[k] += 2.0 * float((x1[:, k] * s) @ column)
                first[k] += 4.0 * float(s @ (column ** 2))
                quad[k] += float(yt @ column)

        deflate = (lams, U)
        with self._quiet_solver():
            self._apply_c(y_tilde, 1, b2, collect, covariates, L["psi"],
                          extra_columns=(depcol,), deflate=deflate,
                          rank_two=rank_two)
        trace, _blocks = self._cc_trace(
            b2, sigma_at, n_draws, block, seed, covariates, L["psi"],
            deflate=deflate, clamp=True, components=tuple(components),
            rank_two=rank_two)

        out = {}
        for k in components:
            s22 = first[k] - 2.0 * trace[k]
            conservative = not (s22 > 0 and s12[k] ** 2 < s11[k] * s22)
            if conservative:
                s22 = first[k]
            out[COMPONENTS[k]] = {
                "lambda_1": float(lams[k]), "b1": float(b1[k]),
                "v_raw": float(v_raw[k]),
                "sigma": np.array([[s11[k], s12[k]], [s12[k], s22]]),
                "conservative": bool(conservative),
                "trace": float(trace[k]), "quadratic": float(quad[k])}
        return out

    def _cc_trace(self, b, sigma_at, n_draws, block, seed, covariates, psi,
                  deflate=None, clamp=False, components=(0, 1, 2),
                  rank_two=None):
        """tr(C Om C Om) per component by random projection.

        E[sum_i s_i (C v)_i^2] = tr(C Om C Om) for v = Om^(1/2) z with
        E[zz'] = I. One application of C per draw, and no normality needed --
        Saggio's reference instead takes the variance of v'Cv across Gaussian
        draws, the same target by way of a factor of two for the same solve.

        Each component carries its own Omega, so the draw is scaled per
        component rather than once for all three. `sigma_at(k, span)` gives the
        smoothed sigma2 of component k; `deflate` is passed to `_apply_c` for
        the deflated kernel C2; `clamp` floors sigma2 at zero in the weighting
        as well as in the draw. All three components share one application of
        C per block, each with its own input; components not listed get a zero
        input and are not collected. Returns (traces, blocks).
        """
        total = np.zeros(3)
        drawn, blocks = 0, 0
        with self._quiet_solver():
            while drawn < n_draws:
                size = min(block, n_draws - drawn)
                draws = self.rademacher(seed=seed, first_draw=drawn)

                def scaled(ordinal, _chunk, n, m, draws=draws):
                    z = draws(ordinal, n, m)
                    out = np.zeros((n, 3, m))
                    for k in components:
                        s = sigma_at(k, slice(ordinal, ordinal + n))
                        out[:, k, :] = z * np.sqrt(np.maximum(s, 0.0))[:, None]
                    return out

                def collect(ordinal, _chunk, cv):
                    span = slice(ordinal, ordinal + cv.shape[0])
                    for k in components:
                        s = sigma_at(k, span)
                        if clamp:
                            s = np.maximum(s, 0.0)
                        total[k] += float(s @ (cv[:, k, :] ** 2).sum(1))

                self._apply_c(scaled, size, b, collect, covariates, psi,
                              deflate=deflate, per_component=True,
                              rank_two=rank_two)
                drawn += size
                blocks += 1
        return total / n_draws, blocks

    def _fit_smoother(self, sigma2, leverage, b_ii, n_bins):
        """The smoother of sigma2 on (P_ii, B_ii); see `fit_smoother`."""
        return fit_smoother(sigma2, leverage, b_ii, n_bins)

    def _smooth_at(self, table, sigma2, leverage, b_ii, span):
        """The smoothed sigma2 for one stretch of rows."""
        if table is None:
            return np.asarray(sigma2[span], dtype=np.float64)
        return smooth_eval(table, leverage[span],
                           None if b_ii is None else b_ii[span])


# --------------------------------------------------------------------------
# the smoother of sigma2
# --------------------------------------------------------------------------

def _auto_bins(n, dims):
    """Bins per axis matching the reference's lowess bandwidth.

    Saggio's llr_fit uses span n^(-1/3): each local fit sees n^(2/3) points. A
    grid whose cells hold that many has n^(1/6) bins per axis in two dimensions
    and n^(1/3) in one. A fixed count is wrong at both ends: twelve per axis is
    11 rows a cell on a 1,600-row panel.
    """
    power = 1.0 / 6.0 if dims == 2 else 1.0 / 3.0
    return int(np.clip(round(n ** power), 2, 500))


def _knots(values, n_bins, take):
    """Quantile knots, strictly increasing, for a continuous rank transform."""
    q = np.quantile(np.asarray(values[take], dtype=np.float64),
                    np.linspace(0.0, 1.0, n_bins + 1))
    q = np.maximum.accumulate(q)
    span = max(float(q[-1] - q[0]), 1e-300)
    return q + np.arange(len(q)) * span * 1e-12


def _rank(knots, values):
    """Continuous, piecewise-linear map to [0, 1] through the quantile knots."""
    return np.interp(np.asarray(values, dtype=np.float64), knots,
                     np.linspace(0.0, 1.0, len(knots)))


def fit_smoother(sigma2, leverage, b_ii=None, n_bins=None):
    """Fit sigma2 against (P_ii, B_ii); return a small table to evaluate from.

    Saggio's reference runs a lowess fit of the leave-out sigma2 on those two
    regressors and uses the fitted values. This is the same idea in a form that
    streams: each coordinate is mapped to its empirical quantile, the rows are
    averaged in a grid of cells there, and `smooth_eval` interpolates the cell
    means bilinearly between cell centers.

    The interpolation matters. A plain lookup of the cell mean is a step
    function, and the quantities that rest on a few rows -- V[b1] in the weak
    identification interval, above all -- then jump whenever random-projection
    noise in P_ii or B_ii moves one of those rows across an edge: on a
    1,600-row panel V[b1] ranged from 0.0195 to 0.0344 across five seeds of the
    same fit, around a truth of 0.0225. Lowess is continuous in its regressors,
    and so is this.

    `n_bins=None` matches the reference's bandwidth (see `_auto_bins`).
    `b_ii=None` smooths on the leverage alone. The knots come from a bounded
    random subsample, since an exact quantile sorts a copy of the column, and
    the cell sums from a single blocked pass, so nothing row-sized is held.
    The table is (knots, knots, means, n_bins) -- a few hundred floats.
    """
    n_rows = len(sigma2)
    dims = 1 if b_ii is None else 2
    n_bins = _auto_bins(n_rows, dims) if n_bins is None else int(n_bins)
    if n_bins < 1:
        return None
    rng = np.random.default_rng(0)
    take = (np.arange(n_rows) if n_rows <= _QUANTILE_SAMPLE
            else np.sort(rng.choice(n_rows, _QUANTILE_SAMPLE, replace=False)))
    knots = (_knots(leverage, n_bins, take),
             None if b_ii is None else _knots(b_ii, n_bins, take))

    shape = (n_bins,) * dims
    totals = np.zeros(shape)
    counts = np.zeros(shape)
    for start in range(0, n_rows, _BLOCK_ROWS):
        stop = min(start + _BLOCK_ROWS, n_rows)
        cells = [np.minimum((_rank(knots[0], leverage[start:stop])
                             * n_bins).astype(np.int64), n_bins - 1)]
        if b_ii is not None:
            cells.append(np.minimum((_rank(knots[1], b_ii[start:stop])
                                     * n_bins).astype(np.int64), n_bins - 1))
        flat = np.ravel_multi_index(cells, shape)
        s = np.asarray(sigma2[start:stop], dtype=np.float64)
        totals += np.bincount(flat, weights=s, minlength=totals.size).reshape(shape)
        counts += np.bincount(flat, minlength=totals.size).reshape(shape)
    # an empty cell -- the two regressors are correlated, so corners of the
    # grid can be -- takes the overall mean rather than dragging the
    # interpolation toward zero
    overall = float(totals.sum() / max(counts.sum(), 1.0))
    means = np.where(counts > 0, totals / np.maximum(counts, 1.0), overall)
    return knots, means, n_bins


def smooth_eval(table, leverage, b_ii=None):
    """Evaluate a fitted smoother at the given rows, bilinearly."""
    knots, means, n_bins = table

    def weights(values, k):
        f = _rank(knots[k], values) * n_bins - 0.5
        j0 = np.clip(np.floor(f).astype(np.int64), 0, n_bins - 2)
        return j0, np.clip(f - j0, 0.0, 1.0)

    i0, a = weights(leverage, 0)
    if knots[1] is None:
        return (1.0 - a) * means[i0] + a * means[i0 + 1]
    j0, b = weights(b_ii, 1)
    return ((1.0 - a) * (1.0 - b) * means[i0, j0]
            + a * (1.0 - b) * means[i0 + 1, j0]
            + (1.0 - a) * b * means[i0, j0 + 1]
            + a * b * means[i0 + 1, j0 + 1])

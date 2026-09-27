"""Leave-out estimation: Johnson-Lindenstrauss leverages, out of core.

Statistical leverages are what leave-out estimation is built on. The exact
`P_ii = x_i' S^- x_i` needs one solve per observation, which is hopeless at any
interesting scale, so Kline, Saggio and Solvsten approximate them by random
projection (their JLA). This module is that approximation, running against the
streaming operator in `inverse.py`.

The whole algorithm is five running sums per row. For Rademacher draws q_s, with
Pq_s the projection of q_s onto the design:

    Phat_ii     = mean_s (Pq_s)_i^2                 unbiased for P_ii
    Mhat_ii     = mean_s (q_si - (Pq_s)_i)^2        unbiased for 1 - P_ii
    m(P^2)      = mean_s (Pq_s)_i^4
    m(M^2)      = mean_s (q_si - (Pq_s)_i)^4
    m(P, M)     = mean_s (Pq_s)_i^2 (q_si - (Pq_s)_i)^2

Both of the first two come out of a single solve per draw -- Mhat is free once
Phat is computed -- and the other three are the same numbers raised to a higher
power, so the marginal cost of the non-linearity correction is nil.

Two refinements from the authors' 2021 follow-up note (`improved_JLA.pdf` in
Saggio's LeaveOutTwoWay), both implemented here:

1. P_ii + M_ii = 1 identically, so the estimates are normalized to satisfy it:
   Pbar = Phat/(Phat + Mhat). This guarantees an estimate inside [0, 1] and
   costs nothing.

2. sigma^2 divides by M_ii, so substituting an estimate is non-linear and biased
   even though the estimate is not. `factor` below removes that bias to order
   1/p. See prototypes/JLA_NOTES.md: the note's version is the one implemented,
   which differs from pytwoway 0.3.21 in a sign and a coefficient -- raced
   against the exact answer there, and the note wins.

Accuracy is set by the number of draws and does not need to grow with the data;
KSS report that the error *falls* with sample size at fixed p, and use p in the
250-500 range for panels with millions of effects.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np

from .report import _log

from .results import HDFEResult

# Number of per-row running sums the algorithm keeps.
_N_MOMENTS = 5
# rows per block when reducing the accumulators in place; bounds the working set
_COMBINE_BLOCK = 1_000_000


@dataclass
class JLALeverages:
    """Per-row leverage estimates, aligned with the residual file's row order."""

    leverage: np.ndarray          # Pbar_ii, constrained to [0, 1]
    complement: np.ndarray        # Mbar_ii = 1 - leverage
    factor: np.ndarray            # non-linearity correction for sigma^2
    n_draws: int
    diagnostics: dict = field(default_factory=dict)

    @property
    def n_rows(self):
        return len(self.leverage)

    def summary(self):
        d = self.diagnostics
        return (f"JLA leverages: {self.n_rows:,} rows, {self.n_draws} draws in "
                f"{d.get('blocks', '?')} block(s)\n"
                f"  leverage   min {self.leverage.min():.6f}  "
                f"max {self.leverage.max():.6f}  mean {self.leverage.mean():.6f}\n"
                f"  correction min {self.factor.min():.6f}  "
                f"max {self.factor.max():.6f}  mean {self.factor.mean():.6f}")


# Shared-state contract with the other mixins
# -------------------------------------------
# Uses only the public surface of _InverseMixin (project_rows, rademacher,
# row_count) plus the run directory for scratch. Sets nothing others read.


class _LeaveOutMixin:

    @contextmanager
    def _quiet_solver(self):
        """Silence the per-iteration solver chatter for the duration.

        The leave-out phases run hundreds of solves. Left alone they would bury
        anything useful under a CG line per block, so each phase logs its own
        progress instead.
        """
        was = self.verbose
        self.verbose = False
        try:
            yield
        finally:
            self.verbose = was

    def _model_covariates(self, covariates):
        """Which variables belong in the design whose leverages we want.

        Defaulting to the fitted model's own covariates matters: leverages from
        the fixed effects alone are leverages of a *different design*, and using
        them for a model with controls would be quietly wrong rather than
        obviously wrong. Pass `covariates=()` to ask for the fixed-effect design
        deliberately.
        """
        if covariates is not None:
            return tuple(covariates)
        distinct = {tuple(model["x"]) for model in self.models}
        if len(distinct) > 1:
            raise ValueError(
                "this fit covers several models with different covariates "
                f"({sorted(distinct)}), so there is no single design to take "
                "leverages of; pass covariates= explicitly")
        return next(iter(distinct))

    def jla_leverages(self, n_draws=250, block=None, seed=0, in_memory=False,
                      covariates=None):
        """Estimate every row's leverage by random projection.

        n_draws    total Rademacher draws. Accuracy goes as 1/sqrt(n_draws) and
                   does not need to scale with the data; 250-500 is the range
                   KSS use on panels with millions of effects.
        block      draws per pass over the rows. Bounds memory: the level-sized
                   accumulator is (threads x levels x block). Defaults to
                   `rhs_block`, the same knob the solver uses.
        seed       choice of draw sequence; results are reproducible given it.
        in_memory  keep the five running sums in RAM rather than memory-mapping
                   them into the run directory. They are row-sized -- five
                   float64 per row -- so at tens of millions of rows this is the
                   difference between a few hundred MB of RAM and none.
        covariates which variables join the fixed effects in the design.
                   Defaults to the covariates of the fitted model, which is
                   almost always what is wanted; `()` asks for the fixed-effect
                   design alone.

        Returns a `JLALeverages`, whose arrays are in the same row order as the
        residual file written by the fit, so they can be used side by side.

        With weights these are the leverages of the square-root-weight hat
        matrix, P_ii = w_i x_i' (A'WA)^- x_i. Of the two hat matrices a weighted
        fit admits, only that one is symmetric and idempotent, and the random
        projection estimates the diagonal of the operator's square -- so only
        that one gives leverages at all. Unweighted the two coincide.
        """
        # not `block or self.rhs_block`: that would silently turn 0 into the
        # default rather than rejecting it
        block = self.rhs_block if block is None else int(block)
        if block < 1 or n_draws < 1:
            raise ValueError("n_draws and block must be at least 1")
        covariates = self._model_covariates(covariates)
        n_rows = self.row_count()
        moments = self._moment_store(n_rows, in_memory)
        _log(self.verbose,
             f"leave-out: leverages, {n_draws} draws in blocks of {block}"
             + (f", design includes {len(covariates)} covariate(s)"
                if covariates else ""),
             self.logger, self.log_level)

        blocks, drawn, info = 0, 0, {}
        # R, out, P, M and the squares and products the sink forms, all per chunk
        with self._quiet_solver(), self._scratch_capped(12 * block):
            while drawn < n_draws:
                size = min(block, n_draws - drawn)
                # a distinct stretch of one sequence per block, never a repeat
                vectors = self.rademacher(seed=seed, first_draw=drawn)

                def sink(ordinal, chunk, projected, size=size, vectors=vectors):
                    r = vectors(ordinal, len(projected), size)
                    P = projected ** 2
                    M = (r - projected) ** 2
                    rows = slice(ordinal, ordinal + len(projected))
                    moments[0][rows] += P.sum(axis=1)
                    moments[1][rows] += M.sum(axis=1)
                    moments[2][rows] += (P ** 2).sum(axis=1)
                    moments[3][rows] += (M ** 2).sum(axis=1)
                    moments[4][rows] += (P * M).sum(axis=1)

                info = self.project_rows(vectors, size, sink,
                                         covariates=covariates,
                                         sqrt_weights=self.weights is not None)
                drawn += size
                blocks += 1

        return self._combine_moments(moments, n_draws, blocks, info, covariates)

    def _moment_store(self, n_rows, in_memory):
        """The five row-sized running sums, in RAM or memory-mapped."""
        if in_memory:
            return [np.zeros(n_rows) for _ in range(_N_MOMENTS)]
        store = np.memmap(self._store_path("jla_moments"), dtype=np.float64,
                          mode="w+", shape=(_N_MOMENTS, n_rows))
        store[:] = 0.0
        return [store[k] for k in range(_N_MOMENTS)]

    @staticmethod
    def _combine_moments(moments, n_draws, blocks, info, covariates=()):
        """Turn the running sums into leverages and the correction factor.

        Everything here is the 2021 note's algebra, applied row-wise -- but done
        *in place*, in bounded row blocks, writing the three results back over
        three of the five accumulator slots. Those slots are memory-mapped unless
        the caller asked otherwise, so what comes back costs no resident memory
        and the working set is one block rather than one array per intermediate.
        The naive version allocated about fifteen row-sized temporaries to
        produce three, which at a hundred million rows is the difference between
        a few hundred MB and twelve GB.
        """
        P_slot, M_slot, F_slot, mMM_slot, mPM_slot = moments
        n_rows = len(P_slot)
        worst_total, lo, hi, running = 0.0, np.inf, -np.inf, 0.0

        for start in range(0, n_rows, _COMBINE_BLOCK):
            stop = min(start + _COMBINE_BLOCK, n_rows)
            P = np.asarray(P_slot[start:stop], dtype=np.float64) / n_draws
            M = np.asarray(M_slot[start:stop], dtype=np.float64) / n_draws
            m_PP = np.asarray(F_slot[start:stop], dtype=np.float64) / n_draws
            m_MM = np.asarray(mMM_slot[start:stop], dtype=np.float64) / n_draws
            m_PM = np.asarray(mPM_slot[start:stop], dtype=np.float64) / n_draws

            # impose P_ii + M_ii = 1, which the raw estimates satisfy only in
            # expectation. The total is positive by construction (both parts are
            # sums of squares) except for a row with no weight at all.
            total = P + M
            safe = total > 0
            P_bar = np.divide(P, total, out=np.zeros_like(P), where=safe)
            M_bar = 1.0 - P_bar

            # non-linearity correction for estimating 1/M_ii
            V = (M_bar ** 2 * m_PP + P_bar ** 2 * m_MM
                 - 2 * P_bar * M_bar * m_PM) / n_draws
            B = (M_bar * m_PP - P_bar * m_MM
                 + (M_bar - P_bar) * m_PM) / n_draws
            usable = M_bar > 0
            factor = np.ones_like(M_bar)
            np.divide(V, M_bar ** 2, out=factor, where=usable)
            factor = np.where(usable,
                              1.0 - factor + np.divide(B, M_bar,
                                                       out=np.zeros_like(B),
                                                       where=usable),
                              1.0)

            worst_total = max(worst_total, float(np.abs(total - 1.0).max())
                              if stop > start else 0.0)
            if stop > start:
                lo = min(lo, float(P_bar.min()))
                hi = max(hi, float(P_bar.max()))
                running += float(P_bar.sum())

            P_slot[start:stop] = P_bar
            M_slot[start:stop] = M_bar
            F_slot[start:stop] = factor

        empty = n_rows == 0
        mean = running / n_rows if n_rows else float("nan")
        return JLALeverages(
            leverage=P_slot, complement=M_slot, factor=F_slot, n_draws=n_draws,
            diagnostics={
                "blocks": blocks,
                "max_leverage": float("nan") if empty else hi,
                "min_leverage": float("nan") if empty else lo,
                "mean_leverage": mean,
                "leave_one_out_connected": bool(not empty and hi < 1.0),
                "raw_sum_deviation": float("nan") if empty else worst_total,
                "solver": info.get("solver"),
                "converged": info.get("converged"),
                "covariates": list(covariates),
            },
        )


# --------------------------------------------------------------------------
# sample selection: the leave-one-out connected set
# --------------------------------------------------------------------------

@dataclass
class LeaveOutSet:
    """The result of pruning a panel to its leave-one-out connected set."""

    data: "object"                # a Polars LazyFrame, filtered
    workers: np.ndarray           # surviving levels of the streamed dimension
    firms: np.ndarray             # surviving levels of the other dimension
    diagnostics: dict = field(default_factory=dict)

    def summary(self):
        d = self.diagnostics
        return (
            "leave-one-out connected set\n"
            f"  workers {d['workers_before']:,} -> {d['workers_after']:,} "
            f"({d['workers_after'] / max(d['workers_before'], 1):.1%})\n"
            f"  firms   {d['firms_before']:,} -> {d['firms_after']:,} "
            f"({d['firms_after'] / max(d['firms_before'], 1):.1%})\n"
            f"  matches {d['matches_before']:,} -> {d['matches_after']:,}\n"
            f"  articulation-point workers removed: {d['cut_workers']:,} "
            f"in {d['pruning_rounds']} round(s)\n"
            f"  workers observed once, removed: "
            f"{d['single_observation_workers']:,}\n"
            f"  components before pruning: {d['components_before']:,}")


def leave_one_out_connected(data, worker="worker_id", firm="firm_id"):
    """Prune a panel to the largest leave-one-out connected set.

    Leave-out estimation needs every fixed effect to stay estimable when any
    single observation is dropped, which for a two-way model means the bipartite
    worker-firm network must stay connected when any one worker is removed. That
    fails exactly at the workers who are *articulation points* of the network.
    Following LeaveOutTwoWay (`pruning_unbal_v3.m`) and VarianceComponentsHDFE.jl:

        1. take the largest connected component
        2. drop workers observed only once -- their leverage is exactly 1
        3. delete every worker that is an articulation point, and take the
           largest connected component of what is left
        4. repeat 3 until no articulation point remains

    Step 3 has to be repeated: deleting cut vertices can create new ones. In a
    4-cycle W1-F1-W2-F2-W1 with a firm attached only to W1, W1 is a cut vertex,
    and once it is gone W2 is the only link between F1 and F2. Step 2 needs doing
    only once, because a worker seen once is a pendant vertex and removing a
    pendant never creates a cut vertex.

    `data` is a Parquet path, glob, or Polars LazyFrame. The return value's
    `.data` is the same source filtered to the surviving workers and firms, ready
    to pass to `feols_stream` or `StreamingHDFE.fit`.

    Expect this to remove a lot: in KSS's own application it drops roughly half
    the firms, mostly ones attached to a single mover. That is a property of the
    estimand, not of this implementation -- the leave-out parameters are simply
    not identified off those firms.

    Only the network is held in memory, never the rows: one vertex per worker and
    per firm, one edge per distinct match.
    """
    import polars as pl

    from .kernels_graph import (_nb_articulation_points, _nb_components,
                                build_bipartite)

    frame = data if isinstance(data, pl.LazyFrame) else pl.scan_parquet(data)

    # one streaming pass for the distinct matches and how many rows each has
    matches = (frame.group_by(worker, firm).len("rows")
               .collect(engine="streaming"))
    worker_ids = matches[worker].unique().sort()
    firm_ids = matches[firm].unique().sort()
    n_workers, n_firms = len(worker_ids), len(firm_ids)

    # map ids to dense codes by sorted search, which works for any id type
    worker_code = np.searchsorted(worker_ids.to_numpy(), matches[worker].to_numpy())
    firm_code = np.searchsorted(firm_ids.to_numpy(), matches[firm].to_numpy())

    indptr, indices = build_bipartite(worker_code, firm_code, n_workers, n_firms)
    n_vertices = n_workers + n_firms

    # step 1: the largest connected component
    alive = np.ones(n_vertices, bool)
    label = np.empty(n_vertices, np.int64)
    n_components = _nb_components(indptr, indices, alive, label)
    if n_components > 1:
        sizes = np.bincount(label[label >= 0], minlength=n_components)
        alive = label == int(np.argmax(sizes))

    def largest(alive):
        n_left = _nb_components(indptr, indices, alive, label)
        if n_left == 0:
            return np.zeros_like(alive)
        sizes = np.bincount(label[label >= 0], minlength=n_left)
        return label == int(np.argmax(sizes))

    # step 2: workers observed once. A worker's rows all sit at firms in its own
    # component, so its row count is fixed by the data, not by the pruning.
    rows_per_worker = np.bincount(worker_code, weights=matches["rows"].to_numpy(),
                                  minlength=n_workers)
    single = (rows_per_worker == 1) & alive[:n_workers]
    single_observation = int(single.sum())
    alive[:n_workers] &= ~single
    alive = largest(alive)

    # steps 3-4: delete articulation-point workers until there are none
    cut_workers, rounds = 0, 0
    is_cut = np.zeros(n_vertices, bool)
    while True:
        is_cut[:] = False
        _nb_articulation_points(indptr, indices, is_cut, alive)
        cut = is_cut[:n_workers] & alive[:n_workers]
        if not cut.any():
            break
        cut_workers += int(cut.sum())
        rounds += 1
        alive[:n_workers] &= ~cut
        alive = largest(alive)

    keep_workers = worker_ids.to_numpy()[alive[:n_workers]]
    keep_firms = firm_ids.to_numpy()[alive[n_workers:]]
    kept_matches = int((alive[worker_code] & alive[n_workers + firm_code]).sum())

    # `implode` makes the membership test unambiguous: the right-hand side is
    # one list to look inside, not a column to compare against element by
    # element. Polars deprecated the bare form.
    filtered = frame.filter(
        pl.col(worker).is_in(pl.Series(worker, keep_workers).implode())
        & pl.col(firm).is_in(pl.Series(firm, keep_firms).implode()))

    return LeaveOutSet(
        data=filtered, workers=keep_workers, firms=keep_firms,
        diagnostics={
            "workers_before": n_workers, "workers_after": len(keep_workers),
            "firms_before": n_firms, "firms_after": len(keep_firms),
            "matches_before": matches.height, "matches_after": kept_matches,
            "cut_workers": cut_workers, "components_before": n_components,
            "single_observation_workers": single_observation,
            "pruning_rounds": rounds,
        },
    )


# --------------------------------------------------------------------------
# the bias term: Hutchinson's trace estimator
# --------------------------------------------------------------------------

@dataclass
class TraceEstimate:
    """The bias terms of the three AKM variance components."""

    var_psi: float
    var_alpha: float
    cov: float
    n_draws: int
    diagnostics: dict = field(default_factory=dict)

    def as_dict(self):
        return {"var(psi)": self.var_psi, "var(alpha)": self.var_alpha,
                "cov(psi, alpha)": self.cov}

    def summary(self):
        d = self.diagnostics
        return (f"bias terms from {self.n_draws} Hutchinson draws in "
                f"{d.get('blocks', '?')} block(s)\n"
                f"  var(psi)        {self.var_psi:+.6f}\n"
                f"  var(alpha)      {self.var_alpha:+.6f}\n"
                f"  cov(psi, alpha) {self.cov:+.6f}")


class _TraceMixin:
    # Defined by other mixins:
    #   _InverseMixin:       apply_inverse, _chunks, _chunk_layout,
    #                        _covariate_matrix, _border, _scratch_capped
    #   _LeaveOutMixin:      _model_covariates, _quiet_solver
    #   _StandardErrorMixin: _moment_divisor
    #   StreamingHDFE:       _offsets, o_fe, g_fe, n_levels, offs, rhs_block,
    #                        verbose, logger, log_level

    def hutchinson_trace(self, sigma2, n_draws=250, block=None, seed=0,
                         covariates=None, psi=None):
        """Estimate tr(Q S^- A' Omega A S^-) for the three variance components.

        This is the term KSS subtract from the plug-in estimates. Omega is
        diag(sigma2), so `sigma2` is the per-row variance estimate, in the row
        order the residual file and `jla_leverages` both use.

        For Rademacher Z, E[Z' M Z] = tr(M), so each draw contributes
        Z'Q u2 with u2 = S^- A' Omega A S^- Z. That is two applications of the
        inverse per draw, and the estimate is their average.

        psi names the non-streamed dimension whose variance is wanted -- the
        firms, normally -- and the streamed dimension plays the part of alpha.

        Returns a `TraceEstimate`. Blocking works as it does for the leverages:
        `block` draws share each pair of solves, and the answer does not depend
        on how the draws are divided up.
        """
        from .kernels_inverse import (_nb_group_sums, _nb_reduce_rows,
                                      _nb_trace_moments)

        block = self.rhs_block if block is None else int(block)
        if block < 1 or n_draws < 1:
            raise ValueError("n_draws and block must be at least 1")
        covariates = self._model_covariates(covariates)
        psi = psi or self.o_fe[0]
        if psi not in self.o_fe:
            raise ValueError(f"psi must be a non-streamed dimension, one of "
                             f"{self.o_fe}; got {psi!r}")

        offsets, total_levels = self._offsets()
        n_groups = self.n_levels[self.g_fe]
        columns, _, _ = self._border(covariates)
        psi_column = self.o_fe.index(psi)
        psi_offset = offsets[psi]
        sigma2 = np.asarray(sigma2, dtype=np.float64)

        _log(self.verbose,
             f"leave-out: bias term, {n_draws} draws in blocks of {block} "
             "(two solves each)", self.logger, self.log_level)
        totals = np.zeros(8)
        blocks, drawn = 0, 0
        with self._quiet_solver(), self._scratch_capped(10 * block):
            while drawn < n_draws:
                size = min(block, n_draws - drawn)
                totals += self._trace_block(
                    size, drawn, seed, columns, covariates, total_levels,
                    n_groups, sigma2, psi_column, psi_offset,
                    _nb_group_sums, _nb_reduce_rows, _nb_trace_moments)
                drawn += size
                blocks += 1

        # turn the accumulated moments into the three traces
        weight, a1a2, b1b2, cross, a1, a2, b1, b2 = totals / n_draws
        var_psi = a1a2 / weight - (a1 / weight) * (a2 / weight)
        var_alpha = b1b2 / weight - (b1 / weight) * (b2 / weight)
        covariance = 0.5 * (cross / weight
                            - (a1 / weight) * (b2 / weight)
                            - (b1 / weight) * (a2 / weight))
        # those divide by the weight total; the components divide by n - 1
        scale = weight / self._moment_divisor(weight)
        var_psi, var_alpha, covariance = (scale * var_psi, scale * var_alpha,
                                          scale * covariance)

        return TraceEstimate(
            var_psi=float(var_psi), var_alpha=float(var_alpha),
            cov=float(covariance), n_draws=n_draws,
            diagnostics={"blocks": blocks, "psi": psi, "alpha": self.g_fe,
                         "covariates": list(covariates),
                         "n_obs": float(weight)})

    def _trace_block(self, size, first_draw, seed, columns, covariates,
                     total_levels, n_groups, sigma2, psi_column, psi_offset,
                     group_sums, reduce_rows, trace_moments):
        """One block of draws: two applications of the inverse, then moments."""
        import numba as nb

        from .kernels_inverse import _nb_rademacher

        n_threads = nb.get_num_threads()
        # One coefficient-space Rademacher vector per draw, from the same
        # counter-based generator the leverages use. Keying on (position, draw)
        # rather than reseeding per block is what makes the answer independent
        # of how the draws are blocked: block b asks for its own stretch of one
        # sequence. The three parts occupy disjoint stretches of the position
        # space, so they are independent of each other.
        n_cov = len(columns)
        Z = np.empty((total_levels + n_groups + n_cov, size))
        _nb_rademacher(0, Z.shape[0], size, seed, first_draw, Z)
        Z_levels = np.ascontiguousarray(Z[:total_levels])
        Z_groups = np.ascontiguousarray(Z[total_levels:total_levels + n_groups])
        Z_cov = np.ascontiguousarray(Z[total_levels + n_groups:])

        # u = S^- Z, accumulating v = A' Omega A u in the same pass
        acc = np.zeros((n_threads, total_levels, size))
        acc_x = np.zeros((n_threads, len(columns), size))
        v_groups = np.zeros((n_groups, size))

        def collect(ordinal, chunk, rows):
            weighted = rows * sigma2[ordinal:ordinal + len(rows), None]
            starts, codes, w = self._chunk_layout(chunk)
            X = self._covariate_matrix(chunk, columns, len(w))
            reduce_rows(starts, codes, self.offs, w, X, weighted, acc, acc_x)
            first = int(chunk["gcode"][0])
            group_sums(starts, w, weighted,
                       v_groups[first:first + len(starts) - 1])

        self.apply_inverse(Z_levels, Z_cov, Z_groups, covariates=covariates,
                           sink=collect)
        v_levels = acc.sum(axis=0)
        v_cov = acc_x.sum(axis=0)

        # u2 = S^- v
        U_levels, _, U_groups = self.apply_inverse(
            v_levels, v_cov, v_groups, covariates=covariates)

        # Z' Q u2, for all three forms, in one pass
        moments = np.zeros((n_threads, size, 8))
        for _, chunk, starts, codes, w in self._chunks():
            first = int(chunk["gcode"][0])
            n_chunk_groups = len(starts) - 1
            trace_moments(starts, codes, psi_column, psi_offset, w,
                          Z_levels, Z_groups[first:first + n_chunk_groups],
                          U_levels, U_groups[first:first + n_chunk_groups],
                          moments)
        return moments.sum(axis=0).sum(axis=0)


# --------------------------------------------------------------------------
# the estimator
# --------------------------------------------------------------------------

@dataclass
class LeaveOutComponents:
    """KSS leave-out variance components, with the plug-in ones for comparison."""

    plug_in: dict
    bias: dict
    leave_out: dict
    n_obs: int
    n_movers: int
    leverages: object
    trace: object
    se: object = None
    weak_id: object = None
    diagnostics: dict = field(default_factory=dict)
    pruning: object = None
    fit: object = None
    match_fit: object = None

    def tidy(self):
        """The three components as a Polars DataFrame."""
        import polars as pl

        keys = list(self.plug_in)
        return pl.DataFrame({
            "component": keys,
            "plug_in": [self.plug_in[k] for k in keys],
            "bias": [self.bias[k] for k in keys],
            "leave_out": [self.leave_out[k] for k in keys],
        })

    def summary(self):
        d = self.diagnostics
        lines = [f"leave-out variance components ({d['psi']} and {d['alpha']})"]
        if d.get("leave_out") == "match":
            # stayers' matches have leverage one by construction, so the
            # maximum that matters is over movers' matches
            lines += [f"  {self.n_obs:,} observations in {d['n_matches']:,} "
                      f"matches, {self.n_movers:,} of them movers'",
                      f"  leaving out a match, {self.leverages.n_draws} draws; "
                      f"max mover leverage {d['max_leverage']:.6f}; "
                      f"mean sigma2 {d['sigma2_mean']:.6f}"]
        else:
            lines += [f"  {self.n_obs:,} observations, {self.n_movers:,} of "
                      f"them movers; {self.leverages.n_draws} draws",
                      f"  max leverage {d['max_leverage']:.6f}   "
                      f"mean sigma2 {d['sigma2_mean']:.6f}"]
        if self.pruning is not None:
            p = self.pruning.diagnostics
            lines.append(
                f"  sample: {p['workers_before']:,} -> {p['workers_after']:,} "
                f"{d['alpha']}, {p['firms_before']:,} -> {p['firms_after']:,} "
                f"{d['psi']} after pruning to the leave-one-out connected set")
        wide = self.se is not None
        confidence = self.diagnostics.get("confidence", 0.95)
        from scipy.stats import norm
        z = float(norm.ppf(0.5 + confidence / 2.0))
        header = f"{'component':<20}{'plug-in':>12}{'bias':>12}{'leave-out':>12}"
        if wide:
            header += f"{'se':>11}{f'{confidence:.0%} interval':>26}"
        lines += ["", header, "-" * (93 if wide else 56)]
        for key in self.plug_in:
            row = (f"{key:<20}{self.plug_in[key]:>12.6f}"
                   f"{self.bias[key]:>12.6f}{self.leave_out[key]:>12.6f}")
            if wide:
                se = self.se.se[key]
                row += f"{se:>11.6f}"
                if np.isfinite(se):
                    lo = self.leave_out[key] - z * se
                    hi = self.leave_out[key] + z * se
                    flag = ("*" if self.weak_id is not None
                            and self.weak_id.q[key] >= 1 else " ")
                    row += f"{f'[{lo:.6f}, {hi:.6f}]':>26}{flag}"
                else:
                    row += f"{'--':>26}"
            lines.append(row)
        if wide:
            lines.append(
                f"\nstandard errors from {self.se.n_draws or 'no'} trace draws"
                + ("" if self.se.diagnostics["trace_included"]
                   else "; conservative, trace term omitted"))
            if self.weak_id is not None and self.weak_id.weakly_identified:
                lines.append("* weakly identified (q >= 1): the normal interval "
                             "understates the uncertainty for this component")
            weak = self.se.weak or {}
            if self.diagnostics.get("leave_out") == "match":
                source = self.diagnostics.get("se_variance", "person_year")
                lines.append(
                    "each match's error variance: "
                    + ("from its person-year residuals, as leave_out_COMPLETE "
                       "does (assumes errors independent within a spell)"
                       if source == "person_year" else
                       "from the leave-match-out residual, robust to errors "
                       "correlated within a spell")
                    + f" [se_variance={source!r}]")
                lines.append("var(alpha): no standard error when leaving out a "
                             "match, as in LeaveOutTwoWay's leave_out_COMPLETE; "
                             "leave_out='observation' gives one")
                key = "cov(psi, alpha)"
                if (self.weak_id is not None and self.weak_id.q[key] >= 1
                        and key not in weak):
                    lines.append(f"{key}: no q = 1 interval when leaving out a "
                                 "match, since stayers load on its weak "
                                 "direction (docs/kss_methodological_differences.md)")
            shown = [k for k in self.plug_in if "interval" in weak.get(k, {})]
            if shown:
                level = weak[shown[0]]["confidence"]
                lines += ["", f"intervals valid under weak identification "
                              f"(KSS Theorem 3, q = 1, {level:.0%})",
                          f"{'component':<20}{'leave-out':>12}{'interval':>28}"
                          f"{'kappa':>8}{'width vs normal':>18}", "-" * 86]
                for key in shown:
                    piece = weak[key]
                    lo, hi = piece["interval"]
                    normal = 2 * z * self.se.se[key]
                    ratio = ((hi - lo) / normal if np.isfinite(normal) and normal
                             else float("nan"))
                    width = f"{ratio:.2f}x" if np.isfinite(ratio) else "--"
                    lines.append(
                        f"{key:<20}{self.leave_out[key]:>12.6f}"
                        f"{f'[{lo:.6f}, {hi:.6f}]':>28}{piece['kappa']:>8.2f}"
                        f"{width:>18}")
                if any(piece.get("conservative") for piece in weak.values()):
                    lines.append("  covariance estimate fell back to its "
                                 "conservative form for at least one component")
                if any(self.weak_id.q[k] >= 2 for k in shown):
                    lines.append("  q >= 2 for at least one component: the q = 1 "
                                 "interval does not account for the second "
                                 "direction and may undercover")
        if self.weak_id is not None:
            lines += ["", self.weak_id.summary()]
        return "\n".join(lines)


class _ComponentsMixin:
    # Defined by other mixins:
    #   _LeaveOutMixin:      jla_leverages
    #   _TraceMixin:         hutchinson_trace
    #   _StandardErrorMixin: component_variances, _moment_divisor
    #   _WeakIdMixin:        weak_id_diagnostics
    #   _InverseMixin:       _chunks, row_count, row_weights
    #   StreamingHDFE:       o_fe, g_fe, n_levels, weights, verbose, logger

    def leave_out_components(self, result, n_draws=250, block=None, seed=0,
                             psi=None, covariates=None, stayers="own",
                             se=False, se_draws=None, se_trace=True,
                             smooth_bins=None, diagnose=None, diagnose_draws=64,
                             weak_interval=True, confidence=0.95,
                             stayer_sigma2=None, centering="weighted",
                             se_variance="person_year", match_within=None):
        """Kline-Saggio-Solvsten variance components for a fitted model.

        The plug-in estimates of var(psi), var(alpha) and cov(psi, alpha) are
        biased -- upward for the variances, toward zero for the covariance --
        because the fixed effects are estimated with error. This removes that
        bias without assuming homoskedasticity, by combining

          * `jla_leverages`, for the leverage each observation carries;
          * a leave-one-out variance estimate built from those and the
            residuals;
          * `hutchinson_trace`, for the bias the two imply.

        `result` is the `HDFEResult` from the fit that produced this estimator's
        intermediates, read for its residuals and fitted effects.

        The panel must be leave-one-out connected -- see
        `leave_one_out_connected` -- or some observation's leave-one-out
        residual is undefined and this will say so.

        With weights the target is the *weighted* variance decomposition, and
        every piece follows from that one choice: the leverages are the ones of
        the square-root-weight hat matrix, the plug-in moments are weighted, and
        sigma2 carries a factor of w -- see `_leave_one_out_sigma2`.

        `diagnose` runs KSS's weak-identification diagnostic (see
        `weak_id_diagnostics`): whether a few eigen-directions dominate the
        estimator's variance, in which case the normal interval understates the
        uncertainty. It defaults to `se`, since it judges the interval and the
        point estimate is unbiased regardless. `diagnose_draws` is its trace
        budget.

        With `se=True` and `weak_interval=True`, every component the diagnostic
        flags (q >= 1) also gets KSS's q = 1 interval, which stays valid under
        weak identification: the Andrews-Mikusheva projection of the joint
        ellipse for (b1, theta1). It is at `confidence` (default 95%), in
        `se.weak[name]["interval"]`. It costs roughly one more set of
        standard-error draws per flagged component, and nothing when none is.

        `centering` is how sigma2 centers the outcome: "weighted" (default) at
        its weighted mean, "reference" as LeaveOutTwoWay does at match level --
        see `_leave_one_out_sigma2`. Unweighted the two are identical.
        `stayers="within_match"` with `stayer_sigma2` is for `leave_out_match`,
        as are `se_variance` and `match_within`: how each match's error
        variance is estimated for the standard errors, "person_year" (the
        reference's, the within-match variance `match_within` plus the
        collapsed sigma2 over T) or "match" (the collapsed sigma2 alone, which
        stays right when errors are correlated within a spell).

        Not supported here: models with several outcomes.
        """
        from .kernels_inverse import _nb_group_distinct

        if stayers not in ("own", "firm_mean", "drop", "within_match"):
            raise ValueError("stayers must be 'own', 'firm_mean', 'drop' or "
                             "'within_match'")
        if (stayers == "within_match") != (stayer_sigma2 is not None):
            raise ValueError("stayers='within_match' needs stayer_sigma2, the "
                             "per-match values from leave_out_match")

        psi = psi or self.o_fe[0]
        if psi not in self.o_fe:
            raise ValueError(f"psi must be a non-streamed dimension, one of "
                             f"{self.o_fe}; got {psi!r}")
        alpha = self.g_fe

        match_level = stayers == "within_match"

        leverages = self.jla_leverages(n_draws=n_draws, block=block, seed=seed,
                                       covariates=covariates)
        movers, psi_code = self._mover_mask(psi, _nb_group_distinct)
        # A stayer's single match has leverage exactly 1 when matches are left
        # out -- that is what the within-match rule is for -- so only movers'
        # matches are held to the connectedness requirement there.
        checked = movers if match_level else np.ones(len(movers), bool)
        worst = (float(np.max(np.asarray(leverages.leverage)[checked]))
                 if checked.any() else 0.0)
        if worst >= 1.0 - 1e-9:
            unit = "match" if match_level else "observation"
            raise ValueError(
                f"maximum leverage is {worst:.6f}: the panel is not leave-one-out "
                f"connected (leaving out one {unit} at a time), so the leave-out "
                f"residual is undefined for at least one {unit}. Prune with "
                "hdfe_stream.leaveout.leave_one_out_connected() and refit")

        # The plug-in moments have to be weighted the same way the trace is,
        # since the bias is subtracted from them: _nb_trace_moments accumulates
        # with w, so these do too. Without weights this is np.var and np.cov.
        #
        # They are second moments of two columns, so Polars computes them in a
        # streaming pass. Collecting the columns to take a variance of them would
        # be two row-sized arrays held for no other purpose.
        plug_in = self._plug_in_moments(result, psi, alpha)

        y, residual = self._outcome_and_residual(result, leverages.n_rows)
        w = self.row_weights()
        sigma2 = self._leave_one_out_sigma2(y, residual, leverages, movers,
                                            psi_code, stayers, w, centering=centering)
        del y, residual
        if stayers == "within_match":
            self._assign_stayer_sigma2(result, sigma2, movers, stayer_sigma2, psi)
        trace = self.hutchinson_trace(sigma2, n_draws=n_draws, block=block,
                                      seed=seed + 1_000_003, psi=psi,
                                      covariates=covariates)
        bias = trace.as_dict()
        leave_out = {k: plug_in[k] - bias[k] for k in plug_in}

        # The weak-identification diagnostic is about whether the *normal
        # interval* is justified, so by default it runs exactly when there is an
        # interval to judge. The point estimate is unbiased either way.
        if diagnose is None:
            diagnose = bool(se)
        weak_id = None
        if diagnose:
            weak_id = self.weak_id_diagnostics(
                result, sigma2, leverages, n_draws=int(diagnose_draws),
                block=block,
                seed=seed + 9_900_019, covariates=covariates, psi=psi)

        errors = None
        if se:
            options = {}
            sigma2_se = sigma2
            if match_level:
                sigma2_se = self._match_se_sigma2(
                    result, sigma2, leverages, movers, psi_code, w, psi,
                    centering, se_variance, match_within, stayer_sigma2)
                options = {"stayer_rows": ~movers, "centering": centering,
                           "reported": (0, 2), "sigma2_point": sigma2}
            errors = self.component_variances(
                result, sigma2_se, leverages,
                n_draws=n_draws if se_draws is None else int(se_draws),
                block=block, seed=seed + 5_500_011, covariates=covariates,
                psi=psi, trace=se_trace, smooth_bins=smooth_bins,
                weak_id=weak_id if weak_interval else None, **options)

        # KSS Section 6: for a weakly identified component, the interval that
        # stays valid projects a 2-d ellipse around (b1, theta1) through
        # lambda_1 b^2 + t, with Andrews and Mikusheva's curvature-adjusted
        # critical value. theta1 is formed from the reported estimate with the
        # same sigma2 it used, so the two intervals share a center as
        # lambda_1 -> 0.
        if errors is not None and errors.weak:
            from .am_interval import am_interval

            for name, piece in errors.weak.items():
                theta1 = leave_out[name] - piece["lambda_1"] * (
                    piece["b1"] ** 2 - piece["v_raw"])
                lo, hi, info = am_interval(piece["lambda_1"], piece["b1"],
                                           theta1, piece["sigma"],
                                           1.0 - confidence)
                piece.update(theta1=float(theta1), interval=(lo, hi),
                             confidence=confidence, **info)

        return LeaveOutComponents(
            plug_in=plug_in, bias=bias, leave_out=leave_out,
            n_obs=leverages.n_rows, n_movers=int(movers.sum()),
            leverages=leverages, trace=trace, se=errors, weak_id=weak_id,
            diagnostics={"psi": psi, "alpha": alpha, "stayers": stayers,
                         "confidence": confidence, "max_leverage": worst,
                         "leave_out": "match" if match_level else "observation",
                         "sigma2_mean": float(np.mean(sigma2 / w)),
                         "covariates": leverages.diagnostics["covariates"],
                         "n_levels": {psi: self.n_levels[psi],
                                      alpha: self.n_levels[alpha]}})

    def _plug_in_moments(self, result, psi, alpha):
        """Weighted var(psi), var(alpha) and cov, as a streaming aggregate."""
        import polars as pl

        weight = (pl.col("weights") if self.weights is not None
                  else pl.lit(1.0)).alias("_w")
        a, b = pl.col(f"fe_{psi}"), pl.col(f"fe_{alpha}")
        totals = (result.resid()
                  .select(weight, a.alias("_a"), b.alias("_b"))
                  .select(pl.col("_w").sum().alias("t"),
                          (pl.col("_w") * pl.col("_a")).sum().alias("a"),
                          (pl.col("_w") * pl.col("_b")).sum().alias("b"),
                          (pl.col("_w") * pl.col("_a") ** 2).sum().alias("aa"),
                          (pl.col("_w") * pl.col("_b") ** 2).sum().alias("bb"),
                          (pl.col("_w") * pl.col("_a") * pl.col("_b")).sum()
                          .alias("ab"))
                  .collect(engine="streaming").row(0, named=True))
        total = totals["t"]
        mean_a, mean_b = totals["a"] / total, totals["b"] / total
        # divide by n - 1, as the references do (see _moment_divisor)
        scale = total / self._moment_divisor(total)
        return {
            "var(psi)": scale * (totals["aa"] / total - mean_a * mean_a),
            "var(alpha)": scale * (totals["bb"] / total - mean_b * mean_b),
            "cov(psi, alpha)": scale * (totals["ab"] / total - mean_a * mean_b),
        }

    def _outcome_and_residual(self, result, expected):
        """The outcome and residual columns, as plain arrays.

        Kept only as long as sigma2 needs them, and read without the two fitted
        effect columns beside them -- those are aggregated separately.
        """
        frame = (result.resid().select(result.depvar, "resid")
                 .collect(engine="streaming"))
        if len(frame) != expected:
            raise ValueError(
                f"the residual file has {len(frame):,} rows but the leverages "
                f"have {expected:,}; they must come from the same fit")
        return (frame[result.depvar].to_numpy(), frame["resid"].to_numpy())

    def _match_se_sigma2(self, result, sigma2, leverages, movers, psi_code, w,
                         psi, centering, se_variance, match_within,
                         stayer_sigma2):
        """Each match row's error variance, as the standard errors use it.

        What the variance formula needs per collapsed row is the variance of
        sqrt(T) times the match mean. Two estimates of it
        (docs/kss_methodological_differences.md 5.7):

          "match"        beyond the reference. The collapsed leave-match-out
                         sigma2 itself, weighted-
                         centered whatever the point estimate's centering, since
                         the level carried by the reference's adds only noise;
                         stayers' within-match values from `sigma_for_stayers`.
                         Estimates the whole block, 1' Omega_m 1 / T, so it
                         stays right when errors are correlated within a spell.
          "person_year"  the reference implementation's, LeaveOutTwoWay's
                         leave_out_COMPLETE with matches (beta there; xhdfe
                         ports it): sum over the match of y_i eta_h,i, with y
                         centered, divided by T. In
                         collapsed terms that is the within-match variance plus
                         the "match" estimate over T -- the two pooled, which is
                         less noisy when errors are independent within a spell
                         and understates the variance when they are not. A
                         stayer gets its within-match variance alone.
        """
        if se_variance not in ("match", "person_year"):
            raise ValueError("se_variance must be 'match' or 'person_year'")
        if centering == "weighted":
            block = np.array(sigma2, dtype=np.float64, copy=True)
        else:
            y, residual = self._outcome_and_residual(result, leverages.n_rows)
            block = self._leave_one_out_sigma2(
                y, residual, leverages, movers, psi_code, "within_match", w,
                centering="weighted")
            del y, residual
        if se_variance == "match":
            self._assign_stayer_sigma2(result, block, movers, stayer_sigma2,
                                       psi)
            return block
        if match_within is None:
            raise ValueError("se_variance='person_year' needs the within-match "
                             "variances from leave_out_match")
        within = self._per_match(result, match_within, "within", psi)
        out = within.copy()
        out[movers] = within[movers] + block[movers] / w[movers]
        return out

    def _per_match(self, result, table, column, psi):
        """A per-match column of `table`, on the match fit's rows, in the
        residual file's order."""
        alpha = self.g_fe
        keys = (result.resid().select(alpha, psi)
                .with_row_index("_row").collect(engine="streaming"))
        joined = keys.join(table, on=[alpha, psi], how="left").sort("_row")
        values = joined[column].to_numpy().astype(np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{int(np.sum(~np.isfinite(values))):,} matches "
                             f"have no {column!r} value")
        return values

    def _assign_stayer_sigma2(self, result, sigma2, movers, table, psi):
        """Put `leave_out_match`'s stayer values on the stayers' match rows.

        The rows are read in the residual file's order, which is the order the
        leverages and sigma2 are in.
        """
        alpha = self.g_fe
        keys = (result.resid().select(alpha, psi)
                .with_row_index("_row").collect(engine="streaming"))
        joined = keys.join(table, on=[alpha, psi], how="left").sort("_row")
        values = joined["sigma2"].to_numpy()
        stayer_rows = ~movers
        missing = stayer_rows & ~np.isfinite(values)
        if missing.any():
            raise ValueError(f"{int(missing.sum()):,} stayer matches have no "
                             "within-match variance; they need at least two "
                             "person-years, which pruning ensures")
        sigma2[stayer_rows] = values[stayer_rows]

    def _mover_mask(self, psi, group_distinct):
        """Per-row: is this observation's group seen at more than one level of
        `psi`? Also returns the row's `psi` level code, which the stayer
        imputation needs."""
        column = self.o_fe.index(psi)
        n_groups = self.n_levels[self.g_fe]
        distinct = np.zeros(n_groups, np.int64)
        codes_out = np.empty(self.row_count(), np.int64)

        for ordinal, chunk, starts, codes, _w in self._chunks():
            first = int(chunk["gcode"][0])
            group_distinct(starts, codes, column,
                           distinct[first:first + len(starts) - 1])
            codes_out[ordinal:ordinal + len(codes)] = codes[:, column]

        movers = np.empty(len(codes_out), bool)
        for ordinal, chunk, starts, _codes, _w in self._chunks():
            gcode = np.asarray(chunk["gcode"])
            movers[ordinal:ordinal + len(gcode)] = distinct[gcode] > 1
        return movers, codes_out

    def _leave_one_out_sigma2(self, y, residual, leverages, movers, psi_code,
                              stayers, w, centering="weighted"):
        """Unbiased per-observation variance, without assuming homoskedasticity.

        sigma2_i = w_i (y_i - ybar) e_i / M_ii: the algebraic form of
        y_i(y_i - x_i' beta_{-i}), whose expectation is Var(e_i) whatever the
        other observations' variances are. The outcome is demeaned, as in
        LeaveOutTwoWay's `leave_out_KSS`, VarianceComponentsHDFE.jl, xhdfe and
        pytwoway.

        What comes back is w_i Var(e_i), not Var(e_i), which is what the trace
        wants: the sandwich is S^- A'W Omega W A S^-, carrying two factors of w,
        and the trace chain applies one of them itself. The other rides along
        here. Unweighted, w is one and this is exactly the KSS expression.

        `stayers` says what workers seen at a single firm get:

          "own"        (default) their own sigma2_i, like every other row. This
                       is LeaveOutTwoWay's rule when leaving out an observation:
                       a stayer seen at least twice has a well-defined leave-out
                       residual, since dropping one of its years leaves the
                       worker effect identified by the others.
          "firm_mean"  the mean of the movers' estimates at the same level of psi
                       -- pytwoway's default, designed for spell-level data where
                       a stayer has one observation. The factor of w is divided
                       out before averaging and put back after.
          "drop"       zero: stayers contribute nothing to the bias correction.
          "within_match"  for rows that are matches: left NaN here and filled
                       by the caller from `leave_out_match`'s stayer values.

        `centering="reference"` centers in the square-root-weight metric
        instead, (sqrt(w) y - mean of sqrt(w) y) sqrt(w) e / M, as LeaveOutTwoWay
        does when leaving out a match. Unweighted the two are identical.
        """
        sigma2 = np.full(len(y), np.nan)
        rows = np.ones(len(y), bool) if stayers == "own" else movers
        if centering == "reference":
            root = np.sqrt(w)
            y_t = root * y
            sigma2[rows] = ((y_t[rows] - y_t.mean()) * root[rows] * residual[rows]
                            / leverages.complement[rows] * leverages.factor[rows])
        else:
            y_bar = float(w @ y / w.sum())
            sigma2[rows] = (w[rows] * (y[rows] - y_bar) * residual[rows]
                            / leverages.complement[rows] * leverages.factor[rows])
        if stayers in ("own", "within_match"):
            return sigma2
        if stayers == "drop":
            sigma2[~movers] = 0.0
            return sigma2

        n_levels = int(psi_code.max()) + 1 if len(psi_code) else 0
        # sum(w^2 Var(e)) / sum(w^2): the w-weighted mean of sigma2 / w
        totals = np.bincount(psi_code[movers], weights=sigma2[movers] * w[movers],
                             minlength=n_levels)
        counts = np.bincount(psi_code[movers], weights=w[movers] ** 2,
                             minlength=n_levels)
        level_mean = np.divide(totals, counts, out=np.full(n_levels, np.nan),
                               where=counts > 0)
        sigma2[~movers] = level_mean[psi_code[~movers]] * w[~movers]

        if not np.all(np.isfinite(sigma2)):
            orphaned = int(np.sum(~np.isfinite(sigma2)))
            raise ValueError(
                f"{orphaned:,} observations sit at a level of the chosen "
                "dimension with no movers, so there is nothing to impute their "
                "variance from; prune to the leave-one-out connected set first")
        return sigma2


def _check_leave_out(leave_out, se, stayers=None, centering="reference",
                     se_variance="person_year", weights=None):
    """Refuse an impossible combination before any fitting is done."""
    if leave_out not in ("match", "observation"):
        raise ValueError("leave_out must be 'match' or 'observation'")
    if se_variance not in ("match", "person_year"):
        raise ValueError("se_variance must be 'match' or 'person_year'")
    if (leave_out == "match" and se and se_variance == "person_year"
            and weights is not None):
        raise ValueError(
            "se_variance='person_year' is LeaveOutTwoWay's, which has no user "
            "weights; use se_variance='match' with a weighted fit")
    if centering not in ("reference", "weighted"):
        raise ValueError("centering must be 'reference' or 'weighted'")
    if leave_out == "match" and stayers is not None:
        raise ValueError(
            "stayers applies when leaving out an observation; leaving out a "
            "match uses LeaveOutTwoWay's within-match rule")


def leave_out_kss(fml, data, workdir=None, *, psi=None, n_draws=250, seed=0,
                  block=None, prune=True, leave_out="match", stayers=None,
                  se=False, se_draws=None, se_trace=True, diagnose=None,
                  diagnose_draws=64, weak_interval=True, confidence=0.95,
                  centering="reference", se_variance="person_year", verbose=True,
                  logger=None, **options):
    """Kline-Saggio-Solvsten leave-out variance components, end to end.

        lo = leave_out_kss("log_earn ~ age_squared | worker_id + firm_id",
                           "data/*.parquet", workdir="scratch")
        lo.summary()

    The plug-in AKM decomposition is biased -- upward for the variances, toward
    zero for the covariance -- because the fixed effects are estimated with
    error. This removes that bias, following Kline, Saggio and Solvsten (2020),
    without assuming the errors are identically distributed.

    Three things happen, in this order, because the order matters:

    1. the panel is pruned to its largest leave-one-out connected set. Leave-out
       estimation needs every effect to stay estimable when any one match is
       dropped, and that fails at workers who are articulation points of the
       worker-firm network; they are removed, repeatedly, until none is left,
       and workers observed once are dropped too. **This changes the estimation
       sample**, often substantially -- in KSS's own application it removes
       about half the firms -- which is why it happens before the fit;
    2. the model is fitted on what survives;
    3. leverages are approximated by random projection, and from them the bias.

    `leave_out="match"` (default) leaves out a whole worker-firm match, as
    LeaveOutTwoWay, VarianceComponentsHDFE.jl and xhdfe do: the fit's
    partialled-out outcome is collapsed to one row per match and the correction
    runs on that weighted regression (see `leaveout_match`). Stayers' variances
    follow LeaveOutTwoWay's within-match rule, and `centering` chooses how
    sigma2 centers the outcome ("reference", the default, or "weighted"; see
    `HDFEResult.leave_out_kss`). With `se=True` there, standard errors are
    reported for var(psi) and the covariance, following leave_out_COMPLETE's
    match-level option (beta there), and `se_variance` chooses how each match's
    error variance is estimated.
    `leave_out="observation"` leaves out one row at a time; `stayers` applies
    there ("own" by default) and all three components get standard errors.

    `psi` names the non-streamed dimension whose variance is wanted (the firms);
    the streamed dimension plays the part of alpha (the workers). Both default
    to the first two fixed effects in the formula, and the streamed dimension is
    pinned to the first so that the sample that was pruned and the decomposition
    that follows are about the same pair.

    `prune=False` skips step 1 and fails if the panel is not already
    leave-one-out connected, for callers who have pruned upstream and want to be
    told if they are wrong.

    Extra options are passed to the fit, so `weights=` and the memory knobs
    remain available. Several outcomes at once are not supported.
    """
    from .api import feols_stream
    from .feterms import _parse_fe, _parse_fe_term
    from .report import _log, _warn

    if "|" not in fml:
        raise ValueError(
            f"{fml!r} has no fixed effects; leave-out estimation needs at least "
            "two dimensions, as in 'y ~ x | worker_id + firm_id'")
    dimensions = [_parse_fe_term(term)[0] for term in _parse_fe(fml.split("|")[1])]
    if len(dimensions) < 2:
        raise ValueError(
            f"{fml!r} has one fixed effect; leave-out estimation needs at least "
            "two, one of which is streamed")
    alpha = dimensions[0]
    psi = psi or dimensions[1]
    if psi == alpha or psi not in dimensions:
        raise ValueError(
            f"psi must be one of the fixed effects other than {alpha!r} "
            f"(which plays the part of alpha); got {psi!r} from {dimensions}")

    pruning = None
    if prune:
        pruning = leave_one_out_connected(data, worker=alpha, firm=psi)
        d = pruning.diagnostics
        message = (
            f"pruned to the leave-one-out connected set: "
            f"{d['workers_before']:,} -> {d['workers_after']:,} {alpha}, "
            f"{d['firms_before']:,} -> {d['firms_after']:,} {psi}, "
            f"{d['cut_workers']:,} articulation-point {alpha} removed in "
            f"{d['pruning_rounds']} round(s), "
            f"{d['single_observation_workers']:,} observed once")
        _log(verbose, message, logger)
        dropped = 1.0 - d["firms_after"] / max(d["firms_before"], 1)
        if dropped > 0.2:
            _warn(f"leave-one-out pruning removed {dropped:.0%} of {psi}; "
                  "the components below describe the surviving set, which is "
                  "the sample the leave-out parameters are identified on, not "
                  "the panel you supplied", logger)
        data = pruning.data

    # the streamed dimension is pinned so that the pair the sample was pruned on
    # is the pair the decomposition is about
    options.setdefault("stream", alpha)
    _check_leave_out(leave_out, se, stayers, centering, se_variance,
                     options.get("weights"))
    # only leaving out an observation reads the row-level fit's sorted rows;
    # leaving out a match reads its residual file and refits on the matches
    fit = feols_stream(fml, data, workdir=workdir,
                       keep_intermediates=(leave_out == "observation"),
                       verbose=verbose, logger=logger, **options)
    if not isinstance(fit, HDFEResult):
        raise ValueError(
            "this formula expands to several models; leave-out estimation needs "
            "a single one, so fit them separately")

    components = fit.leave_out_kss(n_draws=n_draws, seed=seed, block=block,
                                   psi=psi, leave_out=leave_out,
                                   stayers=stayers, se=se,
                                   se_draws=se_draws, se_trace=se_trace,
                                   diagnose=diagnose,
                                   diagnose_draws=diagnose_draws,
                                   weak_interval=weak_interval,
                                   confidence=confidence, centering=centering,
                                   se_variance=se_variance)
    components.pruning = pruning
    components.fit = fit
    return components

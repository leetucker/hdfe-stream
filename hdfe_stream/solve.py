"""Step 2: solve  S Gamma = D_o' M_0 V  for the non-streamed fixed effects."""

from __future__ import annotations

import time

import numba as nb
import numpy as np
import scipy.sparse as sp

from .kernels_base import (_nb_count_triples, _nb_csr_matmat,
                           _nb_emit_triples, _nb_rhs, _nb_stream_matvec)
from .kernels_slopes import (_nb_emit_triples_sl, _nb_rhs_sl,
                             _nb_stream_matvec_sl)
from .report import _warn
from .utils import _scatter


class _STooLarge(Exception):
    """The explicit reduced matrix S would exceed `max_s_gb`."""


# A covariate spanned by the *streamed* fixed effect -- one that is constant
# within every fe[0] group, say -- is annihilated by M_0, so its right-hand
# side b = D_o' M_0 V is zero up to round-off rather than exactly zero. The CG
# convergence test is relative, ||r|| <= tol * ||b||, which for such a column
# asks for a residual far below double precision: it can never be met, so the
# solve would run to `maxiter` and report failure for a column whose answer is
# already right (b = 0 means Gamma = 0 solves it). The column is dropped as
# collinear in step 3 anyway.
#
# The test is b's norm against the square root of the variable's own total sum
# of squares, which pass 0 has already computed, so it costs nothing. The
# measured separation is enormous: such a column comes in around 1e-16, while
# the smallest genuine column seen is around 0.3, so this threshold is nowhere
# near anything real.
#
# Note this is specific to the streamed dimension. A covariate collinear with a
# *non-streamed* fixed effect (i(year) alongside a year FE) has a perfectly
# large b; its degeneracy lives in the null space of S, where CG is already
# well behaved.
_RHS_ZERO_TOL = 1e-11


# Shared-state contract with the other mixins
# -------------------------------------------
# Reads, set by _PassesMixin: the identifying cells (starts, codes, n, sums,
# and st/ainv/cmat with varying slopes) plus n_levels, offs, obs, comp, diag.
# From StreamingHDFE: solver, precond, tol, maxiter, rhs_block, max_s_gb,
# triple_budget, dense_max_levels, paths.
#
# Sets nothing that the other mixins read: _solve() returns Gamma and its
# solver-info dict to the caller in estimator.py.


class _SolveMixin:

    # --------------------------------------------------------------- step 2
    def _pcg(self, matvec, precond, b, zero=None):
        """Block PCG: one independent CG per right-hand-side column, run in
        lockstep so every operator application covers all columns.

        `zero` marks columns whose right-hand side is numerically zero (see
        `_RHS_ZERO_TOL`); they are returned as the zero solution, which is what
        solves them, and take no part in the iteration.
        """
        bnorm = np.linalg.norm(b, axis=0)
        bnorm[bnorm == 0] = 1.0
        X = np.zeros_like(b)
        R = b.copy()
        if zero is not None and zero.any():
            R[:, zero] = 0.0            # solved already: X stays zero
            if zero.all():
                return X, 0, True, 0.0
        Z = precond(R)
        P = Z.copy()
        rz = np.einsum("ij,ij->j", R, Z)
        active = np.ones(b.shape[1], bool) if zero is None else ~zero
        rel = np.zeros(b.shape[1])
        it = 0
        for it in range(1, self.maxiter + 1):
            AP = matvec(P)
            pAp = np.einsum("ij,ij->j", P, AP)
            alpha = np.where(active & (pAp > 0), rz / np.where(pAp > 0, pAp, 1), 0.0)
            X += P * alpha
            R -= AP * alpha
            rel = np.linalg.norm(R, axis=0) / bnorm
            active = rel > self.tol
            if self.verbose and (it % 25 == 0 or not active.any()):
                self._log(f"  CG iter {it}: max rel. residual {rel.max():.2e}")
            if not active.any():
                break
            Z = precond(R)
            rz_new = np.einsum("ij,ij->j", R, Z)
            beta = np.where(active, rz_new / np.where(rz != 0, rz, 1), 0.0)
            P = Z + P * beta
            rz = rz_new
        return X, it, bool(not active.any()), float(rel.max())

    def _jacobi(self):
        pos = self.diag > 1e-12
        dinv = np.where(pos, 1.0 / np.where(pos, self.diag, 1.0), 0.0)
        return lambda R: R * dinv[:, None]

    def _build_explicit(self):
        """Form S = D_o' M_0 D_o as a CSR matrix in one parallel pass.

        Entries between two levels of *small* dimensions (at most
        `dense_max_levels` levels each, e.g. years) are accumulated in a dense
        block; everything else is emitted as COO triples. Groups are processed
        in chunks whose triples fit in max(triple_budget, nnz(S so far)); each
        chunk is converted to CSR (summing duplicates) and added to S, which
        keeps the total merge cost linear in the number of triples. Raises
        _STooLarge as soon as S exceeds `max_s_gb`.
        """
        off, L = self._offsets()
        G = len(self.starts) - 1
        doffs, nd = [], 0
        for f in self.o_fe:
            if self.n_levels[f] <= self.dense_max_levels and nd + self.n_levels[f] <= 4096:
                doffs.append(nd)
                nd += self.n_levels[f]
            else:
                doffs.append(-1)
        doffs = np.array(doffs, np.int64)
        nt = nb.get_num_threads()
        dense = np.zeros((nt, max(nd, 1), max(nd, 1)))
        cap = self.max_s_gb * 1e9

        def nbytes(S):
            return S.data.nbytes + S.indices.nbytes + S.indptr.nbytes

        S = sp.csr_matrix((L, L))
        t0 = time.time()
        n_triples = 0
        block = 2_000_000                  # groups per counting block
        for b0 in range(0, G, block):
            b1 = min(G, b0 + block)
            cnt = np.empty(b1 - b0, np.int64)
            _nb_count_triples(self.starts, self.codes, self.n, self.offs, doffs, b0, b1, cnt)
            cum = np.concatenate(([0], np.cumsum(cnt)))
            g0 = b0
            while g0 < b1:
                budget = max(self.triple_budget, S.nnz)
                base = cum[g0 - b0]
                g1 = b0 + int(np.searchsorted(cum, base + budget, side="right")) - 1
                g1 = min(max(g1, g0 + 1), b1)
                toff = cum[g0 - b0:g1 - b0 + 1] - base
                mt = int(toff[-1])
                R = np.empty(mt, np.int32)
                C = np.empty(mt, np.int32)
                V = np.empty(mt)
                if self.slope_vars:
                    _nb_emit_triples_sl(self.starts, self.codes, self.n, self.st, self.ainv,
                                        self.offs, doffs, g0, g1, toff, R, C, V, dense)
                else:
                    _nb_emit_triples(self.starts, self.codes, self.n, self.offs, doffs,
                                     g0, g1, toff, R, C, V, dense)
                if mt:
                    S = S + sp.coo_matrix((V, (R, C)), shape=(L, L)).tocsr()
                del R, C, V
                n_triples += mt
                g0 = g1
                if nbytes(S) > cap:
                    raise _STooLarge(
                        f"S exceeded max_s_gb={self.max_s_gb:.1f} after {g1:,} of {G:,} "
                        f"groups ({S.nnz:,} nonzeros)")
        if nd:
            dsum = dense.sum(axis=0)
            gidx = np.concatenate([off[f] + np.arange(self.n_levels[f])
                                   for f, d in zip(self.o_fe, doffs) if d >= 0])
            rr, cc = np.nonzero(dsum)
            S = S + sp.coo_matrix((dsum[rr, cc], (gidx[rr], gidx[cc])), shape=(L, L)).tocsr()
        S.sum_duplicates()
        S.sort_indices()
        mb = nbytes(S) / 1e6
        self._log(f"  explicit S: {L:,} x {L:,}, nnz {S.nnz:,} ({mb:,.0f} MB), "
                           f"{n_triples:,} triples + {nd}x{nd} dense block, "
                           f"{time.time()-t0:.1f}s")
        return S, {"S_nnz": int(S.nnz), "S_mb": round(mb, 1),
                   "S_build_seconds": round(time.time() - t0, 2)}

    def _make_block_solver(self):
        """Return (solve(b, zero) -> (X, iterations, converged, rel), info dict).

        The `within` solver takes a column range instead and has no `zero`
        argument; it does its own convergence handling.
        """
        solver = self.solver
        info = {}
        if solver in ("auto", "explicit"):
            try:
                S, sinfo = self._build_explicit()
                info.update(sinfo)
                solver = "explicit"
            except _STooLarge as err:
                if self.solver == "explicit":
                    raise MemoryError(f"{err}; raise max_s_gb or use solver='stream_cg'") from None
                self._log(f"  {err}; falling back to stream_cg")
                info["fallback"] = str(err)
                solver = "stream_cg"
        info["solver"] = solver

        if solver == "explicit":
            indptr, indices, data = S.indptr, S.indices, S.data

            def matvec(P):
                out = np.empty_like(P)
                _nb_csr_matmat(indptr, indices, data, np.ascontiguousarray(P), out)
                return out
            if self.precond == "amg":
                import pyamg
                t0 = time.time()
                ml = pyamg.smoothed_aggregation_solver(S, symmetry="symmetric", max_coarse=500)
                M = ml.aspreconditioner(cycle="V")
                info["amg_setup_seconds"] = round(time.time() - t0, 2)
                info["solver"] = "explicit+amg"

                def precond(R):
                    return np.column_stack([M @ R[:, j] for j in range(R.shape[1])])
            else:
                precond = self._jacobi()
            return (lambda b, zero=None: self._pcg(matvec, precond, b, zero)), info

        if solver == "stream_cg":
            nt = nb.get_num_threads()
            L = self.diag.shape[0]

            def matvec_s(P):
                acc = np.empty((nt, L, P.shape[1]))
                if self.slope_vars:
                    _nb_stream_matvec_sl(np.ascontiguousarray(P), self.starts, self.codes, self.n,
                                         self.st, self.ainv, self.offs, acc)
                else:
                    _nb_stream_matvec(np.ascontiguousarray(P), self.starts, self.codes, self.n,
                                      self.offs, acc)
                return acc.sum(axis=0)
            jac = self._jacobi()
            return (lambda b, zero=None: self._pcg(matvec_s, jac, b, zero)), info

        # within: full system on identifying cells, keep the non-streamed blocks
        import within
        sizes = np.diff(self.starts)
        design = np.asfortranarray(np.column_stack(
            [np.repeat(np.arange(len(sizes), dtype=np.uint32), sizes),
             np.asarray(self.codes)]).astype(np.uint32))
        nvec = np.asarray(self.n)
        off, tot = self._offsets()

        def solve_within(j0, j1):
            Y = np.asarray(self.sums[:, j0:j1]) / nvec[:, None]
            res = within.solve_batch(design, Y, weights=nvec,
                                     options=within.LsmrOptions(tol=self.tol, maxiter=self.maxiter))
            x, lay = np.asarray(res.x), res.layout
            Gm = np.zeros((tot, j1 - j0))
            for t, f in enumerate(self.o_fe, start=1):
                start, nl_seen = lay.index(t, 0, 0), lay.n_levels(t)
                Gm[off[f]:off[f] + nl_seen] = x[start:start + nl_seen]
            return Gm, list(res.iterations), bool(all(res.converged)), float("nan")
        return solve_within, info

    def _solve(self):
        """Solve for Gamma (levels x variables) in column blocks; the result
        is written to a memory-mapped .npy file."""
        off, L = self._offsets()
        m = self.m
        t0 = time.time()
        solve, info = self._make_block_solver()
        gamma = np.lib.format.open_memmap(self.paths["gamma"], mode="w+",
                                          dtype=np.float64, shape=(L, m))
        its, conv, rel = [], True, 0.0
        fe_spanned = []
        for j0 in range(0, m, self.rhs_block):
            j1 = min(m, j0 + self.rhs_block)
            if info["solver"] == "within":
                X, it, c, r = solve(j0, j1)
            else:
                b = np.zeros((L, j1 - j0))
                if self.slope_vars:
                    _nb_rhs_sl(self.starts, self.codes, self.sums[:, j0:j1], self.st, self.ainv,
                               self.cmat[:, :, j0:j1], self.offs, b)
                else:
                    _nb_rhs(self.starts, self.codes, self.n, self.sums[:, j0:j1], self.offs, b)
                # compare each right-hand side against the scale of its own
                # variable; see _RHS_ZERO_TOL. raw_ss is the variable's total
                # sum of squares, already computed in pass 0, so this costs
                # nothing beyond the norm of b itself.
                scale = np.sqrt(np.maximum(self.raw_ss[j0:j1], 0.0))
                zero = np.linalg.norm(b, axis=0) <= _RHS_ZERO_TOL * scale
                fe_spanned += [self.var_names[j0 + k] for k in np.flatnonzero(zero)]
                X, it, c, r = solve(b, zero)
            gamma[:, j0:j1] = self._normalize(X)
            its.append(it)
            conv &= c
            rel = max(rel, r)
        gamma.flush()
        info.update({"iterations": its if len(its) > 1 else its[0], "converged": conv,
                     "blocks": len(its), "seconds": round(time.time() - t0, 2)})
        if not np.isnan(rel):
            info["max_rel_residual"] = rel
        if fe_spanned:
            info["fe_spanned"] = fe_spanned
            self._log(f"  {len(fe_spanned)} variable(s) absorbed by {self.g_fe}: "
                      + ", ".join(fe_spanned))
        if not conv:
            _warn(f"solver did not converge within maxiter={self.maxiter}", self.logger)
        return np.load(self.paths["gamma"], mmap_mode="r"), info

    def _normalize(self, Gamma):
        """fe[1] effects have obs-weighted mean zero within each connected
        component; further dimensions have global obs-weighted mean zero. The
        constants move into the fe[0] effects; fitted values are unchanged."""
        off, _ = self._offsets()
        f1 = self.o_fe[0]
        nl, o = self.n_levels[f1], off[f1]
        w = self.obs[o:o + nl]
        cw = np.bincount(self.comp, weights=w, minlength=nl)
        cw[cw == 0] = 1.0
        mean = _scatter(self.comp, w[:, None] * Gamma[o:o + nl], nl) / cw[:, None]
        Gamma[o:o + nl] -= mean[self.comp]
        for f in self.o_fe[1:]:
            nl, o = self.n_levels[f], off[f]
            w = self.obs[o:o + nl]
            Gamma[o:o + nl] -= (w @ Gamma[o:o + nl]) / max(w.sum(), 1.0)
        return Gamma

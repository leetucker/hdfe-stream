"""Step 2: solve  S Gamma = D_o' M_0 V  for the non-streamed fixed effects."""

from __future__ import annotations

import time

import numba as nb
import numpy as np
import scipy.sparse as sp

from .kernels_base import (_nb_csr_matmat, _nb_level_groups, _nb_rhs,
                           _nb_row_fill, _nb_row_nnz, _nb_row_work,
                           _nb_stream_matvec)
from .kernels_slopes import _nb_rhs_sl, _nb_stream_matvec_sl
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
# paths.
#
# Sets nothing that the other mixins read: _solve() returns Gamma and its
# solver-info dict to the caller in estimator.py.


class _SolveMixin:

    # --------------------------------------------------------------- step 2
    def _pcg(self, matvec, precond, b, zero=None, x0=None):
        """Block PCG: one independent CG per right-hand-side column, run in
        lockstep so every operator application covers all columns.

        `zero` marks columns whose right-hand side is numerically zero (see
        `_RHS_ZERO_TOL`); they are returned as the zero solution, which is what
        solves them, and take no part in the iteration.

        `x0` is a starting guess (the GLM passes the previous IRLS step's
        solution); columns it already solves to `tol` take no iterations.
        """
        bnorm = np.linalg.norm(b, axis=0)
        bnorm[bnorm == 0] = 1.0
        if x0 is None:
            X = np.zeros_like(b)
            R = b.copy()
        else:
            X = np.array(x0, dtype=np.float64, order="C")
            R = b - matvec(X)
        if zero is not None and zero.any():
            X[:, zero] = 0.0
            R[:, zero] = 0.0            # solved already: X stays zero
            if zero.all():
                return X, 0, True, 0.0
        active = np.ones(b.shape[1], bool) if zero is None else ~zero
        rel = np.zeros(b.shape[1])
        if x0 is not None:
            rel = np.linalg.norm(R, axis=0) / bnorm
            active &= rel > self.tol
            if not active.any():
                return X, 0, True, float(rel.max())
        Z = precond(R)
        P = Z.copy()
        rz = np.einsum("ij,ij->j", R, Z)
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
                self._debug(f"  CG iter {it}: max rel. residual {rel.max():.2e}")
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
        """Form S = D_o' M_0 D_o as a CSR matrix, row by row in parallel.

        Each row's columns are counted first, so S is allocated once at its
        final size, and nothing larger than S is held while it is built
        besides an index of the groups each level is in. Raises _STooLarge,
        before allocating S, if it would exceed `max_s_gb`.
        """
        _, L = self._offsets()
        t0 = time.time()
        lptr, lgrp = _nb_level_groups(self.starts, self.codes, self.offs, L)
        work = np.empty(L, np.int64)
        _nb_row_work(self.starts, lptr, lgrp, work)
        # blocks of rows with similar work, several per thread to balance them
        cum = np.cumsum(work)
        cuts = np.searchsorted(cum, np.linspace(0, cum[-1], 16 * nb.get_num_threads() + 1)[1:-1])
        bounds = np.unique(np.concatenate(([0], cuts, [L]))).astype(np.int64)
        wide = max(L // (4 * self.codes.shape[1]), 1)
        rnnz = np.empty(L, np.int64)
        _nb_row_nnz(self.starts, self.codes, self.offs, lptr, lgrp, work, wide, bounds, rnnz)
        nnz = int(rnnz.sum())
        idx = np.int32 if max(nnz, L) < 2**31 else np.int64
        nbytes = nnz * (8 + np.dtype(idx).itemsize) + (L + 1) * np.dtype(idx).itemsize
        if nbytes > self.max_s_gb * 1e9:
            raise _STooLarge(f"S would take {nbytes / 1e9:.3g} GB > max_s_gb="
                             f"{self.max_s_gb:.3g} ({nnz:,} nonzeros)")
        indptr = np.zeros(L + 1, idx)
        np.cumsum(rnnz, out=indptr[1:])
        del rnnz, cum
        indices = np.empty(nnz, idx)
        data = np.empty(nnz)
        if self.slope_vars:
            st, ainv, ng = self.st, self.ainv, np.zeros(0)
        else:
            st, ainv = self.n.reshape(-1, 1), np.zeros((0, 1, 1))
            ng = np.add.reduceat(self.n, self.starts[:-1])
        _nb_row_fill(self.starts, self.codes, self.n, st, ainv, ng, self.offs, lptr, lgrp,
                     work, wide, bounds, indptr, indices, data)
        S = sp.csr_matrix((data, indices, indptr), shape=(L, L), copy=False)
        S.has_sorted_indices = True
        mb = nbytes / 1e6
        self._log(f"  explicit S: {L:,} x {L:,}, nnz {nnz:,} ({mb:,.0f} MB), "
                  f"{time.time()-t0:.1f}s")
        return S, {"S_nnz": nnz, "S_mb": round(mb, 1),
                   "S_build_seconds": round(time.time() - t0, 2)}

    def _make_block_solver(self):
        """Return (solve(b, zero, x0) -> (X, iterations, converged, rel), info
        dict).

        The `within` solver takes the block's cell sums instead and has no
        `zero` or `x0` argument; it does its own convergence handling.
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
            return (lambda b, zero=None, x0=None: self._pcg(matvec, precond, b, zero, x0)), info

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
            return (lambda b, zero=None, x0=None: self._pcg(matvec_s, jac, b, zero, x0)), info

        # within: full system on identifying cells, keep the non-streamed blocks
        import within
        sizes = np.diff(self.starts)
        design = np.asfortranarray(np.column_stack(
            [np.repeat(np.arange(len(sizes), dtype=np.uint32), sizes),
             np.asarray(self.codes)]).astype(np.uint32))
        nvec = np.asarray(self.n)
        off, tot = self._offsets()

        def solve_within(sums):
            Y = np.asarray(sums) / nvec[:, None]
            res = within.solve_batch(design, Y, weights=nvec,
                                     options=within.LsmrOptions(tol=self.tol, maxiter=self.maxiter))
            x, lay = np.asarray(res.x), res.layout
            Gm = np.zeros((tot, Y.shape[1]))
            for t, f in enumerate(self.o_fe, start=1):
                start, nl_seen = lay.index(t, 0, 0), lay.n_levels(t)
                Gm[off[f]:off[f] + nl_seen] = x[start:start + nl_seen]
            return Gm, list(res.iterations), bool(all(res.converged)), float("nan")
        return solve_within, info

    def _solve(self, sums=None, names=None, raw_ss=None, x0=None, in_memory=False):
        """Solve for Gamma (levels x variables) in column blocks; the result
        is written to a memory-mapped .npy file.

        By default the variables are the estimator's own, from the cell sums
        of passes 1/1b. The GLM passes its working regression's instead:
        `sums` (cells x variables), their `names` and total sums of squares
        `raw_ss`, the previous step's solution `x0` as a starting guess, and
        `in_memory` to keep Gamma out of the run directory.
        """
        off, L = self._offsets()
        if sums is None:
            sums, names, raw_ss = self.sums, self.var_names, self.raw_ss
        m = sums.shape[1]
        t0 = time.time()
        if L == 0:
            # No dimension besides the streamed one (or none at all): the
            # reduced system is empty, and demeaning by fe[0] is all there is.
            return np.zeros((0, m)), {"solver": "none", "iterations": 0, "converged": True,
                                      "blocks": 0, "seconds": 0.0}
        solve, info = self._make_block_solver()
        gamma = (np.empty((L, m)) if in_memory else
                 np.lib.format.open_memmap(self.paths["gamma"], mode="w+",
                                           dtype=np.float64, shape=(L, m)))
        its, conv, rel = [], True, 0.0
        fe_spanned = []
        for j0 in range(0, m, self.rhs_block):
            j1 = min(m, j0 + self.rhs_block)
            if info["solver"] == "within":
                X, it, c, r = solve(sums[:, j0:j1])
            else:
                b = np.zeros((L, j1 - j0))
                if self.slope_vars:
                    _nb_rhs_sl(self.starts, self.codes, sums[:, j0:j1], self.st, self.ainv,
                               self.cmat[:, :, j0:j1], self.offs, b)
                else:
                    _nb_rhs(self.starts, self.codes, self.n, sums[:, j0:j1], self.offs, b)
                # compare each right-hand side against the scale of its own
                # variable; see _RHS_ZERO_TOL. raw_ss is the variable's total
                # sum of squares, already computed in pass 0, so this costs
                # nothing beyond the norm of b itself.
                scale = np.sqrt(np.maximum(raw_ss[j0:j1], 0.0))
                zero = np.linalg.norm(b, axis=0) <= _RHS_ZERO_TOL * scale
                fe_spanned += [names[j0 + k] for k in np.flatnonzero(zero)]
                X, it, c, r = solve(b, zero, None if x0 is None else x0[:, j0:j1])
            gamma[:, j0:j1] = self._normalize(X)
            its.append(it)
            conv &= c
            rel = max(rel, r)
        if in_memory:
            return gamma, self._solve_info(info, its, conv, rel, fe_spanned, t0)
        gamma.flush()
        return (np.load(self.paths["gamma"], mmap_mode="r"),
                self._solve_info(info, its, conv, rel, fe_spanned, t0))

    def _solve_info(self, info, its, conv, rel, fe_spanned, t0):
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
        return info

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

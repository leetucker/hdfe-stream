"""Weak-identification diagnostic for the leave-out variance components.

The normal interval reported with `se=True` rests on Theorem 2 condition (ii) of
Kline, Saggio and Solvsten (2020):

    lambda_1^2 / sum_l lambda_l^2  =  o(1)

for the eigenvalues of A~ = S^(-1/2) Q S^(-1/2), equivalently of S^- Q. When a
few eigenvalues dominate -- "bottlenecks" in the mobility network, where the
estimate leans heavily on a handful of poorly estimated linear combinations of
the effects -- the estimator is no longer approximately normal and that interval
undercovers. KSS's threshold rule (Section 6.2) takes q to be the number of
leading eigenvalues with lambda_l^2 / sum lambda^2 >= 1/10; only q = 0 justifies
the normal interval. This module computes those ratios, q, and the first-stage F
of the leading direction. It does not compute the weak-identification interval
itself (KSS Section 6); see prototypes/WEAKID_NOTES.md.

How the eigenvalues are found
-----------------------------
Lanczos on the operator Q S^-, which is self-adjoint in the inner product
<x, y> = x' S^- y. In that form each iteration costs exactly one solve and no
products with S, so no extra passes over the rows: alpha_j = y_j' Q y_j with
y_j = S^- x_j already in hand, and the normalizing solve for the next vector is
the solve the next iteration needs anyway. The three components run in lockstep,
one column each, so an iteration is one solve call with three right-hand sides.

The recurrence keeps three vectors, not a basis. Coefficient vectors include the
streamed dimension, so a stored Krylov basis of fifty of them would cost fifty
worker-sized arrays. The price of not storing it is loss of orthogonality, which
shows up as "ghost" copies of eigenvalues that have converged. Those are handled
the standard way (Cullum and Willoughby): copies that agree are merged, and a
lone Ritz value that is also an eigenvalue of the tridiagonal with its first row
and column removed is spurious and dropped. What single-vector Lanczos cannot do
is see an *exactly* repeated eigenvalue twice -- one starting vector spans one
copy -- so q >= 2 can be understated in a perfectly symmetric design. The q = 0
versus q >= 1 decision depends only on lambda_1 and is unaffected.

The eigenvector, needed for the F statistic, is rebuilt by running the same
recurrence a second time and accumulating the Ritz combination. The arithmetic is
deterministic, so the second run retraces the first exactly, and it costs one
extra coefficient vector rather than a stored basis.

sum lambda^2 = tr((S^- Q)^2) is estimated by Hutchinson, with the random vectors
restricted to the blocks Q reads. That restriction is exact -- S^- Q S^- Q has
zero columns everywhere else -- and removes a term that is pure noise.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ._columns import GCODE, vcol
from .kernels_inverse import _nb_group_sums, _nb_rademacher, _nb_reduce_rows
from .kernels_se import _nb_readout
from .leaveout_se import COMPONENTS
from .report import _log, _warn

#: KSS Section 6.2: a squared eigenvalue ratio at or above this counts toward q
WEAK_ID_THRESHOLD = 0.1


@dataclass
class WeakIdDiagnostics:
    """The eigenvalue diagnostic for the three components.

    For each component name:

        eigenvalues   the leading eigenvalues of S^- Q, largest in magnitude
                      first (signed: the covariance's can be negative)
        sum_sq        sum of all squared eigenvalues, tr((S^- Q)^2), estimated
        sum_sq_se     Monte Carlo standard error of that estimate
        ratios        lambda_l^2 / sum_sq for the leading eigenvalues
        q             KSS's threshold rule; 0 means the normal interval is
                      justified. When every reported ratio clears the threshold
                      the true q may be larger, which `q_capped` records.
        q_borderline  a ratio lies within two Monte Carlo standard errors of the
                      threshold, so q could go either way at this draw budget;
                      raise `n_draws` to resolve it
        f_stat        b1^2 / V[b1] for the leading direction: how well its
                      nuisance parameter is identified. Large values mean the
                      delta-method (normal) interval is less at risk even when
                      q >= 1 (KSS Remark 10).
        bounds        Lanczos error bound on each reported eigenvalue
        converged     whether the leading two eigenvalues met the tolerance
    """

    eigenvalues: dict
    sum_sq: dict
    sum_sq_se: dict
    ratios: dict
    q: dict
    q_capped: dict
    q_borderline: dict
    f_stat: dict
    bounds: dict
    converged: dict
    threshold: float = WEAK_ID_THRESHOLD
    diagnostics: dict = field(default_factory=dict)
    # The leading eigenvector of each component, S-normalized (u' S u = 1), as a
    # coefficient triple with one column per component. The weak-identification
    # interval needs it; it is coefficient-sized, so it is kept out of the repr.
    lead_vectors: tuple = field(default=None, repr=False)

    @property
    def weakly_identified(self) -> list[str]:
        """Components for which the normal interval is not justified (q >= 1)."""
        return [name for name in COMPONENTS if self.q[name] >= 1]

    def summary(self) -> str:
        lines = [f"weak-identification diagnostic (KSS threshold "
                 f"{self.threshold:g} on lambda^2 / sum lambda^2)",
                 f"{'component':<18}{'lam1^2/sum':>11}{'lam2^2/sum':>11}"
                 f"{'lam3^2/sum':>11}{'q':>5}{'F':>9}  normal interval",
                 "-" * 82]
        for name in COMPONENTS:
            r = list(self.ratios[name]) + [float("nan")] * 3
            q = (f"{'~' if self.q_borderline[name] else ''}{self.q[name]}"
                 f"{'+' if self.q_capped[name] else ''}")
            verdict = "justified" if self.q[name] == 0 else "NOT justified"
            if not self.converged[name]:
                verdict += " (unconverged)"
            lines.append(f"{name:<18}{r[0]:>11.4f}{r[1]:>11.4f}{r[2]:>11.4f}"
                         f"{q:>5}{self.f_stat[name]:>9.2f}  {verdict}")
        if any(self.q_borderline.values()):
            lines.append("~ a ratio is within two Monte Carlo standard errors of "
                         "the threshold; raise n_draws to settle q")
        return "\n".join(lines)


class _WeakIdMixin:
    # Defined by other mixins:
    #   _StandardErrorMixin: _se_layout, _apply_q, _inverse_root, _outcome_mean,
    #                        _fit_smoother, _smooth_at
    #   _InverseMixin:       apply_inverse, rademacher, _chunks,
    #                        _covariate_matrix, _scratch_capped
    #   _LeaveOutMixin:      _model_covariates, _quiet_solver
    #   StreamingHDFE:       g_fe, offs, vidx, rhs_block, scratch_mb, verbose,
    #                        logger, log_level

    # --------------------------------------------------------------- lanczos
    def _q_block(self, components, U_levels, U_groups, L):
        """Q applied column by column, column j using component components[j]."""
        out_l = np.zeros_like(U_levels)
        out_g = np.zeros_like(U_groups)
        for k in range(3):
            cols = np.flatnonzero(np.asarray(components) == k)
            if len(cols) == 0:
                continue
            sub_l = np.zeros((U_levels.shape[0], len(cols)))
            sub_g = np.zeros((U_groups.shape[0], len(cols)))
            self._apply_q(k, np.ascontiguousarray(U_levels[:, cols]),
                          np.ascontiguousarray(U_groups[:, cols]), L,
                          sub_l, sub_g)
            out_l[:, cols] = sub_l
            out_g[:, cols] = sub_g
        return out_l, out_g

    @staticmethod
    def _dot(a, b):
        """Column-wise inner products over the three coefficient blocks."""
        return sum((x * y).sum(axis=0) for x, y in zip(a, b))

    def _krylov_plan(self, L, max_iter):
        """Per component: the most steps worth taking, and whether to store the
        basis and reorthogonalize fully.

        The Krylov space of each component is bounded by the non-streamed
        dimension, however many workers there are:

          var(psi)   rank Q_psi = n_psi - 1
          cov        rank Q_cov <= 2 (n_psi - 1)
          var(alpha) the streamed block's spectrum is one bulk eigenvalue, 1/T,
                     plus a part of rank at most n_levels + n_cov driven by the
                     other dimensions, so at most that many distinct values plus
                     one -- and one starting vector sees each distinct value once

        Running past that bound divides by a residual that is only noise, and
        the recurrence explodes. So steps are capped there, and when the basis a
        component would need fits the scratch budget -- which is exactly the
        small-space case where exhaustion is reached -- it is stored and every
        new vector is reorthogonalized against it. Large spaces never get near
        exhaustion within `max_iter`, and there the three-term recurrence with
        Cullum-Willoughby filtering is the standard, safe choice.
        """
        n_psi, n_groups = L["n_psi"], L["n_groups"]
        n_other = L["total_levels"] + len(L["columns"])
        dims = [max(n_psi - 1, 0), max(n_other + 1, 1),
                max(2 * (n_psi - 1), 0)]
        support = [n_psi, n_groups, n_psi + n_groups]
        budget = int(getattr(self, "scratch_mb", 0) or 32) << 20
        caps = [min(max_iter, d) for d in dims]
        reorth = [cap * 2 * rows * 8 <= budget for cap, rows in zip(caps, support)]
        return caps, reorth

    def _lanczos(self, L, covariates, n_iter, seed, ritz=None, check=None,
                 every=20, caps=None, reorth=None):
        """Run the recurrence for all three components in lockstep.

        With `ritz=None`, returns (alphas, betas, lengths, steps): per component
        the tridiagonal's diagonal, its off-diagonal (one longer, the last being
        the residual norm the error bounds need), how many steps ran before it
        stopped, and how many steps ran in all. `check(alphas, betas, lengths,
        steps)` is consulted every `every` steps and stops the run when it
        returns True.

        With `ritz` a list of three (m_c, k) coefficient arrays, runs exactly
        `n_iter` steps and returns (ritz_x, ritz_y): coefficient triples with 3k
        columns, column c*k + l holding sum_j ritz[c][j, l] x_j (respectively
        y_j). The arithmetic is deterministic, so this retraces the first run.

        `caps` limits each component's steps and `reorth` says whether to keep
        its basis for full reorthogonalization; see `_krylov_plan`.
        """
        comps = [0, 1, 2]
        caps = [n_iter] * 3 if caps is None else list(caps)
        reorth = [False] * 3 if reorth is None else list(reorth)
        n_cov = len(L["columns"])
        shapes = [(L["total_levels"], 3), (n_cov, 3), (L["n_groups"], 3)]
        psi_slice = slice(L["psi_offset"], L["psi_offset"] + L["n_psi"])

        def support(c, blocks):
            """The part of a coefficient triple where component c's x lives."""
            parts = []
            if c in (0, 2):
                parts.append(blocks[0][psi_slice, c])
            if c in (1, 2):
                parts.append(blocks[2][:, c])
            return np.concatenate(parts) if len(parts) > 1 else parts[0].copy()

        def project(blocks):
            """Put each column back onto range(Q_c): zero mean on its blocks.

            Q_psi 1 = 0 and Q_alpha 1 = 0, so range(Q) is the sum-zero vectors
            on each block the component touches -- and those are in range(S).
            Rounding leaves a trace of the constant, which is invisible to the
            <., .>_{S^-} inner product (the solve drops null-space components
            before it starts), so orthogonalization can never remove it; and
            subtracting stored basis vectors re-injects every one of their
            traces, which compounds. Measured without this, the constant grows
            about 2.5x a step and wrecks a fully reorthogonalized run by step 40.
            Removing it each step keeps it at rounding level for good.
            """
            for c in (0, 2):
                col = blocks[0][psi_slice, c]
                col -= col.mean()
            for c in (1, 2):
                col = blocks[2][:, c]
                col -= col.mean()

        def scatter(c, flat, blocks):
            """Subtract a support-shaped vector back into a coefficient triple."""
            at = 0
            if c in (0, 2):
                n = L["n_psi"]
                blocks[0][psi_slice, c] -= flat[at:at + n]
                at += n
            if c in (1, 2):
                blocks[2][:, c] -= flat[at:]

        # start in range(Q), which lies in range(S): x0 = Q r for random r
        r = [np.empty(s) for s in shapes]
        position = 0
        for block in r:
            _nb_rademacher(position, block.shape[0], 3, seed, 0, block)
            position += block.shape[0]
        ql, qg = self._q_block(comps, r[0], r[2], L)
        x = [ql, np.zeros(shapes[1]), qg]
        project(x)
        y = list(self.apply_inverse(x[0], x[1], x[2], covariates=covariates))
        norm = np.sqrt(np.maximum(self._dot(x, y), 0.0))
        live = (norm > 0) & (np.array(caps) > 0)
        for block in (*x, *y):
            block[:, live] /= norm[live]
            block[:, ~live] = 0.0
        x_prev = [np.zeros(s) for s in shapes]
        beta_prev = np.zeros(3)
        basis = [([], []) for _ in comps]     # (x supports, y supports)

        alphas = [[] for _ in comps]
        betas = [[] for _ in comps]
        lengths = np.zeros(3, dtype=int)
        if ritz is None:
            acc = None
        else:
            k = ritz[0].shape[1]
            wide = [(s[0], 3 * k) for s in shapes]
            acc = ([np.zeros(s) for s in wide], [np.zeros(s) for s in wide])
        beta_max = np.zeros(3)

        steps = 0
        for j in range(n_iter):
            if not live.any():
                break
            if (check is not None and j > 0 and j % every == 0
                    and check([np.array(a) for a in alphas],
                              [np.array(b) for b in betas], lengths, j)):
                break
            steps = j + 1
            if acc is not None:
                for c in np.flatnonzero(live):
                    if j < len(ritz[c]):
                        cols = slice(c * k, (c + 1) * k)
                        for ax, ay, xb, yb in zip(acc[0], acc[1], x, y):
                            ax[:, cols] += np.outer(xb[:, c], ritz[c][j])
                            ay[:, cols] += np.outer(yb[:, c], ritz[c][j])
            for c in np.flatnonzero(live):
                if reorth[c]:
                    basis[c][0].append(support(c, x))
                    basis[c][1].append(support(c, y))

            ql, qg = self._q_block(comps, y[0], y[2], L)
            w = [ql, np.zeros(shapes[1]), qg]
            for wb, xp in zip(w, x_prev):
                wb -= beta_prev * xp
            alpha = self._dot(w, y)
            for wb, xb in zip(w, x):
                wb -= alpha * xb
            # one step of local reorthogonalization against x_j: a dot product,
            # and it keeps the three-term recurrence honest for longer
            fix = self._dot(w, y)
            for wb, xb in zip(w, x):
                wb -= fix * xb
            alpha = alpha + fix
            project(w)
            # full reorthogonalization where the basis is kept; twice is enough
            for c in np.flatnonzero(live):
                if not reorth[c] or not basis[c][0]:
                    continue
                for _sweep in range(2):
                    wc = support(c, w)
                    Xs = np.column_stack(basis[c][0])
                    Ys = np.column_stack(basis[c][1])
                    scatter(c, Xs @ (Ys.T @ wc), w)
            project(w)

            z = list(self.apply_inverse(w[0], w[1], w[2], covariates=covariates))
            beta = np.sqrt(np.maximum(self._dot(w, z), 0.0))
            beta_max = np.maximum(beta_max, beta)

            for c in np.flatnonzero(live):
                alphas[c].append(float(alpha[c]))
                betas[c].append(float(beta[c]))
                lengths[c] = j + 1
            # stop a component when its Krylov space is exhausted -- the
            # residual has fallen to noise, and the eigenvalues found are exact
            # to that level -- or when it reaches its dimension bound
            live = (live & (beta > 1e-8 * np.maximum(beta_max, 1e-300))
                    & (lengths < np.array(caps)))

            x_prev = x
            x = [wb.copy() for wb in w]
            y = z
            for block in (*x, *y):
                block[:, live] /= beta[live]
                block[:, ~live] = 0.0
            beta_prev = np.where(live, beta, 0.0)

        if acc is not None:
            return acc
        return ([np.array(a) for a in alphas], [np.array(b) for b in betas],
                lengths, steps)

    @staticmethod
    def _ritz_top(alpha, beta, top):
        """Genuine leading Ritz values, their error bounds and eigenvectors.

        `beta` carries one more entry than the tridiagonal uses: the residual
        norm after the last step, which scales the error bounds.
        """
        from scipy.linalg import eigh_tridiagonal, eigvalsh_tridiagonal

        m = len(alpha)
        if m == 0:
            return np.array([]), np.array([]), np.zeros((0, 0))
        off = beta[:m - 1]
        theta, S = eigh_tridiagonal(alpha, off)
        bounds = np.abs(beta[m - 1] * S[-1, :])
        norm = max(float(np.abs(theta).max()), 1e-300)

        # Ghosts: after an eigenvalue converges, loss of orthogonality produces
        # further copies of it. They agree with each other to the convergence
        # level, far tighter than any two genuine eigenvalues of a real design,
        # so values within merge_tol are one eigenvalue. Keep the copy with the
        # smallest error bound.
        merge_tol = 1e-8 * norm
        order = np.argsort(theta)
        keep = []
        for idx in order:
            if keep and abs(theta[idx] - theta[keep[-1]]) <= merge_tol:
                if bounds[idx] < bounds[keep[-1]]:
                    keep[-1] = idx
                continue
            keep.append(idx)
        multiplicity = {}
        for idx in order:
            head = min(keep, key=lambda k: abs(theta[k] - theta[idx]))
            multiplicity[head] = multiplicity.get(head, 0) + 1

        # Cullum-Willoughby: a *lone* Ritz value that is also an eigenvalue of
        # the tridiagonal with its first row and column deleted is spurious.
        # Repeated ones are genuine by construction, so the test applies only to
        # singletons.
        if m > 1:
            hat = eigvalsh_tridiagonal(alpha[1:], off[1:]) if m > 2 else alpha[1:]
            spur_tol = 1e-11 * norm
            keep = [k for k in keep
                    if multiplicity[k] > 1
                    or np.min(np.abs(hat - theta[k])) > spur_tol]

        keep = sorted(keep, key=lambda k: -abs(theta[k]))[:top]
        return theta[keep], bounds[keep], S[:, keep]

    # ---------------------------------------------------------------- the sum
    @staticmethod
    def _s_orthonormalize(X, Y, k):
        """Make each component's k Ritz vectors orthonormal in <., .>_{S^-}.

        <x_a, x_b> = x_a' S^- x_b = x_a' y_b. Loss of orthogonality in the
        three-term runs leaves them only approximately so, and the deflation
        identity needs them exactly so. Modified Gram-Schmidt; a vector that
        comes out numerically zero -- a component with fewer than k eigenvalues
        -- is zeroed, which deflates less but keeps the identity exact.
        """
        for c in range(3):
            base = c * k
            for a in range(k):
                ca = base + a
                for p in range(a):
                    cp = base + p
                    coef = sum(xb[:, ca] @ yb[:, cp] for xb, yb in zip(X, Y))
                    for xb, yb in zip(X, Y):
                        xb[:, ca] -= coef * xb[:, cp]
                        yb[:, ca] -= coef * yb[:, cp]
                norm2 = sum(xb[:, ca] @ yb[:, ca] for xb, yb in zip(X, Y))
                lead = sum(xb[:, base] @ yb[:, base] for xb, yb in zip(X, Y))
                if norm2 <= 1e-20 * max(lead, 1e-300) or norm2 <= 0:
                    for block in (*X, *Y):
                        block[:, ca] = 0.0
                    continue
                scale = 1.0 / np.sqrt(norm2)
                for block in (*X, *Y):
                    block[:, ca] *= scale
        return X, Y

    def _sum_sq_eigenvalues(self, L, covariates, n_draws, block, seed, X, Y, k,
                            settled=None, min_draws=16):
        """sum lambda^2 = tr((S^- Q)^2) per component, deflated Hutchinson.

        Write A = Q S^-, self-adjoint in <x, y> = x' S^- y with the same
        spectrum. For any x_1..x_k orthonormal in that inner product and
        zeta with E[zeta zeta'] = S,

            tr(A^2) = sum_l ||A x_l||^2 + E <A zeta_def, A zeta_def>

        where zeta_def removes the x_l components. That holds exactly for *any*
        such x_l; with the Ritz vectors Lanczos found, the first term is almost
        all of the answer and is computed exactly, leaving Monte Carlo only the
        small remainder. The plain estimator is worst precisely when one
        eigenvalue dominates -- relative error sqrt(2/n_draws) -- which is when
        this diagnostic matters, and on a two-block bottleneck it was off by up
        to 160%. Deflated, the same case is essentially exact.

        zeta = X~' xi for row-space Rademacher xi has E[zeta zeta'] = S, so the
        draws are white in the metric where A is symmetric; the estimate is a
        squared norm, never negative, with variance at most 2 sum lambda^4. One
        reduction pass and one solve per block serve all three components.

        `n_draws` is a ceiling. With `settled(estimates, se)` given, draws stop
        as soon as it returns True, once at least `min_draws` are in: Monte
        Carlo is spent only where it could change the answer.

        Returns (estimates, standard errors of the Monte Carlo part, draws used).
        """
        import numba as nb

        n_threads = nb.get_num_threads()
        n_cov = len(L["columns"])
        comps = [c for c in range(3) for _ in range(k)]

        # the exact part: ||A x_l||^2 = (Q y_l)' S^- (Q y_l)
        ql, qg = self._q_block(comps, Y[0], Y[2], L)
        z = self.apply_inverse(ql, np.zeros((n_cov, 3 * k)), qg,
                               covariates=covariates)
        exact_cols = (ql * z[0]).sum(0) + (qg * z[2]).sum(0)
        exact = exact_cols.reshape(3, k).sum(axis=1)

        draws = []
        drawn = 0
        # each draw fans out to one column per component, and those columns
        # are coefficient-sized -- worker-sized -- so keep the total width at
        # `block`, the bound the rest of the library holds its solves to
        size = max(1, block // 3)
        while drawn < n_draws:
            m = min(size, n_draws - drawn)
            vectors = self.rademacher(seed=seed, first_draw=drawn)
            acc = np.zeros((n_threads, L["total_levels"], m))
            acc_x = np.zeros((n_threads, n_cov, m))
            gsum = np.zeros((L["n_groups"], m))
            for ordinal, chunk, starts, codes, w in self._chunks(L["columns"]):
                _root, inv = self._inverse_root(w)
                xi = np.ascontiguousarray(
                    vectors(ordinal, len(w), m) * inv[:, None])
                X_cov = self._covariate_matrix(chunk, L["columns"], len(w))
                _nb_reduce_rows(starts, codes, self.offs, w, X_cov, xi, acc,
                                acc_x)
                first = int(chunk[GCODE][0])
                _nb_group_sums(starts, w, xi, gsum[first:first + len(starts) - 1])
            zeta = [acc.sum(axis=0), acc_x.sum(axis=0), gsum]
            y = self.apply_inverse(zeta[0], zeta[1], zeta[2],
                                   covariates=covariates)

            # deflate, per component: y_def = y - sum_l (zeta' y_l) y_l
            y_def = [np.empty((b.shape[0], 3 * m)) for b in y]
            for c in range(3):
                cols = slice(c * k, (c + 1) * k)
                proj = sum(zb.T @ Yb[:, cols] for zb, Yb in zip(zeta, Y))
                out = slice(c * m, (c + 1) * m)
                for yd, yb, Yb in zip(y_def, y, Y):
                    yd[:, out] = yb - Yb[:, cols] @ proj.T

            comps_m = [c for c in range(3) for _ in range(m)]
            ql, qg = self._q_block(comps_m, y_def[0], y_def[2], L)
            b = self.apply_inverse(ql, np.zeros((n_cov, 3 * m)), qg,
                                   covariates=covariates)
            est = (ql * b[0]).sum(axis=0) + (qg * b[2]).sum(axis=0)
            draws.append(est.reshape(3, m))
            drawn += m

            values = np.hstack(draws)
            if (settled is not None and drawn >= min_draws and drawn < n_draws
                    and settled(*self._summarize_draws(exact, values))):
                break

        estimate, se = self._summarize_draws(exact, np.hstack(draws))
        return estimate, se, drawn

    @staticmethod
    def _summarize_draws(exact, values):
        n = values.shape[1]
        se = (values.std(axis=1, ddof=1) / np.sqrt(n) if n > 1
              else np.full(values.shape[0], np.nan))
        return exact + values.mean(axis=1), se

    # ------------------------------------------------------------ assembling
    def weak_id_diagnostics(self, result, sigma2, leverages, n_draws=64,
                            block=None, seed=0, covariates=None, psi=None,
                            max_iter=150, tol=1e-4, top=3, warn=True,
                            smooth_bins=None):
        """KSS's eigenvalue diagnostic for the three variance components.

        `result` is the fit, read for the outcome; `sigma2` and `leverages` the
        per-row leave-out variance and leverages `leave_out_components` builds.
        They give the variance of the leading direction's estimate, for the F
        statistic, from sigma2 smoothed against the leverage in `smooth_bins`
        quantile bins (by default n^(1/3), the reference's bandwidth). Smoothing is not optional: V[b1] is linear in sigma2, so
        the raw leave-out values are unbiased for it, but they are often
        negative and the leading direction can load on a few rows, and on a
        well-connected 25,000-row panel that sum came out negative for all three
        components. `n_draws` is the Hutchinson budget
        for sum lambda^2, whose relative error is at most sqrt(2 r / n_draws)
        for a leading ratio r -- under 6% at the threshold with the default.

        Lanczos runs to `max_iter` steps, stopping early once the two leading
        eigenvalues of every component have error bounds below `tol` relative to
        the largest. The default is deliberately loose: the output is a ratio
        judged against 1/10 whose denominator carries Monte Carlo error near 1%,
        and measured on a 1.3M-row panel, 1e-4 stops at 30 steps with
        eigenvalues within 6e-5 of a fully converged run, where 1e-6 needs 40
        steps and 1e-10 needs 120. Each step is one solve.

        With `warn=True`, components whose normal interval is not justified
        (q >= 1) are reported through the logger or the warnings module.
        """
        covariates = self._model_covariates(covariates)
        block = self.rhs_block if block is None else int(block)
        if n_draws < 2:
            raise ValueError("n_draws must be at least 2, so the Monte Carlo "
                             "error of sum lambda^2 can be reported")
        L = self._se_layout(covariates, psi)
        names = list(COMPONENTS)

        _log(self.verbose,
             f"leave-out: weak-identification diagnostic, Lanczos up to "
             f"{max_iter} steps, {n_draws} trace draws",
             self.logger, self.log_level)

        caps, reorth = self._krylov_plan(L, max_iter)

        def converged_all(alphas, betas, lengths, steps):
            for c in range(3):
                if lengths[c] < steps:      # stopped: exhausted or at its cap
                    continue
                theta, bounds, _S = self._ritz_top(alphas[c], betas[c], top)
                lead = float(np.abs(theta[0])) if len(theta) else 0.0
                if np.any(bounds[:2] > tol * max(lead, 1e-300)):
                    return False
            return True

        with self._quiet_solver(), self._scratch_capped(12 * max(3, block)):
            # pass 1: the tridiagonals, stopping once the leading two
            # eigenvalues of every component have converged
            alphas, betas, lengths, n_iter = self._lanczos(
                L, covariates, max_iter, seed, check=converged_all, every=10,
                caps=caps, reorth=reorth)
            checked = [self._ritz_top(alphas[c], betas[c], top)
                       for c in range(3)]

            # pass 2: the leading Ritz vectors of each component, by retracing.
            # Up to `top` of them where they fit the scratch budget, for the
            # deflation below; always at least the first, for the F statistic.
            n_coef = L["total_levels"] + len(L["columns"]) + L["n_groups"]
            budget = int(getattr(self, "scratch_mb", 0) or 32) << 20
            k = top if 2 * 3 * top * n_coef * 8 <= budget else 1
            ritz = []
            for _theta, _b, S in checked:
                coef = np.zeros((S.shape[0], k))
                coef[:, :min(k, S.shape[1])] = S[:, :k]
                ritz.append(coef)
            X, Y = self._lanczos(L, covariates, n_iter, seed, ritz=ritz,
                                 caps=caps, reorth=reorth)
            X, Y = self._s_orthonormalize(X, Y, k)
            u = [block[:, ::k].copy() for block in Y]

            leading = [theta for theta, _b, _S in checked]

            def settled(estimate, se):
                """No ratio within two standard errors of the threshold."""
                for c in range(3):
                    if not estimate[c] > 0 or not np.isfinite(se[c]):
                        return False
                    r = leading[c] ** 2 / estimate[c]
                    if np.any(np.abs(r - WEAK_ID_THRESHOLD)
                              < 2.0 * (se[c] / estimate[c]) * r):
                        return False
                return True

            sum_sq, sum_sq_se, used = self._sum_sq_eigenvalues(
                L, covariates, n_draws, block, seed + 3_300_007, X, Y, k,
                settled=settled)

        table = self._fit_smoother(sigma2, leverages.leverage, None, smooth_bins)
        f_stat = self._first_stage_f(result, sigma2, leverages, table, u, L)

        eig, ratios, q, capped, bounds, converged = {}, {}, {}, {}, {}, {}
        borderline = {}
        for c, name in enumerate(names):
            theta, b, _S = checked[c]
            eig[name] = theta.tolist()
            bounds[name] = b.tolist()
            lead = float(np.abs(theta[0])) if len(theta) else 0.0
            converged[name] = bool(lengths[c] < n_iter or lengths[c] >= caps[c]
                                   or np.all(b[:2] <= tol * max(lead, 1e-300)))
            r = (theta ** 2 / sum_sq[c]) if sum_sq[c] > 0 else np.zeros_like(theta)
            ratios[name] = r.tolist()
            count = 0
            for value in r:
                if value >= WEAK_ID_THRESHOLD:
                    count += 1
                else:
                    break
            q[name] = count
            capped[name] = bool(count == len(r) and count > 0)
            # the ratios share the estimated denominator, so their relative
            # Monte Carlo error is that of sum_sq
            rel = sum_sq_se[c] / sum_sq[c] if sum_sq[c] > 0 else np.inf
            borderline[name] = bool(np.any(
                np.abs(r - WEAK_ID_THRESHOLD) < 2.0 * rel * r))

        out = WeakIdDiagnostics(
            eigenvalues=eig, sum_sq=dict(zip(names, sum_sq.tolist())),
            sum_sq_se=dict(zip(names, sum_sq_se.tolist())), ratios=ratios,
            q=q, q_capped=capped, q_borderline=borderline, f_stat=f_stat,
            bounds=bounds, lead_vectors=tuple(u),
            converged=converged,
            diagnostics={"lanczos_steps": int(n_iter), "deflated": k,
                         "lanczos_lengths": lengths.tolist(),
                         "krylov_caps": caps, "full_reorthogonalization": reorth,
                         "n_draws": used, "max_draws": n_draws,
                         "psi": L["psi"],
                         "alpha": self.g_fe, "covariates": list(covariates)})

        if warn and out.weakly_identified:
            detail = ", ".join(
                f"{name} (lambda_1^2/sum = {out.ratios[name][0]:.3f}, "
                f"q = {out.q[name]}{'+' if out.q_capped[name] else ''})"
                for name in out.weakly_identified)
            _warn("leave-out: the normal interval is not justified for "
                  f"{detail}: a few eigen-directions dominate the estimator's "
                  "variance (KSS Theorem 2 condition (ii) fails; threshold "
                  f"{WEAK_ID_THRESHOLD:g}), so its standard error understates "
                  "the uncertainty. The point estimate is still unbiased.",
                  self.logger)
        return out

    def _first_stage_f(self, result, sigma2, leverages, table, u, L):
        """b1^2 / V[b1] for each component's leading direction.

        b1 = sum_i x1bar_i y~_i with x1bar_i = sqrt(w_i) x_i' u1, and
        V[b1] = sum_i x1bar_i^2 sigma2-tilde_i. Both are linear in the rows and
        the ratio is free of u1's scale, so one pass and no normalization.
        """
        depcol = vcol(self.vidx[result.depvar])
        y_bar = self._outcome_mean(depcol)
        wanted = tuple(dict.fromkeys(tuple(L["columns"]) + (depcol,)))
        num = np.zeros(3)
        den = np.zeros(3)
        for ordinal, chunk, starts, codes, w in self._chunks(wanted):
            X = self._covariate_matrix(chunk, L["columns"], len(w))
            first = int(chunk[GCODE][0])
            rows = np.empty((len(w), 3))
            _nb_readout(starts, codes, self.offs, X,
                        np.ascontiguousarray(u[0]), np.ascontiguousarray(u[1]),
                        np.ascontiguousarray(u[2][first:first + len(starts) - 1]),
                        rows)
            root = np.sqrt(w)
            x1bar = rows * root[:, None]
            y_tilde = root * (np.asarray(chunk[depcol], dtype=np.float64) - y_bar)
            span = slice(ordinal, ordinal + len(w))
            num += x1bar.T @ y_tilde
            smooth = self._smooth_at(table, sigma2, leverages.leverage, None,
                                     span)
            den += (x1bar ** 2).T @ smooth
        return {name: float(num[c] ** 2 / den[c]) if den[c] > 0 else float("nan")
                for c, name in enumerate(COMPONENTS)}

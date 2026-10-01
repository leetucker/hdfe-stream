"""`StreamingGLM`: Poisson, logit and probit regression with high-dimensional
fixed effects, by iteratively reweighted least squares (IRLS).

Each IRLS step is a weighted least-squares fit of a working response on the
covariates and the fixed effects -- the regression `StreamingHDFE` solves --
with weights that change from step to step. The rows, their sort and the cell
structure do not change, so pass 0 runs once, and passes 1/1b once for the
structure only (which cells identify the other effects, their levels, the
connected components). A step is then:

  row pass   the linear predictor of every row, from the current
             coefficients: eta = x'beta + a[group] + sum_d fe_d[level] +
             offset; from it the IRLS weight W and the working response z;
             per cell, sum W and W [z, x]; and the within-cell
             cross-products, with BLAS
  solve      the reduced system for z and the covariates, from the cell sums,
             starting from the previous step's solution
  update     X~'WX~ and X~'Wz~ assembled from the cells; beta; the other
             effects; and each fe[0] group's effect, from its cells

The row pass for a step's coefficients also gives their deviance, so a step
costs one read of the rows and one solve. Nothing row-sized is written: the
linear predictor is rebuilt from the coefficients, and the files rewritten
each step are the cells' weight and sums. The fe[0] effects are held in
memory, one float per group (two or three sets at once).

Which cell a row belongs to is found by reading the rows in order: pass 0
sorts each fe[0] group's rows by the other dimensions' codes, so a cell is one
run of rows, and a group with more than one run is one of pass 1b's
identifying groups. Cells are numbered in the order they are read, which is
the order pass 1b stored them in.

The iteration follows pyfixest's (`fepois`, `feglm`): the same starting
values, convergence when the relative change in deviance,
|dev - dev_old| / (0.1 + |dev|), falls below `iwls_tol`, and step-halving when
a step raises the deviance. Inference is that of the final weighted
least-squares step, as in pyfixest: bread (X~'WX~)^-1, scores W e x~ with e
the working residual, and pyfixest's small-sample factors, with the iid vcov
the bread alone (no dispersion is estimated).

Before anything is written, fixed-effect levels whose outcome is always zero
(Poisson) or never varies (logit, probit) are dropped: their effect would be
infinite. For Poisson one pass over the dimensions finds them all, since
dropping zero rows leaves every other level's positive rows in place. For
logit and probit it can take more: dropping a firm whose outcome is all ones
can leave a worker with only zeros, so the passes repeat until one drops
nothing. See `separation_check`.
"""

from __future__ import annotations

import time
from typing import Any, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from types import SimpleNamespace

import numba as nb
import numpy as np
import polars as pl
from scipy import special

from ._columns import PREFIX, WCOL, vcol
from ._types import Variables
from .estimator import StreamingHDFE
from .families import get_family
from .inference import _stack
from .kernels_base import _nb_assemble, _nb_diag
from .kernels_glm import _nb_cell_moments, _nb_group_effects
from .report import _warn
from .utils import _norm_vars

LO, HI = f"{PREFIX}lo", f"{PREFIX}hi"      # per-level outcome range, in the separation check
_OFFSET = "__offset__"          # the offset's variable name in the row files
_MAX_HALVINGS = 30


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


class StreamingGLM(StreamingHDFE):
    """
    Low-level GLM interface (see `fepois_stream` and `feglm_stream` for
    formulas): one dependent variable, any number of models of it.

    Parameters, beyond `StreamingHDFE`'s
    ------------------------------------
    family : "poisson" (log link), "logit" or "probit".
    offset : column name or Polars expression added to the linear predictor
         with a coefficient of one (e.g. log exposure).
    iwls_tol, iwls_maxiter : IRLS convergence tolerance on the relative
         change in deviance, and the most steps (pyfixest's defaults).
    separation_check : drop fixed-effect levels whose outcome is always zero
         (Poisson) or never varies (logit, probit), for logit and probit
         repeating until none is left (default True). Covariates that
         separate the outcome are not detected.

    `assembly` does not apply (the cross-products come from each step's row
    pass), and varying slopes, IV and CRV3 are not available. With
    solver="auto" the first step builds S and later steps use stream_cg (see
    `_wls`).
    """

    cells_contiguous = True
    cell_sums = False

    def __init__(self, y: Variables, x: Variables | None, fe: Sequence[str],
                 family: str = "poisson", offset: str | pl.Expr | None = None,
                 iwls_tol: float = 1e-8, iwls_maxiter: int = 25,
                 separation_check: bool = True, **options: Any) -> None:
        ys = _norm_vars(y)
        if len(ys) != 1:
            raise ValueError("StreamingGLM takes one dependent variable; "
                             "fepois_stream/feglm_stream fit one estimator per outcome")
        options.pop("assembly", None)
        super().__init__(list(ys.items()), x, fe, **options)
        if self.slopes:
            raise ValueError("varying slopes are not available for GLMs")
        self.yname = next(iter(ys))
        for model in self.models:
            if model.get("iv"):
                raise ValueError(f"{model['fml']}: IV (2SLS) is not available for GLMs")
            if model["y"] != self.yname:
                raise ValueError(f"{model['fml']}: every model must have {self.yname!r} "
                                 "as its dependent variable")
        if not 0 < iwls_tol < 1:
            raise ValueError("iwls_tol must be between 0 and 1")
        if int(iwls_maxiter) < 1:
            raise ValueError("iwls_maxiter must be at least 1")
        self.family = get_family(family)
        self.iwls_tol, self.iwls_maxiter = float(iwls_tol), int(iwls_maxiter)
        self.separation_check = bool(separation_check)
        self.assembly = "rows"          # pass 1 builds no cross-products
        self.offset_name = None
        if offset is not None:
            expr = pl.col(offset) if isinstance(offset, str) else offset
            self.var_names.append(_OFFSET)
            self.var_exprs[_OFFSET] = expr.cast(pl.Float64)
            self.vidx[_OFFSET] = self.m
            self.m += 1
            self.offset_name = offset if isinstance(offset, str) else "<expr>"
        self._p2 = None
        self._raw_w = None
        self._bufs = {}

    # ------------------------------------------------------------ sample
    def _restrict_sample(self, lf):
        """Check the outcome, and drop the separated fixed-effect levels."""
        y = pl.col(vcol(self.vidx[self.yname]))
        chk = (lf.select(n=pl.len(), lo=y.min(), hi=y.max(),
                         other=((y != 0) & (y != 1)).sum(), zeros=(y == 0).sum())
                 .collect(engine="streaming"))
        n = chk["n"].item()
        self.n_before_separation = n
        self._separation_tables = {}
        self.separation = {"checked": self.separation_check and bool(self.fe_user),
                           "observations": 0, "levels": {}, "rounds": 0}
        if n == 0:
            return None                 # pass 0 reports it
        self.family.check(chk["lo"].item(), chk["hi"].item(), chk["other"].item())
        if not self.separation["checked"]:
            return None
        binary = self.family.name != "poisson"
        if not binary and chk["zeros"].item() == 0:
            return None
        separated = (pl.col(LO) == pl.col(HI)) if binary else (pl.col(HI) == 0)
        cols = list(dict.fromkeys(c for d in self.fe_user for c in self.fe_cols[d]))
        cur = lf.select([pl.col(c) for c in cols] + [y])
        found = {}
        while True:
            changed = False
            for d in self.fe_user:
                on = self.fe_cols[d]
                bad = (cur.group_by(on).agg(y.min().alias(LO), y.max().alias(HI))
                          .filter(separated).select(on).collect(engine="streaming"))
                if bad.height:
                    found.setdefault(d, []).append(bad)
                    cur = cur.join(bad.lazy(), on=on, how="anti")
                    changed = True
            # rounds counts the passes over the dimensions that dropped something
            self.separation["rounds"] += changed
            # levels of the only dimension cannot separate each other, and
            # dropping rows whose outcome is zero separates nothing new
            if not changed or len(self.fe_user) == 1 or not binary:
                break
        if not found:
            return None
        tables = {d: pl.concat(v) for d, v in found.items()}
        self._separation_tables = tables
        self.separation["levels"] = {d: t.height for d, t in tables.items()}

        def restrict(frame):
            for d, t in tables.items():
                frame = frame.join(t.lazy(), on=self.fe_cols[d], how="anti")
            return frame
        return restrict

    # ------------------------------------------------------------ driver
    def _fit_passes(self, source, keys, extra, default, fe_dof, t0):
        if any(k.startswith("CRV3:") for k in keys):
            raise ValueError("CRV3 standard errors are not available for GLMs; use CRV1")
        self._pass0_code(source, keys, extra)
        dropped = self.n_before_separation - self.n_obs
        self.separation["observations"] = dropped
        if dropped:
            _warn(f"{dropped:,} observations removed because of separation ("
                  + ", ".join(f"{v:,} levels of {d}"
                              for d, v in self.separation["levels"].items()) + ")",
                  self.logger)
        if self.no_fe:
            self._no_cells()
        else:
            self._pass1_cells()
            self._pass1b_identifying()
            self._components()
        self._log("n={:,} cells={:,} identifying groups={:,} cells={:,} components={:,} "
                  "threads={} ".format(self.n_obs, self.n_cells, self.n_identifying,
                                       self.n_ident_cells, self.n_components,
                                       nb.get_num_threads())
                  + " ".join(f"{f}={v:,}" for f, v in self.n_levels.items()))
        nested = {t: self._nested_dims(t) for t in self.clusters}
        ctx = {"k_fe": self._k_fe(fe_dof), "nested": nested, "default": default, "t0": t0}
        self._layout()
        self._track_disk()
        return [self._fit_model(f"m{mi:03d}", model, ctx)
                for mi, model in enumerate(self.models)]

    def _layout(self):
        """Counts that fix where each cell's sums go: identifying cells as
        pass 1b stored them, then one row per single-cell group."""
        G = self.n_levels.get(self.g_fe, 0) if self.fe else 0
        self.G_id = len(self.starts) - 1
        self.G_s = G - self.G_id
        self.M_id = int(self.starts[-1])
        self.s_starts = np.arange(self.G_s + 1, dtype=np.int64)
        self.codes_s = None             # filled by the first row pass
        self._bufs = {}
        self.ybar = float(self.means[self.vidx[self.yname]])

    def _fit_model(self, tag, model, ctx):
        self._log(f"IRLS ({self.family.name}): {model['fml']}")
        fit = self._irls(model)
        m, (_, L) = self.m, self._offsets()
        cols = [self.vidx[self.yname]] + fit.xj
        A = np.zeros((m, m))
        A[np.ix_(cols, cols)] = fit.A
        gamma = np.zeros((L, m))
        gamma[:, cols] = fit.gamma
        self._raw_w = np.zeros(m)
        self._raw_w[cols] = fit.raw
        self._p2 = SimpleNamespace(prev=fit.prev, final=fit.final, yname=model["y"])
        try:
            res = self._estimate(tag, dict(model, x=fit.names), A, gamma,
                                 dict(ctx, info=fit.info))
        finally:
            self._p2 = self._raw_w = None
        ll0 = self.family.loglik_null(fit.sy, fit.sw, fit.slg)
        res.family = self.family.name
        res.deviance, res.loglik = float(fit.dev), float(fit.ll)
        res.pseudo_r2 = float(1 - fit.ll / ll0) if ll0 else np.nan
        res.rss = res.r2 = res.adj_r2 = res.r2_within = res.adj_r2_within = res.rmse = np.nan
        res.collin_vars = fit.dropped + res.collin_vars
        res.diagnostics["irls"] = {"iterations": fit.iterations, "converged": fit.converged,
                                   "step_halvings": fit.halvings,
                                   "deviance": [float(d) for d in fit.path],
                                   "seconds": fit.seconds}
        res.diagnostics["separation"] = dict(self.separation)
        if self.offset_name is not None:
            res.diagnostics["offset"] = self.offset_name
        return res

    @contextmanager
    def _quiet(self, on):
        verbose = self.verbose
        self.verbose = verbose and not on
        try:
            yield
        finally:
            self.verbose = verbose

    # ------------------------------------------------------------ IRLS
    def _irls(self, model):
        yj = self.vidx[self.yname]
        xj = [self.vidx[x] for x in model["x"]]
        names = list(model["x"])
        t0 = time.time()
        binary = self.family.name != "poisson"
        # Two sets of cell weights: the current step's (`slot`) and the
        # candidate's, which become current when the step is accepted. The
        # cell sums, k + 1 columns, are needed only until `_update` has turned
        # them into coefficients, so the candidate's overwrite them: with many
        # covariates they are the largest thing the fit holds.
        slot = 0
        cur = _State(init=True)
        agg = self._row_pass(cur, yj, xj, slot, first=True)
        const = (agg.sy, agg.sw, agg.slg)   # sums of the outcome, which do not change
        dev = agg.dev
        path, dropped, gamma, halvings = [dev], [], None, 0
        converged, last, it = False, None, 0
        for it in range(1, self.iwls_maxiter + 1):
            with self._quiet(it > 1):
                gamma, A, info = self._wls(agg, ["(working response)", *names], gamma)
            if it == 1:
                keep = ~self._collinear(A[1:, 1:], raw=agg.raw[1:])
                if len(keep) and not keep.any():
                    raise ValueError(f"{model['fml']}: all covariates are collinear with the "
                                     "fixed effects")
                if not keep.all():
                    dropped = [nm for nm, kp in zip(names, keep) if not kp]
                    _warn(f"{model['fml']}: {len(dropped)} variables dropped due to "
                          f"multicollinearity: {dropped}", self.logger)
                    sel = np.concatenate(([0], 1 + np.flatnonzero(keep)))
                    A, gamma = A[np.ix_(sel, sel)], gamma[:, sel]
                    agg = self._select(agg, sel, slot)
                    xj = [j for j, kp in zip(xj, keep) if kp]
                    names = [nm for nm, kp in zip(names, keep) if kp]
            new = self._update(agg, gamma, A)
            cand = self._row_pass(new, yj, xj, 1 - slot)
            if not self._accept(cand.dev, dev):
                # step-halving, as pyfixest's feglm: back toward the current
                # coefficients (for logit and probit the start is all zeros;
                # Poisson's start is not a set of coefficients, so its first
                # step is taken as it is)
                if cur.init and not binary:
                    if not np.isfinite(cand.dev):
                        raise FloatingPointError(f"{model['fml']}: the first IRLS step "
                                                 "gives a non-finite deviance")
                else:
                    base = cur.zeros_like() if cur.init else cur
                    target, t = new, 1.0
                    for _ in range(_MAX_HALVINGS):
                        t /= 2
                        halvings += 1
                        new = base.toward(target, t)
                        cand = self._row_pass(new, yj, xj, 1 - slot)
                        if np.isfinite(cand.dev) and cand.dev < dev:
                            break
                    else:
                        raise RuntimeError(f"{model['fml']}: IRLS step-halving failed "
                                           f"(deviance {cand.dev:.10g} vs {dev:.10g})")
            crit = abs(cand.dev - dev) / (0.1 + abs(dev))
            # the step just taken is the final one if this converges: its
            # weights (those of `cur`), Gamma and cross-products give the vcov
            last = SimpleNamespace(cur=cur, gamma=gamma, A=A, info=info, raw=agg.raw)
            cur, agg, dev, slot = new, cand, cand.dev, 1 - slot
            path.append(dev)
            self._log(f"  IRLS {it}: deviance {dev:.12g}, relative change {crit:.2e}"
                      + (f", solver iterations {info['iterations']}"
                         if info.get("solver") != "none" else ""))
            if crit < self.iwls_tol:
                converged = True
                break
        if not converged:
            _warn(f"{model['fml']}: IRLS did not converge within iwls_maxiter="
                  f"{self.iwls_maxiter} steps", self.logger)
        info = dict(last.info)
        return SimpleNamespace(
            prev=last.cur, final=cur, gamma=last.gamma, A=last.A, info=info, raw=last.raw,
            xj=xj, names=names, dropped=dropped, dev=dev, ll=agg.ll, path=path,
            iterations=it, converged=converged, halvings=halvings,
            sy=const[0], sw=const[1], slg=const[2], seconds=round(time.time() - t0, 2))

    def _accept(self, dev_new, dev):
        """A step that lowers the deviance, or changes it by less than the
        tolerance (rounding at convergence can raise it by a hair)."""
        return np.isfinite(dev_new) and (
            dev_new < dev or abs(dev_new - dev) / (0.1 + abs(dev)) < self.iwls_tol)

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
        # from the previous solution and need fewer iterations (docs/glm.md).
        # AMG needs S, so with it S is rebuilt every step.
        if self.solver == "auto":
            self.solver = ("explicit" if self.precond == "amg" and info["solver"] != "stream_cg"
                           else "stream_cg")
        acc = np.zeros((nb.get_num_threads(), c, c))
        _nb_assemble(self.starts, self.codes, self.n, agg.sums_id, self.offs, gamma, acc)
        return gamma, agg.W + acc.sum(axis=0), info

    def _update(self, agg, gamma, A):
        """The coefficients that solve one weighted least-squares step."""
        k = A.shape[0] - 1
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

    # ------------------------------------------------------------ row pass
    def _columns(self, yj, xj):
        cols = [*self.ccols, vcol(yj), *[vcol(j) for j in xj]]
        if self.weights is not None:
            cols.append(WCOL)
        if self.offset_name is not None:
            cols.append(vcol(self.vidx[_OFFSET]))
        return list(dict.fromkeys(cols))

    def _chunk_layout(self, starts, codes, cursor):
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
        ncell = np.diff(np.searchsorted(cstarts, starts))
        is_id = ncell > 1
        cell_id = np.repeat(is_id, ncell)
        lay = SimpleNamespace(starts=starts, cstarts=cstarts, is_id=is_id, cell_id=cell_id,
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
        f = (np.ascontiguousarray(ch[WCOL], dtype=np.float64) if self.weights is not None
             else None)
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

    def _row_pass(self, state, yj, xj, slot, first=False):
        """Read the rows once at coefficients `state`: the deviance and
        log-likelihood there, and the cell sums of the next weighted
        least-squares step (column 0 the working response z, then the
        covariates), with the within-cell cross-products and each column's
        weighted total sum of squares (`raw`). The cell weights go to the
        buffers of `slot`; the cell sums replace the previous ones."""
        fam, k = self.family, len(xj)
        c = k + 1
        out = SimpleNamespace(dev=0.0, ll=0.0, sy=0.0, sw=0.0, slg=0.0,
                              W=np.zeros((c, c)), sv=np.zeros(c), bss=np.zeros(c), wsum=0.0,
                              n_id=self._buffer(("n_id", slot), (self.M_id,)),
                              sums_id=self._buffer(("sums_id", c), (self.M_id, c)),
                              n_s=self._buffer(("n_s", slot), (self.G_s,)),
                              sums_s=self._buffer(("sums_s", c), (self.G_s, c)))
        if first and self.fe:
            self.codes_s = np.zeros((self.G_s, len(self.o_fe)), np.int64)
        cursor = [0, 0, 0]
        for ch, starts, codes in self._row_chunks(self._columns(yj, xj)):
            lay = self._chunk_layout(starts, codes, cursor)
            y, X, f, off = self._chunk_data(ch, yj, xj)
            eta = self._eta(state, y, X, off, codes, lay)
            mu, W, u = fam.irls(y, eta)
            dev, ll = fam.deviance(y, eta, mu), fam.loglik(y, eta, mu)
            out.dev += float(dev.sum() if f is None else dev @ f)
            out.ll += float(ll.sum() if f is None else ll @ f)
            if first:
                out.sy += float(y.sum() if f is None else y @ f)
                out.sw += float(len(y) if f is None else f.sum())
                if fam.name == "poisson":
                    lg = special.gammaln(y + 1.0)
                    out.slg += float(lg.sum() if f is None else lg @ f)
            w = W if f is None else W * f
            V = np.empty((len(y), c))
            V[:, 0] = eta + u if off is None else eta + u - off
            V[:, 1:] = X
            del X
            if lay is None:
                out.sv += w @ V
                out.wsum += w.sum()
                V *= np.sqrt(w)[:, None]
                out.W += V.T @ V
                continue
            nc = np.empty(len(lay.cstarts) - 1)
            sc = np.empty((len(nc), c))
            _nb_cell_moments(lay.cstarts, w, V, nc, sc)
            out.W += V.T @ V
            out.sv += sc.sum(axis=0)
            out.wsum += nc.sum()
            pos = nc > 0
            out.bss += (sc[pos] ** 2 / nc[pos, None]).sum(axis=0)
            out.n_id[lay.c0:lay.c0 + lay.nc] = nc[lay.cell_id]
            out.sums_id[lay.c0:lay.c0 + lay.nc] = sc[lay.cell_id]
            out.n_s[lay.s0:lay.s0 + lay.ns] = nc[~lay.cell_id]
            out.sums_s[lay.s0:lay.s0 + lay.ns] = sc[~lay.cell_id]
            if first:
                self.codes_s[lay.s0:lay.s0 + lay.ns] = codes[starts[:-1][~lay.is_id]]
        if self.fe and (cursor[0], cursor[1], cursor[2]) != (self.M_id, self.G_id, self.G_s):
            raise RuntimeError(
                f"the row files hold {cursor[0]:,} identifying cells in {cursor[1]:,} groups "
                f"and {cursor[2]:,} single-cell groups, where pass 1b found {self.M_id:,}, "
                f"{self.G_id:,} and {self.G_s:,}")
        # sum w v^2: within cells, plus between cells (none without fixed effects)
        ss = np.diag(out.W) + out.bss
        out.raw = np.maximum(ss - out.sv ** 2 / out.wsum, 0.0) if out.wsum > 0 else ss
        return out

    # ------------------------------------------------------------ pass 2
    def _pass2_rows(self, yj, xk):
        """Pass 2 runs on the final step's working regression: z and W from
        the coefficients before the step, Gamma and beta from it. The residual
        file gets the outcome, the linear predictor, the fitted mean and the
        response residual y - mu at the final coefficients, and the working
        residual."""
        p, fam = self._p2, self.family
        cursor = [0, 0, 0]
        fweights = self.weights is not None and self.weights_type == "fweights"

        def rows(ch, starts, codes):
            lay = self._chunk_layout(starts, codes, cursor)
            y, X, f, off = self._chunk_data(ch, yj, xk)
            eta = self._eta(p.prev, y, X, off, codes, lay)
            _, W, u = fam.irls(y, eta)
            z = eta + u if off is None else eta + u - off
            eta_f = self._eta(p.final, y, X, off, codes, lay)
            mu_f = fam.mu(eta_f)
            return SimpleNamespace(
                y=z, w=W if f is None else W * f, fw=f if fweights else None, w_out=f,
                extra=lambda e: {p.yname: y, "eta": eta_f, "fitted": mu_f, "resid": y - mu_f,
                                 "resid_working": e})
        rows.cols = ([vcol(self.vidx[_OFFSET])] if self.offset_name is not None else [])
        rows.weighted = True
        return rows

    def _iid_factor(self, rss, N, K):
        """The bread alone, with pyfixest's (N - 1) / (N - K)."""
        return (N - 1) / (N - K) if N > K else np.nan

    def _collinear(self, Axx, idx=None, raw=None):
        if raw is None and idx is not None and self._raw_w is not None:
            raw = self._raw_w[idx]
        return super()._collinear(Axx, idx, raw)

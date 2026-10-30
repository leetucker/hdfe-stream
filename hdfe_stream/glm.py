"""`StreamingGLM`: Poisson, logit and probit regression with high-dimensional
fixed effects, by iteratively reweighted least squares (IRLS).

Each IRLS step is a weighted least-squares fit of a working response on the
covariates and the fixed effects -- the regression `StreamingHDFE` solves --
with weights that change from step to step. (Those steps, which do not depend
on the family, are `core/irls.py`'s `_IRLSMixin`; the family's row pass, the
loop and the fit statistics are here.) The rows, their sort and the cell
structure do not change, so pass 0 runs once, and pass 1 once for the
structure only (which cells identify the other effects, their levels, the
connected components). A step is then:

  row pass   the linear predictor of every row, from the current
             coefficients: eta = x'beta + a[group] + sum_d fe_d[level] +
             offset; from it the IRLS weight W and the working response z;
             per cell, sum W and W [z, x], all in one multithreaded sweep
             (`kernels_glm._nb_glm_rows`); and the within-cell
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
run of rows, and a group with more than one run is one of pass 1's
identifying groups. Cells are numbered in the order they are read, which is
the order pass 1 stored them in.

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
from types import SimpleNamespace

import numpy as np
import polars as pl

from .core._columns import PREFIX, vcol
from .core.typing import Variables
from .core.families import get_family
from .core.irls import _OFFSET, _IRLSMixin, _State
from .core.kernels_glm import FAMILY_CODES, _nb_glm_rows, _nb_glm_rows_plain
from .core.passes import _drop_levels
from .core.report import warn
from .core.inference import _criteria
from .core.utils import _norm_vars, single_blas
from .estimator import StreamingHDFE

LO, HI = f"{PREFIX}lo", f"{PREFIX}hi"      # per-level outcome range, in the separation check
# blocks of groups the fused row pass reduces its sums over: fixed, so that the
# deviance does not depend on the number of threads
_ROW_BLOCKS = 256
_MAX_HALVINGS = 30
# the row passes hold BLAS to one thread up to this many columns (see
# core.utils.single_blas); wider cross-products gain from BLAS's threads
_SINGLE_BLAS_MAX_COLS = 128
# without batch_rows, chunks of about this many MB of the design (at most
# 2,000,000 rows, at least 100,000): with BLAS on one thread, small chunks
# cost no time, and memory follows the chunk
_CHUNK_MB = 64


class StreamingGLM(_IRLSMixin, StreamingHDFE):
    """Low-level GLM interface: column names or Polars expressions instead of
    a formula (see `fepois_stream` and `feglm_stream` for formulas). One
    dependent variable, any number of models of it.

    Takes `StreamingHDFE`'s parameters, and these. ``assembly`` does not apply
    (the cross-products come from each step's row pass), and varying slopes,
    IV and CRV3 are not available. With ``solver="auto"`` the first step
    builds S and later steps use ``"stream_cg"``.

    Parameters
    ----------
    y : str or tuple
        The dependent variable: a column name, or a ``(name, polars.Expr)``
        pair.
    x : str, tuple, list or None
        Covariates, as in `StreamingHDFE`.
    fe : sequence of str
        Fixed-effect dimensions, as in `StreamingHDFE`, without varying
        slopes.
    family : {"poisson", "logit", "probit"}, default "poisson"
        Poisson (log link), logit or probit.
    offset : str or polars.Expr, optional
        Column name or Polars expression added to the linear predictor with a
        coefficient of one (e.g. log exposure).
    iwls_tol : float, default 1e-8
        IRLS convergence tolerance on the relative change in deviance
        (pyfixest's default).
    iwls_maxiter : int, default 25
        The most IRLS steps (pyfixest's default).
    separation_check : bool, default True
        Drop fixed-effect levels whose outcome is always zero (Poisson) or
        never varies (logit, probit), for logit and probit repeating until
        none is left. Covariates that separate the outcome are not detected.
    bias_correction : bool or dict, optional
        Correct the coefficients for the incidental-parameter bias of the
        fixed effects (logit and probit, one or two fixed effects), by the
        analytic correction of Fernandez-Val and Weidner (2016) as R's
        alpaca computes it (`biasCorr`): ``True`` for strictly exogenous
        regressors, which needs no roles for the fixed effects (the two
        terms have the same form); ``{"time": "year", "lags": L}`` for
        predetermined regressors (a lagged outcome, ...), with two fixed
        effects, the unit and the period ``time``: the unit's term then also
        sums the scores' products up to ``L`` periods apart, so each unit
        and period must have at most one row, and the period column must be
        numeric or temporal. The fixed effects are then re-estimated with
        the corrected coefficients held, and the variances, residuals and
        fixed effects are those of that fit; the fit statistics (deviance,
        log-likelihood, AIC) remain the maximum-likelihood fit's. Its
        coefficients and the bias are in ``diagnostics["bias_correction"]``.
        The correction is O(1/T) in the observations per level, and
        unreliable for very short panels (a warning below five). Not for
        Poisson, whose coefficients have no such bias with one or two fixed
        effects and strictly exogenous regressors.
    within_pseudo_r2 : bool, default False
        Also fit the model with the fixed effects and no covariates, for
        the within pseudo R-squared (`HDFEResult.pseudo_r2_within`) when it
        is not free: with two or more fixed effects, or an offset. With one
        fixed effect and no offset that model is the groups' means, and it
        is always reported.
    **options
        `StreamingHDFE`'s other parameters.

    Examples
    --------
    >>> import polars as pl
    >>> fit = StreamingGLM("visits", ["x"], ["worker_id", "firm_id", "year"],
    ...                    family="poisson", offset=pl.col("log_exposure"),
    ...                    workdir="scratch").fit("data/*.parquet")
    """

    def __init__(self, y: Variables, x: Variables | None, fe: Sequence[str],
                 family: str = "poisson", offset: str | pl.Expr | None = None,
                 iwls_tol: float = 1e-8, iwls_maxiter: int = 25,
                 separation_check: bool = True, within_pseudo_r2: bool = False,
                 bias_correction: bool | dict | None = None, **options: Any) -> None:
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
        self.within_pseudo_r2 = bool(within_pseudo_r2)
        self.bias = self._bias_spec(bias_correction)
        self._bias_acc = None
        self._fixed_beta = None
        self.assembly = "rows"          # the cross-products come from each step
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
        self._dev_rows = None           # device copy of the rows, no-FE fits on the GPU
        if options.get("batch_rows") is None:
            self.batch_rows = min(2_000_000,
                                  max(100_000, int(_CHUNK_MB * 1e6 / (8 * (self.m + 4)))))

    def _bias_spec(self, spec):
        """`bias_correction=`, checked: None, or the dimensions to correct,
        and with lags the unit, the period and L (the unit then streamed, so
        that each chunk holds whole units, their rows in period order)."""
        if spec is None or spec is False:
            return None
        if spec is True:
            spec = {}
        if not isinstance(spec, dict) or set(spec) - {"time", "lags"}:
            raise ValueError("bias_correction is True, or {'time': <period>, 'lags': L} for "
                             "predetermined regressors")
        if self.family.name == "poisson":
            raise ValueError(
                "bias_correction is for logit and probit: Poisson's coefficients have no "
                "incidental-parameter bias with one or two fixed effects and strictly "
                "exogenous regressors, and the three-way (gravity) correction of Weidner "
                "and Zylkin (2021) is not implemented")
        dims = list(self.fe_user)
        if len(dims) not in (1, 2):
            raise ValueError("bias_correction is for models with one or two fixed effects "
                             f"(this one has {len(dims)})")
        lags, time = int(spec.get("lags", 0)), spec.get("time")
        if lags < 0:
            raise ValueError("bias_correction: lags must be a nonnegative integer")
        if time is not None:
            time = "^".join(c.strip() for c in str(time).split("^"))
            if time not in dims:
                raise ValueError(f"bias_correction: the period {time!r} is not one of the "
                                 f"fixed effects {dims}")
        unit = None
        if lags:
            if time is None or len(dims) != 2:
                raise ValueError("bias_correction with lags needs two fixed effects, the unit "
                                 "and the period, and the period named: "
                                 "{'time': 'year', 'lags': L}")
            if len(self.fe_cols[time]) != 1:
                raise ValueError(f"bias_correction: the period {time!r} must be one column")
            unit = next(d for d in dims if d != time)
            if self.stream is not None and self.stream != unit:
                raise ValueError(f"bias_correction with lags reads the rows unit by unit: "
                                 f"stream={unit!r}, or leave stream unset")
            self.stream = unit
        return {"dims": dims, "time": time, "lags": lags, "unit": unit}

    def _bias_correct(self, tag, model, fit, res, ctx, acc):
        """The bias-corrected coefficients from pass 2's sums, as alpaca's
        `biasCorr`: beta + H^-1 (sum over the corrected dimensions of the
        levels' (sum x~ z) / (sum w), halved, plus the lagged term), with H
        = X~' W X~; then the fixed effects re-estimated with them held, and
        the pass 2 of that fit."""
        names = list(res.coefnames)
        if not names:
            return res
        delta = np.linalg.solve(self._last_bread, acc.totals())
        beta = res.beta + delta
        info = acc.report(self)
        self._log("bias correction: re-estimating the fixed effects with the corrected "
                  "coefficients held")
        start = _State(beta=beta.copy(), fe=fit.final.fe, a_id=fit.final.a_id,
                       a_s=fit.final.a_s)
        self._fixed_beta = beta
        try:
            with self._quiet(True):
                try:
                    fit_c = self._irls(dict(model, x=fit.names), start=start)
                except RuntimeError:
                    # from the first fit's effects, which are far from these
                    # coefficients' when the correction is large (very short
                    # panels), the first steps can overshoot beyond what
                    # step-halving recovers: start the effects from zero
                    self._log("bias correction: the re-estimation failed from the first "
                              "fit's fixed effects; again from zero")
                    fit_c = self._irls(dict(model, x=fit.names), start=_State(
                        beta=beta.copy(), fe=np.zeros_like(start.fe),
                        a_id=np.zeros_like(start.a_id), a_s=np.zeros_like(start.a_s)))
        finally:
            self._fixed_beta = None
        out = self._inference(tag, model, fit_c, ctx, beta=beta)
        out.diagnostics["bias_correction"] = {
            "method": "Fernandez-Val and Weidner (2016), as alpaca::biasCorr",
            "dims": list(self.bias["dims"]), "time": self.bias["time"],
            "lags": self.bias["lags"],
            "beta_uncorrected": dict(zip(names, map(float, res.beta))),
            "se_uncorrected": dict(zip(names, map(float, res.se))),
            "bias": dict(zip(names, map(float, -delta))),
            "refit": {"iterations": fit_c.iterations, "converged": fit_c.converged},
            **info}
        return out

    # ------------------------------------------------------------ sample
    def _restrict_sample(self, lf):
        """Check the outcome, and drop the separated fixed-effect levels."""
        if self.bias is not None and self.bias["lags"]:
            t = self.bias["time"]
            dt = lf.collect_schema()[self.fe_cols[t][0]]
            if not (dt.is_numeric() or dt.is_temporal()):
                raise ValueError(f"bias_correction: the period {t!r} must be numeric or a "
                                 f"date or time, to order each unit's rows (it is {dt})")
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

        return lambda frame: _drop_levels(frame, tables, self.fe_cols)

    # ------------------------------------------------------------ driver
    def _after_pass0(self):
        # analytic weights rescaled to average 1 over the estimation sample, as
        # Stata does for maximum likelihood: the fit is the same, but the iid
        # vcov, the log-likelihood and the bias correction would otherwise
        # depend on the weights' scale (frequency weights are counts)
        if self.weights is not None and self.weights_type != "fweights":
            self._wscale = self.n_obs / self.wsum
        dropped = self.n_before_separation - self.n_obs
        self.separation["observations"] = dropped
        if dropped:
            warn(f"{dropped:,} observations removed because of separation ("
                 + ", ".join(f"{v:,} levels of {d}"
                              for d, v in self.separation["levels"].items()) + ")",
                 self.logger)

    def _fit_passes(self, source, keys, extra, default, fe_dof, t0):
        if any(k.startswith("CRV3:") for k in keys):
            raise ValueError("CRV3 standard errors are not available for GLMs; use CRV1")
        if getattr(self, "boot_req", None):
            raise ValueError("the wild cluster bootstrap (bootcluster=) is for linear "
                             "models; it is not available for GLMs, as in pyfixest")
        self._run_passes(source, keys, extra)
        if self.bias is not None and self.bias["lags"] and self.n_cells < self.n_obs:
            raise ValueError(
                f"bias_correction with lags needs at most one row per unit and period; "
                f"{self.n_obs - self.n_cells:,} rows repeat a ({self.bias['unit']}, "
                f"{self.bias['time']}) pair")
        nested = {t: self._nested_dims(t) for t in self.clusters}
        ctx = {"k_fe": self._k_fe(fe_dof), "fe_dof": fe_dof, "nested": nested, "default": default, "t0": t0}
        self._layout()
        self._track_disk()
        return [self._fit_model(f"m{mi:03d}", model, ctx)
                for mi, model in enumerate(self.models)]

    def _fit_model(self, tag, model, ctx):
        self._log(f"IRLS ({self.family.name}): {model['fml']}")
        fit = self._irls(model)
        if self.bias is not None:
            self._bias_acc = _BiasSums(self)
        try:
            res = self._inference(tag, model, fit, ctx)
            acc = self._bias_acc
        finally:
            self._bias_acc = None
        if acc is not None:
            res = self._bias_correct(tag, model, fit, res, ctx, acc)
        return self._fit_statistics(res, model, fit)

    def _inference(self, tag, model, fit, ctx, beta=None):
        """Pass 2 on the final weighted least-squares step of `fit`: the
        variances, the residual and fixed-effect files (`_estimate`); with
        `beta`, at those coefficients."""
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
            with single_blas(len(cols) <= _SINGLE_BLAS_MAX_COLS):
                return self._estimate(tag, dict(model, x=fit.names), A, gamma,
                                      dict(ctx, info=fit.info), beta=beta)
        finally:
            self._p2 = self._raw_w = None

    def _fit_statistics(self, res, model, fit):
        """The family's statistics, of the maximum-likelihood fit `fit`."""
        ll0 = self.family.loglik_null(fit.sy, fit.sw, fit.slg)
        res.family = self.family.name
        res.deviance, res.loglik = float(fit.dev), float(fit.ll)
        res.loglik_null = float(ll0)
        res.pseudo_r2 = float(1 - fit.ll / ll0) if ll0 else np.nan
        aic, bic = _criteria(fit.ll, len(fit.names) + res.k_fe, res.n_obs)
        res.aic, res.bic = float(aic), float(bic)
        if self.fe:
            ll_fe = fit.ll_fe0
            if ll_fe is None and self.within_pseudo_r2:
                self._log(f"IRLS ({self.family.name}), fixed effects only: {model['fml']}")
                with self._quiet(True):
                    ll_fe = self._irls(dict(model, x=[])).ll
            if ll_fe is not None:
                res.loglik_fe_null = float(ll_fe)
                res.pseudo_r2_within = float(1 - fit.ll / ll_fe) if ll_fe else np.nan
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

    # ------------------------------------------------------------ IRLS
    def _irls(self, model, start=None):
        """IRLS for `model`, from `start` (coefficients) or the family's
        starting values."""
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
        cur = start if start is not None else _State(init=True)
        agg = self._row_pass(cur, yj, xj, slot, first=True)
        # sums of the outcome, which do not change, and the fixed-effects-only
        # model's log-likelihood when it is the groups' means
        const = (agg.sy, agg.sw, agg.slg,
                 agg.ll_fe0 - agg.slg if agg.ll_fe0 is not None else None)
        dev = agg.dev
        path, dropped, gamma, halvings = [dev], [], None, 0
        converged, last, it = False, None, 0
        for it in range(1, self.iwls_maxiter + 1):
            with self._quiet(it > 1):
                gamma, A, info = self._wls(agg, ["(working response)", *names], gamma)
            if it == 1 and self._fixed_beta is None:
                keep = ~self._collinear(A[1:, 1:], raw=agg.raw[1:])
                if len(keep) and not keep.any():
                    raise ValueError(f"{model['fml']}: all covariates are collinear with the "
                                     "fixed effects")
                if not keep.all():
                    dropped = [nm for nm, kp in zip(names, keep) if not kp]
                    warn(f"{model['fml']}: {len(dropped)} variables dropped due to "
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
            warn(f"{model['fml']}: IRLS did not converge within iwls_maxiter="
                 f"{self.iwls_maxiter} steps", self.logger)
        info = dict(last.info)
        self._free_device_rows()
        return SimpleNamespace(
            prev=last.cur, final=cur, gamma=last.gamma, A=last.A, info=info, raw=last.raw,
            xj=xj, names=names, dropped=dropped, dev=dev, ll=agg.ll, path=path,
            iterations=it, converged=converged, halvings=halvings,
            sy=const[0], sw=const[1], slg=const[2], ll_fe0=const[3],
            seconds=round(time.time() - t0, 2))

    def _accept(self, dev_new, dev):
        """A step that lowers the deviance, or changes it by less than the
        tolerance (rounding at convergence can raise it by a hair)."""
        return np.isfinite(dev_new) and (
            dev_new < dev or abs(dev_new - dev) / (0.1 + abs(dev)) < self.iwls_tol)

    # ------------------------------------------------------------ row pass
    def _row_pass_plain(self, out, state, y, X, f, off, c, first):
        """The row pass for a chunk of a model without fixed effects: no
        cells, so the moments are plain weighted sums, in one parallel sweep
        (`_nb_glm_rows_plain`). Up to _SINGLE_BLAS_MAX_COLS columns, each
        block of rows forms its own cross-products on its thread; beyond,
        one BLAS product on all threads forms them."""
        empty = np.zeros(0)
        blockwise = c <= _SINGLE_BLAS_MAX_COLS
        V = np.empty((len(y), c))
        Wb = np.empty((_ROW_BLOCKS if blockwise else 0, c, c))
        parts = np.empty((_ROW_BLOCKS, 6 + c))
        _nb_glm_rows_plain(FAMILY_CODES[self.family.name], bool(state.init), float(self.ybar),
                           y, X, empty if f is None else f, empty if off is None else off,
                           empty if state.init else state.beta, V, Wb, parts)
        p = parts.sum(axis=0)
        out.dev += float(p[0])
        out.ll += float(p[1])
        if first:
            out.sy += float(p[2])
            out.sw += float(p[3])
            out.slg += float(p[4])
        out.wsum += float(p[5])
        out.sv += p[6:]
        out.W += Wb.sum(axis=0) if blockwise else V.T @ V

    def _free_device_rows(self):
        if self._dev_rows is not None:
            self._dev_rows = None
            from .core.gpu import cuda_backend
            getattr(cuda_backend(), "release", lambda: None)()

    def _release_intermediates(self):
        self._dev_rows = None
        super()._release_intermediates()

    def _device_chunks(self, be, yj, xj):
        """The row chunks of a model without fixed effects as device arrays
        (y, X, f, offset). The first pass uploads them and, if they fit in
        the device's free memory with room to spare, keeps them: every later
        step then runs without reading the rows again. If they do not fit,
        each pass uploads them again, a chunk at a time."""
        key = (yj, tuple(xj))
        held = self._dev_rows
        if held is not None and held["key"] == key and held["chunks"] is not None:
            yield from held["chunks"]
            return
        budget = 0.6 * getattr(be, "free_bytes", lambda: float("inf"))()
        keep = held is None or held["key"] != key or held["fits"]
        chunks, used = [], 0
        for ch, _, _ in self._row_chunks(self._columns(yj, xj)):
            d = tuple(None if a is None else be.xp.asarray(a)
                      for a in self._chunk_data(ch, yj, xj))
            used += sum(a.nbytes for a in d if a is not None)
            if keep and used > budget:
                keep, chunks = False, []
                warn("device='cuda': the rows do not fit in the device's free memory, so "
                     "every IRLS step uploads them again", self.logger)
            if keep:
                chunks.append(d)
            yield d
        self._dev_rows = {"key": key, "chunks": chunks if keep else None, "fits": keep}

    def _device_pass(self, out, state, yj, xj, first):
        """`_row_pass_plain` for every chunk on the device: the family's
        terms and the cross-products there (`core.gpu.glm_rows_plain`), and
        only their sums come back."""
        from .core.gpu import cuda_backend, glm_rows_plain
        be = cuda_backend()
        for y, X, f, off in self._device_chunks(be, yj, xj):
            p, sv, W = glm_rows_plain(be, self.family.name, bool(state.init), float(self.ybar),
                                      y, X, f, off, None if state.init else state.beta)
            out.dev += float(p[0])
            out.ll += float(p[1])
            if first:
                out.sy += float(p[2])
                out.sw += float(p[3])
                out.slg += float(p[4])
            out.wsum += float(p[5])
            out.sv += sv
            out.W += W

    def _row_pass(self, state, yj, xj, slot, first=False):
        """Read the rows once at coefficients `state`: the deviance and
        log-likelihood there, and the cell sums of the next weighted
        least-squares step (column 0 the working response z, then the
        covariates), with the within-cell cross-products and each column's
        weighted total sum of squares (`raw`). The cell weights go to the
        buffers of `slot`; the cell sums replace the previous ones."""
        fam, k = self.family, len(xj)
        c = k + 1
        # with one fixed effect and no offset, the model with only the fixed
        # effects fits each group's mean: its log-likelihood (less sum log y!)
        means_null = first and bool(self.fe) and not self.o_fe and self.offset_name is None
        out = SimpleNamespace(dev=0.0, ll=0.0, sy=0.0, sw=0.0, slg=0.0,
                              ll_fe0=0.0 if means_null else None,
                              W=np.zeros((c, c)), sv=np.zeros(c), bss=np.zeros(c), wsum=0.0,
                              n_id=self._buffer(("n_id", slot), (self.M_id,)),
                              sums_id=self._buffer(("sums_id", c), (self.M_id, c)),
                              n_s=self._buffer(("n_s", slot), (self.G_s,)),
                              sums_s=self._buffer(("sums_s", c), (self.G_s, c)))
        if first and self.fe:
            self.codes_s = np.zeros((self.G_s, len(self.o_fe)), np.int64)
        cursor = [0, 0, 0]
        code = FAMILY_CODES[fam.name]
        empty = np.zeros(0)
        on_device = self.device == "cuda" and not self.fe
        if on_device:
            self._device_pass(out, state, yj, xj, first)
        with single_blas(c <= _SINGLE_BLAS_MAX_COLS):
            for ch, starts, codes in (() if on_device
                                      else self._row_chunks(self._columns(yj, xj))):
                lay = self._cell_layout(starts, codes, cursor)
                y, X, f, off = self._chunk_data(ch, yj, xj)
                if lay is None:
                    self._row_pass_plain(out, state, y, X, f, off, c, first)
                    continue
                # one fused sweep: the linear predictor, the family's weight and
                # working response, the deviance and log-likelihood, and the cell
                # moments, with V left holding the within-cell deviations
                if means_null:
                    out.ll_fe0 += fam.loglik_means(y, f, starts)
                a = self._group_effects(state, lay)
                V = np.empty((len(y), c))
                w = np.empty(len(y))
                nc = np.empty(len(lay.cstarts) - 1)
                sc = np.empty((len(nc), c))
                parts = np.empty((_ROW_BLOCKS, 5 + c))
                _nb_glm_rows(code, bool(state.init), float(self.ybar), lay.starts, lay.gcell,
                             lay.cstarts, y, X, empty if f is None else f,
                             empty if off is None else off, codes, self.offs,
                             empty if state.init else state.beta,
                             empty if state.init else state.fe, a, V, w, nc, sc, parts)
                del X, w
                p = parts.sum(axis=0)
                out.dev += float(p[0])
                out.ll += float(p[1])
                if first:
                    out.sy += float(p[2])
                    out.sw += float(p[3])
                    out.slg += float(p[4])
                out.bss += p[5:]
                out.W += V.T @ V
                del V
                out.sv += sc.sum(axis=0)
                out.wsum += nc.sum()
                out.n_id[lay.c0:lay.c0 + lay.nc] = nc[lay.cell_id]
                out.sums_id[lay.c0:lay.c0 + lay.nc] = sc[lay.cell_id]
                out.n_s[lay.s0:lay.s0 + lay.ns] = nc[~lay.cell_id]
                out.sums_s[lay.s0:lay.s0 + lay.ns] = sc[~lay.cell_id]
                if first:
                    self.codes_s[lay.s0:lay.s0 + lay.ns] = codes[starts[:-1][~lay.is_id]]
        if self.fe and (cursor[0], cursor[1], cursor[2]) != (self.M_id, self.G_id, self.G_s):
            raise RuntimeError(
                f"the row files hold {cursor[0]:,} identifying cells in {cursor[1]:,} groups "
                f"and {cursor[2]:,} single-cell groups, where pass 1 found {self.M_id:,}, "
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
            lay = self._cell_layout(starts, codes, cursor)
            y, X, f, off = self._chunk_data(ch, yj, xk)
            eta = self._eta(p.prev, y, X, off, codes, lay)
            _, W, u = fam.irls(y, eta)
            z = eta + u if off is None else eta + u - off
            eta_f = self._eta(p.final, y, X, off, codes, lay)
            mu_f = fam.mu(eta_f)
            after = None
            if self._bias_acc is not None:
                acc = self._bias_acc

                def after(h):
                    acc.add(h, y, eta_f, f, starts, codes)
            return SimpleNamespace(
                y=z, w=W if f is None else W * f, fw=f if fweights else None,
                w_out=None if f is None else f / self._wscale,          # as given
                extra=lambda e: {p.yname: y, "eta": eta_f, "fitted": mu_f, "resid": y - mu_f,
                                 "resid_working": e},
                after=after)
        rows.cols = ([vcol(self.vidx[_OFFSET])] if self.offset_name is not None else [])
        rows.weighted = True
        return rows

    def _iid_factor(self, rss, N, K):
        """The bread alone, with pyfixest's (N - 1) / (N - K)."""
        return (N - 1) / (N - K) if N > K else np.nan


class _BiasSums:
    """The sums of the incidental-parameter bias correction, accumulated in
    pass 2 from its fixed-effect-adjusted regressors x~ and the family's
    per-row terms at the final coefficients (score v, Fisher weight w and
    its derivative z, times the weights): for each corrected dimension, the
    sum over its levels of (sum x~ z) / (sum w) -- alpaca's `groupSums` --
    and with lags, over the units, (sum_l sum_t T / (T - l) x~_t w_t
    v_(t-l)) / (sum w) -- its `groupSumsSpectral`, the rows of each unit in
    period order. The streamed dimension's levels are whole in each chunk;
    the other's are summed into level-sized arrays."""

    def __init__(self, est):
        self.fam, self.lags = est.family, est.bias["lags"]
        dims = est.bias["dims"]
        self.o = [(j, est.n_levels[d]) for j, d in enumerate(est.o_fe) if d in dims]
        self.jt = est.o_fe.index(est.bias["time"]) if self.lags else None
        self.g_sum = self.spec = 0.0
        self.o_num, self.o_den = {}, {}
        self.sizes, self.gaps, self.short = [], 0, 0

    def add(self, h, y, eta, f, starts, codes):
        k = h.shape[1]
        if not k:
            return
        v, w, z = self.fam.bias_terms(y, eta)
        if f is not None:
            v, w, z = v * f, w * f, z * f
        Mz = h * z[:, None]
        Ng = np.diff(starts)
        den = np.add.reduceat(w, starts[:-1])
        ok = den > 0
        num = np.add.reduceat(Mz, starts[:-1], axis=0)
        self.g_sum = self.g_sum + (num[ok] / den[ok, None]).sum(axis=0)
        self.sizes.append(Ng)
        for j, L in self.o:
            if j not in self.o_num:
                self.o_num[j], self.o_den[j] = np.zeros((L, k)), np.zeros(L)
            c = codes[:, j]
            for col in range(k):
                self.o_num[j][:, col] += np.bincount(c, weights=Mz[:, col], minlength=L)
            self.o_den[j] += np.bincount(c, weights=w, minlength=L)
        if not self.lags:
            return
        M = h * w[:, None]
        G, n = len(Ng), len(y)
        gid = np.repeat(np.arange(G), Ng)
        S = np.zeros((G, k))
        for lag in range(1, min(self.lags, n - 1) + 1):
            t = np.flatnonzero(gid[lag:] == gid[:-lag]) + lag
            if not len(t):
                continue
            T = Ng[gid[t]]
            C = M[t] * (v[t - lag] * T / (T - lag))[:, None]
            for col in range(k):
                S[:, col] += np.bincount(gid[t], weights=C[:, col], minlength=G)
        self.spec = self.spec + (S[ok] / den[ok, None]).sum(axis=0)
        same = gid[1:] == gid[:-1]
        self.gaps += len(np.unique(gid[1:][same & (np.diff(codes[:, self.jt]) > 1)]))
        self.short += int((Ng <= self.lags).sum())

    def totals(self):
        total = np.asarray(self.g_sum, dtype=float)
        for j, _ in self.o:
            ok = self.o_den[j] > 0
            total = total + (self.o_num[j][ok] / self.o_den[j][ok, None]).sum(axis=0)
        return 0.5 * total + np.asarray(self.spec, dtype=float)

    def report(self, est):
        """The levels' median sizes, with warnings for very short panels and,
        with lags, for gaps and units no longer than L."""
        off, _ = est._offsets()
        med = {est.g_fe: float(np.median(np.concatenate(self.sizes)))}
        for j, L in self.o:
            d = est.o_fe[j]
            med[d] = float(np.median(est.cnt[off[d]:off[d] + L]))
        small = {d: m for d, m in med.items() if m < 5}
        if small:
            warn("bias_correction: the analytic correction is O(1/T) in the rows per level, "
                 f"and unreliable for very short panels; median rows per level: {small}",
                 est.logger)
        out = {"median_rows_per_level": med}
        if self.lags:
            if self.gaps:
                warn(f"bias_correction: {self.gaps:,} units have gaps in "
                     f"{est.bias['time']!r}; lags are counted in observed periods",
                     est.logger)
            if self.short:
                warn(f"bias_correction: {self.short:,} units have no more periods than "
                     f"lags={self.lags}", est.logger)
            out.update(units_with_gaps=self.gaps, units_not_longer_than_lags=self.short)
        return out

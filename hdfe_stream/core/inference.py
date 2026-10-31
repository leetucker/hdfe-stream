"""Steps 3 and 4: cross-products, collinearity, cluster/dof bookkeeping,
the residual pass, and the per-model result.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import numba as nb
import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from ._columns import GCODE, WCOL, kcol, tcol, vcol
from .kernels_base import (
    _nb_assemble,
    _nb_crv3_finish,
    _nb_crv3_groups,
    _nb_crv3_scatter,
    _nb_pass2,
    _nb_residualize_rows,
    _nb_stack,
)
from .kernels_slopes import _nb_pass2_sl, _nb_residualize_rows_sl
from .report import warn
from .results import HDFEResult
from .utils import _safe, _scatter, _tbl_to_np, iter_group_chunks


def _readonly(a):
    """`a` as a C-contiguous float64 array, read-only (a view if need be):
    Arrow's zero-copy columns are read-only, and numba types read-only and
    writable arrays apart, so `_nb_stack`'s list holds one kind."""
    a = np.ascontiguousarray(a, dtype=np.float64)
    if a.flags.writeable:
        a = a.view()
        a.flags.writeable = False
    return a


# `_stack` writes a chunk's columns into one row-major array. Column by column
# that touches every row's cache line once per column; the blocked parallel
# copy (`_nb_stack`) touches it once. What that saves per element, measured on
# a 16-thread machine, grows with the width: nothing for one column, 0.3 ns
# for two, up to 4 ns from 16 columns (a 39-column, 2-million-row chunk: 0.44 s
# down to 0.04 s). Its cost is the typed list that carries the columns, which
# numba compiles in every process (the compiled code is not cached): about
# 0.2 s. So the copy is used once the saving it would have made adds up to
# that, as a narrow design (AKM's three columns: about 0.1 s over 42 million
# rows) never does and a wide one does in its first chunks.
_SAVED_NS = ((16, 4.0), (8, 2.5), (4, 1.0), (3, 0.6), (2, 0.3))   # (from width, ns/element)
_LIST_COMPILE_S = 0.2
_missed_saving = 0.0
_use_blocked = False


def _stack(chunk, columns):
    """The named columns of a chunk as one C-ordered float64 array, built in a
    single allocation (np.column_stack followed by astype would make two),
    by a blocked parallel copy (`_nb_stack`) where that pays (see above)."""
    global _missed_saving, _use_blocked
    n, k = len(chunk[columns[0]]), len(columns)
    out = np.empty((n, k))
    if not _use_blocked:
        ns = next((v for width, v in _SAVED_NS if k >= width), 0.0)
        _missed_saving += n * k * ns * 1e-9
        _use_blocked = _missed_saving >= _LIST_COMPILE_S
    if _use_blocked and n * k >= 65_536:
        _nb_stack(nb.typed.List([_readonly(chunk[c]) for c in columns]), out)
    else:
        for j, c in enumerate(columns):
            out[:, j] = chunk[c]
    return out


def _criteria(ll, p, N):
    """AIC and BIC of a log-likelihood with `p` parameters and `N` observations."""
    return -2 * ll + 2 * p, -2 * ll + p * np.log(N)


def _ratio(num, den):
    """num / den, or nan when den is zero: a saturated model (a fit made only
    for its effects can be) has no residual df, so its variances are undefined
    rather than an error."""
    return num / den if den else np.nan


# Shared-state contract with the other mixins
# -------------------------------------------
# Reads, set by _PassesMixin: the identifying cells (starts, codes, n, sums,
# cnt, means, tmeans, W_within_cell), the level bookkeeping (n_levels, offs,
# comp), the design totals (N, n_obs, wsum, raw_ss) and the counts that go into
# the result (n_cells, n_identifying, n_components, fe_params, stream_choice).
# From StreamingHDFE: the constructor options, the layout, and _run.
#
# Sets, for _PassesMixin: clusters, cmaps and cluster_reqs. _resolve_clusters
# is called from _pass0_code, before the scan that factorizes the ids.
#
# Hooks a subclass may override (StreamingGLM does): _pass2_rows, where pass 2
# gets each chunk's outcome and weights, and _iid_factor, the scale of the iid
# vcov.


class _InferenceMixin:

    # --------------------------------------------------------------- step 3
    def _assemble(self, gamma):
        """V' M_D V for all variables (m x m)."""
        m, nt = self.m, nb.get_num_threads()
        if self.assembly == "cells":
            acc = np.zeros((nt, m, m))
            _nb_assemble(self.starts, self.codes, self.n, self.sums, self.offs, gamma, acc)
            return self.W_within_cell + acc.sum(axis=0)
        self._log("step 3: row pass for V' M_D V")
        wcol = [WCOL] if self.weights is not None else []
        tcols = [tcol(j + 1) for j in range(len(self.slope_vars))]
        cols = [*self.ccols, *wcol, *tcols, *[vcol(j) for j in range(m)]]
        acc = np.zeros((m, m))
        for ch, starts, codes in self._row_chunks(cols):
            V = _stack(ch, [vcol(j) for j in range(m)])
            w = (np.ascontiguousarray(ch[WCOL], dtype=np.float64) if wcol
                 else np.ones(len(V)))
            # residualize in place, then one BLAS product for the k x k sums
            if tcols:
                _nb_residualize_rows_sl(starts, codes, self.offs, w, self._slope_matrix(ch),
                                        V, gamma, nt)
            else:
                _nb_residualize_rows(starts, codes, self.offs, w, V, gamma, not self.no_fe, nt)
            if wcol:
                V *= np.sqrt(w)[:, None]
            acc += V.T @ V
        return acc

    def _row_chunks(self, cols):
        """Yield (chunk, starts, codes) over the row files.

        With fixed effects each chunk holds complete fe[0] groups and `starts`
        marks them. Without, the rows are never grouped: chunks are plain
        batches and `starts` just cuts each into pieces for the threads (the
        kernels are told not to demean). `codes` stacks the non-streamed
        dimensions' codes, and has no columns when there are none.
        """
        if self.no_fe:
            pieces = 8 * nb.get_num_threads()
            for batch in (b for path in self.paths["rows"]
                          for b in pq.ParquetFile(path).iter_batches(
                              batch_size=self.batch_rows, columns=cols or None)):
                ch = _tbl_to_np(pa.Table.from_batches([batch]))
                n = batch.num_rows
                starts = np.unique(np.linspace(0, n, min(n, pieces) + 1).astype(np.int64))
                yield ch, starts, np.zeros((n, 0), np.int64)
            return
        for ch in iter_group_chunks(self.paths["rows"], [GCODE, *cols], self.batch_rows):
            gc = ch[GCODE]
            starts = np.concatenate(([0], np.flatnonzero(gc[1:] != gc[:-1]) + 1, [len(gc)]))
            codes = (np.column_stack([ch[cc] for cc in self.ccols]).astype(np.int64)
                     if self.ccols else np.zeros((len(gc), 0), np.int64))
            yield ch, starts, codes

    def _slope_matrix(self, ch):
        """T = [1, centered slope variables] for a chunk of rows."""
        n = len(ch[GCODE])
        T = np.empty((n, self.p))
        T[:, 0] = 1.0
        for j, mu in enumerate(self.tmeans):
            T[:, j + 1] = ch[tcol(j + 1)] - mu
        return T

    # ------------------------------------------------------- clusters / dof
    def _resolve_clusters(self, keys, extra):
        """Turn cluster requests into cluster *terms*.

        A one-way request 'a' is one term; a multi-way request 'a+b' expands
        to the terms a, b and a^b (and all higher intersections for three or
        more ways), combined by inclusion-exclusion (Cameron, Gelbach and
        Miller 2011), as in pyfixest/fixest. Each term is one of:
          seg     the fe[0] groups themselves: scores summed per group
          segsub  a refinement of fe[0] (e.g. worker_id^firm_id): rows of a cluster
                  are always in the same chunk, so scores are summed per
                  (group, code) inside each chunk; no global array
          fe      another FE dimension: its codes, level-sized accumulator
          extra   anything else: factorized in pass 0, level-sized accumulator
        """
        reqs = [k.split(":", 1)[1] for k in keys if k.startswith(("CRV1:", "CRV3:"))]
        reqs = list(dict.fromkeys(reqs + list(extra)))
        # the requests that also get the cluster jackknife (one term each)
        self.crv3_reqs = [k.split(":", 1)[1] for k in keys if k.startswith("CRV3:")]
        # with no fixed effects there are no groups to be nested in
        g0 = set(self.fe_cols[self.g_fe]) if self.fe else None
        dim_of = {frozenset(self.fe_cols[d]): d for d in self.o_fe}
        self.clusters, self.cmaps, self.cluster_reqs = {}, {}, {}

        def cmap(cols):
            name = "^".join(cols)
            if name not in self.cmaps:
                self.cmaps[name] = {"cols": list(cols), "code": kcol(len(self.cmaps)),
                                    "map": str(self.workdir / f"cluster_{_safe(name)}.parquet")}
            return name

        def term(cols):
            key = "^".join(cols)
            if key in self.clusters:
                return key
            cs = set(cols)
            if g0 is not None and cs == g0:
                spec = {"kind": "seg", "dim": self.g_fe, "code": GCODE}
            elif g0 is not None and cs > g0:
                xcols = [c for c in cols if c not in g0]
                xdim = dim_of.get(frozenset(xcols))
                spec = {"kind": "segsub", "xdim": xdim,
                        "code": self.code_of[xdim] if xdim else None,
                        "map": None if xdim else cmap(xcols)}
                if not xdim:
                    spec["code"] = self.cmaps[spec["map"]]["code"]
            elif frozenset(cols) in dim_of:
                d = dim_of[frozenset(cols)]
                spec = {"kind": "fe", "dim": d, "code": self.code_of[d]}
            else:
                name = cmap(cols)
                spec = {"kind": "extra", "map": name, "code": self.cmaps[name]["code"]}
            spec["cols"] = list(cols)
            self.clusters[key] = spec
            return key

        from itertools import combinations
        for req in reqs:
            parts = [[c.strip() for c in p.split("^")] for p in req.split("+")]
            terms = []
            for r in range(1, len(parts) + 1):
                for combo in combinations(range(len(parts)), r):
                    cols = list(dict.fromkeys(c for i in combo for c in parts[i]))
                    terms.append((term(cols), (-1) ** (r + 1), r == 1))
            self.cluster_reqs[req] = terms

    def _nested_dims(self, key):
        """FE dimensions whose every level lies within a single cluster of
        this term (not counted in K for CRV1, as in fixest/pyfixest)."""
        spec = self.clusters[key]
        ccols = set(spec["cols"])
        out = []
        for d in self.fe:
            if ccols <= set(self.fe_cols[d]):
                out.append(d)
                continue
            if spec["kind"] == "segsub":    # no single code column; subset rule only
                continue
            # a level is inside one cluster iff min == max of the cluster code
            # over its rows; min/max keep the aggregation state O(1) per level
            ccol, dcol = spec["code"], self.code_of[d]
            q = [pl.col(ccol).min().alias("lo"), pl.col(ccol).max().alias("hi")]
            chk = (pl.col("lo") != pl.col("hi")).any()
            if d == self.g_fe:      # bucket by bucket: rows are bucketed by fe[0]
                ok = not any(self._collect(pl.scan_parquet(f).group_by(GCODE).agg(q)
                                           .select(chk), f"nesting {d}", engine="streaming").item()
                             for f in self.paths["rows"])
            else:
                ok = not self._collect(pl.scan_parquet(self.paths["rows"]).group_by(dcol)
                                       .agg(q).select(chk), f"nesting {d}",
                                       engine="streaming").item()
            if ok:
                out.append(d)
        return out

    def _check_crv3(self, nested):
        """Refuse CRV3 where the jackknife cannot be had from one pass.

        CRV3 re-estimates beta leaving out one cluster at a time. On data with
        the fixed effects already removed, dropping a cluster's rows gives
        that re-estimate exactly only if no other cluster's fixed effects
        depend on those rows -- that is, if every fixed effect is nested
        within the clusters (or there are none). Otherwise the exact
        jackknife re-estimates the fixed effects once per cluster, which is
        pyfixest's approach and not one that scales here.
        """
        if not self.crv3_reqs:
            return
        if any(model.get("iv") for model in self.models):
            raise ValueError("CRV3 standard errors are not available for IV (2SLS) "
                             "models, as in pyfixest; use CRV1")
        for req in self.crv3_reqs:
            (term, _, _), = self.cluster_reqs[req]
            loose = [d for d in self.fe if d not in nested[term]]
            if loose:
                raise ValueError(
                    f"CRV3 standard errors need every fixed effect to be nested within "
                    f"the clusters, and {', '.join(map(repr, loose))} "
                    f"{'is' if len(loose) == 1 else 'are'} not nested within {req!r}. "
                    "With fixed effects that are not nested, only CRV1 is supported: "
                    "the jackknife would have to re-estimate the fixed effects once "
                    "per cluster. pyfixest supports CRV3 in this case, by refitting "
                    "the model for each cluster, if the data fit in memory.")

    def _cluster_levels(self, term, paths):
        """The level values of a cluster term, one row per cluster code, in
        code order."""
        spec = self.clusters[term]
        if spec["kind"] == "seg":
            # the fe[0] file has one row per group, in group-code order
            return pl.read_parquet(paths["fe"][self.g_fe], columns=self.fe_cols[self.g_fe])
        if spec["kind"] == "fe":
            path, code = self.paths["maps"][spec["dim"]], spec["code"]
        else:
            path, code = self.cmaps[spec["map"]]["map"], self.cmaps[spec["map"]]["code"]
        return pl.read_parquet(path).sort(code).select(spec["cols"])

    def _check_boot(self, nested):
        """Refuse a bootstrap clustering (`bootcluster=`) the wild cluster
        bootstrap cannot be exact for: IV, clusters within the streamed
        dimension's groups, and fixed effects not nested within the clusters.

        The bootstrap works on the data with the fixed effects removed. A
        draw multiplies each cluster's residuals by one weight, which leaves
        them orthogonal to a fixed effect whose levels each lie within one
        cluster, but not to one that spans clusters: the exact bootstrap
        would then re-estimate the fixed effects for every draw.
        """
        req = getattr(self, "boot_req", None)
        if req is None:
            return
        if any(model.get("iv") for model in self.models):
            raise ValueError("the wild cluster bootstrap (bootcluster=) is not available "
                             "for IV (2SLS) models, as in pyfixest")
        (term, _, _), = self.cluster_reqs[req]
        if self.clusters[term]["kind"] == "segsub":
            raise ValueError(f"bootcluster={req!r}: clusters finer than the "
                             f"{self.g_fe!r} groups cannot have the fixed effects nested "
                             "within them")
        loose = [d for d in self.fe if d not in nested[term]]
        if loose:
            raise ValueError(
                f"the wild cluster bootstrap needs every fixed effect to be nested "
                f"within the clusters, and {', '.join(map(repr, loose))} "
                f"{'is' if len(loose) == 1 else 'are'} not nested within {req!r}: "
                "its levels span clusters, so each draw would have to re-estimate "
                "the fixed effects. A fixed effect with few levels, such as year, can "
                "go in as covariates instead, with the same estimates of the other "
                "coefficients: 'y ~ treat + i(year) | county' rather than "
                "'y ~ treat | county + year'.")

    def _redundant_slopes(self):
        """Slope variables that are a function of another FE dimension (e.g.
        worker trends in `year` alongside year effects). The worker slopes on
        such a variable sum to a trend that the other dimension already
        spans, so one of them is redundant; each counts as one more
        restriction in the FE degrees of freedom. Found once per fit."""
        if self.slope_redundancy is not None:
            return self.slope_redundancy
        out = []
        for j, v in enumerate(self.slope_vars):
            for d in self.o_fe:
                spread = self._collect(
                    pl.scan_parquet(self.paths["rows"])
                      .group_by(self.code_of[d])
                      .agg((pl.col(tcol(j + 1)).max() - pl.col(tcol(j + 1)).min()).alias("r"))
                      .select(pl.col("r").max()), f"slope spread {v}",
                      engine="streaming").item()
                if spread == 0:
                    out.append((v, d))
                    break
        self.slope_redundancy = out
        return out

    def _collinear(self, Axx, idx=None, raw=None):
        """pyfixest's collinearity check on the residualized X'X, plus a
        relative check: a variable whose residual sum of squares is below
        collin_tol_rel of its raw (centered) sum of squares is treated as
        absorbed by the fixed effects. The iterative solve leaves noise of
        order sqrt(tol * condition number) in exactly collinear variables,
        so an absolute test alone can miss them.

        `raw` is those raw sums of squares, on the same weighting as Axx;
        by default the variables' own from pass 0, looked up by `idx`."""
        k = Axx.shape[0]
        if k == 0:
            return np.zeros(0, bool)
        mask = self._collinear_pf(Axx)
        if raw is None and idx is not None:
            raw = self.raw_ss[idx]
        if raw is not None:
            mask |= (raw > 0) & (np.diag(Axx) <= self.collin_tol_rel * raw)
            if mask.any() and not mask.all():   # re-check the rest without them
                keep = np.flatnonzero(~mask)
                mask[keep] = self._collinear_pf(Axx[np.ix_(keep, keep)])
        return mask

    def _collinear_pf(self, Axx):
        k = Axx.shape[0]
        try:
            from pyfixest.core import find_collinear_variables
            mask, _, _ = find_collinear_variables(np.ascontiguousarray(Axx),
                                                            self.collin_tol)
            mask = np.asarray(mask, bool)
        except ImportError:   # fallback: sequential pivot test on the Cholesky
            mask = np.zeros(k, bool)
            for j in range(k):
                keep = np.flatnonzero(~mask[:j])
                r = Axx[j, j]
                if len(keep):
                    r -= Axx[j, keep] @ np.linalg.solve(Axx[np.ix_(keep, keep)], Axx[keep, j])
                mask[j] = r <= self.collin_tol * max(Axx[j, j], 1e-300)
        return mask

    # --------------------------------------------------------------- step 4
    def _pass2_rows(self, yj, xk):
        """Where pass 2 gets each chunk's outcome and weights.

        Returns a function of (chunk, starts, codes) giving y and w, the
        frequency weights `fw` (None unless fweights: the heteroskedastic
        meat divides each row's squared score by its frequency), `w_out` for
        the residual file's weights column (None for none), and `extra`, a
        function of the residuals giving further residual-file columns, or
        None. Its attributes: `cols`, the extra row columns it reads, and
        `weighted`. For OLS these are the variable and the user's weights.
        """
        weighted = self.weights is not None
        fweights = weighted and self.weights_type == "fweights"

        def rows(ch, starts, codes):
            y = np.ascontiguousarray(ch[vcol(yj)], dtype=np.float64)
            w = (np.ascontiguousarray(ch[WCOL], dtype=np.float64) if weighted
                 else np.ones(len(y)))
            return SimpleNamespace(y=y, w=w, fw=w if fweights else None,
                                   w_out=w if weighted else None, extra=None)
        rows.cols, rows.weighted = [], weighted
        return rows

    def _iid_factor(self, rss, N, K):
        """Scale of the iid vcov, times (X~' W X~)^-1: the residual variance."""
        return _ratio(rss, N - K)

    def _pass2(self, tag, ycol_name, gamma, yj, xk, beta, zk, Pi, bread, ivd=None):
        """Row pass for one model: fe[0] effects, residuals, moments, meats;
        writes the residual file and one FE file per dimension.

        xk: regressor columns (residuals use X beta); zk, Pi: instrument
        columns and first-stage coefficients, so the scores use Pi' z~
        (for OLS, zk = xk and Pi = I). bread: the assembled X~' W X~, which
        the CRV3 downdate needs before the pass rather than after it.
        ivd: for an IV model, the plan of its diagnostics' scores (see
        `_iv_plan`), whose meats this pass accumulates into it.
        """
        mdir = self.workdir / "models" / tag
        mdir.mkdir(parents=True, exist_ok=True)
        paths = {"dir": str(mdir),
                 "resid": str(mdir / "resid.parquet") if self.save_resid else None,
                 "fe": {d: str(mdir / f"fe_{_safe(d)}.parquet") for d in self.fe}}
        off, _ = self._offsets()
        k, q, nt = len(xk), len(zk), nb.get_num_threads()
        L = gamma.shape[0]
        # an IV model's diagnostics also need the endogenous regressors
        # residualized: they ride along with the instruments
        zka = list(zk) + (list(ivd["dcols"]) if ivd else [])
        qa = len(zka)
        gam_x = np.ascontiguousarray(gamma[:, xk]) if k else np.zeros((L, 0))
        gam_z = np.ascontiguousarray(gamma[:, zka]) if qa else np.zeros((L, 0))
        gam_y0 = np.ascontiguousarray(gamma[:, yj])
        gam_y = gam_y0 - gam_x @ beta                  # FEs of the y-equation
        beta_c = np.ascontiguousarray(beta, dtype=np.float64)
        Pi_c = np.ascontiguousarray(Pi, dtype=np.float64).reshape(q, k)
        rows = self._pass2_rows(yj, xk)
        weighted = rows.weighted

        acc_s = np.zeros((nt, 4))
        acc_B = np.zeros((k, k))
        acc_hc = np.zeros((k, k))
        acc_g = np.zeros((k, k))
        need_seg = any(spec["kind"] == "seg" for spec in self.clusters.values())
        # CRV3 terms (checked already: OLS, all fixed effects nested). A term
        # that is the fe[0] groups is jackknifed chunk by chunk; any other
        # collects per-cluster sums through the pass, G x k^2 of them.
        crv3 = {}
        for req in (self.crv3_reqs if k else []):
            (t, _, _), = self.cluster_reqs[req]
            spec = self.clusters[t]
            if spec["kind"] == "seg":
                crv3[t] = {"acc": np.zeros((nt, k, k))}
            else:
                need = spec["G"] * k * (k + 1) * 8 / 1e9
                if not self.budget.fits(need):
                    raise MemoryError(
                        f"CRV3 by {req!r} needs {need:.1f} GB for per-cluster sums "
                        f"({spec['G']:,} clusters x {k} covariates squared) and "
                        f"{self.budget.headroom_gb():.1f} GB is free ({self.budget}); use CRV1, "
                        "fewer covariates, or raise max_memory_gb")
                crv3[t] = {"AG": np.zeros((spec["G"], k, k)), "sG": np.zeros((spec["G"], k))}
        bread_c = np.ascontiguousarray(bread, dtype=np.float64)
        # bootcluster=: per-cluster sum w h h' and w e h, and the rows per
        # cluster, for the wild cluster bootstrap (checked already: OLS, all
        # fixed effects nested); shared with CRV3 when it sums the same term
        boot = None
        if getattr(self, "boot_req", None) and k:
            (t, _, _), = self.cluster_reqs[self.boot_req]
            spec = self.clusters[t]
            if t in crv3 and "AG" in crv3[t]:
                boot = {"term": t, "AG": crv3[t]["AG"], "sG": crv3[t]["sG"], "shared": True}
            else:
                need = spec["G"] * k * (k + 1) * 8 / 1e9
                if not self.budget.fits(need):
                    raise MemoryError(
                        f"bootcluster={self.boot_req!r} needs {need:.1f} GB for per-cluster "
                        f"sums ({spec['G']:,} clusters x {k} covariates squared) and "
                        f"{self.budget.headroom_gb():.1f} GB is free ({self.budget}); "
                        "raise max_memory_gb")
                boot = {"term": t, "AG": np.zeros((spec["G"], k, k)),
                        "sG": np.zeros((spec["G"], k)), "shared": False}
            boot["n"] = np.zeros(spec["G"], np.int64)
        # OLS: the instruments are the regressors (Pi = I), so they share one
        # array and the scores are the residualized regressors themselves
        ols = list(zk) == list(xk)
        scores = {t: np.zeros((spec["G"], k)) for t, spec in self.clusters.items()
                  if spec["kind"] in ("fe", "extra")}
        subs = {t: spec for t, spec in self.clusters.items() if spec["kind"] == "segsub"}
        meat_sub = {t: np.zeros((k, k)) for t in subs}
        G_sub = {t: 0 for t in subs}
        ccodes = [spec["code"] for spec in self.clusters.values() if spec["kind"] != "seg"]
        src = [c for c in self._src_cols() if c not in self.keep]
        tcols = [tcol(j + 1) for j in range(len(self.slope_vars))]
        cols = list(dict.fromkeys([GCODE, *self.ccols, *ccodes, *src, *self.keep, *tcols,
                                   *([WCOL] if self.weights is not None else []), vcol(yj),
                                   *[vcol(j) for j in xk], *[vcol(j) for j in zka],
                                   *rows.cols]))
        g_cols = self.fe_cols[self.g_fe] if self.fe else []
        yc = float(self.means[yj])
        r_writer = g_writer = None
        for ch, starts, codes in self._row_chunks([c for c in cols if c != GCODE]):
            r = rows(ch, starts, codes)
            y, w = r.y, r.w
            n = len(y)
            X = _stack(ch, [vcol(j) for j in xk]) if k else np.zeros((n, 0))
            Z = X if ols else (_stack(ch, [vcol(j) for j in zka]) if qa else np.zeros((n, 0)))
            G = len(starts) - 1
            e = np.empty(n)
            zt = np.empty((n, qa))
            sg = np.empty((G if need_seg else 0, qa))
            if tcols:
                T = self._slope_matrix(ch)
                g_coef = np.empty((G, self.p))
                _nb_pass2_sl(starts, codes, self.offs, w, T, y, X, beta_c, gam_y, gam_y0, Z,
                             gam_z, yc, g_coef, e, zt, sg, acc_s)
                # back to the original scale of the slope variables
                g_eff = g_coef[:, 0] - g_coef[:, 1:] @ self.tmeans
                g_row = np.einsum("ij,ij->i", T, np.repeat(g_coef, np.diff(starts), axis=0))
            else:
                g_eff = np.empty(G)
                _nb_pass2(starts, codes, self.offs, w, y, X, beta_c, gam_y, gam_y0, Z, gam_z,
                          yc, not self.no_fe, g_eff, e, zt, sg, acc_s)
                g_row = None
            Ng = np.diff(starts)
            if ivd is not None:
                self._iv_accumulate(ivd, ch, starts, zt, e, w, r.fw)
                zt = zt[:, :q]
                sg = sg[:, :q]

            # The k x k sums, as BLAS products over the chunk rather than k x k
            # scalar updates per row, which fall out of cache once k is in the
            # hundreds. h = Pi' z~ are the scores (for OLS, z~ itself); one
            # n x k buffer is reused for each weighting in turn.
            h = zt if ols else zt @ Pi_c
            del zt
            if getattr(r, "after", None) is not None:
                r.after(h)                  # a subclass's own sums over the scores
            buf = np.empty_like(h)
            if weighted:
                sw = np.sqrt(w)
                np.multiply(h, sw[:, None], out=buf)
                acc_B += buf.T @ buf                              # sum w h h'
                # w e, or w e / sqrt(fw) with frequency weights (sqrt(w) e
                # when they are the only weights)
                me = (w if r.fw is None else sw if r.fw is w else w / np.sqrt(r.fw)) * e
                np.multiply(h, me[:, None], out=buf)
            else:
                acc_B += h.T @ h
                np.multiply(h, e[:, None], out=buf)
            acc_hc += buf.T @ buf               # sum w^2 e^2 h h' (w e^2 h h', fweights)
            if need_seg:
                if not ols:
                    sg = sg @ Pi_c
                acc_g += sg.T @ sg              # group sums of w e h, from the kernel
            for t, c3 in crv3.items():
                if "acc" in c3:
                    _nb_crv3_groups(starts, w, e, h, bread_c, c3["acc"])
                else:
                    _nb_crv3_scatter(ch[self.clusters[t]["code"]].astype(np.int64), w, e, h,
                                     c3["AG"], c3["sG"])
            if boot is not None:
                bc = ch[self.clusters[boot["term"]]["code"]].astype(np.int64)
                if not boot["shared"]:
                    _nb_crv3_scatter(bc, w, e, h, boot["AG"], boot["sG"])
                boot["n"] += np.bincount(bc, minlength=len(boot["n"]))
            if scores or subs:
                he = np.multiply(h, (w * e)[:, None], out=buf)    # scores w e h
            del h
            if scores or subs:
                for t, sc in scores.items():
                    sc += _scatter(ch[self.clusters[t]["code"]].astype(np.int64), he, sc.shape[0])
                if subs:
                    local = np.repeat(np.arange(G, dtype=np.int64), Ng)
                for t, spec in subs.items():
                    # clusters nested in fe[0] groups are complete within the chunk
                    key = local * spec["xL"] + ch[spec["code"]].astype(np.int64)
                    uk, inv = np.unique(key, return_inverse=True)
                    S = _scatter(inv, he, len(uk))
                    meat_sub[t] += S.T @ S
                    G_sub[t] += len(uk)

            if self.save_resid:
                out = {c: ch[c] for c in src}
                out.update({c: ch[c] for c in self.keep})
                if r.w_out is not None:
                    out["weights"] = r.w_out
                out[ycol_name] = y
                # row-level fe[0] contribution (intercept + slopes * variables)
                if self.fe:
                    out[f"fe_{self.g_fe}"] = np.repeat(g_eff, Ng) if g_row is None else g_row
                for j, f in enumerate(self.o_fe):
                    out[f"fe_{f}"] = gam_y[self.offs[j] + codes[:, j]]
                out.update({"xb": X @ beta_c, "resid": e})
                if r.extra is not None:
                    out.update(r.extra(e))
                tbl = pa.table(out)
                if r_writer is None:
                    r_writer = pq.ParquetWriter(paths["resid"], tbl.schema)
                r_writer.write_table(tbl, row_group_size=self.rgs)

            if not self.fe:
                continue
            slopes_out = ({f"fe_{self.g_fe}[{v}]": g_coef[:, j + 1]
                           for j, v in enumerate(self.slope_vars)} if tcols else {})
            gt = pa.table({**{c: ch[c][starts[:-1]] for c in g_cols},
                           f"fe_{self.g_fe}": g_eff, **slopes_out, "n_obs": Ng.astype(np.int64)})
            if g_writer is None:
                g_writer = pq.ParquetWriter(paths["fe"][self.g_fe], gt.schema)
            g_writer.write_table(gt, row_group_size=self.rgs)
        if r_writer is not None:
            r_writer.close()
        if g_writer is not None:
            g_writer.close()

        for f in self.o_fe:
            nl, o = self.n_levels[f], off[f]
            # `component` is only defined for the dimension the connected
            # components were computed on (see _components)
            extra = {"component": pl.Series(self.comp)} if f == self.o_fe[0] else {}
            (pl.scan_parquet(self.paths["maps"][f])
               .with_columns(**{f"fe_{f}": pl.Series(gam_y[o:o + nl]),
                                "n_obs": pl.Series(self.cnt[o:o + nl]).cast(pl.Int64)}, **extra)
               # the dense code is an internal label; the level values identify
               # the rows for the caller
               .drop(self.code_of[f])
               .sink_parquet(paths["fe"][f], engine="streaming"))

        meat = {t: sc.T @ sc for t, sc in scores.items()}
        meat.update(meat_sub)
        for t, spec in self.clusters.items():
            if spec["kind"] == "seg":
                meat[t] = acc_g
            elif spec["kind"] == "segsub":
                spec["G"] = G_sub[t]
        rss, tss_w, sy, syy = acc_s.sum(axis=0)
        ssy = syy - sy * sy / self.wsum     # weighted total SS around the weighted mean
        jack = {}                           # CRV3: sum over clusters of d_g d_g'
        for t, c3 in crv3.items():
            if "acc" not in c3:
                c3["acc"] = np.zeros((nt, k, k))
                _nb_crv3_finish(bread_c, c3["AG"], c3["sG"], c3["acc"])
            jack[t] = c3["acc"].sum(axis=0)
        return rss, tss_w, ssy, acc_B, acc_hc, meat, jack, paths, boot

    # ------------------------------------------------------------ IV diagnostics
    def _iv_plan(self, fml, names, znames, zk, xk, yj, A, ctx):
        """The sums an IV model's diagnostics (`core.ivtests`) need, from the
        assembled cross-products, and which meats pass 2 accumulates for
        them: none for an iid vcov, the heteroskedastic one, or the cluster
        terms of the default vcov. The included exogenous regressors (those
        that are also instruments) are partialled out of the excluded
        instruments Z1 and of the endogenous regressors Y and the outcome y
        (the reduced form, for the Anderson-Rubin test) by Gz and Gy. The
        endogeneity test needs the OLS coefficients, b_ols, and where each
        regressor is among the residualized columns (xpos)."""
        exog = [x for x in names if x in znames]
        endog = [x for x in names if x not in znames]
        x2 = [i for i, z in enumerate(znames) if z in exog]
        z1 = [i for i, z in enumerate(znames) if z not in exog]
        ycols = [self.vidx[x] for x in endog]
        dcols = ycols + [yj]                      # Y, then y
        zj = np.asarray(zk, dtype=np.int64)
        a2, a1 = zj[x2], zj[z1]
        iX2 = np.linalg.pinv(A[np.ix_(a2, a2)]) if len(a2) else np.zeros((0, 0))
        Gz = iX2 @ A[np.ix_(a2, a1)]
        Gy = iX2 @ A[np.ix_(a2, dcols)]
        ZZ = A[np.ix_(a1, a1)] - A[np.ix_(a1, a2)] @ Gz
        ZY = A[np.ix_(a1, dcols)] - A[np.ix_(a1, a2)] @ Gy
        YY = A[np.ix_(dcols, dcols)] - A[np.ix_(a2, dcols)].T @ Gy
        q, L1, Ke = len(zk), len(z1), len(endog)
        b_ols = np.linalg.pinv(A[np.ix_(xk, xk)]) @ A[xk, yj]
        xpos = [znames.index(x) if x in exog else q + endog.index(x) for x in names]
        m = 2 * q + L1 * Ke + L1 * (Ke + 1) + Ke
        default = ctx["default"]
        terms = []
        if default.startswith(("CRV1:", "CRV3:")):
            terms = [t for t, _, _ in self.cluster_reqs[default.split(":", 1)[1]]]
        scores = {}
        for t in terms:
            spec = self.clusters[t]
            if spec["kind"] in ("fe", "extra"):
                need = spec["G"] * m * 8 / 1e9
                if not self.budget.fits(need):
                    warn(f"{fml}: the IV diagnostics by {t!r} need {need:.1f} GB of cluster "
                         f"sums and {self.budget.headroom_gb():.1f} GB is free; not computed "
                         "(raise max_memory_gb)", self.logger)
                    return None
                scores[t] = np.zeros((spec["G"], m))
        return {"ycols": ycols, "dcols": dcols, "x2": x2, "z1": z1, "Gz": Gz, "Gy": Gy,
                "Pi1": np.linalg.pinv(ZZ) @ ZY, "ZZ": ZZ, "ZY": ZY, "YY": YY, "endog": endog,
                "exog": exog, "b_ols": b_ols, "xpos": xpos,
                "hetero": default == "hetero", "terms": terms, "scores": scores,
                "meat": {t: np.zeros((m, m)) for t in terms}, "acc_hc": np.zeros((m, m))}

    def _iv_accumulate(self, ivd, ch, starts, zt, e, w, fw):
        """Pass 2's part for the IV diagnostics, one chunk: each row's scores
        w [e z~, vec(z1~ y~'), vec(z1~ [v vy]'), e_ols z~, e_ols x~_endog]: the
        instruments' moments at the 2SLS residuals; the first stage's at Y
        and at its residuals V, with the reduced form's residual vy; and the
        instruments' and the endogenous regressors' moments at the OLS
        residuals. Summed into the meats of `_iv_plan`. `zt` holds the
        residualized instruments, then the endogenous regressors and y."""
        if not (ivd["hetero"] or ivd["terms"]):
            return
        Ke = len(ivd["endog"])
        q = zt.shape[1] - Ke - 1
        Z, D = zt[:, :q], zt[:, q:]
        X2 = Z[:, ivd["x2"]]
        Z1 = Z[:, ivd["z1"]] - X2 @ ivd["Gz"]
        Ya = D - X2 @ ivd["Gy"]
        Va = Ya - Z1 @ ivd["Pi1"]
        n = len(e)
        U = (Ya[:, :Ke, None] * Z1[:, None, :]).reshape(n, -1)
        V = (Va[:, :, None] * Z1[:, None, :]).reshape(n, -1)
        e_ols = D[:, Ke] - np.hstack([Z, D[:, :Ke]])[:, ivd["xpos"]] @ ivd["b_ols"]
        we, wo = (w * e)[:, None], (w * e_ols)[:, None]
        s = np.hstack([Z * we, U * w[:, None], V * w[:, None], Z * wo, D[:, :Ke] * wo])
        if ivd["hetero"]:
            # frequency weights: each row stands for fw identical ones
            sh = s if fw is None else s / np.sqrt(fw)[:, None]
            ivd["acc_hc"] += sh.T @ sh
        for t in ivd["terms"]:
            spec = self.clusters[t]
            if spec["kind"] == "seg":
                sg = np.add.reduceat(s, starts[:-1], axis=0)
                ivd["meat"][t] += sg.T @ sg
            elif spec["kind"] == "segsub":
                local = np.repeat(np.arange(len(starts) - 1, dtype=np.int64), np.diff(starts))
                key = local * spec["xL"] + ch[spec["code"]].astype(np.int64)
                uk, inv = np.unique(key, return_inverse=True)
                S = _scatter(inv, s, len(uk))
                ivd["meat"][t] += S.T @ S
            else:
                sc = ivd["scores"][t]
                sc += _scatter(ch[spec["code"]].astype(np.int64), s, sc.shape[0])

    def _iv_diag(self, ivd, A, beta, V, names, xk, zk, yj, rss, ctx, n_clusters):
        """The IV diagnostics under the default vcov (`core.ivtests`):
        Kleibergen-Paap (Anderson / Cragg-Donald under iid) and
        Sanderson-Windmeijer identification tests, partial and Shea partial
        R^2, the effective F (one endogenous regressor), the Anderson-Rubin
        test and confidence set, the endogeneity (C) test, and Hansen's J
        (Sargan's statistic under iid) when overidentified. The F statistics
        count the fixed effects' dof as partialled out (ivreg2's sdofminus),
        less those nested in the clusters."""
        from .ivtests import (
            anderson_rubin,
            c_statistic,
            hansen_j,
            identification,
            iid_omegas,
            sargan,
        )
        N, default = self.N, ctx["default"]
        q, k = len(zk), len(xk)
        Ke, L1 = len(ivd["endog"]), len(ivd["z1"])
        m1, ma = L1 * Ke, L1 * (Ke + 1)
        ZX, Zy = A[np.ix_(zk, xk)], A[zk, yj]
        fz = list(zk) + list(ivd["ycols"])          # the instruments, and Y as exogenous
        ZXf, Zyf = A[np.ix_(fz, xk)], A[fz, yj]
        G, dof_extra = None, ctx["k_fe"]
        if default == "iid":
            OUa, OVa = iid_omegas(ivd["ZZ"], ivd["ZY"], ivd["YY"], N)
            OU = OUa[:m1, :m1]
            over = ("sargan", *sargan(A[np.ix_(zk, zk)], Zy - ZX @ beta, rss, N, k))
            b = ivd["b_ols"]
            rss_ols = A[yj, yj] - 2 * b @ A[xk, yj] + b @ A[np.ix_(xk, xk)] @ b
            Sf = rss_ols / N * A[np.ix_(fz, fz)]
        else:
            if default == "hetero":
                Om = ivd["acc_hc"]
            else:
                req = default.split(":", 1)[1]
                Om = np.zeros_like(ivd["acc_hc"])
                for t, sign, _ in self.cluster_reqs[req]:
                    M = ivd["meat"][t]
                    if t in ivd["scores"]:
                        M = M + ivd["scores"][t].T @ ivd["scores"][t]
                    Om += sign * M
                G = min(n_clusters[req])
                nest = list(dict.fromkeys(d for t, _, _ in self.cluster_reqs[req]
                                          for d in ctx["nested"][t]))
                dof_extra = (self._k_fe(ctx["fe_dof"], nest)
                             - sum(self.fe_params[d] for d in nest) + len(nest))
            Om = (Om + Om.T) / 2
            OU = Om[q:q + m1, q:q + m1]
            o = q + m1
            OVa = Om[o:o + ma, o:o + ma]
            Sf = Om[o + ma:, o + ma:]
            over = ("hansen_j", *hansen_j(ZX, Zy, Om[:q, :q]))
        f = {"n_obs": N, "n_inst": q, "n_clusters": G, "dof_extra": dof_extra}
        out = identification(ivd["ZZ"], ivd["ZY"][:, :Ke], ivd["YY"][:Ke, :Ke], OU,
                             OVa[:m1, :m1], endog=ivd["endog"], exog=ivd["exog"], sw="2sls",
                             **f)
        out["vcov"] = default
        name, stat, df, p = over
        out["overid"] = ({"test": name, "statistic": stat, "df": df, "p": p} if df > 0
                         else None)
        ctr = scl = None
        if Ke == 1:
            i = names.index(ivd["endog"][0])
            ctr, scl = float(beta[i]), float(np.sqrt(max(V[i, i], 0.0)))
        out["anderson_rubin"] = anderson_rubin(ivd["ZZ"], ivd["ZY"], OVa, center=ctr,
                                               scale=scl, **f)
        C, dfc, pc = c_statistic(ZXf, Zyf, (Sf + Sf.T) / 2, range(q, q + Ke))
        out["endogeneity"] = {"statistic": C, "df": dfc, "p": pc, "tested": ivd["endog"]}
        return out

    # ------------------------------------------------------------ estimation
    def _estimate(self, tag, model, A, gamma, ctx, beta=None):
        """OLS or 2SLS for one model from the assembled V' M_D V, then its
        row pass and variance estimates. With `beta`, the coefficients are
        those (not solved for), as after a bias correction."""
        fml, yname = model["fml"], model["y"]
        yj = self.vidx[yname]
        xj = [self.vidx[x] for x in model["x"]]
        mask = self._collinear(A[np.ix_(xj, xj)], xj)
        if len(xj) and mask.all():
            raise ValueError(f"{fml}: all covariates are collinear with the fixed effects")
        dropped = [x for x, m_ in zip(model["x"], mask) if m_]
        xk = [j for j, m_ in zip(xj, mask) if not m_]
        names = [x for x, m_ in zip(model["x"], mask) if not m_]
        k = len(xk)
        iv = model.get("iv")
        first_stage, f_stats = [], []
        if iv:
            lost = [e for e in iv["endog"] if e in dropped]
            if lost:
                raise ValueError(f"{fml}: endogenous variables {lost} are collinear with "
                                 "the fixed effects / exogenous regressors")
            zj = [self.vidx[z] for z in iv["z"]]
            zmask = self._collinear(A[np.ix_(zj, zj)], zj)
            zdrop = [z for z, m_ in zip(iv["z"], zmask) if m_]
            dropped += [z for z in zdrop if z not in dropped]
            zk = [j for j, m_ in zip(zj, zmask) if not m_]
            znames = [z for z, m_ in zip(iv["z"], zmask) if not m_]
            if len(zk) < k:
                raise ValueError(f"{fml}: under-identified ({len(zk)} instruments for "
                                 f"{k} regressors after dropping collinear columns)")
            ivd = self._iv_plan(fml, names, znames, zk, xk, yj, A, ctx)
            Azz, Azx, Azy = A[np.ix_(zk, zk)], A[np.ix_(zk, xk)], A[zk, yj]
            Pi = np.linalg.solve(Azz, Azx)             # first-stage coefficients
            XhXh = Azx.T @ Pi
            beta = np.linalg.solve(XhXh, Pi.T @ Azy)
            bread_asm = XhXh
        else:
            ivd = None
            zk, Pi = xk, np.eye(k)
            bread_asm = A[np.ix_(xk, xk)]
            if beta is None:
                beta = np.linalg.solve(bread_asm, A[xk, yj]) if k else np.zeros(0)
        if dropped:
            warn(f"{fml}: {len(dropped)} variables dropped due to "
                 f"multicollinearity: {dropped}", self.logger)
        Axy = A[xk, yj]
        rss_asm = A[yj, yj] - 2 * beta @ Axy + beta @ A[np.ix_(xk, xk)] @ beta if k else A[yj, yj]

        self._log(f"pass 2: {fml}")
        rss, tss_w, ssy, B, meat_hc, meat, jack, paths, boot = self._pass2(
            tag, yname, gamma, yj, xk, beta, zk, Pi, bread_asm, ivd=ivd)
        self._last_bread = B                # X~' W X~, for a subclass's corrections
        N, k_fe, nested = self.N, ctx["k_fe"], ctx["nested"]
        # Small-sample factors follow pyfixest's ssc defaults: iid
        # (N-1)/(N-K); hetero N/(N-K); CRV1 G/(G-1)*(N-1)/(N-Kc), with the
        # levels of FEs nested in any of the request's cluster terms dropped
        # from K (plus one back per nested FE); with fe_dof='exact', a pair of
        # nested FEs adds no redundant levels (see _k_fe). Multi-way: inclusion-exclusion
        # over the terms, each scaled with G = the smallest one-way cluster
        # count (pyfixest's G_df="min"); t-tests use G_min - 1 df.
        K = k + k_fe
        Binv = np.linalg.inv(B) if k else np.zeros((0, 0))
        # a saturated model has no residual df: its variances are undefined,
        # not an error (a fit made only for its effects can be saturated)
        vc = {"iid": (self._iid_factor(rss, N, K) * Binv, N - K),
              "hetero": (_ratio(N, N - K) * Binv @ meat_hc @ Binv, N - K)}
        n_clusters, adjs = {}, {}
        # each vcov's small-sample factor (for OLS's iid, that and the error
        # variance), for a caller that wants other conventions
        factors = {"iid": self._iid_factor(rss, N, K), "hetero": _ratio(N, N - K)}
        for req, terms in self.cluster_reqs.items():
            Gs = [self.clusters[t]["G"] for t, _, single in terms if single]
            Gm = min(Gs)
            nest = list(dict.fromkeys(d for t, _, _ in terms for d in nested[t]))
            Kc = (k + self._k_fe(ctx["fe_dof"], nest)
                  - sum(self.fe_params[d] for d in nest) + len(nest))
            adj = _ratio(Gm, Gm - 1) * _ratio(N - 1, N - Kc)
            Vc = sum(sign * adj * (Binv @ meat[t] @ Binv) for t, sign, _ in terms)
            vc[f"CRV1:{req}"] = (Vc, Gm - 1)
            factors[f"CRV1:{req}"] = adj
            if req in self.crv3_reqs:
                # the jackknife takes the same small-sample factor, as in pyfixest
                (t, _, _), = terms
                vc[f"CRV3:{req}"] = (adj * jack[t] if k else np.zeros((0, 0)), Gm - 1)
                factors[f"CRV3:{req}"] = adj
            n_clusters[req] = Gs
            adjs[req] = adj
        V, dft = vc[ctx["default"]]
        boot_data = None
        if boot is not None:
            # the clusters with rows, with their level values, so that draws
            # can be keyed to the clusters themselves (see HDFEResult._boot)
            spec = self.clusters[boot["term"]]
            levels = self._cluster_levels(boot["term"], paths)
            if levels.height != len(boot["n"]):
                raise RuntimeError(f"{levels.height} cluster levels for {len(boot['n'])} "
                                   "clusters")
            present = boot["n"] > 0
            boot_data = {"cluster": self.boot_req, "cols": list(spec["cols"]),
                         "levels": levels.filter(pl.Series(present)),
                         "AG": boot["AG"][present].copy(), "sG": boot["sG"][present].copy(),
                         "adj": adjs[self.boot_req], "df": min(n_clusters[self.boot_req]) - 1}
        if np.any(np.diag(V) < 0):
            warn(f"{fml}: the {ctx['default']} vcov has negative variances "
                 "(possible with multi-way clustering)", self.logger)

        if iv:
            # one first-stage OLS per endogenous variable (same vcov type), and
            # the Wald F on the excluded instruments, as pyfixest reports it
            excluded = [z for z in znames if z not in names]
            for ei, en in enumerate(iv["endog"]):
                fs_fe = f" | {' + '.join(self.fe)}" if self.fe else ""
                fs_model = {"fml": f"{en} ~ {' + '.join(znames)}{fs_fe}",
                            "y": en, "x": znames}
                fs = self._estimate(f"{tag}_fs{ei}", fs_model, A, gamma, ctx)
                first_stage.append(fs)
                idx = [fs.coefnames.index(z) for z in excluded if z in fs.coefnames]
                b = fs.beta[idx]
                Vs = fs.vcov[np.ix_(idx, idx)]
                f_stats.append(float(b @ np.linalg.solve(Vs, b) / len(idx)) if idx else np.nan)

        iv_diag = None
        if ivd is not None:
            try:
                iv_diag = self._iv_diag(ivd, A, beta, V, names, xk, zk, yj, rss, ctx,
                                        n_clusters)
            except (np.linalg.LinAlgError, ValueError) as err:
                warn(f"{fml}: IV diagnostics not computed: {err}", self.logger)

        # goodness of fit, with pyfixest's definitions (not reported for IV)
        # Without fixed effects the constant is one of the k coefficients, and
        # there is no within R2 (pyfixest reports none either).
        k_fe_r2 = sum(self.fe_params.values()) - len(self.fe) + 1 if self.fe else 0
        r2 = 1 - rss / ssy if ssy > 0 and not iv else np.nan
        # the normal log-likelihood at the ML error variance (not for IV), and
        # AIC/BIC counting the coefficients and the fixed effects' dof (a GLM
        # replaces all three with its family's)
        ll = aic = bic = np.nan
        if not iv and rss > 0:
            ll = -0.5 * N * (np.log(2 * np.pi * rss / N) + 1)
            if self.weights is not None and self.weights_type != "fweights":
                ll += 0.5 * self.wlogsum
            aic, bic = _criteria(ll, k + k_fe, N)
        r2w = 1 - rss / tss_w if tss_w > 0 and not iv and self.fe else np.nan
        diag = {"rss_assembled": float(rss_asm), "rss_rows": float(rss),
                "bread_rel_diff": (float(np.abs(bread_asm - B).max() / np.abs(B).max()) if k
                                   else 0.0),
                "cells_per_obs": round(self.n_cells / self.n_obs, 4),
                "cells_collapsed": None if self.no_fe else bool(self.collapse),
                "assembly": self.assembly, "nested_in_cluster": nested,
                "design_evaluated": getattr(self, "design_evaluated", None),
                "stream": self.stream_choice, "singletons": dict(self.singletons),
                "connected": dict(self.connected_info), "fe_params": dict(self.fe_params),
                "slope_redundancy": getattr(self, "slope_redundancy", None) or [],
                "vcov_factors": {key: float(f) for key, f in factors.items()},
                "seconds_total": round(time.time() - ctx["t0"], 2)}
        return HDFEResult(
            fml=fml, depvar=yname, coefnames=names, fe_names=self.fe, beta=beta, vcov=V,
            vcov_type=ctx["default"], df_t=dft, n_obs=N, n_levels=dict(self.n_levels),
            n_identifying=self.n_identifying, n_components=self.n_components, k_fe=k_fe,
            rss=float(rss), r2_within=float(r2w), solver_info=dict(ctx["info"]), paths=paths,
            loglik=float(ll), aic=float(aic), bic=float(bic),
            collin_vars=dropped, diagnostics=diag, all_vcovs=vc, r2=float(r2),
            adj_r2=float(1 - (1 - r2) * _ratio(N - 1, N - k - k_fe_r2)),
            adj_r2_within=float(1 - (1 - r2w) * _ratio(N - k_fe_r2, N - k - k_fe_r2)),
            rmse=float(np.sqrt(rss / N)) if not iv else np.nan, is_iv=bool(iv), first_stage=first_stage,
            f_stat_1st_stage=f_stats, iv_diag=iv_diag,
            n_clusters=n_clusters,
            weights=self.weights_name, weights_type=self.weights_type, _run=self._run,
            _estimator=self, _sample=self._sample_spec, _boot=boot_data)

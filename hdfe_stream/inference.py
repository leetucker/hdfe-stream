"""Steps 3 and 4: cross-products, collinearity, cluster/dof bookkeeping,
the residual pass, and the per-model result.
"""

from __future__ import annotations

import time

import numba as nb
import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from .kernels_base import _nb_assemble, _nb_assemble_rows, _nb_pass2
from .kernels_slopes import _nb_assemble_rows_sl, _nb_pass2_sl
from .report import _warn
from .results import HDFEResult
from .utils import _safe, _scatter, iter_group_chunks


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


class _InferenceMixin:

    # --------------------------------------------------------------- step 3
    def _assemble(self, gamma):
        """V' M_D V for all variables (m x m)."""
        m, nt = self.m, nb.get_num_threads()
        acc = np.zeros((nt, m, m))
        if self.assembly == "cells":
            _nb_assemble(self.starts, self.codes, self.n, self.sums, self.offs, gamma, acc)
            return self.W_within_cell + acc.sum(axis=0)
        self._log("step 3: row pass for V' M_D V")
        wcol = ["w"] if self.weights is not None else []
        tcols = [f"t{j + 1}" for j in range(len(self.slope_vars))]
        cols = ["gcode", *self.ccols, *wcol, *tcols, *[f"v{j}" for j in range(m)]]
        for ch in iter_group_chunks(self.paths["rows"], cols, self.batch_rows):
            gc = ch["gcode"]
            starts = np.concatenate(([0], np.flatnonzero(gc[1:] != gc[:-1]) + 1, [len(gc)]))
            codes = np.column_stack([ch[cc] for cc in self.ccols]).astype(np.int64)
            V = np.column_stack([ch[f"v{j}"] for j in range(m)]).astype(np.float64)
            w = (np.ascontiguousarray(ch["w"], dtype=np.float64) if wcol
                 else np.ones(len(gc)))
            if tcols:
                _nb_assemble_rows_sl(starts, codes, self.offs, w, self._slope_matrix(ch), V,
                                     gamma, acc)
            else:
                _nb_assemble_rows(starts, codes, self.offs, w, V, gamma, acc)
        return acc.sum(axis=0)

    def _slope_matrix(self, ch):
        """T = [1, centered slope variables] for a chunk of rows."""
        n = len(ch["gcode"])
        T = np.empty((n, self.p))
        T[:, 0] = 1.0
        for j, mu in enumerate(self.tmeans):
            T[:, j + 1] = ch[f"t{j + 1}"] - mu
        return T

    # ------------------------------------------------------- clusters / dof
    def _resolve_clusters(self, keys, extra):
        """Turn cluster requests into cluster *terms*.

        A one-way request 'a' is one term; a multi-way request 'a+b' expands
        to the terms a, b and a^b (and all higher intersections for three or
        more ways), combined by inclusion-exclusion (Cameron, Gelbach and
        Miller 2011), as in pyfixest/fixest. Each term is one of:
          seg     the fe[0] groups themselves: scores summed per group
          segsub  a refinement of fe[0] (e.g. pik^sein): rows of a cluster
                  are always in the same chunk, so scores are summed per
                  (group, code) inside each chunk; no global array
          fe      another FE dimension: its codes, level-sized accumulator
          extra   anything else: factorized in pass 0, level-sized accumulator
        """
        reqs = [k.split(":", 1)[1] for k in keys if k.startswith("CRV1:")]
        reqs += [c for c in extra if c not in reqs]
        g0 = set(self.fe_cols[self.g_fe])
        dim_of = {frozenset(self.fe_cols[d]): d for d in self.o_fe}
        self.clusters, self.cmaps, self.cluster_reqs = {}, {}, {}

        def cmap(cols):
            name = "^".join(cols)
            if name not in self.cmaps:
                self.cmaps[name] = {"cols": list(cols), "code": f"k{len(self.cmaps)}",
                                    "map": str(self.workdir / f"cluster_{_safe(name)}.parquet")}
            return name

        def term(cols):
            key = "^".join(cols)
            if key in self.clusters:
                return key
            cs = set(cols)
            if cs == g0:
                spec = {"kind": "seg", "dim": self.g_fe, "code": "gcode"}
            elif cs > g0:
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
                ok = not any(pl.scan_parquet(f).group_by("gcode").agg(q).select(chk)
                             .collect(engine="streaming").item() for f in self.paths["rows"])
            else:
                ok = not (pl.scan_parquet(self.paths["rows"]).group_by(dcol).agg(q).select(chk)
                          .collect(engine="streaming").item())
            if ok:
                out.append(d)
        return out

    def _redundant_slopes(self):
        """Slope variables that are a function of another FE dimension (e.g.
        worker trends in `year` alongside year effects). The worker slopes on
        such a variable sum to a trend that the other dimension already
        spans, so one of them is redundant; each counts as one more
        restriction in the FE degrees of freedom."""
        out = []
        for j, v in enumerate(self.slope_vars):
            for d in self.o_fe:
                spread = (pl.scan_parquet(self.paths["rows"])
                          .group_by(self.code_of[d])
                          .agg((pl.col(f"t{j + 1}").max() - pl.col(f"t{j + 1}").min()).alias("r"))
                          .select(pl.col("r").max()).collect(engine="streaming").item())
                if spread == 0:
                    out.append((v, d))
                    break
        self.slope_redundancy = out
        return out

    def _collinear(self, Axx, idx=None):
        """pyfixest's collinearity check on the residualized X'X, plus a
        relative check: a variable whose residual sum of squares is below
        collin_tol_rel of its raw (centered) sum of squares is treated as
        absorbed by the fixed effects. The iterative solve leaves noise of
        order sqrt(tol * condition number) in exactly collinear variables,
        so an absolute test alone can miss them."""
        k = Axx.shape[0]
        if k == 0:
            return np.zeros(0, bool)
        mask = self._collinear_pf(Axx)
        if idx is not None:
            raw = self.raw_ss[idx]
            mask |= (raw > 0) & (np.diag(Axx) <= self.collin_tol_rel * raw)
            if mask.any() and not mask.all():   # re-check the rest without them
                keep = np.flatnonzero(~mask)
                mask[keep] = self._collinear_pf(Axx[np.ix_(keep, keep)])
        return mask

    def _collinear_pf(self, Axx):
        k = Axx.shape[0]
        try:
            from pyfixest.core import find_collinear_variables
            mask, _, all_removed = find_collinear_variables(np.ascontiguousarray(Axx),
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
    def _pass2(self, tag, ycol_name, gamma, yj, xk, beta, zk, Pi):
        """Row pass for one model: fe[0] effects, residuals, moments, meats;
        writes the residual file and one FE file per dimension.

        xk: regressor columns (residuals use X beta); zk, Pi: instrument
        columns and first-stage coefficients, so the scores use Pi' z~
        (for OLS, zk = xk and Pi = I).
        """
        mdir = self.workdir / "models" / tag
        mdir.mkdir(parents=True, exist_ok=True)
        paths = {"dir": str(mdir),
                 "resid": str(mdir / "resid.parquet") if self.save_resid else None,
                 "fe": {d: str(mdir / f"fe_{_safe(d)}.parquet") for d in self.fe}}
        off, _ = self._offsets()
        k, q, nt = len(xk), len(zk), nb.get_num_threads()
        L = gamma.shape[0]
        gam_x = np.ascontiguousarray(gamma[:, xk]) if k else np.zeros((L, 0))
        gam_z = np.ascontiguousarray(gamma[:, zk]) if q else np.zeros((L, 0))
        gam_y0 = np.ascontiguousarray(gamma[:, yj])
        gam_y = gam_y0 - gam_x @ beta                  # FEs of the y-equation
        beta_c = np.ascontiguousarray(beta, dtype=np.float64)
        Pi_c = np.ascontiguousarray(Pi, dtype=np.float64).reshape(q, k)
        fweights = self.weights is not None and self.weights_type == "fweights"
        weighted = self.weights is not None

        acc_s = np.zeros((nt, 4))
        acc_B = np.zeros((nt, k, k))
        acc_hc = np.zeros((nt, k, k))
        acc_g = np.zeros((nt, k, k))
        scores = {t: np.zeros((spec["G"], k)) for t, spec in self.clusters.items()
                  if spec["kind"] in ("fe", "extra")}
        subs = {t: spec for t, spec in self.clusters.items() if spec["kind"] == "segsub"}
        meat_sub = {t: np.zeros((k, k)) for t in subs}
        G_sub = {t: 0 for t in subs}
        ccodes = [spec["code"] for spec in self.clusters.values() if spec["kind"] != "seg"]
        src = [c for c in self._src_cols() if c not in self.keep]
        tcols = [f"t{j + 1}" for j in range(len(self.slope_vars))]
        cols = list(dict.fromkeys(["gcode", *self.ccols, *ccodes, *src, *self.keep, *tcols,
                                   *(["w"] if weighted else []), f"v{yj}",
                                   *[f"v{j}" for j in xk], *[f"v{j}" for j in zk]]))
        g_cols = self.fe_cols[self.g_fe]
        yc = float(self.means[yj])
        r_writer = g_writer = None
        for ch in iter_group_chunks(self.paths["rows"], cols, self.batch_rows):
            gc = ch["gcode"]
            starts = np.concatenate(([0], np.flatnonzero(gc[1:] != gc[:-1]) + 1, [len(gc)]))
            codes = np.column_stack([ch[cc] for cc in self.ccols]).astype(np.int64)
            y = np.ascontiguousarray(ch[f"v{yj}"], dtype=np.float64)
            n = len(y)
            w = np.ascontiguousarray(ch["w"], dtype=np.float64) if weighted else np.ones(n)
            X = (np.column_stack([ch[f"v{j}"] for j in xk]).astype(np.float64) if k
                 else np.zeros((n, 0)))
            Z = (np.column_stack([ch[f"v{j}"] for j in zk]).astype(np.float64) if q
                 else np.zeros((n, 0)))
            G = len(starts) - 1
            e = np.empty(n)
            h = np.empty((n, k))
            if tcols:
                T = self._slope_matrix(ch)
                g_coef = np.empty((G, self.p))
                _nb_pass2_sl(starts, codes, self.offs, w, T, y, X, beta_c, gam_y, gam_y0, Z,
                             gam_z, Pi_c, yc, fweights, g_coef, e, h, acc_s, acc_B, acc_hc, acc_g)
                # back to the original scale of the slope variables
                g_eff = g_coef[:, 0] - g_coef[:, 1:] @ self.tmeans
                g_row = np.einsum("ij,ij->i", T, np.repeat(g_coef, np.diff(starts), axis=0))
            else:
                g_eff = np.empty(G)
                _nb_pass2(starts, codes, self.offs, w, y, X, beta_c, gam_y, gam_y0, Z, gam_z,
                          Pi_c, yc, fweights, g_eff, e, h, acc_s, acc_B, acc_hc, acc_g)
                g_row = None
            Ng = np.diff(starts)
            if scores or subs:
                he = h * (w * e)[:, None]
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
                if weighted:
                    out["weights"] = w
                out[ycol_name] = y
                # row-level fe[0] contribution (intercept + slopes * variables)
                out[f"fe_{self.g_fe}"] = np.repeat(g_eff, Ng) if g_row is None else g_row
                for j, f in enumerate(self.o_fe):
                    out[f"fe_{f}"] = gam_y[self.offs[j] + codes[:, j]]
                out.update({"xb": X @ beta_c, "resid": e})
                tbl = pa.table(out)
                if r_writer is None:
                    r_writer = pq.ParquetWriter(paths["resid"], tbl.schema)
                r_writer.write_table(tbl, row_group_size=self.rgs)

            slopes_out = ({f"fe_{self.g_fe}[{v}]": g_coef[:, j + 1]
                           for j, v in enumerate(self.slope_vars)} if tcols else {})
            gt = pa.table({**{c: ch[c][starts[:-1]] for c in g_cols},
                           f"fe_{self.g_fe}": g_eff, **slopes_out, "n_obs": Ng.astype(np.int64)})
            if g_writer is None:
                g_writer = pq.ParquetWriter(paths["fe"][self.g_fe], gt.schema)
            g_writer.write_table(gt, row_group_size=self.rgs)
        if r_writer is not None:
            r_writer.close()
        g_writer.close()

        for f in self.o_fe:
            nl, o = self.n_levels[f], off[f]
            extra = {"component": pl.Series(self.comp)} if f == self.o_fe[0] else {}
            (pl.scan_parquet(self.paths["maps"][f])
               .with_columns(**{f"fe_{f}": pl.Series(gam_y[o:o + nl]),
                                "n_obs": pl.Series(self.cnt[o:o + nl]).cast(pl.Int64)}, **extra)
               .sink_parquet(paths["fe"][f]))

        meat = {t: sc.T @ sc for t, sc in scores.items()}
        meat.update(meat_sub)
        for t, spec in self.clusters.items():
            if spec["kind"] == "seg":
                meat[t] = acc_g.sum(axis=0)
            elif spec["kind"] == "segsub":
                spec["G"] = G_sub[t]
        rss, tss_w, sy, syy = acc_s.sum(axis=0)
        ssy = syy - sy * sy / self.wsum     # weighted total SS around the weighted mean
        return rss, tss_w, ssy, acc_B.sum(axis=0), acc_hc.sum(axis=0), meat, paths

    # ------------------------------------------------------------ estimation
    def _estimate(self, tag, model, A, gamma, ctx):
        """OLS or 2SLS for one model from the assembled V' M_D V, then its
        row pass and variance estimates."""
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
            Azz, Azx, Azy = A[np.ix_(zk, zk)], A[np.ix_(zk, xk)], A[zk, yj]
            Pi = np.linalg.solve(Azz, Azx)             # first-stage coefficients
            XhXh = Azx.T @ Pi
            beta = np.linalg.solve(XhXh, Pi.T @ Azy)
            bread_asm = XhXh
        else:
            zk, Pi = xk, np.eye(k)
            bread_asm = A[np.ix_(xk, xk)]
            beta = np.linalg.solve(bread_asm, A[xk, yj]) if k else np.zeros(0)
        if dropped:
            _warn(f"{fml}: {len(dropped)} variables dropped due to "
                  f"multicollinearity: {dropped}", self.logger)
        Axy = A[xk, yj]
        rss_asm = A[yj, yj] - 2 * beta @ Axy + beta @ A[np.ix_(xk, xk)] @ beta if k else A[yj, yj]

        self._log(f"pass 2: {fml}")
        rss, tss_w, ssy, B, meat_hc, meat, paths = self._pass2(tag, yname, gamma, yj, xk,
                                                               beta, zk, Pi)
        N, k_fe, nested = self.N, ctx["k_fe"], ctx["nested"]
        # Small-sample factors follow pyfixest's ssc defaults: iid
        # (N-1)/(N-K); hetero N/(N-K); CRV1 G/(G-1)*(N-1)/(N-Kc), with the
        # levels of FEs nested in any of the request's cluster terms dropped
        # from K (plus one back per nested FE). Multi-way: inclusion-exclusion
        # over the terms, each scaled with G = the smallest one-way cluster
        # count (pyfixest's G_df="min"); t-tests use G_min - 1 df.
        K = k + k_fe
        Binv = np.linalg.inv(B) if k else np.zeros((0, 0))
        vc = {"iid": (rss / (N - K) * Binv, N - K),
              "hetero": (N / (N - K) * Binv @ meat_hc @ Binv, N - K)}
        n_clusters = {}
        for req, terms in self.cluster_reqs.items():
            Gs = [self.clusters[t]["G"] for t, _, single in terms if single]
            Gm = min(Gs)
            nest = list(dict.fromkeys(d for t, _, _ in terms for d in nested[t]))
            Kc = K - sum(self.fe_params[d] for d in nest) + len(nest)
            adj = Gm / (Gm - 1) * (N - 1) / (N - Kc)
            Vc = sum(sign * adj * (Binv @ meat[t] @ Binv) for t, sign, _ in terms)
            vc[f"CRV1:{req}"] = (Vc, Gm - 1)
            n_clusters[req] = Gs
        V, dft = vc[ctx["default"]]
        if np.any(np.diag(V) < 0):
            _warn(f"{fml}: the {ctx['default']} vcov has negative variances "
                  "(possible with multi-way clustering)", self.logger)

        if iv:
            # one first-stage OLS per endogenous variable (same vcov type), and
            # the Wald F on the excluded instruments, as pyfixest reports it
            excluded = [z for z in znames if z not in names]
            for ei, en in enumerate(iv["endog"]):
                fs_model = {"fml": f"{en} ~ {' + '.join(znames)} | {' + '.join(self.fe)}",
                            "y": en, "x": znames}
                fs = self._estimate(f"{tag}_fs{ei}", fs_model, A, gamma, ctx)
                first_stage.append(fs)
                idx = [fs.coefnames.index(z) for z in excluded if z in fs.coefnames]
                b = fs.beta[idx]
                Vs = fs.vcov[np.ix_(idx, idx)]
                f_stats.append(float(b @ np.linalg.solve(Vs, b) / len(idx)) if idx else np.nan)

        # goodness of fit, with pyfixest's definitions (not reported for IV)
        k_fe_r2 = sum(self.fe_params.values()) - len(self.fe) + 1
        r2 = 1 - rss / ssy if ssy > 0 and not iv else np.nan
        r2w = 1 - rss / tss_w if tss_w > 0 and not iv else np.nan
        diag = {"rss_assembled": float(rss_asm), "rss_rows": float(rss),
                "bread_rel_diff": (float(np.abs(bread_asm - B).max() / np.abs(B).max()) if k
                                   else 0.0),
                "cells_per_obs": round(self.n_cells / self.n_obs, 4),
                "assembly": self.assembly, "nested_in_cluster": nested,
                "stream": self.stream_choice, "fe_params": dict(self.fe_params),
                "slope_redundancy": getattr(self, "slope_redundancy", []),
                "seconds_total": round(time.time() - ctx["t0"], 2)}
        return HDFEResult(
            fml=fml, depvar=yname, coefnames=names, fe_names=self.fe, beta=beta, vcov=V,
            vcov_type=ctx["default"], df_t=dft, n_obs=N, n_levels=dict(self.n_levels),
            n_identifying=self.n_identifying, n_components=self.n_components, k_fe=k_fe,
            rss=float(rss), r2_within=float(r2w), solver_info=dict(ctx["info"]), paths=paths,
            collin_vars=dropped, diagnostics=diag, all_vcovs=vc, r2=float(r2),
            adj_r2=float(1 - (1 - r2) * (N - 1) / (N - k - k_fe_r2)),
            adj_r2_within=float(1 - (1 - r2w) * (N - k_fe_r2) / (N - k - k_fe_r2)),
            rmse=float(np.sqrt(rss / N)) if not iv else np.nan, is_iv=bool(iv), first_stage=first_stage,
            f_stat_1st_stage=f_stats,
            n_clusters=n_clusters,
            weights=self.weights_name, weights_type=self.weights_type, _run=self._run)

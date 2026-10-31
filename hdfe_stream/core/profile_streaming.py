"""The memory and disk model of the shared passes (`profile=True`).

Every estimator on `_StreamingBase` (OLS, IV, Poisson, logit, probit) runs
the same pipeline, so its profile is one model: `_ProfileMixin._profile`
scans the data once, finds the dimensions the passes are sized by (rows N,
levels per fixed effect, the streamed dimension, identifying cells M,
clusters), and lists the arrays each stage holds. The sizes follow the
allocations in passes.py, solve.py, inference.py and irls.py; the comments
here name them.
"""

from __future__ import annotations

import math
from itertools import combinations

import numba as nb
import polars as pl

from .profile import (
    GB, Component, Profile, as_components, baseline_gb, scan_source, source_row_group_rows,
)
from .results_base import _canon_cluster, vcov_key

_DISTINCT_ENOUGH = 0.9          # passes.py: above this share of distinct cells, no collapsing
_SORT_EXPECTED, _SORT_UPPER = 2.5, 3.5   # a sorted bucket's peak over its bytes (arrow, sort copy)


def _roots(expr) -> list[str]:
    try:
        return list(expr.meta.root_names())
    except Exception:                                        # noqa: BLE001
        return []


class _ProfileMixin:

    def _requests(self, vcov, cluster, bootcluster):
        """The vcov keys, extra clusterings and cluster columns a fit with
        these arguments computes (sets ``boot_req`` and ``cluster_cols``)."""
        default = vcov_key(vcov)
        keys = list(dict.fromkeys(["iid", "hetero", default]))
        extra = [_canon_cluster(c) for c in ([cluster] if isinstance(cluster, str) else cluster)]
        self.boot_req = None
        if bootcluster is not None:
            self.boot_req = _canon_cluster(bootcluster)
            if "+" in self.boot_req:
                raise ValueError(f"bootcluster={bootcluster!r}: the wild cluster bootstrap "
                                 "is one-way; give one column or an 'a^b' combination")
            extra = list(dict.fromkeys(extra + [self.boot_req]))
        self.cluster_cols = list(dict.fromkeys(
            c for req in [k.split(":", 1)[1] for k in keys if k.startswith(("CRV1:", "CRV3:"))]
            + extra
            for part in req.split("+") for c in part.split("^")))
        return default, keys, extra

    def _cluster_terms(self, keys, extra):
        """The clusterings that need per-cluster arrays: ``{name: columns}``
        for every non-empty subset of the ways of each CRV1/CRV3 request."""
        reqs = [k.split(":", 1)[1] for k in keys if k.startswith(("CRV1:", "CRV3:"))]
        reqs = list(dict.fromkeys(reqs + list(extra)))
        terms = {}
        for req in reqs:
            parts = [[c.strip() for c in p.split("^")] for p in req.split("+")]
            for r in range(1, len(parts) + 1):
                for sub in combinations(parts, r):
                    cols = [c for p in sub for c in p]
                    terms["^".join(cols)] = cols
        return terms

    def _profile(self, lf, source, vcov="iid", cluster=(), fe_dof="exact",
                 bootcluster=None) -> Profile:
        default, keys, extra = self._requests(vcov, cluster, bootcluster)
        terms = self._cluster_terms(keys, extra)
        crv3 = [k.split(":", 1)[1] for k in keys if k.startswith("CRV3:")]

        # ------------------------------------------------------------ scan
        present = list(self.var_exprs.values())
        fe_src = list(dict.fromkeys(c for d in self.fe_user for c in self.fe_cols[d]))
        present += [pl.col(c) for c in fe_src + [c for c in self.cluster_cols if c not in fe_src]]
        if self.weights is not None:
            present.append(self.weights)
        distinct = {d: self.fe_cols[d] for d in self.fe_user}
        if len(self.fe_user) > 1 or any("^" in d for d in self.fe_user):
            distinct["__cells"] = fe_src
        for name, cols in terms.items():
            if name not in distinct and not any(set(cols) == set(self.fe_cols[d])
                                                for d in self.fe_user):
                distinct[name] = cols
        stats = scan_source(lf, present=present, distinct=distinct)
        src_cols = set(fe_src) | set(self.cluster_cols)
        for e in self.var_exprs.values():
            src_cols |= set(_roots(e))
        if self.weights is not None:
            src_cols |= set(_roots(self.weights))
        return self._profile_from_stats(stats, source, vcov, cluster, bootcluster,
                                        n_src=len(src_cols))

    def _profile_from_stats(self, stats, source, vcov="iid", cluster=(), bootcluster=None, *,
                            n_src) -> Profile:
        """The profile from scan statistics: N, the levels of each fixed effect
        (``distinct[dim]``), of the combination of them all (``"__cells"``) and
        of each cluster term not among them. `n_src` is the number of source
        columns. Separate from the scan so that a model that expands its data
        (logit-stream's multinomial logit with several fixed effects, fitted as
        a Poisson regression on J times the rows) can supply the expanded
        counts without scanning the expanded data."""
        default, keys, extra = self._requests(vcov, cluster, bootcluster)
        terms = self._cluster_terms(keys, extra)
        crv3 = [k.split(":", 1)[1] for k in keys if k.startswith("CRV3:")]
        fe_src = list(dict.fromkeys(c for d in self.fe_user for c in self.fe_cols[d]))
        distinct = {d: self.fe_cols[d] for d in self.fe_user}
        if len(self.fe_user) > 1 or any("^" in d for d in self.fe_user):
            distinct["__cells"] = fe_src
        for name, cols in terms.items():
            if name not in distinct and not any(set(cols) == set(self.fe_cols[d])
                                                for d in self.fe_user):
                distinct[name] = cols
        N = max(stats.n_rows, 1)
        lv = {d: max(stats.distinct.get(d, 1), 1) for d in self.fe_user}

        # --------------------------------------------------------- shape
        m, k = self.m, max((len(mod["x"]) for mod in self.models), default=1)
        k = max(k, 1)
        nt = self.n_threads or nb.get_num_threads()
        pl_threads = pl.thread_pool_size()
        n_src = max(int(n_src), 1)
        width = m + n_src + len(self.keep) + 2          # passes.py: rows' width
        row_bytes = 8 * width
        glm = getattr(self, "family", None) is not None
        warns = []
        comps: list[Component] = []
        add = comps.append

        if self.no_fe:
            G = L = D = M = 0
            streamed = None
        else:
            if self.stream:
                streamed = self.stream
            elif self.slopes:
                streamed = next(iter(self.slopes))
            else:
                streamed = max(self.fe_user, key=lambda d: (lv[d], -self.fe_user.index(d)))
            G = lv[streamed]
            o_fe = [d for d in self.fe_user if d != streamed]
            D = len(o_fe)
            L = sum(lv[d] for d in o_fe) * self.p
            M_dist = min(stats.distinct.get("__cells", lv[streamed]), N)
            # pass 1 collapses runs of equal cells unless nearly every row is its own
            collapses = self.collapse_cells is not False and not (
                self._collapse_optional and self.collapse_cells == "auto"
                and M_dist / N >= _DISTINCT_ENOUGH)
            M = M_dist if (collapses or self.cells_contiguous) else N
        M_up = N

        # numba kernels, Polars' engines and scipy, loaded by the process's first fit:
        # measured at 0.4 GB above the imported package for a 20,000-row fit. A process
        # that has fitted before holds them in the baseline, so this over-counts there.
        add(Component("kernels and library state (first fit)", 0.5e9, 0.9e9,
                      formula="the floor of a small fit"))

        # ------------------------------------------------- pass 0 (memory)
        rpb = self.rows_per_bucket or min(20_000_000, max(500_000, int(384e6 / row_bytes)))
        bucket = min(rpb, N)
        add(Component("pass 0: sort one bucket", _SORT_EXPECTED * bucket * row_bytes,
                      _SORT_UPPER * bucket * row_bytes, phases=("pass0",),
                      formula=f"{_SORT_EXPECTED}-{_SORT_UPPER} x min(rows_per_bucket={rpb:,}, N) "
                              f"x {width} columns x 8"))
        rg = source_row_group_rows(source) or 500_000
        add(Component("pass 0: Polars scan buffers",
                      0.5 * pl_threads * min(rg, N) * 8 * n_src,
                      1.5 * pl_threads * min(rg, N) * 8 * n_src + 0.0,
                      phases=("pass0",),
                      formula=f"{pl_threads} Polars threads x one source row group ({rg:,} rows)"
                              f" x {n_src} source columns x 8"))
        add(Component("pass 0: Polars sink buffers",
                      0.5 * pl_threads * self.rgs * row_bytes, pl_threads * self.rgs * row_bytes,
                      phases=("pass0",),
                      formula=f"{pl_threads} threads x row_group_size={self.rgs:,} x {width} x 8"))
        n_maps = len([d for d in self.fe_user if d != streamed]) + len(
            [t for t in terms if not any(set(terms[t]) == set(self.fe_cols[d])
                                         for d in self.fe_user)])
        map_levels = sum(lv[d] for d in self.fe_user if d != streamed) + sum(
            stats.distinct.get(t, 0) for t in terms if t in distinct)
        add(Component("factor maps (level hash tables)", 24 * map_levels, 40 * map_levels,
                      phases=("pass0",), formula="24-40 bytes per level of each non-streamed "
                                                 "fixed effect and extra cluster"))

        # ------------------------------------------ batches (all passes)
        batch = min(self.batch_rows, N)
        add(Component("row batch (design chunk)", batch * 8 * (3 * m + D + 4),
                      batch * 8 * (5 * m + D + 6), phases=("pass1", "assemble", "residuals"),
                      formula=f"min(batch_rows={self.batch_rows:,}, N) x (3-5 x {m} + {D + 4}) x 8"))

        if not self.no_fe:
            # pass 1 and the solve: level-sized arrays
            add(Component("level arrays (obs, cnt, diag)", 3 * L * 8, 3 * L * 8,
                          phases=("pass1", "solve"), formula=f"3 x L={L:,} x 8"))
            add(Component("rhs and CG blocks", L * min(m, self.rhs_block) * 8 * 7,
                          L * min(m, self.rhs_block) * 8 * 7,
                          phases=("solve",),
                          formula=f"~7 x L x min(m={m}, rhs_block={self.rhs_block}) x 8 "
                                  "(rhs plus 6 block-PCG arrays)"))
            if self.assembly == "cells" and not glm:
                add(Component("cell cross-product blocks", 256 * m * m * 8, 256 * m * m * 8,
                              phases=("pass1",), formula=f"256 x {m}^2 x 8"))
            # identifying cells: memory-mapped files
            cell_row = 4 * D + 8 + 8 * m + (8 * (self.p * self.p + self.p * m) if self.slopes else 0)
            where = "ram" if self.cells_in_memory else "mapped"
            add(Component("identifying cells", M * cell_row, M_up * cell_row, where=where,
                          phases=("pass1", "solve", "assemble", "residuals"),
                          formula=f"M x (4 x D={D} + 8 + 8 x {m}); M={M:,} distinct FE "
                                  f"combinations (<= N)"))
            add(Component("fixed-effect estimates (gamma)", L * m * 8, L * m * 8, where="mapped",
                          phases=("solve", "residuals"), formula=f"L x {m} x 8"))
            # the solver: explicit S, or streamed CG with per-thread accumulators. S may take
            # what the budget has left in this phase
            others = sum(c.expected for c in comps if c.where != "disk"
                         and ("solve" in c.phases or "all" in c.phases))
            s_cap = max(0.0, self.budget.usable_gb - baseline_gb() - others / GB) * GB
            s_exp = min(float(L) * L, float(M) * max(M / max(G, 1), 1)) * 12 + (L + 1) * 4
            s_up = min(float(L) * L * 12, s_cap)
            explicit_ok = s_exp <= s_cap
            copies = max(1, min(nt, int(0.5 * s_cap // max(L * m * 8, 1))))
            if self.solver == "stream_cg" or self.solver == "within":
                s_exp = s_up = 0.0
            if self.solver in ("auto", "explicit", "stream_cg") and D:
                if s_exp > 0:
                    add(Component("explicit S (sparse, CSR)", min(s_exp, s_cap),
                                  s_up, phases=("solve",),
                                  formula=f"nnz ~ min(L^2, M^2/G) x 12 bytes, capped at "
                                          f"{s_cap / GB:.1f} GB of room; larger means stream_cg"))
                    add(Component("S build: level-to-group index", 8 * M * D * 0.5 + 8 * L,
                                  8 * M * D + 16 * L, phases=("solve",),
                                  formula="~ M x D x 8 (lptr, lgrp, row work)"))
                if not explicit_ok or self.solver == "stream_cg":
                    add(Component("stream_cg accumulators", copies * L * m * 8,
                                  copies * L * m * 8, phases=("solve",),
                                  formula=f"{copies} copies x L x {m} x 8"))
            if not explicit_ok and self.solver == "auto":
                warns.append("the explicit S would not fit in the memory left, so the solver streams "
                             "the cells (slower, bounded memory)")
        # step 3 / pass 2
        add(Component("Gram and score accumulators", (nt * m * m + 4 * k * k) * 8,
                      (nt * m * m + 6 * k * k) * 8, phases=("assemble", "residuals"),
                      formula=f"(nt={nt} x {m}^2 + ~4-6 x {k}^2) x 8"))

        # clusters
        g0 = set(self.fe_cols[streamed]) if streamed else None
        for name, cols in terms.items():
            cs = set(cols)
            if g0 is not None and (cs == g0 or cs > g0):
                continue                                    # per streamed group: k x k per thread
            Gt = lv.get(next((d for d in self.fe_user if set(self.fe_cols[d]) == cs), None),
                        None) or stats.distinct.get(name, N)
            add(Component(f"cluster scores: {name}", Gt * k * 8, Gt * k * 8,
                          phases=("residuals",), formula=f"G={Gt:,} x k={k} x 8"))
        for req in crv3:
            Gt = (stats.distinct.get(req) or lv.get(req) or N)
            if streamed and set(req.split("^")) == set(self.fe_cols.get(streamed, [])):
                continue
            add(Component(f"CRV3 per-cluster sums: {req}", Gt * k * (k + 1) * 8,
                          Gt * k * (k + 1) * 8, phases=("residuals",),
                          formula=f"G={Gt:,} x k(k+1) x 8 (refused above 25% of memory)"))
        if self.boot_req:
            Gt = (stats.distinct.get(self.boot_req) or lv.get(self.boot_req) or N)
            add(Component(f"bootcluster sums: {self.boot_req}", Gt * (k * (k + 1) * 8 + 8),
                          Gt * (k * (k + 1) * 8 + 8), phases=("residuals",),
                          formula=f"G={Gt:,} x (k(k+1) x 8 + 8)"))

        # GLM
        if glm:
            c = m
            add(Component("IRLS cell buffers", M * c * 8 * 1.0, M_up * c * 8,
                          where="ram" if self.cells_in_memory else "mapped",
                          phases=("irls",), formula=f"cell sums M x {c} x 8 (glm_*.npy)"))
            add(Component("IRLS row-pass accumulators", nt * c * c * 8, nt * c * c * 8,
                          phases=("irls",), formula=f"nt x {c}^2 x 8"))

        # ------------------------------------------------------ disk
        sorted_up = N * row_bytes
        add(Component("sorted rows (working copy)", 0.7 * sorted_up, sorted_up, where="disk",
                      formula=f"N x {width} columns x 8, Parquet-compressed (0.7-1.0)"))
        n_models = max(len(self.models), 1)
        if self.save_resid:
            ncol = 1 + (len(self.fe_user) if self.fe_user else 0)
            add(Component("result files (residuals, effects)", 0.8 * N * ncol * 8 * n_models,
                          N * ncol * 8 * n_models, where="disk",
                          formula=f"N x (1 + {len(self.fe_user)} effects) x 8 x {n_models} models"))

        prof = Profile(
            model=f"{type(self).__name__} {self.models[0]['fml'] if self.models else ''}".strip()
                  + (f" (+{len(self.models) - 1} more)" if len(self.models) > 1 else ""),
            components=as_components(comps), scan=stats, threads=int(nt),
            memory_limit_gb=self.budget.limit_gb, memory_reserve=self.budget.reserve,
            polars_threads=int(pl_threads), baseline_gb=baseline_gb(), warnings=warns,
            notes={"streamed": streamed, "levels": lv, "cells": M, "rows": N, "k": k, "m": m,
                   "L": L, "G": G, "D": D, "memory_limit_gb": self.budget.limit_gb,
                   "memory_reserve": self.budget.reserve})
        return prof

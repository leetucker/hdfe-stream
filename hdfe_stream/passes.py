"""Passes 0, 1 and 1b: build the design, reduce rows to identifying cells,
and find connected components.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import polars as pl

from .kernels_base import _nb_components, _nb_diag
from .kernels_slopes import _nb_diag_sl
from .utils import _segments, iter_group_chunks


# Shared-state contract with the other mixins
# -------------------------------------------
# Reads, all set by StreamingHDFE itself: the constructor options, and the
# layout fixed by _setup_dims (fe, g_fe, o_fe, ccols, code_of, paths, workdir).
#
# Sets, for _SolveMixin and _InferenceMixin:
#   identifying cells (arrays, or memory maps over <run>/ident_*)
#       starts (G+1,) int64   codes (M, D) uint32   n (M,) f64   sums (M, m) f64
#       st, ainv, cmat        varying-slopes extras; see kernels_slopes
#   level bookkeeping         n_levels, offs, obs, comp, diag, cnt
#   design totals             N, n_obs, wsum, means, tmeans, raw_ss,
#                             W_within_cell
#   reported counts           n_cells, n_ident_cells, n_identifying,
#                             n_components, fe_params, stream_choice
#
# Reads back from _InferenceMixin: clusters and cmaps. _pass0_code calls
# _resolve_clusters so the cluster ids are factorized in the same scan as the
# fixed effects.


class _PassesMixin:

    # ------------------------------------------------------------------ pass 0
    def _pass0_code(self, source, keys, extra):
        lf = source if isinstance(source, pl.LazyFrame) else pl.scan_parquet(source)
        schema = lf.collect_schema()
        src = self._src_cols()
        missing = [c for c in src + self.keep + self.slope_vars if c not in schema]
        if missing:
            raise ValueError(f"columns not found in data: {missing}")
        m = self.m
        wexpr = [self.weights.alias("w")] if self.weights is not None else []
        texpr = [pl.col(v).cast(pl.Float64).alias(f"t{j + 1}")
                 for j, v in enumerate(self.slope_vars)]
        finite = [pl.col(f"v{j}").is_finite() for j in range(m)]
        finite += [pl.col(f"t{j + 1}").is_finite() for j in range(len(self.slope_vars))]
        if self.weights is not None:
            finite.append(pl.col("w").is_finite())
        lf = (lf.select([pl.col(c) for c in src + self.keep] + wexpr + texpr
                        + [self.var_exprs[nm].alias(f"v{j}") for j, nm in enumerate(self.var_names)])
                .drop_nulls(src)
                .filter(pl.all_horizontal(finite)))

        # One scan for counts, means, weight checks and (if the streamed
        # dimension must be chosen by cardinality) HyperLogLog counts.
        wstats = ([pl.col("w").sum().alias("wsum"), pl.col("w").min().alias("wmin")]
                  if self.weights is not None else [])
        tstats = [pl.col(f"t{j + 1}").mean() for j in range(len(self.slope_vars))]
        need_card = self.stream is None and not self.slopes
        card = []
        if need_card:
            for d in self.fe_user:
                cols = self.fe_cols[d]
                key = pl.col(cols[0]) if len(cols) == 1 else pl.struct(cols).hash(seed=7)
                card.append(key.approx_n_unique().alias(f"__card_{d}"))
        self._log("pass 0: scanning data (counts, means"
                  + (", approximate cardinalities)" if need_card else ")"))
        mom = (lf.select([pl.len().alias("n")] + wstats + tstats + card
                         + [pl.col(f"v{j}").mean() for j in range(m)]
                         + [pl.col(f"v{j}").var().alias(f"var{j}") for j in range(m)])
                 .collect(engine="streaming"))
        approx = {d: mom[f"__card_{d}"].item() for d in self.fe_user} if need_card else {}
        g, why = self._choose_stream(approx)
        self._setup_dims(g)
        self.stream_choice = {"dim": g, "reason": why}
        self._log(f"pass 0: streaming {g} ({why})")
        self._resolve_clusters(keys, extra)
        self.tmeans = np.array([mom[f"t{j + 1}"].item() for j in range(len(self.slope_vars))])
        self.n_obs = mom["n"].item()
        if self.n_obs == 0:
            raise ValueError("no complete observations")
        if self.weights is not None and mom["wmin"].item() <= 0:
            raise ValueError("weights must be strictly positive")
        self.wsum = mom["wsum"].item() if self.weights is not None else float(self.n_obs)
        self.N = self.wsum if self.weights_type == "fweights" else self.n_obs
        self.means = np.array([mom[f"v{j}"].item() for j in range(m)])
        self.raw_ss = np.array([(mom[f"var{j}"].item() or 0.0) * max(self.n_obs - 1, 1)
                                for j in range(m)])
        P = self.n_buckets or max(1, -(-self.n_obs // self.rows_per_bucket))
        self.n_buckets = P

        # Codes for non-streamed dimensions and extra cluster variables:
        # one level-sized hash table each.
        self.n_levels = {}
        maps = [(d, self.fe_cols[d], cc, self.paths["maps"][d])
                for d, cc in zip(self.o_fe, self.ccols)]
        maps += [(name, cm["cols"], cm["code"], cm["map"]) for name, cm in self.cmaps.items()]
        coded = lf
        for name, cols, cc, path in maps:
            self._log(f"pass 0: factorizing {name}")
            lf.select(cols).unique().sort(cols).with_row_index(cc).sink_parquet(path)
            nl = pl.scan_parquet(path).select(pl.len()).collect().item()
            if name in self.o_fe:
                self.n_levels[name] = nl
            else:
                self.cmaps[name]["G"] = nl
            coded = coded.join(pl.scan_parquet(path), on=cols)

        # Hash-partition rows by fe[0]: every group lands in exactly one bucket.
        g_cols = self.fe_cols[self.g_fe]
        key = pl.col(g_cols[0]) if len(g_cols) == 1 else pl.struct(g_cols)
        self._log(f"pass 0: hash-partitioning rows into {P} {self.g_fe} buckets")
        shutil.rmtree(self.paths["partitioned"], ignore_errors=True)
        (coded.with_columns(_bucket=(key.hash(seed=20260921) % P).cast(pl.UInt32))
              .sink_parquet(pl.PartitionBy(self.paths["partitioned"], key="_bucket",
                                           include_key=False)))

        # Sort each bucket by fe[0] and assign dense codes that keep
        # increasing across buckets.
        self._log(f"pass 0: sorting buckets and assigning {self.g_fe} codes")
        new_group = pl.any_horizontal([(pl.col(c) != pl.col(c).shift()).fill_null(True)
                                       for c in g_cols])
        offset = 0
        self.paths["rows"] = []
        for bkt in range(P):
            srcdir = Path(self.paths["partitioned"]) / f"_bucket={bkt}"
            if not srcdir.exists():
                continue
            out = str(self.workdir / f"rows_b{bkt:04d}.parquet")
            (pl.scan_parquet(str(srcdir / "*.parquet"))
               .sort(g_cols)
               .with_columns(gcode=(new_group.cast(pl.UInt32).cum_sum() - 1 + offset)
                             .cast(pl.UInt32))
               .sink_parquet(out, row_group_size=self.rgs))
            offset = pl.scan_parquet(out).select(pl.col("gcode").max()).collect().item() + 1
            self.paths["rows"].append(out)
        self.n_levels = {self.g_fe: offset, **self.n_levels}
        self._track_disk()
        shutil.rmtree(self.paths["partitioned"], ignore_errors=True)
        for spec in self.clusters.values():
            if spec["kind"] in ("seg", "fe"):
                spec["G"] = self.n_levels[spec["dim"]]
            elif spec["kind"] == "extra":
                spec["G"] = self.cmaps[spec["map"]]["G"]
            else:                                   # segsub: counted in pass 2
                spec["G"] = None
                spec["xL"] = (self.n_levels[spec["xdim"]] if spec["xdim"] is not None
                              else self.cmaps[spec["map"]]["G"])

    # ------------------------------------------------------------------ pass 1
    def _pass1_cells(self):
        """group_by(fe[0], other codes) -> cell table with counts and sums of
        the (globally centered) variables, plus within-cell cross-products
        when assembly == 'cells'."""
        self._log(f"pass 1: group_by({self.g_fe}, {', '.join(self.o_fe)}) -> cells")
        m = self.m
        z = [(pl.col(f"v{j}") - mu).alias(f"z{j}") for j, mu in enumerate(self.means)]
        weighted = self.weights is not None
        if weighted:
            z.append(pl.col("w"))
        wz = (lambda e: pl.col("w") * e) if weighted else (lambda e: e)
        # "n" is the cell's weight sum (its count when unweighted); "cnt" the count
        aggs = [(pl.col("w").sum() if weighted else pl.len().cast(pl.Float64)).alias("n"),
                pl.len().cast(pl.UInt32).alias("cnt")]
        aggs += [wz(pl.col(f"z{j}")).sum().alias(f"s{j}") for j in range(m)]
        if self.assembly == "cells":
            aggs += [wz(pl.col(f"z{i}") * pl.col(f"z{j}")).sum().alias(f"cp{i}_{j}")
                     for i in range(m) for j in range(i, m)]
        ps = len(self.slope_vars)
        if ps:
            # slope variables centered at their means (keeps the per-group
            # p x p systems well conditioned); moments needed for the
            # within-group regression on [1, slopes]
            z += [(pl.col(f"t{j + 1}") - mu).alias(f"u{j + 1}") for j, mu in enumerate(self.tmeans)]
            aggs += [wz(pl.col(f"u{a}")).sum().alias(f"sl{a}") for a in range(1, ps + 1)]
            aggs += [wz(pl.col(f"u{a}") * pl.col(f"u{b}")).sum().alias(f"sll{a}_{b}")
                     for a in range(1, ps + 1) for b in range(a, ps + 1)]
            aggs += [wz(pl.col(f"u{a}") * pl.col(f"z{j}")).sum().alias(f"slz{a}_{j}")
                     for a in range(1, ps + 1) for j in range(m)]
        self.paths["cells"] = []
        for rows in self.paths["rows"]:
            out = rows.replace("rows_b", "cells_b")
            (pl.scan_parquet(rows)
               .select("gcode", *self.ccols, *z)
               .group_by("gcode", *self.ccols)
               .agg(aggs)
               .sort("gcode", *self.ccols)
               .sink_parquet(out, row_group_size=self.rgs))
            self.paths["cells"].append(out)

        exprs = [pl.len().alias("n_cells")]
        if self.assembly == "cells":
            exprs += [(pl.col(f"cp{i}_{j}") - pl.col(f"s{i}") * pl.col(f"s{j}") / pl.col("n"))
                      .sum().alias(f"w{i}_{j}") for i in range(m) for j in range(i, m)]
        w = pl.scan_parquet(self.paths["cells"]).select(exprs).collect(engine="streaming")
        self.n_cells = w["n_cells"].item()
        if self.assembly == "cells":
            W = np.zeros((m, m))
            for i in range(m):
                for j in range(i, m):
                    W[i, j] = W[j, i] = w[f"w{i}_{j}"].item()
            self.W_within_cell = W

    # ----------------------------------------------------------------- pass 1b
    def _pass1b_identifying(self):
        """Keep cells of fe[0] groups with more than one cell (the only ones
        that identify the other effects) as flat binary arrays for memory
        mapping, and build the Jacobi diagonal."""
        self._log("pass 1b: extracting identifying cells")
        m = self.m
        off, tot = self._offsets()
        scols = [f"s{j}" for j in range(m)]
        obs = np.zeros(tot)             # weight per level (all cells)
        cnt = np.zeros(tot)             # observations per level (all cells)
        sizes = []
        ps, p = len(self.slope_vars), self.p
        slcols = ([f"sl{a}" for a in range(1, ps + 1)]
                  + [f"sll{a}_{b}" for a in range(1, ps + 1) for b in range(a, ps + 1)]
                  + [f"slz{a}_{j}" for a in range(1, ps + 1) for j in range(m)])
        kinds = ("codes", "n", "sums") + (("st", "ainv", "cmat") if ps else ())
        files = {k: open(self.paths["ident"][k], "wb") for k in kinds}
        self.fe0_rank = 0               # fe[0] parameters: sum of per-group ranks
        for ch in iter_group_chunks(self.paths["cells"],
                                    ["gcode", *self.ccols, "n", "cnt", *scols, *slcols],
                                    self.batch_rows):
            for f, cc in zip(self.o_fe, self.ccols):
                sl = slice(off[f], off[f] + self.n_levels[f])
                obs[sl] += np.bincount(ch[cc], weights=ch["n"], minlength=self.n_levels[f])
                cnt[sl] += np.bincount(ch[cc], weights=ch["cnt"], minlength=self.n_levels[f])
            starts, local = _segments(ch["gcode"])
            ncells = np.diff(np.append(starts, len(local)))
            keep = ncells[local] > 1
            if ps:
                # per cell: st = sum w [1, u]; stt = sum w [1,u][1,u]'; stz = sum w [1,u] z'
                Mc = len(local)
                st = np.empty((Mc, p))
                st[:, 0] = ch["n"]
                for a in range(1, p):
                    st[:, a] = ch[f"sl{a}"]
                stt = np.empty((Mc, p, p))
                stt[:, 0, :] = st
                stt[:, :, 0] = st
                for a in range(1, p):
                    for b in range(a, p):
                        stt[:, a, b] = stt[:, b, a] = ch[f"sll{a}_{b}"]
                stz = np.empty((Mc, p, m))
                for j in range(m):
                    stz[:, 0, j] = ch[f"s{j}"]
                    for a in range(1, p):
                        stz[:, a, j] = ch[f"slz{a}_{j}"]
                A = np.add.reduceat(stt, starts, axis=0)
                self.fe0_rank += int(np.linalg.matrix_rank(A, hermitian=True).sum())
            else:
                self.fe0_rank += len(starts)
            if not keep.any():
                continue
            sizes.append(ncells[ncells > 1])
            np.column_stack([ch[cc][keep] for cc in self.ccols]).astype(np.uint32).tofile(files["codes"])
            ch["n"][keep].astype(np.float64).tofile(files["n"])
            np.column_stack([ch[c][keep] for c in scols]).tofile(files["sums"])
            if ps:
                gk = ncells > 1
                st[keep].tofile(files["st"])
                np.linalg.pinv(A[gk], rcond=1e-12, hermitian=True).tofile(files["ainv"])
                np.add.reduceat(stz, starts, axis=0)[gk].tofile(files["cmat"])
        for fh in files.values():
            fh.close()
        sizes = np.concatenate(sizes) if sizes else np.zeros(0, np.int64)
        np.save(self.paths["ident"]["starts"],
                np.concatenate(([0], np.cumsum(sizes))).astype(np.int64))
        self.obs, self.cnt, self.n_identifying = obs, cnt, len(sizes)
        self._track_disk()
        if not self.keep_intermediates:         # cell tables are no longer needed
            for f in self.paths["cells"]:
                Path(f).unlink(missing_ok=True)
        self._load_ident()
        self.diag = np.zeros(tot)
        if ps:
            _nb_diag_sl(self.starts, self.codes, self.n, self.st, self.ainv, self.offs, self.diag)
        else:
            _nb_diag(self.starts, self.codes, self.n, self.offs, self.diag)
        self.fe_params = {self.g_fe: self.fe0_rank,
                          **{d: self.n_levels[d] for d in self.o_fe}}

    def _load_ident(self):
        """Memory-map the identifying cells (or load them, if cells_in_memory)."""
        m, D = self.m, len(self.o_fe)
        self.starts = np.load(self.paths["ident"]["starts"])
        M = int(self.starts[-1])
        load = np.array if self.cells_in_memory else (lambda a: a)

        def mm(key, dtype, shape):
            if M == 0:
                return np.zeros(shape, dtype)
            return load(np.memmap(self.paths["ident"][key], dtype=dtype, mode="r", shape=shape))
        self.codes = mm("codes", np.uint32, (M, D))
        self.n = mm("n", np.float64, (M,))
        self.sums = mm("sums", np.float64, (M, m))
        if self.slope_vars:
            p, G = self.p, len(self.starts) - 1
            self.st = mm("st", np.float64, (M, p))
            self.ainv = (load(np.memmap(self.paths["ident"]["ainv"], dtype=np.float64, mode="r",
                                        shape=(G, p, p))) if G else np.zeros((0, p, p)))
            self.cmat = (load(np.memmap(self.paths["ident"]["cmat"], dtype=np.float64, mode="r",
                                        shape=(G, p, m))) if G else np.zeros((0, p, m)))
        off, _ = self._offsets()
        self.offs = np.array([off[f] for f in self.o_fe], np.int64)
        self.n_ident_cells = M

    # ----------------------------------------------------------- components
    def _components(self):
        """Connected components of the (fe[0], fe[1]) graph by union-find.
        Only the first two dimensions get exact redundancy accounting; any
        further dimension contributes one more restriction (see `fe_dof`)."""
        self._log("computing connected components")
        lab = _nb_components(self.starts, self.codes, self.n_levels[self.o_fe[0]])
        self.comp = lab
        self.n_components = int(np.count_nonzero(lab == np.arange(len(lab))))

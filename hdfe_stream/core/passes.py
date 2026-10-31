"""Passes 0 and 1: build the design, reduce rows to identifying cells,
and find connected components.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import polars as pl

from ._columns import BUCKET, GCODE, PREFIX, WCOL, check_source, tcol, vcol
from .connected import COMPONENT, _find_components
from .kernels_base import _nb_components, _nb_diag, _nb_rhs, _nb_run_sums, _nb_union_pairs
from .kernels_slopes import _nb_diag_sl, _nb_rhs_sl
from .report import warn
from .utils import as_lazy, _not_nan, _row_wise, _segments, iter_group_chunks, row_index

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
#
# Hooks a subclass may override (StreamingGLM does): _restrict_sample, which
# may drop rows from the estimation sample before anything is written;
# `cells_contiguous`, which always sorts each fe[0] group's rows by cell; and
# `cell_sums`, which pass 1 consults before summing the variables per cell.


# Blocks of cells pass 1's kernel splits its sums into: fixed, so that they
# do not depend on the thread count
_RUN_BLOCKS = 256

# The rows are not sorted into cells (collapse_cells="auto") when about this
# share of them are distinct combinations of fixed-effect levels already
_DISTINCT_ENOUGH = 0.9


def _drop_levels(frame, tables, fe_cols):
    """The rows of `frame` in none of the levels `tables` lists, a table of
    levels per fixed effect. An anti-join, not an is_in filter: is_in costs
    over ten times as much once there are a million levels to test."""
    for d, t in tables.items():
        frame = frame.join(t.lazy(), on=fe_cols[d], how="anti")
    return frame


class _PassesMixin:

    # ------------------------------------------------------------------ pass 0
    def _pass0_code(self, source, keys, extra):
        lf = as_lazy(source)
        schema = lf.collect_schema()
        check_source(schema.names())
        if self.row_id in schema:
            raise ValueError(f"the data already has a column named {self.row_id!r}, which "
                             "the estimator uses for each row's position in the source; "
                             "pass row_id= to name it something else")
        # Positions in the source as given, before any row is dropped: what
        # lets the residuals, and sample(), be matched to the source again.
        raw = lf
        lf = row_index(lf, self.row_id)
        schema = lf.collect_schema()
        src = self._src_cols()
        missing = [c for c in src + self.keep + self.slope_vars if c not in schema]
        if missing:
            raise ValueError(f"columns not found in data: {missing}")
        m = self.m
        source = lf
        wexpr = [self.weights.alias(WCOL)] if self.weights is not None else []
        texpr = [pl.col(v).cast(pl.Float64).alias(tcol(j + 1))
                 for j, v in enumerate(self.slope_vars)]
        design = [self.var_exprs[nm].alias(vcol(j)) for j, nm in enumerate(self.var_names)]
        others_finite = [pl.col(tcol(j + 1)).is_finite() for j in range(len(self.slope_vars))]
        if self.weights is not None:
            others_finite.append(pl.col(WCOL).is_finite())
        others_finite += _not_nan(schema, src)
        finite =[pl.col(vcol(j)).is_finite() for j in range(m)] + others_finite

        def evaluate(frame):
            return (frame.select([pl.col(c) for c in src + self.keep] + wexpr + texpr + design)
                         .drop_nulls(src)
                         .filter(pl.all_horizontal(finite)))
        lf = evaluate(source)
        # Row restrictions, in order: the largest connected set, singletons,
        # then whatever the subclass drops (separation, for GLMs); each sees
        # the rows left by those before it.
        restricts = []
        for hook in (self._connected_restriction, self._singleton_restriction,
                     self._restrict_sample):
            restrict = hook(lf)
            if restrict is not None:
                lf = restrict(lf)
                restricts.append(restrict)
        if restricts:
            # The same rows, with the restrictions applied to the source
            # before the design is evaluated: they are anti-joins on the
            # fixed-effect columns, and a streaming join of the evaluated
            # design buffers every design column (with 500 indicators, the
            # scan below peaked near 11 GB this way instead of 4 GB).
            narrow = source
            for restrict in restricts:
                narrow = restrict(narrow)
            lf = evaluate(narrow)
        self._record_sample(raw, src)

        # One scan for counts, means, weight checks and (if the streamed
        # dimension must be chosen by cardinality) HyperLogLog counts.
        wstats = ([pl.col(WCOL).sum().alias("wsum"), pl.col(WCOL).min().alias("wmin"),
                   pl.col(WCOL).log().sum().alias("wlogsum")]
                  if self.weights is not None else [])
        tstats = [pl.col(tcol(j + 1)).mean() for j in range(len(self.slope_vars))]
        need_card = self.stream is None and not self.slopes and len(self.fe_user) > 1
        card = []
        if need_card:
            for d in self.fe_user:
                cols = self.fe_cols[d]
                key = pl.col(cols[0]) if len(cols) == 1 else pl.struct(cols).hash(seed=7)
                card.append(key.approx_n_unique().alias(f"__card_{d}"))
        # whether the rows are already their own cells: the approximate
        # number of distinct combinations of every fixed effect's columns
        combo = []
        if self.collapse_cells == "auto" and self._collapse_optional:
            cols = list(dict.fromkeys(c for d in self.fe_user for c in self.fe_cols[d]))
            combo = [pl.struct(cols).hash(seed=7).approx_n_unique().alias("__combos")]
        self._log("pass 0: scanning data (counts, means"
                  + (", approximate cardinalities)" if need_card or combo else ")"))
        mom = self._collect(lf.select([pl.len().alias("n")] + wstats + tstats + card + combo
                                      + [pl.col(vcol(j)).mean() for j in range(m)]
                                      + [pl.col(vcol(j)).var().alias(f"var{j}")
                                         for j in range(m)]),
                            "pass 0 scan", engine="streaming")
        approx = {d: mom[f"__card_{d}"].item() for d in self.fe_user} if need_card else {}
        g, why = self._choose_stream(approx)
        self._setup_dims(g)
        self.stream_choice = {"dim": g, "reason": why}
        self._log(f"pass 0: streaming {g} ({why})" if g is not None
                  else "pass 0: no fixed effects (ordinary least squares)")
        self._resolve_clusters(keys, extra)
        self.tmeans = np.array([mom[tcol(j + 1)].item() for j in range(len(self.slope_vars))])
        self.n_obs = mom["n"].item()
        if self.n_obs == 0:
            raise ValueError("no complete observations"
                             + (" left after dropping singletons"
                                if self.singletons["observations"] else ""))
        if self.singletons["observations"]:
            warn(f"{self.singletons['observations']:,} singleton observations dropped from "
                 "the model (fixef_rm='none' keeps them)", self.logger)
        if self.weights is not None and mom["wmin"].item() <= 0:
            raise ValueError("weights must be strictly positive")
        self.wsum = mom["wsum"].item() if self.weights is not None else float(self.n_obs)
        # sum of log weights, for the Gaussian log-likelihood with analytic weights
        self.wlogsum = mom["wlogsum"].item() if self.weights is not None else 0.0
        if combo:
            self.combos_ratio = min(1.0, mom["__combos"].item() / self.n_obs)
            self.collapse = self.combos_ratio < _DISTINCT_ENOUGH
        else:
            self.combos_ratio = None
            self.collapse = self.collapse_cells is not False
        self.N = self.wsum if self.weights_type == "fweights" else self.n_obs
        self.means = np.array([mom[vcol(j)].item() for j in range(m)])
        self.raw_ss = np.array([(mom[f"var{j}"].item() or 0.0) * max(self.n_obs - 1, 1)
                                for j in range(m)])
        # Each bucket is sorted in memory, so its rows set pass 0's peak: by
        # default, about 384 MB of them at the rows' width (the design and the
        # source columns they carry), between 500,000 and 20,000,000 rows.
        # (An ACS logit of 3 million rows and 41 columns: one 20M-row bucket
        # peaked at 3.3 GB, 1M-row buckets at 1.7 GB, in less time.)
        rpb = self.rows_per_bucket
        if rpb is None:
            width = self.m + len(src) + len(self.keep) + 2
            # a bucket's sort takes about 3.5 x its bytes: 10% of the headroom, at most
            # 384 MB and at least 64 MB, so a small allocation gets smaller buckets
            target = min(384e6, max(64e6, 0.1 * self.budget.headroom_gb() * 1e9))
            rpb = min(20_000_000, max(500_000, int(target / (8 * width))))
        P = self.n_buckets or max(1, -(-self.n_obs // rpb))
        self.n_buckets = P

        # Codes for non-streamed dimensions and extra cluster variables:
        # one level-sized hash table each.
        self.n_levels = {}
        maps = [(d, self.fe_cols[d], cc, self.paths["maps"][d])
                for d, cc in zip(self.o_fe, self.ccols)]
        maps += [(name, cm["cols"], cm["code"], cm["map"]) for name, cm in self.cmaps.items()]
        # The rows that are partitioned carry either the evaluated design
        # (`lf`) or, when that is safe, only the columns it is computed from,
        # with the design evaluated bucket by bucket at the sort below. See
        # _defer_design. The filter is the same either way, so both select
        # exactly the same rows.
        raw_extra = self._defer_design(schema, src)
        if raw_extra is None:
            coded = lf
        else:
            coded = (source.select([pl.col(c) for c in src + self.keep + raw_extra]
                                   + wexpr + texpr)
                           .drop_nulls(src)
                           .filter(pl.all_horizontal(
                               [self.var_exprs[nm].is_finite() for nm in self.var_names]
                               + others_finite)))
            for restrict in restricts:
                coded = restrict(coded)
        for name, cols, cc, path in maps:
            self._log(f"pass 0: factorizing {name}")
            self._sink(row_index(lf.select(cols).unique().sort(cols), cc), path,
                       f"pass 0 factorize {name}")
            nl = pl.scan_parquet(path).select(pl.len()).collect(engine="streaming").item()
            if name in self.o_fe:
                self.n_levels[name] = nl
            else:
                self.cmaps[name]["G"] = nl
            coded = coded.join(pl.scan_parquet(path), on=cols)

        if self.no_fe:
            # Nothing to group by, so nothing to partition or sort: the rows
            # go to disk in the order they arrive, in one file.
            self._log("pass 0: writing rows (no fixed effects: no grouping)")
            out = str(self.workdir / "rows_b0000.parquet")
            self._sink(coded, out, "pass 0 write rows", row_group_size=self.rgs)
            self.paths["rows"] = [out]
            self._track_disk()
            for spec in self.clusters.values():
                spec["G"] = self.cmaps[spec["map"]]["G"]
            return

        # Hash-partition rows by fe[0]: every group lands in exactly one bucket.
        g_cols = self.fe_cols[self.g_fe]
        key = pl.col(g_cols[0]) if len(g_cols) == 1 else pl.struct(g_cols)
        self._log(f"pass 0: hash-partitioning rows into {P} {self.g_fe} buckets")
        shutil.rmtree(self.paths["partitioned"], ignore_errors=True)
        self._sink(coded.with_columns((key.hash(seed=20260921) % P).cast(pl.UInt32)
                                      .alias(BUCKET)),
                   pl.PartitionBy(self.paths["partitioned"], key=BUCKET, include_key=False),
                   "pass 0 partition",
                   before_retry=lambda: shutil.rmtree(self.paths["partitioned"],
                                                      ignore_errors=True))

        # Sort each bucket by fe[0] and assign dense codes that keep
        # increasing across buckets.
        self._log(f"pass 0: sorting buckets and assigning {self.g_fe} codes")
        new_group = pl.any_horizontal([(pl.col(c) != pl.col(c).shift()).fill_null(True)
                                       for c in g_cols])
        offset = 0
        self.paths["rows"] = []
        for bkt in range(P):
            srcdir = Path(self.paths["partitioned"]) / f"{BUCKET}={bkt}"
            if not srcdir.exists():
                continue
            out = str(self.workdir / f"rows_b{bkt:04d}.parquet")
            # Every fit sorts each bucket by fe[0], because the streaming
            # algorithm needs whole groups together. A fit whose intermediates
            # are being kept sorts by more than that: the other dimensions'
            # codes, then the variables.
            #
            # That wider key makes the stored row order a function of the data
            # rather than of the order it arrived in, which is what leave-out
            # estimation needs -- it draws random vectors keyed on a row's
            # position, so without it the same data written in a different order
            # gives a different (equally valid, but unreproducible) answer. Both
            # halves matter: without the codes a re-exported panel changes the
            # answer, and without the variables two observations of the same
            # worker at the same firm stay tied and can swap between runs.
            #
            # Keeping the intermediates is the condition because it is exactly
            # what makes the operator in inverse.py reachable: if you can get at
            # the row files, their order is canonical. An ordinary fit pays
            # nothing for a guarantee it cannot use.
            #
            # Groups stay contiguous either way, since fe[0] leads the key, so
            # `gcode` does not depend on this.
            #
            # Otherwise the rows are sorted by the other dimensions' codes too
            # when they are to be collapsed into cells (`collapse`): each cell
            # is then one run of rows, which pass 1 collapses as it reads them.
            # That costs a wider key here, a few percent of the sort, and is
            # skipped when nearly every row is a cell of its own anyway.
            sort_by = list(g_cols)
            if self.keep_intermediates:
                sort_by += [*self.ccols, *[vcol(j) for j in range(self.m)]]
            elif self.collapse:
                sort_by += self.ccols
            bucket = pl.scan_parquet(str(srcdir / "*.parquet"))
            if raw_extra is not None:
                bucket = bucket.with_columns(design).drop(raw_extra)
            # A sort holds the whole bucket anyway, so it is collected and
            # written rather than sunk: Polars 2.0's streaming sort peaks at
            # about 1.5x the memory of its in-memory engine.
            query = (bucket
               .sort(sort_by)
               .with_columns((new_group.cast(pl.UInt32).cum_sum() - 1 + offset)
                             .cast(pl.UInt32).alias(GCODE)))
            if self._sort_on_gpu(query, out):
                offset = (pl.scan_parquet(out).select(pl.col(GCODE).max())
                          .collect(engine="streaming").item() + 1)
            else:
                sorted_bucket = query.collect(engine="in-memory")
                sorted_bucket.write_parquet(out, row_group_size=self.rgs)
                offset = sorted_bucket[GCODE].max() + 1
                del sorted_bucket
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
        """Collapse the sorted rows into cells, one per run of rows with the
        same group and the same levels of the other dimensions, with their
        weight, count and the sums of the (globally centered) variables; keep
        the cells of fe[0] groups with more than one cell (the only ones that
        identify the other effects) as flat binary arrays for memory mapping.

        The Jacobi diagonal, and with the estimator's own variables the
        right-hand side of step 2, are built here too, chunk by chunk while
        the cells are in memory, instead of by reading the arrays back: each
        of those reads is a pass over data the size of the cell table."""
        if self.collapse:
            self._log("pass 1: collapsing rows into cells, extracting identifying cells")
        else:
            why = ("collapse_cells=False" if self.combos_ratio is None else
                   f"~{self.combos_ratio:.1%} of the rows are distinct combinations of "
                   "fixed-effect levels")
            self._log(f"pass 1: rows as cells ({why}), extracting identifying cells")
        m = self.m if self.cell_sums else 0
        off, tot = self._offsets()
        offs = np.array([off[f] for f in self.o_fe], np.int64)
        obs = np.zeros(tot)             # weight per level (all cells)
        cnt = np.zeros(tot)             # observations per level (all cells)
        diag = np.zeros(tot)            # Jacobi diagonal (identifying cells)
        # step 2's right-hand side D_o' M_0 V, when all the variables are
        # solved in one block: it is then the size the solve allocates anyway
        rhs = np.zeros((tot, m)) if 0 < m <= self.rhs_block else None
        sizes = []
        ps, p = len(self.slope_vars), self.p
        kinds = ("codes", "n") + (("sums",) if m else ()) + (("st", "ainv", "cmat") if ps else ())
        files = {k: open(self.paths["ident"][k], "wb") for k in kinds}
        self.fe0_rank = 0               # fe[0] parameters: sum of per-group ranks
        # Union-find for each pair of non-streamed dimensions, for the
        # degrees of freedom (see _components). Every cell joins its two
        # levels, so these take all cells, not only the identifying ones.
        nl = [self.n_levels[f] for f in self.o_fe]
        uf = {} if ps else {(i, j): np.arange(nl[i] + nl[j])
                            for i in range(len(nl)) for j in range(i + 1, len(nl))}
        self.n_cells = 0
        self.W_within_cell = np.zeros((m, m)) if self.assembly == "cells" else None
        for ch in self._cell_chunks(m, ps):
            self.n_cells += len(ch[GCODE])
            for f, cc in zip(self.o_fe, self.ccols):
                sl = slice(off[f], off[f] + self.n_levels[f])
                obs[sl] += np.bincount(ch[cc], weights=ch["n"], minlength=self.n_levels[f])
                cnt[sl] += np.bincount(ch[cc], weights=ch["cnt"], minlength=self.n_levels[f])
            for (i, j), parent in uf.items():
                _nb_union_pairs(parent, ch[self.ccols[i]], ch[self.ccols[j]], nl[i])
            starts, local = _segments(ch[GCODE])
            ncells = np.diff(np.append(starts, len(local)))
            keep = ncells[local] > 1
            S = ch["S"]
            if ps:
                # per cell: st = sum w [1, u]; stt = sum w [1,u][1,u]'; stz = sum w [1,u] z'
                Mc = len(local)
                st = np.empty((Mc, p))
                st[:, 0] = ch["n"]
                st[:, 1:] = ch["sl"]
                stt = np.empty((Mc, p, p))
                stt[:, 0, :] = st
                stt[:, :, 0] = st
                stt[:, 1:, 1:] = ch["sll"]
                stz = np.empty((Mc, p, m))
                stz[:, 0, :] = S
                stz[:, 1:, :] = ch["slz"]
                A = np.add.reduceat(stt, starts, axis=0)
                self.fe0_rank += int(np.linalg.matrix_rank(A, hermitian=True).sum())
            else:
                self.fe0_rank += len(starts)
            if not keep.any():
                continue
            kept = ncells[ncells > 1]
            sizes.append(kept)
            kstarts = np.concatenate(([0], np.cumsum(kept))).astype(np.int64)
            kcodes = np.column_stack([ch[cc][keep] for cc in self.ccols]).astype(np.uint32)
            kn = ch["n"][keep].astype(np.float64)
            kcodes.tofile(files["codes"])
            kn.tofile(files["n"])
            if m:
                ksums = np.ascontiguousarray(S[keep])
                ksums.tofile(files["sums"])
            if ps:
                gk = ncells > 1
                kst = np.ascontiguousarray(st[keep])
                kainv = np.linalg.pinv(A[gk], rcond=1e-12, hermitian=True)
                kcmat = np.add.reduceat(stz, starts, axis=0)[gk]
                kst.tofile(files["st"])
                kainv.tofile(files["ainv"])
                kcmat.tofile(files["cmat"])
                _nb_diag_sl(kstarts, kcodes, kn, kst, kainv, offs, diag)
                if rhs is not None:
                    _nb_rhs_sl(kstarts, kcodes, ksums, kst, kainv, kcmat, offs, rhs)
            else:
                _nb_diag(kstarts, kcodes, kn, offs, diag)
                if rhs is not None:
                    _nb_rhs(kstarts, kcodes, kn, ksums, offs, rhs)
        for fh in files.values():
            fh.close()
        sizes = np.concatenate(sizes) if sizes else np.zeros(0, np.int64)
        np.save(self.paths["ident"]["starts"],
                np.concatenate(([0], np.cumsum(sizes))).astype(np.int64))
        self.obs, self.cnt, self.n_identifying = obs, cnt, len(sizes)
        self.diag, self._rhs_pass1 = diag, rhs
        self._other_pair_components = {
            (self.o_fe[i], self.o_fe[j]): int(np.count_nonzero(parent == np.arange(len(parent))))
            for (i, j), parent in uf.items()}
        self._track_disk()
        self._load_ident()
        self.fe_params = {self.g_fe: self.fe0_rank,
                          **{d: self.n_levels[d] for d in self.o_fe}}

    def _cell_chunks(self, m, ps):
        """Chunks of whole fe[0] groups' cells, read from the sorted rows, as
        dicts of arrays: the group code and the other dimensions' codes of each
        cell, "n" (its weight sum), "cnt" (its rows) and "S" (cells x m, the
        weighted sums of the centered variables); with varying slopes also
        "sl", "sll" and "slz", the weighted sums of the centered slope
        variables u, of u u' (the upper triangle filled both ways) and of u z'.

        A cell is a run of consecutive rows with the same group and codes.
        When the rows were sorted by the codes (`collapse`), that is every row
        of the combination; otherwise only identical rows that happen to be
        adjacent share one, and the rest are cells of their own, which changes
        nothing but the number of cells. With assembly="cells" the within-cell
        cross-products are accumulated into W_within_cell, exactly, from each
        row's deviation from its cell's mean."""
        weighted = self.weights is not None
        cols = [GCODE, *self.ccols, *[vcol(j) for j in range(m)],
                *[tcol(a + 1) for a in range(ps)]] + ([WCOL] if weighted else [])
        means = np.asarray(self.means[:m], np.float64)
        tmeans = np.asarray(self.tmeans[:ps], np.float64)
        W = self.W_within_cell
        nblk = _RUN_BLOCKS
        for ch in iter_group_chunks(self.paths["rows"], cols, self.batch_rows):
            g = ch[GCODE]
            n_rows = len(g)
            same = g[1:] == g[:-1]
            for cc in self.ccols:
                same &= ch[cc][1:] == ch[cc][:-1]
            cs = np.flatnonzero(np.concatenate(([True], ~same)))     # cell starts
            bounds = np.append(cs, n_rows).astype(np.int64)
            run = np.diff(bounds)
            w = ch[WCOL].astype(np.float64) if weighted else np.zeros(0)
            out = {GCODE: g[cs], **{cc: ch[cc][cs] for cc in self.ccols}}
            out["cnt"] = run.astype(np.float64)
            out["n"] = np.add.reduceat(w, cs) if weighted else out["cnt"]
            # the centered variables, one contiguous row each, and the cells'
            # sums of them (with the within-cell scatter, for assembly="cells")
            ZT = np.empty((m, n_rows))
            for j in range(m):
                np.subtract(ch.pop(vcol(j)), means[j], out=ZT[j])
            ST = np.empty((m, len(cs)))
            parts = np.empty((nblk, m, m) if W is not None else (nblk, 0, 0))
            _nb_run_sums(bounds, ZT, w, ST, parts)
            if W is not None:
                W += parts.sum(axis=0)
            if ps:
                Z = ZT.T
                U = np.column_stack([ch.pop(tcol(a + 1)).astype(np.float64) - tmeans[a]
                                     for a in range(ps)])
                wU = U * w[:, None] if weighted else U
                out["sl"] = np.add.reduceat(wU, cs, axis=0)
                out["sll"] = np.add.reduceat(wU[:, :, None] * U[:, None, :], cs, axis=0)
                out["slz"] = np.add.reduceat(wU[:, :, None] * Z[:, None, :], cs, axis=0)
                del U, wU, Z
            del ZT
            out["S"] = ST.T
            yield out

    def _load_ident(self):
        """Memory-map the identifying cells (or load them, if cells_in_memory)."""
        m, D = (self.m if self.cell_sums else 0), len(self.o_fe)
        self.starts = np.load(self.paths["ident"]["starts"])
        M = int(self.starts[-1])
        load = np.array if self.cells_in_memory else (lambda a: a)

        def mm(key, dtype, shape):
            if M == 0 or 0 in shape:
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

    def _defer_design(self, schema, src):
        """Decide where the design is evaluated; see `utils._row_wise`.

        Returns None to evaluate it before partitioning (the rows then carry
        the m design columns through the partition and the sort), or the list
        of extra raw columns to partition instead, with the design evaluated
        one bucket at a time. The second is what keeps a wide design -- one
        categorical expanded into hundreds of indicators -- from multiplying
        the partition's memory by the number of indicators; it is taken
        whenever every design expression is row-wise, which formula-built
        designs always are. Otherwise the fit falls back, with a warning: the
        estimates are the same, only the memory differs.
        """
        self.design_evaluated = "before partitioning"
        if self.no_fe:                  # nothing is partitioned
            return None
        for name in self.var_names:
            ok, why = _row_wise(self.var_exprs[name])
            if not ok:
                self.design_evaluated += f" ({name!r} is not row-wise: {why})"
                warn(f"the expression for {name!r} is not row-wise ({why}), so the "
                     "design is evaluated before the rows are partitioned. The "
                     "estimates are unaffected; with many covariates it costs "
                     "memory. To avoid it, define the column in the input "
                     "LazyFrame and name it here.", self.logger)
                return None
        taken = set(src) | set(self.keep)
        roots = dict.fromkeys(r for name in self.var_names
                              for r in self.var_exprs[name].meta.root_names())
        extra = [r for r in roots if r not in taken]
        missing = [r for r in extra if r not in schema]
        if missing:
            # let the eager path raise Polars' own error
            self.design_evaluated += f" (column names {missing})"
            return None
        self.design_evaluated = "per bucket"
        return extra

    def _record_sample(self, raw, src):
        """What `HDFEResult.sample()` needs to say which rows of the source the
        fit used: the source itself, its row count, the test for a usable row
        (as pass 0 applies it, but on the source's own columns), and the
        small tables of levels dropped for being singletons or separated."""
        usable = pl.all_horizontal(
            [pl.col(c).is_not_null() for c in src]
            + _not_nan(raw.collect_schema(), src)
            + [self.var_exprs[nm].is_finite() for nm in self.var_names]
            + [pl.col(v).cast(pl.Float64).is_finite() for v in self.slope_vars]
            + ([self.weights.is_finite()] if self.weights is not None else [])
        ).fill_null(False)
        self._sample_spec = {
            "source": raw, "row_id": self.row_id, "usable": usable,
            "n_raw": self._collect(raw.select(pl.len()), "row count", engine="streaming").item(),
            "fe_cols": dict(self.fe_cols),
            "tables": {"min_degree": dict(self._pruned_tables),
                       "disconnected": dict(self._disconnected_tables),
                       "leave_one_out": dict(self._loo_tables),
                       "singleton": dict(self._singleton_tables),
                       "separation": dict(self._separation_tables)},
        }

    def _connected_restriction(self, lf):
        """A function that drops the rows `min_degree` prunes, then those
        outside the largest connected set (`connected="largest"`) or the
        leave-one-out connected set (`connected="leave_one_out"`), all of the
        pair `connected_fe`; or None when there is nothing to drop.

        All of it comes from one call to `connected.py`'s `_find_components`,
        as in `connected_components`: pruning first, then the components of
        what is left. A row survives the pruning when both of its levels do,
        so the rows a pruning drops are those with a removed level of either
        dimension (a table of them per dimension). The two levels of a
        surviving row are in the same component, so the rows outside the
        largest are those whose level, of either dimension, is outside it,
        and the dimension with fewer such levels gives the table."""
        self.connected_info = {}
        self._pruned_tables, self._disconnected_tables, self._loo_tables = {}, {}, {}
        if self.connected is None and self.min_degree == (1, 1):
            return None
        dims = self.connected_fe
        loo = self.connected == "leave_one_out"
        cc = _find_components(lf, dims, self.fe_cols, min_degree=self.min_degree,
                              batch_rows=self.batch_rows, log=self._log, leave_one_out=loo)
        diag = cc.diagnostics
        info = {"fe": list(dims), "rows": diag["rows"]}
        removed = {d: cc.levels[d][COMPONENT].is_null() for d in dims}
        if self.min_degree != (1, 1):
            by_degree = ({d: ~pl.Series(cc._after_min_degree[d]) for d in dims} if loo
                         else removed)
            tables = {d: cc.levels[d].filter(by_degree[d]).drop(COMPONENT) for d in dims}
            pruned = diag["pruned_rows"]
            info["min_degree"] = {"min_degree": list(self.min_degree), "observations": pruned,
                                  "levels": {d: t.height for d, t in tables.items()},
                                  "rounds": diag["pruning_rounds"]}
            self._pruned_tables = {d: t for d, t in tables.items() if t.height}
            if pruned:
                self._log(f"pass 0: min_degree={self.min_degree} on {dims[0]} x {dims[1]}: "
                          f"{pruned:,} observations dropped")
            if loo:
                removed = {d: removed[d] & ~by_degree[d] for d in dims}
        if loo:
            lo = diag["leave_one_out"]
            tables = {d: cc.levels[d].filter(removed[d]).drop(COMPONENT) for d in dims}
            self._loo_tables = {d: t for d, t in tables.items() if t.height}
            after = {d: int(cc.sizes[d][0]) if cc.n_components else 0 for d in dims}
            info["leave_one_out"] = {
                **lo, "observations": lo["rows"],
                "levels_before": {d: after[d] + lo["levels"][d] for d in dims},
                "levels_after": after}
            self._log(f"pass 0: kept the leave-one-out connected set of {dims[0]} x "
                      f"{dims[1]}: {lo['rows']:,} observations dropped "
                      f"({lo['single_observation']:,} {dims[0]} observed once and "
                      f"{lo['cut']:,} whose removal would disconnect it, in "
                      f"{lo['rounds']} round(s), and those outside the largest of "
                      f"{lo['components_before']:,} components)")
        elif self.connected is not None:
            kept = cc.sizes["rows"][0] if cc.n_components else 0
            outside = {d: cc.levels[d].filter(pl.col(COMPONENT) > 0).drop(COMPONENT)
                       for d in dims}
            dropped = diag["rows"] - diag.get("pruned_rows", 0) - kept
            info.update(components=cc.n_components, observations=dropped,
                        levels={d: t.height for d, t in outside.items()})
            if cc.n_components > 1:
                d = min(dims, key=lambda d: outside[d].height)
                self._disconnected_tables = {d: outside[d]}
                self._log(f"pass 0: keeping the largest connected set of {dims[0]} x "
                          f"{dims[1]}: {dropped:,} observations in {cc.n_components - 1:,} "
                          f"other component{'s' if cc.n_components > 2 else ''} dropped")
        self.connected_info = info
        steps = [t for t in (self._pruned_tables, self._disconnected_tables,
                             self._loo_tables) if t]
        if not steps:
            return None

        def restrict(frame):
            for t in steps:
                frame = _drop_levels(frame, t, self.fe_cols)
            return frame
        return restrict

    def _singleton_restriction(self, lf):
        """A function that drops the singleton observations (`fixef_rm`), or
        None when there are none to drop.

        A singleton is a row alone in its level of some fixed effect.
        Dropping it can leave another row alone in a level of a different
        dimension, so the dimensions are checked in turn, each one's
        singletons dropped before the next is counted, until every dimension
        has been checked since the last drop and found none. A dimension's
        own drop cannot leave a singleton in it, so after a drop the others
        are checked and that one is not. The result is the largest set of rows
        in which no level is alone, whatever the order of the dimensions.
        Each count is a streaming group_by over the fixed-effect columns, one
        dimension at a time, so memory is one hash table of levels plus the
        singleton levels found, which are kept. A singleton level holds one
        row, so the rows dropped are the levels found."""
        self.singletons = {"observations": 0, "levels": {}, "rounds": 0}
        self._singleton_tables = {}
        if self.fixef_rm != "singleton" or not self.fe_user:
            return None
        cols = list(dict.fromkeys(c for d in self.fe_user for c in self.fe_cols[d]))
        cur = lf.select([pl.col(c) for c in cols])
        dims, found, rounds = self.fe_user, {}, set()
        quiet, i = 0, 0             # checks in a row that found nothing
        while quiet < len(dims) - bool(found):
            d = dims[i % len(dims)]
            on = self.fe_cols[d]
            bad = self._collect(cur.group_by(on).agg(pl.len().alias(f"{PREFIX}n"))
                                   .filter(pl.col(f"{PREFIX}n") == 1).select(on),
                                f"singletons {d}", engine="streaming")
            if bad.height:
                found.setdefault(d, []).append(bad)
                cur = _drop_levels(cur, {d: bad}, self.fe_cols)
                rounds.add(i // len(dims))
                quiet = 0
            else:
                quiet += 1
            i += 1
        self.singletons["rounds"] = len(rounds)
        if not found:
            return None
        tables = {d: pl.concat(v) for d, v in found.items()}
        self._singleton_tables = tables
        self.singletons["levels"] = {d: t.height for d, t in tables.items()}
        self.singletons["observations"] = sum(self.singletons["levels"].values())
        return lambda frame: _drop_levels(frame, tables, self.fe_cols)

    def _restrict_sample(self, lf):
        """A function that drops rows from the estimation sample, applied to
        the rows before pass 0 writes them, or None to keep them all. `lf`
        is the filtered design (source columns and the variables as v0, v1,
        ...). OLS keeps every complete row."""
        return None

    # ------------------------------------------------------------- no FE
    def _no_cells(self):
        """Stand-in for pass 1 when there are no fixed effects.

        There are no cells and nothing for the solver to do: every count the
        later steps read is set to what it means for plain OLS.
        """
        self.n_levels, self.fe_params = {}, {}
        self.n_cells, self.n_ident_cells, self.n_identifying = self.n_obs, 0, 0
        self.n_components, self.fe0_rank = 0, 0
        self.pair_components = {}
        self.obs = self.cnt = self.diag = np.zeros(0)
        self.comp = np.zeros(0, np.int64)
        self.starts = np.zeros(1, np.int64)
        self.codes = np.zeros((0, 0), np.uint32)
        self.n = np.zeros(0)
        self.sums = np.zeros((0, self.m))
        self.offs = np.zeros(0, np.int64)

    # ----------------------------------------------------------- components
    def _rank_components(self, root):
        """Component labels 0, 1, ... for the levels of fe[1], numbered by
        the rows in each (component 0 is the largest), from the union-find
        roots `root`, each the component's smallest fe[1] code. Codes are
        in sorted order of the levels, so a stable sort gives ties to the
        component with the smallest fe[1] level: the numbering of
        `connected_components` on the estimation sample, by rows, with fe[1]
        as its first dimension."""
        nl = len(root)
        is_root = root == np.arange(nl)
        dense = (np.cumsum(is_root) - 1)[root]
        f1 = self.o_fe[0]
        o = self._offsets()[0][f1]
        rows = np.bincount(dense, weights=self.cnt[o:o + nl], minlength=int(is_root.sum()))
        order = np.argsort(-rows, kind="stable")
        rank = np.empty(len(order), np.int64)
        rank[order] = np.arange(len(order))
        return rank[dense], len(order)

    def _components(self):
        """Connected components of the (fe[0], fe[1]) graph by union-find,
        which the solver and the reported effects use, and the number of
        components of every pair of dimensions, which `_k_fe` uses. The
        labels, on the levels of fe[1], are ranked by size (see
        `_rank_components`).

        A pair with C components has exactly C redundant levels between its
        two dimensions, and each further dimension adds at least one more, so
        any pair gives a lower bound on the redundant levels; `_k_fe` takes
        the largest. With varying slopes only the (fe[0], fe[1]) pair is
        counted."""
        self.pair_components = {}
        if not self.o_fe:           # one dimension: no graph to connect
            self.comp = np.zeros(0, np.int64)
            self.n_components = 0
            return
        self._log("computing connected components")
        lab = _nb_components(self.starts, self.codes, self.n_levels[self.o_fe[0]])
        self.comp, self.n_components = self._rank_components(lab)
        self.pair_components[(self.g_fe, self.o_fe[0])] = self.n_components
        if self.slope_vars:
            return
        for j, d in enumerate(self.o_fe[1:], 1):
            lab = _nb_components(self.starts, self.codes, self.n_levels[d], j)
            self.pair_components[(self.g_fe, d)] = int(
                np.count_nonzero(lab == np.arange(len(lab))))
        self.pair_components.update(self._other_pair_components)

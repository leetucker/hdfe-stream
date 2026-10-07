"""Passes 0, 1 and 1b: build the design, reduce rows to identifying cells,
and find connected components.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import polars as pl

from ._columns import BUCKET, GCODE, PREFIX, WCOL, check_source, tcol, vcol
from .kernels_base import _nb_components, _nb_diag, _nb_union_pairs
from .kernels_slopes import _nb_diag_sl
from .report import _warn
from .utils import _as_lazy, _row_wise, _segments, iter_group_chunks

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
# `cells_contiguous`, which sorts each fe[0] group's rows by cell; and
# `cell_sums`, which passes 1/1b consult before summing the variables per cell.


def _not_nan(schema, cols):
    """A NaN in a fixed-effect or cluster column is missing, as a null is
    (and as in pyfixest): one test per floating-point column."""
    return [pl.col(c).is_not_nan() for c in cols if schema[c].is_float()]


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
        lf = _as_lazy(source)
        schema = lf.collect_schema()
        check_source(schema.names())
        if self.row_id in schema:
            raise ValueError(f"the data already has a column named {self.row_id!r}, which "
                             "the estimator uses for each row's position in the source; "
                             "pass row_id= to name it something else")
        # Positions in the source as given, before any row is dropped: what
        # lets the residuals, and sample(), be matched to the source again.
        raw = lf
        lf = lf.with_row_index(self.row_id)
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
        # Row restrictions, in order: singletons, then whatever the subclass
        # drops (separation, for GLMs), which sees the rows left by the first.
        restricts = []
        for hook in (self._singleton_restriction, self._restrict_sample):
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
        wstats = ([pl.col(WCOL).sum().alias("wsum"), pl.col(WCOL).min().alias("wmin")]
                  if self.weights is not None else [])
        tstats = [pl.col(tcol(j + 1)).mean() for j in range(len(self.slope_vars))]
        need_card = self.stream is None and not self.slopes and len(self.fe_user) > 1
        card = []
        if need_card:
            for d in self.fe_user:
                cols = self.fe_cols[d]
                key = pl.col(cols[0]) if len(cols) == 1 else pl.struct(cols).hash(seed=7)
                card.append(key.approx_n_unique().alias(f"__card_{d}"))
        self._log("pass 0: scanning data (counts, means"
                  + (", approximate cardinalities)" if need_card else ")"))
        mom = (lf.select([pl.len().alias("n")] + wstats + tstats + card
                         + [pl.col(vcol(j)).mean() for j in range(m)]
                         + [pl.col(vcol(j)).var().alias(f"var{j}") for j in range(m)])
                 .collect(engine="streaming"))
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
            _warn(f"{self.singletons['observations']:,} singleton observations dropped from "
                  "the model (fixef_rm='none' keeps them)", self.logger)
        if self.weights is not None and mom["wmin"].item() <= 0:
            raise ValueError("weights must be strictly positive")
        self.wsum = mom["wsum"].item() if self.weights is not None else float(self.n_obs)
        self.N = self.wsum if self.weights_type == "fweights" else self.n_obs
        self.means = np.array([mom[vcol(j)].item() for j in range(m)])
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
            lf.select(cols).unique().sort(cols).with_row_index(cc).sink_parquet(path)
            nl = pl.scan_parquet(path).select(pl.len()).collect().item()
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
            coded.sink_parquet(out, row_group_size=self.rgs)
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
        (coded.with_columns((key.hash(seed=20260921) % P).cast(pl.UInt32).alias(BUCKET))
              .sink_parquet(pl.PartitionBy(self.paths["partitioned"], key=BUCKET,
                                           include_key=False)))

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
            # A fit that re-reads the rows cell by cell (`cells_contiguous`:
            # the GLM, whose weights change every iteration) sorts by the
            # codes as well, so that each cell's rows are one run.
            sort_by = list(g_cols)
            if self.keep_intermediates:
                sort_by += [*self.ccols, *[vcol(j) for j in range(self.m)]]
            elif self.cells_contiguous:
                sort_by += self.ccols
            bucket = pl.scan_parquet(str(srcdir / "*.parquet"))
            if raw_extra is not None:
                bucket = bucket.with_columns(design).drop(raw_extra)
            # A sort holds the whole bucket anyway, so it is collected and
            # written rather than sunk: Polars 2.0's streaming sort peaks at
            # about 1.5x the memory of its in-memory engine.
            sorted_bucket = (bucket
               .sort(sort_by)
               .with_columns((new_group.cast(pl.UInt32).cum_sum() - 1 + offset)
                             .cast(pl.UInt32).alias(GCODE))
               .collect(engine="in-memory"))
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
        """group_by(fe[0], other codes) -> cell table with counts and sums of
        the (globally centered) variables, plus within-cell cross-products
        when assembly == 'cells'."""
        self._log(f"pass 1: group_by({self.g_fe}, {', '.join(self.o_fe)}) -> cells")
        # without `cell_sums` only the cell structure and counts are built
        m = self.m if self.cell_sums else 0
        z = [(pl.col(vcol(j)) - mu).alias(f"z{j}") for j, mu in enumerate(self.means[:m])]
        weighted = self.weights is not None
        if weighted:
            z.append(pl.col(WCOL))
        wz = (lambda e: pl.col(WCOL) * e) if weighted else (lambda e: e)
        # "n" is the cell's weight sum (its count when unweighted); "cnt" the count
        aggs = [(pl.col(WCOL).sum() if weighted else pl.len().cast(pl.Float64)).alias("n"),
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
            z += [(pl.col(tcol(j + 1)) - mu).alias(f"u{j + 1}") for j, mu in enumerate(self.tmeans)]
            aggs += [wz(pl.col(f"u{a}")).sum().alias(f"sl{a}") for a in range(1, ps + 1)]
            aggs += [wz(pl.col(f"u{a}") * pl.col(f"u{b}")).sum().alias(f"sll{a}_{b}")
                     for a in range(1, ps + 1) for b in range(a, ps + 1)]
            aggs += [wz(pl.col(f"u{a}") * pl.col(f"z{j}")).sum().alias(f"slz{a}_{j}")
                     for a in range(1, ps + 1) for j in range(m)]
        self.paths["cells"] = []
        for rows in self.paths["rows"]:
            out = rows.replace("rows_b", "cells_b")
            (pl.scan_parquet(rows)
               .select(GCODE, *self.ccols, *z)
               .group_by(GCODE, *self.ccols)
               .agg(aggs)
               .sort(GCODE, *self.ccols)
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
        m = self.m if self.cell_sums else 0
        off, tot = self._offsets()
        scols = [f"s{j}" for j in range(m)]
        obs = np.zeros(tot)             # weight per level (all cells)
        cnt = np.zeros(tot)             # observations per level (all cells)
        sizes = []
        ps, p = len(self.slope_vars), self.p
        slcols = ([f"sl{a}" for a in range(1, ps + 1)]
                  + [f"sll{a}_{b}" for a in range(1, ps + 1) for b in range(a, ps + 1)]
                  + [f"slz{a}_{j}" for a in range(1, ps + 1) for j in range(m)])
        kinds = ("codes", "n") + (("sums",) if m else ()) + (("st", "ainv", "cmat") if ps else ())
        files = {k: open(self.paths["ident"][k], "wb") for k in kinds}
        self.fe0_rank = 0               # fe[0] parameters: sum of per-group ranks
        # Union-find for each pair of non-streamed dimensions, for the
        # degrees of freedom (see _components). Every cell joins its two
        # levels, so these take all cells, not only the identifying ones.
        nl = [self.n_levels[f] for f in self.o_fe]
        uf = {} if ps else {(i, j): np.arange(nl[i] + nl[j])
                            for i in range(len(nl)) for j in range(i + 1, len(nl))}
        for ch in iter_group_chunks(self.paths["cells"],
                                    [GCODE, *self.ccols, "n", "cnt", *scols, *slcols],
                                    self.batch_rows):
            for f, cc in zip(self.o_fe, self.ccols):
                sl = slice(off[f], off[f] + self.n_levels[f])
                obs[sl] += np.bincount(ch[cc], weights=ch["n"], minlength=self.n_levels[f])
                cnt[sl] += np.bincount(ch[cc], weights=ch["cnt"], minlength=self.n_levels[f])
            for (i, j), parent in uf.items():
                _nb_union_pairs(parent, ch[self.ccols[i]], ch[self.ccols[j]], nl[i])
            starts, local = _segments(ch[GCODE])
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
            if m:
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
        self._other_pair_components = {
            (self.o_fe[i], self.o_fe[j]): int(np.count_nonzero(parent == np.arange(len(parent))))
            for (i, j), parent in uf.items()}
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
                _warn(f"the expression for {name!r} is not row-wise ({why}), so the "
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
            "n_raw": raw.select(pl.len()).collect(engine="streaming").item(),
            "fe_cols": dict(self.fe_cols),
            "tables": {"singleton": dict(self._singleton_tables),
                       "separation": dict(self._separation_tables)},
        }

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
            bad = (cur.group_by(on).agg(pl.len().alias(f"{PREFIX}n"))
                      .filter(pl.col(f"{PREFIX}n") == 1).select(on)
                      .collect(engine="streaming"))
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
        """Stand-in for passes 1 and 1b when there are no fixed effects.

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
    def _components(self):
        """Connected components of the (fe[0], fe[1]) graph by union-find,
        which the solver and the reported effects use, and the number of
        components of every pair of dimensions, which `_k_fe` uses.

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
        self.comp = lab
        self.n_components = int(np.count_nonzero(lab == np.arange(len(lab))))
        self.pair_components[(self.g_fe, self.o_fe[0])] = self.n_components
        if self.slope_vars:
            return
        for j, d in enumerate(self.o_fe[1:], 1):
            lab = _nb_components(self.starts, self.codes, self.n_levels[d], j)
            self.pair_components[(self.g_fe, d)] = int(
                np.count_nonzero(lab == np.arange(len(lab))))
        self.pair_components.update(self._other_pair_components)

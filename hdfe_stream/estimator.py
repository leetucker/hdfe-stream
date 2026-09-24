"""`StreamingHDFE`: the estimator itself -- options, layout, and the driver
that runs the passes.

The pipeline stages live in `passes.py`, `solve.py` and `inference.py` as
mixins; each of those files documents the attributes it reads and sets.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
import time
from pathlib import Path

import numba as nb
import polars as pl

from .feterms import _parse_fe_term
from .inference import _InferenceMixin
from .passes import _PassesMixin
from .report import _log
from .results import HDFEMulti, _canon_cluster, _vcov_key
from .solve import _SolveMixin
from .utils import _norm_vars, _phys_mem_gb, _safe
from .workspace import _ACTIVE_RUNS, _MARKER, _Run, _dir_bytes


# The pipeline stages are mixed in from passes.py, solve.py and inference.py;
# each of those files opens with the shared-state contract it takes part in.
# Everything on `self` is set here -- in __init__, _init_run or _setup_dims --
# except these, which the mixins set and this file reads back: n_levels,
# n_obs, n_cells, n_ident_cells, n_identifying, n_components and fe_params
# (pass 1/1b), and clusters (inference).


class StreamingHDFE(_PassesMixin, _SolveMixin, _InferenceMixin):
    """
    Low-level interface (see `feols_stream` for formulas).

    Parameters
    ----------
    y : dependent variable(s): a column name, or a list of names and/or
        (name, Polars expression) pairs. Each y is estimated as its own model
        (sharing one solve) unless `models` says otherwise.
    x : covariates, as for `y`. May be empty (FE-only model).
    fe : fixed-effect dimensions, e.g. ["pik", "sein", "year"] or
         ["pik", "sein^year"] ('^' interacts columns). One dimension may carry
         varying slopes, "pik[exper]" or "pik[exper, exper2]" (fixest syntax;
         FE intercepts plus slopes); it is then the streamed dimension.
    stream : name of the dimension to stream. Default: the dimension with
         slopes, else the highest approximate cardinality (ties: listed
         first). The choice is logged and stored in diagnostics["stream"];
         pass it explicitly when cardinality is a poor guide (e.g. firm-year
         effects that outnumber workers in a subsample).
    workdir : directory under which each fit creates its own run directory
         (hdfe_run_<time>_<id>/) for intermediate and result files. Default:
         the system temporary directory (honours TMPDIR). Put it on a disk
         with room for several times the input's Parquet size (the sorted
         working copy of the rows is less compressible than typical input;
         about 4-5x in tests); diagnostics["disk_peak_gb"] reports it.
    outputs : "auto" (default): result files (residuals, fixed effects) are
             deleted when all result objects of the fit are garbage-collected
             or the interpreter exits; keep the result alive while using
             resid()/fixef(), or collect/sink what you need.
         "keep": result files stay until result.cleanup() or
             hdfe_stream.cleanup(workdir). Intermediates are always removed.
    save_resid : write the row-level residual file (the largest output;
         default True). With False, resid() is unavailable.
    keep_intermediates : keep intermediate files and, on errors, the run
         directory (for debugging only).
    solver : "auto" (default): "explicit" unless S would exceed `max_s_gb`,
             then "stream_cg".
         "explicit": build S = D_o' M_0 D_o once as a sparse matrix and
             iterate in memory. RAM ~ nnz(S): grows with distinct level
             co-occurrences (firm-year pairs, firm pairs sharing a mover),
             not with rows.
         "stream_cg": never form S; each iteration streams the identifying
             cells. Fallback when S is too big.
         "within": identifying cells in RAM, solved by `within`.
    precond : "jacobi" (default) or "amg" (pyamg smoothed aggregation;
         explicit solver only).
    max_s_gb : size cap for S (default: 25% of physical memory).
    rhs_block : variables solved at a time; bounds solver memory at about
         6 x (total non-streamed levels) x rhs_block doubles.
    assembly : "cells" (within-cell cross-products computed in pass 1),
         "rows" (an extra row pass after the solve; no m^2 aggregation
         columns), or "auto" (cells when there are at most 8 variables).
    collin_tol : tolerance for dropping collinear covariates (pyfixest's
         algorithm applied to the residualized X'X).
    collin_tol_rel : also drop a covariate when its variation left after
         removing the fixed effects is below this share of its raw variation
         (default 1e-6, i.e. residual SD < 0.1% of raw SD).
    weights : column name or Polars expression with strictly positive
         weights (weighted least squares; rows with missing weights are
         dropped).
    weights_type : "aweights" (default; N = rows) or "fweights" (frequency
         weights; N = sum of weights), as in pyfixest.
    keep : extra columns to carry into the residual output.
    n_buckets / rows_per_bucket : rows are hash-partitioned by fe[0] into
         buckets so the sort and the cell group_by are bucket-sized.
    batch_rows : rows per chunk when streaming Parquet.
    cells_in_memory : load the identifying cells into RAM instead of
         memory-mapping them.
    triple_budget, dense_max_levels : tuning for building S (see
         `_build_explicit`).
    n_threads : numba threads (default: all available).
    verbose : emit progress messages (default True).
    logger : a logging.Logger; if given, progress messages go to it (at
         `log_level`, default INFO) as they happen, and warnings (dropped
         collinear variables, non-convergence, negative variances) go to
         logger.warning. Default: print to stdout / warnings module.
    models : list of {"fml", "y", "x"} dicts selecting which of the variables
         enter each model; all models share passes 0-1 and the solve. An IV
         model adds "iv": {"endog": [...], "z": [...]}, where z lists the
         full instrument set (excluded instruments and exogenous regressors).
    """

    def __init__(self, y, x, fe, workdir=None, solver="auto", precond="jacobi", keep=(),
                 tol=1e-10, maxiter=5000, batch_rows=2_000_000, row_group_size=500_000,
                 n_buckets=None, rows_per_bucket=20_000_000, cells_in_memory=False,
                 triple_budget=5_000_000, dense_max_levels=1000, rhs_block=8,
                 assembly="auto", max_s_gb=None, collin_tol=1e-10, collin_tol_rel=1e-6,
                 n_threads=None,
                 weights=None, weights_type="aweights", stream=None, models=None,
                 verbose=True, logger=None, log_level=logging.INFO, outputs="auto",
                 save_resid=True, keep_intermediates=False):
        ys, xs = _norm_vars(y), _norm_vars(x if x is not None else [])
        self.var_names = list(ys) + [k for k in xs if k not in ys]
        self.var_exprs = {**xs, **ys}
        self.m = len(self.var_names)
        self.vidx = {nm: j for j, nm in enumerate(self.var_names)}
        parsed = [_parse_fe_term(t) for t in fe]
        self.fe_user = [nm for nm, _ in parsed]
        if len(self.fe_user) < 2:
            raise ValueError("need at least two fixed-effect dimensions")
        if len(set(self.fe_user)) < len(self.fe_user):
            raise ValueError(f"duplicate fixed-effect dimensions in {self.fe_user}")
        self.slopes = {nm: sl for nm, sl in parsed if sl}
        if len(self.slopes) > 1:
            raise ValueError("varying slopes are supported on one fixed-effect dimension only "
                             f"(the streamed one); got {list(self.slopes)}")
        self.stream = "^".join(c.strip() for c in stream.split("^")) if stream else None
        if self.stream is not None and self.stream not in self.fe_user:
            raise ValueError(f"stream={stream!r} is not one of the fixed effects {self.fe_user}")
        if self.slopes and self.stream is not None and self.stream not in self.slopes:
            raise ValueError("varying slopes are only supported on the streamed dimension; "
                             f"slopes are on {list(self.slopes)[0]!r}, stream={self.stream!r}")
        self.slope_vars = next(iter(self.slopes.values())) if self.slopes else []
        self.p = 1 + len(self.slope_vars)
        self.fe_cols = {d: [c.strip() for c in d.split("^")] for d in self.fe_user}
        if models is None:
            rhs = " + ".join(xs) or "1"
            models = [{"fml": f"{yn} ~ {rhs} | {' + '.join(fe)}", "y": yn, "x": list(xs)}
                      for yn in ys]
        self.models = models
        self.base_dir = Path(workdir) if workdir else Path(tempfile.gettempdir())
        if outputs not in ("auto", "keep"):
            raise ValueError("outputs must be 'auto' or 'keep'")
        self.outputs, self.save_resid = outputs, save_resid
        self.keep_intermediates = keep_intermediates
        if solver not in ("auto", "explicit", "stream_cg", "within"):
            raise ValueError("solver must be 'auto', 'explicit', 'stream_cg' or 'within'")
        if precond not in ("jacobi", "amg"):
            raise ValueError("precond must be 'jacobi' or 'amg'")
        if assembly not in ("auto", "cells", "rows"):
            raise ValueError("assembly must be 'auto', 'cells' or 'rows'")
        self.solver, self.precond = solver, precond
        if self.slopes and solver == "within":
            raise ValueError("solver='within' does not support varying slopes; "
                             "use 'auto', 'explicit' or 'stream_cg'")
        if self.slopes and assembly == "cells":
            raise ValueError("varying slopes need assembly='rows'")
        self.assembly = (assembly if assembly != "auto" else
                         ("cells" if self.m <= 8 and not self.slopes else "rows"))
        fe_src = {c for d in self.fe_user for c in self.fe_cols[d]}
        self.keep = [c for c in dict.fromkeys(keep) if c not in fe_src]
        self.tol, self.maxiter = tol, maxiter
        self.batch_rows, self.rgs = batch_rows, row_group_size
        self.n_buckets, self.rows_per_bucket = n_buckets, rows_per_bucket
        self.cells_in_memory = cells_in_memory
        self.triple_budget = triple_budget
        self.dense_max_levels = dense_max_levels
        self.rhs_block = max(1, int(rhs_block))
        self.max_s_gb = max_s_gb if max_s_gb is not None else 0.25 * _phys_mem_gb()
        self.collin_tol, self.collin_tol_rel = collin_tol, collin_tol_rel
        if weights_type not in ("aweights", "fweights"):
            raise ValueError("weights_type must be 'aweights' or 'fweights'")
        self.weights = (None if weights is None else
                        (pl.col(weights) if isinstance(weights, str) else weights).cast(pl.Float64))
        self.weights_name = weights if isinstance(weights, str) else ("<expr>" if weights is not None else None)
        self.weights_type = weights_type
        if n_threads:
            nb.set_num_threads(n_threads)
        self.verbose = verbose
        self.logger, self.log_level = logger, log_level

    def _init_run(self):
        """Create this fit's run directory and the intermediate file paths."""
        self._run = _Run(self.base_dir, self.outputs == "auto")
        self.workdir = p = self._run.path
        self.disk_peak = 0
        self.paths = {
            "maps": {},
            "partitioned": str(p / "rows_partitioned"),
            "rows": [], "cells": [],
            "ident": {"codes": str(p / "ident_codes.u32"), "n": str(p / "ident_n.f64"),
                      "sums": str(p / "ident_sums.f64"), "starts": str(p / "ident_starts.npy"),
                      "st": str(p / "ident_st.f64"), "ainv": str(p / "ident_ainv.f64"),
                      "cmat": str(p / "ident_cmat.f64")},
            "gamma": str(p / "gamma.npy"),
        }

    def _log(self, msg):
        _log(self.verbose, msg, self.logger, self.log_level)

    def _track_disk(self):
        self.disk_peak = max(self.disk_peak, _dir_bytes(self.workdir))

    def _release_intermediates(self):
        """Drop memory maps and delete everything in the run directory
        except the per-model result files."""
        for a in ("codes", "n", "sums", "st", "ainv", "cmat", "starts"):
            if hasattr(self, a):
                setattr(self, a, None)
        if self.keep_intermediates or not self.workdir.exists():
            return
        for child in self.workdir.iterdir():
            if child.name in ("models", _MARKER):
                continue
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                child.unlink(missing_ok=True)

    def _setup_dims(self, g):
        """Fix the streamed dimension g; the others keep the user's order."""
        self.fe = [g] + [d for d in self.fe_user if d != g]
        self.g_fe, self.o_fe = self.fe[0], self.fe[1:]
        self.ccols = [f"c{d}" for d in range(1, len(self.fe))]   # code columns
        self.code_of = {d: cc for d, cc in zip(self.o_fe, self.ccols)}
        self.code_of[self.g_fe] = "gcode"
        self.paths["maps"] = {d: str(self.workdir / f"map_{_safe(d)}.parquet") for d in self.o_fe}

    def _choose_stream(self, approx):
        """Streamed dimension: explicit `stream`, else the dimension with
        varying slopes, else the highest (approximate) cardinality, ties going
        to the dimension listed first."""
        if self.stream is not None:
            return self.stream, "stream= option"
        if self.slopes:
            return next(iter(self.slopes)), "varying slopes"
        best = max(self.fe_user, key=lambda d: (approx[d], -self.fe_user.index(d)))
        return best, "highest cardinality (" + ", ".join(
            f"{d} ~{approx[d]:,}" for d in self.fe_user) + ")"

    # --------------------------------------------------------------- layout
    def _offsets(self):
        """Start offset of each other-dimension block in the stacked vector."""
        off, o = {}, 0
        for f in self.o_fe:
            off[f] = o
            o += self.n_levels[f]
        return off, o

    def _src_cols(self):
        cols = [c for d in self.fe_user for c in self.fe_cols[d]]
        cols += list(self.cluster_cols)
        return list(dict.fromkeys(cols))


    def _fit_passes(self, source, keys, extra, default, fe_dof, t0):
        self._pass0_code(source, keys, extra)
        self._pass1_cells()
        self._pass1b_identifying()
        self._components()
        self._log("n={:,} cells={:,} identifying groups={:,} cells={:,} "
                           "components={:,} threads={} ".format(
            self.n_obs, self.n_cells, self.n_identifying, self.n_ident_cells,
            self.n_components, nb.get_num_threads())
            + " ".join(f"{f}={v:,}" for f, v in self.n_levels.items()))

        self._log(f"step 2: solving for {self.m} variables "
                           f"(blocks of {self.rhs_block}), solver={self.solver}")
        gamma, info = self._solve()
        self._log("step 3: assembling normal equations")
        A = self._assemble(gamma)
        nested = {t: self._nested_dims(t) for t in self.clusters}
        total = sum(self.fe_params.values())   # fe[0]: one per group, or its rank with slopes
        n_red = (self.n_components + len(self.o_fe) - 1) if fe_dof == "exact" else len(self.o_fe)
        n_red += len(self._redundant_slopes())
        ctx = {"k_fe": total - n_red, "nested": nested, "default": default, "t0": t0,
               "info": info}
        self._track_disk()
        results = [self._estimate(f"m{mi:03d}", model, A, gamma, ctx)
                   for mi, model in enumerate(self.models)]
        return results

    # --------------------------------------------------------------- driver
    def fit(self, source, vcov="iid", cluster=(), fe_dof="exact"):
        """
        source : Parquet path/glob or a Polars LazyFrame.
        vcov   : default vcov: 'iid', 'hetero'/'HC1', {'CRV1': var} or
                 'CRV1:var'; multi-way clustering as {'CRV1': 'a+b'} (any
                 number of ways). iid and hetero are always computed; switch
                 with result.with_vcov(...).
        cluster: further cluster specs to compute CRV1 for, e.g.
                 ["sein", "pik+sein"]. Any column or 'a^b' combination works;
                 fe[0] itself and the other FE dimensions reuse their codes.
        fe_dof : 'exact' counts one redundant level per connected component of
                 the (fe[0], fe[1]) graph, plus one per further dimension;
                 'pyfixest' counts one per dimension beyond the first.
        Returns an HDFEResult, or an HDFEMulti when there are several models.
        """
        t0 = time.time()
        default = _vcov_key(vcov)
        keys = list(dict.fromkeys(["iid", "hetero", default]))
        extra = [_canon_cluster(c) for c in cluster]
        self.cluster_cols = list(dict.fromkeys(
            c for req in [k.split(":", 1)[1] for k in keys if k.startswith("CRV1:")] + extra
            for part in req.split("+") for c in part.split("^")))
        self._init_run()
        try:
            results = self._fit_passes(source, keys, extra, default, fe_dof, t0)
        except BaseException:
            # nothing from a failed fit is kept (unless debugging)
            self._release_intermediates()
            if not self.keep_intermediates:
                self._run.cleanup()
            raise
        self._release_intermediates()
        _ACTIVE_RUNS.discard(str(self.workdir))
        peak = self.disk_peak / 1e9
        kept = _dir_bytes(self.workdir) / 1e9
        for r in results:
            r.diagnostics["disk_peak_gb"] = round(peak, 3)
            r.diagnostics["disk_results_gb"] = round(kept, 3)
        self._log(f"done in {time.time()-t0:.1f}s; peak disk use {peak:.2f} GB, result files "
                  f"{kept:.2f} GB in {self.workdir}"
                  + (" (removed with the results)" if self.outputs == "auto" else ""))
        return results[0] if len(results) == 1 else HDFEMulti(results)

    @classmethod
    def from_formula(cls, fml, workdir=None, **options):
        """Estimator for a pyfixest-style formula; call .fit(data, vcov=...)."""
        from .api import _FormulaEstimator      # api imports this module
        return _FormulaEstimator(fml, workdir, options)

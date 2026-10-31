"""`_StreamingBase`: the estimator's driver -- options, layout, and the run
of the passes -- without the leave-out (KSS) machinery.

The pipeline stages live in `passes.py`, `solve.py` and `inference.py` as
mixins; each of those files documents the attributes it reads and sets.
`hdfe_stream.StreamingHDFE` is this class with the leave-out mixins added,
and `hdfe_stream.StreamingGLM` builds on that (see `irls.py`).
"""

from __future__ import annotations

import gc
import logging
import time
import traceback
from pathlib import Path
from typing import Any, Sequence

import numba as nb
import polars as pl

from ._columns import GCODE, PREFIX, ccol
from .typing import PathLike, Source, Variables, Vcov
from .connected import _min_degree, _parse_pair
from .feterms import _parse_fe_term
from .inference import _InferenceMixin
from .passes import _PassesMixin
from .profile_streaming import _ProfileMixin
from .report import _log
from .resources import MemoryBudget, peak_rss_gb, reset_peak_rss, thread_scope
from .results import HDFEMulti, HDFEResult
from .results_base import _canon_cluster, vcov_key
from .solve import _SolveMixin
from .utils import _norm_vars, _safe
from .workspace import _ACTIVE_RUNS, _MARKER, Run, _base_dir, _dir_bytes, _remove_path

# The pipeline stages are mixed in from passes.py, solve.py and inference.py;
# each of those files opens with the shared-state contract it takes part in.
# Everything on `self` is set here -- in __init__, _init_run or _setup_dims --
# except these, which the mixins set and this file reads back: n_levels,
# n_obs, n_cells, n_ident_cells, n_identifying, n_components and fe_params
# (pass 1), and clusters (inference).


def _polars_panic_message(exc):
    """What to tell the user when Polars panics (a Rust assertion failing
    inside Polars, not an error in the data or the model)."""
    msg = (f"Polars {pl.__version__} crashed while running the fit's queries "
           f"(Rust panic: {exc}). This is a bug in Polars, not a problem with "
           "the data or the model.")
    if pl.__version__.startswith("1.41."):
        return (msg + " Polars 1.41 panics in streaming joins once a frame "
                "has about 130 columns or more, as a model with over 120 "
                "covariates does; upgrade with `pip install -U 'polars>=1.42'`.")
    return (msg + " Upgrading Polars may fix it; if it does not, please report "
            "it at https://github.com/leetucker/hdfe-stream/issues with the "
            "Polars version and the formula.")


class _StreamingBase(_ProfileMixin, _PassesMixin, _SolveMixin, _InferenceMixin):
    """The driver that `StreamingHDFE` documents: everything but the
    leave-out estimators, which it adds as mixins."""

    # Hooks for subclasses (see passes.py): sort each fe[0] group's rows by
    # cell, and sum the variables per cell in pass 1.
    cells_contiguous = False
    cell_sums = True

    def __init__(self, y: Variables, x: Variables | None, fe: Sequence[str],
                 workdir: PathLike | None = None, solver: str = "auto",
                 precond: str = "jacobi", keep: Sequence[str] = (),
                 tol: float = 1e-10, maxiter: int = 5000, batch_rows: int | None = None,
                 row_group_size: int = 500_000, n_buckets: int | None = None,
                 rows_per_bucket: int | None = None, cells_in_memory: bool = False,
                 rhs_block: int = 8, assembly: str = "auto",
                 collin_tol: float = 1e-10, collin_tol_rel: float = 1e-6,
                 n_threads: int | None = None, scratch_mb: int = 32,
                 fixef_rm: str = "singleton", connected: str | None = None,
                 min_degree: int | tuple[int, int] = 1,
                 connected_fe: str | Sequence[str] | None = None, row_id: str = "row_id",
                 weights: str | pl.Expr | None = None, weights_type: str = "aweights",
                 stream: str | None = None, models: list[dict[str, Any]] | None = None,
                 verbose: bool = True, logger: logging.Logger | None = None,
                 log_level: int = logging.INFO, outputs: str = "auto",
                 save_resid: bool = True, keep_intermediates: bool = False,
                 collapse_cells: bool | str = "auto", device: str = "cpu",
                 polars_engine: str = "cpu", max_memory_gb: float | None = None,
                 memory_reserve: float | None = None) -> None:
        ys, xs = _norm_vars(y), _norm_vars(x if x is not None else [])
        self.var_names = list(ys) + [k for k in xs if k not in ys]
        self.var_exprs = {**xs, **ys}
        parsed = [_parse_fe_term(t) for t in (fe or [])]
        self.fe_user = [nm for nm, _ in parsed]
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
        # With no fixed effects there is no dimension to group the rows by:
        # every pass reads them in plain batches and nothing is demeaned.
        self.no_fe = not self.fe_user
        self.p = 1 + len(self.slope_vars)
        self.fe_cols = {d: [c.strip() for c in d.split("^")] for d in self.fe_user}
        if models is None:
            rhs = " + ".join(xs) or "1"
            if self.no_fe:
                # OLS keeps its constant as a coefficient, as pyfixest reports it
                if "Intercept" not in self.var_exprs:
                    self.var_names.append("Intercept")
                    self.var_exprs["Intercept"] = pl.lit(1.0)
                models = [{"fml": f"{yn} ~ {rhs}", "y": yn, "x": ["Intercept", *xs]}
                          for yn in ys]
            else:
                models = [{"fml": f"{yn} ~ {rhs} | {' + '.join(fe)}", "y": yn,
                           "x": list(xs)} for yn in ys]
        self.models = models
        self.m = len(self.var_names)
        self.vidx = {nm: j for j, nm in enumerate(self.var_names)}
        self.base_dir = _base_dir(workdir)
        if outputs not in ("auto", "keep"):
            raise ValueError("outputs must be 'auto' or 'keep'")
        self.outputs, self.save_resid = outputs, save_resid
        # Also what enables leave-out estimation: it needs the row files and
        # cell arrays, which are otherwise deleted when the fit finishes.
        self.keep_intermediates = keep_intermediates
        if fixef_rm not in ("none", "singleton"):
            raise ValueError("fixef_rm must be 'none' or 'singleton'")
        self.fixef_rm = fixef_rm
        if connected not in (None, "largest", "leave_one_out"):
            raise ValueError("connected must be None, 'largest' or 'leave_one_out'")
        self.connected = connected
        self.min_degree = _min_degree(min_degree)
        if connected is not None or self.min_degree != (1, 1):
            if len(self.fe_user) < 2:
                raise ValueError(
                    ("min_degree" if connected is None else f"connected={connected!r}")
                    + " needs at least two fixed effects: it works on the graph of a "
                    "pair of them")
        self.connected_fe = tuple(self.fe_user[:2])
        if connected_fe is not None:
            self.connected_fe, _ = _parse_pair(connected_fe)
            other = [d for d in self.connected_fe if d not in self.fe_user]
            if other:
                raise ValueError(f"connected_fe names {other}, which are not fixed effects "
                                 f"of the model ({self.fe_user})")
        if solver not in ("auto", "explicit", "stream_cg", "within"):
            raise ValueError("solver must be 'auto', 'explicit', 'stream_cg' or 'within'")
        if precond not in ("jacobi", "amg"):
            raise ValueError("precond must be 'jacobi' or 'amg'")
        if assembly not in ("auto", "cells", "rows"):
            raise ValueError("assembly must be 'auto', 'cells' or 'rows'")
        self.solver, self.precond = solver, precond
        if device not in ("cpu", "cuda"):
            raise ValueError("device must be 'cpu' or 'cuda'")
        if device == "cuda" and (solver not in ("auto", "explicit") or precond != "jacobi"):
            raise ValueError("device='cuda' runs the explicit solver with the Jacobi "
                             "preconditioner: use solver='auto' or 'explicit' and "
                             "precond='jacobi'")
        if polars_engine not in ("cpu", "gpu"):
            raise ValueError("polars_engine must be 'cpu' or 'gpu'")
        if device == "cuda":
            from .gpu import cuda_backend
            cuda_backend()          # fail now, not after pass 0, without CuPy or a GPU
        if polars_engine == "gpu":
            from .gpu import check_polars_gpu
            check_polars_gpu()
        self.device, self.polars_engine = device, polars_engine
        self.polars_gpu_fallbacks = []
        if self.slopes and solver == "within":
            raise ValueError("solver='within' does not support varying slopes; "
                             "use 'auto', 'explicit' or 'stream_cg'")
        if self.slopes and assembly == "cells":
            raise ValueError("varying slopes need assembly='rows'")
        if self.no_fe and assembly == "cells":
            raise ValueError("assembly='cells' needs a fixed effect to group the rows "
                             "by; without one the cross-products come from a row pass")
        self.assembly = (assembly if assembly != "auto" else
                         ("cells" if self.m <= 8 and not self.slopes and not self.no_fe
                          else "rows"))
        if collapse_cells not in ("auto", True, False):
            raise ValueError("collapse_cells must be 'auto', True or False")
        # Whether to sort each group's rows by cell, so that pass 1 collapses
        # every cell, is a choice only for linear models with two or more
        # fixed effects whose intermediates are not kept: the GLM re-reads the
        # rows cell by cell, and the leave-out operator wants the canonical
        # order, so both always sort.
        self._collapse_optional = (not self.cells_contiguous and not keep_intermediates
                                   and not self.no_fe and len(self.fe_user) >= 2)
        if collapse_cells is False and not self._collapse_optional:
            raise ValueError("collapse_cells=False is available for linear models with two "
                             "or more fixed effects, without keep_intermediates=True")
        self.collapse_cells = collapse_cells
        fe_src = {c for d in self.fe_user for c in self.fe_cols[d]}
        if row_id.startswith(PREFIX):
            raise ValueError(f"row_id={row_id!r} starts with {PREFIX!r}, which is reserved "
                             "for the estimator's own columns")
        self.row_id = row_id
        self.keep = [row_id] + [c for c in dict.fromkeys(keep)
                                if c not in fe_src and c != row_id]
        self._pruned_tables, self._disconnected_tables, self._loo_tables = {}, {}, {}
        self._singleton_tables, self._separation_tables = {}, {}
        self.connected_info = {}
        self._sample_spec = None
        self.tol, self.maxiter = tol, maxiter
        # Without fixed effects nothing groups the rows, and chunks of about 64 MB
        # of the design (100,000 to 2,000,000 rows) keep memory to the design's
        # width at no cost in time (3M x 40: peak 3.9 -> 2.1 GB); with fixed
        # effects 2,000,000 rows (StreamingGLM sets its own).
        self.batch_rows = (batch_rows if batch_rows is not None else
                           min(2_000_000, max(100_000, int(64e6 / (8 * (self.m + 4)))))
                           if self.no_fe else 2_000_000)
        self.rgs = row_group_size
        self.scratch_mb = int(scratch_mb)
        self.n_buckets, self.rows_per_bucket = n_buckets, rows_per_bucket
        self.cells_in_memory = cells_in_memory
        self.rhs_block = max(1, int(rhs_block))
        # what may still be allocated, for the guards that choose between a fast path
        # and a slower one (see _s_limit_gb)
        self.budget = MemoryBudget(max_memory_gb, memory_reserve)
        self.collin_tol, self.collin_tol_rel = collin_tol, collin_tol_rel
        if weights_type not in ("aweights", "fweights"):
            raise ValueError("weights_type must be 'aweights' or 'fweights'")
        self.weights = (None if weights is None else
                        (pl.col(weights) if isinstance(weights, str) else weights).cast(pl.Float64))
        self.weights_name = weights if isinstance(weights, str) else ("<expr>" if weights is not None else None)
        self.weights_type = weights_type
        self.n_threads = n_threads      # held for the length of fit(), then restored
        self.verbose = verbose
        self.logger, self.log_level = logger, log_level

    def _init_run(self):
        """Create this fit's run directory and the intermediate file paths."""
        self._run = Run(self.base_dir, self.outputs == "auto")
        self.workdir = p = self._run.path
        self.disk_peak = 0
        self.slope_redundancy = None    # found once per fit by _redundant_slopes
        self.polars_gpu_fallbacks = []
        self.paths = {
            "maps": {},
            "partitioned": str(p / "rows_partitioned"),
            "rows": [],
            "ident": {"codes": str(p / "ident_codes.u32"), "n": str(p / "ident_n.f64"),
                      "sums": str(p / "ident_sums.f64"), "starts": str(p / "ident_starts.npy"),
                      "st": str(p / "ident_st.f64"), "ainv": str(p / "ident_ainv.f64"),
                      "cmat": str(p / "ident_cmat.f64")},
            "gamma": str(p / "gamma.npy"),
        }

    def _log(self, msg):
        _log(self.verbose, msg, self.logger, self.log_level)

    # The fit's own Polars queries go through these two, so that
    # polars_engine="gpu" can send them to the cudf-polars engine. The engine
    # is asked to raise rather than fall back silently, and a query it cannot
    # run is run again on the CPU and recorded under `what`, which is how the
    # benchmarks can tell what actually ran on the GPU. On the CPU the engine
    # is always named, as everywhere in the library, so that Polars 1 and 2
    # run the same one.
    def _gpu_engine(self):
        # one engine, and so one GPU memory pool, for all of a fit's queries
        if getattr(self, "_polars_gpu", None) is None:
            self._polars_gpu = pl.GPUEngine(raise_on_fail=True)
        return self._polars_gpu

    def _gpu_failed(self, what, err):
        reason = (str(err).strip().splitlines() or [type(err).__name__])[0][:200]
        self.polars_gpu_fallbacks.append({"query": what, "reason": reason})
        self._debug(f"  polars GPU engine: {what} ran on the CPU ({reason})")

    def _collect(self, lf, what, engine):
        if self.polars_engine == "gpu":
            try:
                return lf.collect(engine=self._gpu_engine())
            except Exception as err:            # noqa: BLE001
                self._gpu_failed(what, err)
        return lf.collect(engine=engine)

    def _sink(self, lf, target, what, before_retry=None, **kw):
        if self.polars_engine == "gpu":
            try:
                return lf.sink_parquet(target, engine=self._gpu_engine(), **kw)
            except Exception as err:            # noqa: BLE001
                self._gpu_failed(what, err)
                if before_retry is not None:
                    before_retry()
        return lf.sink_parquet(target, engine="streaming", **kw)

    def _sort_on_gpu(self, query, out):
        """Pass 0's bucket sort, written straight from the GPU; False if
        the CPU is to do it (no GPU engine asked for, or it could not)."""
        if self.polars_engine != "gpu":
            return False
        try:
            query.sink_parquet(out, row_group_size=self.rgs, engine=self._gpu_engine())
            return True
        except Exception as err:                # noqa: BLE001
            self._gpu_failed("pass 0 sort", err)
            Path(out).unlink(missing_ok=True)
            return False

    def _s_limit_gb(self, upcoming_gb=0.0):
        """The most memory, in GB, the explicit S may take (and, at half of it,
        the per-thread accumulators): the memory headroom now, less
        `upcoming_gb` still to be allocated in the phase."""
        return self.budget.headroom_gb(upcoming_gb)

    def _acc_copies(self, levels, width):
        """How many per-thread copies of a levels x width float64 accumulator
        a parallel pass gets: one per thread, as long as all of them together
        stay within half of `_s_limit_gb()`, and at least one.

        The kernels split their groups among the copies, so fewer copies
        means fewer threads on that pass rather than more memory. Half the
        budget, because the leave-out passes can hold two such sets at once
        (their own, and the solver's)."""
        per_copy = max(int(levels) * max(int(width), 1) * 8, 1)
        fit = int(0.5 * self._s_limit_gb() * 1e9 // per_copy)
        return max(1, min(nb.get_num_threads(), fit))

    def _debug(self, msg):
        """Fine-grained progress (e.g. per-iteration CG residuals): a DEBUG
        record, so it shows only if the logger is enabled for DEBUG. Never
        printed when there is no logger."""
        if self.logger is not None:
            _log(self.verbose, msg, self.logger, logging.DEBUG)

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
            _remove_path(child)

    def _setup_dims(self, g):
        """Fix the streamed dimension g; the others keep the user's order.
        With no fixed effects g is None and every list is empty."""
        self.fe = [] if g is None else [g] + [d for d in self.fe_user if d != g]
        self.g_fe, self.o_fe = (self.fe[0] if self.fe else None), self.fe[1:]
        self.ccols = [ccol(d) for d in range(1, len(self.fe))]   # code columns
        self.code_of = {d: cc for d, cc in zip(self.o_fe, self.ccols)}
        self.code_of[self.g_fe] = GCODE
        self.paths["maps"] = {d: str(self.workdir / f"map_{_safe(d)}.parquet") for d in self.o_fe}

    def _choose_stream(self, approx):
        """Streamed dimension: explicit `stream`, else the dimension with
        varying slopes, else the highest (approximate) cardinality, ties going
        to the dimension listed first."""
        if self.no_fe:
            return None, "no fixed effects"
        if self.stream is not None:
            return self.stream, "stream= option"
        if len(self.fe_user) == 1:
            return self.fe_user[0], "the only fixed effect"
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


    def _run_passes(self, source, keys, extra):
        """Passes 0 and 1 and the connected components: everything a fit
        does before the solve, shared with StreamingGLM."""
        self._pass0_code(source, keys, extra)
        self._after_pass0()
        if self.no_fe:
            self._no_cells()
        else:
            self._pass1_cells()
            self._components()
        self._log("n={:,} cells={:,} identifying groups={:,} cells={:,} "
                  "components={:,} threads={} ".format(
                      self.n_obs, self.n_cells, self.n_identifying, self.n_ident_cells,
                      self.n_components, nb.get_num_threads())
                  + " ".join(f"{f}={v:,}" for f, v in self.n_levels.items()))

    def _after_pass0(self):
        """Hook between pass 0 and pass 1 (StreamingGLM reports separation)."""

    def _fit_passes(self, source, keys, extra, default, fe_dof, t0):
        self._run_passes(source, keys, extra)

        # which fixed effects each cluster term nests; checked before the solve
        # so that a CRV3 request that cannot be met fails early
        nested = {t: self._nested_dims(t) for t in self.clusters}
        self._check_crv3(nested)
        self._check_boot(nested)

        self._log(f"step 2: solving for {self.m} variables "
                           f"(blocks of {self.rhs_block}), solver={self.solver}")
        gamma, info = self._solve()
        self._log("step 3: assembling normal equations")
        A = self._assemble(gamma)
        ctx = {"k_fe": self._k_fe(fe_dof), "fe_dof": fe_dof, "nested": nested, "default": default, "t0": t0,
               "info": info}
        self._track_disk()
        # a loop, not a comprehension: before Python 3.12 a comprehension is a
        # function whose closure would hold `gamma` (a memory map) alive
        # through a traceback, and Windows cannot delete a mapped file
        results = []
        for mi, model in enumerate(self.models):
            results.append(self._estimate(f"m{mi:03d}", model, A, gamma, ctx))
        return results

    def _k_fe(self, fe_dof, nested=()):
        """Fixed-effect parameters net of the redundant ones (see `fit`).

        'exact' counts the components of the pair of dimensions with the
        most (see `_components`), plus one per further dimension. `nested`
        are dimensions whose levels a CRV1 request drops from K: a pair of
        two of them is skipped, since its redundant levels are among levels
        that are not counted, and without another pair nothing beyond
        pyfixest's count is subtracted."""
        total = sum(self.fe_params.values())   # fe[0]: one per group, or its rank with slopes
        if len(self.fe) < 2:
            n_red = 0                   # one dimension, or none: nothing is redundant
        elif fe_dof == "exact":
            comps = max((c for (a, b), c in self.pair_components.items()
                         if a not in nested or b not in nested), default=1)
            n_red = comps + len(self.o_fe) - 1
        else:
            n_red = len(self.o_fe)
        return total - n_red - len(self._redundant_slopes())

    # --------------------------------------------------------------- driver
    def fit(self, source: Source, vcov: Vcov = "iid", cluster: Sequence[str] | str = (),
            fe_dof: str = "exact", bootcluster: str | None = None) -> HDFEResult | HDFEMulti:
        """Fit the model(s) to the data.

        Parameters
        ----------
        source : str, Path, polars.LazyFrame or polars.DataFrame
            A Parquet path or glob, a Polars LazyFrame, or a DataFrame.
        vcov : str or dict, default "iid"
            The default vcov: ``"iid"``, ``"hetero"`` (or ``"HC1"``),
            ``{"CRV1": var}`` or ``"CRV1:var"``; multi-way clustering as
            ``{"CRV1": "a+b"}`` (any number of ways). ``{"CRV3": var}`` is the
            cluster jackknife (one-way, OLS), available when every fixed
            effect is nested within the clusters or there are none. iid and
            hetero are always computed; switch with `HDFEResult.with_vcov`.
        cluster : str or sequence of str, optional
            Further cluster specifications to compute CRV1 for, e.g.
            ``["firm_id", "worker_id+firm_id"]``. Any column or ``"a^b"``
            combination works; the streamed dimension and the other fixed
            effects reuse their codes.
        fe_dof : {"exact", "pyfixest"}, default "exact"
            ``"exact"`` counts one redundant level per connected component of
            the pair of fixed-effect dimensions with the most components, plus
            one per further dimension: exact with two dimensions, and with
            more a conservative count (never more redundant levels than there
            are) that does not depend on their order. For CRV1, a pair of
            fixed effects both nested in the clusters is not used.
            ``"pyfixest"`` counts one per dimension beyond the first.
        bootcluster : str, optional
            A clustering (one column, or ``"a^b"``) to keep the per-cluster
            sums for that `HDFEResult.wildboottest`, `hdfe_stream.rwolf` and
            `hdfe_stream.wyoung` need: about G x k^2 numbers per model, for G
            clusters and k coefficients, collected in the residual pass that
            the fit makes anyway. CRV1 by it is computed too. OLS only (not
            IV), and every fixed effect must be nested within the clusters,
            as for CRV3: firm effects clustered by state are, if no firm is
            in two states; year effects never are, so put them in as
            covariates instead (``"y ~ treat + i(year) | county"`` estimates
            the same ``treat`` as ``"y ~ treat | county + year"``).

        Returns
        -------
        HDFEResult or HDFEMulti
            One result, or several when there are several models.
        """
        t0 = time.time()
        default, keys, extra = self._requests(vcov, cluster, bootcluster)
        self._init_run()
        reset_peak_rss()
        try:
            with thread_scope(self.n_threads):
                results = self._fit_passes(source, keys, extra, default, fe_dof, t0)
        except BaseException as exc:
            # nothing from a failed fit is kept (unless debugging). The
            # traceback's frames still hold memory maps of the intermediates,
            # and Windows cannot delete a mapped file, so release them first.
            traceback.clear_frames(exc.__traceback__)
            gc.collect()
            self._release_intermediates()
            if not self.keep_intermediates:
                self._run.cleanup()
            if isinstance(exc, pl.exceptions.PanicException):
                raise RuntimeError(_polars_panic_message(exc)) from exc
            raise
        self._release_intermediates()
        _ACTIVE_RUNS.discard(str(self.workdir))
        rss = peak_rss_gb()
        rss = None if rss is None else round(rss, 3)
        peak = self.disk_peak / 1e9
        kept = _dir_bytes(self.workdir) / 1e9
        for r in results:
            r.diagnostics["disk_peak_gb"] = round(peak, 3)
            r.diagnostics["disk_results_gb"] = round(kept, 3)
            r.diagnostics["peak_rss_gb"] = rss
            if self.polars_engine == "gpu":
                r.diagnostics["polars_gpu_fallbacks"] = list(self.polars_gpu_fallbacks)
        self._log(f"done in {time.time()-t0:.1f}s; peak disk use {peak:.2f} GB, result files "
                  f"{kept:.2f} GB in {self.workdir}"
                  + (" (removed with the results)" if self.outputs == "auto" else ""))
        return results[0] if len(results) == 1 else HDFEMulti(results)

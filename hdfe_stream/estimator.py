"""`StreamingHDFE`: the estimator itself -- options, layout, and the driver
that runs the passes.

The pipeline stages live in `passes.py`, `solve.py` and `inference.py` as
mixins; each of those files documents the attributes it reads and sets.
"""

from __future__ import annotations

import gc
import logging
import time
import traceback
from typing import Any, Sequence

import numba as nb
import polars as pl

from ._columns import GCODE, PREFIX, ccol
from ._types import PathLike, Source, Variables, Vcov
from .connected import _min_degree, _parse_pair
from .feterms import _parse_fe_term
from .inference import _InferenceMixin
from .inverse import _InverseMixin
from .leaveout import _ComponentsMixin, _LeaveOutMixin, _TraceMixin
from .leaveout_se import _StandardErrorMixin
from .leaveout_weakid import _WeakIdMixin
from .passes import _PassesMixin
from .report import _log
from .results import HDFEMulti, HDFEResult, _canon_cluster, _vcov_key
from .solve import _SolveMixin
from .utils import _norm_vars, _phys_mem_gb, _safe
from .workspace import _ACTIVE_RUNS, _MARKER, _Run, _base_dir, _dir_bytes, _remove_path


# The pipeline stages are mixed in from passes.py, solve.py and inference.py;
# each of those files opens with the shared-state contract it takes part in.
# Everything on `self` is set here -- in __init__, _init_run or _setup_dims --
# except these, which the mixins set and this file reads back: n_levels,
# n_obs, n_cells, n_ident_cells, n_identifying, n_components and fe_params
# (pass 1/1b), and clusters (inference).


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


class StreamingHDFE(_PassesMixin, _SolveMixin, _InferenceMixin, _InverseMixin,
                    _LeaveOutMixin, _TraceMixin, _ComponentsMixin,
                    _StandardErrorMixin, _WeakIdMixin):
    """Low-level interface: column names or Polars expressions instead of a
    formula (see `feols_stream` for formulas). It needs no pyfixest.

    Build the estimator, then call `fit` on the data.

    Parameters
    ----------
    y : str, tuple or list
        Dependent variable(s): a column name, or a list of names and/or
        ``(name, polars.Expr)`` pairs. Each y is estimated as its own model
        (sharing one solve) unless `models` says otherwise.
    x : str, tuple, list or None
        Covariates, as for `y`. May be empty (a model of fixed effects only).
    fe : sequence of str
        Fixed-effect dimensions, e.g. ``["worker_id", "firm_id", "year"]`` or
        ``["worker_id", "firm_id^year"]`` (``^`` interacts columns). One
        dimension may carry varying slopes, ``"worker_id[t]"`` or
        ``"worker_id[t, t2]"`` (fixest syntax: fixed-effect intercepts plus
        slopes); it is then the streamed dimension. One dimension is a within
        regression: the streamed dimension is demeaned group by group and
        there is nothing left to solve for. None (or ``[]``) is ordinary least
        squares: the rows are read in batches and never grouped, and an
        ``"Intercept"`` column is added to each model's covariates unless
        `models` lists them explicitly.
    workdir : str or Path, optional
        Directory under which each fit creates its own run directory
        (``hdfe_run_<time>_<id>/``) for intermediate and result files.
        Default: the ``HDFE_STREAM_WORKDIR`` environment variable if set, else
        the system temporary directory (honors ``TMPDIR``). Put it on a disk
        with room for several times the input's Parquet size (the sorted
        working copy of the rows is less compressible than typical input;
        about 4-5x in tests); ``diagnostics["disk_peak_gb"]`` reports it.
    stream : str, optional
        Name of the dimension to stream. Default: the dimension with slopes,
        else the highest approximate cardinality (ties: listed first). The
        choice is logged and stored in ``diagnostics["stream"]``; pass it
        explicitly when cardinality is a poor guide (e.g. firm-year effects
        that outnumber workers in a subsample).
    weights : str or polars.Expr, optional
        Column name or Polars expression with strictly positive weights
        (weighted least squares; rows with missing weights are dropped).
    weights_type : {"aweights", "fweights"}, default "aweights"
        Analytic weights (N = rows) or frequency weights (N = sum of
        weights), as in pyfixest.
    fixef_rm : {"singleton", "none"}, default "singleton"
        ``"singleton"`` (as in pyfixest) drops observations whose level of
        some fixed effect appears in no other observation, repeating until
        none is left, since dropping one can create another in a different
        dimension. They contribute nothing to the coefficients but count
        toward N, the number of clusters and the fixed-effect parameters.
        ``"none"`` keeps them. The search takes a few extra passes over the
        fixed-effect columns (one per dimension per round, and typically two
        or three rounds), and holds one row per singleton level in memory. It
        runs after missing values are removed and does not look at weights;
        the number dropped is in ``diagnostics["singletons"]``. Nothing is
        dropped without fixed effects.
    connected : {"largest", "leave_one_out"}, optional
        ``"largest"`` keeps only the rows in the largest connected set (by
        rows) of the pair `connected_fe`, by default the first two fixed
        effects as listed (worker_id and firm_id in
        ``"worker_id + firm_id + year"``): the largest component of the graph
        in which each row joins its two levels, as `connected_components`
        finds it. ``"leave_one_out"`` keeps the leave-one-out connected set of
        the pair instead, the sample KSS estimation needs, with its first
        dimension as the workers (``connected_components(...,
        leave_one_out=True)`` describes it); `leave_out_kss` uses it. Either
        is found among the rows left after missing values are removed (and
        `min_degree` has pruned), and comes before the singletons are dropped
        (and before the separation check of a GLM), so the fit is that of the
        data restricted beforehand. With two fixed effects the set stays as it
        is after them; with more, dropping a singleton of a further dimension
        can split it, and connectedness of one pair is necessary but not
        sufficient for identification (see `connected_components`). Default
        None keeps every component. `HDFEResult.sample` reports the rows
        dropped as ``"disconnected"`` or ``"leave_one_out"``; the counts are
        in ``diagnostics["connected"]``.
    min_degree : int or (int, int), default 1
        Drop the rows of levels of the pair `connected_fe` with fewer distinct
        partners in the other than this, repeating until none is left, before
        `connected` and the singletons; an int, or a pair (first, second).
        ``(2, 1)`` keeps movers only: workers seen at two or more firms, with
        ``"worker_id + firm_id"``. As in `connected_components`, which
        documents it. `HDFEResult.sample` reports the rows as
        ``"min_degree"``; the count is in
        ``diagnostics["connected"]["min_degree"]``. The default, 1, drops
        nothing.
    connected_fe : str or (str, str), optional
        The pair of fixed effects `connected` and `min_degree` work on, as
        ``"worker_id + firm_id"`` or ``("worker_id", "firm_id")``; both must
        be fixed effects of the model. Default: the first two listed.
    models : list of dict, optional
        ``{"fml", "y", "x"}`` dicts selecting which of the variables enter
        each model; all models share passes 0-1 and the solve. An IV model
        adds ``"iv": {"endog": [...], "z": [...]}``, where z lists the full
        instrument set (excluded instruments and exogenous regressors).

    Other Parameters
    ----------------
    solver : {"auto", "explicit", "stream_cg", "within"}, default "auto"
        ``"auto"``: ``"explicit"`` unless S would exceed `max_s_gb`, then
        ``"stream_cg"``. ``"explicit"``: build S = D_o' M_0 D_o once as a
        sparse matrix and iterate in memory; its memory is about nnz(S), which
        grows with distinct level co-occurrences (firm-year pairs, firm pairs
        sharing a mover), not with rows. ``"stream_cg"``: never form S; each
        iteration streams the identifying cells; the fallback when S is too
        big. ``"within"``: identifying cells in memory, solved by the
        ``within`` package (no varying slopes).
    precond : {"jacobi", "amg"}, default "jacobi"
        ``"amg"`` is pyamg smoothed aggregation, for the explicit solver
        only.
    tol : float, default 1e-10
        Conjugate-gradient convergence tolerance, on each column's residual
        relative to its right-hand side.
    maxiter : int, default 5000
        The most conjugate-gradient iterations; a solve that reaches it warns.
    max_s_gb : float, optional
        Size cap for S. Default: 25% of physical memory.
    rhs_block : int, default 8
        Variables solved at a time; bounds solver memory at about
        6 x (total non-streamed levels) x rhs_block doubles.
    batch_rows : int, default 2_000_000
        Rows per chunk when streaming Parquet, and per slice of whole groups
        of the streamed dimension when building the cell table.
    rows_per_bucket : int, default 20_000_000
        Target rows per bucket: rows are hash-partitioned by the streamed
        dimension into buckets so the sort is bucket-sized.
    n_buckets : int, optional
        The number of buckets, instead of `rows_per_bucket`.
    row_group_size : int, default 500_000
        Rows per row group of the working and result Parquet files.
    assembly : {"auto", "cells", "rows"}, default "auto"
        ``"cells"``: within-cell cross-products computed in pass 1;
        ``"rows"``: an extra row pass after the solve, with no m^2
        aggregation columns; ``"auto"``: cells when there are at most 8
        variables.
    collin_tol : float, default 1e-10
        Tolerance for dropping collinear covariates (pyfixest's algorithm
        applied to the residualized X'X).
    collin_tol_rel : float, default 1e-6
        Also drop a covariate when its variation left after removing the
        fixed effects is below this share of its raw variation (the default
        means residual SD < 0.1% of raw SD).
    cells_in_memory : bool, default False
        Load the identifying cells into memory instead of memory-mapping
        them.
    n_threads : int, optional
        Numba threads. Default: all available.
    scratch_mb : int, default 32
        Budget, in MB, for the row-blocked scratch the leave-out passes hold.
        Those passes want about a dozen float64 columns per row of the chunk,
        so the chunk is shrunk below `batch_rows` as needed to stay inside
        this. It bounds memory only: every random vector involved is keyed on
        the absolute row ordinal, so results do not depend on it. Raising it
        trades memory for speed and the default sits at the knee: on a
        1.3M-row panel, 32 MB runs the correction in 30s holding 67 MB, while
        512 MB gets to 20s but holds 669 MB.
    keep : sequence of str, optional
        Extra columns to carry into the residual output.
    row_id : str, default "row_id"
        Name of the column that gives each row's position in the source
        (counted before any row is dropped). It is written to the residual
        file, so `HDFEResult.resid` joins back onto the source on it, and
        `HDFEResult.sample` adds it to the source. The source must not
        already have a column of this name.
    outputs : {"auto", "keep"}, default "auto"
        ``"auto"``: result files (residuals, fixed effects) are deleted when
        all result objects of the fit are garbage-collected or the interpreter
        exits; keep the result alive while using ``resid()`` and ``fixef()``,
        or collect or sink what you need. ``"keep"``: result files stay until
        `HDFEResult.cleanup` or `hdfe_stream.cleanup`. Intermediates are
        always removed.
    save_resid : bool, default True
        Write the row-level residual file (the largest output). With False,
        ``resid()`` is unavailable.
    keep_intermediates : bool, default False
        Keep the intermediate files, and on errors the run directory. Needed
        for leave-out (Kline-Saggio-Solvsten) estimation when leaving out an
        observation, which reads the sorted rows and cell tables after the
        fit; see `HDFEResult.leave_out_kss`. Otherwise for debugging. It also
        widens the pass-0 sort key so the stored row order is a function of
        the data, which is what makes the leave-out draws reproducible.
    verbose : bool, default True
        Emit progress messages.
    logger : logging.Logger, optional
        If given, progress messages go to it (at `log_level`) as they happen,
        per-iteration CG residuals go to it at DEBUG (and are not printed
        without a logger), and warnings (dropped collinear variables,
        non-convergence, negative variances) go to ``logger.warning``.
        Default: print to stdout and warn through the warnings module.
    log_level : int, default logging.INFO
        The level progress messages are logged at.

    Examples
    --------
    >>> import polars as pl
    >>> est = StreamingHDFE(
    ...     y="y",
    ...     x=["x", ("late", (pl.col("year") >= 2010).cast(pl.Float64))],
    ...     fe=["worker_id[t]", "firm_id"],
    ...     workdir="scratch")
    >>> fit = est.fit("data/*.parquet", vcov={"CRV1": "firm_id"})
    """

    # Hooks for subclasses (see passes.py): sort each fe[0] group's rows by
    # cell, and sum the variables per cell in passes 1/1b.
    cells_contiguous = False
    cell_sums = True

    def __init__(self, y: Variables, x: Variables | None, fe: Sequence[str],
                 workdir: PathLike | None = None, solver: str = "auto",
                 precond: str = "jacobi", keep: Sequence[str] = (),
                 tol: float = 1e-10, maxiter: int = 5000, batch_rows: int = 2_000_000,
                 row_group_size: int = 500_000, n_buckets: int | None = None,
                 rows_per_bucket: int = 20_000_000, cells_in_memory: bool = False,
                 rhs_block: int = 8, assembly: str = "auto", max_s_gb: float | None = None,
                 collin_tol: float = 1e-10, collin_tol_rel: float = 1e-6,
                 n_threads: int | None = None, scratch_mb: int = 32,
                 fixef_rm: str = "singleton", connected: str | None = None,
                 min_degree: int | tuple[int, int] = 1,
                 connected_fe: str | Sequence[str] | None = None, row_id: str = "row_id",
                 weights: str | pl.Expr | None = None, weights_type: str = "aweights",
                 stream: str | None = None, models: list[dict[str, Any]] | None = None,
                 verbose: bool = True, logger: logging.Logger | None = None,
                 log_level: int = logging.INFO, outputs: str = "auto",
                 save_resid: bool = True, keep_intermediates: bool = False) -> None:
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
        self.batch_rows, self.rgs = batch_rows, row_group_size
        self.scratch_mb = int(scratch_mb)
        self.n_buckets, self.rows_per_bucket = n_buckets, rows_per_bucket
        self.cells_in_memory = cells_in_memory
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
        self.slope_redundancy = None    # found once per fit by _redundant_slopes
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
        """Passes 0, 1 and 1b and the connected components: everything a fit
        does before the solve, shared with StreamingGLM."""
        self._pass0_code(source, keys, extra)
        self._after_pass0()
        if self.no_fe:
            self._no_cells()
        else:
            self._pass1_cells()
            self._pass1b_identifying()
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
            fe_dof: str = "exact") -> HDFEResult | HDFEMulti:
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

        Returns
        -------
        HDFEResult or HDFEMulti
            One result, or several when there are several models.
        """
        t0 = time.time()
        default = _vcov_key(vcov)
        keys = list(dict.fromkeys(["iid", "hetero", default]))
        extra = [_canon_cluster(c) for c in cluster]
        self.cluster_cols = list(dict.fromkeys(
            c for req in [k.split(":", 1)[1] for k in keys if k.startswith(("CRV1:", "CRV3:"))]
            + extra
            for part in req.split("+") for c in part.split("^")))
        self._init_run()
        try:
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
        peak = self.disk_peak / 1e9
        kept = _dir_bytes(self.workdir) / 1e9
        for r in results:
            r.diagnostics["disk_peak_gb"] = round(peak, 3)
            r.diagnostics["disk_results_gb"] = round(kept, 3)
        self._log(f"done in {time.time()-t0:.1f}s; peak disk use {peak:.2f} GB, result files "
                  f"{kept:.2f} GB in {self.workdir}"
                  + (" (removed with the results)" if self.outputs == "auto" else ""))
        return results[0] if len(results) == 1 else HDFEMulti(results)

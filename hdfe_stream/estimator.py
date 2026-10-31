"""`StreamingHDFE`: the estimator itself -- the driver in
`core/estimator.py`, with the leave-out (Kline-Saggio-Solvsten) estimators
mixed in from `inverse.py`, `leaveout.py`, `leaveout_se.py` and
`leaveout_weakid.py`.
"""

from __future__ import annotations

from .core.estimator import _StreamingBase
from .inverse import _InverseMixin
from .leaveout import _ComponentsMixin, _LeaveOutMixin, _TraceMixin
from .leaveout_se import _StandardErrorMixin
from .leaveout_weakid import _WeakIdMixin


class StreamingHDFE(_StreamingBase, _InverseMixin, _LeaveOutMixin, _TraceMixin,
                    _ComponentsMixin, _StandardErrorMixin, _WeakIdMixin):
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
        ``"auto"``: ``"explicit"`` unless S would not fit in memory, then
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
    device : {"cpu", "cuda"}, default "cpu"
        Experimental. ``"cuda"`` runs the solve on an NVIDIA GPU through CuPy
        (``pip install cupy-cuda13x``, or the build for your CUDA version):
        S is built on the CPU, copied to the GPU once, and the
        conjugate-gradient iterations run there. Needs ``solver="auto"`` or
        ``"explicit"`` and ``precond="jacobi"``; if S does not fit,
        ``"auto"`` falls back to ``"stream_cg"`` on the CPU, as without a
        GPU. Only the solve moves: the passes over the rows, and everything
        after the fit, stay on the CPU. The estimates agree with the CPU's to
        within the solver tolerance, not to the last digit. A GPU's double
        precision is only fast on data-center cards (A100, H100, B200). On
        an H100 it did not make the AKM benchmark's solve faster; see the
        "CPU and GPU" benchmark in the documentation.
    polars_engine : {"cpu", "gpu"}, default "cpu"
        Experimental. ``"gpu"`` runs the fit's Polars queries (pass 0's scan,
        factorization, partitioning and sorts, and the checks after the
        solve) on the cudf-polars GPU engine (``pip install
        cudf-polars-cu13``). A query the engine cannot run is run on the CPU
        instead, and listed in ``diagnostics["polars_gpu_fallbacks"]``.
        cudf-polars supports a narrow range of Polars versions, so installing
        it can change yours. With ``device="cuda"`` as well, on an H100, it
        made the AKM benchmark about 10% faster at 5 million workers, and
        slower at a million and below.
    tol : float, default 1e-10
        Conjugate-gradient convergence tolerance, on each column's residual
        relative to its right-hand side.
    maxiter : int, default 5000
        The most conjugate-gradient iterations; a solve that reaches it warns.
    max_memory_gb : float, optional
        The memory this fit may use, in GB. Default: the job's allocation (a
        cgroup limit, ``STREAM_LIMITS_MEMORY_GB``) or else the machine's. The
        choices that trade speed for memory are made against what is left of
        it (the limit, less `memory_reserve`, less what the process holds
        now): the explicit matrix S is built if it fits, and otherwise
        ``"stream_cg"`` streams it; per-thread accumulators drop to fewer
        copies (fewer threads on those passes) when they would not fit; the
        sort buckets of pass 0 shrink in a small allocation; and an
        estimate that cannot be made smaller (CRV3, the wild-bootstrap sums)
        raises ``MemoryError`` up front with the numbers, not at the
        operating system's kill. Polars' own buffers follow its out-of-core
        budget instead, which is fixed at import (see `stream_limits`).
    memory_reserve : float, optional
        The share of the memory limit held back from those choices for what
        they do not model (allocator slack, transient copies, pages a cgroup
        charges for files), 0 to 1. Default 0.15
        (``STREAM_LIMITS_MEMORY_RESERVE``): lower it to let the fast paths
        run closer to the limit, raise it for a job that must not be killed.
    rhs_block : int, default 8
        Variables solved at a time; bounds solver memory at about
        6 x (total non-streamed levels) x rhs_block doubles.
    batch_rows : int, optional
        Rows per chunk when streaming Parquet, and per slice of whole groups
        of the streamed dimension when building the cell table. Default:
        chunks of about 64 MB of the design, between 100,000 and 2,000,000
        rows, for GLMs (`StreamingGLM`) and for linear models without fixed
        effects, which keeps memory proportional to the design's width
        rather than to a row count; 2,000,000 rows for linear models with
        fixed effects.
    rows_per_bucket : int, optional
        Target rows per bucket: rows are hash-partitioned by the streamed
        dimension into buckets so the sort is bucket-sized. Each bucket is
        sorted in memory, which sets pass 0's peak; the default is about
        384 MB of rows at their width (the design and the source columns),
        between 500,000 and 20,000,000 rows.
    n_buckets : int, optional
        The number of buckets, instead of `rows_per_bucket`.
    row_group_size : int, default 500_000
        Rows per row group of the working and result Parquet files.
    assembly : {"auto", "cells", "rows"}, default "auto"
        ``"cells"``: within-cell cross-products accumulated in pass 1;
        ``"rows"``: an extra row pass after the solve, with no m^2
        aggregation columns; ``"auto"``: cells when there are at most 8
        variables.
    collapse_cells : {"auto", True, False}, default "auto"
        Whether the rows are collapsed into cells, one per combination of
        fixed-effect levels within each group of the streamed dimension.
        Collapsing makes the later passes cheaper when combinations repeat (a
        worker's years at one firm, with ``worker_id + firm_id``). It costs a
        wider sort key in pass 0, and buys nothing when they do not repeat
        (``worker_id + firm_id^year`` on a worker-year panel, or
        ``worker_id + firm_id + year``), where every row is its own cell.
        ``"auto"`` counts the distinct combinations approximately in pass 0's
        scan, at little cost, and collapses unless they are at least 90% of
        the rows; True always collapses, False never does (rows that happen to
        be adjacent and identical still share a cell). The estimates are the
        same either way. GLMs and fits with ``keep_intermediates`` always
        collapse. ``diagnostics["cells_collapsed"]`` says which happened.
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
        Numba and BLAS threads for this fit, restored afterwards. Default:
        all available (or the scheduler's allocation; see `stream_limits`).
        Polars' threads are fixed for the process and are not affected: cap
        them with ``STREAM_LIMITS_MAX_THREADS`` or `stream_limits.limit_threads`
        before polars is imported.
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

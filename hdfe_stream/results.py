"""Fit results: `HDFEResult` for one model, `HDFEMulti` for a formula that
expands to several.
"""

from __future__ import annotations

import json
import logging
import math
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

import numpy as np
import polars as pl
from scipy import special

from ._columns import PREFIX, check_source
from ._types import Vcov
from .report import _emit

if TYPE_CHECKING:
    from .leaveout import LeaveOutComponents


def _json_safe(x: Any) -> Any:
    """Convert numpy scalars/arrays, paths and nested containers to plain
    JSON types; nan and infinity become null (JSON has no such numbers)."""
    if isinstance(x, dict):
        return {str(k): _json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [_json_safe(v) for v in x]
    if isinstance(x, np.ndarray):
        return _json_safe(x.tolist())
    if isinstance(x, np.generic):
        return _json_safe(x.item())
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if x is None or isinstance(x, (bool, int, str)):
        return x
    return str(x)


def _dump_json(obj: Any, path: str | Path | None, indent: int | None) -> str:
    text = json.dumps(obj, indent=indent, allow_nan=False)
    if path is not None:
        Path(path).write_text(text + "\n", encoding="utf-8")
    return text


# --------------------------------------------------------------------------
# results
# --------------------------------------------------------------------------

@dataclass
class HDFEResult:
    """The result of one model: coefficients, standard errors, fit statistics,
    and lazy access to the fixed effects and the row-level output.

    Returned by `feols_stream`, `fepois_stream`, `feglm_stream` and
    `StreamingHDFE.fit` (or within an `HDFEMulti`). The fixed effects and
    residuals are Parquet files in the fit's run directory, which live as
    long as this object (unless the fit used ``outputs="keep"``): collect or
    sink what you need from `fixef` and `resid` before dropping it.

    Attributes
    ----------
    fml : str
        The model's formula.
    depvar : str
        The dependent variable.
    coefnames : list of str
        Coefficient names, as pyfixest names them.
    beta : numpy.ndarray
        The coefficients, in the order of `coefnames`.
    se : numpy.ndarray
        Their standard errors, under the current vcov.
    vcov : numpy.ndarray
        The variance-covariance matrix of the coefficients.
    vcov_type : str
        Which vcov that is: ``"iid"``, ``"hetero"``, ``"CRV1:var"``, ...
    all_vcovs : dict
        Every vcov computed at fit time, for `with_vcov`.
    n_clusters : dict
        The number of clusters of each clustering computed.
    df_t : int
        Degrees of freedom of the t distribution for p-values.
    n_obs : int
        Observations used (the sum of the weights for frequency weights).
    fe_names : list of str
        The fixed-effect dimensions.
    n_levels : dict
        The number of levels of each dimension.
    k_fe : int
        Fixed-effect parameters, net of the redundant ones (see ``fe_dof`` in
        `StreamingHDFE.fit`).
    n_components : int
        Connected components of the first two dimensions.
    n_identifying : int
        Groups of the streamed dimension that carry identifying information
        (movers, in an AKM model).
    rss : float
        Residual sum of squares; nan for GLMs.
    r2 : float
        R-squared; nan for GLMs.
    adj_r2 : float
        Adjusted R-squared; nan for GLMs.
    r2_within : float
        Within R-squared, net of the fixed effects; nan for GLMs.
    adj_r2_within : float
        Adjusted within R-squared; nan for GLMs.
    rmse : float
        Root mean squared error; nan for GLMs.
    family : str or None
        ``"poisson"``, ``"logit"`` or ``"probit"`` for a GLM; None for OLS.
    deviance : float
        For a GLM, the deviance; nan for OLS.
    loglik : float
        For a GLM, the log-likelihood; nan for OLS.
    pseudo_r2 : float
        For a GLM, McFadden's pseudo R-squared; nan for OLS.
    is_iv : bool
        Whether the model was estimated by 2SLS.
    first_stage : list of HDFEResult
        For IV, the first-stage regressions.
    f_stat_1st_stage : list of float
        For IV, the Wald F of the excluded instruments, under the same vcov.
    collin_vars : list of str
        Covariates dropped as collinear.
    weights : str or None
        The weights column, if any.
    weights_type : str
        ``"aweights"`` or ``"fweights"``.
    solver_info : dict
        The solver used, its iterations, convergence and timing.
    diagnostics : dict
        Everything else the fit recorded: the streamed dimension and why
        (``"stream"``), dropped observations (``"singletons"``,
        ``"connected"``, ``"separation"``), IRLS steps (``"irls"``), disk and
        time (``"disk_peak_gb"``, ``"seconds_total"``), and more.
    paths : dict
        The result files.
    """

    fml: str
    depvar: str
    coefnames: list
    fe_names: list
    beta: np.ndarray
    vcov: np.ndarray
    vcov_type: str
    df_t: int
    n_obs: int
    n_levels: dict
    n_identifying: int
    n_components: int
    k_fe: int
    rss: float
    r2_within: float
    solver_info: dict
    paths: dict
    collin_vars: list = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)
    all_vcovs: dict = field(default_factory=dict)
    r2: float = np.nan
    adj_r2: float = np.nan
    adj_r2_within: float = np.nan
    rmse: float = np.nan
    is_iv: bool = False
    first_stage: list = field(default_factory=list)
    f_stat_1st_stage: list = field(default_factory=list)
    n_clusters: dict = field(default_factory=dict)
    weights: str | None = None
    weights_type: str = "aweights"
    # GLMs (fepois_stream, feglm_stream): the family, and fit statistics in
    # place of the least-squares ones (rss, r2 and rmse are then nan)
    family: str | None = None
    deviance: float = np.nan
    loglik: float = np.nan
    pseudo_r2: float = np.nan
    _run: object = field(default=None, repr=False, compare=False)
    _estimator: object = field(default=None, repr=False, compare=False)
    _sample: dict | None = field(default=None, repr=False, compare=False)

    @property
    def se(self) -> np.ndarray:
        """Standard errors, under the current vcov."""
        return np.sqrt(np.diag(self.vcov))

    def coef(self) -> dict:
        """The coefficients, as ``{name: estimate}``."""
        return dict(zip(self.coefnames, self.beta))

    def tidy(self) -> pl.DataFrame:
        """Coefficient table. p-values use the t distribution with `df_t`
        degrees of freedom, or the normal for GLMs, as in pyfixest."""
        se = self.se
        t = np.divide(self.beta, se, out=np.full_like(self.beta, np.nan), where=se > 0)
        p = 2 * (special.ndtr(-np.abs(t)) if self.family else special.stdtr(self.df_t, -np.abs(t)))
        return pl.DataFrame({
            "Coefficient": self.coefnames,
            "Estimate": self.beta,
            "Std. Error": se,
            "t value": t,
            "Pr(>|t|)": p,
        }, schema_overrides={"Coefficient": pl.Utf8})

    def with_vcov(self, vcov: Vcov) -> "HDFEResult":
        """A copy of this result with another of the vcovs computed at fit
        time: ``"iid"``, ``"hetero"``, or a clustering given as
        ``{"CRV1": var}`` or ``"CRV1:var"`` (likewise CRV3, when it was the
        fit's vcov). Ask for clusterings at fit time with ``vcov=`` or
        ``cluster=``."""
        import copy
        key = _vcov_key(vcov)
        if key not in self.all_vcovs:
            raise KeyError(f"{key} was not computed; available: {list(self.all_vcovs)}. "
                           "Request it at fit time with vcov= or cluster=.")
        out = copy.copy(self)
        v, dft = self.all_vcovs[key]
        out.vcov, out.vcov_type, out.df_t = v, key, dft
        return out

    def to_pyfixest(self) -> Any:
        """A pyfixest ``Feols`` (``Feiv`` for IV) view of this result, for
        ``pf.etable``, ``pf.summary``, ``pf.coefplot`` and ``pf.iplot``. Only
        the stored results are available; methods that need the data
        (``predict``, recomputing the vcov, the wild bootstrap) are not."""
        from .reporting import _to_pyfixest   # reporting imports this module
        return _to_pyfixest(self)

    def leave_out_kss(self, n_draws: int = 250, seed: int = 0, block: int | None = None,
                      psi: str | None = None, leave_out: str = "match",
                      stayers: str | None = None, se: bool = False,
                      se_draws: int | None = None, se_trace: bool = True,
                      diagnose: bool | None = None, diagnose_draws: int = 64,
                      weak_interval: bool = True, confidence: float = 0.95,
                      centering: str = "reference",
                      se_variance: str = "person_year") -> LeaveOutComponents:
        """Kline-Saggio-Solvsten leave-out variance components for this fit.

        This is the secondary way in, for when you want the regression in its
        own right as well. The top-level `hdfe_stream.leave_out_kss` is the
        usual one, and does the whole sequence.

        The fit must be on a panel that is already leave-one-out connected
        (fit with ``connected="leave_one_out"``, or on data pruned with
        ``connected_components(..., leave_one_out=True)``). If you have not
        pruned, use the top-level function: pruning changes the estimation
        sample, so a fit made before pruning is a fit of a different model.

        Parameters
        ----------
        n_draws, seed, block, psi, se, se_draws, se_trace, diagnose, diagnose_draws, weak_interval, confidence
            As in `hdfe_stream.leave_out_kss`.
        leave_out : {"match", "observation"}, default "match"
            The unit left out. ``"match"`` is a whole worker-firm spell, as
            LeaveOutTwoWay, VarianceComponentsHDFE.jl and xhdfe do by default.
            It is robust to errors correlated within a spell. Everything
            except the worker and firm effects is partialled out first and the
            data collapsed to one row per match, so this fit is the
            regression, not the leave-out one. Standard errors for var(psi)
            and the covariance, following LeaveOutTwoWay's
            leave_out_COMPLETE, whose match-level option is marked beta there.
            ``"observation"`` is a single person-year, and needs this fit made
            with ``keep_intermediates=True``.
        stayers : str, optional
            At observation level only (``"own"`` by default); at match level
            the reference's within-match rule is used.
        centering : {"reference", "weighted"}, default "reference"
            At match level only. ``"reference"`` is LeaveOutTwoWay's:
            sqrt(w) ybar less its mean over matches. ``"weighted"`` centers
            ybar at its weighted mean first. They coincide when spells are of
            equal length or the outcome is centered near zero; otherwise both
            are unbiased but the reference's carries the outcome's level into
            sigma2, and with log earnings its standard deviation was 1.7 to 3
            times the weighted one's (docs/kss/differences.md, 3.4).
        se_variance : {"person_year", "match"}, default "person_year"
            At match level with ``se=True``: how each match's error variance
            is estimated. ``"person_year"`` is the reference implementation's
            (leave_out_COMPLETE, matches, beta there), from the person-year
            residuals, pooling the variation within the spell; it assumes
            errors independent within a spell and needs an unweighted fit.
            ``"match"`` goes beyond the reference: the collapsed
            leave-match-out estimate alone, which stays right when errors are
            correlated within a spell, at the cost of more noise
            (docs/kss/differences.md, 5.7).

        Returns
        -------
        LeaveOutComponents
        """
        if self.family:
            raise ValueError("leave-out variance components are for linear models; "
                             f"this is a {self.family} model")
        estimator = self._estimator
        if estimator is None:
            raise RuntimeError(
                "this result has no estimator attached, so its intermediates "
                "cannot be reached; use the top-level hdfe_stream.leave_out_kss")
        from .leaveout import _check_leave_out

        if len(estimator.fe_user) < 2:
            raise ValueError("leave-out variance components need two fixed effects "
                             "(worker and firm); this model has "
                             f"{len(estimator.fe_user) or 'none'}")

        _check_leave_out(leave_out, se, stayers, centering, se_variance,
                         self.weights)

        if leave_out == "match":
            if len(estimator.fe_user) != 2 and psi is None:
                raise ValueError(
                    "with more than two fixed effects, name the second one of "
                    "the pair with psi=; the rest are partialled out")
            from .leaveout_match import leave_out_match

            alpha = estimator.g_fe
            psi = psi or estimator.o_fe[0]
            return leave_out_match(
                self, alpha, psi, estimator.workdir / "leave_out_match",
                n_draws=n_draws, seed=seed, block=block, se=se,
                se_draws=se_draws, se_trace=se_trace, diagnose=diagnose,
                diagnose_draws=diagnose_draws, weak_interval=weak_interval,
                confidence=confidence, verbose=estimator.verbose,
                logger=estimator.logger, centering=centering,
                se_variance=se_variance)

        if not getattr(estimator, "keep_intermediates", False):
            raise RuntimeError(
                "leaving out an observation needs the fit's intermediates, which "
                "are deleted unless the fit was made with keep_intermediates="
                "True. Refit with feols_stream(..., keep_intermediates=True), or "
                "use the top-level hdfe_stream.leave_out_kss, which handles this "
                "and the pruning for you")
        estimator.reload_intermediates()
        try:
            return estimator.leave_out_components(
                self, n_draws=n_draws, block=block, seed=seed, psi=psi,
                stayers="own" if stayers is None else stayers, se=se,
                se_draws=se_draws, se_trace=se_trace, diagnose=diagnose,
                diagnose_draws=diagnose_draws, weak_interval=weak_interval,
                confidence=confidence)
        finally:
            estimator.unload_intermediates()

    def _scan(self, path, what):
        if path is None:
            raise FileNotFoundError(f"{what} was not saved (save_resid=False)")
        if not Path(path).exists():
            raise FileNotFoundError(
                f"{what} file is gone: result files were cleaned up (cleanup() was called, "
                "or, with outputs='auto', every result object of the fit was garbage-"
                "collected). Keep the result object alive while using its lazy frames, "
                "collect/sink what you need, or fit with outputs='keep'.")
        return pl.scan_parquet(path)

    def fixef(self, name: str) -> pl.LazyFrame:
        """Lazy scan of one dimension's estimated fixed effects: one row per
        level, with its effect (``fe_<name>``, and a column per slope with
        varying slopes), its number of observations and, for the first
        non-streamed dimension, its connected component. The file lives as
        long as this result object (``outputs="auto"``): collect or sink what
        you need before dropping the result."""
        if name not in self.paths["fe"]:
            raise KeyError(f"{name!r} is not a fixed effect of this model; it has "
                           + (", ".join(map(repr, self.paths["fe"])) or "none"))
        return self._scan(self.paths["fe"][name], f"fixed effects for {name!r}")

    def resid(self) -> pl.LazyFrame:
        """Lazy scan of the row-level output: the outcome, ``xb``, each row's
        fixed effects and the residual (for GLMs also ``eta``, ``fitted`` and
        ``resid_working``), plus any ``keep=`` columns. Same lifetime caveat as
        `fixef`. The ``row_id`` column is each row's position in the source,
        so the residuals join back onto `sample` (or onto the source with
        ``with_row_index("row_id")``) on it. The file is in the estimator's
        own order, not the source's; sort by ``row_id`` to restore it."""
        return self._scan(self.paths["resid"], "residual")

    def sample(self, check: bool = True) -> pl.LazyFrame:
        """The source, lazily, with three columns added saying what the fit
        did with each row.

        ``row_id``
            The row's position in the source (the first column); the same
            column `resid` carries.
        ``in_sample``
            True for the rows the model was estimated on.
        ``dropped_because``
            Null for those rows. Otherwise ``"missing"`` (a null in any column
            the fit uses, a NaN in a fixed effect or cluster, or a non-finite
            outcome, covariate or weight), ``"min_degree"`` (a level with too
            few distinct partners, ``min_degree``), ``"disconnected"`` (outside
            the largest connected set, ``connected="largest"``),
            ``"leave_one_out"`` (outside the leave-one-out connected set,
            ``connected="leave_one_out"``), ``"singleton"`` (``fixef_rm``) or
            ``"separation"`` (GLMs).

        Nothing is stored per row: the frame is the source plus an expression
        and the small tables of levels the fit dropped, so it costs a scan of
        the source when collected, and a join with those tables.

        The source must be the same rows in the same order as at the fit. A
        DataFrame is, and so is a file or glob (Polars keeps the file list and
        metadata in the plan, and reading a rewritten file fails); a LazyFrame
        is only if its plan is, which a group_by, an unordered join or a
        random sample is not.

        Parameters
        ----------
        check : bool, default True
            Compare the source's row count with the count at the fit. This
            catches a plan that returns a different number of rows each time
            it runs, not one that returns the same rows in a different order.

        Returns
        -------
        polars.LazyFrame
        """
        spec = self._sample
        if spec is None:
            raise ValueError("this result holds no reference to its source (it was "
                             "converted, or fitted without one)")
        source, row_id = spec["source"], spec["row_id"]
        check_source(source.collect_schema().names())
        names = ("in_sample", "dropped_because")
        clash = [c for c in (row_id, *names) if c in source.collect_schema()]
        if clash:
            raise ValueError(f"the source has columns named {clash}, which sample() adds; "
                             "rename them")
        if check:
            n = source.select(pl.len()).collect(engine="streaming").item()
            if n != spec["n_raw"]:
                raise ValueError(
                    f"the source has {n:,} rows, and had {spec['n_raw']:,} when the model "
                    "was fitted, so row_id would no longer line up with the residuals")
        frame = source.with_row_index(row_id)
        flags = {}
        for kind, tables in spec["tables"].items():
            for d, table in tables.items():
                flag = f"{PREFIX}{kind}_{len(flags.get(kind, []))}"
                # the source's own order, which a join keeps only when asked
                frame = frame.join(table.lazy().with_columns(pl.lit(True).alias(flag)),
                                   on=spec["fe_cols"][d], how="left",
                                   maintain_order="left")
                flags.setdefault(kind, []).append(flag)
        reason = pl.when(~spec["usable"]).then(pl.lit("missing"))
        for kind in ("min_degree", "disconnected", "leave_one_out", "singleton",
                     "separation"):
            if kind in flags:
                reason = reason.when(pl.any_horizontal(
                    [pl.col(f).is_not_null() for f in flags[kind]])).then(pl.lit(kind))
        levels = ["missing", "min_degree", "disconnected", "leave_one_out", "singleton",
                  "separation"]
        return (frame.with_columns(reason.otherwise(None).cast(pl.Enum(levels))
                                   .alias("dropped_because"))
                     .with_columns(pl.col("dropped_because").is_null().alias("in_sample"))
                     .drop([f for fs in flags.values() for f in fs])
                     .select(row_id, *source.collect_schema().names(), "in_sample",
                             "dropped_because"))

    @property
    def files_dir(self) -> Path | None:
        """Directory holding this model's result files."""
        return self.paths.get("dir")

    def cleanup(self) -> None:
        """Delete this model's result files (and its first-stage files for
        IV); the run directory goes once no model files remain."""
        for fs in self.first_stage:
            fs.cleanup()
        if self.paths.get("dir"):
            shutil.rmtree(self.paths["dir"], ignore_errors=True)
        run = self._run
        if run is not None and run.exists:
            models = run.path / "models"
            if not models.exists() or not any(models.iterdir()):
                run.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.cleanup()
        return False

    def summary_text(self) -> str:
        """The summary() report as a string."""
        lines = [f"### {self.fml}",
                 f"Streaming HDFE{f' ({self.family})' if self.family else ''}   "
                 f"vcov: {self.vcov_type}",
                 "obs: {:,}   ".format(self.n_obs)
                 + "   ".join(f"{k}: {v:,}" for k, v in self.n_levels.items())]
        st = self.diagnostics.get("stream")
        if st and st["dim"] is not None:
            lines.append(f"streamed dimension: {st['dim']} ({st['reason']})")
        if len(self.fe_names) >= 2:
            lines.append(f"identifying {self.fe_names[0]} groups: {self.n_identifying:,}   "
                         f"({self.fe_names[0]} x {self.fe_names[1]}) components: "
                         f"{self.n_components:,}")
        conn = self.diagnostics.get("connected", {})
        pruned = conn.get("min_degree", {})
        if pruned.get("observations"):
            lines.append(f"dropped by min_degree={tuple(pruned['min_degree'])} on "
                         f"{' x '.join(conn['fe'])}: {pruned['observations']:,} observations")
        loo = conn.get("leave_one_out", {})
        if loo.get("observations"):
            lines.append(f"dropped outside the leave-one-out connected set of "
                         f"{' x '.join(conn['fe'])}: {loo['observations']:,} observations")
        if conn.get("observations"):
            lines.append(f"dropped outside the largest connected set of "
                         f"{' x '.join(conn['fe'])}: {conn['observations']:,} observations "
                         f"in {conn['components'] - 1:,} other component"
                         + ("s" if conn["components"] > 2 else ""))
        singles = self.diagnostics.get("singletons", {})
        if singles.get("observations"):
            lines.append(f"dropped as singletons: {singles['observations']:,} observations")
        if self.family:
            fit = (f"deviance: {self.deviance:.8g}   log-likelihood: {self.loglik:.8g}   "
                   f"pseudo R2: {self.pseudo_r2:.6f}")
            irls = self.diagnostics.get("irls", {})
            if irls:
                fit += (f"   IRLS steps: {irls['iterations']}"
                        + ("" if irls["converged"] else " (not converged)"))
            if self.fe_names:
                fit = f"FE dof: {self.k_fe:,}   {fit}"
            sep = self.diagnostics.get("separation", {})
            if sep.get("observations"):
                lines.append(f"dropped for separation: {sep['observations']:,} observations")
        else:
            fit = f"RSS: {self.rss:.6g}   RMSE: {self.rmse:.4g}   R2: {self.r2:.6f}"
            if self.fe_names:
                fit = (f"FE dof: {self.k_fe:,}   {fit}   "
                       f"within R2: {self.r2_within:.6f}")
        lines.append(fit)
        if self.weights:
            lines.append(f"weights: {self.weights} ({self.weights_type})")
        if self.is_iv:
            lines.append("first-stage F (excluded instruments, same vcov): "
                         + ", ".join(f"{fs.depvar}: {f:.4g}"
                                     for fs, f in zip(self.first_stage, self.f_stat_1st_stage)))
        if self.collin_vars:
            lines.append(f"dropped as collinear: {self.collin_vars}")
        lines.append(f"solver: {self.solver_info}")
        lines.append(str(self.tidy()))
        return "\n".join(lines)

    def summary_dict(self, coefficients: bool = False) -> dict:
        """The model-level information of summary() as a dict of plain Python
        types (JSON-ready; nan is None). Fit statistics not defined for the
        model (e.g. r2 for a GLM) are absent. `coefficients=True` adds the
        `tidy()` table as a list of row dicts."""
        st = self.diagnostics.get("stream") or {}
        d: dict[str, Any] = {
            "formula": self.fml,
            "depvar": self.depvar,
            "family": self.family or "gaussian",
            "is_iv": self.is_iv,
            "vcov": self.vcov_type,
            "df_t": self.df_t,
            "n_obs": self.n_obs,
            "fixed_effects": list(self.fe_names),
            "n_levels": dict(self.n_levels),
            "n_clusters": dict(self.n_clusters),
            "streamed_dimension": st.get("dim"),
            "streamed_reason": st.get("reason") if st.get("dim") is not None else None,
            "n_identifying": self.n_identifying if len(self.fe_names) >= 2 else None,
            "n_components": self.n_components if len(self.fe_names) >= 2 else None,
            "fe_dof": self.k_fe if self.fe_names else None,
            "weights": self.weights,
            "weights_type": self.weights_type if self.weights else None,
            "collinear_dropped": list(self.collin_vars),
            "solver": self.solver_info,
        }
        if self.family:
            d.update(deviance=self.deviance, loglik=self.loglik, pseudo_r2=self.pseudo_r2,
                     irls=self.diagnostics.get("irls"),
                     separation=self.diagnostics.get("separation"))
        else:
            d.update(rss=self.rss, rmse=self.rmse, r2=self.r2, adj_r2=self.adj_r2)
            if self.fe_names:
                d.update(r2_within=self.r2_within, adj_r2_within=self.adj_r2_within)
        if self.is_iv:
            d["first_stage_f"] = {fs.depvar: f for fs, f
                                  in zip(self.first_stage, self.f_stat_1st_stage)}
        if coefficients:
            d["coefficients"] = self.tidy().to_dicts()
        return _json_safe(d)

    def summary_json(self, path: str | Path | None = None, indent: int | None = 2,
                     coefficients: bool = False) -> str:
        """The model-level information of summary() as a JSON string (see
        `summary_dict`); with `path`, also written there as UTF-8. Reload with
        `json.loads` or `json.load`."""
        return _dump_json(self.summary_dict(coefficients), path, indent)

    def summary(self, logger: logging.Logger | None = None,
                level: int = logging.INFO) -> None:
        """Print the report, or write it to `logger` (one record, at `level`)."""
        _emit(self.summary_text(), logger, level)

class HDFEMulti:
    """Results of a formula that expands to several models.

    Iterate over it, or take one model with `fetch_model`, to get each
    `HDFEResult`; ``len()`` counts them. The models are in the formula's
    order, and keyed by formula in ``all_fitted_models``.
    """

    def __init__(self, results: Iterable[HDFEResult]) -> None:
        self.all_fitted_models = {r.fml: r for r in results}

    def fetch_model(self, i: int) -> HDFEResult:
        """The i-th model, in the formula's order."""
        return list(self.all_fitted_models.values())[i]

    def to_pyfixest(self) -> list:
        """A pyfixest view of each model (see `HDFEResult.to_pyfixest`)."""
        return [r.to_pyfixest() for r in self]

    def sample(self, check: bool = True) -> dict[str, pl.LazyFrame]:
        """`{formula: HDFEResult.sample()}` for each model. Models fitted
        together on the same fixed effects share one sample, so their frames
        are the same; only a stepwise change in the fixed effects (or, for
        GLMs, in the outcome) changes it. Every frame uses the same `row_id`."""
        return {f: r.sample(check) for f, r in self.all_fitted_models.items()}

    def cleanup(self) -> None:
        """Delete the result files of all models."""
        for r in self:
            r.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.cleanup()
        return False

    def tidy(self) -> pl.DataFrame:
        """Every model's coefficient table, with a column ``fml`` first."""
        return pl.concat([r.tidy().with_columns(pl.lit(f).alias("fml"))
                          for f, r in self.all_fitted_models.items()]).select("fml", pl.all().exclude("fml"))

    def summary_text(self) -> str:
        """Every model's report, as one string."""
        return "\n\n".join(r.summary_text() for r in self.all_fitted_models.values())

    def summary_dict(self, coefficients: bool = False) -> dict:
        """`{formula: HDFEResult.summary_dict(...)}` for each model."""
        return {f: r.summary_dict(coefficients) for f, r in self.all_fitted_models.items()}

    def summary_json(self, path: str | Path | None = None, indent: int | None = 2,
                     coefficients: bool = False) -> str:
        """JSON object keyed by formula, each value as `HDFEResult.summary_json`;
        with `path`, also written there."""
        return _dump_json(self.summary_dict(coefficients), path, indent)

    def summary(self, logger: logging.Logger | None = None, level: int = logging.INFO,
                per_model: bool = False) -> None:
        """Print all reports, or write them to `logger`: one record in total,
        or one per model with per_model=True."""
        if logger is not None and per_model:
            for r in self.all_fitted_models.values():
                r.summary(logger, level)
        else:
            _emit(self.summary_text(), logger, level)

    def __len__(self):
        return len(self.all_fitted_models)

    def __iter__(self):
        return iter(self.all_fitted_models.values())


def _canon_cluster(var):
    """'worker_id + firm_id' -> 'worker_id+firm_id'; 'firm_id ^ year' -> 'firm_id^year'."""
    return "+".join("^".join(c.strip() for c in part.split("^")) for part in var.split("+"))


def _vcov_key(v):
    if v is None:
        return None
    if isinstance(v, dict):
        (kind, var), = v.items()
        v = f"{kind}:{var}"
    v = str(v)
    if v.lower() in ("iid",):
        return "iid"
    if v.lower() in ("hetero", "hc1"):
        return "hetero"
    kind, _, var = v.partition(":")
    if kind.upper() == "CRV1" and var:
        return "CRV1:" + _canon_cluster(var)
    if kind.upper() == "CRV3" and var:
        var = _canon_cluster(var)
        if "+" in var:
            raise ValueError(f"CRV3 is one-way only, got {var!r}; use CRV1 for "
                             "multi-way clustering")
        return "CRV3:" + var
    if kind.upper() in ("CRV1", "CRV3"):
        raise ValueError(f"{kind} needs a cluster variable, as in {{'{kind}': 'firm_id'}}")
    raise ValueError(f"unsupported vcov {v!r}: use 'iid', 'hetero'/'HC1', {{'CRV1': var}} "
                     "(multi-way: {'CRV1': 'a+b'}) or {'CRV3': var}")

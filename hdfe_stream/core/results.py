"""Fit results: `HDFEResult` for one model, `HDFEMulti` for a formula that
expands to several. What does not depend on the model -- inference, tests,
reports and export -- is their base classes' (results_base.py).
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

from ._columns import PREFIX, check_source
from .results_base import RegressionMulti, RegressionResult
from .utils import row_index

if TYPE_CHECKING:
    from ..leaveout import LeaveOutComponents


# --------------------------------------------------------------------------
# results
# --------------------------------------------------------------------------

@dataclass(kw_only=True)
class HDFEResult(RegressionResult):
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
        The log-likelihood: for a GLM, the family's; for OLS (not IV), the
        normal one at the maximum-likelihood error variance RSS / N (with
        analytic weights, the weighted least-squares one, which adds
        sum(log w) / 2; frequency weights count as copies of the row).
    loglik_null : float
        For a GLM, the log-likelihood of the constant-only model; nan for
        OLS.
    pseudo_r2 : float
        For a GLM, McFadden's pseudo R-squared, 1 - loglik / loglik_null: it
        credits the covariates and the fixed effects together; nan for OLS.
    loglik_fe_null : float
        For a GLM with fixed effects, the log-likelihood of the model with
        the fixed effects and no covariates: the groups' means with one
        fixed effect (and no offset), otherwise a fit of its own, made only
        with ``within_pseudo_r2=True``; else nan.
    pseudo_r2_within : float
        1 - loglik / loglik_fe_null: what the covariates add beyond the fixed
        effects (fixest's "within pseudo R2"); nan when `loglik_fe_null` is.
    aic, bic : float
        Akaike's and Schwarz's information criteria, -2 loglik + 2 p and
        -2 loglik + p log(N), with p the coefficients plus the fixed effects'
        degrees of freedom (`k_fe`), as fixest counts them. For OLS the error
        variance is not counted, as in statsmodels (R's ``lm`` counts it,
        adding 2 to the AIC); nan for IV.
    is_iv : bool
        Whether the model was estimated by 2SLS.
    first_stage : list of HDFEResult
        For IV, the first-stage regressions.
    f_stat_1st_stage : list of float
        For IV, the Wald F of the excluded instruments, under the same vcov.
    iv_diag : dict or None
        For IV, the identification and overidentification tests under the
        same vcov, with ivreg2's conventions (`hdfe_stream.core.ivtests`):
        ``kp_lm`` (Kleibergen-Paap rk LM underidentification test; Anderson's
        LM under iid), ``kp_wald`` (the rk Wald statistic and its ``F``, the
        weak-identification F; Cragg-Donald's under iid),
        ``sanderson_windmeijer`` (one conditional F per endogenous regressor),
        ``partial_r2`` (each endogenous regressor's first-stage partial R^2
        and Shea's partial R^2), ``effective_f`` (Montiel Olea and
        Pflueger's effective F and its 2SLS critical value for a 10% worst-case
        bias; one endogenous regressor only, else None), ``anderson_rubin``
        (the weak-instrument-robust test that the endogenous regressors'
        coefficients are zero, and with one of them its 95% confidence set),
        ``endogeneity`` (the C statistic for treating them as exogenous, as
        ivreg2's ``endog()``) and ``overid`` (Hansen's J, or Sargan's
        statistic under iid; None when exactly identified). Each test has its
        statistic, df and p.
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

    # with RegressionResult's fields: fml, depvar, coefnames, beta, vcov,
    # vcov_type, df_t, n_obs, all_vcovs, n_clusters, collin_vars, diagnostics,
    # weights and weights_type
    fe_names: list
    n_levels: dict
    n_identifying: int
    n_components: int
    k_fe: int
    rss: float
    r2_within: float
    solver_info: dict
    paths: dict
    r2: float = np.nan
    adj_r2: float = np.nan
    adj_r2_within: float = np.nan
    rmse: float = np.nan
    is_iv: bool = False
    first_stage: list = field(default_factory=list)
    f_stat_1st_stage: list = field(default_factory=list)
    iv_diag: dict | None = None
    # GLMs (fepois_stream, feglm_stream): the family, and fit statistics in
    # place of the least-squares ones (rss, r2 and rmse are then nan)
    family: str | None = None
    deviance: float = np.nan
    loglik: float = np.nan
    pseudo_r2: float = np.nan
    loglik_null: float = np.nan
    loglik_fe_null: float = np.nan
    pseudo_r2_within: float = np.nan
    aic: float = np.nan
    bic: float = np.nan
    _run: object = field(default=None, repr=False, compare=False)
    _estimator: object = field(default=None, repr=False, compare=False)
    _sample: dict | None = field(default=None, repr=False, compare=False)
    # bootcluster=: the per-cluster sums for the wild cluster bootstrap
    # (`wildboottest`): "AG" (G x k x k, sum w h h'), "sG" (G x k, sum w e h),
    # "levels" (the clusters' values, one row each), "cluster", "cols", "adj"
    # (the CRV1 small-sample factor) and "df" (G - 1)
    _boot: dict | None = field(default=None, repr=False, compare=False)

    def wildboottest(self, param: str, reps: int = 9999,
                     weights_type: str = "rademacher", impose_null: bool = True,
                     seed: int | None = None) -> dict:
        """Wild cluster bootstrap test of ``param = 0``.

        Needs the fit made with ``bootcluster=`` (see `StreamingHDFE.fit`);
        the bootstrap clusters by that variable, works from the per-cluster
        sums the fit kept, and reads no data. A test costs about reps x G x k
        operations, for G clusters and k coefficients: well under a second
        with 50 clusters.

        The bootstrap is the restricted wild cluster bootstrap (WCR, the
        "11" variant of pyfixest and fwildclusterboot), or the unrestricted
        one with ``impose_null=False``. Its draws are those of pyfixest's
        `wildboottest` with the same seed and weights, so the p-values agree
        with pyfixest's up to rounding, where pyfixest can fit the model
        (it puts the fixed effects in as dummy variables). With Rademacher
        weights and 2^G < reps, every sign pattern is used once instead, as
        in pyfixest.

        A draw that gives every cluster the same weight reproduces the sample
        (up to scale), and so the t statistic exactly. With few clusters and
        two-point weights such draws are common (with 12 clusters, about 2%
        of Mammen draws), and they count as at least as extreme as the t
        statistic: the conservative choice, and a deterministic one.
        pyfixest's comparison counts or skips them depending on rounding.

        Parameters
        ----------
        param : str
            The coefficient tested.
        reps : int, default 9999
            Bootstrap draws.
        weights_type : {"rademacher", "mammen", "webb", "norm"}, default "rademacher"
            The distribution of the cluster weights.
        impose_null : bool, default True
            Impose the null hypothesis on the bootstrap data-generating
            process (WCR), or not (WCU).
        seed : int, optional
            Seed for the draws.

        Returns
        -------
        dict
            ``param``; ``t value``, the t statistic under CRV1 by the bootstrap
            clustering (the one `tidy` reports under that vcov);
            ``Pr(>|t|)``, the bootstrap p-value, the share of draws whose
            absolute t statistic is at least as large; ``reps``, the draws
            used;
            ``weights_type``, ``impose_null`` and ``cluster``.
        """
        from .wildboot import _bootstrap, _coef_index

        if self._boot is None:
            raise ValueError("wildboottest needs the per-cluster sums, which the fit keeps "
                             "only when asked: refit with bootcluster='<cluster variable>'")
        t, tstar, _ = _bootstrap([(self, _coef_index(self, param))], reps, weights_type,
                                 impose_null, seed)
        return {"param": param, "t value": float(t[0]),
                "Pr(>|t|)": float(np.mean(_at_least(tstar[:, 0], t[0]))),
                "reps": tstar.shape[0], "weights_type": weights_type,
                "impose_null": impose_null, "cluster": self._boot["cluster"]}

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
        from ..leaveout import _check_leave_out

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
            from ..leaveout_match import leave_out_match

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
        frame = row_index(source, row_id)
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

    def predict(self, newdata: Any = None, type: str = "link",
                name: str = "prediction") -> pl.LazyFrame:
        """Predictions on new data, or on the source, as a lazy frame.

        Returns `newdata`, lazily and in its own order, with one column added:
        the linear predictor (each row's covariates times the coefficients,
        plus its fixed effects and, for a GLM, its offset), or with
        ``type="response"``, the mean. Nothing is computed until the frame is
        collected or sunk, so `newdata` can be larger than memory. The fixed
        effects come from the fit's files, which live as long as this result
        (see `fixef`): collect or sink the predictions before dropping it.

        A row gets a null prediction if any of its fixed-effect levels has no
        estimated effect (a level the estimation sample does not contain:
        new, or dropped, e.g. as a singleton), as pyfixest gives NaN, or if a
        covariate is null.

        The fixed effects are identified only within a connected set of the
        first two dimensions (workers and firms, say), so a prediction that
        combines levels from different components, such as a worker from one
        and a firm from another, depends on the normalization and means
        little. ``fixef(...)`` reports each level's ``component`` for the
        first non-streamed dimension.

        Parameters
        ----------
        newdata : DataFrame, LazyFrame, or Parquet path or glob, optional
            The rows to predict for, with every column the model uses. A
            categorical covariate's indicators are those of the fit, so a
            level the fit did not see counts as the base level. Default: the
            fit's own source, every row of it (the estimation sample's rows
            then reproduce the fitted values in `resid`).
        type : {"link", "response"}, default "link"
            For a GLM, the linear predictor, or the mean (exp for Poisson, the
            logistic or normal CDF for logit and probit). The same for OLS.
        name : str, default "prediction"
            The name of the added column.

        Returns
        -------
        polars.LazyFrame
        """
        from .utils import as_lazy

        if type not in ("link", "response"):
            raise ValueError(f"type must be 'link' or 'response', got {type!r}")
        est = self._estimator
        if est is None:
            raise ValueError("this result has no estimator attached (it was converted), "
                             "so it cannot predict")
        if newdata is None:
            if self._sample is None:
                raise ValueError("this result holds no reference to its source; pass "
                                 "newdata")
            frame = self._sample["source"]
        else:
            frame = as_lazy(newdata)
        schema = frame.collect_schema()
        check_source(schema.names())
        if name in schema:
            raise ValueError(f"newdata already has a column named {name!r}; pass "
                             "another name=")

        terms = [b * est.var_exprs[c] for c, b in zip(self.coefnames, self.beta)]
        offset = getattr(est, "offset_name", None)
        if offset is not None:
            terms.append(est.var_exprs["__offset__"])
        tmp, keys_tmp = [], []
        for i, d in enumerate(self.fe_names):
            keys = est.fe_cols[d]
            table = self.fixef(d)
            tschema = table.collect_schema()
            tkeys = [f"{PREFIX}key{i}_{j}" for j in range(len(keys))]
            eff = f"{PREFIX}fe{i}"
            slopes = [(f"{PREFIX}fe{i}_{j}", v) for j, v in enumerate(
                est.slope_vars if d in est.slopes else [])]
            table = table.select(*[pl.col(k).alias(t) for k, t in zip(keys, tkeys)],
                                 pl.col(f"fe_{d}").alias(eff),
                                 *[pl.col(f"fe_{d}[{v}]").alias(s) for s, v in slopes])
            frame = frame.with_columns(pl.col(k).cast(tschema[k], strict=False).alias(t)
                                       for k, t in zip(keys, tkeys))
            frame = frame.join(table, on=tkeys, how="left", maintain_order="left")
            terms.append(pl.col(eff))
            terms += [pl.col(s) * pl.col(v).cast(pl.Float64) for s, v in slopes]
            tmp += [eff, *(s for s, _ in slopes)]
            keys_tmp += tkeys

        needed = {r for t in terms for r in t.meta.root_names() if not r.startswith(PREFIX)}
        missing = sorted(needed - set(schema.names()))
        if missing:
            raise ValueError(f"newdata lacks columns the model uses: {missing}")
        pred = pl.sum_horizontal(terms) if terms else pl.lit(0.0)
        # sum_horizontal skips nulls; a null term (a level without an effect,
        # a null covariate) must make the prediction null
        if terms:
            pred = pl.when(pl.any_horizontal(t.is_null() for t in terms)).then(None) \
                     .otherwise(pred)
        if type == "response" and self.family:
            mu = est.family.mu          # the fit's own inverse link, on numpy
            pred = pl.when(pred.is_null()).then(None).otherwise(pred.map_batches(
                lambda s: pl.Series(mu(s.to_numpy())), return_dtype=pl.Float64,
                is_elementwise=True))
        return frame.with_columns(pred.cast(pl.Float64).alias(name)).drop(tmp + keys_tmp)

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

    # ------------------------------------------------------------ hooks
    @property
    def _normal_inference(self) -> bool:
        # GLMs: normal p-values and intervals, as pyfixest has them
        return bool(self.family)

    def _summary_lines(self) -> list[str]:
        lines = [f"Streaming HDFE{f' ({self.family})' if self.family else ''}   "
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
            if not np.isnan(self.pseudo_r2_within):
                fit += f"   within pseudo R2: {self.pseudo_r2_within:.6f}"
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
        bc = self.diagnostics.get("bias_correction")
        if bc:
            lines.append("coefficients bias-corrected for the incidental parameters of "
                         + " and ".join(bc["dims"])
                         + (f" (lags: {bc['lags']} periods of {bc['time']})" if bc["lags"]
                            else "")
                         + "; uncorrected in diagnostics['bias_correction']")
        if not np.isnan(self.aic):
            lines.append(f"AIC: {self.aic:.8g}   BIC: {self.bic:.8g}"
                         + ("" if self.family else f"   log-likelihood: {self.loglik:.8g}"))
        if self.weights:
            lines.append(f"weights: {self.weights} ({self.weights_type})")
        if self.is_iv:
            lines.append("first-stage F (excluded instruments, same vcov): "
                         + ", ".join(f"{fs.depvar}: {f:.4g}"
                                     for fs, f in zip(self.first_stage, self.f_stat_1st_stage)))
            lines += _iv_diag_lines(self.iv_diag)
        if self.collin_vars:
            lines.append(f"dropped as collinear: {self.collin_vars}")
        lines.append(f"solver: {self.solver_info}")
        return lines

    def _summary_fields(self) -> dict:
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
                     loglik_null=self.loglik_null, loglik_fe_null=self.loglik_fe_null,
                     pseudo_r2_within=self.pseudo_r2_within,
                     irls=self.diagnostics.get("irls"),
                     separation=self.diagnostics.get("separation"))
        else:
            d.update(rss=self.rss, rmse=self.rmse, r2=self.r2, adj_r2=self.adj_r2,
                     loglik=self.loglik)
            if self.fe_names:
                d.update(r2_within=self.r2_within, adj_r2_within=self.adj_r2_within)
        d.update(aic=self.aic, bic=self.bic)
        if self.is_iv:
            d["first_stage_f"] = {fs.depvar: f for fs, f
                                  in zip(self.first_stage, self.f_stat_1st_stage)}
            d["iv_diag"] = self.iv_diag
        return d

    def _pyfixest_attrs(self) -> dict:
        from .families import get_family
        from .reporting import _fixef_string    # reporting imports this module
        method = get_family(self.family).method if self.family else "feols"
        glm = method != "feols"
        return {
            # the FE string as written in the formula, without spaces: maketables
            # matches FE rows across models by the text between '+' signs, spaces
            # included, so "a + b + c" and "a + b" would not share their "b" row
            "_fixef": (_fixef_string(self.fml.split("|")[1] if "|" in self.fml
                                     else " + ".join(self.fe_names)) if self.fe_names else None),
            "_has_fixef": bool(self.fe_names),
            "_r2": self.r2, "_adj_r2": self.adj_r2, "_r2_adj": self.adj_r2,
            "_r2_within": self.r2_within, "_adj_r2_within": self.adj_r2_within,
            "_rmse": self.rmse,
            "deviance": self.deviance if glm else None, "_method": method,
            "_loglik": self.loglik if glm else None,
            "_pseudo_r2": self.pseudo_r2 if glm else None,
            "_is_iv": self.is_iv,
            "_f_stat_1st_stage": (self.f_stat_1st_stage[0]
                                  if self.is_iv and len(self.f_stat_1st_stage) == 1 else None),
        }


class HDFEMulti(RegressionMulti):
    """Results of a formula that expands to several models.

    Iterate over it, or take one model with `fetch_model`, to get each
    `HDFEResult`; ``len()`` counts them. The models are in the formula's
    order, and keyed by formula in ``all_fitted_models``.
    """

    def sample(self, check: bool = True) -> dict[str, pl.LazyFrame]:
        """`{formula: HDFEResult.sample()}` for each model. Models fitted
        together on the same fixed effects share one sample, so their frames
        are the same; only a stepwise change in the fixed effects (or, for
        GLMs, in the outcome) changes it. Every frame uses the same `row_id`."""
        return {f: r.sample(check) for f, r in self.all_fitted_models.items()}


def _iv_diag_lines(d):
    """The summary block of `HDFEResult.iv_diag`."""
    if not d:
        return []
    iid = d["vcov"] == "iid"
    lm, wd = d["kp_lm"], d["kp_wald"]
    out = [(f"underidentification: {'Anderson' if iid else 'Kleibergen-Paap rk'} LM "
            f"chi2({lm['df']}) = {lm['statistic']:.4g}, p = {lm['p']:.4g}"),
           (f"weak identification: {'Cragg-Donald' if iid else 'Kleibergen-Paap rk'} Wald F "
            f"= {wd['F']:.4g}")]
    sw = d["sanderson_windmeijer"]
    if len(sw) > 1:
        out.append("  Sanderson-Windmeijer F: " + ", ".join(
            f"{e['variable']}: F({e['df']}, {e['df2']}) = {e['F']:.4g}" for e in sw))
    ef = d.get("effective_f")
    if ef:
        out.append(f"  effective F (Montiel Olea-Pflueger) = {ef['F']:.4g}; 2SLS critical value "
                   f"(10% worst-case bias, 5% size) {ef['critical_value']:.4g}")
    pr = d.get("partial_r2") or {}
    if pr:
        out.append("  partial R2 / Shea partial R2: " + ", ".join(
            f"{k}: {v['partial']:.4g} / {v['shea']:.4g}" for k, v in pr.items()))
    ar = d.get("anderson_rubin")
    if ar:
        line = (f"weak-instrument-robust: Anderson-Rubin chi2({ar['df']}) = "
                f"{ar['statistic']:.4g}, p = {ar['p']:.4g} (H0: endogenous coefficients = 0)")
        cs = ar.get("confidence_set")
        if cs is not None:
            ivs = cs["intervals"]
            line += (f"; {cs['level']:.0%} set: "
                     + (" U ".join(f"[{lo:.4g}, {hi:.4g}]" for lo, hi in ivs) if ivs
                        else "empty"))
        out.append(line)
    en = d.get("endogeneity")
    if en and en["df"] > 0:
        out.append(f"endogeneity (C statistic, {', '.join(en['tested'])}): chi2({en['df']}) = "
                   f"{en['statistic']:.4g}, p = {en['p']:.4g}")
    o = d.get("overid")
    if o:
        name = "Sargan" if o["test"] == "sargan" else "Hansen J"
        out.append(f"overidentification: {name} chi2({o['df']}) = {o['statistic']:.4g}, "
                   f"p = {o['p']:.4g}")
    return out


def _at_least(tstar, t):
    """Bootstrap draws at least as extreme as t: |t*| >= |t|, with a draw that
    reproduces t (to rounding) counted as a tie, and so as at least as
    extreme, rather than left to the last bit of floating point."""
    return np.abs(tstar) >= np.abs(t) * (1 - 1e-9)

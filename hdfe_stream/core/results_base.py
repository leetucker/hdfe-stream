"""The model-independent part of a fit's results: `RegressionResult` for one
model, `RegressionMulti` for several.

A result here is a flat vector of estimates with its variance matrix, and
everything that follows from those alone: standard errors, the coefficient
table, confidence intervals, Wald tests, the summary report and its JSON,
the pyfixest view behind `etable`, and the multiple-testing corrections of
`multcomp.py`. `HDFEResult` (results.py) is the subclass the fixed-effects
estimators return; other estimators built on the same passes subclass it in
the same way.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import polars as pl
from scipy import special

from .report import _emit
from .typing import Vcov


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


@dataclass(kw_only=True)
class RegressionResult:
    """Estimates, their variance matrix, and what follows from them alone.

    The estimates are one flat vector, `beta`, named by `coefnames`, which is
    what lets `tidy`, `confint`, `wald_test`, `etable` and the corrections in
    `multcomp.py` treat every model the same way. A model with several
    equations or with ancillary parameters names them into the same vector:
    ``"<outcome level>:<variable>"`` for a multinomial logit's equations,
    ``"cut1"``, ``"cut2"``, ... for an ordered logit's cutpoints.

    A subclass adds its model's own fields and fills in four hooks, each of
    whose defaults suits a linear model without fixed effects:

    `_normal_inference`
        True for p-values and intervals from the normal distribution (and a
        chi-squared Wald test by default), False for the t distribution with
        `df_t` degrees of freedom. GMM and maximum-likelihood estimators set
        it True.
    `_summary_lines()`
        The report's lines between its title and the coefficient table.
    `_summary_fields()`
        The model-level fields of `summary_dict`, in their order.
    `_pyfixest_attrs()`
        The model-specific attributes of the pyfixest view (`to_pyfixest`):
        fixed effects, the estimation method, fit statistics, IV.

    A dynamic panel GMM result, say, would add the number of panel units and
    instruments and the Hansen and Arellano-Bond tests, and would name its
    vcovs (``"onestep"``, ``"twostep"``) in `all_vcovs`, overriding
    `with_vcov` if they need another vocabulary than `vcov_key`'s; an
    ordered or multinomial logit result would add the log-likelihood, the
    pseudo R-squared and the outcome's levels.

    Attributes
    ----------
    fml : str
        The model's formula, or another one-line description of the model.
    depvar : str
        The dependent variable.
    coefnames : list of str
        The names of the estimates.
    beta : numpy.ndarray
        The estimates, in the order of `coefnames`.
    vcov : numpy.ndarray
        Their variance-covariance matrix.
    vcov_type : str
        Which vcov that is, as a key of `all_vcovs`.
    df_t : int
        Degrees of freedom of the t distribution for p-values.
    n_obs : int
        Observations used.
    all_vcovs : dict
        ``{vcov_type: (vcov, df_t)}`` for every vcov computed, for
        `with_vcov`.
    n_clusters : dict
        The number of clusters of each clustering computed.
    collin_vars : list of str
        Variables dropped as collinear.
    diagnostics : dict
        Anything else the fit recorded.
    weights : str or None
        The weights column, if any.
    weights_type : str
        ``"aweights"`` or ``"fweights"``.
    """

    fml: str
    depvar: str
    coefnames: list
    beta: np.ndarray
    vcov: np.ndarray
    vcov_type: str
    df_t: int
    n_obs: int
    all_vcovs: dict = field(default_factory=dict)
    n_clusters: dict = field(default_factory=dict)
    collin_vars: list = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)
    weights: str | None = None
    weights_type: str = "aweights"

    # not dataclass fields: class-level defaults that subclasses override
    is_iv = False                       # the pyfixest view is a Feiv
    _normal_inference = False

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
        p = 2 * (special.ndtr(-np.abs(t)) if self._normal_inference else special.stdtr(self.df_t, -np.abs(t)))
        return pl.DataFrame({
            "Coefficient": self.coefnames,
            "Estimate": self.beta,
            "Std. Error": se,
            "t value": t,
            "Pr(>|t|)": p,
        }, schema_overrides={"Coefficient": pl.Utf8})

    def confint(self, alpha: float = 0.05, keep: str | list | None = None,
                drop: str | list | None = None, exact_match: bool = False,
                joint: bool = False, reps: int = 10_000,
                seed: int | None = None) -> pl.DataFrame:
        """Confidence intervals, one coefficient at a time or simultaneous.

        Pointwise intervals use the t distribution with `df_t` degrees of
        freedom, or the normal for GLMs, as `tidy` does. With ``joint=True``,
        the intervals cover all the selected coefficients at once with
        probability 1 - alpha: the critical value is the 1 - alpha quantile of
        the largest absolute t statistic, simulated from the normal with the
        coefficients' correlation (the multiplier bootstrap of pyfixest's
        ``confint(joint=True)``; Chernozhukov et al., *Applied Causal
        Inference Powered by ML and AI*, remark 4.4.1).

        Parameters
        ----------
        alpha : float, default 0.05
            One minus the coverage.
        keep, drop : str or list of str, optional
            Regular expressions selecting coefficients by name, as in
            pyfixest: those matching a `keep` pattern, in the order of the
            patterns (all, by default), less those matching a `drop`
            pattern. A joint interval covers exactly these.
        exact_match : bool, default False
            Match names exactly instead of as regular expressions.
        joint : bool, default False
            Simultaneous intervals.
        reps : int, default 10000
            Draws for the joint critical value.
        seed : int, optional
            Seed for those draws.

        Returns
        -------
        polars.DataFrame
            ``Coefficient`` and the lower and upper bounds, in columns named
            by their percentiles (``"2.5%"``, ``"97.5%"``).
        """
        idx = _select_coefs(self.coefnames, keep, drop, exact_match)
        se = self.se[idx]
        if not joint:
            crit = (special.ndtri(1 - alpha / 2) if self._normal_inference
                    else special.stdtrit(self.df_t, 1 - alpha / 2))
        else:
            corr = self.vcov[np.ix_(idx, idx)] / np.outer(se, se)
            vals, vecs = np.linalg.eigh(corr)
            root = (vecs * np.sqrt(np.clip(vals, 0, None))) @ vecs.T
            draws = root @ np.random.default_rng(seed).normal(size=(len(idx), reps))
            crit = float(np.quantile(np.abs(draws).max(axis=0), 1 - alpha))
        beta = self.beta[idx]
        return pl.DataFrame({
            "Coefficient": [self.coefnames[i] for i in idx],
            f"{alpha / 2 * 100:.1f}%": beta - crit * se,
            f"{(1 - alpha / 2) * 100:.1f}%": beta + crit * se,
        }, schema_overrides={"Coefficient": pl.Utf8})

    def wald_test(self, R: Any = None, q: Any = None, distribution: str | None = None,
                  keep: str | list | None = None, drop: str | list | None = None,
                  exact_match: bool = False) -> dict:
        """Wald test of the linear hypothesis R beta = q, under the current
        vcov (see `with_vcov`).

        By default the hypothesis is that every coefficient is zero; `keep`
        and `drop` narrow it to some of them (the coefficients of an event
        study's leads, say), and `R` and `q` state any other.

        The statistic is W = (R b - q)' (R V R')^+ (R b - q), and F = W / m for
        m restrictions. F is referred to the F distribution with m and `df_t`
        degrees of freedom, so that for one coefficient the p-value is the
        one `tidy` reports; W, to the chi-squared with m. Two differences from
        pyfixest's `wald_test`: it reports chi-squared whenever R is not the
        identity or q is not zero, where this test reports what is asked for;
        and its F's denominator degrees of freedom, without clustering,
        subtract every fixed-effect level, including the redundant ones that
        `df_t` does not.

        Parameters
        ----------
        R : array-like, optional
            m x k, one row per restriction, one column per coefficient in the
            order of `coefnames`; a 1-d array is one restriction.
        q : float or array-like, optional
            The m right-hand sides; zero by default.
        distribution : {"F", "chi2"}, optional
            Default "F" for OLS and "chi2" for GLMs, whose `tidy` p-values are
            normal.
        keep, drop, exact_match
            Without `R`: test that the coefficients they select are zero (or
            `q`), as in `confint`.

        Returns
        -------
        dict
            ``statistic`` (F or W), ``pvalue``, ``df1`` (m), ``df2`` (`df_t`
            for F, None for chi-squared) and ``distribution``.
        """
        k = len(self.beta)
        if R is None:
            idx = _select_coefs(self.coefnames, keep, drop, exact_match)
            R = np.eye(k)[idx]
        else:
            if keep is not None or drop is not None:
                raise ValueError("pass either R or keep/drop, not both")
            R = np.atleast_2d(np.asarray(R, dtype=float))
            if R.ndim != 2 or R.shape[1] != k:
                raise ValueError(f"R must have {k} columns, one per coefficient "
                                 f"({self.coefnames}); got shape {np.shape(R)}")
        m = R.shape[0]
        q = np.zeros(m) if q is None else np.asarray(q, dtype=float)
        if q.ndim == 0:
            q = np.full(m, float(q))
        if q.shape != (m,):
            raise ValueError(f"q must be a scalar or have {m} elements, one per row of R")
        distribution = distribution or ("chi2" if self._normal_inference else "F")
        d = R @ self.beta - q
        W = float(d @ np.linalg.pinv(R @ self.vcov @ R.T) @ d)
        if distribution == "F":
            F = W / m
            return {"statistic": F, "pvalue": float(special.fdtrc(m, self.df_t, F)),
                    "df1": m, "df2": self.df_t, "distribution": "F"}
        if distribution == "chi2":
            return {"statistic": W, "pvalue": float(special.chdtrc(m, W)),
                    "df1": m, "df2": None, "distribution": "chi2"}
        raise ValueError(f"distribution must be 'F' or 'chi2', got {distribution!r}")

    def with_vcov(self, vcov: Vcov) -> "RegressionResult":
        """A copy of this result with another of the vcovs computed at fit
        time: ``"iid"``, ``"hetero"``, or a clustering given as
        ``{"CRV1": var}`` or ``"CRV1:var"`` (likewise CRV3, when it was the
        fit's vcov). Ask for clusterings at fit time with ``vcov=`` or
        ``cluster=``."""
        import copy
        key = vcov_key(vcov)
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
        (recomputing the vcov, the wild bootstrap) are not, and ``predict``
        is this result's own `predict`."""
        from .reporting import _to_pyfixest  # reporting imports this module
        return _to_pyfixest(self)

    def cleanup(self) -> None:
        """Delete any files the result holds (none here)."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.cleanup()
        return False

    def summary_text(self) -> str:
        """The summary() report as a string."""
        return "\n".join([f"### {self.fml}", *self._summary_lines(), str(self.tidy())])

    def summary_dict(self, coefficients: bool = False) -> dict:
        """The model-level information of summary() as a dict of plain Python
        types (JSON-ready; nan is None). Fit statistics not defined for the
        model (e.g. r2 for a GLM) are absent. `coefficients=True` adds the
        `tidy()` table as a list of row dicts."""
        d = self._summary_fields()
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

    # ------------------------------------------------------------ hooks
    def _summary_lines(self) -> list[str]:
        """The report's lines between its title and the coefficient table."""
        lines = [f"vcov: {self.vcov_type}", f"obs: {self.n_obs:,}"]
        if self.weights:
            lines.append(f"weights: {self.weights} ({self.weights_type})")
        if self.collin_vars:
            lines.append(f"dropped as collinear: {self.collin_vars}")
        return lines

    def _summary_fields(self) -> dict:
        """The model-level fields of `summary_dict`, in their order."""
        return {
            "formula": self.fml,
            "depvar": self.depvar,
            "vcov": self.vcov_type,
            "df_t": self.df_t,
            "n_obs": self.n_obs,
            "n_clusters": dict(self.n_clusters),
            "weights": self.weights,
            "weights_type": self.weights_type if self.weights else None,
            "collinear_dropped": list(self.collin_vars),
        }

    def _pyfixest_attrs(self) -> dict:
        """The pyfixest view's model-specific attributes (see
        `reporting._fill_pyfixest`, which sets the rest)."""
        return {
            "_fixef": None, "_has_fixef": False,
            "_r2": np.nan, "_adj_r2": np.nan, "_r2_adj": np.nan,
            "_r2_within": np.nan, "_adj_r2_within": np.nan, "_rmse": np.nan,
            "deviance": None, "_method": "feols", "_loglik": None, "_pseudo_r2": None,
            "_is_iv": self.is_iv, "_f_stat_1st_stage": None,
        }


class RegressionMulti:
    """Results of several models, fitted together: iterate over it, or take
    one model with `fetch_model`; ``len()`` counts them. The models are in
    the order fitted, and keyed by formula in ``all_fitted_models``.
    """

    def __init__(self, results: Iterable[RegressionResult]) -> None:
        self.all_fitted_models = {r.fml: r for r in results}

    def fetch_model(self, i: int) -> RegressionResult:
        """The i-th model, in the formula's order."""
        return list(self.all_fitted_models.values())[i]

    def to_pyfixest(self) -> list:
        """A pyfixest view of each model (see `HDFEResult.to_pyfixest`)."""
        return [r.to_pyfixest() for r in self]

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


def _select_coefs(names, keep, drop, exact_match):
    """Indices of the coefficients `keep` and `drop` select, in pyfixest's
    way: those matching a keep pattern, in the order of the patterns (all,
    without any), less those matching a drop pattern."""
    import re

    def matches(pattern, name):
        return name == pattern if exact_match else re.search(pattern, name) is not None

    keep = [keep] if isinstance(keep, str) else list(keep or [])
    drop = [drop] if isinstance(drop, str) else list(drop or [])
    order = ([i for p in keep for i, nm in enumerate(names) if matches(p, nm)]
             if keep else list(range(len(names))))
    idx = [i for i in dict.fromkeys(order)
           if not any(matches(p, names[i]) for p in drop)]
    if not idx:
        raise ValueError(f"no coefficient matches keep={keep} and drop={drop}; the "
                         f"coefficients are {names}")
    return idx


def _canon_cluster(var):
    """'worker_id + firm_id' -> 'worker_id+firm_id'; 'firm_id ^ year' -> 'firm_id^year'."""
    return "+".join("^".join(c.strip() for c in part.split("^")) for part in var.split("+"))


def vcov_key(v):
    """The canonical key of a vcov specification: ``"iid"``, ``"hetero"``,
    ``"CRV1:<var>"`` (multi-way ``"CRV1:a+b"``) or ``"CRV3:<var>"``, from
    any of the spellings the estimators accept (``"HC1"``, ``"CRV1:var"``,
    ``{"CRV1": "var"}``, ...); None for None. Raises ValueError for anything
    else. The keys index `RegressionResult.all_vcovs`."""
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

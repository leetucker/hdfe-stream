"""Reporting through pyfixest: `pf.etable`, `pf.summary`, `pf.coefplot`.

Output side of the pyfixest interface, and entirely optional -- nothing here
is needed to fit a model. Contrast `formula.py`, which is on the input path,
is required by `feols_stream`, and leans on pyfixest internals.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import Any

import numpy as np
from scipy import special

from .feterms import interactions_to_caret
from .results_base import RegressionMulti, RegressionResult


# --------------------------------------------------------------------------
# pyfixest adapter (pf.etable, pf.summary, pf.coefplot, ...)
#
# pyfixest's reporting functions accept Feols/Feiv instances, and maketables
# reads a documented set of attributes plus tidy(). The adapter subclasses
# Feols/Feiv without running their constructors and fills those attributes
# from an HDFEResult. Anything that needs the data itself is unavailable.
#
# The attributes live in one dict, `_stream_attrs`, and the adapter answers
# for them in `__getattribute__`. pyfixest 0.70 replaced these private
# attributes with typed components (`fit.coeftable`, `fit.sample_info`, ...)
# and keeps the old names as deprecated descriptors that read the typed state,
# which an adapter built without a fit does not have. Answering first keeps
# the old names working there, and is the same code on pyfixest 0.60.
# --------------------------------------------------------------------------

_ADAPTER_CLASSES = {}


def _adapter_classes():
    if not _ADAPTER_CLASSES:
        import pandas as pd
        from pyfixest.estimation import Feiv, Feols

        class _StreamMixin:
            def __init__(self, res):            # no Feols.__init__: no data here
                _fill_pyfixest(self, res)

            def __getattribute__(self, name):
                attrs = object.__getattribute__(self, "__dict__").get("_stream_attrs")
                if attrs is not None and name in attrs:
                    return attrs[name]
                return super().__getattribute__(name)

            def __setattr__(self, name, value):
                attrs = self.__dict__.get("_stream_attrs")
                if attrs is not None and name in attrs:
                    # copy on write: `copy.copy(adapter)` shares the dict
                    self.__dict__["_stream_attrs"] = {**attrs, name: value}
                else:
                    super().__setattr__(name, value)

            def _index(self):
                return pd.Index(self._coefnames, name="Coefficient")

            def coef(self):
                return pd.Series(self._beta_hat, index=self._index(), name="Estimate")

            def se(self):
                return pd.Series(self._se, index=self._index(), name="Std. Error")

            def tstat(self):
                return pd.Series(self._tstat, index=self._index(), name="t value")

            def pvalue(self):
                return pd.Series(self._pvalue, index=self._index(), name="Pr(>|t|)")

            def confint(self, alpha=0.05, **_):
                crit = _crit(self, 1 - alpha / 2)
                lo, hi = self._beta_hat - crit * self._se, self._beta_hat + crit * self._se
                return pd.DataFrame({f"{alpha / 2 * 100:.1f}%": lo,
                                     f"{(1 - alpha / 2) * 100:.1f}%": hi}, index=self._index())

            def tidy(self, alpha=0.05, **_):
                ci = self.confint(alpha)
                return pd.DataFrame({"Estimate": self._beta_hat, "Std. Error": self._se,
                                     "t value": self._tstat, "Pr(>|t|)": self._pvalue,
                                     ci.columns[0]: ci.iloc[:, 0].to_numpy(),
                                     ci.columns[1]: ci.iloc[:, 1].to_numpy()},
                                    index=self._index())

            def vcov(self, *args, **kwargs):
                raise NotImplementedError(
                    "streaming results cannot recompute the vcov; use "
                    "HDFEResult.with_vcov() and convert again")

        class StreamFeols(_StreamMixin, Feols):
            pass

        class StreamFeiv(_StreamMixin, Feiv):
            pass

        _ADAPTER_CLASSES.update(ols=StreamFeols, iv=StreamFeiv)
    return _ADAPTER_CLASSES["ols"], _ADAPTER_CLASSES["iv"]


def _crit(obj, q):
    """Quantile for confidence intervals: normal when the result's inference
    is (GLMs, as pyfixest has it), t otherwise."""
    if obj._stream_result._normal_inference:
        return special.ndtri(q)
    return special.stdtrit(obj._df_t, q)


def _has_typed_components():
    """True for pyfixest 0.70 and later, whose fitted models carry typed
    components (`fit.coeftable`, ...) beside the old private attributes."""
    from importlib.util import find_spec
    return find_spec("pyfixest.estimation.models._deprecated_attributes") is not None


def _fill_pyfixest(obj, r):
    """The attributes pyfixest's reporting reads, from a `RegressionResult`:
    the coefficient table and vcov here, the rest from its `_pyfixest_attrs`."""
    se = r.se
    t = np.divide(r.beta, se, out=np.full_like(r.beta, np.nan), where=se > 0)
    kind = r.vcov_type
    if kind.startswith(("CRV1:", "CRV3:")):
        detail, cv = kind.split(":", 1)
        Gs = list(r.n_clusters[cv])
        # pyfixest reports min(G) for every term of a multi-way vcov
        vtype, clustervar, G = ("CRV", cv.split("+"),
                                Gs if len(Gs) == 1 else [min(Gs)] * len(Gs))
    else:
        vtype, detail, clustervar, G = kind, kind, None, None
    normal = r._normal_inference
    crit = special.ndtri(0.975) if normal else special.stdtrit(r.df_t, 0.975)
    obj.__dict__["_stream_attrs"] = {
        "_fml": r.fml, "_depvar": r.depvar,
        "_coefnames": list(r.coefnames), "_k": len(r.coefnames),
        "_beta_hat": np.asarray(r.beta), "_se": se, "_tstat": t,
        "_pvalue": 2 * (special.ndtr(-np.abs(t)) if normal
                        else special.stdtr(r.df_t, -np.abs(t))),
        "_conf_int": np.vstack([r.beta - crit * se, r.beta + crit * se]),
        "_vcov": r.vcov, "_vcov_type": vtype, "_vcov_type_detail": detail,
        "_clustervar": clustervar, "_G": G, "_df_t": r.df_t,
        "_N": int(r.n_obs) if float(r.n_obs).is_integer() else r.n_obs,
        "_F_stat": None,
        "_use_mundlak": False, "_sample_split_var": None, "_sample_split_value": "all",
        "_quantile": None, "_data": None, "_weights_name": r.weights,
        "_weights_type": r.weights_type, "_collin_vars": list(r.collin_vars),
        "_model_name": r.fml, "_model_name_plot": r.fml,
        "_icovars": [c for c in r.coefnames if "::" in c] or None,     # for pf.iplot
        **r._pyfixest_attrs(),
        "_stream_result": r,
    }
    if _has_typed_components():
        obj.__dict__["_stream_attrs"].update(_typed_components(obj.__dict__["_stream_attrs"], r, crit))


def _typed_components(attrs, r, crit):
    """Stand-ins for the typed components pyfixest 0.70 reads in place of the
    private attributes (`fit.model`, `fit.sample_info`, ...): only what its
    reporting functions (`summary`, `iplot`, `coefplot`) look at, built from
    the attributes already set."""
    se = attrs["_se"]
    t = attrs["_tstat"]
    fixef = attrs.get("_fixef")
    icov = attrs["_icovars"]
    return {
        "model": SimpleNamespace(
            depvar=attrs["_depvar"], formula=attrs["_fml"], model_name=attrs["_model_name"],
            method=attrs["_method"], is_iv=attrs["_is_iv"], has_fixef=bool(attrs["_has_fixef"]),
            fixef=fixef, sample_split=None, interacted_covariates=icov),
        "sample_info": SimpleNamespace(n_obs=attrs["_N"]),
        "variance_covariance": SimpleNamespace(
            vcov=attrs["_vcov"], df_t=attrs["_df_t"],
            spec=SimpleNamespace(vcov_type=attrs["_vcov_type"],
                                 vcov_type_detail=attrs["_vcov_type_detail"])),
        "fitstat": SimpleNamespace(
            rmse=attrs["_rmse"], r2=attrs["_r2"], r2_within=attrs["_r2_within"],
            adj_r2=attrs["_adj_r2"], deviance=_nan_if_none(attrs.get("deviance"))),
        "coeftable": SimpleNamespace(se=se, tstat=t, pvalue=attrs["_pvalue"]),
    }


def _nan_if_none(x):
    return np.nan if x is None else x


def _to_pyfixest(r):
    ols, iv = _adapter_classes()
    return (iv if r.is_iv else ols)(r)


def _fixef_string(text: str) -> str:
    """'a + b ^ c' -> 'a+b^c' (and 'b:c' -> 'b^c')."""
    text = interactions_to_caret(text)
    return "+".join("^".join(c.strip() for c in part.split("^")) for part in text.split("+"))


_TEX_SPECIAL = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "#": r"\#",
                "_": r"\_", "^": r"\^{}"}


def _tex_escape(text: str) -> str:
    return "".join(_TEX_SPECIAL.get(c, c) for c in text)


def _tex_labels(models: list, labels: dict | None, felabels: dict | None) -> tuple[dict, dict]:
    """LaTeX-escaped labels for every variable and fixed-effect name in
    `models`, under the caller's own. maketables writes names into LaTeX
    unescaped, so an underscore would set the next character as a subscript."""
    names, fes = set(), set()
    for m in models:
        names.update(getattr(m, "_coefnames", None) or [])
        if getattr(m, "_depvar", None):
            names.add(m._depvar)
        fx = getattr(m, "_fixef", None)
        if fx and fx != "0":
            fes.update(x for x in fx.split("+") if x)
    auto = {n: _tex_escape(n) for n in names | fes if _tex_escape(n) != n}
    fe_auto = {n: _tex_escape(n) for n in fes if _tex_escape(n) != n}
    return {**auto, **(labels or {})}, {**fe_auto, **(felabels or {})}


def etable(models: Any, **kwargs: Any) -> Any:
    """``pf.etable`` for streaming results: a regression table of
    `HDFEResult` and `HDFEMulti` models, mixed freely with ordinary pyfixest
    models.

    A multi-model result expands into one column per model. The spacing of
    the fixed-effect side is normalized, so ``"a + b"`` and ``"a+b"`` share a
    row (``pf.etable`` itself matches rows on the literal text). With
    ``type="tex"``, labels are escaped for LaTeX.

    Parameters
    ----------
    models : HDFEResult, HDFEMulti, or a list of them and pyfixest models
        The models, one column each.
    **kwargs
        Passed to ``pf.etable``: ``type``, ``model_stats``, ``labels``,
        ``felabels``, ``file_name``, ...

    Returns
    -------
    object
        What ``pf.etable`` returns for the ``type`` asked for.

    Examples
    --------
    >>> etable([fit1, fit2], type="df", model_stats=["N", "se_type", "r2"])
    """
    import pyfixest as pf
    if isinstance(models, (RegressionResult, RegressionMulti)):
        models = [models]
    out = []
    for m in models:
        if isinstance(m, RegressionMulti):
            out += m.to_pyfixest()
        elif isinstance(m, RegressionResult):
            out.append(m.to_pyfixest())
        else:
            out.append(m)
    # the same spelling of fixed effects across models, so their rows merge
    for k, m in enumerate(out):
        fx = getattr(m, "_fixef", None)
        if isinstance(fx, str) and fx != _fixef_string(fx):
            out[k] = copy.copy(m)
            out[k]._fixef = _fixef_string(fx)
    if kwargs.get("type") == "tex":
        kwargs["labels"], kwargs["felabels"] = _tex_labels(
            out, kwargs.get("labels"), kwargs.get("felabels"))
    return pf.etable(out, **kwargs)

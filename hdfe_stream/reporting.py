"""Reporting through pyfixest: `pf.etable`, `pf.summary`, `pf.coefplot`.

Output side of the pyfixest interface, and entirely optional -- nothing here
is needed to fit a model. Contrast `formula.py`, which is on the input path,
is required by `feols_stream`, and leans on pyfixest internals.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
from scipy import stats

from .results import HDFEMulti, HDFEResult


# --------------------------------------------------------------------------
# pyfixest adapter (pf.etable, pf.summary, pf.coefplot, ...)
#
# pyfixest's reporting functions accept Feols/Feiv instances, and maketables
# reads a documented set of attributes plus tidy(). The adapter subclasses
# Feols/Feiv without running their constructors and fills those attributes
# from an HDFEResult. Anything that needs the data itself is unavailable.
# --------------------------------------------------------------------------

_ADAPTER_CLASSES = {}


def _adapter_classes():
    if not _ADAPTER_CLASSES:
        import pandas as pd
        from pyfixest.estimation.models.feiv_ import Feiv
        from pyfixest.estimation.models.feols_ import Feols

        class _StreamMixin:
            def __init__(self, res):            # no Feols.__init__: no data here
                _fill_pyfixest(self, res)

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

            def tidy(self, alpha=0.05):
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
    """Quantile for confidence intervals: normal for GLMs, t otherwise, as
    pyfixest has it."""
    if obj._method != "feols":
        return stats.norm.ppf(q)
    return stats.t.ppf(q, obj._df_t)


def _fill_pyfixest(obj, r):
    from .families import get_family
    method = get_family(r.family).method if r.family else "feols"
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
    glm = method != "feols"
    crit = stats.norm.ppf(0.975) if glm else stats.t.ppf(0.975, r.df_t)
    obj.__dict__.update({
        "_fml": r.fml, "_depvar": r.depvar,
        # the FE string as written in the formula, without spaces: maketables
        # matches FE rows across models by the text between '+' signs, spaces
        # included, so "a + b + c" and "a + b" would not share their "b" row
        "_fixef": (_fixef_string(r.fml.split("|")[1] if "|" in r.fml
                                 else " + ".join(r.fe_names)) if r.fe_names else None),
        "_has_fixef": bool(r.fe_names), "_coefnames": list(r.coefnames), "_k": len(r.coefnames),
        "_beta_hat": np.asarray(r.beta), "_se": se, "_tstat": t,
        "_pvalue": 2 * (stats.norm.sf(np.abs(t)) if glm else stats.t.sf(np.abs(t), r.df_t)),
        "_conf_int": np.vstack([r.beta - crit * se, r.beta + crit * se]),
        "_vcov": r.vcov, "_vcov_type": vtype, "_vcov_type_detail": detail,
        "_clustervar": clustervar, "_G": G, "_df_t": r.df_t,
        "_N": int(r.n_obs) if float(r.n_obs).is_integer() else r.n_obs,
        "_r2": r.r2, "_adj_r2": r.adj_r2, "_r2_adj": r.adj_r2,
        "_r2_within": r.r2_within, "_adj_r2_within": r.adj_r2_within, "_rmse": r.rmse,
        "_F_stat": None, "deviance": r.deviance if glm else None, "_method": method,
        "_loglik": r.loglik if glm else None, "_pseudo_r2": r.pseudo_r2 if glm else None,
        "_is_iv": r.is_iv,
        "_f_stat_1st_stage": (r.f_stat_1st_stage[0] if r.is_iv and len(r.f_stat_1st_stage) == 1
                              else None),
        "_use_mundlak": False, "_sample_split_var": None, "_sample_split_value": "all",
        "_quantile": None, "_data": None, "_weights_name": r.weights,
        "_weights_type": r.weights_type, "_collin_vars": list(r.collin_vars),
        "_model_name": r.fml, "_model_name_plot": r.fml,
        "_icovars": [c for c in r.coefnames if "::" in c] or None,     # for pf.iplot
        "_stream_result": r,
    })


def _to_pyfixest(r):
    ols, iv = _adapter_classes()
    return (iv if r.is_iv else ols)(r)


def _fixef_string(text: str) -> str:
    """'a + b ^ c' -> 'a+b^c'."""
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
    """pf.etable for streaming results (HDFEResult, HDFEMulti, or a list that
    may mix them with ordinary pyfixest models). kwargs go to pf.etable."""
    import pyfixest as pf
    if isinstance(models, (HDFEResult, HDFEMulti)):
        models = [models]
    out = []
    for m in models:
        if isinstance(m, HDFEMulti):
            out += m.to_pyfixest()
        elif isinstance(m, HDFEResult):
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

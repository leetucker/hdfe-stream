"""Fit results: `HDFEResult` for one model, `HDFEMulti` for a formula that
expands to several.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats

from .report import _emit


# --------------------------------------------------------------------------
# results
# --------------------------------------------------------------------------

@dataclass
class HDFEResult:
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
    _run: object = field(default=None, repr=False, compare=False)

    @property
    def se(self):
        return np.sqrt(np.diag(self.vcov))

    def coef(self) -> dict:
        return dict(zip(self.coefnames, self.beta))

    def tidy(self) -> pl.DataFrame:
        se = self.se
        t = np.divide(self.beta, se, out=np.full_like(self.beta, np.nan), where=se > 0)
        p = 2 * stats.t.sf(np.abs(t), self.df_t)
        return pl.DataFrame({
            "Coefficient": self.coefnames,
            "Estimate": self.beta,
            "Std. Error": se,
            "t value": t,
            "Pr(>|t|)": p,
        }, schema_overrides={"Coefficient": pl.Utf8})

    def with_vcov(self, vcov) -> "HDFEResult":
        """Switch among the vcovs computed in pass 2: 'iid', 'hetero', or a
        cluster variable given as {'CRV1': var} or 'CRV1:var'."""
        import copy
        key = _vcov_key(vcov)
        if key not in self.all_vcovs:
            raise KeyError(f"{key} was not computed; available: {list(self.all_vcovs)}. "
                           "Request it at fit time with vcov= or cluster=.")
        out = copy.copy(self)
        v, dft = self.all_vcovs[key]
        out.vcov, out.vcov_type, out.df_t = v, key, dft
        return out

    def to_pyfixest(self):
        """A pyfixest Feols (Feiv for IV) view of this result, for
        pf.etable, pf.summary, pf.coefplot, ... Only the stored results are
        available; methods that need the data (predict, re-computing vcov,
        wild bootstrap) are not."""
        from .reporting import _to_pyfixest   # reporting imports this module
        return _to_pyfixest(self)

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
        """Lazy scan of one dimension's estimated fixed effects. The file lives
        as long as this result object (outputs='auto'): collect or sink what
        you need before dropping the result."""
        return self._scan(self.paths["fe"][name], f"fixed effects for {name!r}")

    def resid(self) -> pl.LazyFrame:
        """Lazy scan of row-level output (FE contributions + residuals); same
        lifetime caveat as fixef()."""
        return self._scan(self.paths["resid"], "residual")

    @property
    def files_dir(self):
        """Directory holding this model's result files."""
        return self.paths.get("dir")

    def cleanup(self):
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
        lines = [f"### {self.fml}", f"Streaming HDFE   vcov: {self.vcov_type}",
                 "obs: {:,}   ".format(self.n_obs)
                 + "   ".join(f"{k}: {v:,}" for k, v in self.n_levels.items())]
        st = self.diagnostics.get("stream")
        if st:
            lines.append(f"streamed dimension: {st['dim']} ({st['reason']})")
        lines.append(f"identifying {self.fe_names[0]} groups: {self.n_identifying:,}   "
                     f"({self.fe_names[0]} x {self.fe_names[1]}) components: "
                     f"{self.n_components:,}")
        lines.append(f"FE dof: {self.k_fe:,}   RSS: {self.rss:.6g}   RMSE: {self.rmse:.4g}   "
                     f"R2: {self.r2:.6f}   within R2: {self.r2_within:.6f}")
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

    def summary(self, logger=None, level=logging.INFO):
        """Print the report, or write it to `logger` (one record, at `level`)."""
        _emit(self.summary_text(), logger, level)

class HDFEMulti:
    """Results of a formula that expands to several models."""

    def __init__(self, results):
        self.all_fitted_models = {r.fml: r for r in results}

    def fetch_model(self, i) -> HDFEResult:
        return list(self.all_fitted_models.values())[i]

    def to_pyfixest(self) -> list:
        return [r.to_pyfixest() for r in self]

    def cleanup(self):
        """Delete the result files of all models."""
        for r in self:
            r.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.cleanup()
        return False

    def tidy(self) -> pl.DataFrame:
        return pl.concat([r.tidy().with_columns(pl.lit(f).alias("fml"))
                          for f, r in self.all_fitted_models.items()]).select("fml", pl.all().exclude("fml"))

    def summary_text(self) -> str:
        return "\n\n".join(r.summary_text() for r in self.all_fitted_models.values())

    def summary(self, logger=None, level=logging.INFO, per_model=False):
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
        if kind.upper() != "CRV1":
            raise ValueError(f"only CRV1 clustering is supported, got {kind}")
        return f"CRV1:{_canon_cluster(var)}"
    v = str(v)
    if v.lower() in ("iid",):
        return "iid"
    if v.lower() in ("hetero", "hc1"):
        return "hetero"
    if v.upper().startswith("CRV1:"):
        return "CRV1:" + _canon_cluster(v.split(":", 1)[1])
    raise ValueError(f"unsupported vcov {v!r}: use 'iid', 'hetero'/'HC1' or {{'CRV1': var}} "
                     "(multi-way: {'CRV1': 'a+b'})")

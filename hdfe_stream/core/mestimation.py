"""Influence functions that account for an estimated first stage: a binary
logit or probit (a propensity score).

A second-stage estimate theta(gamma) that uses the fitted score, with
per-row contributions phi_i (theta_hat - theta ~= sum phi_i at the true
gamma), is, to first order,

    theta_hat - theta ~= sum_i phi_i + J (gamma_hat - gamma)
    gamma_hat - gamma ~= H^-1 sum_i s_i

with J = d theta / d gamma, s_i the first stage's score contributions and
H = -sum_i d s_i / d gamma its observed information. The corrected
contributions phi_i + J H^-1 s_i go to `if_vcov` (clustered as needed):
this is the stacked M-estimation (just-identified GMM) sandwich, as in
Stata's ``teffects ipw`` or R's ``WeightIt`` M-estimation, written as an
influence function so that it streams and clusters like any other.

For a binary model with P(D = 1) = F(eta), eta = x' gamma:

    s_i = (D_i - F_i) g_i x_i,     g = f / (F (1 - F))
    H   = sum_i [f_i g_i - (D_i - F_i) g'_i] x_i x_i'

(for the logit g = 1 and H = sum F (1 - F) x x', observed = expected; for
the probit they differ, and the observed one is what the stacked sandwich
uses). Every sum is one streaming pass over Polars expressions.

Absorbed fixed effects (one or several dimensions) are first-stage
parameters too, possibly millions of them, so H cannot be formed. Every
derivative needed here has the form J = X~' c, X~ the full design (the
covariates and every fixed-effect dummy) and c one number per row
(c_i = d theta / d eta_i). Then

    H^-1 J = argmin_a  sum_i w_i (c_i / w_i - x~_i' a)^2,

a weighted least-squares regression of c / w on the covariates with the
fixed effects absorbed, weights w the information weights (the
fixed-effect regression hdfe-stream's `feols_stream` solves out of core).
So s_i' H^-1 J = r_i * fitted_i, with s_i = r_i x~_i, and the quadratic form
J' H^-1 J = sum_i w_i fitted_i^2 (`project`). The weights are the rows'
observed information, which is positive for every row (the logit and probit
log-likelihoods are concave in eta row by row), so this is exactly the
observed-information correction the dense path computes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl
from scipy import special

LINKS = ("logit", "probit")


def _np_cdf_pdf(link: str, eta: np.ndarray):
    if link == "logit":
        F = special.expit(eta)
        return F, F * (1 - F)
    return special.ndtr(eta), np.exp(-0.5 * eta * eta) / np.sqrt(2 * np.pi)


@dataclass
class BinaryFirstStage:
    """A fitted binary-choice first stage: covariate expressions `x` (with
    the intercept as ``pl.lit(1.0)`` if there is one), coefficients `beta`,
    the `link` ("logit" or "probit") and the outcome column `outcome`.
    With absorbed fixed effects, `fe` names them (formula terms such as
    ``"firm"`` or ``"firm^year"``), `fit` is the hdfe-stream result (it
    predicts eta, the fixed effects included) and `rhs` the covariate part
    of its formula (for `project`)."""

    names: list[str]
    x: list[pl.Expr]
    beta: np.ndarray
    link: str
    outcome: str
    fe: list[str] = field(default_factory=list)
    fit: object = None
    rhs: str = ""

    def __post_init__(self):
        if self.link not in LINKS:
            raise ValueError(f"link must be one of {LINKS}, not {self.link!r}")
        self.beta = np.asarray(self.beta, np.float64)
        if len(self.x) != self.beta.shape[0]:
            raise ValueError("one coefficient per covariate expression")
        if self.fe and self.fit is None:
            raise ValueError("a first stage with fixed effects needs its fit (to predict)")
        self._fits = []          # fixed-effect regressions behind project()'s columns

    @classmethod
    def from_fit(cls, fit, outcome: str) -> BinaryFirstStage:
        """From an hdfe-stream ``feglm_stream`` logit or probit result, with
        or without absorbed fixed effects (not varying slopes)."""
        link = str(fit.family).lower()
        est = fit._estimator
        fe = list(fit.fe_names)
        if fe and getattr(est, "slopes", None):
            raise NotImplementedError("the first-stage correction with varying slopes on "
                                      "fixed effects is not built")
        rhs = ""
        if fe:
            rhs = fit.fml.split("~", 1)[1].split("|", 1)[0].strip()
        return cls(names=list(fit.coefnames), x=[est.var_exprs[c] for c in fit.coefnames],
                   beta=np.asarray(fit.beta), link=link, outcome=outcome, fe=fe,
                   fit=fit if fe else None, rhs=rhs)

    # -------------------------------------------------------------- pieces
    def eta(self) -> pl.Expr:
        """The linear predictor: an expression without fixed effects; with
        them, the column __eta that `with_eta` adds."""
        if self.fe:
            return pl.col("__eta")
        return pl.sum_horizontal([float(b) * x.cast(pl.Float64) for b, x in
                                  zip(self.beta, self.x)])

    def with_eta(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """`lf` with the linear predictor in __eta (the fixed effects joined
        from the fit's files when there are any)."""
        lf = lf.lazy()
        if self.fe:
            return self.fit.predict(lf, type="link", name="__eta")
        return lf.with_columns(self.eta().alias("__eta"))

    def prob(self) -> pl.Expr:
        """F(eta), the fitted probability (with fixed effects, after `with_eta`)."""
        if self.link == "logit":
            return 1.0 / (1.0 + (-self.eta()).exp())
        return self.eta().map_batches(lambda s: pl.Series(special.ndtr(s.to_numpy())),
                                      return_dtype=pl.Float64, is_elementwise=True)

    def _parts(self) -> list[pl.Expr]:
        """Per row: F, f, g = f / (F (1 - F)) and g' (as columns __F, __f,
        __g, __gp), from eta."""
        link = self.link

        def parts(s: pl.Series) -> pl.Series:
            e = s.to_numpy()
            F, f = _np_cdf_pdf(link, e)
            if link == "logit":
                g = np.ones_like(e)
                gp = np.zeros_like(e)
            else:
                v = F * (1 - F)
                g = f / v
                # g' = [f' v - f v'] / v^2, f' = -eta f, v' = f (1 - 2F)
                gp = (-e * f * v - f * f * (1 - 2 * F)) / (v * v)
            return pl.DataFrame({"F": F, "f": f, "g": g, "gp": gp}).to_struct("p")

        st = self.eta().map_batches(parts, is_elementwise=True,
                                    return_dtype=pl.Struct({"F": pl.Float64, "f": pl.Float64,
                                                            "g": pl.Float64,
                                                            "gp": pl.Float64}))
        return [st.struct.field(k).alias(f"__{k}") for k in ("F", "f", "g", "gp")]

    def with_parts(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """`lf` with __eta, __F, __f, __g, __gp and the covariates as __x0, ..."""
        lf = lf.lazy()
        if self.fe and "__eta" not in lf.collect_schema():
            lf = self.with_eta(lf)
        return lf.with_columns(*self._parts(),
                               *[x.cast(pl.Float64).alias(f"__x{k}")
                                 for k, x in enumerate(self.x)])

    def score_exprs(self) -> list[pl.Expr]:
        """s_i, one expression per coefficient (needs `with_parts` columns)."""
        r = self.residual()
        return [r * pl.col(f"__x{k}") for k in range(len(self.x))]

    def residual(self) -> pl.Expr:
        """r_i with s_i = r_i x~_i: (D - F) g (needs `with_parts` columns)."""
        d = pl.col(self.outcome).cast(pl.Float64)
        return (d - pl.col("__F")) * pl.col("__g")

    def info_weight(self) -> pl.Expr:
        """Each row's observed-information weight, -d^2 loglik_i / d eta^2 =
        f g - (D - F) g' (the logit's is F (1 - F)). Positive for every row:
        the logit's and the probit's log-likelihoods are concave in eta row
        by row, so the weighted regression in `project` is well defined."""
        d = pl.col(self.outcome).cast(pl.Float64)
        return pl.col("__f") * pl.col("__g") - (d - pl.col("__F")) * pl.col("__gp")

    def information(self, lf: pl.LazyFrame) -> np.ndarray:
        """The observed information H (k x k), one streaming pass. Not
        available with fixed effects (use `project`)."""
        if self.fe:
            raise NotImplementedError("with fixed effects H is not formed; use project()")
        p = len(self.x)
        d = pl.col(self.outcome).cast(pl.Float64)
        a = pl.col("__f") * pl.col("__g") - (d - pl.col("__F")) * pl.col("__gp")
        aggs = [(a * pl.col(f"__x{i}") * pl.col(f"__x{j}")).sum().alias(f"h{i}_{j}")
                for i in range(p) for j in range(i, p)]
        vals = self.with_parts(lf).select(aggs).collect(engine="streaming").row(0, named=True)
        H = np.zeros((p, p))
        for i in range(p):
            for j in range(i, p):
                H[i, j] = H[j, i] = vals[f"h{i}_{j}"]
        return H

    def score_sums(self, lf: pl.LazyFrame) -> np.ndarray:
        """sum_i s_i over the covariates: about 0 at the maximum likelihood
        estimate (a check)."""
        sums = [e.sum().alias(f"s{k}") for k, e in enumerate(self.score_exprs())]
        return np.array(self.with_parts(lf).select(sums).collect(engine="streaming").row(0))

    def project(self, lf: pl.LazyFrame, c: dict[str, pl.Expr], *, workdir=None,
                weight: pl.Expr | None = None) -> pl.LazyFrame:
        """For each named per-row derivative c (J = X~' c), the column
        ``<name>_fit`` = x~_i' H^-1 J, by the weighted least-squares
        regression of c / w on the covariates with the fixed effects
        absorbed (`feols_stream`, out of core), w the observed-information
        weight (`info_weight`). `lf` must have the `with_parts` columns
        and the source's columns."""
        if not self.fe:
            raise NotImplementedError("project() is for a first stage with fixed effects; "
                                      "without them H is formed directly")
        from hdfe_stream import feols_stream

        if not self.rhs or self.rhs in ("1", "0"):
            raise NotImplementedError("a score model with fixed effects and no covariates")
        w = self.info_weight() if weight is None else weight
        lf = lf.lazy().with_columns(w.alias("__pw"))
        lf = lf.with_columns([(e / pl.col("__pw")).alias(f"__u_{k}") for k, e in c.items()])
        fe = " + ".join(self.fe)
        for k in c:
            fit = feols_stream(f"__u_{k} ~ {self.rhs} | {fe}", lf, workdir=workdir,
                               weights="__pw", fixef_rm="none", vcov="iid", verbose=False)
            lf = fit.predict(lf, type="link", name=f"{k}_fit")
            # the prediction reads the fit's fixed-effect files when collected:
            # keep the fit as long as this first stage
            self._fits.append(fit)
        return lf

    def quad(self, lf: pl.LazyFrame, c: dict[str, pl.Expr], *, workdir=None) -> dict:
        """J' H^-1 J for each named per-row derivative c (J = X~' c)."""
        if not self.fe:
            p = self.with_parts(lf)
            out = {}
            H = self.information(lf)
            for k, e in c.items():
                J = np.array(p.select([(e * pl.col(f"__x{j}")).sum().alias(f"j{j}")
                                       for j in range(len(self.x))])
                             .collect(engine="streaming").row(0))
                out[k] = float(J @ np.linalg.solve(H, J))
            return out
        p = self.project(self.with_parts(lf), c, workdir=workdir)
        sums = p.select([(pl.col("__pw") * pl.col(f"{k}_fit") ** 2).sum().alias(k)
                         for k in c]).collect(engine="streaming").row(0, named=True)
        return {k: float(v) for k, v in sums.items()}


def corrected_contributions(lf: pl.LazyFrame, phi: str, first: BinaryFirstStage,
                            J: np.ndarray | None = None, H: np.ndarray | None = None,
                            name: str = "__phi_c", *, c: pl.Expr | None = None,
                            workdir=None) -> pl.LazyFrame:
    """`lf` (which must hold the first stage's outcome and covariates) with
    column `name` = phi_i + J H^-1 s_i. Give `J` (d theta / d gamma, length
    k; without fixed effects) or `c`, the per-row derivative with
    J = X~' c (required with fixed effects). `H`: the observed information
    (computed when not given; no fixed effects)."""
    p = first.with_parts(lf)
    if first.fe:
        if c is None:
            raise ValueError("with fixed effects give c, the per-row derivative")
        p = first.project(p, {"__corr": c}, workdir=workdir)
        adj = first.residual() * pl.col("__corr_fit")
        return p.with_columns((pl.col(phi) + adj).alias(name))
    if J is None:
        if c is None:
            raise ValueError("give J or c")
        J = np.array(p.select([(c * pl.col(f"__x{j}")).sum().alias(f"j{j}")
                               for j in range(len(first.x))]).collect(engine="streaming").row(0))
    if H is None:
        H = first.information(lf)
    a = np.linalg.solve(H, np.asarray(J, np.float64))      # H^-1 J' (H symmetric)
    adj = pl.sum_horizontal([float(v) * s for v, s in zip(a, first.score_exprs())])
    return p.with_columns((pl.col(phi) + adj).alias(name))

"""Instrumental-variable diagnostics from sums: identification and
overidentification tests for any linear IV or GMM estimator whose moments
are sums over rows or clusters.

Everything here works on small matrices -- cross-products of the
instruments and regressors, and the (cluster-)robust covariances of their
scores -- never on data, so an estimator that streams its rows computes
those sums in a pass it makes anyway and hands them over. feols_stream's IV
fits use it, and so does dpd-stream for GMM.

The statistics, with ivreg2's conventions (Baum, Schaffer and Stillman):

- Underidentification: Kleibergen and Paap's (2006) rk LM test (Anderson's
  canonical-correlation LM under iid errors).
- Weak identification: the Kleibergen-Paap rk Wald F (Cragg-Donald's F under
  iid errors), and per endogenous regressor Sanderson and Windmeijer's (2016)
  conditional F. The F statistics take ivreg2's small-sample factor,
  (N - L)/N, times (G - 1)/G with G clusters (with N/(N - 1) in place of the
  first factor's N), so that with one endogenous regressor the SW F is the
  first-stage F.
- Overidentification: Hansen's J at the two-step efficient GMM estimate,
  weighted by the moments' covariance at the given residuals (ivreg2's J
  after robust 2SLS), or Sargan's statistic under iid errors.
- With `identification`, also each endogenous regressor's first-stage
  partial R^2 and Shea's (1997) partial R^2 (Godfrey's 1999 formula), and,
  with one endogenous regressor, Montiel Olea and Pflueger's (2013)
  effective F with its simplified critical value for 2SLS.
- Weak-instrument-robust inference: the Anderson-Rubin (1949) test of the
  endogenous regressors' coefficients (robust as the variance is), and with
  one endogenous regressor its confidence set (`anderson_rubin`).
- Endogeneity: the C statistic (difference-in-Sargan/Hansen) for treating
  the endogenous regressors as exogenous, as ivreg2's ``endog()``
  (`c_statistic`).

`kp_rk` follows ranktest's SVD algebra (Kleibergen and Paap 2006, section 3).
"""

from __future__ import annotations

import numpy as np
from scipy import stats

__all__ = ["anderson_rubin", "c_statistic", "effective_f", "hansen_j", "identification",
           "iid_omegas", "kp_rk", "partial_r2", "sargan", "weak_f"]


def _invsym(A, tol=1e-13):
    """Stata's invsym: the inverse of symmetric A, or with A singular a
    generalized inverse, sweeping the pivots in column order and dropping
    any that is negligible relative to its diagonal element. Returns
    (inverse, kept)."""
    A = np.array(A, dtype=np.float64, copy=True)
    n = A.shape[0]
    kept = np.zeros(n, bool)
    if n == 0:
        return A, kept
    d0 = np.abs(np.diag(A)).copy()
    for k in range(n):
        d = A[k, k]
        if d0[k] == 0.0 or abs(d) <= tol * d0[k]:
            A[k, :] = 0.0
            A[:, k] = 0.0
            continue
        row, col = A[k, :].copy(), A[:, k].copy()
        A -= np.outer(col / d, row)
        A[k, :] = row / d
        A[:, k] = col / d
        A[k, k] = -1.0 / d
        kept[k] = True
    out = np.zeros_like(A)
    idx = np.flatnonzero(kept)
    out[np.ix_(idx, idx)] = -A[np.ix_(idx, idx)]
    return (out + out.T) / 2, kept


def _sqrtm_psd(A):
    w, v = np.linalg.eigh((A + A.T) / 2)
    return v @ np.diag(np.sqrt(np.clip(w, 0.0, None))) @ v.T


def kp_rk(Qzz, Pi, Qyy, Omega):
    """Kleibergen and Paap's rk statistic for rank(Pi) = K - 1.

    Qzz = Z'Z (L x L) and Qyy = Y'Y (K x K), both net of the included
    exogenous regressors; Pi = Qzz^-1 Z'Y, the first-stage coefficients;
    Omega the covariance of vec(Z'u) (columns stacked, one per regressor),
    unnormalized, with u = Y for the LM form and the first-stage residuals
    for the Wald form. Returns (statistic, degrees of freedom).
    """
    L, K = Pi.shape
    rQzz = np.linalg.cholesky(Qzz)
    irQyy = np.linalg.inv(np.linalg.cholesky(Qyy).T)
    irQzz = np.linalg.inv(rQzz.T)
    that = rQzz.T @ Pi @ irQyy
    T = np.kron(irQyy.T, irQzz.T)
    kpvar = T @ Omega @ T.T
    u, _, vt = np.linalg.svd(that)
    v = vt.T
    kk = K - 1
    u12, u22 = u[:kk, kk:], u[kk:, kk:]
    v12, v22 = v[:kk, kk:], v[kk:, kk:]
    aq = np.vstack([u12, u22]) @ np.linalg.inv(u22) @ _sqrtm_psd(u22 @ u22.T)
    bq = _sqrtm_psd(v22 @ v22.T) @ np.linalg.inv(v22.T) @ np.vstack([v12, v22]).T
    M = np.kron(bq, aq.T)
    lab = M @ that.reshape(-1, order="F")
    ivlab, kept = _invsym(M @ kpvar @ M.T)
    return float(lab @ ivlab @ lab), (L - kk) * (K - kk) - int((~kept).sum())


def weak_f(wald, df1, n_obs, n_inst, n_clusters=None, dof_extra=0):
    """ivreg2's F form of a Wald statistic of `df1` restrictions on the
    first stage: wald/N (N - L - dof_extra)/df1, or with G clusters
    wald/(N - 1) (N - L - dof_extra) (G - 1)/G / df1. L counts all the
    instruments (excluded and included exogenous, the constant among them);
    `dof_extra` the parameters partialled out besides them, such as absorbed
    fixed effects. Returns (F, df2)."""
    N, L = n_obs, n_inst
    if n_clusters is None:
        return wald / N * (N - L - dof_extra) / df1, N - L - dof_extra
    G = n_clusters
    return wald / (N - 1) * (N - L - dof_extra) * (G - 1) / G / df1, G - 1


def iid_omegas(ZZ, ZY, YY, n_obs):
    """The covariances of vec(Z'Y) and vec(Z'V) under iid errors, V the
    first-stage residuals: kron(Y'Y/N, Z'Z) and kron(V'V/N, Z'Z), for
    Anderson's LM and Cragg and Donald's Wald statistic."""
    iZZ, _ = _invsym(ZZ)
    VV = YY - ZY.T @ iZZ @ ZY
    return np.kron(YY / n_obs, ZZ), np.kron(VV / n_obs, ZZ)


def identification(ZZ, ZY, YY, OU, OV, *, n_obs, n_inst, n_clusters=None, dof_extra=0,
                   endog=None, exog=None, sw="2sls"):
    """The identification tests from the partialled moments.

    ZZ = Z1'Z1 (L1 x L1), ZY = Z1'Y (L1 x Ke), YY = Y'Y: the excluded
    instruments Z1 and the endogenous regressors Y, both net of the included
    exogenous regressors (and of anything absorbed); OU, OV the covariances
    of vec(Z1'Y) and vec(Z1'V), V = Y - Z1 Pi (from `iid_omegas` for iid
    errors). `n_obs`, `n_inst` (all instruments), `n_clusters` and
    `dof_extra` give the F statistics' factor (`weak_f`).

    `sw` picks the Sanderson-Windmeijer statistic:

    - "2sls" (ivreg2's): for each endogenous regressor, its 2SLS regression
      on the others; the Wald test, with OV, that the excluded instruments
      do not explain the residuals, chi2(L1 - Ke + 1), and its F.
    - "gmm" (Windmeijer's robust test, as Stata's underid): the J statistic
      of that auxiliary regression estimated by two-step efficient GMM with
      the moments' covariance from OU, chi2(L1 - Ke + 1).

    Collinear excluded instruments are dropped first, as ranktest does.
    Returns a dict with ``kp_lm``, ``kp_wald`` (with ``F``) and
    ``sanderson_windmeijer`` (one entry per endogenous regressor).
    """
    L1, Ke = ZY.shape
    endog = list(endog) if endog is not None else [f"y{k}" for k in range(Ke)]
    _, zk = _invsym(ZZ)
    if not zk.all():
        idx = np.flatnonzero(zk)
        sel = np.concatenate([k * L1 + idx for k in range(Ke)])
        ZZ, ZY = ZZ[np.ix_(idx, idx)], ZY[idx]
        OU, OV = OU[np.ix_(sel, sel)], OV[np.ix_(sel, sel)]
        L1 = len(idx)
    if L1 < Ke:
        raise ValueError(f"{L1} excluded instruments for {Ke} endogenous regressors")
    iZZ, _ = _invsym(ZZ)
    Pi = iZZ @ ZY
    lm, df = kp_rk(ZZ, Pi, YY, OU)
    wald, dfw = kp_rk(ZZ, Pi, YY, OV)
    F, _ = weak_f(wald, L1, n_obs, n_inst, n_clusters, dof_extra)
    out = {"endogenous": endog, "exogenous": list(exog or []), "excluded_instruments": L1,
           "kp_lm": {"statistic": lm, "df": df, "p": float(stats.chi2.sf(lm, df))},
           "kp_wald": {"statistic": wald, "df": dfw, "p": float(stats.chi2.sf(wald, dfw)),
                       "F": F},
           "sanderson_windmeijer": []}
    df_sw = L1 - Ke + 1
    for k in range(Ke):
        others = [o for o in range(Ke) if o != k]
        A, yk = ZY[:, others], ZY[:, k]
        c = np.zeros(Ke)
        c[k] = 1.0
        if others:
            c[others] = -np.linalg.solve(A.T @ iZZ @ A, A.T @ iZZ @ yk)
        C = np.kron(c[:, None], np.eye(L1))
        entry = {"variable": endog[k], "df": df_sw}
        if sw == "gmm":
            S = C.T @ OU @ C
            iS, _ = _invsym(S)
            m = yk
            if others:
                m = yk - A @ np.linalg.solve(A.T @ iS @ A, A.T @ iS @ yk)
            J = float(m @ iS @ m)
            entry.update(statistic=J, p=float(stats.chi2.sf(J, df_sw)))
        else:
            # the residual's first-stage coefficients Pi c, tested with the
            # covariance of Z1'V c
            b = Pi @ c
            Vb = iZZ @ (C.T @ OV @ C) @ iZZ
            iVb, _ = _invsym(Vb)
            W = float(b @ iVb @ b)
            Fk, df2 = weak_f(W, df_sw, n_obs, n_inst, n_clusters, dof_extra)
            entry.update(statistic=W, p=float(stats.chi2.sf(W, df_sw)), F=Fk, df2=df2,
                         F_p=float(stats.f.sf(Fk, df_sw, df2)))
        out["sanderson_windmeijer"].append(entry)
    out["partial_r2"] = partial_r2(ZZ, ZY, YY, endog)
    out["effective_f"] = (effective_f(ZZ, ZY, OV, n_obs=n_obs, n_inst=n_inst,
                                      n_clusters=n_clusters, dof_extra=dof_extra)
                          if Ke == 1 else None)
    return out


def partial_r2(ZZ, ZY, YY, endog=None):
    """Each endogenous regressor's first-stage partial R^2 (of the excluded
    instruments, net of the included exogenous regressors) and Shea's (1997)
    partial R^2, by Godfrey's (1999) formula: [(Y'Y)^-1]_kk /
    [(Y^'Y^)^-1]_kk, Y^ the first-stage fitted values (all net of the
    exogenous regressors, which leaves those diagonal elements as they are
    with them). With one endogenous regressor the two are equal. Returns
    {name: {"partial": ..., "shea": ...}}."""
    Ke = ZY.shape[1]
    endog = list(endog) if endog is not None else [f"y{k}" for k in range(Ke)]
    iZZ, _ = _invsym(ZZ)
    F = ZY.T @ iZZ @ ZY                    # Y^'Y^
    iY, _ = _invsym(YY)
    iF, _ = _invsym(F)
    out = {}
    for k, name in enumerate(endog):
        part = float(F[k, k] / YY[k, k]) if YY[k, k] > 0 else np.nan
        shea = float(iY[k, k] / iF[k, k]) if iF[k, k] > 0 else 0.0
        out[name] = {"partial": part, "shea": shea}
    return out


def effective_f(ZZ, ZY, OV, *, n_obs, n_inst, n_clusters=None, dof_extra=0, tau=0.1,
                alpha=0.05):
    """Montiel Olea and Pflueger's (2013) effective F for one endogenous
    regressor: pi'(Z'Z) pi / tr(Sigma (Z'Z)), with pi its first-stage
    coefficients on the excluded instruments (net of the exogenous
    regressors) and Sigma their robust variance, from OV (the covariance of
    Z'v), with ivreg2's small-sample factor (`weak_f`): under iid errors it
    is the first-stage F. Also the simplified critical value for 2SLS
    (Patnaik's approximation, worst-case bias `tau` relative to OLS's, size
    `alpha`), with its effective degrees of freedom. Returns a dict."""
    L1 = ZZ.shape[0]
    iZZ, _ = _invsym(ZZ)
    pi = iZZ @ ZY[:, 0]
    Sig = iZZ @ OV @ iZZ                    # variance of pi, unnormalized
    num = float(pi @ ZZ @ pi)
    den = float(np.trace(Sig @ ZZ))
    F, _ = weak_f(L1 * num / den, L1, n_obs, n_inst, n_clusters, dof_extra)
    # the critical value: Sigma in the instruments' normalized coordinates
    rZZ = _sqrtm_psd(ZZ)
    W = rZZ @ Sig @ rZZ
    W = W / np.trace(W)
    x = 1.0 / tau
    lmax = float(np.linalg.eigvalsh((W + W.T) / 2).max())
    k_eff = float((1 + 2 * x) / (np.trace(W @ W) + 2 * x * lmax))
    cv = float(stats.ncx2.ppf(1 - alpha, k_eff, x * k_eff) / k_eff)
    return {"F": float(F), "critical_value": cv, "k_eff": k_eff, "tau": tau,
            "alpha": alpha}


def anderson_rubin(ZZ, ZYa, OVa, *, n_obs, n_inst, n_clusters=None, dof_extra=0,
                   beta0=None, level=0.95, center=None, scale=None):
    """The Anderson-Rubin test of H0: the endogenous regressors'
    coefficients are `beta0` (default 0): the Wald test that the excluded
    instruments do not explain y - Y beta0 (net of the exogenous
    regressors), with the variance of the first-stage residuals' moments,
    so as robust as that is. Valid however weak the instruments.

    ZZ = Z1'Z1; ZYa = Z1'[Y y] (L1 x (Ke + 1)), y last; OVa the covariance of
    vec(Z1'[V v]), V and v their first-stage (reduced-form) residuals.
    Returns the chi2(L1) statistic, its F (`weak_f`) and p-values, and with
    one endogenous regressor the confidence set at `level`: the beta0 the
    test does not reject, as a list of intervals (they can be unbounded, or
    two rays, or everything). `center` and `scale` (an estimate and its
    standard error) place the search's grid."""
    L1, K1 = ZYa.shape
    Ke = K1 - 1
    iZZ, _ = _invsym(ZZ)
    Pa = iZZ @ ZYa

    def stat(b0):
        c = np.concatenate([-np.atleast_1d(b0).astype(float), [1.0]])
        C = np.kron(c[:, None], np.eye(L1))
        V = iZZ @ (C.T @ OVa @ C) @ iZZ
        iV, _ = _invsym(V)
        p = Pa @ c
        return float(p @ iV @ p)
    b0 = np.zeros(Ke) if beta0 is None else np.atleast_1d(beta0).astype(float)
    W = stat(b0)
    F, df2 = weak_f(W, L1, n_obs, n_inst, n_clusters, dof_extra)
    out = {"beta0": b0.tolist(), "statistic": W, "df": L1,
           "p": float(stats.chi2.sf(W, L1)), "F": F, "df2": df2,
           "F_p": float(stats.f.sf(F, L1, df2)), "confidence_set": None}
    if Ke != 1:
        return out
    crit = stats.chi2.ppf(level, L1)
    c0 = float(center) if center is not None else 0.0
    sc = float(scale) if scale is not None and scale > 0 else 1.0
    # a grid over the whole line, dense near the estimate: b = c0 + sc tan(t)
    th = np.linspace(-np.pi / 2, np.pi / 2, 4003)[1:-1]
    grid = c0 + 20 * sc * np.tan(th)
    f = np.array([stat(b) - crit for b in grid])
    inside = f <= 0
    if not inside.any():
        out["confidence_set"] = {"level": level, "intervals": []}
        return out
    from scipy.optimize import brentq
    intervals, lo = [], (-np.inf if inside[0] else None)
    for i in range(1, len(grid)):
        if inside[i] != inside[i - 1]:
            r = brentq(lambda b: stat(b) - crit, grid[i - 1], grid[i], xtol=1e-12 * sc)
            if inside[i]:
                lo = r
            else:
                intervals.append([lo, r])
                lo = None
    if lo is not None:
        intervals.append([lo, np.inf])
    out["confidence_set"] = {"level": level, "intervals": intervals}
    return out


def c_statistic(ZX, Zy, S, tested):
    """The C (difference-in-Sargan/Hansen) statistic for the moments
    `tested` (indices): Hansen's J with every moment less J without them,
    both weighted by (submatrices of) the full set's covariance S, so it is
    never negative (Hayashi 2000). As ivreg2's ``endog()``, with the
    endogenous regressors added to the instruments as the tested moments,
    and S at the residuals of the model that treats them as exogenous (OLS).
    Returns (C, df, p); df = the tested moments' rank."""
    L = ZX.shape[0]
    keep = np.array([i for i in range(L) if i not in set(tested)], np.int64)
    Jf, dff, _ = hansen_j(ZX, Zy, S)
    Jr, dfr, _ = hansen_j(ZX[keep], Zy[keep], S[np.ix_(keep, keep)])
    C, df = Jf - Jr, dff - dfr
    return float(C), int(df), float(stats.chi2.sf(C, df)) if df > 0 else np.nan


def hansen_j(ZX, ZY, S):
    """Hansen's J: the minimized two-step GMM criterion with weight S^-1
    (S the moments' covariance, unnormalized, at the residuals of a first
    estimate). ZX = Z'X, ZY = Z'y. Returns (J, df, p); df = rank(S) - K."""
    A, kept = _invsym(S)
    XZA = ZX.T @ A
    V, _ = _invsym(XZA @ ZX)
    b = V @ (XZA @ ZY)
    g = ZY - ZX @ b
    J = float(g @ A @ g)
    df = int(kept.sum()) - ZX.shape[1]
    return J, df, float(stats.chi2.sf(J, df)) if df > 0 else np.nan


def sargan(ZZ, Ze, ee, n_obs, k):
    """Sargan's statistic, N e'P_Z e / e'e, from Z'Z, Z'e and e'e at the
    2SLS residuals; chi2(L - k). Returns (statistic, df, p)."""
    iZZ, kept = _invsym(ZZ)
    s = float(n_obs * (Ze @ iZZ @ Ze) / ee)
    df = int(kept.sum()) - k
    return s, df, float(stats.chi2.sf(s, df)) if df > 0 else np.nan

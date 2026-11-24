"""The fast path of `feglm_stream` and `fepois_stream`: a logit, probit or
Poisson model with no fixed effect or one, fitted by Newton's method on
`core.newton`'s stored rows, with the result `StreamingGLM` would give.

`StreamingGLM` partitions and sorts the rows as Parquet files for its
demeaning, which a model with one fixed effect does not need: the effect is
profiled out exactly. Here pass 0 writes the rows once, as compact blocks
(`core.newton.glm_design`: the 0/1 columns one byte each); each Newton step
is one read of them through the fused pass (`core.kernels_newton`); and one
final read gives the scores for the variances and writes the residual file.
The result is an `HDFEResult` with `StreamingGLM`'s conventions:

- the same sample: incomplete rows, then the fixed effect's singletons
  (``fixef_rm="singleton"``), then the groups whose effect would be infinite
  (logit and probit: an outcome that never varies; Poisson: one that is
  always zero), with `sample()` saying which;
- the same coefficient names (``Intercept`` first without a fixed effect),
  and the same collinearity rule (pyfixest's, on the information at the
  start, with the relative check);
- the variances of the final step: bread the inverse information (the
  expected information, Fisher scoring's for the probit), scores w e x~ with
  x~ the covariates less each group's information-weighted mean, and
  pyfixest's small-sample factors -- iid (N - 1) / (N - K), hetero N / (N - K),
  CRV1 G / (G - 1) (N - 1) / (N - K), K counting the fixed effect's levels
  unless the fixed effect is nested in the clusters; multi-way clusters by
  inclusion and exclusion, with the smallest G;
- the same fit statistics and files (`fixef`, `resid`).

The estimates stop on the Newton decrement (below 1e-8, or `iwls_tol` if
smaller, or relative to the deviance below a thousandth of `iwls_tol`):
closer to the maximum than IRLS's test on the relative change in deviance.

With ``device="cuda"`` the Newton steps run on the device
(`core.gpu.newton_rows`), the rows uploaded once.

Anything else -- two or more fixed effects, varying slopes, multiple
estimation, CRV3, `bias_correction`, `bootcluster`, the separation check
switched off, `fe_dof` other than ``"exact"`` or ``"pyfixest"``, or an option
this path does not know -- is `StreamingGLM`'s.
"""

from __future__ import annotations

import time
from itertools import combinations
from types import SimpleNamespace

import numpy as np
import polars as pl
from scipy import sparse, special

from .core.families import get_family
from .core.feterms import parse_fe
from .core.formula import parse_formula
from .core.inference import _criteria, _ratio
from .core.report import warn
from .core.results import HDFEResult
from .core.results_base import _canon_cluster, vcov_key
from .core.utils import as_lazy

# options of feglm_stream/fepois_stream this path takes (others: StreamingGLM)
_OPTIONS = {"weights", "weights_type", "workdir", "verbose", "logger", "max_memory_gb",
            "memory_reserve", "polars_engine", "n_threads", "storage", "dtype", "device",
            "block_mb", "chunk_mb", "save_resid"}
# the most memory the information kept for a one-way fixed effect may take
# (`HDFEResult._newton_information`, for packages that draw its levels): at
# most this, and at most half the memory budget's headroom
INFORMATION_GB = 2.0
_MAX_HALVINGS = 30
# whether feglm_stream and fepois_stream take this path when they can (tests
# compare it with StreamingGLM's)
ENABLED = True


def eligible(fml, data, *, vcov, cluster, fe_dof, separation_check, profile, options):
    """Whether the fast path fits this model (see the module docstring)."""
    if profile or not separation_check or fe_dof not in ("exact", "pyfixest"):
        return False
    if any(k not in _OPTIONS for k in options) or options.get("device", "cpu") not in (
            "cpu", "cuda"):
        return False
    try:
        key = vcov_key(vcov)
    except ValueError:
        return False
    if key.startswith("CRV3:"):
        return False
    specs = parse_formula(fml)
    if len(specs) != 1 or specs[0].iv is not None:
        return False
    fe = parse_fe(specs[0].fe)
    return len(fe) <= 1 and not any("[" in d for d in fe)


def _cluster_requests(vcov, cluster):
    """The vcov keys (iid, hetero, the default, and one CRV1 per extra
    clustering), and for each CRV1 request its terms: (term, sign, one-way)
    by inclusion and exclusion; and every term's columns."""
    default = vcov_key(vcov)
    extra = [_canon_cluster(c) for c in ([cluster] if isinstance(cluster, str)
                                         else list(cluster or []))]
    reqs = list(dict.fromkeys(([default.split(":", 1)[1]] if default.startswith("CRV1:")
                               else []) + extra))
    keys = list(dict.fromkeys(["iid", "hetero", default] + [f"CRV1:{r}" for r in extra]))
    requests, terms = {}, {}
    for req in reqs:
        parts = [[c.strip() for c in p.split("^")] for p in req.split("+")]
        out = []
        for r in range(1, len(parts) + 1):
            for sub in combinations(parts, r):
                cols = [c for p in sub for c in p]
                name = "^".join(cols)
                terms[name] = cols
                out.append((name, (-1) ** (r + 1), r == 1))
        requests[req] = out
    return default, keys, requests, terms


def _collinear(C, raw, tol=1e-10, tol_rel=1e-6):
    """hdfe-stream's collinearity rule (`_InferenceMixin._collinear`):
    pyfixest's check on C, and a variable whose C diagonal is below tol_rel of
    its raw sum of squares."""
    k = C.shape[0]
    if not k:
        return np.zeros(0, bool)

    def pf(A):
        try:
            from pyfixest.core import find_collinear_variables
            mask, _, _ = find_collinear_variables(np.ascontiguousarray(A), tol)
            return np.asarray(mask, bool)
        except ImportError:
            m = np.zeros(A.shape[0], bool)
            for j in range(A.shape[0]):
                keep = np.flatnonzero(~m[:j])
                r = A[j, j]
                if len(keep):
                    r -= A[j, keep] @ np.linalg.solve(A[np.ix_(keep, keep)], A[keep, j])
                m[j] = r <= tol * max(A[j, j], 1e-300)
            return m

    mask = pf(C)
    if raw is not None:
        mask |= (raw > 0) & (np.diag(C) <= tol_rel * raw)
        if mask.any() and not mask.all():
            keep = np.flatnonzero(~mask)
            mask[keep] = pf(C[np.ix_(keep, keep)])
    return mask


def _terms(family, y, eta):
    """Per row: (log-likelihood, score in eta, expected information, mean,
    working residual)."""
    with np.errstate(all="ignore"):
        if family == "logit":
            p, q = special.expit(eta), special.expit(-eta)
            ll = np.where(y > 0, special.log_expit(eta), special.log_expit(-eta))
            return ll, y - p, p * q, p, (y - p) / (p * q)
        if family == "probit":
            lp, lq = special.log_ndtr(eta), special.log_ndtr(-eta)
            lphi = -0.5 * eta * eta - 0.9189385332046727
            mu = special.ndtr(eta)
            r = np.where(y > 0, np.exp(lphi - lp), -np.exp(lphi - lq))
            return (np.where(y > 0, lp, lq), r, np.exp(2 * lphi - lp - lq), mu,
                    (y - mu) / np.exp(lphi))
        mu = np.exp(eta)
        return y * eta - mu - special.gammaln(y + 1.0), y - mu, mu, mu, (y - mu) / mu


def fit(fml, data, family, *, vcov, cluster, fe_dof, iwls_tol, iwls_maxiter, offset,
        options):
    """The fast path's fit (call `eligible` first)."""
    from .core.newton import glm_design, newton_direction
    from .core.resources import peak_rss_gb, reset_peak_rss, thread_scope
    from .core.workspace import Run, _base_dir

    t0 = time.time()
    reset_peak_rss()
    options = dict(options)
    weights = options.pop("weights", None)
    weights_type = options.pop("weights_type", "aweights")
    n_threads = options.pop("n_threads", None)
    device = options.pop("device", "cpu")
    save_resid = bool(options.pop("save_resid", True))
    verbose = options.get("verbose", True)
    logger = options.get("logger")
    fam = get_family(family)
    default, keys, requests, terms = _cluster_requests(vcov, cluster)
    lf = as_lazy(data)
    spec = parse_formula(fml)[0]
    with thread_scope(n_threads):
        ld = glm_design(fml, lf, family=fam.name, weights=weights, weights_type=weights_type,
                        offset=offset, clusters=terms, drop_singletons=True, **options)
        try:
            out = _fit(ld, fml, spec, lf, fam, weights, weights_type, offset, default, keys,
                       requests, terms, fe_dof, iwls_tol, iwls_maxiter, verbose, logger, t0,
                       Run, _base_dir, options.get("workdir"), newton_direction, device,
                       save_resid)
        finally:
            ld.cleanup()
    rss = peak_rss_gb()
    out.diagnostics["peak_rss_gb"] = None if rss is None else round(rss, 3)
    return out


def _log(verbose, logger, msg):
    if verbose:
        if logger is not None:
            logger.info(msg)
        else:
            print(f"[hdfe {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _fit(ld, fml, spec, lf, fam, weights, weights_type, offset, default, keys, requests,
         terms, fe_dof, iwls_tol, iwls_maxiter, verbose, logger, t0, Run, _base_dir, workdir,
         newton_direction, device="cpu", save_resid=True):
    fe = ld.fe
    G = ld.n_groups
    if ld.diagnostics["singletons"]:
        warn(f"{ld.diagnostics['singletons']:,} singleton observations dropped from the model "
             "(fixef_rm='none' keeps them)", logger)
    k = len(ld.names)
    with_fe = fe is not None
    # parameters: with a fixed effect the slopes (the effects profiled out),
    # without one the intercept and the slopes
    names = list(ld.names) if with_fe else ["Intercept", *ld.names]
    p = len(names)
    sy = ld.group_outcomes.sum(axis=0)
    sw = ld.group_weights.sum(axis=0)
    mean = np.clip(sy / np.where(sw > 0, sw, 1.0), 1e-10, None)
    if fam.name == "poisson":
        a0 = np.log(mean)
    else:
        mean = np.clip(mean, 1e-10, 1 - 1e-10)
        a0 = np.log(mean / (1 - mean)) if fam.name == "logit" else special.ndtri(mean)
    b = np.zeros(k)
    a = a0 if with_fe else (np.zeros(1) if offset is not None else a0)

    def system(mom):
        """(information, gradient) over the parameters."""
        if with_fe:
            return mom.profiled(0)
        H = np.empty((p, p))
        H[0, 0] = mom.effects_information[0, 0]
        H[0, 1:] = H[1:, 0] = mom.cross[0, 0]
        H[1:, 1:] = mom.information[0]
        return H, np.r_[mom.effects_gradient[0, 0], mom.gradient[0]]

    passes = [0]
    on_device = device == "cuda"
    if on_device:
        from .core.gpu import cuda_backend, group_indicator, newton_rows
        from .core.newton import NewtonMoments
        be = cuda_backend()
        dev = []
        budget = 0.6 * getattr(be, "free_bytes", lambda: float("inf"))()
        used = 0
        for X, y, w, off, f, _s in ld._blocks():
            d = (be.xp.asarray(X), be.xp.asarray(y), None if w is None else be.xp.asarray(w),
                 None if off is None else be.xp.asarray(off), f if with_fe else None,
                 group_indicator(be, f, G) if with_fe else None)
            used += sum(x.nbytes for x in d[:4] if x is not None)
            if used > budget:
                raise MemoryError("device='cuda': the rows do not fit in the device's free "
                                  "memory")
            dev.append(d)

    # the probit's Newton steps take the observed information (quadratic
    # convergence); its expected one, Fisher scoring's, is the variance's bread
    newton_obs = fam.name == "probit"

    def npass(b, a, info=True, observed=False):
        passes[0] += 1
        if not on_device:
            mom = ld.newton_pass(b, a, information=info, observed=observed)
            return float(mom.loglik.sum()), mom
        ll, gb, ga = 0.0, np.zeros(k), np.zeros(G)
        H, Dg, Cg = np.zeros((k, k)), np.zeros(G), np.zeros((G, k))
        for X, y, w, off, f, F in dev:
            o = newton_rows(be, fam.name, X, y, w, off, f, F, b, a, info, G, observed)
            ll += o[0]
            gb += o[1]
            ga += o[2]
            if info:
                H += o[3]
                Dg += o[4]
                Cg += o[5]
        return ll, NewtonMoments(
            loglik=np.array([[ll]]), gradient=gb[None], effects_gradient=ga[None],
            information=((H + H.T) / 2)[None] if info else None,
            effects_information=Dg[None] if info else None,
            cross=Cg[None] if info else None, skipped=np.zeros((1, 1), np.int64),
            columns=np.arange(k))

    ll, mom = npass(b, a)
    # the collinearity check, on the information at the start (IRLS's first
    # step's), with each variable's raw sum of squares on the same weighting
    Cb, _ = system(mom)
    hs = float(mom.effects_information[0].sum())
    hx = mom.cross[0].sum(axis=0)
    rawx = np.diag(mom.information[0]) - hx * hx / hs if hs > 0 else None
    raw = rawx if with_fe else (None if rawx is None else np.r_[0.0, rawx])
    drop = _collinear(Cb, raw)
    if p and drop.all():
        raise ValueError(f"{fml}: all covariates are collinear with the fixed effects")
    dropped = [nm for nm, d in zip(names, drop) if d]
    if dropped:
        warn(f"{fml}: {len(dropped)} variables dropped due to multicollinearity: {dropped}",
             logger)
    sel = np.flatnonzero(~drop)
    tol_abs, tol_rel = min(1e-8, iwls_tol), 1e-3 * iwls_tol
    it, halvings, decrement, converged = 0, 0, np.inf, False

    def unpack(theta):
        return (theta, None) if with_fe else (theta[1:], theta[:1])

    theta = b.copy() if with_fe else np.r_[a[0], b]
    # a step this small moves each row's linear predictor by about
    # sqrt(decrement / weight sum) < 1e-8, and the information as little: the
    # information it was taken from serves as the variances' bread, unread
    unread = 1e-16 * ld.weight_sum
    polish = False
    for it in range(1, iwls_maxiter + 1):
        C, g = system(mom)
        d = np.zeros(p)
        d[sel], decrement = newton_direction(C[np.ix_(sel, sel)], g[sel])
        # once the decrement passes the test, one more step: near a regular
        # maximum it squares the error (what IRLS's final step does after
        # `StreamingGLM`'s Newton start). It is read, for the bread at the
        # estimates, unless it is too small to change it
        passed = decrement < max(tol_abs, tol_rel * (0.1 + 2.0 * abs(ll)))
        if polish or decrement == 0.0 or (passed and decrement < unread):
            theta = theta + d
            if with_fe:
                a = a + mom.effects_step(d, 0)
            converged = True
            break
        polish = passed
        da = mom.effects_step(d, 0) if with_fe else None
        t = 1.0
        for _ in range(_MAX_HALVINGS):
            th = theta + t * d
            bb, aa = unpack(th)
            if with_fe:
                aa = a + t * da
            ll_new, mom_new = npass(bb, aa, observed=newton_obs)
            if np.isfinite(ll_new) and ll_new >= ll - 1e-12 * (1.0 + abs(ll)):
                break
            t /= 2
            halvings += 1
        else:
            break
        theta, ll, mom = th, ll_new, mom_new
        if with_fe:
            a = aa
        _log(verbose, logger, f"  Newton {it}: log-likelihood {ll:.12g}, decrement "
                              f"{decrement:.2e}" + (f", step {t:g}" if t < 1 else ""))
    if not converged:
        warn(f"{fml}: Newton's method did not converge within iwls_maxiter={iwls_maxiter} "
             "steps", logger)
    b, ab = unpack(theta)
    if not with_fe:
        a = ab
    if newton_obs and it > 1:
        # the expected information at the estimates, for the variances
        ll, mom = npass(b, a)
    keep = sel
    names_k = [names[j] for j in keep]
    C, _ = system(mom)
    Ck = C[np.ix_(keep, keep)]
    Binv = np.linalg.inv(Ck) if len(keep) else np.zeros((0, 0))
    # each group's information-weighted mean of the covariates, for the
    # profiled scores
    D = mom.effects_information[0]
    mu_g = (np.where(D[:, None] > 0, mom.cross[0] / np.where(D > 0, D, 1.0)[:, None], 0.0)
            if with_fe else None)

    # the final read: scores, cluster sums, outcome sums, and the residual file
    run = Run(_base_dir(workdir), auto_cleanup=True)
    mdir = run.path / "models" / "m000"
    mdir.mkdir(parents=True)
    resid_path = str(mdir / "resid.parquet") if save_resid else None
    fweights = weights is not None and weights_type == "fweights"
    ctabs = ld.diagnostics["cluster_tables"]
    cl_names = list(terms)
    csums = {t: np.zeros((ctabs[t].height, len(keep))) for t in cl_names}
    cl_seen = {t: np.zeros(ctabs[t].height, bool) for t in cl_names}
    slg = 0.0
    ysum = wsum = 0.0
    lsat = 0.0
    n_obs_g = np.zeros(G, np.int64)
    fe_cols = fe.split("^") if with_fe else []
    keep_cols = list(dict.fromkeys(fe_cols + [c for t in cl_names for c in terms[t]]))
    writer = None
    import pyarrow as pa
    import pyarrow.parquet as pq
    beta_full = np.asarray(b, dtype=np.float64)
    # the robust variance's meat, sum (w r)^2 x x' (w r^2 x x' with frequency
    # weights), in the information's arrow shape -- x stands for (1, x)
    # without a fixed effect, x - (its group's mean) with one -- from the
    # row kernel's squared-score sums
    msums = ld._sums(beta_full, np.atleast_2d(a), meat=1 if fweights else 2)
    all_keep = len(keep) == p
    for blk in ld._rows_full():
        X, y, w, off, f = blk["X"], blk["y"], blk["w"], blk["off"], blk["f"]
        n = len(y)
        xb = X @ beta_full + (0.0 if with_fe else a[0])
        eff = a[f] if with_fe else np.zeros(n)
        eta = xb + eff + (0.0 if off is None else off)
        _, r, _, mu, uw = _terms(fam.name, y, eta)
        wt = np.ones(n) if w is None else w
        msums.add(X, y, w, off, f, blk["s"])
        if cl_names:
            Xs = (X - mu_g[f]) if with_fe else np.column_stack([np.ones(n), X])
            if not all_keep:
                Xs = Xs[:, keep]
            rows = np.arange(n)
            for j, t in enumerate(cl_names):
                code = blk["cl"][:, (1 if ld._split_of is not None else 0) + j]
                # the clusters' summed scores, sum over the rows of w r x
                csums[t] += sparse.csr_matrix((wt * r, (code, rows)),
                                              shape=(len(cl_seen[t]), n)) @ Xs
                cl_seen[t][code] = True
        ysum += float(wt @ y)
        wsum += float(wt.sum())
        if fam.name == "poisson":
            lg = special.gammaln(y + 1.0)
            slg += float(wt @ lg)
            lsat += float(wt @ (special.xlogy(y, y) - y - lg))
        if with_fe:
            n_obs_g += np.bincount(f, minlength=G)
        if not save_resid:
            continue
        cols = {}
        for c in keep_cols:
            cols[c] = _values_of(ld, c, blk, cl_names, terms, fe_cols)
        cols["row_id"] = pa.array(blk["row"], type=pa.int64())
        if w is not None:                       # the weights as given
            cols["weights"] = pa.array(w / ld.diagnostics["weight_scale"])
        cols[ld.depvar] = pa.array(y)
        if with_fe:
            cols[f"fe_{fe}"] = pa.array(eff)
        cols["xb"] = pa.array(xb)
        cols["resid"] = pa.array(y - mu)
        cols["eta"] = pa.array(eta)
        cols["fitted"] = pa.array(mu)
        cols["resid_working"] = pa.array(uw)
        table = pa.table(cols)
        if writer is None:
            writer = pq.ParquetWriter(resid_path, table.schema)
        writer.write_table(table)
    if writer is not None:
        writer.close()

    mm = msums.moments()
    Hm, Cm, Dm = mm.information[0], mm.cross[0], mm.effects_information[0]
    if with_fe:
        Cmu = Cm.T @ mu_g
        meat = Hm - Cmu - Cmu.T + mu_g.T @ (Dm[:, None] * mu_g)
    else:
        meat = np.empty((p, p))
        meat[0, 0] = Dm[0]
        meat[0, 1:] = meat[1:, 0] = Cm[0]
        meat[1:, 1:] = Hm
    meat_hc = meat[np.ix_(keep, keep)]

    # the variances
    N = ld.weight_sum if fweights else ld.n_obs
    k_fe = G if with_fe else 0
    K = len(keep) + k_fe
    vc = {"iid": (_ratio(N - 1, N - K) * Binv, N - K),
          "hetero": (_ratio(N, N - K) * Binv @ meat_hc @ Binv, N - K)}
    factors = {"iid": _ratio(N - 1, N - K), "hetero": _ratio(N, N - K)}
    n_clusters, nested_by = {}, {}
    nested = {t: _nested(ld, terms[t], fe_cols, t, cl_names) if with_fe else False
              for t in cl_names}
    for req, parts in requests.items():
        Gs = [int(cl_seen[t].sum()) for t, _, single in parts if single]
        Gm = min(Gs)
        nest = any(nested[t] for t, _, _ in parts)
        Kc = len(keep) + (1 if nest else k_fe)
        adj = _ratio(Gm, Gm - 1) * _ratio(N - 1, N - Kc)
        Vc = sum(sign * adj * (Binv @ (csums[t].T @ csums[t]) @ Binv) for t, sign, _ in parts)
        vc[f"CRV1:{req}"] = (Vc, Gm - 1)
        factors[f"CRV1:{req}"] = adj
        n_clusters[req] = Gs
        nested_by[f"CRV1:{req}"] = bool(nest)
    V, df_t = vc[default]
    all_vcovs = {kk: vc[kk] for kk in keys}

    # the fit statistics
    ll0 = float(fam.loglik_null(ysum, wsum, slg))
    deviance = -2.0 * (ll - lsat)
    ll_fe0 = np.nan
    if with_fe and offset is None:
        gy, gw = ld.group_outcomes.sum(axis=0), ld.group_weights.sum(axis=0)
        ll_fe0 = float(fam._means_ll(gy, gw).sum() - slg)
    aic, bic = _criteria(ll, len(keep) + k_fe, N)

    # the fixed effect's file
    fe_paths = {}
    if with_fe:
        fe_path = str(mdir / f"fe_{fe.replace('^', '_')}.parquet")
        ld.fe_levels.with_columns(pl.Series(f"fe_{fe}", a),
                                  pl.Series("n_obs", n_obs_g)).write_parquet(fe_path)
        fe_paths[fe] = fe_path

    exprs = {**ld.exprs, "Intercept": pl.lit(1.0)}
    if offset is not None:
        exprs["__offset__"] = ld.offset_expr
    est = SimpleNamespace(var_exprs=exprs, fe_cols={fe: fe_cols} if with_fe else {},
                          family=fam, offset_name=(offset if isinstance(offset, str)
                                                   else None if offset is None else "<expr>"),
                          slopes={}, slope_vars=[], fe_user=[fe] if with_fe else [],
                          fe=[fe] if with_fe else [], device="cpu")
    sample_spec = {"source": lf, "row_id": "row_id", "usable": ld.usable,
                   "n_raw": ld.diagnostics["n_raw"], "fe_cols": {fe: fe_cols} if with_fe else {},
                   "tables": {"min_degree": {}, "disconnected": {}, "leave_one_out": {},
                              "singleton": ({fe: ld.dropped_levels["singleton"]}
                                            if with_fe and ld.dropped_levels["singleton"].height
                                            else {}),
                              "separation": ({fe: ld.dropped_levels["separation"]}
                                             if with_fe and
                                             ld.dropped_levels["separation"].height else {})}}
    rhs = spec.rhs
    if with_fe and rhs.startswith("1 + "):     # the effects carry the constant
        rhs = rhs[4:]
    shown = f"{spec.dep} ~ {rhs}" + (f" | {fe}" if with_fe else "")
    diagnostics = {
        "fit": "newton",
        "newton": {"iterations": it, "converged": converged, "passes": passes[0] + 1,
                   "step_halvings": halvings, "decrement": float(decrement),
                   "nonzeros_only": ld.use_nonzeros and not on_device,
                   "fe_sorted": ld.fe_sorted, "device": be.name if on_device else None},
        "separation": dict(ld.separation),
        "singletons": {"observations": ld.diagnostics["singletons"],
                       "levels": ({fe: ld.dropped_levels["singleton"].height}
                                  if with_fe else {}),
                       "rounds": 1 if with_fe else 0},
        "vcov_factors": {kk: float(v) for kk, v in factors.items()},
        "nested_in_cluster": nested_by, "seconds_total": round(time.time() - t0, 2)}
    if offset is not None:
        diagnostics["offset"] = est.offset_name
    res = HDFEResult(
        fml=shown, depvar=spec.dep, coefnames=names_k, beta=theta[keep], vcov=V,
        vcov_type=default, df_t=df_t, n_obs=int(N) if float(N).is_integer() else N,
        all_vcovs=all_vcovs, n_clusters=n_clusters, collin_vars=dropped,
        diagnostics=diagnostics, weights=weights, weights_type=weights_type,
        fe_names=[fe] if with_fe else [], n_levels={fe: G} if with_fe else {},
        n_identifying=0, n_components=0, k_fe=k_fe, rss=np.nan, r2_within=np.nan,
        solver_info={"solver": "none", "iterations": 0, "converged": True},
        paths={"dir": str(mdir), "resid": resid_path, "fe": fe_paths},
        family=fam.name, deviance=float(deviance), loglik=float(ll),
        pseudo_r2=float(1 - ll / ll0) if ll0 else np.nan, loglik_null=ll0,
        loglik_fe_null=ll_fe0,
        pseudo_r2_within=float(1 - ll / ll_fe0) if np.isfinite(ll_fe0) and ll_fe0 else np.nan,
        aic=float(aic), bic=float(bic), _run=run, _estimator=est, _sample=sample_spec)
    # the information at the estimates, by level, for a package that draws the
    # levels jointly with the coefficients (logit-stream's draw_parameters):
    # each level's own (D, G), its cross block with the kept coefficients
    # (G, k), and the inverse of the coefficients' information net of the
    # levels (the iid bread before its small-sample factor)
    info_gb = G * (max(len(keep), 1) + 1) * 8 / 1e9
    if (with_fe and info_gb <= INFORMATION_GB
            and (ld._budget is None or ld._budget.fits(info_gb, share=0.5))):
        res._newton_information = {"D": np.asarray(D, float).copy(),
                                   "cross": np.ascontiguousarray(mom.cross[0][:, keep]),
                                   "Vred": Binv}
    return res


def _values_of(ld, col, blk, cl_names, terms, fe_cols):
    """Column `col` (a fixed-effect or cluster column) for a block's rows,
    from the codes stored and the levels' tables."""
    import pyarrow as pa

    from .core.newton import CODE
    ctabs = ld.diagnostics["cluster_tables"]
    off = 1 if ld._split_of is not None else 0
    for j, t in enumerate(cl_names):
        if col in terms[t]:
            tab = ctabs[t].sort(CODE)
            return tab[col].to_arrow().take(pa.array(blk["cl"][:, off + j]))
    tab = ld.fe_table.sort(CODE)
    return tab[col].to_arrow().take(pa.array(blk["fe"]))


def _nested(ld, cols, fe_cols, term, cl_names):
    """Whether every level of the fixed effect lies within one cluster of
    `term`."""
    if set(cols) <= set(fe_cols):
        return True
    j = cl_names.index(term) + (1 if ld._split_of is not None else 0)
    lo = hi = None
    for blk in ld._rows_full():
        f, c = blk["f"], blk["cl"][:, j]
        G = ld.n_groups
        mn = np.full(G, np.iinfo(np.int64).max)
        mx = np.full(G, -1)
        np.minimum.at(mn, f, c)
        np.maximum.at(mx, f, c)
        lo = mn if lo is None else np.minimum(lo, mn)
        hi = mx if hi is None else np.maximum(hi, mx)
    return bool(lo is not None and np.all(lo == hi))

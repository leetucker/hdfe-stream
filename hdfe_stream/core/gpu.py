"""Optional GPU execution (experimental): the reduced solve on CUDA through
CuPy (``device="cuda"``), and the fit's Polars queries on the cudf-polars GPU
engine (``polars_engine="gpu"``).

Neither changes what is computed. The solve is the same block PCG as
`_SolveMixin._pcg` with the Jacobi preconditioner, on the same explicit
matrix S: S is built on the CPU as usual, copied to the GPU once, and every
iteration runs there; only the right-hand sides go over and the solutions
come back. Sums are formed in a different order than on the CPU, so the
estimates differ in the last digits, far inside the solver tolerance.

A GLM without fixed effects has no solve. With ``device="cuda"`` its rows are
uploaded once and each IRLS step runs on the device (`glm_rows_plain`): the
family's terms and the cross-products, of which only the sums come back.

Both packages are imported only when asked for, so nothing here is needed to
use the library without a GPU.
"""

from __future__ import annotations

import importlib.util
from types import SimpleNamespace

import numpy as np

_INV_SQRT_2PI = 0.3989422804014327
_LOG_2PI_HALF = 0.9189385332046727       # log(2 pi) / 2
_CUPY_HINT = ("device='cuda' needs CuPy built for your CUDA version, e.g. "
              "`pip install cupy-cuda13x` (or cupy-cuda12x), and an NVIDIA GPU")


def cuda_backend():
    """The array namespace the device solver runs on: CuPy and its sparse
    module. Tests replace this with a numpy/scipy stand-in."""
    try:
        import cupy
        import cupyx.scipy.sparse as cusparse
    except ImportError as err:
        raise ImportError(_CUPY_HINT) from err
    if cupy.cuda.runtime.getDeviceCount() < 1:
        raise RuntimeError("device='cuda': CuPy found no CUDA device")
    import cupyx.scipy.special as cusp
    return SimpleNamespace(
        xp=cupy, asnumpy=cupy.asnumpy, special=cusp,
        free_bytes=lambda: cupy.cuda.runtime.memGetInfo()[0],
        release=lambda: cupy.get_default_memory_pool().free_all_blocks(),
        csr=lambda S: cusparse.csr_matrix(
            (cupy.asarray(S.data), cupy.asarray(S.indices), cupy.asarray(S.indptr)),
            shape=S.shape),
        name=cupy.cuda.runtime.getDeviceProperties(0)["name"].decode())


def glm_rows_plain(be, family, init, ybar, y, X, f, off, beta):
    """`_nb_glm_rows_plain` on the device, for a chunk of rows whose arrays
    (`y`, `X`, and `f`, `off` or None) are already there: the same terms, row
    by row, from the same formulas (`families.py`), so the sums agree with
    the CPU's to rounding.

    Returns the scalars (deviance, log-likelihood, f y, f, f log(y!), sum w),
    the sums of w v (v = [z, x], z = eta + u less the offset, w = W f) and the
    cross-products V'V of V = sqrt(w) v, all as host arrays: a handful of
    numbers however many rows."""
    xp = be.xp
    sp = getattr(be, "special", None)
    if sp is None:
        import scipy.special as sp
    with np.errstate(all="ignore"):         # the stand-in evaluates both branches of where
        if init:
            eta = xp.log(0.5 * (y + ybar)) if family == "poisson" else xp.zeros_like(y)
        else:
            eta = X @ xp.asarray(beta)
            if off is not None:
                eta = eta + off
        pos = y > 0
        lg = None
        if family == "poisson":
            mu = xp.exp(eta)
            W = mu
            u = xp.where(mu > 0, y / xp.where(mu > 0, mu, 1.0) - 1.0, 0.0)
            dev = 2.0 * (sp.xlogy(y, y) - y * eta - y + mu)
            lg = sp.gammaln(y + 1.0)
            ll = y * eta - mu - lg
        elif family == "logit":
            p, q = sp.expit(eta), sp.expit(-eta)
            W = p * q
            resid = xp.where(pos, q, -p)
            u = xp.where(W > 0, resid / xp.where(W > 0, W, 1.0), 0.0)
            dev = -2.0 * xp.where(pos, sp.log_expit(eta), sp.log_expit(-eta))
            ll = -0.5 * dev
        else:
            p, q = sp.ndtr(eta), sp.ndtr(-eta)
            pdf = xp.exp(-0.5 * eta * eta) * _INV_SQRT_2PI
            var = p * q
            W = xp.where(var > 0, pdf * pdf / xp.where(var > 0, var, 1.0), 0.0)
            resid = xp.where(pos, q, -p)
            u = xp.where((W > 0) & (pdf > 0), resid / xp.where(pdf > 0, pdf, 1.0), 0.0)
            dev = -2.0 * xp.where(pos, sp.log_ndtr(eta), sp.log_ndtr(-eta))
            ll = -0.5 * dev
        fw = xp.ones_like(y) if f is None else f
        w = W * fw
        z = eta + u
        if off is not None:
            z = z - off
        Z = xp.concatenate([z[:, None], X], axis=1)
        V = Z * xp.sqrt(w)[:, None]
        scal = xp.stack([(fw * dev).sum(), (fw * ll).sum(), (fw * y).sum(), fw.sum(),
                         (fw * lg).sum() if lg is not None else xp.zeros(()), w.sum()])
        return be.asnumpy(scal), be.asnumpy(w @ Z), be.asnumpy(V.T @ V)


def group_indicator(be, f, G):
    """The (G, n) 0/1 matrix of each row's group, on the device (sparse), for
    the groups' sums of a chunk whose group codes `f` (host) are 0..G-1."""
    import scipy.sparse as sps
    n = len(f)
    S = sps.csr_matrix((np.ones(n), (np.asarray(f, np.int64), np.arange(n))), shape=(G, n))
    return be.csr(S)


def newton_rows(be, family, X, y, w, off, f, F, beta, effects, info, G, observed=False):
    """One model's sums of `core.newton`'s pass on the device, for a chunk
    whose arrays are there (`X`, `y`, and `w`, `off` or None; `f` the group
    codes and `F` their indicator, `group_indicator`, both None without a
    fixed effect): the log-likelihood, the gradient for b and for the
    effects, and with `info` the information for b, for each effect, and
    between them. The terms are those of `kernels_newton.row_terms`.

    Returns host arrays: (ll, gb (k,), ga (G,), H (k, k), D (G,), C (G, k)),
    the last three None without `info`."""
    xp = be.xp
    sp = getattr(be, "special", None)
    if sp is None:
        import scipy.special as sp
    with np.errstate(all="ignore"):         # the stand-in evaluates both branches of where
        eta = X @ xp.asarray(beta)
        a = xp.asarray(effects)
        eta = eta + (a[xp.asarray(f)] if f is not None else a[0])
        if off is not None:
            eta = eta + off
        pos = y > 0
        if family == "logit":
            p, q = sp.expit(eta), sp.expit(-eta)
            ll = xp.where(pos, sp.log_expit(eta), sp.log_expit(-eta))
            r = xp.where(pos, q, -p)
            h = p * q
        elif family == "probit":
            lp, lq = sp.log_ndtr(eta), sp.log_ndtr(-eta)
            lphi = -0.5 * eta * eta - _LOG_2PI_HALF
            ll = xp.where(pos, lp, lq)
            r = xp.where(pos, xp.exp(lphi - lp), -xp.exp(lphi - lq))
            h = (r * (r + eta) if observed
                 else xp.exp(2.0 * lphi - lp - lq))
        else:
            mu = xp.exp(eta)
            ll = y * eta - mu - sp.gammaln(y + 1.0)
            r = y - mu
            h = mu
        if w is not None:
            ll, r, h = w * ll, w * r, w * h
        gb = X.T @ r
        if F is None:
            ga = r.sum()[None]
        else:
            ga = F @ r
        out = [float(be.asnumpy(ll.sum())), be.asnumpy(gb), be.asnumpy(ga)]
        if not info:
            return (*out, None, None, None)
        Xh = X * h[:, None]
        H = X.T @ Xh
        if F is None:
            D, C = h.sum()[None], Xh.sum(axis=0)[None, :]
        else:
            D, C = F @ h, F @ Xh
        return (*out, be.asnumpy(H), be.asnumpy(D), be.asnumpy(C))


def check_polars_gpu():
    """Raise ImportError, saying what to install, unless the cudf-polars
    GPU engine is importable (``polars_engine="gpu"``)."""
    if importlib.util.find_spec("cudf_polars") is None:
        raise ImportError("polars_engine='gpu' needs the cudf-polars package for your CUDA "
                          "version, e.g. `pip install cudf-polars-cu13`, and an NVIDIA GPU")


def explicit_solver(S, dinv, tol, maxiter, debug=None):
    """(solve(b, zero, x0) -> (X, iterations, converged, rel), device name):
    S and the preconditioner are copied to the device here, once, and stay
    there for every right-hand side the solver is given."""
    be = cuda_backend()
    A = be.csr(S)
    d = be.xp.asarray(dinv)[:, None]

    def solve(b, zero=None, x0=None):
        return pcg(be, A, d, b, zero, x0, tol, maxiter, debug)
    return solve, be.name


def pcg(be, A, dinv, b, zero=None, x0=None, tol=1e-10, maxiter=5000, debug=None):
    """Jacobi-preconditioned block CG on the device, column by column in
    lockstep: the algorithm of `_SolveMixin._pcg`, step for step, with the
    same handling of `zero` columns, the starting guess `x0` and convergence.
    `dinv` is the inverse diagonal as a device column; `b` and `x0` are host
    arrays, and so is the solution returned."""
    xp = be.xp
    b = np.asarray(b, dtype=np.float64)
    k = b.shape[1]
    bnorm = np.linalg.norm(b, axis=0)
    bnorm[bnorm == 0] = 1.0
    zero = np.zeros(k, bool) if zero is None else np.asarray(zero, bool)
    B = xp.asarray(b)
    if x0 is None:
        X = xp.zeros_like(B)
        R = B.copy()
    else:
        X = xp.asarray(np.asarray(x0, dtype=np.float64))
        R = B - A @ X
    if zero.any():
        cols = xp.asarray(np.flatnonzero(zero))
        X[:, cols] = 0.0
        R[:, cols] = 0.0            # solved already: X stays zero
        if zero.all():
            return be.asnumpy(X), 0, True, 0.0
    active = ~zero
    rel = np.zeros(k)
    if x0 is not None:
        rel = be.asnumpy(xp.sqrt((R * R).sum(axis=0))) / bnorm
        active &= rel > tol
        if not active.any():
            return be.asnumpy(X), 0, True, float(rel.max())
    P = R * dinv
    rz = (R * P).sum(axis=0)
    it = 0
    for it in range(1, maxiter + 1):
        AP = A @ P
        pAp = (P * AP).sum(axis=0)
        alpha = xp.where(xp.asarray(active) & (pAp > 0),
                         rz / xp.where(pAp > 0, pAp, 1.0), 0.0)
        X += alpha * P
        R -= alpha * AP
        del AP
        rel = be.asnumpy(xp.sqrt((R * R).sum(axis=0))) / bnorm
        active = rel > tol
        if debug is not None and (it % 25 == 0 or not active.any()):
            debug(f"  CG iter {it}: max rel. residual {rel.max():.2e}")
        if not active.any():
            break
        Z = R * dinv
        rz_new = (R * Z).sum(axis=0)
        beta = xp.where(xp.asarray(active), rz_new / xp.where(rz != 0, rz, 1.0), 0.0)
        P = Z + beta * P
        rz = rz_new
    return be.asnumpy(X), it, bool(not active.any()), float(rel.max())

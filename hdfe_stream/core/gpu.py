"""Optional GPU execution (experimental): the reduced solve on CUDA through
CuPy (``device="cuda"``), and the fit's Polars queries on the cudf-polars GPU
engine (``polars_engine="gpu"``).

Neither changes what is computed. The solve is the same block PCG as
`_SolveMixin._pcg` with the Jacobi preconditioner, on the same explicit
matrix S: S is built on the CPU as usual, copied to the GPU once, and every
iteration runs there; only the right-hand sides go over and the solutions
come back. Sums are formed in a different order than on the CPU, so the
estimates differ in the last digits, far inside the solver tolerance.

Both packages are imported only when asked for, so nothing here is needed to
use the library without a GPU.
"""

from __future__ import annotations

import importlib.util
from types import SimpleNamespace

import numpy as np

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
    return SimpleNamespace(
        xp=cupy, asnumpy=cupy.asnumpy,
        csr=lambda S: cusparse.csr_matrix(
            (cupy.asarray(S.data), cupy.asarray(S.indices), cupy.asarray(S.indptr)),
            shape=S.shape),
        name=cupy.cuda.runtime.getDeviceProperties(0)["name"].decode())


def check_polars_gpu():
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

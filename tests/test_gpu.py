"""The GPU options, without a GPU.

``device="cuda"`` runs the solve through `hdfe_stream.core.gpu`, whose array
backend is CuPy. Here the backend is replaced by numpy and scipy, which run
the same code, so everything but CuPy itself is exercised: the device PCG
(zero columns, starting guesses, convergence), and how the fit, the GLM and
KSS reach it. On a real GPU the sums are formed in a different order, so the
agreement there is to the solver tolerance; benchmarks/gpu_benchmark.py
checks it on the hardware.

``polars_engine="gpu"`` is checked for its fallback: with an engine that
cannot run anything, every query must be run again on the CPU, recorded, and
give the same fit.
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest
import scipy.sparse as sp
from conftest import rel

import hdfe_stream.core.gpu as gpu
from hdfe_stream import StreamingHDFE, feols_stream, fepois_stream, leave_out_kss
from hdfe_stream.simulate import simulate_akm

FML = "log_earn ~ age_squared + age_cubed | worker_id + firm_id + year"


@pytest.fixture
def numpy_device(monkeypatch):
    """Run device="cuda" on numpy and scipy instead of CuPy."""
    backend = SimpleNamespace(xp=np, asnumpy=np.asarray, name="numpy stand-in",
                              csr=lambda S: sp.csr_matrix(S, copy=True))
    monkeypatch.setattr(gpu, "cuda_backend", lambda: backend)
    return backend


def test_device_fit_matches_the_cpu(akm, tmp_path, numpy_device):
    kw = dict(vcov={"CRV1": "worker_id"}, verbose=False, tol=1e-12)
    cpu = feols_stream(FML, akm.src, workdir=tmp_path / "cpu", **kw)
    dev = feols_stream(FML, akm.src, workdir=tmp_path / "dev", device="cuda", **kw)
    assert dev.solver_info["device"] == "numpy stand-in"
    assert dev.solver_info["solver"] == "explicit"
    assert rel(list(dev.coef().values()), list(cpu.coef().values())) < 1e-9
    assert rel(dev.se, cpu.se) < 1e-8
    fe = [r.fixef("firm_id").sort("firm_id").collect()["fe_firm_id"].to_numpy()
          for r in (cpu, dev)]
    assert np.max(np.abs(fe[0] - fe[1])) < 1e-8


def test_device_pcg_matches_the_cpu_pcg(akm, tmp_path, numpy_device):
    """The two PCGs side by side on the same S, right-hand sides, zero
    columns and starting guess: the same iterations and solutions."""
    est = StreamingHDFE("log_earn", ["age_squared"], ["worker_id", "firm_id", "year"],
                        workdir=tmp_path, verbose=False, keep_intermediates=True,
                        solver="explicit", tol=1e-11)
    est.fit(akm.src)
    est.reload_intermediates()
    S, _ = est._build_explicit()
    rng = np.random.default_rng(3)
    b = S @ rng.standard_normal((S.shape[0], 4))
    zero = np.array([False, True, False, False])
    b[:, 1] = 0.0
    x0 = np.zeros_like(b)
    x0[:, 2] = np.linalg.lstsq(S.toarray(), b[:, 2], rcond=None)[0]   # already solved
    cpu_solve, _ = est._make_block_solver()
    est.device = "cuda"
    dev_solve, info = est._make_block_solver()
    assert info["device"] == "numpy stand-in"
    for start in (None, x0):
        X_cpu, it_cpu, ok_cpu, _ = cpu_solve(b, zero, start)
        X_dev, it_dev, ok_dev, _ = dev_solve(b, zero, start)
        assert ok_cpu and ok_dev
        assert abs(it_cpu - it_dev) <= 1
        assert np.all(X_dev[:, 1] == 0.0)
        assert np.max(np.abs(X_dev - X_cpu)) < 1e-8 * np.max(np.abs(X_cpu))


def test_device_glm_matches_the_cpu(akm, tmp_path, numpy_device):
    frame = akm.frame.with_columns(
        y=pl.Series(np.random.default_rng(1).poisson(
            np.exp(0.2 * akm.frame["age_squared"].to_numpy())).astype(float)))
    path = tmp_path / "glm.parquet"
    frame.write_parquet(path)
    fml = "y ~ age_squared + age_cubed | worker_id + firm_id"
    kw = dict(verbose=False, tol=1e-12, iwls_tol=1e-10)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cpu = fepois_stream(fml, str(path), workdir=tmp_path / "cpu", **kw)
        dev = fepois_stream(fml, str(path), workdir=tmp_path / "dev", device="cuda", **kw)
    assert rel(list(dev.coef().values()), list(cpu.coef().values())) < 1e-7
    assert rel(dev.se, cpu.se) < 1e-6


@pytest.mark.parametrize("leave_out", ["match", "observation"])
def test_device_kss_matches_the_cpu(tmp_path, numpy_device, leave_out):
    path = tmp_path / "panel.parquet"
    simulate_akm(n_workers=400, n_firms=30, seed=7).write_parquet(path)
    kw = dict(n_draws=50, seed=1, verbose=False, leave_out=leave_out, tol=1e-12)
    cpu = leave_out_kss("log_earn ~ 1 | worker_id + firm_id", str(path),
                        workdir=tmp_path / "cpu", **kw)
    dev = leave_out_kss("log_earn ~ 1 | worker_id + firm_id", str(path),
                        workdir=tmp_path / "dev", device="cuda", **kw)
    for key in cpu.leave_out:
        assert dev.leave_out[key] == pytest.approx(cpu.leave_out[key], rel=1e-7, abs=1e-12)


def test_device_options_are_checked(akm, tmp_path, numpy_device):
    with pytest.raises(ValueError, match="device must be"):
        StreamingHDFE("log_earn", [], ["worker_id", "firm_id"], device="gpu")
    with pytest.raises(ValueError, match="explicit solver"):
        StreamingHDFE("log_earn", [], ["worker_id", "firm_id"], device="cuda",
                      solver="stream_cg")
    with pytest.raises(ValueError, match="explicit solver"):
        StreamingHDFE("log_earn", [], ["worker_id", "firm_id"], device="cuda",
                      precond="amg")
    with pytest.raises(ValueError, match="polars_engine must be"):
        StreamingHDFE("log_earn", [], ["worker_id", "firm_id"], polars_engine="cuda")


def test_device_without_cupy_says_what_to_install(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def no_cupy(name, *args, **kwargs):
        if name.split(".")[0] in ("cupy", "cupyx"):
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", no_cupy)
    with pytest.raises(ImportError, match="cupy-cuda13x"):
        StreamingHDFE("log_earn", [], ["worker_id", "firm_id"], device="cuda")


def test_device_falls_back_to_the_cpu_when_s_is_too_large(akm, tmp_path, numpy_device):
    fit = feols_stream(FML, akm.src, workdir=tmp_path, verbose=False, device="cuda",
                       max_memory_gb=0.01)
    assert fit.solver_info["solver"] == "stream_cg"
    assert fit.solver_info["device"] == "cpu"


def test_polars_gpu_engine_falls_back_query_by_query(akm, tmp_path, monkeypatch):
    """An engine that runs nothing: every query is run again on the CPU and
    recorded, and the fit is the CPU's."""
    monkeypatch.setattr(gpu, "check_polars_gpu", lambda: None)

    def refuse(self):
        raise NotImplementedError("this engine runs nothing")
    monkeypatch.setattr(StreamingHDFE, "_gpu_engine", refuse)
    kw = dict(vcov={"CRV1": "worker_id"}, verbose=False)
    cpu = feols_stream(FML, akm.src, workdir=tmp_path / "cpu", **kw)
    dev = feols_stream(FML, akm.src, workdir=tmp_path / "dev", polars_engine="gpu", **kw)
    # the same queries on the same engine; rows tied in pass 0's sort can come
    # out in another order, as between any two fits, hence not exact
    assert rel(list(dev.coef().values()), list(cpu.coef().values())) < 1e-10
    queries = {f["query"] for f in dev.diagnostics["polars_gpu_fallbacks"]}
    assert {"pass 0 scan", "pass 0 partition", "pass 0 sort"} <= queries
    assert any(q.startswith("pass 0 factorize") for q in queries)
    assert all(f["reason"] == "this engine runs nothing"
               for f in dev.diagnostics["polars_gpu_fallbacks"])
    assert "polars_gpu_fallbacks" not in cpu.diagnostics


def test_polars_gpu_engine_without_cudf_polars_says_what_to_install(monkeypatch):
    import importlib.util
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name, *a: None if name == "cudf_polars" else real(name, *a))
    with pytest.raises(ImportError, match="cudf-polars"):
        StreamingHDFE("log_earn", [], ["worker_id", "firm_id"], polars_engine="gpu")


# ------------------------------------------------ fits without fixed effects

def _no_fe_frame(n=3000):
    rng = np.random.default_rng(11)
    x1, x2 = rng.standard_normal(n), rng.standard_normal(n)
    eta = 0.3 * x1 - 0.5 * x2 + 0.1
    return pl.DataFrame({
        "x1": x1, "x2": x2, "w": rng.uniform(0.5, 2.0, n),
        "yb": (rng.uniform(size=n) < 1 / (1 + np.exp(-eta))).astype(float),
        "yp": rng.poisson(np.exp(0.2 * eta)).astype(float)})


@pytest.mark.parametrize("fml, fit_fn, kw", [
    ("yb ~ x1 + x2", "feglm_stream", dict(family="logit")),
    ("yb ~ x1 + x2", "feglm_stream", dict(family="probit")),
    ("yb ~ x1 + x2", "feglm_stream", dict(family="logit", weights="w")),
    ("yp ~ x1 + x2", "fepois_stream", {}),
], ids=["logit", "probit", "weighted-logit", "poisson"])
def test_device_glm_without_fixed_effects(tmp_path, numpy_device, fml, fit_fn, kw):
    import hdfe_stream
    path = tmp_path / "d.parquet"
    _no_fe_frame().write_parquet(path)
    fn = getattr(hdfe_stream, fit_fn)
    opts = dict(verbose=False, iwls_tol=1e-12, batch_rows=700, **kw)
    cpu = fn(fml, str(path), workdir=tmp_path / "cpu", **opts)
    dev = fn(fml, str(path), workdir=tmp_path / "dev", device="cuda", **opts)
    assert rel(list(dev.coef().values()), list(cpu.coef().values())) < 1e-9
    assert rel(dev.se, cpu.se) < 1e-8
    assert rel(dev.deviance, cpu.deviance) < 1e-10


def test_device_glm_rows_that_do_not_fit_are_streamed(tmp_path, numpy_device):
    import hdfe_stream
    path = tmp_path / "d.parquet"
    _no_fe_frame().write_parquet(path)
    opts = dict(verbose=False, iwls_tol=1e-12, batch_rows=700)
    cpu = hdfe_stream.feglm_stream("yb ~ x1 + x2", str(path), family="logit",
                                   workdir=tmp_path / "cpu", **opts)
    numpy_device.free_bytes = lambda: 10_000.0      # far less than the rows
    with pytest.warns(UserWarning, match="do not fit"):
        dev = hdfe_stream.feglm_stream("yb ~ x1 + x2", str(path), family="logit",
                                       workdir=tmp_path / "dev", device="cuda", **opts)
    assert rel(list(dev.coef().values()), list(cpu.coef().values())) < 1e-9

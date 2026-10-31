"""The guards that choose between a fast path and a smaller one follow the
memory *headroom* (the limit, less a reserve, less what the process holds),
not a fixed share of the machine; `max_memory_gb` sets the limit."""

from __future__ import annotations

import pytest

from conftest import TOL_BETA, rel

from hdfe_stream import StreamingHDFE, feols_stream

FML = "log_earn ~ age_squared + age_cubed | worker_id + firm_id + year"


def test_a_generous_limit_builds_s_and_a_tiny_one_streams(akm, workdir):
    big = feols_stream(FML, akm.src, workdir=workdir / "b", verbose=False, max_memory_gb=500)
    tiny = feols_stream(FML, akm.src, workdir=workdir / "t", verbose=False, max_memory_gb=0.05)
    assert big.solver_info["solver"] == "explicit"
    assert tiny.solver_info["solver"] == "stream_cg"
    assert rel(tiny.beta, big.beta) < TOL_BETA


def test_the_fixed_cap_is_gone(akm, workdir):
    with pytest.raises(TypeError, match="max_s_gb"):
        feols_stream(FML, akm.src, workdir=workdir, verbose=False, max_s_gb=1.0)


def test_the_reserve_moves_the_line(workdir):
    est = StreamingHDFE("y", ["x"], ["worker_id", "firm_id"], workdir=workdir,
                        max_memory_gb=100, memory_reserve=0.0)
    more = est._s_limit_gb()
    est2 = StreamingHDFE("y", ["x"], ["worker_id", "firm_id"], workdir=workdir,
                         max_memory_gb=100, memory_reserve=0.5)
    assert more > est2._s_limit_gb() + 30


def test_copies_follow_the_headroom(workdir):
    est = StreamingHDFE("y", ["x"], ["worker_id", "firm_id"], workdir=workdir, max_memory_gb=0.2)
    assert est._acc_copies(10**8, 8) == 1            # nothing left: one copy


def test_crv3_is_refused_up_front_when_it_does_not_fit(workdir):
    import numpy as np
    import polars as pl
    rng = np.random.default_rng(0)
    n = 20_000
    w = rng.integers(0, 2000, n)
    df = pl.DataFrame({"w": w, "st": w % 40, "x": rng.normal(size=n)}).with_columns(
        y=pl.col("x") + rng.normal(size=n))
    ok = feols_stream("y ~ x | w", df, workdir=workdir, verbose=False, vcov={"CRV3": "st"},
                      max_memory_gb=500)
    assert ok.se[0] > 0
    with pytest.raises(MemoryError, match="max_memory_gb"):
        feols_stream("y ~ x | w", df, workdir=workdir, verbose=False, vcov={"CRV3": "st"},
                     max_memory_gb=0.01)

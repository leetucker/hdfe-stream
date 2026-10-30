"""Polars' 64-bit runtime (issue #5): row indices are UInt64, not UInt32.

`with_row_index` returns polars' index type, which is UInt32 on the default
runtime and UInt64 with `polars-runtime-64`, whatever the size of the frame.
The library codes levels and numbers rows with it, and NumPy and numba take
UInt64 codes badly (no safe cast to int64; mixed with an int64 offset, numba
types the sum as float64). The runtime is chosen at install time, so a test
cannot switch it; instead `_widen_row_index` makes `with_row_index` return UInt64
on any install, which is what the 64-bit runtime does. With that runtime
installed it changes nothing, so this file also passes there.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from hdfe_stream import StreamingHDFE

X = ["age_squared", "age_cubed"]
FE = ["worker_id", "firm_id", "year"]


def _widen_row_index(monkeypatch):
    """`with_row_index` returns UInt64 from here on, as on the 64-bit runtime."""
    for cls in (pl.DataFrame, pl.LazyFrame):
        original = cls.with_row_index

        def with_row_index(self, name="index", offset=None, *, _original=original):
            out = _original(self, name, offset)
            return out.with_columns(pl.col(name).cast(pl.UInt64))

        monkeypatch.setattr(cls, "with_row_index", with_row_index)
    assert pl.DataFrame({"a": [1]}).with_row_index().schema["index"] == pl.UInt64


def _fit(panel, workdir, solver):
    est = StreamingHDFE("log_earn", X, FE, workdir=workdir, solver=solver,
                        tol=1e-11, verbose=False, batch_rows=5_000, n_buckets=3,
                        outputs="keep")
    return est.fit(panel.src, vcov="iid", cluster=("firm_id",))


@pytest.mark.parametrize("solver", ["explicit", "stream_cg"])
def test_fit_with_uint64_row_index(akm, tmp_path, monkeypatch, solver):
    """A fit with UInt64 row indices equals the fit on the installed runtime."""
    native = _fit(akm, tmp_path / "native", solver)
    with monkeypatch.context() as m:
        _widen_row_index(m)
        wide = _fit(akm, tmp_path / "wide", solver)
    try:
        assert wide.coefnames == native.coefnames
        np.testing.assert_allclose(wide.beta, native.beta, rtol=1e-10)
        for spec in ("iid", {"CRV1": "firm_id"}):
            np.testing.assert_allclose(wide.with_vcov(spec).se,
                                       native.with_vcov(spec).se, rtol=1e-10)
        assert wide.n_obs == native.n_obs
        got, want = (r.resid().sort("row_id").collect() for r in (wide, native))
        assert got["row_id"].to_list() == want["row_id"].to_list()
        np.testing.assert_allclose(got["resid"].to_numpy(), want["resid"].to_numpy(),
                                   atol=1e-9)
    finally:
        native.cleanup()
        wide.cleanup()

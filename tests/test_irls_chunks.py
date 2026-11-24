"""IRLS's chunk arrays have one numba signature whatever the chunk.

A column comes from the row files zero-copy (read-only) or copied (writable),
depending on how it was stored. numba compiles a kernel once per signature,
and read-only is part of it, so a fit whose chunks came the other way from the
ones the cache held compiled the row kernels again inside the fit."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from hdfe_stream.core._columns import vcol
from hdfe_stream.core.irls import _OFFSET, _IRLSMixin


@pytest.mark.parametrize("readonly", [False, True])
def test_chunk_arrays_are_read_only(readonly):
    n = 50
    r = np.random.default_rng(0)
    ch = {vcol(0): r.normal(size=n), vcol(1): r.normal(size=n), vcol(2): r.normal(size=n)}
    for a in ch.values():
        a.flags.writeable = not readonly
    est = SimpleNamespace(weights=None, offset_name="lexp", vidx={_OFFSET: 2}, _wscale=1.0)
    y, X, w, off = _IRLSMixin._chunk_data(est, ch, 0, [1])
    assert not y.flags.writeable
    assert not off.flags.writeable
    np.testing.assert_array_equal(y, ch[vcol(0)])
    np.testing.assert_array_equal(off, ch[vcol(2)])
    np.testing.assert_array_equal(X[:, 0], ch[vcol(1)])
    assert w is None

"""The fits use stream-limits: n_threads is scoped to the fit, and peak memory is
recorded. (The limits themselves are tested in the stream-limits package.)"""

from __future__ import annotations

import numpy as np
import polars as pl
import stream_limits


def test_fit_reports_peak_rss_and_restores_threads(tmp_path):
    import numba
    import polars as pl
    from hdfe_stream import feols_stream
    rng = __import__("numpy").random.default_rng(0)
    n = 5000
    df = pl.DataFrame({"y": rng.normal(size=n), "x": rng.normal(size=n),
                       "g": rng.integers(0, 50, n), "h": rng.integers(0, 20, n)})
    before = numba.get_num_threads()
    fit = feols_stream("y ~ x | g + h", df, workdir=tmp_path, n_threads=1, verbose=False)
    if stream_limits.peak_rss_gb() is not None:      # not readable on Windows
        assert fit.diagnostics["peak_rss_gb"] > 0
    else:
        assert fit.diagnostics["peak_rss_gb"] is None
    assert numba.get_num_threads() == before

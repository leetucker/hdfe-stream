"""Pass 0's row restrictions on a narrow copy of the rows.

For a wide design the restrictions (connected set, singletons, separation)
read a narrow file of the source columns and the design variables they need,
written a batch at a time, rather than a query over the evaluated design. The
rows they drop, and so the fit, must be the same either way; here the narrow
path is forced by making Polars' morsels look enormous.
"""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl
import pytest

from hdfe_stream import feglm_stream, feols_stream
from hdfe_stream.core import passes


@pytest.fixture(scope="module")
def panel(tmp_path_factory):
    """Singleton workers and firms (and a chain found only in a second
    round), a categorical for a row-wise design, and a binary outcome with a
    firm whose outcome never varies."""
    rng = np.random.default_rng(11)
    n = 3000
    w = rng.integers(0, 400, n)
    f = rng.integers(0, 40, n)
    w[:30] = 10_000 + np.arange(30)
    f[30:33] = 1000 + np.arange(3)
    w[33], f[33] = 20_000, 2000
    w[34], f[34] = 20_000, 2001
    w[35], f[35] = 20_001, 2001
    x = rng.normal(size=n)
    q = rng.integers(0, 5, n)
    yb = (rng.random(n) < 1 / (1 + np.exp(-x))).astype(float)
    yb[f == 7] = 1.0                                   # firm 7: separated
    frame = pl.DataFrame({"y": x + 0.3 * q + rng.normal(size=n) + 0.1 * (w % 7), "x": x,
                          "q": q, "w": w, "f": f, "yb": yb})
    path = tmp_path_factory.mktemp("restrict") / "d.parquet"
    frame.write_parquet(path)
    return str(path)


def _both(monkeypatch, fit):
    """The fit by the direct query, and by the narrow copy (forced)."""
    from hdfe_stream.core.estimator import _StreamingBase
    calls = []
    real = _StreamingBase._restriction_rows

    def spy(self, *a, **k):
        out = real(self, *a, **k)
        calls.append(out is not None)
        return out

    monkeypatch.setattr(_StreamingBase, "_restriction_rows", spy)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        direct = fit()
        monkeypatch.setattr(passes, "_MORSEL_ROWS", 10**15)
        narrow = fit()
    assert calls == [False, True]
    return direct, narrow


@pytest.mark.parametrize("options", [{}, {"connected": "largest"}])
def test_ols_singletons_and_connected_set_are_the_same(panel, monkeypatch, options):
    direct, narrow = _both(monkeypatch, lambda: feols_stream(
        "y ~ x + i(q) | w + f", panel, vcov="hetero", verbose=False, **options))
    assert narrow.n_obs == direct.n_obs
    assert narrow.diagnostics["singletons"] == direct.diagnostics["singletons"]
    assert direct.diagnostics["singletons"]["observations"] > 0
    np.testing.assert_allclose(narrow.beta, direct.beta, rtol=1e-12)
    np.testing.assert_allclose(narrow.se, direct.se, rtol=1e-10)


def test_glm_separation_is_the_same(panel, monkeypatch):
    direct, narrow = _both(monkeypatch, lambda: feglm_stream(
        "yb ~ x + i(q) | w + f", panel, family="logit", fixef_rm="none", verbose=False))
    sep = direct.diagnostics["separation"]
    assert sep["observations"] > 0
    assert narrow.diagnostics["separation"] == sep
    assert narrow.n_obs == direct.n_obs
    np.testing.assert_allclose(narrow.beta, direct.beta, rtol=1e-9)

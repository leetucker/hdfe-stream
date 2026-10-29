"""`_stack`: the same array from either copy, and the blocked one (whose typed
list costs about 0.2 s of compilation in each process) only where it pays."""

import numpy as np
import pytest

from hdfe_stream.core import inference


@pytest.fixture(autouse=True)
def fresh_state(monkeypatch):
    monkeypatch.setattr(inference, "_missed_saving", 0.0)
    monkeypatch.setattr(inference, "_use_blocked", False)


def _chunk(n, k, seed=0):
    rng = np.random.default_rng(seed)
    cols = [f"v{j}" for j in range(k)]
    chunk = {c: rng.normal(size=n) for c in cols}
    chunk[cols[0]].flags.writeable = False           # Arrow's columns are read-only
    return chunk, cols


@pytest.mark.parametrize("n, k", [(10, 3), (70_000, 1), (70_000, 3), (50_001, 40)])
def test_same_array_either_way(n, k, monkeypatch):
    chunk, cols = _chunk(n, k)
    expected = np.column_stack([chunk[c] for c in cols])
    monkeypatch.setattr(inference, "_use_blocked", True)
    monkeypatch.setattr(inference, "_missed_saving", 1e9)
    np.testing.assert_array_equal(inference._stack(chunk, cols), expected)
    monkeypatch.setattr(inference, "_use_blocked", False)
    monkeypatch.setattr(inference, "_missed_saving", -1e9)
    out = inference._stack(chunk, cols)
    np.testing.assert_array_equal(out, expected)
    assert out.flags.c_contiguous


def test_narrow_stacks_never_switch():
    """AKM's three columns over 42 million rows saves about 0.1 s: not worth 0.2."""
    chunk, cols = _chunk(2_000_000, 3)
    for _ in range(21):
        inference._stack(chunk, cols)
    assert not inference._use_blocked
    chunk, cols = _chunk(1_000_000, 1)
    for _ in range(100):
        inference._stack(chunk, cols)
    assert not inference._use_blocked


def test_wide_stacks_switch_in_the_first_chunks():
    chunk, cols = _chunk(200_000, 39)
    inference._stack(chunk, cols)                    # 7.8M elements: 0.03 s
    assert not inference._use_blocked
    for _ in range(6):
        out = inference._stack(chunk, cols)
    assert inference._use_blocked
    np.testing.assert_array_equal(out, np.column_stack([chunk[c] for c in cols]))

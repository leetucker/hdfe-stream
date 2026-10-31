"""`profile=True`: the estimate is returned instead of a fit, it needs no run
directory, and it brackets what a real fit uses."""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import polars as pl
import pytest
import stream_limits

from hdfe_stream import Profile, feglm_stream, feols_stream, fepois_stream
from hdfe_stream.core.profile import Component, Request

# These tests compare the profile with a measured peak, so they need a peak-memory reading (Linux,
# macOS and, from stream-limits 0.1.1, Windows). The model was calibrated on Linux and checked on
# macOS; on Windows a failure here is information about the model, not a broken test.
needs_peak_rss = pytest.mark.skipif(
    stream_limits.peak_rss_gb() is None, reason="peak memory cannot be read on this platform")


@pytest.fixture(scope="module")
def panel(tmp_path_factory):
    rng = np.random.default_rng(1)
    n = 200_000
    df = pl.DataFrame({
        "w": rng.integers(0, n // 8, n), "f": rng.integers(0, 400, n),
        "yr": rng.integers(2000, 2010, n), "st": rng.integers(0, 30, n),
        "x1": rng.normal(size=n), "x2": rng.normal(size=n), "z": rng.normal(size=n)})
    df = df.with_columns(y=pl.col("x1") + rng.normal(size=n), cnt=rng.poisson(2.0, n),
                         d=(rng.random(n) < 0.4).astype("int8"))
    path = tmp_path_factory.mktemp("prof") / "panel.parquet"
    df.write_parquet(path, row_group_size=50_000)
    return str(path)


def test_returns_a_profile_and_leaves_no_run_directory(panel, tmp_path):
    p = feols_stream("y ~ x1 + x2 | w + f", panel, workdir=tmp_path, profile=True, verbose=False)
    assert isinstance(p, Profile)
    assert list(tmp_path.iterdir()) == []
    assert p.scan.n_rows == 200_000 and p.notes["streamed"] == "w"
    lo, hi = p.memory_gb
    assert 0 < lo <= hi
    assert p.disk_gb[0] <= p.disk_gb[1]


def test_request_and_report(panel):
    p = feols_stream("y ~ x1 | w + f", panel, profile=True, verbose=False)
    r = p.recommend()
    assert isinstance(r, Request) and r.mem_gb >= 1 and r.ncpus == p.threads
    assert r.pbs.startswith("select=1:ncpus=") and "gb" in r.pbs
    text = str(p)
    assert "memory" in text and "components" in text and "pass 0: sort one bucket" in text
    d = p.to_dict()
    json.dumps(d)                                   # serializable
    assert d["memory_gb"]["upper"] >= d["memory_gb"]["expected"]
    assert p.recommend(safety=100).mem_gb <= int(np.ceil(p.memory_gb[1]))


def test_every_estimator_profiles(panel):
    for p in (feols_stream("y ~ x1 | w + f | x2 ~ z", panel, profile=True, verbose=False),
              feols_stream("y ~ x1 + x2", panel, profile=True, verbose=False),
              feols_stream("y ~ x1 | w", panel, profile=True, verbose=False, vcov={"CRV1": "st"}),
              fepois_stream("cnt ~ x1 | w + f", panel, profile=True, verbose=False),
              feglm_stream("d ~ x1 | w + f", panel, "logit", profile=True, verbose=False)):
        assert isinstance(p, Profile) and p.memory_gb[1] > 0


def test_clusters_add_their_arrays(panel):
    plain = feols_stream("y ~ x1 | w + f", panel, profile=True, verbose=False)
    clustered = feols_stream("y ~ x1 | w + f", panel, vcov={"CRV1": "st"}, profile=True,
                             verbose=False)
    names = [c.name for c in clustered.components]
    assert any(n.startswith("cluster scores: st") for n in names)
    assert not any(c.name.startswith("cluster scores") for c in plain.components)


def test_several_fixed_effect_sets_combine(panel):
    one = feols_stream("y ~ x1 | w + f", panel, profile=True, verbose=False)
    both = feols_stream("y ~ x1 | w + sw(f, yr)", panel, profile=True, verbose=False)
    assert len(both.notes["estimators"]) == 2
    # the fits run one after another: memory is the larger one's, disk adds up
    assert both.memory_gb[1] >= one.memory_gb[1] - 0.05
    assert both.disk_gb[1] > one.disk_gb[1]


def test_options_are_validated_like_a_fit(panel):
    with pytest.raises(ValueError, match="solver"):
        feols_stream("y ~ x1 | w + f", panel, profile=True, solver="bogus", verbose=False)
    with pytest.raises(ValueError):
        feols_stream("y ~ nothere | w + f", panel, profile=True, verbose=False)


def test_a_frame_source_works(panel):
    df = pl.read_parquet(panel)
    p = feols_stream("y ~ x1 | w + f", df, profile=True, verbose=False)
    assert p.scan.n_rows == len(df)


def test_phases_take_the_largest_not_the_sum():
    comps = [Component("a", 1e9, phases=("p1",)), Component("b", 2e9, phases=("p2",)),
             Component("c", 0.5e9), Component("d", 3e9, where="disk")]
    p = Profile(model="x", components=comps, baseline_gb=0.25, threads=1)
    assert p.memory_gb[0] == pytest.approx(0.25 + 2.5)
    assert p.disk_gb[0] == pytest.approx(3.0)


MEASURE = r'''
import json, sys
import hdfe_stream as h
fn, fml, kw, path = sys.argv[1], sys.argv[2], json.loads(sys.argv[3]), sys.argv[4]
f = getattr(h, fn)
p = f(fml, path, profile=True, verbose=False, **kw)
r = f(fml, path, verbose=False, **kw)
d = r.diagnostics
print(json.dumps({"mem": p.memory_gb, "disk": p.disk_gb, "rss": d["peak_rss_gb"],
                  "disk_peak": d["disk_peak_gb"]}))
'''


@needs_peak_rss
@pytest.mark.parametrize("fn,fml,kw", [
    ("feols_stream", "y ~ x1 + x2 | w + f + yr", {}),
    ("feols_stream", "y ~ x1 | w + f", {"vcov": {"CRV1": "st"}}),
    ("feols_stream", "y ~ x1 + x2", {}),
    ("fepois_stream", "cnt ~ x1 | w + f", {}),
    ("feglm_stream", "d ~ x1 | w + f", {"family": "logit"}),
])
def test_the_upper_bound_holds_and_the_expected_is_not_wild(panel, fn, fml, kw):
    out = subprocess.run([sys.executable, "-c", MEASURE, fn, fml, json.dumps(kw), panel],
                         capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr[-500:]
    r = json.loads(out.stdout.strip().splitlines()[-1])
    assert r["rss"] <= r["mem"][1]
    assert r["rss"] >= 0.4 * r["mem"][0]            # not an order of magnitude too high
    assert r["disk_peak"] <= r["disk"][1] + 0.01

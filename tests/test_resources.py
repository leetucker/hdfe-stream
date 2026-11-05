"""Memory tests: the streamed fits stay inside a memory limit on data that
would not fit in it.

Run with ``pytest -m resource`` (the default run leaves them out: they take
about a minute). Each fit runs in a child process under stream-limits'
``resource_runner``: a watchdog on every platform, or a kernel limit when
``STREAM_LIMITS_TEST_CGROUP_PARENT`` names a cgroup set up for it.

The panel is 10.2 million rows, 0.51 GB in memory. With ``max_memory_gb=0.5``
(a job given little memory) a fit holds about 1.1-1.3 GB of private memory
(not counting its memory-mapped scratch files, which the kernel can reclaim),
most of it buffers whose sizes are capped, and the same at 20 million rows.
The limit, 1.6 GB, leaves 0.3 GB for platform differences; a fit that held
the panel in memory would need at least 1.8 GB. Measured on Linux (WSL2,
2 threads): OLS 1.30 GB, logit 1.24 GB.

The limit is on private memory, which stream-limits measures on Linux and
Windows. On macOS it measures resident memory, which counts the mapped
scratch files (0.6 GB more here), so there the limit is a looser safety net.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import polars as pl
import pytest

from hdfe_stream.simulate import simulate_akm

pytestmark = pytest.mark.resource

ROWS_WORKERS, ROWS_YEARS = 300_000, 40          # 10.2 million rows
BUDGET_GB = 0.5                                  # the fit's max_memory_gb
LIMIT_GB = 1.6 if sys.platform != "darwin" else 2.6
FML = "~ age_squared + age_cubed | worker_id + firm_id + year"


@pytest.fixture(scope="module")
def big_panel(tmp_path_factory):
    """Workers and firms fixed, many years: rows grow, levels do not."""
    path = tmp_path_factory.mktemp("big") / "panel.parquet"
    d = simulate_akm(n_workers=ROWS_WORKERS, n_firms=20_000,
                     years=range(1980, 1980 + ROWS_YEARS), seed=1)
    d = d.with_columns(high=(pl.col("log_earn") > pl.col("log_earn").median())
                       .cast(pl.Int8))
    d.write_parquet(path)
    assert d.estimated_size() > 0.5e9
    return path


def _fit_in_child(resource_runner, call: str, path, workdir):
    code = f"""
import json
from hdfe_stream import feglm_stream, feols_stream
r = {call}
print(json.dumps(dict(beta=r.beta.tolist(), solver=r.solver_info.get("solver"))))
"""
    code = code.replace("SRC", repr(str(path))).replace("WORK", repr(str(workdir)))
    result = resource_runner([sys.executable, "-c", code], memory_gb=LIMIT_GB,
                             threads=2, timeout=900)
    report = {k: result.as_dict()[k] for k in ("status", "mode", "memory_measure",
                                               "peak_private_bytes", "peak_rss_bytes",
                                               "elapsed_s")}
    print(json.dumps(report))
    assert result.status == "ok", (report, result.stderr[-3000:])
    if result.kernel:
        assert result.kernel["events"].get("oom_kill", 0) == 0
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_ols_streams_inside_the_limit(resource_runner, big_panel, tmp_path):
    out = _fit_in_child(
        resource_runner,
        f"feols_stream('log_earn {FML}', SRC, workdir=WORK, verbose=False, "
        f"max_memory_gb={BUDGET_GB})",
        big_panel, tmp_path / "work")
    assert out["solver"] == "stream_cg"
    # the simulated age profile (hdfe_stream.simulate): a sanity check only;
    # agreement with pyfixest is the rest of the suite's job
    assert np.all(np.isfinite(out["beta"])) and out["beta"][0] > 0 > out["beta"][1]


def test_logit_streams_inside_the_limit(resource_runner, big_panel, tmp_path):
    out = _fit_in_child(
        resource_runner,
        f"feglm_stream('high {FML}', SRC, family='logit', workdir=WORK, "
        f"verbose=False, max_memory_gb={BUDGET_GB})",
        big_panel, tmp_path / "work")
    assert np.all(np.isfinite(out["beta"])) and out["beta"][0] > 0 > out["beta"][1]

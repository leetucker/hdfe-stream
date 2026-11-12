"""The refit bootstrap's shared machinery: whole-group resampling (units and clusters),
worker processes, the refit runner and the result's intervals."""

import sys
import time
from concurrent.futures import as_completed

import numpy as np
import polars as pl
import pytest
from numpy.testing import assert_allclose
from scipy.stats import norm
from stream_limits import MemoryBudget

from hdfe_stream.core.partition import whole_group_batches
from hdfe_stream.core.process import ProcessBatch
from hdfe_stream.core.refit import RefitBootstrapResult, refit_draws
from hdfe_stream.core.resample import DiskGroupResampler


def test_whole_group_batches_cross_reader_boundaries(tmp_path):
    frame = pl.DataFrame(
        {"unit": [0] * 7 + [1] * 2 + [2] * 9 + [3], "value": list(range(19))}
    )
    path = tmp_path / "groups.parquet"
    frame.write_parquet(path, row_group_size=3)
    batches = list(whole_group_batches(path, "unit", 2, read_rows=4))
    assert pl.concat(batches).equals(frame)
    assert sum(b["unit"].n_unique() for b in batches) == 4
    assert all(b["unit"].n_unique() <= 2 for b in batches)


def test_unit_resamples_copy_whole_histories_and_rename_duplicates(tmp_path):
    data = pl.DataFrame(
        {
            "id": ["a", "a", "b", "c", "c", "c"],
            "time": [1, 3, 2, 1, 2, 3],
            "value": np.arange(6),
        }
    )
    raw = tmp_path / "input.parquet"
    data.write_parquet(raw)
    root = tmp_path / "stage"
    root.mkdir()
    sampler = DiskGroupResampler(
        raw,
        root,
        MemoryBudget(None),
        1,
        key="id",
        sort_columns=["id", "time"],
        select=lambda f: f,
        index_column="source_index",
        draw_column="boot",
    )
    assert sampler.n_units == 3 and sampler.n_rows == 6
    expected = np.random.default_rng(4).integers(0, 3, size=3, dtype=np.int64)
    path, dates, n, _ = sampler.materialize(
        np.random.default_rng(4), tmp_path / "draw", time_column="time", chunk_rows=2
    )
    result = pl.read_parquet(path)
    assert result["boot"].n_unique() == 3
    assert result["id"].n_unique() < 3  # this seed repeats a source unit
    for j, i in enumerate(expected):
        assert (
            result.filter(pl.col("boot") == j)
            .drop("boot")
            .sort("time")
            .equals(data.filter(pl.col("id") == ["a", "b", "c"][i]).sort("time"))
        )
    assert n == len(result) and dates == sorted(result["time"].unique().to_list())


def test_cluster_resamples_copy_whole_clusters_with_new_unit_ids(tmp_path):
    """With within=, clusters are drawn whole, and each copy's units get IDs of their own."""
    data = pl.DataFrame(
        {
            "state": ["x", "x", "x", "x", "y", "y", "z"],
            "id": [1, 1, 2, 2, 3, 3, 4],
            "time": [1, 2, 1, 2, 1, 2, 1],
            "value": np.arange(7.0),
        }
    )
    raw = tmp_path / "input.parquet"
    data.write_parquet(raw)
    root = tmp_path / "stage"
    root.mkdir()
    sampler = DiskGroupResampler(
        raw,
        root,
        MemoryBudget(None),
        1,
        key="state",
        sort_columns=["state", "id", "time"],
        select=lambda f: f,
        index_column="source_index",
        draw_column="boot",
        within="id",
    )
    assert (sampler.n_units, sampler.n_within, sampler.max_within) == (3, 4, 2)
    expected = np.random.default_rng(1).integers(0, 3, size=3, dtype=np.int64)
    path, _, n, _ = sampler.materialize(
        np.random.default_rng(1), tmp_path / "draw", time_column=None
    )
    result = pl.read_parquet(path)
    assert n == len(result)
    states = ["x", "y", "z"]
    for copy, cluster in enumerate(expected):
        rows = result.filter(pl.col("boot") // 2 == copy)
        source = data.filter(pl.col("state") == states[cluster])
        # the copy is the whole cluster, each unit with its own ID in this copy
        assert rows.drop("boot").sort("id", "time").equals(source.sort("id", "time"))
        assert rows.group_by("boot").agg(pl.col("id").n_unique())["id"].max() == 1
        assert rows["boot"].n_unique() == source["id"].n_unique()
    # a cluster drawn twice gives distinct unit IDs in each copy
    assert result.group_by("boot").agg(pl.len()).height == sum(
        data.filter(pl.col("state") == states[c])["id"].n_unique() for c in expected
    )


def test_pointwise_interval_definitions_and_no_implicit_inference():
    draws = np.array([[0.0, 1.0], [1.0, 3.0], [4.0, 5.0], [9.0, 9.0], [16.0, 11.0]])
    r = RefitBootstrapResult(
        fml="y~1",
        estimator="test",
        coefnames=["a", "b"],
        beta=np.array([3.0, 4.0]),
        diagnostics={},
        _draws=draws,
        _failures=[],
        _warnings=[],
    )
    q = np.quantile(draws, [0.1, 0.9], axis=0, method="linear")
    for method in ["percentile", "basic", "normal"]:
        tab = r.confint(0.2, method=method)
        expected = (
            q
            if method == "percentile"
            else np.array([2 * r.beta - q[1], 2 * r.beta - q[0]])
            if method == "basic"
            else np.array(
                [
                    r.beta - norm.ppf(0.9) * draws.std(axis=0, ddof=1),
                    r.beta + norm.ppf(0.9) * draws.std(axis=0, ddof=1),
                ]
            )
        )
        assert_allclose(tab["Lower"], expected[0])
        assert_allclose(tab["Upper"], expected[1])
    for method in ["bca", "studentized"]:
        with pytest.raises(ValueError, match="method"):
            r.confint(method=method)


def mean_fit(task):
    """A refit for the tests: the mean of y over the draw's units (fails on request)."""
    frame = pl.read_parquet(task["path"])
    if task.get("fail_below") is not None and frame["y"].mean() < task["fail_below"]:
        raise ValueError("draw mean too small")
    if task.get("crash"):
        raise KeyError("programming error")
    if task.get("report_limits"):
        return np.array([float(task["threads"]), float(task["memory_gb"] or 0.0)])
    if task.get("die") and task["draw"] == 3:
        import os

        os._exit(7)  # a worker that dies partway through the run
    per_unit = frame.group_by(task["unit_column"]).agg(pl.col("y").mean())
    return np.array([per_unit["y"].mean(), float(frame.height)])


def _source(tmp_path):
    rng = np.random.default_rng(0)
    frame = pl.DataFrame(
        {
            "id": np.repeat(np.arange(40), 3),
            "state": np.repeat(np.arange(40) // 5, 3),
            "time": np.tile([1, 2, 3], 40),
            "y": rng.normal(0, 1, 120),
        }
    )
    path = tmp_path / "source.parquet"
    frame.write_parquet(path)
    return frame, path


def _run(tmp_path, name, **options):
    _, path = _source(tmp_path)
    scratch = tmp_path / name
    scratch.mkdir()
    defaults = {
        "fit": mean_fit,
        "task": {},
        "width": 2,
        "reps": 6,
        "seed": 3,
        "budget": MemoryBudget(None),
        "preprocess_mb": 4,
        "key": "id",
        "sort_columns": ["id", "time"],
        "time_column": "time",
        "expected_groups": 40,
        "expected_rows": 120,
        "threads": 2,
    }
    return refit_draws(path, scratch, **(defaults | options))


def test_refits_are_reproducible_and_the_same_in_workers(tmp_path):
    serial = _run(tmp_path, "serial")
    again = _run(tmp_path, "again")
    parallel = _run(tmp_path, "parallel", workers=2, worker_minimum_gb=0.05)
    assert_allclose(again.draws, serial.draws, rtol=1e-13)
    assert_allclose(parallel.draws, serial.draws, rtol=1e-13)
    assert (
        parallel.diagnostics["draw_plan_sha256"]
        == serial.diagnostics["draw_plan_sha256"]
    )
    assert (
        parallel.diagnostics["unit_order_sha256"]
        == serial.diagnostics["unit_order_sha256"]
    )
    # two persistent workers served all six draws
    resources = parallel.diagnostics["worker_resources"]
    assert parallel.diagnostics["persistent_workers"] is True
    assert parallel.diagnostics["parallel_batches"][0]["worker_processes"] == 2
    assert len({r["pid"] for r in resources}) == 2 and len(resources) == 6
    assert all(r["polars_threads"] == 1 for r in resources)
    # one process per draw, and workers restarted after two draws: the same draws
    fresh = _run(
        tmp_path, "fresh", workers=2, worker_minimum_gb=0.05, persistent_workers=False
    )
    assert_allclose(fresh.draws, serial.draws, rtol=1e-13)
    assert [b["refits"] for b in fresh.diagnostics["parallel_batches"]] == [2, 2, 2]
    assert len({r["pid"] for r in fresh.diagnostics["worker_resources"]}) == 6
    restarted = _run(
        tmp_path, "restarted", workers=2, worker_minimum_gb=0.05, worker_draws=2
    )
    assert_allclose(restarted.draws, serial.draws, rtol=1e-13)
    assert restarted.diagnostics["parallel_batches"][0]["worker_processes"] >= 3
    assert not list(tmp_path.rglob("draw-*"))
    assert (
        np.isfinite(serial.draws).all() and serial.draws[:, 1].tolist() == [120.0] * 6
    )


def test_cluster_refits_draw_whole_clusters(tmp_path):
    out = _run(
        tmp_path,
        "clusters",
        key="state",
        within="id",
        sort_columns=["state", "id", "time"],
        expected_groups=40,
    )
    assert (
        out.diagnostics["resampled"] == "clusters (state)"
        and out.diagnostics["clusters"] == 8
    )
    # every cluster has 5 units of 3 rows, so every draw has 120 rows
    assert out.draws[:, 1].tolist() == [120.0] * 6


def test_failed_draws_are_recorded_not_redrawn_and_errors_propagate(tmp_path):
    out = _run(tmp_path, "fail", task={"fail_below": 0.1}, reps=10)
    failed = {f["draw"] for f in out.failures}
    assert failed and all(not np.isfinite(out.draws[b]).any() for b in failed)
    assert all(f["error_type"] == "ValueError" for f in out.failures)
    assert len(out.diagnostics["draw_plan_sha256"]) == 10
    with pytest.raises(KeyError, match="programming error"):
        _run(tmp_path, "crash", task={"crash": True})
    with pytest.raises(KeyError, match="programming error"):
        _run(
            tmp_path,
            "crash_parallel",
            task={"crash": True},
            workers=2,
            worker_minimum_gb=0.05,
        )
    with pytest.raises(RuntimeError, match="exited 7"):
        _run(tmp_path, "die", task={"die": True}, workers=2, worker_minimum_gb=0.05)
    with pytest.raises(RuntimeError, match="differs from the original"):
        _run(tmp_path, "mismatch", expected_rows=119)


@pytest.mark.parametrize("error", [RuntimeError, KeyboardInterrupt])
def test_process_batch_stops_running_sibling_on_failure_or_interrupt(error, tmp_path):
    ready = tmp_path / "ready"
    processes = []
    start = time.monotonic()
    with pytest.raises(error), ProcessBatch(2) as batch:
        sleeper = batch.submit(
            [
                sys.executable,
                "-c",
                f"from pathlib import Path; import time; Path({str(ready)!r}).touch(); time.sleep(60)",
            ],
            tmp_path / "sleep.log",
        )
        deadline = time.monotonic() + 10
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready.exists()
        with batch._lock:
            processes = list(batch._children)
        if error is KeyboardInterrupt:
            raise KeyboardInterrupt()
        bad = batch.submit(
            [sys.executable, "-c", "raise SystemExit(42)"], tmp_path / "bad.log"
        )
        for future in as_completed([sleeper, bad]):
            if future.result():
                raise RuntimeError("worker failed")
    assert time.monotonic() - start < 15
    assert processes and all(p.poll() is not None for p in processes)


@pytest.mark.parametrize("persistent", [True, False])
def test_refits_get_their_workers_share_of_threads_and_memory(persistent, tmp_path):
    out = _run(
        tmp_path,
        "limits",
        task={"report_limits": True},
        workers=2,
        threads=2,
        memory_gb=8.0,
        worker_minimum_gb=0.05,
        persistent_workers=persistent,
    )
    share = out.diagnostics["parallel_batches"][0]["worker_memory_gb"]
    assert out.draws[:, 0].tolist() == [1.0] * 6  # 2 job threads over 2 workers
    assert_allclose(out.draws[:, 1], share)
    serial = _run(
        tmp_path, "limits_serial", task={"report_limits": True}, memory_gb=8.0
    )
    assert (
        serial.draws[:, 0].tolist() == [2.0] * 6
        and serial.draws[:, 1].tolist() == [8.0] * 6
    )

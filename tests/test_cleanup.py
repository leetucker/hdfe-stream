"""Disk lifecycle: what is written, when it is deleted, and what survives.

This matters more than usual here. The whole point of the library is datasets
that do not fit in memory, so a fit writes a sorted copy of the rows plus cell
tables and memory-mapped arrays. Leaving those behind fills the disk; deleting
them too early breaks `resid()` and `fixef()`.

Each fit works in its own run directory under `workdir`, marked with a file so
that `cleanup()` can tell its own directories from anything else there.
"""

from __future__ import annotations

import gc
import os
import socket
import time
from pathlib import Path

import pytest

import hdfe_stream
from hdfe_stream import StreamingHDFE, feols_stream, workspace

pytest.importorskip("pyfixest")

FML = "log_earn ~ age_squared + age_cubed | worker_id + firm_id + year"


def run_dirs(workdir):
    return sorted(p.name for p in Path(workdir).glob("hdfe_run_*")) if Path(workdir).exists() else []


def files_in(run):
    return sorted(str(p.relative_to(run)) for p in Path(run).rglob("*") if p.is_file())


def run_of(fit):
    """The run directory holding a result's files."""
    return Path(fit.files_dir).parents[1]


def test_only_result_files_survive_a_fit(rich, workdir):
    """Intermediates are deleted as soon as the fit no longer needs them, so
    what is left is the residual file, the fixed-effect files and the marker."""
    fit = feols_stream(FML, rich.src, workdir=workdir, verbose=False)
    remaining = files_in(run_of(fit))

    assert any(f.endswith("resid.parquet") for f in remaining)
    assert sum(f.startswith("models/") and "fe_" in f for f in remaining) == 3
    for junk in ("rows_partitioned", "ident_", "gamma", "map_"):
        assert not any(junk in f for f in remaining), (junk, remaining)
    fit.cleanup()


def test_result_files_go_when_the_result_is_dropped(rich, workdir):
    """The default `outputs="auto"` ties the files to the result's lifetime."""
    fit = feols_stream(FML, rich.src, workdir=workdir, verbose=False)
    assert fit.resid().select("resid").collect().height > 0
    assert run_dirs(workdir)

    del fit
    gc.collect()
    assert run_dirs(workdir) == []


def test_reported_disk_use(rich, workdir):
    fit = feols_stream(FML, rich.src, workdir=workdir, verbose=False)
    diagnostics = fit.diagnostics
    assert diagnostics["disk_peak_gb"] > 0
    # the peak includes the intermediates, so it exceeds what is kept
    assert diagnostics["disk_peak_gb"] >= diagnostics["disk_results_gb"]
    fit.cleanup()


def test_copies_share_the_run_directory(rich, workdir):
    """`with_vcov` returns a copy; it must keep the files alive, and one
    `cleanup()` must release them for every copy."""
    fit = feols_stream(FML, rich.src, workdir=workdir, verbose=False, cluster=["firm_id"])
    copy = fit.with_vcov("hetero")
    del fit
    gc.collect()

    assert copy.resid().select("resid").collect().height > 0
    copy.cleanup()
    assert run_dirs(workdir) == []


def test_reading_after_cleanup_says_so(rich, workdir):
    fit = feols_stream(FML, rich.src, workdir=workdir, verbose=False)
    fit.cleanup()
    with pytest.raises(FileNotFoundError, match="cleaned up"):
        fit.resid()


def test_outputs_keep_survives_the_result(rich, workdir):
    """`outputs="keep"` leaves the files for a later process to read; they go
    when `cleanup(workdir)` is called."""
    fit = feols_stream(FML, rich.src, workdir=workdir, verbose=False, outputs="keep")
    del fit
    gc.collect()
    assert len(run_dirs(workdir)) == 1

    removed = hdfe_stream.cleanup(workdir)
    assert len(removed) == 1
    assert sum(size for _, size in removed) > 0
    assert run_dirs(workdir) == []


def test_context_manager_releases_the_files(rich, workdir):
    with feols_stream(FML, rich.src, workdir=workdir, verbose=False) as fit:
        assert fit.fixef("firm_id").collect().height > 0
    assert run_dirs(workdir) == []


def test_save_resid_false_skips_the_row_level_file(rich, workdir):
    """The residual file is the big one -- one row per observation. Skipping it
    still leaves the fixed effects, which are level-sized."""
    with feols_stream(FML, rich.src, workdir=workdir, verbose=False,
                      save_resid=False) as fit:
        remaining = files_in(run_of(fit))
        assert not any("resid" in f for f in remaining)
        assert sum(f.startswith("models/") and "fe_" in f for f in remaining) == 3
        with pytest.raises(FileNotFoundError, match="save_resid"):
            fit.resid()


def test_nothing_is_left_after_a_failure_in_the_solve(rich, workdir, monkeypatch):
    """An interrupt part way through must not leave the sorted row copy behind
    -- that is the largest intermediate."""
    def boom(self):
        raise KeyboardInterrupt

    monkeypatch.setattr(StreamingHDFE, "_solve", boom)
    with pytest.raises(KeyboardInterrupt):
        feols_stream(FML, rich.src, workdir=workdir, verbose=False)
    assert run_dirs(workdir) == []


def test_nothing_is_left_after_an_estimation_error(rich, workdir):
    """All covariates collinear: the fit fails after files have been written."""
    with pytest.raises(ValueError):
        feols_stream("log_earn ~ i(year) | worker_id + firm_id + year", rich.src,
                     workdir=workdir, verbose=False)
    assert run_dirs(workdir) == []


def test_a_later_failure_removes_the_earlier_models_files(rich, workdir):
    """With `sw()` over FE sets, the first set succeeds and the second fails.
    The whole run goes, not just the part that failed."""
    with pytest.raises(ValueError, match="not found"):
        feols_stream("log_earn ~ age_squared | sw(worker_id + firm_id, worker_id + nosuch)",
                     rich.src, workdir=workdir, verbose=False)
    assert run_dirs(workdir) == []


def test_feis_and_clustering_leave_no_intermediates(rich, workdir):
    """Varying slopes and cluster factorization write extra intermediates
    (Ainv, C, cluster maps); none of them may linger."""
    fit = feols_stream("log_earn ~ age_squared | worker_id[year] + firm_id", rich.src,
                       workdir=workdir, verbose=False, cluster=["worker_id+firm_id"])
    remaining = files_in(run_of(fit))
    assert all(f.startswith("models/") or f == workspace._MARKER for f in remaining), remaining

    del fit
    gc.collect()
    assert os.listdir(workdir) == []


def test_cleanup_removes_leftovers_from_a_dead_process(workdir):
    """A killed job leaves a run directory whose marker names a pid that no
    longer exists; `cleanup()` must claim it."""
    workdir = Path(workdir)
    dead = workdir / "hdfe_run_19700101_000000_deadbeef"
    (dead / "rows").mkdir(parents=True)
    (dead / "rows" / "x.bin").write_bytes(b"0" * 1_000_000)
    (dead / workspace._MARKER).write_text(
        f"{socket.gethostname()} 999999 {time.time()}\n")

    planned = hdfe_stream.cleanup(workdir, dry_run=True)
    assert [Path(p).name for p, _ in planned] == [dead.name]
    assert planned[0][1] >= 1_000_000       # it reports the size it would free
    assert dead.exists()                    # dry run really did nothing

    assert len(hdfe_stream.cleanup(workdir)) == 1
    assert not dead.exists()


def test_cleanup_ignores_directories_it_did_not_create(workdir):
    """Only directories carrying hdfe_stream's marker are touched, so pointing
    `cleanup()` at a shared scratch area is safe."""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "hdfe_run_not_ours").mkdir()
    (workdir / "hdfe_run_not_ours" / "precious.txt").write_text("keep me")
    (workdir / "unrelated").mkdir()

    assert hdfe_stream.cleanup(workdir) == []
    assert (workdir / "hdfe_run_not_ours" / "precious.txt").read_text() == "keep me"
    assert (workdir / "unrelated").exists()


def test_cleanup_of_a_missing_directory_is_not_an_error(tmp_path):
    assert hdfe_stream.cleanup(tmp_path / "does_not_exist") == []


def test_concurrent_fits_use_separate_run_directories(rich, workdir):
    first = feols_stream(FML, rich.src, workdir=workdir, verbose=False, outputs="keep")
    second = feols_stream(FML, rich.src, workdir=workdir, verbose=False, outputs="keep")
    assert run_of(first) != run_of(second)
    assert len(run_dirs(workdir)) == 2
    hdfe_stream.cleanup(workdir)


def test_invalid_outputs_option_rejected(rich, workdir):
    with pytest.raises(ValueError, match="outputs must be"):
        feols_stream(FML, rich.src, workdir=workdir, outputs="forever")

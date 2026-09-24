"""Run directories on disk, and the public `cleanup()`."""

from __future__ import annotations

import os
import shutil
import socket
import tempfile
import time
import uuid
import weakref
from pathlib import Path


# --------------------------------------------------------------------------
# disk management
#
# Each fit works in its own run directory, <workdir>/hdfe_run_<time>_<id>/,
# so concurrent fits never collide and cleanup never touches anything else in
# workdir. Intermediates (the sorted copy of the rows, cell tables, memory-
# mapped arrays, the solution Gamma, factor maps) are deleted as soon as the
# fit no longer needs them. Result files (residuals and fixed effects, read
# lazily through resid()/fixef()) live in <run>/models/ and are removed when
# the results are garbage-collected or the interpreter exits
# (outputs="auto"), when .cleanup() is called, or never (outputs="keep").
# On any exception, the whole run directory is removed.
# --------------------------------------------------------------------------

_MARKER = ".hdfe_stream_run"
_ACTIVE_RUNS = set()      # run directories of fits currently running in this process


def _rmtree_quiet(path):
    _ACTIVE_RUNS.discard(str(path))
    shutil.rmtree(path, ignore_errors=True)


def _dir_bytes(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


class _Run:
    """A run directory, shared by all results of one fit."""

    def __init__(self, base, auto_cleanup):
        base = Path(base)
        base.mkdir(parents=True, exist_ok=True)
        self.path = base / f"hdfe_run_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        self.path.mkdir()
        (self.path / _MARKER).write_text(f"{socket.gethostname()} {os.getpid()} {time.time()}\n")
        _ACTIVE_RUNS.add(str(self.path))
        self._fin = (weakref.finalize(self, _rmtree_quiet, str(self.path))
                     if auto_cleanup else None)

    def cleanup(self):
        if self._fin is not None:
            self._fin()                      # runs once; detaches itself
        else:
            _rmtree_quiet(str(self.path))

    @property
    def exists(self):
        return self.path.exists()


def cleanup(workdir=None, older_than_hours=None, force=False, dry_run=False):
    """Remove leftover run directories (e.g. from killed processes) in
    `workdir` (default: the system temporary directory). Only directories
    created by hdfe_stream (they carry a marker file) are touched.

    Runs of this process are removed unless a fit is still running in them
    (this includes result files of finished fits; their resid()/fixef() stop
    working). Skipped unless force=True: runs of other processes still alive
    on this host, runs from other hosts (their liveness can't be checked),
    and runs younger than `older_than_hours` if given.
    Returns a list of (path, bytes) that were (or, with dry_run, would be)
    removed.
    """
    base = Path(workdir) if workdir else Path(tempfile.gettempdir())
    host, removed = socket.gethostname(), []
    if not base.exists():
        return removed
    for d in sorted(base.glob("hdfe_run_*")):
        marker = d / _MARKER
        if not marker.exists():
            continue
        try:
            h, pid, started = marker.read_text().split()
            pid, started = int(pid), float(started)
        except ValueError:
            h, pid, started = "?", -1, 0.0
        if str(d) in _ACTIVE_RUNS:
            continue                        # a fit is running in it right now
        mine = h == host and pid == os.getpid()
        if not force and not mine:
            if older_than_hours is not None and time.time() - started < older_than_hours * 3600:
                continue
            if h != host:
                continue
            if pid > 0:
                try:
                    os.kill(pid, 0)
                    continue                # process still running
                except ProcessLookupError:
                    pass
                except PermissionError:
                    continue                # exists, owned by someone else
        size = _dir_bytes(d)
        if not dry_run:
            _rmtree_quiet(str(d))
        removed.append((str(d), size))
    return removed

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
WORKDIR_ENV = "HDFE_STREAM_WORKDIR"
_ACTIVE_RUNS = set()      # run directories of fits currently running in this process


def _rmtree_quiet(path):
    _ACTIVE_RUNS.discard(str(path))
    shutil.rmtree(path, ignore_errors=True)


def _pid_alive(pid):
    """Whether a process with this pid exists on this host (or is owned by
    someone else, which counts as alive). Signal 0 is not usable on Windows,
    where os.kill would terminate the process."""
    if os.name == "nt":
        import ctypes
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.restype = ctypes.c_void_p
        handle = kernel32.OpenProcess(0x1000, False, pid)   # QUERY_LIMITED_INFORMATION
        if not handle:
            return ctypes.get_last_error() == 5             # access denied: it exists
        code = ctypes.c_ulong()
        ok = kernel32.GetExitCodeProcess(ctypes.c_void_p(handle), ctypes.byref(code))
        kernel32.CloseHandle(ctypes.c_void_p(handle))
        return bool(ok) and code.value == 259               # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


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


def _base_dir(workdir=None):
    """The directory run directories go under: `workdir` if given, else
    $HDFE_STREAM_WORKDIR if set, else the system temporary directory."""
    return Path(workdir or os.environ.get(WORKDIR_ENV) or tempfile.gettempdir())


def cleanup(workdir: str | Path | None = None,
            older_than_hours: float | None = None, force: bool = False,
            dry_run: bool = False) -> list[tuple[str, int]]:
    """Remove leftover run directories (e.g. from killed processes) in
    `workdir` (default: $HDFE_STREAM_WORKDIR if set, else the system
    temporary directory). Only directories
    created by hdfe_stream (they carry a marker file) are touched.

    Runs of this process are removed unless a fit is still running in them
    (this includes result files of finished fits; their resid()/fixef() stop
    working). Skipped unless force=True: runs of other processes still alive
    on this host, runs from other hosts (their liveness can't be checked),
    and runs younger than `older_than_hours` if given.
    Returns a list of (path, bytes) that were (or, with dry_run, would be)
    removed.
    """
    base = _base_dir(workdir)
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
                if _pid_alive(pid):
                    continue                # process still running
        size = _dir_bytes(d)
        if not dry_run:
            _rmtree_quiet(str(d))
        removed.append((str(d), size))
    return removed

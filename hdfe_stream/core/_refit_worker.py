"""Private fresh-process entry point of the refit bootstrap (core.refit).

Run as a script, not as ``-m``: importing hdfe_stream would load Polars before the
worker's thread and memory limits are set. It sets them, imports the consumer's
``preload`` modules (which may set scheduler defaults before their numerics), then
runs refits: with ``--serve``, one after another as the parent sends them (a
persistent worker); otherwise the one task of a pickled job file.
"""

import os
import sys

# Running as a script puts this directory first on the path, where core's own
# modules (typing.py) would shadow the standard library's: drop it first.
_here = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.curdir) != _here]

import json
import pickle
import traceback
from pathlib import Path


def _prepare(threads, memory, spill, header):
    """Set the limits, then import the consumer's packages (before any numerics)."""
    import stream_limits

    spill.mkdir()
    stream_limits.set_limits(
        threads=int(threads), memory_gb=float(memory), spill_dir=spill, override=True
    )
    header = json.loads(Path(header).read_text())
    # Import the parent's packages, including from a checkout that is not
    # installed, or from a notebook's working directory.
    for entry in reversed(header["sys_path"]):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    for module in header["preload"]:
        __import__(module)


def _error_reply(error):
    # Preserve resource and programming exceptions instead of treating them as
    # statistical failures or returning a partial bootstrap result.
    try:
        pickle.dumps(error)
    except Exception:  # noqa: BLE001 (an unpicklable error is sent as text)
        error = RuntimeError(f"{type(error).__name__}: {error}")
    return {"error": error, "traceback": traceback.format_exc()}


def serve(header, threads, memory):
    """A persistent worker: one draw per message on stdin, one reply on stdout.

    The channel is the process's original stdout; anything the fits print goes
    to stderr (the worker's log) instead. A ``None`` message, or the end of the
    input, stops the worker.
    """
    channel = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)
    source = sys.stdin.buffer
    try:
        _prepare(threads, memory, Path(header).parent / "spill", header)
        import polars as pl

        from hdfe_stream.core.refit import _attempt, _receive, _send

        ready = None
    except BaseException as error:  # noqa: BLE001 (sent back to the parent, which re-raises it)
        ready = _error_reply(error)
        _receive = None
    if ready is not None:
        # report the failed start as the reply to the first draw
        import struct

        head = source.read(8)
        if len(head) == 8:
            source.read(struct.unpack("<Q", head)[0])
        data = pickle.dumps(ready, protocol=pickle.HIGHEST_PROTOCOL)
        channel.write(struct.pack("<Q", len(data)) + data)
        channel.flush()
        return
    while True:
        message = _receive(source)
        if message is None:
            return
        try:
            reply = {
                "result": _attempt(message["fit"], message["task"], message["width"]),
                "resources": {
                    "pid": os.getpid(),
                    "polars_threads": pl.thread_pool_size(),
                },
            }
        except BaseException as error:  # noqa: BLE001 (sent back to the parent, which re-raises it)
            reply = _error_reply(error)
        _send(channel, reply)


def main():
    if sys.argv[1] == "--serve":
        serve(*sys.argv[2:])
        return
    job, output, threads, memory = sys.argv[1:]
    import stream_limits

    spill = Path(job).parent / "spill"
    spill.mkdir()
    stream_limits.set_limits(
        threads=int(threads), memory_gb=float(memory), spill_dir=spill, override=True
    )
    try:
        header = json.loads(Path(job).with_name("job.json").read_text())
        # Import the parent's packages, including from a checkout that is not
        # installed, or from a notebook's working directory.
        for entry in reversed(header["sys_path"]):
            if entry not in sys.path:
                sys.path.insert(0, entry)
        for module in header["preload"]:
            __import__(module)
        import polars as pl

        from hdfe_stream.core.refit import _attempt

        with open(job, "rb") as f:
            payload = pickle.load(
                f
            )  # Private, locally written task; never external input.
        result = _attempt(payload["fit"], payload["task"], payload["width"])
        out = {
            "result": result,
            "resources": {"pid": os.getpid(), "polars_threads": pl.thread_pool_size()},
        }
    except BaseException as error:  # noqa: BLE001 (sent back to the parent, which re-raises it)
        # Preserve resource and programming exceptions instead of treating them as
        # statistical failures or returning a partial bootstrap result.
        try:
            pickle.dumps(error)
        except Exception:  # noqa: BLE001 (an unpicklable error is sent as text)
            error = RuntimeError(f"{type(error).__name__}: {error}")
        out = {"error": error, "traceback": traceback.format_exc()}
    with open(output, "wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()

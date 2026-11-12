"""The refit bootstrap: resample whole groups with replacement and refit on each draw.

`refit_draws` stages a source by group (`core.resample`), and for each of ``reps``
draws writes a resample to disk and calls the consumer's ``fit(task)``, serially or
in fresh worker processes (`core.process`), each with its share of the job's
threads and memory. Failed draws (a ``ValueError``, ``FloatingPointError`` or
``LinAlgError`` from ``fit``) are recorded and never redrawn; any other exception,
including ``MemoryError``, stops the run. `RefitBootstrapResult` holds the draws
and gives bootstrap standard errors and pointwise intervals.

What a draw's sample must satisfy and which estimator is refitted belong to the
consumer: ``fit`` checks the draw (``task["dates"]``, ``task["rows"]``) and fits it,
with ``task["unit_column"]`` as its unit identifier.
"""

from __future__ import annotations

import json
import pickle
import shutil
import struct
import subprocess
import sys
import threading
import warnings
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.stats import norm
from stream_limits import detect_cpus

from .process import ProcessBatch
from .resample import DiskGroupResampler

_WORKER = Path(__file__).with_name("_refit_worker.py")


class BootstrapFailureWarning(UserWarning):
    """Some refits of a bootstrap failed; inference on the rest needs explicit acknowledgment."""


@dataclass(kw_only=True)
class RefitBootstrapResult:
    """Point estimates, refit draws and pointwise bootstrap inference.

    No source or fit scratch survives. Failed attempts remain in the ledger and
    are never redrawn. Inference on successful draws requires allow_incomplete
    when any attempt failed; such inference is conditional on successful fits.
    """

    fml: str
    estimator: str
    coefnames: list
    beta: np.ndarray
    diagnostics: dict
    _draws: np.ndarray = field(repr=False)
    _failures: list = field(repr=False)
    _warnings: list = field(repr=False)

    @property
    def successful_reps(self):
        return int(np.isfinite(self._draws).all(axis=1).sum())

    @property
    def se(self):
        """Sample SD of refitted estimates; refuses incomplete bootstrap runs."""
        return self.standard_errors()

    def _valid(self, allow_incomplete):
        if self._failures and not allow_incomplete:
            raise ValueError(
                "some refits failed; inspect failures() and explicitly set allow_incomplete=True for inference conditional on success"
            )
        data = self._draws[np.isfinite(self._draws).all(axis=1)]
        if len(data) < 2:
            raise ValueError(
                "at least two successful refits are required for bootstrap inference"
            )
        return data

    def standard_errors(self, *, allow_incomplete=False):
        return np.std(self._valid(allow_incomplete), axis=0, ddof=1)

    def confint(self, alpha=0.05, *, method="percentile", allow_incomplete=False):
        """Pointwise percentile/type-7, basic, or uncorrected normal intervals.

        Normal intervals are point estimate +/- normal critical value times
        bootstrap SD, without boot.ci's normal bias correction. No simultaneous,
        studentized or BCa intervals are supplied.
        """
        if isinstance(alpha, bool) or not np.isfinite(alpha) or not 0 < alpha < 1:
            raise ValueError("alpha must be between zero and one")
        if method not in ("percentile", "basic", "normal"):
            raise ValueError("method must be percentile, basic or normal")
        draws = self._valid(allow_incomplete)
        if method == "normal":
            spread = norm.ppf(1 - alpha / 2) * np.std(draws, axis=0, ddof=1)
            lower, upper = self.beta - spread, self.beta + spread
        else:
            low, high = np.quantile(
                draws, [alpha / 2, 1 - alpha / 2], axis=0, method="linear"
            )
            lower, upper = (
                (low, high)
                if method == "percentile"
                else (2 * self.beta - high, 2 * self.beta - low)
            )
        return pl.DataFrame(
            {
                "Coefficient": self.coefnames,
                "Estimate": self.beta,
                "Lower": lower,
                "Upper": upper,
            }
        )

    def tidy(self, alpha=0.05, *, method="percentile", allow_incomplete=False):
        draws = self._valid(allow_incomplete)
        return self.confint(
            alpha, method=method, allow_incomplete=allow_incomplete
        ).with_columns(
            pl.Series("Std. Error", np.std(draws, axis=0, ddof=1)),
            pl.Series("Bootstrap Mean", draws.mean(axis=0)),
            pl.Series("Bias", draws.mean(axis=0) - self.beta),
            pl.lit(len(draws)).alias("Successful Reps"),
        )

    def draws(self):
        """Copy attempted estimates; failed rows are NaN for every coefficient."""
        return pl.DataFrame(
            {
                "draw": np.arange(len(self._draws)),
                "successful": np.isfinite(self._draws).all(axis=1),
                **{n: self._draws[:, j].copy() for j, n in enumerate(self.coefnames)},
            }
        )

    def failures(self):
        return pl.DataFrame(
            self._failures,
            schema={"draw": pl.Int64, "error_type": pl.String, "reason": pl.String},
        )

    def refit_warnings(self):
        return pl.DataFrame(
            self._warnings,
            schema={"draw": pl.Int64, "category": pl.String, "message": pl.String},
        )

    def summary_dict(self):
        return {
            "fml": self.fml,
            "estimator": self.estimator,
            "coefficients": dict(zip(self.coefnames, self.beta.tolist(), strict=True)),
            "requested_reps": len(self._draws),
            "successful_reps": self.successful_reps,
            "failures": self.failures().to_dicts(),
            "refit_warnings": self.refit_warnings().to_dicts(),
            "diagnostics": self.diagnostics,
        }

    def summary(self):
        """Run/sample/failure diagnostics; inference is available through tidy()."""
        return self.summary_dict()

    def summary_json(self, path=None):
        content = json.dumps(
            self.summary_dict(), indent=2, allow_nan=False, default=str
        )
        if path is not None:
            Path(path).write_text(content + "\n")
        return content


@dataclass
class RefitDraws:
    """What `refit_draws` returns: the draws (reps by coefficients, NaN rows for
    failed draws), the failures, the warnings the refits raised, and diagnostics of
    how the draws were made (seed, group order and draw-plan hashes, workers)."""

    draws: np.ndarray
    failures: list
    warnings: list
    diagnostics: dict


def refit_workers(reps, workers, threads):
    """Validate the run's options; return (workers used, job threads, threads per worker)."""
    for name, value, minimum in [("reps", reps, 2), ("n_workers", workers, 1)]:
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if threads is not None and (
        isinstance(threads, bool) or not isinstance(threads, int) or threads < 1
    ):
        raise ValueError("n_threads must be a positive integer")
    active = min(workers, reps)
    job_threads = threads if threads is not None else detect_cpus()
    if active > 1 and active > job_threads:
        raise ValueError(
            "n_workers cannot exceed the total thread budget; increase n_threads or reduce n_workers"
        )
    return active, job_threads, max(1, job_threads // active)


def _free_name(names, base):
    while base in names:
        base += "_"
    return base


def refit_draws(
    snapshot,
    scratch,
    *,
    fit,
    task,
    width,
    reps,
    budget,
    preprocess_mb,
    key,
    sort_columns,
    seed=None,
    workers=1,
    threads=None,
    memory_gb=None,
    select=None,
    within=None,
    time_column=None,
    expected_groups=None,
    expected_rows=None,
    worker_minimum_gb=0.5,
    preload=(),
    persistent_workers=True,
    worker_draws=200,
):
    """Draw ``reps`` resamples of the groups of ``snapshot`` and refit each with ``fit``.

    ``snapshot`` is a Parquet file of the analysis sample's source rows and
    ``scratch`` an existing directory for the run. ``fit`` is an importable
    function (it is pickled by reference for the workers) called with a dict:
    ``task`` plus ``draw``, ``path`` (the resample), ``unit_column`` (each copy's
    unit ID), ``workdir``, ``dates`` (the values of ``time_column`` it covers),
    ``rows``, and the ``threads`` and ``memory_gb`` the refit may use. It returns
    the ``width`` coefficients, or raises ``ValueError`` (or ``FloatingPointError``,
    ``LinAlgError``) for a draw that cannot be estimated.

    Groups are ``key`` (with ``within``: clusters of ``within`` units), sorted by
    ``sort_columns`` and filtered by ``select(frame)``; ``expected_groups`` and
    ``expected_rows`` check the staging against the original fit. ``seed`` fixes
    the draws for a given group order. ``workers`` > 1 runs refits in worker
    processes, which first import the modules in ``preload`` (a package's
    scheduler defaults) and need ``worker_minimum_gb`` each. They are started once
    and given one draw after another, each restarted after ``worker_draws`` draws
    (so that memory a runtime keeps cannot accumulate); with
    ``persistent_workers=False`` each draw has a fresh process instead. Either way
    the draws are the same, and only one resample per worker is on disk at a time.
    """
    if seed is not None and (
        isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
    ):
        raise ValueError("seed must be a nonnegative integer or None")
    active, job_threads, worker_threads = refit_workers(reps, workers, threads)
    scratch = Path(scratch)
    with pq.ParquetFile(snapshot) as pf:
        names = set(pf.schema_arrow.names)
    index_column = _free_name(names, "__refit_source_group")
    draw_column = _free_name(names | {index_column}, "__refit_draw_unit")
    seed_sequence = np.random.SeedSequence(seed)
    rng = np.random.default_rng(seed_sequence)
    stage = scratch / "groups"
    stage.mkdir()
    sampler = DiskGroupResampler(
        snapshot,
        stage,
        budget,
        preprocess_mb,
        key=key,
        sort_columns=list(sort_columns),
        select=select if select is not None else (lambda frame: frame),
        index_column=index_column,
        draw_column=draw_column,
        within=within,
    )
    groups = sampler.n_within if within is not None else sampler.n_units
    if (expected_groups is not None and groups != expected_groups) or (
        expected_rows is not None and sampler.n_rows != expected_rows
    ):
        raise RuntimeError("bootstrap staging differs from the original fitted sample")
    pa.default_memory_pool().release_unused()
    if not budget.fits((8 * 4 * reps * width + 48e6) / 1e9):
        raise MemoryError(
            "bootstrap draw matrix exceeds job headroom; reduce reps/coefficients"
        )
    draws = np.full((reps, width), np.nan)
    failures, captured, hashes, batches, worker_resources = [], [], [], [], []

    def prepare(b):
        """Write draw b (in draw order, so the plan is the same whatever the workers do)."""
        attempt = scratch / f"draw-{b}"
        attempt.mkdir()
        path, dates, rows, identity = sampler.materialize(
            rng, attempt / "data", time_column=time_column
        )
        hashes.append(identity)
        return _task_for(
            task, b, path, draw_column, attempt, dates, rows, threads, memory_gb
        )

    def record(b, outcome, resources):
        values, failure, notes = outcome
        if values is not None:
            draws[b] = values
        if failure is not None:
            failures.append(failure)
        captured.extend(notes)
        if resources is not None:
            worker_resources.append(dict(draw=b, **resources))
        shutil.rmtree(scratch / f"draw-{b}", ignore_errors=True)

    worker_memory = None
    if active > 1:
        pa.default_memory_pool().release_unused()
        worker_memory = _worker_memory_gb(budget, active)
        if worker_memory * (1 - budget.reserve) < worker_minimum_gb:
            raise MemoryError(
                "insufficient whole-job memory for parallel refits including worker runtimes; "
                "reduce n_workers/preprocess_mb or increase max_memory_gb"
            )
    try:
        if active == 1:
            for b in range(reps):
                record(b, _attempt(fit, prepare(b), width), None)
        elif persistent_workers:
            starts = _persistent_refits(
                prepare,
                record,
                reps,
                active,
                worker_threads,
                worker_memory,
                scratch=scratch,
                fit=fit,
                width=width,
                preload=tuple(preload),
                restart_after=worker_draws,
            )
            batches.append(
                {"worker_processes": starts, "worker_memory_gb": worker_memory}
            )
        else:
            for start in range(0, reps, active):
                tasks = [prepare(b) for b in range(start, min(start + active, reps))]
                batches.append(
                    {
                        "first_draw": start,
                        "refits": len(tasks),
                        "worker_memory_gb": worker_memory,
                    }
                )
                completed = _parallel_refits(
                    tasks,
                    active,
                    worker_threads,
                    worker_memory,
                    fit=fit,
                    width=width,
                    preload=tuple(preload),
                )
                # Closing the generator stops and reaps workers before the draws'
                # directories are removed, including on interruption.
                with closing(completed) as results:
                    for b, outcome, resources in results:
                        record(b, outcome, resources)
    finally:
        for leftover in scratch.glob("draw-*"):
            shutil.rmtree(leftover, ignore_errors=True)
    failures.sort(key=lambda x: x["draw"])
    captured.sort(key=lambda x: x["draw"])
    worker_resources.sort(key=lambda x: x["draw"])
    diagnostics = {
        "workers_requested": workers,
        "workers": active,
        "job_threads": job_threads,
        "worker_threads": worker_threads if active > 1 else threads,
        "parallel_batches": batches,
        "persistent_workers": bool(persistent_workers) if active > 1 else None,
        "worker_resources": worker_resources,
        "seed": seed,
        "seed_entropy": seed_sequence.entropy,
        "bit_generator": type(rng.bit_generator).__name__,
        "reps": reps,
        "resampled": f"clusters ({key})" if within is not None else "units",
        "clusters": sampler.n_units if within is not None else None,
        "unit_count": groups,
        "unit_order": "hash leaf traversal, sorted source IDs within leaf; fixed input/layout reproducibility",
        "unit_order_sha256": sampler.identity.hexdigest(),
        "draw_plan_sha256": hashes,
        "partitions": sampler.partition_count,
        "preprocess_mb": preprocess_mb,
    }
    return RefitDraws(draws, failures, captured, diagnostics)


def _task_for(task, b, path, unit_column, attempt, dates, rows, threads, memory_gb):
    """The task of draw b: the consumer's task and what the draw is."""
    return dict(
        task,
        draw=b,
        path=path,
        unit_column=unit_column,
        workdir=attempt / "fit",
        dates=dates,
        rows=rows,
        threads=threads,
        memory_gb=memory_gb,
    )


def _worker_memory_gb(budget, active):
    """Each worker's memory: the job's headroom (its staging is idle while children
    fit, and the job reserve is kept) shared among the workers, which also keep
    their own MemoryBudget reserve."""
    return budget.headroom_gb() / active


def _send(stream, message):
    data = pickle.dumps(message, protocol=pickle.HIGHEST_PROTOCOL)
    stream.write(struct.pack("<Q", len(data)) + data)
    stream.flush()


def _receive(stream):
    """The next message, or None at the end of the stream."""
    head = stream.read(8)
    if len(head) < 8:
        return None
    (size,) = struct.unpack("<Q", head)
    data = stream.read(size)
    if len(data) < size:
        return None
    return pickle.loads(data)  # Private channel to our own worker.


class _WorkerPool:
    """Persistent refit workers, each fed one draw at a time over its own pipe."""

    def __init__(self, count, threads, memory_gb, *, scratch, preload, restart_after):
        self.count, self.threads, self.memory_gb = count, threads, memory_gb
        self.scratch, self.preload, self.restart_after = (
            Path(scratch),
            preload,
            restart_after,
        )
        self.children, self.served, self.logs = (
            [None] * count,
            [0] * count,
            [None] * count,
        )
        self.starts = 0
        self._lock = threading.Lock()
        self._closing = False

    def __enter__(self):
        for i in range(self.count):
            self._start(i)
        return self

    def _start(self, i):
        with self._lock:
            if self._closing:
                raise RuntimeError("the worker pool is closing")
            root = self.scratch / f"worker-{self.starts}"
            root.mkdir()
            header = root / "worker.json"
            header.write_text(
                json.dumps({"sys_path": sys.path, "preload": list(self.preload)})
            )
            log = root / "worker.log"
            command = [
                sys.executable,
                str(_WORKER),
                "--serve",
                str(header),
                str(self.threads),
                repr(self.memory_gb),
            ]
            with log.open("wb") as output:
                self.children[i] = subprocess.Popen(
                    command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=output,
                )
            self.logs[i], self.served[i] = log, 0
            self.starts += 1

    def run(self, i, message):
        """Send one draw to worker i and return its reply (in a waiting thread)."""
        if self.served[i] >= self.restart_after:
            self._stop(i)
            self._start(i)
        child = self.children[i]
        try:
            _send(child.stdin, message)
            reply = _receive(child.stdout)
        except (BrokenPipeError, OSError):
            reply = None
        if reply is None:
            child.wait()
            detail = self.logs[i].read_bytes()[-4000:].decode(errors="replace")
            raise RuntimeError(
                f"bootstrap worker for draw {message['task']['draw']} exited "
                f"{child.returncode}: {detail}"
            )
        self.served[i] += 1
        return reply

    def _stop(self, i):
        child = self.children[i]
        if child is None:
            return
        try:
            _send(child.stdin, None)
            child.stdin.close()
            child.wait(timeout=30)
        except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
            child.kill()
            child.wait()
        self.children[i] = None

    def __exit__(self, exc_type, exc, tb):
        with self._lock:
            self._closing = True
        for child in self.children:
            if child is None or child.poll() is not None:
                continue
            if exc_type is None:
                try:
                    _send(child.stdin, None)
                    child.stdin.close()
                except (BrokenPipeError, OSError):
                    pass
            else:
                child.terminate()
        for child in self.children:
            if child is None:
                continue
            try:
                child.wait(timeout=30 if exc_type is None else 2)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()


def _persistent_refits(
    prepare,
    record,
    reps,
    workers,
    threads,
    memory_gb,
    *,
    scratch,
    fit,
    width,
    preload,
    restart_after,
):
    """Fit draws 0..reps-1 with persistent workers; return how many processes started.

    Draws are prepared in order, each when a worker is free, so the plan is the
    serial one and at most one resample per worker is on disk. A worker's error
    stops the run: the workers are stopped (the pool closes first) before the
    threads waiting on them finish and the consumer removes the scratch.
    """
    with (
        ThreadPoolExecutor(max_workers=workers) as waiting,
        _WorkerPool(
            workers,
            threads,
            memory_gb,
            scratch=scratch,
            preload=preload,
            restart_after=restart_after,
        ) as pool,
    ):
        idle, in_flight, next_draw = list(range(workers)), {}, 0
        while next_draw < reps or in_flight:
            while idle and next_draw < reps:
                worker = idle.pop()
                # each refit runs within its worker's share of threads and memory
                task = dict(prepare(next_draw), threads=threads, memory_gb=memory_gb)
                message = {"fit": fit, "task": task, "width": width}
                in_flight[waiting.submit(pool.run, worker, message)] = (
                    worker,
                    next_draw,
                )
                next_draw += 1
            done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
            for future in done:
                worker, b = in_flight.pop(future)
                reply = future.result()
                if "error" in reply:
                    # Keep the exception type (notably MemoryError) and the remote trace.
                    raise reply["error"] from RuntimeError(reply["traceback"])
                record(b, reply["result"], reply["resources"])
                idle.append(worker)
        return pool.starts


def _attempt(fit, task, width):
    """One refit, with the same failure policy in the parent and in workers."""
    values, failure, captured = None, None, []
    b = task["draw"]
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                values = np.asarray(fit(task), dtype=float)
                if values.shape != (width,) or not np.isfinite(values).all():
                    raise ValueError(
                        "requested coefficients are nonfinite or suppressed"
                    )
            finally:
                captured.extend(
                    {
                        "draw": b,
                        "category": w.category.__name__,
                        "message": str(w.message)[:2000],
                    }
                    for w in caught
                )
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
        values = None
        failure = {
            "draw": b,
            "error_type": type(exc).__name__,
            "reason": str(exc)[:2000],
        }
    finally:
        pa.default_memory_pool().release_unused()
    return values, failure, captured


def _parallel_refits(tasks, workers, threads, memory_gb, *, fit, width, preload=()):
    """Yield completed refits from fresh worker processes; stop children before scratch cleanup."""
    with ProcessBatch(workers) as batch:
        pending = {}
        for task in tasks:
            root = Path(task["workdir"]).parent
            job, output, log = (
                root / "job.pkl",
                root / "result.pkl",
                root / "worker.log",
            )
            header = root / "job.json"
            task = dict(task, threads=threads, memory_gb=memory_gb)
            # The worker sets its limits and imports ``preload`` before unpickling the
            # task, which imports the consumer's modules (and their numerics).
            header.write_text(
                json.dumps({"sys_path": sys.path, "preload": list(preload)})
            )
            with job.open("wb") as f:
                pickle.dump(
                    {"fit": fit, "task": task, "width": width},
                    f,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
            command = [
                sys.executable,
                str(_WORKER),
                str(job),
                str(output),
                str(threads),
                repr(memory_gb),
            ]
            pending[batch.submit(command, log)] = (task["draw"], output, log)
        for future in as_completed(pending):
            b, output, log = pending[future]
            returncode = future.result()
            if returncode or not output.exists():
                with log.open("rb") as f:
                    f.seek(max(0, log.stat().st_size - 4000))
                    detail = f.read().decode(errors="replace")
                raise RuntimeError(
                    f"bootstrap worker for draw {b} exited {returncode}: {detail}"
                )
            with output.open("rb") as f:
                payload = pickle.load(f)  # Private output of our own worker.
            if "error" in payload:
                error = payload["error"]
                # Keep the exception type (notably MemoryError) and the remote trace.
                raise error from RuntimeError(payload["traceback"])
            yield b, payload["result"], payload["resources"]

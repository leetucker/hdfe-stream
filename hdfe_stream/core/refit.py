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
import sys
import warnings
from concurrent.futures import as_completed
from contextlib import closing, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory

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
    the draws for a given group order. ``workers`` > 1 runs refits in fresh
    processes, which first import the modules in ``preload`` (a package's
    scheduler defaults) and need ``worker_minimum_gb`` each.
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
    for start in range(0, reps, active):
        with TemporaryDirectory(prefix="batch-", dir=scratch) as batch_root:
            tasks = []
            for b in range(start, min(start + active, reps)):
                attempt = Path(batch_root) / f"draw-{b}"
                attempt.mkdir()
                path, dates, rows, identity = sampler.materialize(
                    rng, attempt / "data", time_column=time_column
                )
                hashes.append(identity)
                tasks.append(
                    dict(
                        task,
                        draw=b,
                        path=path,
                        unit_column=draw_column,
                        workdir=attempt / "fit",
                        dates=dates,
                        rows=rows,
                        threads=threads,
                        memory_gb=memory_gb,
                    )
                )
            if active == 1:
                completed = [(start, _attempt(fit, tasks[0], width), None)]
            else:
                pa.default_memory_pool().release_unused()
                # The parent's staging is idle while children fit. Reserve its
                # current memory plus the job reserve; workers share the rest and
                # keep their own MemoryBudget reserve.
                worker_memory = budget.headroom_gb() / active
                if worker_memory * (1 - budget.reserve) < worker_minimum_gb:
                    raise MemoryError(
                        "insufficient whole-job memory for parallel refits including worker runtimes; "
                        "reduce n_workers/preprocess_mb or increase max_memory_gb"
                    )
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
            # Closing the generator stops and reaps workers before the batch's
            # temporary directory is removed, including on interruption.
            with (
                closing(completed) if active > 1 else nullcontext(completed) as results
            ):
                for b, (values, failure, notes), resources in results:
                    if values is not None:
                        draws[b] = values
                    if failure is not None:
                        failures.append(failure)
                    captured.extend(notes)
                    if resources is not None:
                        worker_resources.append(dict(draw=b, **resources))
    failures.sort(key=lambda x: x["draw"])
    captured.sort(key=lambda x: x["draw"])
    worker_resources.sort(key=lambda x: x["draw"])
    diagnostics = {
        "workers_requested": workers,
        "workers": active,
        "job_threads": job_threads,
        "worker_threads": worker_threads if active > 1 else threads,
        "parallel_batches": batches,
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

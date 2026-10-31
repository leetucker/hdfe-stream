"""Estimates of the memory and disk a fit needs, from one scan of the data.

``profile=True`` on an estimator returns a `Profile` instead of fitting. The
estimator's package supplies the model (which arrays it holds, and how big
they are for the dimensions it found); this module has what the packages
share: the one streaming scan (`scan_source`), the `Component` a model is a
list of, and the `Profile` that adds them up, prints them and turns them into
a resource request.

How the numbers add up
    A fit runs in phases (pass 0, pass 1, the solve, ...), and most arrays
    live in one or two of them. The peak memory is the baseline (what the
    process holds now) plus the largest phase's sum, not the sum of
    everything. Each component has an *expected* size and an *upper* bound,
    because some quantities (the number of distinct fixed-effect combinations,
    the nonzeros of the reduced matrix, the instruments left after empty
    columns are dropped) are known only after a pass. Memory-mapped files are
    counted in memory as well as on disk: a scheduler's cgroup can charge the
    pages of a mapped file to the job.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import polars as pl

from .resources import current_rss_gb, detect_cpus, detect_memory_gb

__all__ = ["Component", "Profile", "Request", "ScanStats", "combine_profiles", "scan_source",
           "source_row_group_rows"]

GB = 1e9


@dataclass(frozen=True)
class Component:
    """One array (or file) a fit holds: its size, expected and as an upper
    bound, in bytes.

    ``where`` is ``"ram"``, ``"mapped"`` (a memory-mapped file: counted in
    both memory and disk) or ``"disk"``. ``phases`` names the phases it lives
    in; ``("all",)`` is the whole fit. ``formula`` says how the size was
    computed, so a surprising number can be traced."""
    name: str
    expected: float
    upper: float | None = None
    where: str = "ram"
    phases: tuple[str, ...] = ("all",)
    formula: str = ""

    def __post_init__(self):
        if self.upper is None:
            object.__setattr__(self, "upper", self.expected)
        if self.where not in ("ram", "mapped", "disk"):
            raise ValueError(f"where must be 'ram', 'mapped' or 'disk', not {self.where!r}")


@dataclass
class ScanStats:
    """What the one scan of the data found."""
    n_raw: int                                  # rows in the source
    n_rows: int                                 # rows with every used value present and finite
    distinct: dict[str, int] = field(default_factory=dict)     # approximate level counts
    minmax: dict[str, tuple] = field(default_factory=dict)
    seconds: float = 0.0

    def to_dict(self):
        return {"n_raw": self.n_raw, "n_rows": self.n_rows, "distinct": dict(self.distinct),
                "minmax": {k: list(v) for k, v in self.minmax.items()}, "seconds": self.seconds}


def scan_source(lf: pl.LazyFrame, *, present: Sequence[pl.Expr] = (),
                distinct: Mapping[str, Sequence[str]] | None = None,
                minmax: Sequence[str] = ()) -> ScanStats:
    """One streaming pass over the columns the fit uses.

    Parameters
    ----------
    present : expressions that must be non-null and, for floats, finite, for a
        row to count (the design's expressions, the fixed effects, the
        clusters, the weights): the rows the fit would keep.
    distinct : name -> source columns; the (approximate, HyperLogLog) number of
        distinct values of the combination of the columns, among kept rows.
    minmax : columns whose minimum and maximum among kept rows to report.
    """
    import time
    t0 = time.time()
    schema = lf.collect_schema()
    checks = []
    for e in present:
        checks.append(e.is_not_null())
        try:
            dtype = lf.select(e.alias("_")).collect_schema()["_"]
        except Exception:                                    # noqa: BLE001
            dtype = None
        if dtype is not None and dtype.is_float():
            checks.append(e.is_finite())
    keep = pl.all_horizontal(*checks) if checks else pl.lit(True)
    aggs = [pl.len().alias("__raw"), keep.sum().alias("__n")]
    for name, cols in (distinct or {}).items():
        for c in cols:
            if c not in schema:
                raise ValueError(f"column {c!r} is not in the data")
        key = pl.col(cols[0]) if len(cols) == 1 else pl.struct([pl.col(c) for c in cols])
        aggs.append(key.hash().filter(keep).approx_n_unique().alias(f"u:{name}"))
    for c in minmax:
        aggs.append(pl.col(c).filter(keep).min().alias(f"lo:{c}"))
        aggs.append(pl.col(c).filter(keep).max().alias(f"hi:{c}"))
    row = lf.select(aggs).collect(engine="streaming").row(0, named=True)
    return ScanStats(
        n_raw=int(row["__raw"]), n_rows=int(row["__n"] or 0),
        distinct={k[2:]: int(v or 0) for k, v in row.items() if k.startswith("u:")},
        minmax={c: (row[f"lo:{c}"], row[f"hi:{c}"]) for c in minmax},
        seconds=round(time.time() - t0, 3))


def source_row_group_rows(source) -> int | None:
    """Rows per Parquet row group of the first file of a path or glob source
    (what the streaming scan holds per thread), or None for frames."""
    import glob

    import pyarrow.parquet as pq
    if isinstance(source, (pl.DataFrame, pl.LazyFrame)):
        return None
    try:
        paths = sorted(glob.glob(os.fspath(source))) or [os.fspath(source)]
        md = pq.ParquetFile(paths[0]).metadata
        return int(max(md.row_group(0).num_rows, 1)) if md.num_row_groups else None
    except (OSError, ValueError, TypeError, IndexError):
        return None


@dataclass(frozen=True)
class Request:
    """A resource request in a form a scheduler takes."""
    mem_gb: int
    disk_gb: int
    ncpus: int

    def __str__(self):
        return f"ncpus={self.ncpus}  mem={self.mem_gb}gb  disk={self.disk_gb}gb"

    @property
    def pbs(self) -> str:
        """The ``-l select=`` line: ``select=1:ncpus=8:mem=14gb``. Disk is a
        site-specific resource (``scratch``, ``file``), so it is left out."""
        return f"select=1:ncpus={self.ncpus}:mem={self.mem_gb}gb"


@dataclass
class Profile:
    """The estimated resources of a fit, from `profile=True`.

    ``memory_gb`` and ``disk_gb`` are ``(expected, upper)`` pairs. ``print(p)``
    shows the components; ``p.recommend()`` turns the numbers into a request.
    The figures are for the threads shown: more threads mean more per-thread
    accumulators."""
    model: str
    components: list[Component]
    scan: ScanStats | None = None
    threads: int = 1
    polars_threads: int | None = None
    baseline_gb: float = 0.0
    memory_limit_gb: float = field(default_factory=detect_memory_gb)
    memory_reserve: float = 0.15
    available_cpus: int = field(default_factory=detect_cpus)
    warnings: list[str] = field(default_factory=list)
    notes: dict = field(default_factory=dict)
    time: dict | None = None            # reserved: per-pass timings, not estimated yet

    # ------------------------------------------------------------ totals
    def phases(self) -> list[str]:
        out = []
        for c in self.components:
            for p in c.phases:
                if p != "all" and p not in out:
                    out.append(p)
        return out or ["all"]

    def phase_peaks_gb(self) -> dict[str, tuple[float, float]]:
        """Memory by phase, baseline included: ``{phase: (expected, upper)}``."""
        res = {}
        for ph in self.phases():
            e = u = 0.0
            for c in self.components:
                if c.where != "disk" and (ph in c.phases or "all" in c.phases):
                    e += c.expected
                    u += c.upper
            res[ph] = (self.baseline_gb + e / GB, self.baseline_gb + u / GB)
        return res

    @property
    def memory_gb(self) -> tuple[float, float]:
        peaks = self.phase_peaks_gb().values()
        return max(p[0] for p in peaks), max(p[1] for p in peaks)

    @property
    def disk_gb(self) -> tuple[float, float]:
        e = sum(c.expected for c in self.components if c.where in ("disk", "mapped"))
        u = sum(c.upper for c in self.components if c.where in ("disk", "mapped"))
        return e / GB, u / GB

    def recommend(self, safety: float = 1.25) -> Request:
        """A request: memory is the expected peak times `safety`, but no more
        than the upper bound; disk likewise. CPUs are the threads profiled."""
        mem_e, mem_u = self.memory_gb
        dsk_e, dsk_u = self.disk_gb
        mem = min(mem_e * safety, mem_u)
        dsk = min(dsk_e * safety, dsk_u)
        return Request(mem_gb=max(1, math.ceil(mem)), disk_gb=max(1, math.ceil(dsk)),
                       ncpus=int(self.threads))

    # ----------------------------------------------------------- output
    def to_dict(self) -> dict:
        return {
            "model": self.model, "threads": self.threads, "polars_threads": self.polars_threads,
            "memory_gb": {"expected": self.memory_gb[0], "upper": self.memory_gb[1]},
            "disk_gb": {"expected": self.disk_gb[0], "upper": self.disk_gb[1]},
            "baseline_gb": self.baseline_gb,
            "phases_gb": {k: {"expected": v[0], "upper": v[1]}
                          for k, v in self.phase_peaks_gb().items()},
            "components": [{"name": c.name, "expected_gb": c.expected / GB,
                            "upper_gb": c.upper / GB, "where": c.where,
                            "phases": list(c.phases), "formula": c.formula}
                           for c in self.components],
            "scan": self.scan.to_dict() if self.scan else None,
            "available": {"memory_limit_gb": self.memory_limit_gb, "reserve": self.memory_reserve,
                          "cpus": self.available_cpus},
            "warnings": list(self.warnings), "notes": dict(self.notes),
            "recommend": str(self.recommend()),
        }

    def __str__(self):
        me, mu = self.memory_gb
        de, du = self.disk_gb
        lines = [f"Profile: {self.model}"]
        if self.scan:
            lines.append(f"  {self.scan.n_rows:,} rows kept of {self.scan.n_raw:,} "
                         f"(scan {self.scan.seconds:.1f}s); {self.threads} threads"
                         + (f", Polars {self.polars_threads}" if self.polars_threads else ""))
        lines.append(f"  memory  {me:7.2f} GB expected   {mu:7.2f} GB upper bound"
                     f"   (this process holds {self.baseline_gb:.2f} GB now)")
        lines.append(f"  disk    {de:7.2f} GB expected   {du:7.2f} GB upper bound")
        peaks = self.phase_peaks_gb()
        if len(peaks) > 1:
            lines.append("  by phase: " + ", ".join(f"{k} {v[0]:.2f}" for k, v in peaks.items()))
        lines.append(f"  request: {self.recommend()}   ({self.recommend().pbs})")
        usable = self.memory_limit_gb * (1 - self.memory_reserve)
        if mu > usable:
            lines.append(f"  note: the upper bound exceeds the {usable:.1f} GB usable "
                         f"({self.memory_limit_gb:.1f} GB limit, {self.memory_reserve:.0%} reserve)"
                         + ("; the expected peak does too" if me > usable else ""))
        lines.append("  components (GB expected / upper):")
        order = sorted(self.components, key=lambda c: -c.upper)
        width = max((len(c.name) for c in order), default=10)
        for c in order:
            lines.append(f"    {c.name:<{width}}  {c.expected / GB:8.3f} / {c.upper / GB:8.3f}"
                         f"  {c.where:<6} {','.join(c.phases)}")
        for w in self.warnings:
            lines.append(f"  warning: {w}")
        return "\n".join(lines)

    __repr__ = __str__


def baseline_gb() -> float:
    """What this process holds now (imports, compiled kernels, any data in
    memory): the fit holds it too."""
    return current_rss_gb() or 0.5


def as_components(items: Iterable[Component | None]) -> list[Component]:
    return [c for c in items if c is not None and (c.expected > 0 or c.upper > 0)]


def combine_profiles(profiles: Sequence[Profile]) -> Profile:
    """One profile for a formula that runs several estimators one after
    another (models with different fixed effects): the memory is the largest
    one's, since each fit ends before the next starts, and the disk adds up,
    since each fit's result files stay."""
    if len(profiles) == 1:
        return profiles[0]
    base = max(profiles, key=lambda p: p.memory_gb[1])
    comps = list(base.components)
    for p in profiles:
        if p is base:
            continue
        for c in p.components:
            if c.where == "disk":
                comps.append(Component(f"{c.name} ({p.model})", c.expected, c.upper, "disk",
                                       c.phases, c.formula))
    out = Profile(model=f"{len(profiles)} estimators, the largest: {base.model}",
                  components=comps, scan=base.scan, threads=base.threads,
                  polars_threads=base.polars_threads, baseline_gb=base.baseline_gb,
                  warnings=[w for p in profiles for w in p.warnings], notes=dict(base.notes))
    out.notes["estimators"] = [p.model for p in profiles]
    return out

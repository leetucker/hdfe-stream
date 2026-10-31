"""Threads and memory a fit may use, the memory left to allocate, and the peak
memory it did use.

The code is in the `stream-limits` package (module ``stream_limits``), which
depends on nothing outside the standard library so that it can run before
polars, numba and numpy are imported (a module inside this package cannot:
importing it imports `hdfe_stream` first, and that imports polars). The names
are re-exported here for the estimators and for packages built on this layer.
"""

from stream_limits import (
    MemoryBudget,
    apply_scheduler_defaults,
    current_rss_gb,
    describe,
    detect_cpus,
    detect_memory_gb,
    limit_memory,
    limit_threads,
    peak_rss_gb,
    reset_peak_rss,
    thread_scope,
)

__all__ = ["MemoryBudget", "apply_scheduler_defaults", "current_rss_gb", "describe",
           "detect_cpus", "detect_memory_gb", "limit_memory", "limit_threads", "peak_rss_gb",
           "reset_peak_rss", "thread_scope"]

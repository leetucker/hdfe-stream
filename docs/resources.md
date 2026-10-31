# Resource limits and profiling

(quickstart)=
## Quickstart for cluster jobs

Two habits cover most of it.

**1. Profile your fits before you run them.** `profile=True` reads the data once and returns
the estimated memory and disk instead of fitting, so you can request the right resources:

```python
from hdfe_stream import feols_stream

p = feols_stream("log_earn ~ age | worker_id + firm_id", "data/*.parquet",
                 vcov={"CRV1": "worker_id"}, profile=True)
print(p.recommend())      # ncpus=8  mem=14gb  disk=40gb
print(p.recommend().pbs)  # select=1:ncpus=8:mem=14gb
```

**2. If your script imports Polars, run `stream_limits.set_limits()` *before* the import.**

```python
import stream_limits
stream_limits.set_limits()     # caps threads and Polars' memory at the job's allocation

import polars as pl            # only now
from hdfe_stream import feols_stream
```

Polars fixes its thread pool and its memory budget when it is imported, and nothing can change
them afterwards, so a limit set after the import cannot reach it (you get a warning that says
so). On a cluster, importing hdfe-stream applies the limits for you; the line above matters when
something imports Polars *before* hdfe-stream does, such as your own `import polars` at the top of the
script.

## What the packages do about resource limits

On a cluster, a job gets a fixed number of CPUs and a fixed amount of memory,
and is killed if it exceeds either. Two things make that hard to get right with
regression analysis in packages like `hdfe-stream`. It's not obvious in advance
how much memory a fit needs, because the amount of data in memory is indirectly
related to the source data's size. And, the packages sit on libraries that
each decide for themselves how many threads to use: Polars, numba, BLAS and
pyarrow all default to every core of the *node*, not of the job. These packages
then need to be constrained to avoid exceeding resource limits and leading your
job to be killed.

This page covers what hdfe-stream, logit-stream and dpd-stream do about both:

1. [`profile=True`](#profile) estimates a fit's memory and disk from one scan of
   the data, without fitting.
2. [Thread and memory limits](#limits): the packages cap themselves at a
   scheduler's allocation, using the standalone [`stream-limits`](https://leetucker.github.io/stream-limits/) package, which has
   its own documentation.
3. [`max_memory_gb`](#headroom): the choices that trade speed for memory follow
   the memory that is *left*, not a fixed share of the machine.
4. [`peak_rss_gb`](#measured): every fit reports the peak memory it used.

## A job script { #job-script }

```bash
#!/bin/bash
#PBS -l select=1:ncpus=8:mem=16gb
#PBS -l walltime=04:00:00

cd $PBS_O_WORKDIR
python profile_it.py     # step 1: the profile call above, printing p.recommend() (seconds, little memory)
python fit_it.py         # step 2: the real fit, with the request it printed
```

Nothing else in `fit_it.py` has to change: the packages read the job's CPU and
memory allocation themselves (see below), and `fit_it.py` needs the `set_limits()` line only if it
imports Polars first (see the [quickstart](#quickstart)).

(profile)=
## `profile=True`

Every estimator in the three packages takes `profile=True`: `feols_stream`,
`fepois_stream`, `feglm_stream`, `leave_out_kss`; logit-stream's
`mlogit_stream`, `ologit_stream`,
`oprobit_stream`, `clogit_stream`, `cfelogit_stream`, `nlogit_stream`,
`mixlogit_stream` (`logit_stream` and `probit_stream` are `feglm_stream` and profile as
it does); `dpd_stream`. The call validates the formula and the options exactly as a
fit does, scans the data once (streaming, the columns the fit uses), and returns a
{py:class}`~hdfe_stream.Profile` instead of a result. It writes no run directory.

```{eval-rst}
.. currentmodule:: hdfe_stream

.. autoclass:: Profile
   :members: memory_gb, disk_gb, phase_peaks_gb, recommend, to_dict
```

### Reading a profile

```
Profile: StreamingHDFE y ~ x1 + x2 | w + f + yr
  3,000,000 rows kept of 3,000,000 (scan 0.7s); 8 threads, Polars 8
  memory     1.96 GB expected      3.44 GB upper bound   (this process holds 0.86 GB now)
  disk       0.42 GB expected      0.53 GB upper bound
  by phase: pass0 1.96, pass1 1.22, assemble 1.22, residuals 1.22, solve 1.39
  request: ncpus=8  mem=3gb  disk=1gb   (select=1:ncpus=8:mem=3gb)
  components (GB expected / upper):
    explicit S (sparse, CSR)         0.376 /    2.402  ram    solve
    pass 0: sort one bucket          0.780 /    1.092  ram    pass0
    ...
```

- **Expected and upper bound.** Some quantities are known only after a pass:
  the number of distinct fixed-effect combinations, the nonzeros of the reduced
  matrix S (hdfe-stream), the instruments left after empty columns are dropped
  (dpd-stream), the outcome's levels when `levels=` is not given (logit-stream).
  The profile gives an expected size and an upper bound for each component.
  `recommend()` asks for the expected peak times a safety factor (1.25), but
  never more than the upper bound.
- **Phases.** A fit runs in phases (pass 0, pass 1, the solve, ...) and most
  arrays live in one or two of them, so the peak is the baseline plus the
  *largest phase*, not the sum of every component.
- **Baseline.** What the process already holds: imported libraries, compiled
  kernels, any DataFrame you have in memory. The fit holds it too. Profile in a
  fresh process (as in the job script) for the cleanest number. A process that has
  already fitted something holds the kernels in its baseline, and the profile's
  fixed allowance for them ("kernels and library state", about 0.5 GB) is then
  counted twice.
- **Threads.** The numbers are for the thread count shown. Accumulators have one
  copy per thread (hdfe-stream's Gram matrices, dpd-stream's `S1` and `Z'Z`,
  logit-stream's information matrices), so fewer threads is also less memory.
  Profile with the `n_threads=` you will fit with.
- **Memory-mapped files** count as memory *and* as disk. hdfe-stream memory-maps
  its identifying cells and the fixed-effect estimates; the pages are file-backed
  and the kernel can drop them, but a cgroup limit can charge them to the job.
  They are listed with `where = mapped`.
- **Disk** is the peak of the run directory (the sorted working copy, the
  memory-mapped cells, the result files). Parquet compression is not
  predictable, so the expected figure assumes 0.7 of the uncompressed size and
  the upper bound 1.0.
- **Limit and reserve.** The report says when the upper bound exceeds the usable
  memory (the limit less the reserve, see [below](#headroom)), and the
  `warnings` list the guards that would refuse the fit.

### How accurate is it

On the benchmark datasets used while developing this feature (hdfe-stream: 3 million rows,
OLS with one to three fixed effects, clustered, IV, Poisson and logit; dpd-stream:
difference and system GMM with 92 to 3,542 instruments; logit-stream: multinomial
and ordered logit, with a fixed effect, with clusters, and with two fixed effects),
the upper bound was above the measured peak in every case (1.6 to 2.6 times it), and the
expected figure was 2% to 70% above it (hdfe-stream 30% to 70%, dpd-stream 6% to
36%, logit-stream 2% to 34%): the model is deliberately conservative. The test
suites of the three packages check this on every run (`tests/test_profile.py`:
upper bound >= measured, expected within a factor of 2.5 of it). It is an estimate,
not a guarantee: ask for the upper bound when a job that gets killed is costly, and
use `diagnostics["peak_rss_gb"]` from a first real run to tighten the next request.

What is not modeled, and the profile says so in `warnings` or `notes`:
logit-stream's random intercepts (the EM's group-level arrays), the mixed and
nested logit's workspaces beyond the per-thread information accumulators, and
`leave_out_kss`'s pruning graph and trace vectors. Time is not estimated.

In logit-stream, the multinomial logit with absorbed fixed effects has additional sizing considerations; see its [documentation](https://leetucker.github.io/logit-stream/resources.html#the-multinomial-logit-with-several-absorbed-fixed-effects).

(limits)=
## Thread and memory limits

These are in the standalone [`stream-limits`](https://leetucker.github.io/stream-limits/) package, which depends on nothing but the
standard library, works without any regression package, and has its own documentation:

- [Thread limits](https://leetucker.github.io/stream-limits/threads.html): what Polars, numba, BLAS and pyarrow allow, and why the
  limits must be in place before they are imported.
- [Memory limits](https://leetucker.github.io/stream-limits/memory.html): the memory a job has, Polars' spill-to-disk budget, and
  `MemoryBudget`.
- [Schedulers and containers](https://leetucker.github.io/stream-limits/schedulers.html): PBS/`qstat`, Slurm, cgroups, and what to do on a system that does it
  differently. `python -m stream_limits` shows what was found, from inside a job.

What hdfe-stream, logit-stream and dpd-stream add:

- **Automatic limits.** Importing `hdfe_stream`, `logit_stream` or `dpd_stream` applies the job's
  allocation to Polars, numba, BLAS and pyarrow, and gives Polars 2.x its memory budget, **if it
  finds a limit**; with no limit found it does nothing, so a workstation is untouched.
- **`n_threads=`** holds numba and BLAS to that many threads for the length of one fit and
  restores them afterwards. It cannot affect Polars, whose pool is fixed for the process: for
  Polars, see the [quickstart](#quickstart).

(headroom)=
## `max_memory_gb`: choices that follow the memory left

Several places in the packages choose between a fast path that needs a lot of memory
and a slower one that needs less, or refuse an allocation that cannot shrink. They
used to compare with a fixed share of the machine's memory, which a correctly
sized PBS job always hit. They now ask whether *their exact allocation
fits in the headroom*:

> headroom = limit x (1 - reserve) - memory the process holds now

(this is {py:class}`stream_limits.MemoryBudget`; the [stream-limits documentation](https://leetucker.github.io/stream-limits/memory.html)
explains it for code of your own)

- **`max_memory_gb`** is the limit: the memory this fit may use. Default: the job's
  allocation (found as [described](https://leetucker.github.io/stream-limits/memory.html)),
  else the machine's. However, it is an option of every estimator, in case you
  want to constrain a fit further.
- **`memory_reserve`** is the share of the limit held back for what the estimates do not
  model: allocator slack, transient copies, pages a cgroup charges for files. Default
  0.15 (`STREAM_LIMITS_MEMORY_RESERVE`). Lower it to let the fast paths run closer to
  the limit; raise it for a job that must not be killed.

What follows the headroom:

| Package | Choice |
|---|---|
| hdfe-stream | the explicit matrix S is built if it fits and otherwise `stream_cg` streams it; per-thread accumulators drop to fewer copies (fewer threads on those passes) when they would not fit; pass 0's sort buckets shrink in a small allocation (10% of the headroom, between 64 and 384 MB); CRV3, the wild-bootstrap sums and the IV diagnostics' cluster sums are refused (or, for the diagnostics, skipped with a warning) up front, with the numbers, when they do not fit |
| logit-stream | the absorbed fixed effect's cross blocks (G x J x P), the random intercepts' blocks and CRV3's cluster information are refused up front when they do not fit (the first two were a fixed 8 GB) |
| dpd-stream | pass 1's per-thread moment accumulators, the cluster sums' copies and the absorbed cells' copies drop to fewer copies when they would not fit; cluster sums and absorbed cells that cannot shrink are refused (they were a fixed 8 GB) |

Decisions are made against the *exact* size of the allocation at
the point where it is made (S's nonzeros are counted before it is allocated), not the
profile's upper bound, so a job is not refused for an estimate's pessimism.

(measured)=
## Measured peak memory

Every fit records `diagnostics["peak_rss_gb"]`: the peak resident memory during
the fit, from the kernel's own high-water mark (`VmHWM`), read once at the end;
there is no polling, so it costs nothing. On Linux the counter is reset when the
fit starts, so a second fit in the same process reports its own peak; elsewhere
it is the process's lifetime peak. Child processes are not included.
`diagnostics["disk_peak_gb"]` is the disk counterpart.

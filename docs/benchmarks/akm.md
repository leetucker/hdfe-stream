# AKM regression

A standard three-way AKM specification,
`log_earn ~ age_squared + age_cubed | worker_id + firm_id + year`, clustered by
worker, on the simulated panel that ships with the library. It is compared with
[pyfixest](https://github.com/py-econometrics/pyfixest), with its default
demeaner (LSMR), with [xhdfe](https://github.com/reisportela/xhdfe-xfe), on its CPU
backend, and with R's [fixest](https://lrberge.github.io/fixest/) and Julia's
[FixedEffectModels.jl](https://github.com/FixedEffects/FixedEffectModels.jl).
Sizes run from 25,000 to 5,000,000 workers, with firms = workers / 15 at every
size and about 8.5 rows per worker, so the largest panel has 42.5 million rows.
Every configuration agrees with every other on the coefficients to within
2e-9, so these are measurements of equally good answers. Wall time includes
reading the data, because the other libraries need it in memory before they can
start and hdfe_stream reads it itself; [how it is
measured](index.md#how-it-is-measured) has the details.

```bash
python benchmarks/akm_benchmark.py
python benchmarks/plot_benchmarks.py
```

The numbers are in
[benchmarks/results/akm.csv](https://github.com/leetucker/hdfe-stream/blob/main/benchmarks/results/akm.csv).

```{image} ../figures/akm_time.light.svg
:class: only-light benchmark
:alt: AKM regression: wall time against the number of workers, one line per configuration, log-log
```
```{image} ../figures/akm_time.dark.svg
:class: only-dark benchmark
:alt: AKM regression: wall time against the number of workers, one line per configuration, log-log
```

```{image} ../figures/akm_memory.light.svg
:class: only-light benchmark
:alt: AKM regression: peak memory against the number of workers, one line per configuration, log-log
```
```{image} ../figures/akm_memory.dark.svg
:class: only-dark benchmark
:alt: AKM regression: peak memory against the number of workers, one line per configuration, log-log
```

```{image} ../figures/akm_disk.light.svg
:class: only-light benchmark
:alt: AKM regression: peak disk use of hdfe_stream against the number of workers, log-log
```
```{image} ../figures/akm_disk.dark.svg
:class: only-dark benchmark
:alt: AKM regression: peak disk use of hdfe_stream against the number of workers, log-log
```

At 5 million workers:

| configuration | wall time | peak memory | peak disk |
|---|---:|---:|---:|
| pyfixest (LSMR, its default) | 59 s | 22.0 GB | — |
| xhdfe | 79 s | 19.3 GB | — |
| fixest (R) | 152 s | 9.1 GB | — |
| FixedEffectModels.jl (Julia) | 79 s | 9.5 GB | — |
| hdfe_stream (`explicit`, its default) | 23 s | 3.9 GB | 2.6 GB |
| hdfe_stream (`stream_cg`) | 23 s | 3.8 GB | 2.6 GB |
| hdfe_stream (`stream_cg`, low memory) | 26 s | 3.9 GB | 2.7 GB |
| **hdfe_stream (`max_memory_gb=2`)** | **25 s** | **3.2 GB** | 2.6 GB |

## Reading the figures

**Memory is the point.** At 5 million workers pyfixest and xhdfe peak at
19–22 GB: pyfixest needed 22.0 GB of this machine's 30.8 GB. fixest and
FixedEffectModels.jl are leaner, at 9.1 and 9.5 GB. hdfe_stream peaks at
3.9 GB with its defaults. Telling it how much it may use (`max_memory_gb=2`)
brings that to 3.2 GB, for about 10% more time and the same answer, because
the fit then streams the reduced system instead of building it. Turning the
batch and bucket sizes down ("low memory": `rows_per_bucket=250_000`,
`batch_rows=200_000`) saves memory below 5 million workers (1.5 GB against
2.3 GB at a million) but not at 5 million, where the peak is set by something
else (see the floor below). What hdfe_stream spends instead is disk: about 61
bytes per row, 2.6 GB at 42.5 million rows, freed when the fit finishes. The
in-memory libraries' peak grows in proportion to the data. hdfe_stream's
low-memory setting grows far more slowly: 0.8 GB at 25,000 workers, 1.5 GB at a
million, 3.9 GB at five million.

**It is not slower for it.** From 400,000 workers up, hdfe_stream is the
fastest configuration measured: 2.4 s against pyfixest's 2.8 s and xhdfe's
3.4 s there, and at 5 million workers 23 s against 59 s for pyfixest, 79 s for
xhdfe and FixedEffectModels.jl, and 152 s for fixest. Below that the in-memory
libraries are faster, by a second or less: at 100,000 workers fixest takes
0.8 s, pyfixest 1.0 s and xhdfe 1.1 s, against hdfe_stream's 1.3 s. fixest's time grows faster
than the data from a million workers up (15 s there, 152 s for five times the
rows). With AKM panel data,
reducing rows to worker-firm cells before solving more than pays for the disk
traffic. The simulated panel has about 8.5 rows per worker and one cell per
row, which suits this approach; a specification where the cell table is no
smaller than the data, and the fixed effects are less local, may do worse.

**On small data the in-memory libraries use less memory.** Running any
hdfe_stream fit costs about 0.8 GB, most of it importing polars, numba and
scipy and starting Polars' streaming engine and numba's thread pools. On small
data that floor dominates: at 25,000 workers fixest peaks at 0.31 GB, xhdfe at
0.35 GB and pyfixest at 0.5 GB (FixedEffectModels.jl at 0.9 GB, most of it
the Julia runtime and compiled code). fixest stays the leanest up to 400,000
workers, at 1.0 GB against hdfe_stream's 1.8 GB (1.1 GB in the low-memory
setting) and 2.5 GB for xhdfe; at a million workers fixest,
FixedEffectModels.jl and hdfe_stream's defaults all need about 2.3–2.5 GB, and
from there hdfe_stream pulls away. On small panels, then, an in-memory library
is the lighter and quicker choice.

**hdfe_stream's `within` solver** is measured but no longer drawn; its rows
are in akm.csv. It is no faster than the default at any size here (1.0 s
against 1.0 s at 25,000 workers, 57 s against 23 s at 5 million), and uses
more memory at scale: 14.7 GB at 5 million workers, where `stream_cg` and
`explicit` stay near 4 GB.

## Trading memory for time

Same data, same solver, same answer, varying only how much is in flight at
once:

```bash
python benchmarks/akm_benchmark.py --memory-sweep 1000000
```

**8,500,353 rows, 1,000,000 worker effects**

| `rows_per_bucket` | `batch_rows` | wall time | peak memory | peak disk |
|---:|---:|---:|---:|---:|
| 20,000,000 (default) | 2,000,000 | 4.9 s | 2,554 MB | 513 MB |
| 1,000,000 | 500,000 | 5.0 s | 1,571 MB | 518 MB |
| 250,000 | 200,000 | 5.5 s | 1,518 MB | 527 MB |
| 100,000 | 100,000 | 5.8 s | 1,619 MB | 526 MB |

Most of the saving comes from the first step down, at almost no cost in time;
the second saves a little more for 10% more time, and below that the peak rises
again. The defaults are tuned for a machine with room to spare; if memory is
the binding constraint, set `rows_per_bucket` to something near what you can
afford and leave the rest alone (with many covariates, scale `batch_rows` and
`row_group_size` down too; see [many covariates](covariates.md)). `rhs_block`
bounds the other big array (levels × variables).

**There is a floor, and it is specific to the regression design.** Turning these
settings down only reduces the memory that is proportional to the batches and
buckets. What is left grows with the number of workers (fixed-effect levels and
clusters), and no setting shrinks it. In this benchmark's design (worker, firm
and year effects, clustered by worker) that floor is about 0.8 GB of fixed
overhead plus roughly 0.5 KB per worker: about 1.5 GB at a million workers and
3.2 GB at five million (the `max_memory_gb=2` fit reached 3.2 GB; the low-memory
setting 3.9 GB). That is why the low-memory curve flattens at the largest
sizes. A different design (fewer levels, no clustering by a high-cardinality
variable) has a different floor, so these figures describe this benchmark, not a
general limit.

`max_memory_gb` (by default the job's allocation) decides whether the
explicit reduced matrix is built, falling back to `stream_cg` by itself when it
does not fit. With `max_memory_gb=2` it did so at 5 million workers (and not at
the smaller sizes, where the matrix fits), which is why that fit peaked lower
than the defaults and took 10% longer. [Out of core](../examples/out_of_core.md#memory-budgets) describes
each option.

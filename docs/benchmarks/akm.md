# AKM regression

A standard three-way AKM specification,
`log_earn ~ age_squared + age_cubed | worker_id + firm_id + year`, clustered by
worker, on the simulated panel that ships with the library. It is compared with
[pyfixest](https://github.com/py-econometrics/pyfixest), under both of its
demeaners, with [xhdfe](https://github.com/reisportela/xhdfe-xfe), on its CPU
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
| pyfixest (MAP, its default) | 919 s | 20.8 GB | — |
| pyfixest (LSMR) | 74 s | 27.0 GB | — |
| xhdfe | 79 s | 19.3 GB | — |
| fixest (R) | 152 s | 9.1 GB | — |
| FixedEffectModels.jl (Julia) | 79 s | 9.5 GB | — |
| hdfe_stream (`explicit`, its default) | 22 s | 3.9 GB | 2.6 GB |
| hdfe_stream (`stream_cg`) | 22 s | 3.8 GB | 2.6 GB |
| **hdfe_stream (`stream_cg`, low memory)** | **24 s** | **3.7 GB** | 2.7 GB |

## Reading the figures

**Memory is the point.** At 5 million workers pyfixest and xhdfe peak at
19–27 GB: pyfixest's LSMR run needed 27.0 GB of this machine's 30.8 GB. fixest
and FixedEffectModels.jl are leaner, at 9.1 and 9.5 GB. hdfe_stream peaks at
3.9 GB with its defaults and 3.7 GB with the batch and
bucket sizes turned down ("low memory": `rows_per_bucket=250_000`,
`batch_rows=200_000`), for about 10% more time and the same answer. What it
spends instead is disk: about 61 bytes per row, 2.6 GB at 42.5 million rows,
freed when the fit finishes. The in-memory libraries' peak grows in proportion
to the data. hdfe_stream's low-memory setting grows far more slowly: 0.8 GB at
25,000 workers, 1.4 GB at a million, 3.7 GB at five million.

**It is not slower for it.** From 400,000 workers up, hdfe_stream is the
fastest configuration measured: 2.5 s against xhdfe's 3.4 s there, and at 5
million workers 22 s against 74 s for pyfixest (LSMR; 919 s with MAP), 79 s for
xhdfe and FixedEffectModels.jl, and 152 s for fixest. Below that the in-memory
libraries are faster, by a second or less: at 100,000 workers fixest takes
0.8 s and xhdfe 1.1 s, against hdfe_stream's 1.4 s. fixest's time grows faster
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
0.35 GB and pyfixest at about 0.6 GB (FixedEffectModels.jl at 0.9 GB, most of it
the Julia runtime and compiled code). fixest stays the leanest up to 400,000
workers, at 1.0 GB against hdfe_stream's 1.8 GB (1.1 GB in the low-memory
setting) and 2.5 GB for xhdfe; at a million workers fixest,
FixedEffectModels.jl and hdfe_stream's defaults all need about 2.3–2.5 GB, and
from there hdfe_stream pulls away. On small panels, then, an in-memory library
is the lighter and quicker choice.

**The default pyfixest demeaner is not the one to compare against.**
`MapDemeaner` (alternating projections) is 9 to 12 times slower than
`LsmrDemeaner` from 400,000 workers up, since many worker effects are the case
alternating projections struggles with. If you are comparing, compare against
LSMR.

**hdfe_stream's `within` solver** is measured but no longer drawn; its rows
are in akm.csv. It is no faster than the default at any size here (1.18 s
against 1.15 s at 25,000 workers, 49 s against 22 s at 5 million), and uses
more memory at scale: 15.3 GB at 5 million workers, where `stream_cg` and
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
| 20,000,000 (default) | 2,000,000 | 4.9 s | 2,411 MB | 512 MB |
| 1,000,000 | 500,000 | 4.9 s | 1,532 MB | 518 MB |
| 250,000 | 200,000 | 5.3 s | 1,457 MB | 526 MB |
| 100,000 | 100,000 | 5.7 s | 1,521 MB | 524 MB |

Most of the saving comes from the first step down, at no cost in time; the
second saves a little more for 8% more time, and below that the peak rises
again. The defaults are tuned for a machine with room to spare; if memory is
the binding constraint, set `rows_per_bucket` to something near what you can
afford and leave the rest alone (with many covariates, scale `batch_rows` and
`row_group_size` down too; see [many covariates](covariates.md)). `rhs_block`
bounds the other big array (levels × variables), and `max_s_gb` caps the
explicit reduced matrix, above which `solver="auto"` falls back to `stream_cg`
by itself. [Out of core](../examples/out_of_core.md#memory-budgets) describes
each option.

# AKM regression

A standard three-way AKM specification,
`log_earn ~ age_squared + age_cubed | worker_id + firm_id + year`, clustered by
worker, on the simulated panel that ships with the library. It is compared with
[pyfixest](https://github.com/py-econometrics/pyfixest), under both of its
demeaners, and with [xhdfe](https://github.com/reisportela/xhdfe-xfe), on its
CPU backend. Sizes run from 25,000 to 5,000,000 workers, with firms = workers /
15 at every size and about 8.5 rows per worker, so the largest panel has 42.5
million rows. Every configuration agrees with every other on the coefficients
to within 3e-10, so these are measurements of equally good answers. Wall time
includes reading the data, because pyfixest and xhdfe need it in memory before
they can start and hdfe_stream reads it itself; [how it is
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
| pyfixest (MAP, its default) | 1,960 s | 20.7 GB | — |
| pyfixest (LSMR) | 138 s | 25.3 GB | — |
| xhdfe | 238 s | 19.0 GB | — |
| hdfe_stream (`explicit`, its default) | 52 s | 4.1 GB | 4.1 GB |
| hdfe_stream (`stream_cg`) | 54 s | 4.0 GB | 4.1 GB |
| **hdfe_stream (`stream_cg`, low memory)** | **60 s** | **3.2 GB** | 4.3 GB |

## Reading the figures

**Memory is the point.** At 5 million workers the in-memory libraries peak at
19–25 GB: pyfixest's LSMR run needed almost all of this machine's 26 GB.
hdfe_stream peaks at 4.1 GB with its defaults and 3.2 GB with the batch and
bucket sizes turned down ("low memory": `rows_per_bucket=250_000`,
`batch_rows=200_000`), for about 16% more time and the same answer. What it
spends instead is disk: about 97 bytes per row, 4.1 GB at 42.5 million rows,
freed when the fit finishes. The in-memory libraries' peak grows in proportion
to the data. hdfe_stream's low-memory setting grows far more slowly: 0.9 GB at
25,000 workers, 1.4 GB at a million, 3.2 GB at five million.

**It is not slower for it.** From about a million workers up, hdfe_stream is
the fastest configuration measured, and at 400,000 it is within half a second
of xhdfe (4.1 s against 3.7 s). At 5 million workers it takes 52 s against
pyfixest's 138 s (LSMR) or 1,960 s (MAP) and xhdfe's 238 s. With AKM panel data,
reducing rows to worker-firm cells before solving more than pays for the disk
traffic. The simulated panel has about 8.5 rows per worker and one cell per
row, which suits this approach; a specification where the cell table is no
smaller than the data, and the fixed effects are less local, may do worse.

**Below about 400,000 workers, the in-memory libraries use less memory.**
Running any hdfe_stream fit costs about 0.9 GB, most of it importing polars,
numba and scipy and starting Polars' streaming engine and numba's thread pools.
On small data that floor dominates: at 25,000 workers xhdfe peaks at 0.29 GB and
pyfixest at about 0.5 GB. This is the concrete version of "use pyfixest (or
xhdfe) if your data fits".

**The default pyfixest demeaner is not the one to compare against.**
`MapDemeaner` (alternating projections) is 6 to 14 times slower than
`LsmrDemeaner` from 400,000 workers up, since many worker effects are the case
alternating projections struggles with. If you are comparing, compare against
LSMR.

**hdfe_stream's `within` solver** uses more memory at scale: 14.9 GB at 5
million workers, where `stream_cg` and `explicit` stay near 4 GB. Prefer
those when memory is the constraint.

## Trading memory for time

Same data, same solver, same answer, varying only how much is in flight at
once:

```bash
python benchmarks/akm_benchmark.py --memory-sweep 1000000
```

**8,500,353 rows, 1,000,000 worker effects**

| `rows_per_bucket` | `batch_rows` | wall time | peak memory | peak disk |
|---:|---:|---:|---:|---:|
| 20,000,000 (default) | 2,000,000 | 11.2 s | 2,341 MB | 818 MB |
| 1,000,000 | 500,000 | 11.0 s | 1,849 MB | 821 MB |
| 250,000 | 200,000 | 11.6 s | 1,384 MB | 863 MB |
| 100,000 | 100,000 | 11.6 s | 1,555 MB | 876 MB |

The saving comes from the first two steps down; below that, the peak rises
again. The defaults are tuned for a machine with room to spare; if memory is
the binding constraint, set `rows_per_bucket` to something near what you can
afford and leave the rest alone (with many covariates, scale `batch_rows` and
`row_group_size` down too; see [many covariates](covariates.md)). `rhs_block`
bounds the other big array (levels × variables), and `max_s_gb` caps the
explicit reduced matrix, above which `solver="auto"` falls back to `stream_cg`
by itself. [Out of core](../examples/out_of_core.md#memory-budgets) describes
each option.

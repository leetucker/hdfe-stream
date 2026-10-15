# Many covariates

The same kind of comparison as the [AKM regression](akm.md), holding the data
fixed and varying its width: the million-worker panel (8.5 million rows, 66,665
firms) with age entered as indicators,
`log_earn ~ i(age_bin) | worker_id + firm_id`, clustered by worker, in bins from
five years wide (8 indicators) down to one month (503). hdfe_stream runs at its
defaults and "sized to the design", with bucket, batch and row-group sizes
scaled down as the design widens (the rule is below). Every configuration that
finished agrees with every other to within 1e-8.

```bash
python benchmarks/covariates_benchmark.py
```

The numbers are in
[benchmarks/results/covariates.csv](https://github.com/leetucker/hdfe-stream/blob/main/benchmarks/results/covariates.csv).

```{image} ../figures/covariates_memory.light.svg
:class: only-light benchmark
:alt: Many covariates: peak memory against the number of covariates, one line per configuration, log-log
```
```{image} ../figures/covariates_memory.dark.svg
:class: only-dark benchmark
:alt: Many covariates: peak memory against the number of covariates, one line per configuration, log-log
```

```{image} ../figures/covariates_time.light.svg
:class: only-light benchmark
:alt: Many covariates: wall time against the number of covariates, one line per configuration, log-log
```
```{image} ../figures/covariates_time.dark.svg
:class: only-dark benchmark
:alt: Many covariates: wall time against the number of covariates, one line per configuration, log-log
```

```{image} ../figures/covariates_disk.light.svg
:class: only-light benchmark
:alt: Many covariates: peak disk use of hdfe_stream against the number of covariates, log-log
```
```{image} ../figures/covariates_disk.dark.svg
:class: only-dark benchmark
:alt: Many covariates: peak disk use of hdfe_stream against the number of covariates, log-log
```

Wall time and peak memory ("—": ran out of this machine's 26 GB):

| covariates | pyfixest (MAP) | pyfixest (LSMR) | xhdfe | hdfe_stream | hdfe_stream, sized |
|---:|---:|---:|---:|---:|---:|
| 8 | 132 s, 5.7 GB | 16 s, 7.3 GB | 15 s, 9.1 GB | 7 s, 3.2 GB | 7 s, 3.0 GB |
| 41 | 520 s, 13.8 GB | 62 s, 18.2 GB | 60 s, 18.2 GB | 22 s, 6.9 GB | 20 s, 3.7 GB |
| 83 | 976 s, 25.2 GB | 156 s, 25.8 GB | 125 s, 26.0 GB | 58 s, 12.4 GB | 44 s, 4.9 GB |
| 167 | — | — | — | 127 s, 23.0 GB | 81 s, 5.1 GB |
| 503 | — | — | — | — | **329 s, 8.6 GB** |

## Reading the figures

**The in-memory libraries run out first.** Their peak grows with rows ×
covariates. At 8.5 million rows all three were within about 1 GB of this
machine's 26 GB at 83 indicators, and out of memory at 167. hdfe_stream sized
to the design goes from 3.0 GB at 8 indicators to 5.1 GB at 167 and 8.6 GB at
503. Of that last figure, about 4.9 GB is the memory-mapped cell table: file
pages the kernel can drop under pressure. The memory the fit itself allocated
peaked at 4.6 GB (a separate profile, reading the process's anonymous and
file-backed pages apart).

**Size the batches to the design.** At its defaults hdfe_stream holds a whole
bucket of up to 20 million rows, with every covariate, while it sorts; that is
fine for a handful of covariates and not for hundreds (23.0 GB at 167, out of
memory at 503). Every step that holds rows holds all the covariates, so scale
`rows_per_bucket`, `batch_rows` and `row_group_size` down with the width. The
benchmark aims at about 1 GB of design per bucket and 250 MB per batch and row
group, never above the defaults:

```python
per_row = 8 * (k + 4)                   # k covariates, plus outcome and ids
batch = min(2_000_000, int(250e6 // per_row))
fit = feols_stream(fml, data, rows_per_bucket=min(20_000_000, int(1e9 // per_row)),
                   batch_rows=batch, row_group_size=min(500_000, batch))
```

[Wide designs](../examples/wide_designs.md) puts this together.

**Time grows with the square of the width.** The cross-products cost k × k per
row; they are formed chunk by chunk as BLAS matrix products, and 503 indicators
take 5½ minutes where 167 take under 1½. hdfe_stream is the fastest
configuration at every width measured.

**Timings on this machine are noisy where memory is tight.** Each point is one
run. Repeated runs of the default configuration at 83 indicators, which holds
12.4 GB, took between 53 and 83 s; the sized configuration varied by about 15%.
The differences between libraries above are far larger than that.

**Disk grows with the width too**, since the working copy of the rows carries
every covariate: 0.3 GB at 8 indicators, 5.5 GB at 503.

**How the design is built matters.** A formula term such as `i(age_bin)` is
carried through the partitioning as the one column it is computed from, and the
indicators are built one bucket at a time. An expression that needs the whole
column, possible only with the low-level interface, makes the fit build the
design up front instead; see [wide designs](../examples/wide_designs.md).

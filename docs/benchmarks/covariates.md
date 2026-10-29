# Many covariates

The same kind of comparison as the [AKM regression](akm.md), holding the data
fixed and varying its width: the million-worker panel (8.5 million rows, 66,665
firms) with age entered as indicators,
`log_earn ~ i(age_bin) | worker_id + firm_id`, clustered by worker, in bins from
five years wide (8 indicators) down to one month (503). It is compared with
pyfixest (both demeaners), xhdfe, R's fixest and Julia's FixedEffectModels.jl.
hdfe_stream runs at its defaults and "sized to the design", with bucket, batch
and row-group sizes scaled down as the design widens (the rule is below). Every
configuration that finished agrees with every other to within 1e-7.

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

Wall time and peak memory ("—": out of memory at this machine's limit, 29.6 GB
of its 30.8 GB):

| covariates | pyfixest (MAP) | pyfixest (LSMR) | xhdfe | fixest (R) | FixedEffectModels.jl | hdfe_stream | hdfe_stream, sized |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 90 s, 5.8 GB | 13 s, 8.6 GB | 12 s, 9.5 GB | 18 s, 3.5 GB | 23 s, 3.3 GB | 5 s, 2.4 GB | 5 s, 3.3 GB |
| 41 | 280 s, 13.9 GB | 34 s, 21.0 GB | 36 s, 22.4 GB | 63 s, 10.5 GB | 89 s, 7.8 GB | 11 s, 4.4 GB | 11 s, 3.8 GB |
| 83 | 507 s, 25.2 GB | — | — | 119 s, 19.4 GB | 177 s, 13.7 GB | 21 s, 4.9 GB | 21 s, 4.5 GB |
| 167 | — | — | — | — | 322 s, 24.1 GB | 43 s, 8.0 GB | 43 s, 5.4 GB |
| 503 | — | — | — | — | — | 102 s, 21.9 GB | 114 s, 10.9 GB |

## Reading the figures

**The in-memory libraries run out first.** Their peak grows with rows ×
covariates. At 8.5 million rows pyfixest's LSMR demeaner and xhdfe were out of
memory at 83 indicators, and MAP used 25.2 GB there and was out of memory at
167, as was fixest. FixedEffectModels.jl, the leanest of them, finished 167
indicators in 24.1 GB; at 503 the design alone is 34 GB, and none of them could
hold it. hdfe_stream sized to the design goes from 3.3 GB at 8
indicators to 5.4 GB at 167 and 10.9 GB at 503. Part of that last figure is the
memory-mapped cell table, whose pages count towards the peak although the
kernel can drop them under pressure, so the memory the fit itself needs is less.

**Size the batches to the design.** At its defaults hdfe_stream sorts buckets of about 384 MB of rows,
but each batch is 2 million rows with every covariate, which is 8 GB at 503; that
is fine for a handful of covariates and costly for hundreds (8.0 GB at 167 and
21.9 GB at 503, against 5.4 and 10.9 GB sized to the design). Every step that holds rows holds all the covariates, so scale
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

**Time grows with the width.** The cross-products cost k × k per row; they are
formed chunk by chunk as BLAS matrix products. With this machine's 16 cores
the time grew about in proportion to the width from 83 indicators up, though
(2.1 times from 83 to 167, 2.6 times from 167 to 503), and 503 indicators took
2 minutes where 167 took 43 seconds. hdfe_stream is the fastest
configuration at every width measured.

**Timings are noisy where memory is tight.** Each point is one run, so
differences of 15% or so between configurations should not be read as
meaningful. The differences between libraries above are far larger than that.

**Disk grows with the width too**, since the working copy of the rows carries
every covariate: 0.3 GB at 8 indicators, 5.3 GB at 503.

**How the design is built matters.** A formula term such as `i(age_bin)` is
carried through the partitioning as the one column it is computed from, and the
indicators are built one bucket at a time. An expression that needs the whole
column, possible only with the low-level interface, makes the fit build the
design up front instead; see [wide designs](../examples/wide_designs.md).

# CPU and GPU

hdfe_stream has two experimental GPU options: `device="cuda"` runs the solve on
an NVIDIA GPU, through CuPy, and `polars_engine="gpu"` runs the fit's Polars
queries on the cudf-polars GPU engine. This benchmark measures what they buy,
on the [AKM regression](akm.md)
(`log_earn ~ age_squared + age_cubed | worker_id + firm_id + year`, clustered
by worker, 25,000 to 5,000,000 workers), on one machine with an NVIDIA H100,
against the two other libraries with a GPU path:
[pyfixest](https://github.com/py-econometrics/pyfixest)'s LSMR demeaner, on its
default CPU backend and on PyTorch on the GPU, and
[xhdfe](https://github.com/reisportela/xhdfe-xfe), on its CPU and CUDA
backends. Every library on the CPU is the reference point for itself on the
GPU, on the same machine. All seven configurations agree on the coefficients
to within 2e-10 at every size.

```bash
python benchmarks/gpu_benchmark.py      # on a machine with an NVIDIA GPU
python benchmarks/plot_benchmarks.py
```

The numbers are in
[benchmarks/results/gpu.csv](https://github.com/leetucker/hdfe-stream/blob/main/benchmarks/results/gpu.csv).

```{image} ../figures/gpu_time.light.svg
:class: only-light benchmark
:alt: AKM regression on a GPU machine: wall time against the number of workers, each library on the CPU and on the GPU, log-log
```
```{image} ../figures/gpu_time.dark.svg
:class: only-dark benchmark
:alt: AKM regression on a GPU machine: wall time against the number of workers, each library on the CPU and on the GPU, log-log
```

```{image} ../figures/gpu_memory.light.svg
:class: only-light benchmark
:alt: AKM regression on a GPU machine: peak memory against the number of workers, each library on the CPU and on the GPU, log-log
```
```{image} ../figures/gpu_memory.dark.svg
:class: only-dark benchmark
:alt: AKM regression on a GPU machine: peak memory against the number of workers, each library on the CPU and on the GPU, log-log
```

```{image} ../figures/gpu_gpu_memory.light.svg
:class: only-light benchmark
:alt: AKM regression on a GPU machine: peak GPU memory of the GPU configurations against the number of workers, log-log
```
```{image} ../figures/gpu_gpu_memory.dark.svg
:class: only-dark benchmark
:alt: AKM regression on a GPU machine: peak GPU memory of the GPU configurations against the number of workers, log-log
```

At 1 and 5 million workers (memory: the process's peak, then the GPU's):

| configuration | 1M workers | 5M workers |
|---|---:|---:|
| hdfe_stream, CPU | 6.2 s, 4.4 GB | 28.0 s, 9.0 GB |
| hdfe_stream, GPU solve | 6.6 s, 4.9 + 0.6 GB | 28.2 s, 9.9 + 0.8 GB |
| hdfe_stream, GPU solve and GPU Polars | 8.4 s, 4.3 + 4.7 GB | 25.6 s, 6.6 + 10.7 GB |
| pyfixest (LSMR), CPU | 13.3 s, 6.0 GB | 101.6 s, 27.0 GB |
| pyfixest (LSMR), GPU | 13.7 s, 5.8 + 3.8 GB | 133.2 s, 23.4 + 16.5 GB |
| xhdfe, CPU | 12.7 s, 5.0 GB | 106.6 s, 19.3 GB |
| xhdfe, GPU | **4.2 s**, 3.0 + 1.8 GB | **24.3 s**, 12.7 + 7.0 GB |

## Reading the figures

**hdfe_stream gains little from the GPU, and only at the largest size.** Its
design keeps the solve small: the rows are reduced to sums over worker-firm
cells before solving, so at 5 million workers the solve is 3.8 of 28 seconds,
and on the GPU it took 3.6. Moving the Polars queries to the GPU as well saved
2.4 seconds there (9%) and 2.4 GB of host memory, with the data held in 10.7 GB
of GPU memory instead. Up to a million workers the solve itself was slower on
the GPU, and with the Polars queries on the GPU too the whole fit was slower,
by up to 3 seconds on the smallest panels, where starting the GPU engines costs
more than they save. Two of the fit's Polars queries run on the CPU even then,
as `diagnostics["polars_gpu_fallbacks"]` reports: cudf-polars does not yet
support the approximate distinct count in pass 0's scan, or the partitioned
write that follows it.

**pyfixest's PyTorch backend gained nothing reliable.** It was slower than
pyfixest's CPU backend at every size here; at 5 million workers it took 133 s
against 102 s on the CPU, and 93 s in an earlier run on an identical machine,
the one measurement that moved between the two runs. pyfixest describes the
backend as experimental.

**xhdfe gains the most:** its CUDA backend is 3 to 4.4 times as fast as its CPU
backend from a million workers up, and the fastest configuration on this
machine from 100,000 workers up. At 5 million workers it took 24.3 s,
against 28.0 s for hdfe_stream on this machine's CPU and 25.6 s with both of
hdfe_stream's GPU options, with 12.7 GB of host memory and 7.0 GB on the GPU.

## Should I use a GPU?

For this problem, the evidence here says probably not. A GPU-first library
gains a lot from the GPU, but against its own CPU path; against a CPU-first one
the gain is small, and the GPU machine costs far more. In the
[AKM benchmark](akm.md), on a 16-vCPU CPU machine (a `c8id.4xlarge`),
hdfe_stream fitted 5 million workers in 22 s and 3.9 GB. That is as fast as
xhdfe on the H100 (24 s, and 12.7 GB of memory plus 7.0 GB on the GPU), on a
machine that costs about an eighth as much per hour: $0.89 an hour on demand,
against $6.88 for the `p5.4xlarge`, AWS's smallest H100 machine.

If you want to try a GPU yourself:

- **For hdfe_stream, expect modest gains, and only on large problems.** The
  options are experimental, and on this benchmark they paid off only at the
  largest size, by about 10%.
- **If GPU performance is the priority, a GPU-oriented library such as xhdfe is
  the better choice.** Its CUDA backend made the most of the card here.
- **The GPU needs fast double precision.** These fits run in float64, which is
  fast only on data-center GPUs (A100, H100, B200); consumer and inference
  cards run it at a small fraction of their single-precision speed.

## Using the GPU options

Both are experimental, and both leave the estimates unchanged: they agree with
the CPU's to within the solver tolerance.

```bash
pip install cupy-cuda13x cudf-polars-cu13      # or the builds for CUDA 12
```

```python
fit = feols_stream(fml, "panel.parquet", device="cuda", polars_engine="gpu")
```

- `device="cuda"` builds the reduced system on the CPU as usual, copies it to
  the GPU once, and runs the conjugate-gradient iterations there. It needs
  `solver="auto"` or `"explicit"`; above `max_s_gb` the solve falls back to the
  CPU, as it would without a GPU.
- `polars_engine="gpu"` sends the fit's Polars queries to cudf-polars, and runs
  any query the engine cannot on the CPU instead.
- cudf-polars supports a narrow range of Polars versions (version 26.6 needs
  Polars 1.39), so installing it can change your Polars version.

## How it was measured

On an AWS EC2 `p5.4xlarge`: one NVIDIA H100 (80 GB), 16 vCPUs of an AMD EPYC
7R13, NVIDIA driver 595.91 (CUDA 13.2), and the instance's local NVMe drive for
the data. The benchmarks were limited to 29.6 GB of memory, as on the CPU
machine, so that every configuration had the same room as there. This was
hdfe_stream 0.5.0, the release that added the GPU options (the other benchmarks
ran on 0.4.0). Python and its packages were those of the other benchmarks,
except Polars 1.39.3, which cudf-polars 26.6 needs (2.0.0 elsewhere), with CuPy
14.2.0, cudf-polars 26.6.0 and PyTorch 2.14.1 (CUDA 13.0). The CPU numbers here are therefore not those of
the other pages: this is a different, older CPU, with an older Polars, and
hdfe_stream took 28 s at 5 million workers here against 22 s there.

- **Peak GPU memory** is the configuration's process's, polled from
  `nvidia-smi`. Each configuration runs in a fresh process, so the GPU ones pay
  for starting CUDA each time, as a script would. What is compiled once and
  cached is not charged: numba's, CuPy's and cudf's kernels, and pyfixest's
  `torch.compile`, are warmed up on the smallest panel first.
- **pyfixest** silently falls back to a CPU demeaner if PyTorch is missing, so
  a GPU result is kept only if PyTorch saw the GPU and the fit allocated memory
  on it.
- **xhdfe** is commit `f71420b` (2.28.1) built with `XHDFE_ENABLE_CUDA=ON`, with
  one change: its source calls `cub::Max`, which the CUDA 13 toolkit removed,
  and the build replaces it with `cuda::maximum<>{}`, the same operation. Its
  result is kept only if xhdfe reports that it used the GPU.
- Each point is one run. A second complete run on an identical machine agreed
  with it to within a few percent, except for pyfixest's GPU backend at 5
  million workers (above).

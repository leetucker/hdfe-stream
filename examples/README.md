# Examples

Every example runs on simulated data, so there is nothing to download and
nothing to configure. Install the package and run any file:

```bash
pip install -e ".[formula]"      # pyfixest is needed for formula syntax
python examples/quickstart.py
```

The first run writes a simulated panel to `examples/output/`; later runs reuse
it. Delete that directory to start fresh — it also holds each example's scratch
space and result files, and nothing in it is needed afterwards.

Run them from anywhere; paths are resolved relative to the example file, not
your working directory.

## The files

| File | What it shows |
|---|---|
| [`quickstart.py`](quickstart.py) | The smallest useful thing: fit a three-way fixed-effects model on a Parquet file and read the results. Start here. |
| [`akm_variance.py`](akm_variance.py) | The application this library was written for — decomposing the variance of log earnings into worker, firm and sorting components, aggregated in a streaming pass so the rows are never collected. |
| [`formulas.py`](formulas.py) | Formula syntax: transformations, categoricals, event studies, interacted fixed effects, several models in one call, missing values, collinear terms, and clustering on anything. |
| [`weights_and_iv.py`](weights_and_iv.py) | Analytic and frequency weights, and 2SLS with a first-stage F. |
| [`varying_slopes.py`](varying_slopes.py) | Worker-specific time trends (FEIS), how the streamed dimension is chosen, and the low-level `StreamingHDFE` interface. |
| [`out_of_core.py`](out_of_core.py) | What is held in memory and what is not, the three solvers, the memory knobs, disk lifecycle, and logging. Read this before pointing the library at something large. |
| [`reporting.py`](reporting.py) | Regression tables and event-study plots via pyfixest, mixing streaming and in-memory models. |

[`simulated_data.py`](simulated_data.py) is the shared helper the examples
import for their data paths and scratch directories. It is not an example
itself, but it is short and worth a look if you want to know what the simulated
columns mean.

## The simulated data

`hdfe_stream.simulate` is part of the installed package, so you can use it
outside these examples:

```python
from hdfe_stream.simulate import simulate_akm

panel = simulate_akm(n_workers=50_000)
panel.write_parquet("sim.parquet")
```

- `simulate_akm` — the base matched worker–firm panel. Workers move between
  firms, and log earnings are a worker effect plus a firm effect plus an age
  profile plus noise. Effects are positively assorted, so the fixed effects
  genuinely have to be estimated jointly, and there is enough mobility to link
  the firms into one connected component.
- `simulate_rich` — adds categoricals, cluster variables that are not fixed
  effects, weights, an instrumental-variables block, a second outcome, and a
  column with missing values.
- `simulate_trends` — adds worker-specific time trends, for varying slopes.

All three take a `seed` and are deterministic. Because the worker and firm
effects are drawn explicitly, you know the right answer, which is what makes
these useful for checking your own code as well as ours.

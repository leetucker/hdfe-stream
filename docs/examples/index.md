# Examples

Each page here follows one of the scripts in
[`examples/`](https://github.com/leetucker/hdfe-stream/tree/main/examples) of
the repository, explains what it shows, and ends with the script itself. They
all run on simulated data, so there is nothing to download and nothing to
configure. Clone the repository, install the package and run any of them:

```bash
pip install -e ".[formula]"      # pyfixest is needed for formula syntax
python examples/quickstart.py
```

The first run writes a simulated panel of 20,000 workers (about 170,000 rows)
to `examples/output/`; later runs reuse it. Delete that directory to start
fresh. It also holds each example's scratch space and result files, and nothing
in it is needed afterwards. Run them from anywhere: paths are resolved relative
to the example file, not your working directory. The outputs shown on these
pages are from those runs, abridged.

| | |
|---|---|
| [Quickstart](quickstart.md) | fit a three-way fixed-effects model on a Parquet file and read the results. Start here. |
| [AKM variance decomposition](akm_variance.md) | the application the library was written for, aggregated in a streaming pass |
| [Formulas, standard errors and the sample](formulas.md) | formula syntax end to end, missing values and `sample()`, clustering, degrees of freedom |
| [Weights and IV](weights_and_iv.md) | analytic and frequency weights, and 2SLS with a first-stage F |
| [One fixed effect, or none](fewer_fixed_effects.md) | a within regression, OLS with an intercept, and `csw0()` |
| [Poisson, logit and probit](glm.md) | `fepois_stream` and `feglm_stream`: how they work, separation, and the differences from pyfixest |
| [Wide designs](wide_designs.md) | hundreds of covariates, and sizing the batches to them |
| [Varying slopes](varying_slopes.md) | worker-specific trends, the streamed dimension, and the low-level interface |
| [Out of core](out_of_core.md) | memory, disk, solvers and logging: read this before pointing the library at something large |
| [Reporting](reporting.md) | regression tables and plots via pyfixest |
| [Connected sets](connected_sets.md) | connected components, `min_degree` pruning and the leave-one-out connected set, compared |
| [Leave-out variance components](leave_out_kss.md) | the KSS bias correction, checked against the effects the data was built from |

## The simulated data

`hdfe_stream.simulate` ships with the package, so you can try the library, or
check your own code, against data whose answer you know:

```python
from hdfe_stream.simulate import simulate_akm

simulate_akm(n_workers=1_000_000).write_parquet("sim.parquet")
```

- {py:func}`simulate_akm <hdfe_stream.simulate.simulate_akm>`
  is the base matched worker–firm panel. Workers move between firms, and log
  earnings are a worker effect plus a firm effect plus an age profile plus
  noise. Effects are positively assorted, so the fixed effects genuinely have to
  be estimated jointly, and there is enough mobility to link the firms into one
  connected component.
- {py:func}`simulate_rich <hdfe_stream.simulate.simulate_rich>`
  adds categoricals, cluster variables that are not fixed effects, weights, an
  instrumental-variables block, a second outcome, and a column with missing
  values.
- {py:func}`simulate_trends <hdfe_stream.simulate.simulate_trends>`
  adds worker-specific time trends, for varying slopes.
- {py:func}`simulate_bottleneck <hdfe_stream.simulate.simulate_bottleneck>`
  builds a panel whose variance components are weakly identified: two blocks of
  firms joined by a handful of movers.

All of them take a `seed` and are deterministic. Because the worker and firm
effects are drawn explicitly, you know the right answer, which is what makes
these useful for checking your own code as well as the library's.

[`examples/simulated_data.py`](https://github.com/leetucker/hdfe-stream/blob/main/examples/simulated_data.py)
is the helper the examples import for their data paths and scratch
directories. It is not an example itself, but it is short and worth a look if
you want to know what the simulated columns mean.

```{toctree}
:hidden:

quickstart
akm_variance
formulas
weights_and_iv
fewer_fixed_effects
glm
wide_designs
varying_slopes
out_of_core
reporting
connected_sets
leave_out_kss
```

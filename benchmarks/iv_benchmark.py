"""Time and peak memory of two-stage least squares with no fixed effects and
with one: feols_stream against the other implementations.

    python benchmarks/iv_benchmark.py                    # the default sizes
    python benchmarks/iv_benchmark.py 100000 1000000     # pick your own
    python benchmarks/plot_benchmarks.py                 # redraw the figures

The data are `_harness.general_data_path`: n rows, 10 covariates, an endogenous
regressor d with two instruments, and one group variable with n / 20 groups.
The models are

    yiv ~ x0 + ... + x9 | d ~ z1 + z2           no fixed effects
    yiv ~ x0 + ... + x9 | g | d ~ z1 + z2       one fixed effect

with the classical standard errors where a library gives them. polars-ols has
no 2SLS: its row runs the two stages by hand, which gives 2SLS's coefficients
but not its standard errors, so it reports none. linearmodels absorbs fixed
effects in least squares only, so for one fixed effect it is given the data
demeaned within g, counted in its time. Results go to
benchmarks/results/iv.csv. See _regression.py for the rest.
"""

from __future__ import annotations

from pathlib import Path

from _regression import main

SCRIPT = Path(__file__).resolve()

DEFAULT_SIZES = [100_000, 1_000_000, 5_000_000, 20_000_000]

# label -> (library, fixed effects)
CONFIGS = {
    "hdfe_stream": ("hdfe", 0),
    "linearmodels IV2SLS": ("linearmodels", 0),
    "pyfixest": ("pyfixest", 0),
    "fixest (R)": ("fixest", 0),
    "polars-ols (two stages)": ("polars_ols", 0),
    "NumPy normal equations": ("numpy", 0),
    "hdfe_stream (1 FE)": ("hdfe", 1),
    "pyfixest (1 FE)": ("pyfixest", 1),
    "fixest (R, 1 FE)": ("fixest", 1),
    "linearmodels IV2SLS (demeaned, 1 FE)": ("linearmodels", 1),
    "polars-ols (two stages, demeaned, 1 FE)": ("polars_ols", 1),
}

if __name__ == "__main__":
    main("iv", SCRIPT, CONFIGS, DEFAULT_SIZES, __doc__, "iv.csv")

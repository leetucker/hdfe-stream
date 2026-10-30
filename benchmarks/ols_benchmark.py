"""Time and peak memory of OLS with no fixed effects and with one: feols_stream
against the fast Python and R implementations people reach for.

    python benchmarks/ols_benchmark.py                   # the default sizes
    python benchmarks/ols_benchmark.py 100000 1000000    # pick your own
    python benchmarks/plot_benchmarks.py                 # redraw the figures

The data are `_harness.general_data_path`: n rows, 10 covariates and one group
variable with n / 20 groups. The models are

    y ~ x0 + ... + x9               no fixed effects (with an intercept)
    y ~ x0 + ... + x9 | g           one fixed effect

with the classical standard errors where a library gives them. scikit-learn,
and polars-ols and polars-ds where they are listed as coefficients only, do
less than the others, which also compute standard errors; the "NumPy normal
equations" row is the least work an OLS fit can do. For one fixed effect the
plugins and NumPy have no absorbing option, so they are given the data
demeaned within g, which is counted in their time. Results go to
benchmarks/results/ols.csv, one row per (size, configuration), replacing that
size's rows when it is re-run. See _regression.py for the rest.
"""

from __future__ import annotations

from pathlib import Path

from _regression import main

SCRIPT = Path(__file__).resolve()

DEFAULT_SIZES = [100_000, 1_000_000, 5_000_000, 20_000_000]

# label -> (library, fixed effects)
CONFIGS = {
    "hdfe_stream": ("hdfe", 0),
    "scikit-learn": ("sklearn", 0),
    "polars-ols": ("polars_ols", 0),
    "polars-ds": ("polars_ds", 0),
    "statsmodels": ("statsmodels", 0),
    "pyfixest": ("pyfixest", 0),
    "fixest (R)": ("fixest", 0),
    "lm (R)": ("base", 0),
    "NumPy normal equations": ("numpy", 0),
    "hdfe_stream (1 FE)": ("hdfe", 1),
    "pyfixest (1 FE)": ("pyfixest", 1),
    "fixest (R, 1 FE)": ("fixest", 1),
    "linearmodels AbsorbingLS (1 FE)": ("linearmodels", 1),
    "polars-ols (demeaned, 1 FE)": ("polars_ols", 1),
}

if __name__ == "__main__":
    main("ols", SCRIPT, CONFIGS, DEFAULT_SIZES, __doc__, "ols.csv")

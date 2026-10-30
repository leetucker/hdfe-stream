"""Time and peak memory of logit and probit regressions with no fixed effects:
feglm_stream against the other implementations.

    python benchmarks/binary_benchmark.py                   # the default sizes
    python benchmarks/binary_benchmark.py 100000 1000000    # pick your own
    python benchmarks/plot_benchmarks.py                    # redraw the figures

The data are `_harness.general_data_path`: n rows, 10 covariates and a binary
outcome from a logistic index. The models are

    yb ~ x0 + ... + x9       logit and probit, with an intercept

with the classical standard errors where a library gives them. scikit-learn,
glum and polars-ds return coefficients only, and fit logit alone, so they do
less than the others. Results go to benchmarks/results/binary.csv, one row per
(size, configuration). See _regression.py for the rest.
"""

from __future__ import annotations

from pathlib import Path

from _regression import main

SCRIPT = Path(__file__).resolve()

DEFAULT_SIZES = [100_000, 1_000_000, 5_000_000, 20_000_000]

# label -> (library, fixed effects); logit or probit as the label says
CONFIGS = {
    "hdfe_stream logit": ("hdfe", 0),
    "statsmodels logit": ("statsmodels", 0),
    "pyfixest logit": ("pyfixest", 0),
    "fixest logit (R)": ("fixest", 0),
    "glm logit (R)": ("base", 0),
    "scikit-learn logit": ("sklearn", 0),
    "glum logit": ("glum", 0),
    "polars-ds logit": ("polars_ds", 0),
    "hdfe_stream probit": ("hdfe", 0),
    "statsmodels probit": ("statsmodels", 0),
    "pyfixest probit": ("pyfixest", 0),
    "fixest probit (R)": ("fixest", 0),
    "glm probit (R)": ("base", 0),
}

if __name__ == "__main__":
    main("binary", SCRIPT, CONFIGS,
         DEFAULT_SIZES, __doc__, "binary.csv")

"""Smallest useful thing: fit a two-way fixed-effects model on a Parquet file.

    python examples/quickstart.py
"""

from hdfe_stream import feols_stream
from simulated_data import OUTPUT, panel_path, workdir

# A Parquet path (a glob like "data/part-*.parquet" works too), never loaded
# into memory as a whole.
data = panel_path()

# The formula follows pyfixest/fixest syntax:
#
#     outcome ~ covariates | fixed effects
#
# Here: log earnings on an age profile, with worker, firm and year effects.
# `workdir` is where intermediates go -- this needs disk, not memory, and is
# the one argument with no pyfixest equivalent.
fit = feols_stream(
    "log_earn ~ age_squared + age_cubed | worker_id + firm_id + year",
    data,
    workdir=workdir("quickstart"),
    vcov={"CRV1": "worker_id"},        # cluster by worker
)

# Progress printed as it goes; `verbose=False` turns that off. The summary
# includes the coefficient table; `fit.tidy()` returns it as a Polars DataFrame.
fit.summary()

print("\nage_squared:", fit.coef()["age_squared"])
print("standard errors:", dict(zip(fit.coefnames, fit.se)))

# How many levels each dimension had, and how many parameters that cost
print("\nlevels:", fit.n_levels)
print("fixed-effect parameters (net of redundancy):", fit.k_fe)
print("observations:", f"{fit.n_obs:,.0f}")

# The estimated fixed effects, as a lazy scan -- nothing is read until you ask
print("\nlargest firm effects:")
print(fit.fixef("firm_id").sort("fe_firm_id", descending=True).head(5).collect())

# The model-level information of the summary (observations, levels, fit
# statistics, solver), saved as JSON for later. coefficients=True adds the
# coefficient table. summary_dict() returns the same as a dict, and
# summary_text() the printed summary as a string.
path = OUTPUT / "quickstart_fit.json"
fit.summary_json(path, coefficients=True)
print(f"\nsummary written to {path.name}:")
print(fit.summary_json(indent=None)[:240] + " ...")

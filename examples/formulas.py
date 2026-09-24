"""Formula syntax: everything you can put on either side of the bars.

Formulas are parsed by pyfixest's own parser, so the syntax and the resulting
coefficient names are pyfixest's. Every covariate term is then compiled into a
Polars expression, which is why the design matrix never has to be built in
memory.

    python examples/formulas.py
"""

import polars as pl

from hdfe_stream import feols_stream
from simulated_data import panel_path, workdir

data = panel_path()


def show(title, fit):
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")
    print(fit.tidy())


# ---------------------------------------------------------------- transformations
# I() for arithmetic, log()/exp()/sqrt(), and `:` for interactions. These are
# computed as Polars expressions while the data streams past.
show("transformations and interactions", feols_stream(
    "log_earn ~ I(age ** 2) + log(age) + age_squared:age_cubed"
    " | worker_id + firm_id + year",
    data, workdir=workdir("terms"), verbose=False))

# ------------------------------------------------------------------ categoricals
# C(x) for a categorical covariate; string columns work too. One level is
# absorbed as the reference, exactly as pyfixest chooses it.
show("categorical covariates", feols_stream(
    "log_earn ~ C(occ) + cat:age_squared | worker_id + firm_id + year",
    data, workdir=workdir("categorical"), verbose=False))

# ------------------------------------------------------------------ event study
# i(over, by, ref=...) gives one coefficient per level of `over`, interacted
# with `by`, dropping the reference level. The simulated data has a treatment
# effect switching on in 2010, which is what this should find.
event = feols_stream(
    "log_earn ~ i(year, treat, ref=2009) + age_squared | worker_id + firm_id",
    data, workdir=workdir("event"), vcov={"CRV1": "firm_id"}, verbose=False)
show("event study: i(year, treat, ref=2009)", event)
print("\n(the simulated effect turns on in 2010 and is worth 0.1)")

# ------------------------------------------------------ interacted fixed effects
# `^` combines columns into one dimension: a separate effect per firm-year cell.
show("firm-by-year effects: firm_id^year", feols_stream(
    "log_earn ~ age_squared + age_cubed | worker_id + firm_id^year",
    data, workdir=workdir("interacted"), vcov={"CRV1": "firm_id"}, verbose=False))

# ------------------------------------------------------------ multiple estimation
# Several outcomes (`y1 + y2 ~ ...`), cumulative or plain stepwise covariate
# sets (csw()/sw()), and stepwise fixed-effect sets. Models sharing a
# fixed-effect set share one pass over the data and one solve, which is the
# whole point of asking for them together.
multi = feols_stream(
    "log_earn + y2 ~ csw(age_squared, age_cubed)"
    " | sw(worker_id + firm_id, worker_id + firm_id + year)",
    data, workdir=workdir("multi"), vcov="hetero", verbose=False)
print(f"\n{'=' * 72}\nmultiple estimation: 2 outcomes x 2 covariate sets x 2 FE sets"
      f"\n{'=' * 72}")
print(f"{len(multi)} models fitted in one call\n")
print(multi.tidy())

# Individual models come back by position or by iteration
print("\nfirst model:", multi.fetch_model(0).fml)

# ------------------------------------------------------------- missing values
# Rows with a null or non-finite value in any variable the formula uses are
# dropped, as pyfixest drops them. `x_na` is null in about 5% of rows.
with_missing = feols_stream(
    "log_earn ~ age_squared + x_na | worker_id + firm_id + year",
    data, workdir=workdir("missing"), verbose=False)
total = pl.scan_parquet(data).select(pl.len()).collect().item()
print(f"\n{'=' * 72}\nmissing values\n{'=' * 72}")
print(f"rows in the file: {total:,}")
print(f"rows used:        {with_missing.n_obs:,.0f}")

# ---------------------------------------------------------------- collinearity
# Terms spanned by the fixed effects are dropped, exactly as pyfixest drops
# them: i(year) duplicates the year effect, and `treat` is constant within
# worker so C(treat) duplicates the worker effect.
collinear = feols_stream(
    "log_earn ~ i(year) + C(treat) + age_squared | worker_id + firm_id + year",
    data, workdir=workdir("collinear"), verbose=False)
print(f"\n{'=' * 72}\ncollinear terms are dropped\n{'=' * 72}")
print("dropped:", collinear.collin_vars)
print("kept:   ", collinear.coefnames)
# A covariate the streamed dimension absorbs entirely is recognised during the
# solve rather than costing iterations
print("absorbed by the streamed dimension:",
      collinear.solver_info.get("fe_spanned", []))

# ------------------------------------------------------------- clustering
# Cluster on anything: a fixed effect, a column that is not a fixed effect, an
# interaction, or several dimensions at once (Cameron-Gelbach-Miller).
clustered = feols_stream(
    "log_earn ~ age_squared + age_cubed | worker_id + firm_id + year",
    data, workdir=workdir("clusters"),
    vcov={"CRV1": "worker_id"},
    # ask for more in the same pass, then switch between them for free
    cluster=["firm_id", "state", "worker_id+firm_id", "firm_id^year"],
    verbose=False)
print(f"\n{'=' * 72}\nstandard errors under different clusterings\n{'=' * 72}")
print(f"{'vcov':<26}{'se(age_squared)':>18}{'clusters':>16}")
for key in ["iid", "hetero", "CRV1:worker_id", "CRV1:firm_id", "CRV1:state",
            "CRV1:worker_id+firm_id", "CRV1:firm_id^year"]:
    switched = clustered.with_vcov(key)
    groups = switched.n_clusters.get(key.removeprefix("CRV1:"), "-")
    print(f"{key:<26}{switched.se[0]:>18.6f}{str(groups):>16}")

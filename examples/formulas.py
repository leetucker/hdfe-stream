"""Formula syntax: everything you can put on either side of the bars.

Formulas are parsed by pyfixest's own parser, so the syntax and the resulting
coefficient names are pyfixest's. Every covariate term is then compiled into a
Polars expression, which is why the design matrix never has to be built in
memory.

    python examples/formulas.py
"""

import warnings

import polars as pl

from hdfe_stream import feols_stream, holm, rwolf
from simulated_data import panel_path, thin_panel_path, workdir

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

# sample() says which rows were used and why the others were not: the source,
# lazily, with `row_id` (each row's position in the file), `in_sample` and
# `dropped_because`. Nothing is stored per row to make this work.
rows = with_missing.sample()
print("\nwhy rows were dropped:")
print(rows.group_by("dropped_because").len().sort("dropped_because").collect())

# The residual file is in the estimator's own order, but carries `row_id`, so
# it joins back onto the source -- here to see the residuals by occupation, a
# column the model did not use.
by_occ = (rows.join(with_missing.resid().select("row_id", "resid"), on="row_id", how="left")
              .group_by("occ").agg(pl.col("resid").mean().alias("mean resid"),
                                   pl.col("resid").is_null().sum().alias("rows not used"))
              .sort("occ").collect())
print("\nresiduals joined back onto the source on row_id, by occupation:")
print(by_occ)
# row_id= renames the column, for a source that already has one by that name.

# For several models, sample() returns {formula: frame}. Models fitted together
# on the same fixed effects share one sample: all rows complete in every one of
# their variables.
samples = multi.sample()
used = {f: s.filter("in_sample").select(pl.len()).collect().item() for f, s in samples.items()}
print(f"\nmulti.sample(): one frame per formula ({len(samples)}), rows used: "
      + ", ".join(f"{n:,}" for n in sorted(set(used.values()))))

# ---------------------------------------------------------------- predictions
# predict() adds a prediction column to new rows, lazily: covariates times the
# coefficients plus each row's fixed effects, looked up by their levels. A row
# with a worker or firm the fit never saw (or dropped) gets null.
panel = pl.scan_parquet(data)
early = feols_stream("log_earn ~ age_squared + age_cubed | worker_id + firm_id",
                     panel.filter(pl.col("year") < 2012), workdir=workdir("formulas"),
                     verbose=False)
later = early.predict(panel.filter(pl.col("year") >= 2012)).collect()
print(f"\npredict(): {later.height:,} later rows, "
      f"{later['prediction'].null_count():,} with a worker or firm the fit never saw")

# ------------------------------------------------- fixed-effect degrees of freedom
# The standard errors count the fixed-effect parameters, net of the redundant
# ones. fe_dof="exact" (the default) finds those from the connected components:
# one redundant level per component of the worker-firm network. fe_dof="pyfixest"
# counts one per fixed effect beyond the first, as pyfixest does. On a connected
# panel, like the one above, the two agree exactly. On a panel whose network
# falls apart, pyfixest's count includes a parameter for every component past
# the first that cannot be estimated, so its standard errors are a little too
# large. (With three or more fixed effects "exact" is an upper bound, so its
# standard errors are conservative.)
thin = thin_panel_path()
# (the fits below drop a few singletons; their warnings are beside the point here)
warnings.filterwarnings("ignore", message=".*singleton observations dropped")
print(f"\n{'=' * 72}\nfe_dof: counting the fixed-effect parameters\n{'=' * 72}")
print(f"{'panel':<12}{'fe_dof':<10}{'components':>11}{'FE params':>11}{'df':>9}"
      f"{'se(age_squared)':>17}")
for name, source in (("connected", data), ("thin", thin)):
    for dof in ("exact", "pyfixest"):
        fit = feols_stream("log_earn ~ age_squared | worker_id + firm_id", source,
                           workdir=workdir("fe_dof"), fe_dof=dof, verbose=False)
        print(f"{name:<12}{dof:<10}{fit.n_components:>11,}{fit.k_fe:>11,}{fit.df_t:>9,}"
              f"{fit.se[0]:>17.7f}")

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
# A covariate the streamed dimension absorbs entirely is recognized during the
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

# CRV3, the cluster jackknife, is computed by downdating the fit, which is exact
# when every fixed effect is nested within the clusters: here firm-by-year
# effects, clustered by firm. (OLS and one-way only.)
nested = feols_stream(
    "log_earn ~ age_squared + age_cubed | firm_id^year", data,
    workdir=workdir("crv3"), vcov={"CRV3": "firm_id"}, cluster=["firm_id"],
    verbose=False)
print(f"\nfirm-year effects clustered by firm: se(age_squared) "
      f"CRV3 {nested.se[0]:.6f}, CRV1 {nested.with_vcov({'CRV1': 'firm_id'}).se[0]:.6f}")
# With fixed effects that are not nested -- worker and firm effects clustered by
# worker -- the jackknife would have to refit once per cluster, and the fit says so.
try:
    feols_stream("log_earn ~ age_squared | worker_id + firm_id", data,
                 workdir=workdir("crv3"), vcov={"CRV3": "worker_id"}, verbose=False)
except ValueError as err:
    print(f"worker and firm effects clustered by worker: {str(err).split('.')[0]}.")

# ------------------------------------------------- tests and confidence intervals
# From the stored coefficients and vcov (the current one: switch with
# with_vcov). wald_test tests R beta = q; keep= selects coefficients to test
# jointly against zero. confint(joint=True) covers all of them at once.
print(f"\n{'=' * 72}\ntests and confidence intervals (clustered by worker)\n{'=' * 72}")
wald = clustered.wald_test(keep="age")
print(f"age_squared = age_cubed = 0: F = {wald['statistic']:.1f} "
      f"on ({wald['df1']}, {wald['df2']:,}) df, p = {wald['pvalue']:.2g}")
print(clustered.confint(joint=True, seed=0))

# Bonferroni and Holm corrections for one coefficient tested in several models
models = feols_stream("log_earn ~ x2 + csw0(age_squared, age_cubed) | worker_id + firm_id",
                      data, workdir=workdir("multcomp"), vcov={"CRV1": "worker_id"},
                      verbose=False)
print(holm(models, "x2").select("fml", "Pr(>|t|)", "Holm Pr(>|t|)"))

# ------------------------------------------------------ wild cluster bootstrap
# With 12 states, CRV1 by state is unreliable; the wild cluster bootstrap is
# the usual remedy. bootcluster= keeps the per-cluster sums it needs during the
# fit. Every fixed effect must be nested within the clusters: firms are (each
# is in one state), years are not, so they go in as covariates, which leaves
# the other coefficients as they would be with year effects.
by_state = feols_stream("log_earn + y2 ~ x2 + age_squared + i(year) | firm_id", data,
                        workdir=workdir("wildboot"), vcov={"CRV1": "state"},
                        bootcluster="state", verbose=False)
earnings = by_state.fetch_model(0)
boot = earnings.wildboottest("age_squared", reps=9999, seed=0)
print(f"\n{'=' * 72}\nwild cluster bootstrap, 12 states\n{'=' * 72}")
print(f"age_squared: t = {boot['t value']:.2f}, CRV1 p = {earnings.tidy()['Pr(>|t|)'][1]:.4f}, "
      f"bootstrap p = {boot['Pr(>|t|)']:.4f}")
# Romano-Wolf: x2 in both models, on the same draws
print(rwolf(by_state, "x2", reps=9999, seed=0).select("fml", "Pr(>|t|)", "RW Pr(>|t|)"))

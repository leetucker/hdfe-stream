"""AKM variance decomposition, without ever holding the data in memory.

This is the application the library was written for. Having estimated worker
and firm effects, you want to know how much of the variance of log earnings
each one explains, and how much comes from the covariance between them --
whether good workers sort into good firms.

Fitted values split additively into the covariate part and one term per fixed
effect dimension,

    y = Xb + worker + firm + year + residual

so the variance of y is the sum of those five variances plus twice every
pairwise covariance between them. Each term is an aggregate over rows, and
`resid()` hands back a lazy scan of the row-level output, so Polars computes
them in a streaming pass: the numbers come back without the rows ever being
collected.

    python examples/akm_variance.py
"""

import polars as pl

from hdfe_stream import feols_stream
from simulated_data import panel_path, workdir

fit = feols_stream(
    "log_earn ~ age_squared + age_cubed | worker_id + firm_id + year",
    panel_path(),
    workdir=workdir("akm"),
    vcov={"CRV1": "worker_id"},
    verbose=False,
)

# The row-level file holds, per observation: the outcome, the covariate
# contribution `xb`, one column per fixed effect dimension with that row's
# estimated effect, and the residual.
rows = fit.resid()
print("columns available per row:", rows.collect_schema().names())

# One streaming pass over the file computes the whole decomposition.
# engine="streaming" keeps Polars from materializing the frame.
PARTS = ["xb", "fe_worker_id", "fe_firm_id", "fe_year", "resid"]
stats = rows.select(
    pl.col("log_earn").var().alias("var_y"),
    *[pl.col(c).var().alias(f"var_{c}") for c in PARTS],
    pl.cov("fe_worker_id", "fe_firm_id").alias("cov_worker_firm"),
    pl.corr("fe_worker_id", "fe_firm_id").alias("corr_worker_firm"),
).collect(engine="streaming").row(0, named=True)

var_y = stats["var_y"]
variances = {c: stats[f"var_{c}"] for c in PARTS}
# everything not in the individual variances is covariance between the parts
covariances = var_y - sum(variances.values())

print(f"\nVariance of log earnings: {var_y:.4f}\n")
print(f"{'component':<34}{'variance':>11}{'share':>9}")
print("-" * 54)
for label, key in [("worker effects", "fe_worker_id"),
                   ("firm effects", "fe_firm_id"),
                   ("year effects", "fe_year"),
                   ("covariates (Xb)", "xb"),
                   ("residual", "resid")]:
    value = variances[key]
    print(f"{label:<34}{value:>11.4f}{value / var_y:>8.1%}")
print(f"{'2 x covariances between the parts':<34}{covariances:>11.4f}"
      f"{covariances / var_y:>8.1%}")
print("-" * 54)
print(f"{'total':<34}{var_y:>11.4f}{1.0:>8.1%}")

# The sorting term on its own: this is the one people quote
sorting = 2 * stats["cov_worker_firm"]
print(f"\n2 x cov(worker, firm) alone: {sorting:.4f} ({sorting / var_y:.1%} of Var(y))")

print(f"\nCorrelation of worker and firm effects: {stats['corr_worker_firm']:.3f}")
print("(positive means assortative matching: better workers at better firms)")

# Identification: worker and firm effects are only comparable within a
# connected set of firms, linked by workers who moved between them. With more
# than one component, effects in different components are not comparable.
print(f"\nconnected components: {fit.n_components}")
print("worker groups that carry identifying information:", f"{fit.n_identifying:,}")

# Firm effects, with the component each firm belongs to
firms = fit.fixef("firm_id").collect()
print("\nfirm effects by component:")
print(firms.group_by("component").agg(
    firms=pl.len(),
    observations=pl.col("n_obs").sum(),
    mean_effect=pl.col("fe_firm_id").mean(),
    sd_effect=pl.col("fe_firm_id").std(),
).sort("firms", descending=True))

# A caution worth knowing: with few observations per worker, Var(worker) is
# biased up and Cov(worker, firm) biased down by estimation error -- the
# "limited mobility bias" of Andrews et al. This example reports the raw
# plug-in decomposition, which is what the estimated effects give you.
print("\nobservations per worker: "
      f"{fit.n_obs / fit.n_levels['worker_id']:.1f} on average")

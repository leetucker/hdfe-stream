"""Usage examples for hdfe_stream."""
import polars as pl
import pyfixest as pf
from hdfe_stream import StreamingHDFE, etable, feols_stream

# ---------------------------------------------------------------- formulas
# pyfixest syntax; the default vcov (like pyfixest) clusters by the first FE.
fit = feols_stream(
    "log_earn ~ age_squared + age_cubed | pik + sein + year",
    "simf.parquet",              # Parquet path/glob, or any Polars LazyFrame
    workdir="hdfe_work/akm",     # intermediates + outputs (needs disk!)
    vcov={"CRV1": "pik"},
    cluster=["sein"],            # also compute CRV1 by firm; switch with with_vcov()
    keep=["year"],               # extra columns carried into the residual file
)
fit.summary()
print(fit.with_vcov({"CRV1": "sein"}).tidy())

# multi-way clustering (Cameron-Gelbach-Miller), any number of ways; request
# several at once with cluster= and switch between them afterwards
tw = feols_stream("log_earn ~ age_squared + age_cubed | pik + sein + year", "simf.parquet",
                  workdir="hdfe_work/twoway", vcov={"CRV1": "pik+sein"},
                  cluster=["pik", "sein+year", "pik+sein+year"], verbose=False)
print(tw.tidy(), tw.with_vcov({"CRV1": "sein+year"}).tidy())

# fixed effects and residuals come back as lazy scans; nothing is loaded yet
print(fit.fixef("sein").head(3).collect())
print(fit.resid().select(
    pl.col("log_earn").var().alias("var_y"),
    pl.col("fe_pik").var().alias("var_worker"),
    pl.col("fe_sein").var().alias("var_firm"),
    (2 * pl.cov("fe_pik", "fe_sein")).alias("2cov"),
    pl.col("resid").var().alias("var_resid"),
).collect(engine="streaming"))

# event study with firm-by-year effects, clustered by firm (any column works)
es = feols_stream(
    "log_earn ~ i(year, treat, ref=2009) + age_squared | pik + sein^year",
    "simf.parquet", workdir="hdfe_work/es", vcov={"CRV1": "sein"},
)
print(es.tidy())

# multiple estimation: 2 outcomes x 2 covariate sets x 2 FE sets = 8 models.
# Models with the same FEs share passes 0-1 and a single solve.
multi = feols_stream(
    "log_earn + y2 ~ csw(age_squared, age_cubed) | sw(pik + sein, pik + sein + year)",
    "simf.parquet", workdir="hdfe_work/multi", vcov="hetero", verbose=False,
)
print(multi.tidy())

# categorical terms, interactions, transformations; collinear terms are
# dropped exactly as pyfixest does (here: i(year) with year FE)
fit2 = feols_stream(
    "log_earn ~ C(occ) + cat:age_squared + log(age) + i(year) | pik + sein + year",
    "simf.parquet", workdir="hdfe_work/terms", verbose=False,
)
print("dropped:", fit2.collin_vars)
print(fit2.tidy())

# ------------------------------------------------------ weights and IV
# weighted least squares (aweights by default; weights_type="fweights" for
# frequency weights, where N = sum of weights)
wfit = feols_stream("log_earn ~ age_squared + age_cubed | pik + sein + year",
                    "simiv.parquet", workdir="hdfe_work/wls", weights="wa", verbose=False)

# 2SLS: y ~ exogenous | FEs | endogenous ~ instruments
iv = feols_stream("y_iv ~ age_squared | pik + sein | x2 ~ z1 + z2",
                  "simiv.parquet", workdir="hdfe_work/iv", vcov={"CRV1": "pik"},
                  verbose=False)
iv.summary()                      # includes the first-stage F (same vcov as the model)
print(iv.first_stage[0].tidy())   # the first-stage regression itself

# ------------------------------------------------------ pyfixest reporting
# etable accepts streaming results, multi-model results, and ordinary
# pyfixest models, mixed freely; everything else is passed to pf.etable
print(etable([wfit, iv], type="df", model_stats=["N", "se_type", "r2", "r2_within"]))
etable(multi, type="tex", file_name="hdfe_work/table.tex")
pf.iplot(es.to_pyfixest(), plot_backend="matplotlib")     # event-study plot
pf.summary(iv.to_pyfixest())

# ------------------------------------------------------ varying slopes (FEIS)
# worker-specific intercepts and slopes in year (individual trends); the
# dimension with slopes is always the streamed one
feis = feols_stream("log_earn ~ age_cubed | pik[year] + sein + year", "simf.parquet",
                    workdir="hdfe_work/feis", vcov={"CRV1": "pik"}, verbose=False)
feis.summary()
print(feis.fixef("pik").head(3).collect())     # pik, fe_pik, fe_pik[year], n_obs

# choosing the streamed dimension explicitly (otherwise: slopes, else the
# highest approximate cardinality)
fit_s = feols_stream("log_earn ~ age_squared | sein + pik + year", "simf.parquet",
                     workdir="hdfe_work/stream", stream="pik", verbose=False)
print(fit_s.diagnostics["stream"])

# ------------------------------------------------------ disk use
# Each fit works in its own run directory under workdir; intermediates are
# deleted as soon as they're no longer needed, and everything is removed if
# the fit fails. Result files (resid(), fixef()) live as long as the result:
with feols_stream("log_earn ~ age_squared | pik + sein + year", "simf.parquet",
                  workdir="hdfe_work", verbose=False) as tmp:
    fe_firm = tmp.fixef("sein").collect()        # collect what you need ...
# ... the files are gone here. outputs="keep" keeps them until you call
# result.cleanup() or hdfe_stream.cleanup(workdir); save_resid=False skips the
# row-level residual file. Leftovers from killed jobs:
import hdfe_stream
print(hdfe_stream.cleanup("hdfe_work", dry_run=True))

# ------------------------------------------------------ logging
# Progress messages and warnings go to a logger as they happen; summaries too.
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s",
                    force=True)   # force: an earlier import may have configured logging
log = logging.getLogger("akm")
fit_l = feols_stream("log_earn ~ age_squared + age_cubed | pik + sein + year", "simf.parquet",
                     workdir="hdfe_work/logged", logger=log)   # verbose=False to log warnings only
fit_l.summary(logger=log)          # one multi-line INFO record
text = fit_l.summary_text()        # or just the string

# ------------------------------------------------------ low-level interface
# Column names or (name, Polars expression) pairs; handy when a covariate is
# easier to write in Polars than in formula syntax.
est = StreamingHDFE(
    y="log_earn",
    x=["age_squared", ("over_40", (pl.col("age") > 40).cast(pl.Float64))],
    fe=["pik", "sein", "year"],
    workdir="hdfe_work/lowlevel",
    solver="auto",               # explicit S unless it exceeds max_s_gb, then stream_cg
    max_s_gb=8,
    rhs_block=8,                 # variables solved at a time (bounds solver memory)
    n_threads=None,              # numba threads; None = all available
)
res = est.fit("simf.parquet", vcov={"CRV1": "pik"})
res.summary()
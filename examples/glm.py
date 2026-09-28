"""Poisson, logit and probit with fixed effects.

`fepois_stream` and `feglm_stream` fit generalized linear models by
iteratively reweighted least squares. Each step is a weighted version of the
linear regression `feols_stream` solves, so everything about reading the data
carries over: the rows are streamed, sorted once, and read again at each step.

The data here are simulated with a known coefficient of 0.3 on `x`, so the
estimates can be checked against it. The binary outcome has firm effects but
no worker effects, which makes the incidental parameter problem visible: add
worker effects, estimated from about eight rows each, and the logit
coefficient moves away from the truth even though the model is correct.
Neither this package nor pyfixest corrects that bias.

    python examples/glm.py
"""

from hdfe_stream import feglm_stream, fepois_stream
from simulated_data import glm_panel_path, workdir

data = glm_panel_path()

# --------------------------------------------------------------- Poisson
# An offset enters the linear predictor with a coefficient of one (here, log
# exposure). Workers whose count is zero in every year have a worker effect of
# minus infinity; they are dropped before the fit, and the result says so.
pois = fepois_stream("visits ~ x | worker_id + firm_id + year", data, offset="log_exposure",
                     workdir=workdir("glm_poisson"), verbose=False)
pois.summary()
sep = pois.diagnostics["separation"]
print(f"separation: {sep['observations']:,} rows dropped ({sep['levels']})")
irls = pois.diagnostics["irls"]
print(f"IRLS: {irls['iterations']} steps, deviance by step {[round(d) for d in irls['deviance']]}")

# The residual file holds the linear predictor, the fitted mean and the
# response residual for every row, as a lazy scan.
rows = pois.resid().select("visits", "eta", "fitted", "resid").head(3).collect()
print(rows)

# ------------------------------------------------------------ logit, probit
# Firm and year effects only: many rows per firm, so the effects are well
# estimated and the coefficient is close to 0.3 (probit's is on a different
# scale, about 0.3 / 1.6).
for family in ("logit", "probit"):
    fit = feglm_stream("promoted ~ x | firm_id + year", data, family,
                       workdir=workdir(f"glm_{family}"), verbose=False)
    print(f"{family:7} firm + year effects:   x = {fit.coef()['x']:.4f}  "
          f"(se {fit.se[0]:.4f}, clustered by firm)")
    fit.cleanup()

# Adding worker effects: workers whose outcome never changes are dropped
# (their effect would be infinite), and the rest have few rows each.
wk = feglm_stream("promoted ~ x | worker_id + firm_id + year", data, "logit",
                  workdir=workdir("glm_logit_workers"), verbose=False)
sep = wk.diagnostics["separation"]
print(f"logit   + worker effects:        x = {wk.coef()['x']:.4f}  "
      f"({sep['observations']:,} rows of workers with an unchanging outcome dropped, "
      f"in {sep['rounds']} round{'' if sep['rounds'] == 1 else 's'})")

for fit in (pois, wk):
    fit.cleanup()

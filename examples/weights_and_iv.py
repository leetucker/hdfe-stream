"""Weighted least squares and two-stage least squares.

    python examples/weights_and_iv.py
"""

from hdfe_stream import feols_stream
from simulated_data import panel_path, workdir

data = panel_path()
FML = "log_earn ~ age_squared + age_cubed | worker_id + firm_id + year"

# ---------------------------------------------------------------------- weights
# weights= names a column. Two kinds, and the difference is not cosmetic:
#
#   aweights (default)  analytic/precision weights. N stays the number of rows.
#   fweights            frequency weights: each row stands for `w` identical
#                       observations, so N is the sum of the weights and every
#                       degrees-of-freedom correction changes with it.
unweighted = feols_stream(FML, data, workdir=workdir("ols"), verbose=False)
analytic = feols_stream(FML, data, workdir=workdir("aweights"),
                        weights="wa", verbose=False)
frequency = feols_stream(FML, data, workdir=workdir("fweights"),
                         weights="wf", weights_type="fweights", verbose=False)

print(f"{'model':<24}{'N':>14}{'age_squared':>14}{'se':>12}")
print("-" * 64)
for label, fit in [("unweighted", unweighted), ("aweights (wa)", analytic),
                   ("fweights (wf)", frequency)]:
    print(f"{label:<24}{fit.n_obs:>14,.0f}{fit.beta[0]:>14.6f}{fit.se[0]:>12.6f}")
print("\nfweights inflate N to the weight total, which is why its standard "
      "errors\nare smaller: it is told there are more observations.")

# -------------------------------------------------------------------------- IV
# Four parts:  outcome ~ exogenous | fixed effects | endogenous ~ instruments
#
# In the simulated data x2 is correlated with an unobservable that also enters
# y_iv, so OLS on x2 is biased upward; z1 and z2 are excluded instruments. The
# true coefficient on x2 is 0.2.
ols = feols_stream("y_iv ~ age_squared + x2 | worker_id + firm_id",
                   data, workdir=workdir("iv_ols"), vcov={"CRV1": "worker_id"},
                   verbose=False)
iv = feols_stream("y_iv ~ age_squared | worker_id + firm_id | x2 ~ z1 + z2",
                  data, workdir=workdir("iv"), vcov={"CRV1": "worker_id"},
                  verbose=False)

print(f"\n{'=' * 64}\n2SLS: x2 is endogenous, instrumented by z1 and z2\n{'=' * 64}")
print("true coefficient on x2:   0.200000")
print(f"OLS  (biased):          {ols.coef()['x2']:>9.6f}")
print(f"2SLS (consistent):      {iv.coef()['x2']:>9.6f}")

iv.summary()

# The first stage is a fitted model in its own right
print("\nfirst stage (x2 on the instruments, same fixed effects):")
print(iv.first_stage[0].tidy())

# The reported F is the Wald statistic on the excluded instruments, computed
# under the same vcov as the model -- the usual weak-instrument diagnostic.
print(f"\nfirst-stage F: {iv.f_stat_1st_stage[0]:,.1f}"
      "   (>10 is the conventional rule of thumb)")

# Weights and IV together
iv_weighted = feols_stream("y_iv ~ age_squared | worker_id + firm_id | x2 ~ z1 + z2",
                           data, workdir=workdir("iv_weighted"), weights="wa",
                           vcov={"CRV1": "worker_id"}, verbose=False)
print(f"\nweighted 2SLS on x2:    {iv_weighted.coef()['x2']:>9.6f}")

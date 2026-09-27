"""Leave-out (Kline-Saggio-Solvsten) variance components.

The plug-in AKM decomposition is biased. Worker and firm effects are estimated
from a handful of observations each, and that estimation error inflates their
measured variance and attenuates the covariance between them -- so the plug-in
numbers overstate how much of wage inequality is worker and firm heterogeneity,
and understate how much is sorting. With few observations per worker the bias is
large; this is the "limited mobility bias" of Andrews et al. (2008).

Kline, Saggio and Solvsten (2020) remove it exactly, without assuming the errors
are identically distributed, by subtracting a bias term built from each
observation's statistical leverage. Computing those leverages exactly would need
one linear solve per observation, so they are approximated by random projection
-- which is what makes the whole thing affordable here.

What is "left out" is a whole worker-firm match -- every year of a spell --
following LeaveOutTwoWay, VarianceComponentsHDFE.jl and xhdfe. Earnings errors
are correlated within a spell, and leaving out one year of it would treat the
others as independent of it. `leave_out="observation"` leaves out a single year
instead, and is what standard errors need; both appear below.

    python examples/leave_out_kss.py
"""

import numpy as np
import polars as pl

from hdfe_stream import feols_stream, leave_one_out_connected, leave_out_kss
from hdfe_stream.simulate import simulate_akm, simulate_bottleneck
from simulated_data import OUTPUT, workdir

# The simulator can hand back the worker and firm effects it generated, so for
# once the right answer is known. `log_earn` is built as
#
#     alpha + psi + 0.08 age_squared - 0.012 age_cubed + noise
#
# so the model below is correctly specified, and the true variance components
# are moments of columns we have.
OUTPUT.mkdir(parents=True, exist_ok=True)
data = OUTPUT / "leave_out_panel.parquet"
if not data.exists():
    simulate_akm(n_workers=8_000, seed=17, keep_effects=True).write_parquet(data)

FML = "log_earn ~ age_squared + age_cubed | worker_id + firm_id"

# KSS operate at 250-500 draws on panels with millions of effects, and report
# the error *falling* with sample size at fixed draws -- so this is a budget,
# not something that has to scale with the data. The last section shows how to
# check it is enough for your own.
DRAWS = 250

# ----------------------------------------------------------------- one call
# This prunes, fits and decomposes. The order matters and is not negotiable:
# pruning changes the estimation sample, so it has to happen before the fit.
lo = leave_out_kss(FML, str(data), workdir=workdir("leave_out"),
                   n_draws=DRAWS, seed=0)

print()
print(lo.summary())

# ------------------------------------------------------- against the truth
# The estimand is an observation-level moment of the true effects, taken over
# the sample that survived pruning -- so it must be computed on that sample,
# not on the panel we started with.
kept = lo.pruning.data.select("true_worker_effect", "true_firm_effect").collect()
alpha = kept["true_worker_effect"].to_numpy()
psi = kept["true_firm_effect"].to_numpy()
# Like the reference implementations, the components divide by n - 1.
truth = {
    "var(psi)": float(np.var(psi, ddof=1)),
    "var(alpha)": float(np.var(alpha, ddof=1)),
    "cov(psi, alpha)": float(np.cov(psi, alpha, ddof=1)[0, 1]),
}

print(f"\n{'=' * 72}\nagainst the answer the data was built from\n{'=' * 72}")
print(f"{'component':<18}{'truth':>10}{'plug-in':>11}{'leave-out':>11}"
      f"{'plug-in err':>13}{'leave-out err':>15}")
print("-" * 78)
for key, true_value in truth.items():
    plug_in_error = abs(lo.plug_in[key] - true_value)
    corrected_error = abs(lo.leave_out[key] - true_value)
    print(f"{key:<18}{true_value:>10.5f}{lo.plug_in[key]:>11.5f}"
          f"{lo.leave_out[key]:>11.5f}{plug_in_error:>13.5f}"
          f"{corrected_error:>15.5f}")

print("\nThe plug-in errors run in the direction theory predicts: both variances")
print("too high, the covariance too low. Correcting them changes the story about")
print("how much of wage variance is sorting rather than heterogeneity.")
print(f"\nobservations per worker: "
      f"{lo.n_obs / lo.diagnostics['n_levels']['worker_id']:.1f} -- the bias grows")
print("as this falls, so a short panel needs the correction more, not less.")

# --------------------------------------------------------------- the sample
# Pruning is not a detail. Leave-out estimation needs every effect to stay
# estimable when any one observation is dropped, which fails at workers who are
# the only link between two parts of the worker-firm network. Here the simulated
# panel is well connected and little is lost; on administrative data KSS report
# losing roughly half the firms.
p = lo.pruning.diagnostics
print(f"\n{'=' * 72}\nwhat pruning cost\n{'=' * 72}")
print(f"  workers {p['workers_before']:,} -> {p['workers_after']:,}")
print(f"  firms   {p['firms_before']:,} -> {p['firms_after']:,}")
print(f"  articulation-point workers removed: {p['cut_workers']:,}")
print(f"  components before pruning: {p['components_before']:,}")

# The pruned panel is available on its own, if you want to fit other things on
# the same sample.
print(f"\n  the surviving panel: {lo.pruning.data.select(pl.len()).collect().item():,} rows")

# ------------------------------------------------------ the two-step version
# `leave_out_kss` is a wrapper. If you want the regression in its own right --
# coefficients, residuals, fixed effects -- do the steps yourself. Leaving out a
# match reads only the fit's residual file, which every fit keeps, and refits on
# the collapsed matches. (Leaving out an observation reads the sorted rows and
# cell tables a fit otherwise deletes, so that needs keep_intermediates=True.)
print(f"\n{'=' * 72}\nthe same thing in steps, keeping the regression\n{'=' * 72}")
panel = leave_one_out_connected(str(data))
fit = feols_stream(FML, panel.data, workdir=workdir("leave_out_steps"),
                   stream="worker_id", verbose=False)
step = fit.leave_out_kss(n_draws=DRAWS, seed=0)

print(f"  regression: {dict(zip(fit.coefnames, fit.beta.round(6)))}")
print("              (the data was built with 0.08 and -0.012)")
print(f"  and the same components: "
      f"{np.allclose([step.leave_out[k] for k in truth], [lo.leave_out[k] for k in truth])}")

# ------------------------------------------------------------ how many draws
# The leverages are approximated, so the answer carries Monte Carlo noise. The
# accuracy does not have to grow with the data -- KSS report the error *falling*
# with sample size at fixed draws -- but you should know how big it is, and the
# only way to find out is to vary the seed.
print(f"\n{'=' * 72}\nhow much the approximation moves the answer\n{'=' * 72}")
runs = [step] + [fit.leave_out_kss(n_draws=DRAWS, seed=s) for s in (1, 2, 3)]
values = np.array([[run.leave_out[k] for k in truth] for run in runs])
print(f"across {len(runs)} seeds at {DRAWS} draws:\n")
print(f"{'component':<18}{'mean':>11}{'spread':>11}{'relative':>11}")
for index, key in enumerate(truth):
    column = values[:, index]
    print(f"{key:<18}{column.mean():>11.5f}{np.ptp(column):>11.5f}"
          f"{np.ptp(column) / abs(column.mean()):>11.2%}")

print("\nThat spread shrinks with more draws; docs/kss.md reports it across five")
print("seeds at 250 and 1,000 draws. What matters is that it is small next to the")
print("sampling uncertainty of the components themselves -- about a tenth of the")
print("standard error when leaving out a match -- so the approximation is not")
print("what limits this answer.")
# --------------------------------------------------------- standard errors
# `se=True` adds standard errors and 95% intervals. This section leaves out an
# observation, so that all three components get one. Leaving out a match
# reports them for var(psi) and the covariance only, following LeaveOutTwoWay's
# leave_out_COMPLETE, whose match-level option is marked beta (see docs/kss.md).
# The estimator is a quadratic form y'Cy whose kernel has a zero diagonal by
# construction, which is what makes it unbiased and what makes its variance
#
#     V = 4 sum_i sigma2_i (C mu)_i^2 + 2 tr(C Omega C Omega)
#
# hold with no normality assumption. The first term alone is conservative, so it
# is reported as well; `se_trace=False` computes only that and saves roughly
# three solves per draw.
print(f"\n{'=' * 72}\nstandard errors\n{'=' * 72}")
with_se = leave_out_kss(FML, str(data), workdir=workdir("leave_out_se"),
                        n_draws=DRAWS, seed=0, leave_out="observation", se=True)
print()
print(with_se.summary())

# Do the intervals contain the answer the data was built from? With three
# components and one draw of the data this is an anecdote, not a coverage study
# -- docs/kss.md reports coverage measured over many redraws -- but it is the
# check worth making on your own data's scale.
print(f"\n{'component':<18}{'truth':>10}{'leave-out':>11}{'se':>10}"
      f"{'z':>7}{'covers':>8}")
for key, true_value in truth.items():
    se = with_se.se.se[key]
    z = (with_se.leave_out[key] - true_value) / se
    print(f"{key:<18}{true_value:>10.5f}{with_se.leave_out[key]:>11.5f}"
          f"{se:>10.5f}{z:>7.2f}{'yes' if abs(z) < 1.96 else 'no':>8}")

# Those intervals come with a check on whether they are justified at all. The
# normal approximation needs no single eigen-direction to dominate the
# estimator's variance; `q` counts the ones that do, by KSS's threshold of 1/10
# on lambda^2 / sum lambda^2, and only q = 0 justifies the interval. A bottleneck
# in the mobility network -- two groups of firms joined by a few movers -- is
# what breaks it. This panel is well connected, so q should be 0 throughout.
print("\nis the normal interval justified?  q per component:",
      with_se.weak_id.q)

# `se.theta` recomputes the same estimand as y'Cy -- per-row B_ii by random
# projection rather than the coefficient-space trace.
# It shares almost no code with the point estimate, so agreement is a real check.
print("\ncross-check, two independent routes to the same estimand:")
for key in truth:
    gap = abs(with_se.se.theta[key] - with_se.leave_out[key]) / with_se.se.se[key]
    print(f"  {key:<18}{with_se.se.theta[key]:>11.6f} vs "
          f"{with_se.leave_out[key]:>11.6f}   {gap:.2f} se apart")

# ------------------------------------------------- when the interval fails
# Everything above was on a well-connected panel. A bottleneck in the mobility
# network -- two groups of firms joined by a handful of movers -- is where the
# normal interval stops being justified: one linear combination of the effects
# is estimated far worse than the rest, and it dominates the estimator's
# variance. The output then flags it, and for each flagged component adds KSS's
# q = 1 interval, which stays valid there.
print(f"\n{'=' * 72}\na weakly identified panel\n{'=' * 72}")
weak_data = OUTPUT / "bottleneck_panel.parquet"
simulate_bottleneck(n_bridge=6, keep_effects=True).write_parquet(weak_data)
weak = leave_out_kss("log_earn ~ 1 | worker_id + firm_id", str(weak_data),
                     workdir=workdir("leave_out_weak"), n_draws=DRAWS, seed=0,
                     leave_out="observation", se=True, verbose=False)
print(weak.summary())

# Against the truth, once more. One draw of the data is an anecdote; docs/kss.md
# reports coverage over many. Note the q = 1 interval is not symmetric about the
# estimate: the badly estimated direction enters squared, so its uncertainty is
# skewed, which a normal interval cannot express.
kept = weak.pruning.data.select("true_worker_effect", "true_firm_effect").collect()
a_true = kept["true_worker_effect"].to_numpy()
p_true = kept["true_firm_effect"].to_numpy()
weak_truth = {"var(psi)": float(np.var(p_true, ddof=1)),
              "var(alpha)": float(np.var(a_true, ddof=1)),
              "cov(psi, alpha)": float(np.cov(p_true, a_true, ddof=1)[0, 1])}
print(f"\n{'component':<18}{'truth':>10}{'normal covers':>15}{'q=1 covers':>12}")
for key, true_value in weak_truth.items():
    se = weak.se.se[key]
    normal = abs(weak.leave_out[key] - true_value) <= 1.96 * se
    lo, hi = weak.se.weak[key]["interval"] if key in (weak.se.weak or {}) else (
        float("nan"), float("nan"))
    print(f"{key:<18}{true_value:>10.5f}{'yes' if normal else 'no':>15}"
          f"{'yes' if lo <= true_value <= hi else 'no':>12}")

# ------------------------------------------------------------------ weights
# `weights=` works here too, and then the estimand is the *weighted* variance
# decomposition -- the weighted variances and covariance of the effects, over
# observations. That one choice fixes everything else: the leverages become the
# ones of the square-root-weight hat matrix, which is the only version that is a
# projection, and sigma2 carries a factor of w. Nothing to pass but the weights.

print("\nFor a paper: fix the seed and the number of draws and report both, and")
print("report the estimate across several seeds rather than one. The claim worth")
print("making is that your result is stable to the approximation -- not that one")
print("particular draw reproduces.")

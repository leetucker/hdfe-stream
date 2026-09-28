# Methodological differences from the reference KSS implementations

`hdfe-stream` implements the standard cases of Kline, Saggio and Sølvsten
(2020): the leave-out bias correction for AKM variance components, its standard
errors, and the weak-identification diagnostic and q = 1 interval. It is not an
extension of that methodology. Where it departs from the published reference
implementations, this document says so, with what the references do, what this
package does, why, and the measured effect.

How to use the implementation is in [kss.md](kss.md); this document is only
about how it differs from the references.

Everything below was checked against the source code, not documentation alone,
of the versions listed. Claims about another package's behavior cite the file
they come from.

## References compared

| package | language | version read | scope |
|---|---|---|---|
| **LeaveOutTwoWay** (Saggio) — `leave_out_KSS.m` | MATLAB | `8b957ff`, 2024-10-17 | the current entry point: point estimates, `lincom` |
| **LeaveOutTwoWay** — `leave_out_COMPLETE.m` | MATLAB | same commit | the older entry point: also component standard errors (on by default, `do_SE`), eigenvalue diagnostics, the AM q = 1 interval. Leaves out an observation by default; its match-level option is documented as "currently in beta mode and needs further testing" |
| **VarianceComponentsHDFE.jl** (Corcuera) | Julia | `fa3d66e`, 2024-01-04 | point estimates, `lincom` |
| **xhdfe** (Portela) | C++/R/Stata/Python | `e0c2362`, v2.28.0, 2026-09-24 | point estimates; SEs and weak-ID on request (`compute_se` is off by default), ported from `leave_out_COMPLETE`; states machine-precision agreement with LeaveOutTwoWay |
| **pytwoway** (Lamadon, Oppenheimer) | Python | 0.3.21 (bipartitepandas 1.1.9) | point estimates; homoskedastic correction by default |

Where the two Saggio entry points disagree, both are given; `leave_out_KSS` is
what the repository's README describes as current.

**Reference implementation for standard errors when leaving out a match.**
Only one exists: `leave_out_COMPLETE` with `leave_out_level='matches'`, a
non-default option its own documentation calls beta. xhdfe ports that code
path, and since MATLAB was not run here, xhdfe is what the port in
`prototypes/` was validated against. `leave_out_KSS`,
VarianceComponentsHDFE.jl and pytwoway report no component standard errors at
all. The person-year variance estimate described in 5.7 is the only one
`leave_out_COMPLETE` implements, so it is the reference's method, not a default
it chose over an alternative.

## How to read the entries

Each difference is classed as one of:

- **Estimand** — changes what is estimated, or the point estimate beyond Monte
  Carlo error.
- **Estimator** — same estimand, a different estimator of it: numbers differ
  within sampling or Monte Carlo error.
- **Numerical** — a different computational route to the same quantity; results
  agree to solver or Monte Carlo precision.
- **Reference issue** — the reference's published code departs from the paper,
  verified here by direct test.

Entries marked **agrees** are not differences. They are listed where the
references disagree among themselves, so which one this package follows needs
saying, or where the behavior is easy to get wrong.

---

## 1. The estimation sample

### 1.1 Leave-out unit — observation vs match  ·  *Estimand; follows the current references*

| | |
|---|---|
| LeaveOutTwoWay `leave_out_KSS` | leaves out a whole worker–firm **match** by default (`leave_out_level='matches'`); README: "By default, the code runs a leave-out correction by leaving a match out as opposed to leaving an observation out" |
| LeaveOutTwoWay `leave_out_COMPLETE` | leaves out an **observation** by default; `'matches'` marked beta |
| VarianceComponentsHDFE.jl | **match** by default (`leave_out_level::String = "match"`) |
| xhdfe | **match** by default, described as "Saggio's default"; `"obs"` available |
| pytwoway | user's choice through bipartitepandas `connectedness` (observation, spell, match, worker, firm) |
| **hdfe-stream** | **match** by default (`leave_out="match"`), following the references; `leave_out="observation"` available |

**Effect.** Leaving out an observation treats a worker's years at one firm as
independent. With serially correlated errors within a spell — typical of
earnings — the leave-one-observation-out residual is not independent of the
observation, and the variance estimate `σ̂²ᵢ` is biased. Leaving out the match
removes that dependence. Measured in the dense prototype (2,568 person-years,
486 matches, 200 replications, half the error variance a spell-level shock):

| | observation level: bias (t) | match level: bias (t) |
|---|---|---|
| var(psi), truth 0.1059 | +0.0066 (6.9) | −0.0003 (−0.3) |
| cov(psi, alpha), truth 0.0340 | −0.0051 (−11.7) | +0.0000 (0.0) |
| var(alpha), truth 0.1338 | +0.0400 (47.8) | +0.0237 (25.8) |

With no spell shock both levels are unbiased. `var(alpha)` stays biased at match
level: a stayer's worker effect cannot be separated from its single spell's
shock by any leave-out scheme (3.5).

**Implementation.** The reference's three steps (`leave_out_KSS.m`): partial
out everything but the two effects using the full model, collapse to one row per
match (mean outcome, weighted by spell length times any user weight), and run
the correction on that weighted regression. In the square-root-weight metric a
collapsed row's leverage is its match's, so the rest of the machinery applies
unchanged; the quadratic forms remain person-year moments. The prototype's
version (`kss_match`) agrees with xhdfe to 5e-14–5e-12 on two panels; the
streaming version agrees with the prototype to within Monte Carlo error.
Standard errors at match level are described in 5.7. The q = 1 interval for the
covariance is not available at match level (6.9).

### 1.2 Pruning to the leave-one-out connected set — iteration  ·  *agrees*

| | |
|---|---|
| LeaveOutTwoWay | `pruning_unbal_v3.m` loops `while n_of_bad_workers>=1`: remove articulation-point workers, take the largest connected set, recompute |
| VarianceComponentsHDFE.jl | loops `while nbadfirsts>0` (`prunning_connected_set`) |
| **hdfe-stream** | loops like the references: remove articulation-point workers, keep the largest component, repeat until none remain |

One round is not enough, because deleting cut vertices can create new ones. A
4-cycle W1–F1–W2–F2–W1 with a firm F3 attached only to W1: W1 is a cut vertex;
deleting it leaves F1–W2–F2, where W2 is a new one. This panel is a test case.
Were W2 kept, `leave_out_components` would find leverage 1 and raise "not
leave-one-out connected".

The pruned sample matches xhdfe's row for row on three panels, with the same
number of rounds (3, 1, 2). Diagnostics report `pruning_rounds`.

### 1.3 Workers with a single observation  ·  *agrees*

| | |
|---|---|
| LeaveOutTwoWay (both) | drop them after pruning (`sel=T>1`) |
| VarianceComponentsHDFE.jl | `drop_single_obs` |
| **hdfe-stream** | dropped once, before the articulation loop, as in xhdfe |

A worker seen once has leverage exactly 1: the worker effect is determined by that
observation. A pendant worker is never an articulation point, so pruning alone
never removes it, and the leverage check would raise. Tested on a minimal
panel. The resulting sample matches xhdfe's. Diagnostics report
`single_observation_workers`.

### 1.4 The graph searched for articulation points  ·  *Numerical*

LeaveOutTwoWay searches the movers-only bipartite graph; this package searches
the full one. Stayers are pendant vertices and never articulation points, so the
two agree.

---

## 2. The model

### 2.1 Controls  ·  *Estimator*

| | |
|---|---|
| LeaveOutTwoWay `leave_out_KSS`, VarianceComponentsHDFE.jl, xhdfe | partial them out first — fit the full model, form `y − Xb̂`, then run the leave-out machinery on the two-way design without `X` |
| LeaveOutTwoWay `leave_out_COMPLETE` | default `resid_controls=0`: controls stay in the design |
| **hdfe-stream** | controls stay in the design; leverages and `S⁻` use a bordered Schur complement |

**Effect.** Partialling out treats `b̂` as known, so the leverages omit the
controls' contribution — an O(k/n) difference for k controls, negligible for a
handful of year effects. Keeping them follows the paper's formula for the full
design.

### 2.2 Weights  ·  *Estimand for weighted fits*

| | |
|---|---|
| LeaveOutTwoWay, VarianceComponentsHDFE.jl | no user weights; match length is used as an internal weight at match level |
| xhdfe | frequency weights only, match-level point estimates only |
| pytwoway | analytic weights; leverages in the square-root-weight metric |
| **hdfe-stream** | analytic weights; the estimand is the weighted variance decomposition; leverages in the square-root-weight metric, weighted plug-in moments, `σ̂²` carrying a factor of `w` |

This is the one feature beyond the MATLAB, Julia and xhdfe references. The
leverage convention matches pytwoway. At match level user weights compose with
spell length: a match's weight is its total user weight, and the stayer rule
(3.2) weights each person-year by its share of it; tested against the prototype. Validated by Monte Carlo against known effects in `prototypes/`.

---

## 3. The per-observation variance `σ̂²ᵢ`

### 3.1 Demeaned outcome  ·  *agrees with current references*

`σ̂²ᵢ = (yᵢ − ȳ)·η̂ᵢ` with `η̂ᵢ` the leave-out residual, in LeaveOutTwoWay
`leave_out_KSS` (`sigma_i=(y-mean(y)).*eta_h`), VarianceComponentsHDFE.jl,
xhdfe and pytwoway — and here. Only the older `leave_out_COMPLETE` uses the raw
outcome (`y'*Lambda_B*eta_h`). The paper's `yᵢ(yᵢ − xᵢ'β̂₋ᵢ)` is exactly
unbiased; demeaning trades an O(1/n) bias for a large variance reduction,
because `yᵢ` otherwise carries the outcome's whole mean.

### 3.2 Stayers  ·  *agrees with LeaveOutTwoWay by default; pytwoway's rule optional*

| | |
|---|---|
| LeaveOutTwoWay, VarianceComponentsHDFE.jl, xhdfe — observation level | every observation uses its own `σ̂²ᵢ`, stayers included |
| same — match level | a stayer's single match cannot be left out, so its variance comes from its own leave-one-*observation*-out residuals, averaged within the match (`sigma_for_stayers.m`) |
| pytwoway | `Sii_stayers='firm_mean'` (default): stayers get the mean of movers' estimates at their firm |
| **hdfe-stream** | LeaveOutTwoWay's rules at both levels: own `σ̂²ᵢ` at observation level (`stayers="own"`, the default) and `sigma_for_stayers.m` at match level. At observation level, `stayers="firm_mean"` (pytwoway's rule) and `"drop"` are available |

**Effect of the options.** The firm-mean imputation exists for data where a
stayer has one observation, as with spell-level data; at observation level a
stayer seen twice or more has a well-defined leave-out residual. Measured: the
imputation moves `var(alpha)` by under 0.1 standard errors on the simulated
panels and leaves `var(psi)` and `cov` unchanged, but it would matter more where
stayers' error variances differ from movers'. With it, the reported estimate
also differs slightly from `se.theta`, because the standard errors describe the
un-imputed estimator. Under the default every row uses its own estimate, so the
reported point estimate and `se.theta` describe the same estimator.

**How the match-level rule meets the collapsed fit.** Collapsing to matches is
what makes the rule necessary: a stayer's one match row has leverage exactly 1.
So its variance comes from its person-year rows, which the collapsed table no
longer holds. The rule is computed in a streaming pass over the row-level fit's
residual file: each year's `(y~ᵢₜ − ȳ)(y~ᵢₜ − ŷᵢ)/(1 − wᵢₜ/Wᵢ)`, with `ŷᵢ` the
collapsed fit's prediction for the match and `y~` the partialled-out outcome,
averaged within the match as `Σ w²ᵢₜ rᵢₜ / Σ wᵢₜ` — the match row's variance in
the square-root-weight metric, and a plain mean when unweighted. Agrees with the
prototype to 1e-12 (it has no randomness).

### 3.3 The random-projection correction  ·  *agrees*

The five-moment leverage construction with the non-linearity correction of
Saggio's `improved_JLA.pdf` is used by LeaveOutTwoWay `leave_out_KSS`
(`leverages.m`), VarianceComponentsHDFE.jl, pytwoway and here. The older
`leave_out_COMPLETE` uses an earlier projection (`eff_res.m`, "JLL") without it.

### 3.4 Centering at match level  ·  *Estimator, optional: the reference's by default*

| | |
|---|---|
| LeaveOutTwoWay `leave_out_KSS`, xhdfe | `σ̂²` uses `√w·ȳ − mean(√w·ȳ)`: transform by the square root of spell length, then subtract the unweighted mean over matches |
| **hdfe-stream** | the same by default (`centering="reference"`); `centering="weighted"` uses `√w·(ȳ − ȳ_w)`, centering at the weighted mean before transforming. **The option is a documented deviation.** |

At observation level, and at match level when every spell has the same length,
the two are the same number (tested). The option only matters at match level
with spells of varying length.

**Motivation.** Write `c̄` for the mean of `√w` over matches. The reference's
centered outcome for match `i` is `√wᵢ·ȳᵢ − mean(√w·ȳ)`, which contains
`(√wᵢ − c̄)·ȳ_w`: a term proportional to the outcome's *level* that does not
vanish when spell lengths differ. With log earnings, mean about 10, it is the
dominant part of each `σ̂²ᵢ`. Centering `ȳ` before transforming removes it.

The same algebra shows that the reference's estimate depends on the outcome's
location: adding a constant `k` to the outcome changes each centered value by
`k·(√wᵢ − c̄)`, so, for example, earnings in logs of euros and in logs of
thousands of euros give different estimates. The weighted centering does not
change when a constant is added. Checked exactly in the dense prototype, on one
pruned 400-worker panel with log earnings of mean about 10:

| outcome | reference: var(psi) / var(alpha) / cov | weighted |
|---|---|---|
| `y − 10` | 0.0563 / 0.1666 / 0.0300 | 0.0569 / 0.1671 / 0.0295 |
| `y` | 0.0435 / 0.1573 / 0.0415 | same |
| `y + 5` | 0.0371 / 0.1527 / 0.0472 | same |

Each difference is a different realization of sampling noise, not a bias.
Across repeated samples both centerings average to the truth (below). But a
reported estimate that depends on the units of the outcome is a property worth
knowing. The weighted centering's invariance is tested in the streaming
implementation.

Both are unbiased. `σ̂²ᵢ` is the centered outcome times the leave-out residual
divided by `Mᵢᵢ`, and the leave-out residual has mean zero and is uncorrelated
with any fixed constant subtracted from the outcome. So the level term adds
variance, not bias. (Each centering constant is itself estimated, which adds an
O(1/n) term to both; the Monte Carlo below finds neither biased.)

**Benefits of `centering="weighted"`**, measured with the outcome at mean 10 and
spells of varying length:

- *Lower sampling variability.* Over 200 samples in the dense prototype the
  standard deviations were 0.0122 / 0.0090 / 0.0063 (var(psi) / var(alpha) /
  cov), against 0.0205 / 0.0269 / 0.0151 with the reference centering: 1.7 to
  3 times smaller. Both were unbiased. With the outcome at mean 0 the two
  were identical.
- *Lower Monte Carlo variability at a given number of draws.* The larger
  `σ̂²` under the reference's centering inflate the Hutchinson trace's noise in
  proportion. Streaming at 2,048 draws over three seeds on one panel, the
  spread was 0.4% / 0.3% / 0.2% of the value, against 6.4% / 1.5% / 13.9%.
  Since Monte Carlo error falls as one over the square root of the number of
  draws, the reference's centering would need roughly 25 to several thousand
  times as many draws to match (three seeds each, so only indicative).
- *An illustration, not a test.* On the 8,000-worker panel in
  `examples/leave_out_kss.py`, the error against the known truth in
  var(psi) / var(alpha) / cov was +0.0020 / +0.0001 / −0.0004 with the
  weighted centering and +0.0043 / +0.0071 / −0.0019 with the reference's,
  identical across two seeds. That is one sample, so it shows the size of the
  difference that sampling noise can make, not a systematic one.

**Costs of `centering="weighted"`.**

- *It does not reproduce the reference.* With spells of varying length its
  estimates differ from LeaveOutTwoWay's and xhdfe's on the same data, by an
  amount of the order of the reference's own sampling noise. Anyone
  replicating a published result computed with those packages needs the
  default.
- *Its advantage depends on the outcome's level.* When the outcome's mean is
  near zero or the spells are all the same length, there is nothing to gain.
  Equivalently, a user of the reference can recover most of the benefit by
  demeaning the outcome first. Under the reference's centering that changes
  the estimate, because of the location dependence above.
- No computational cost: the weighted mean is one more scalar in a pass that
  already happens.

**Status.** The default follows the reference, and
`centering="weighted"` is available as a documented deviation of the
*estimator* class: same estimand, lower variance.

### 3.5 What match-level stayers' variance targets  ·  *Reference property, kept*

`sigma_for_stayers.m` uses the stayer's *within-match* residuals. Those remove
any spell-level shock entirely, so under within-spell correlation the rule
estimates only the idiosyncratic part of the variance, while movers' match-level
`σ̂²` include the spell shock. This is an inconsistency in the reference, kept
here to match it; it is also unavoidable in part, since for a stayer the spell
shock is indistinguishable from the worker effect. It is the reason `var(alpha)`
remains biased at match level under spell shocks (1.1 table); `var(psi)` and the
covariance, which rest on movers, are not affected.

---

## 4. The point estimate

### 4.1 Divisor of the plug-in moments  ·  *agrees with `leave_out_KSS`*

| | |
|---|---|
| LeaveOutTwoWay `leave_out_KSS`, VarianceComponentsHDFE.jl | `n − 1` in both the plug-in (`cov`) and the bias (`1/dof`, `dof = n−1`) |
| LeaveOutTwoWay `leave_out_COMPLETE` | `n − 1` in the plug-in, `n` in the bias — inconsistent, an O(1/n) difference |
| **hdfe-stream** | `n − 1` person-years in the plug-in, the bias and the standard errors, at both leave-out levels |

With weights the weight total `W` is scaled the same way, `W(n − 1)/n`. The
centering of the moments uses `W`.

### 4.2 Leverages  ·  *Numerical*

References offer an exact algorithm for small samples (LeaveOutTwoWay's default
below 10,000 observations) and 200 random-projection draws otherwise. This
package always projects, 250 draws by default. Its random streams are keyed on
the absolute row ordinal, so results do not depend on thread count or chunk size;
VarianceComponentsHDFE.jl seeds one generator per thread, so its results do.

### 4.3 The bias term  ·  *Estimator*

| | |
|---|---|
| LeaveOutTwoWay, VarianceComponentsHDFE.jl, xhdfe | per-row `Bᵢᵢ` by random projection, then `Σ Bᵢᵢσ̂²ᵢ` |
| pytwoway | Hutchinson estimate of the trace, `ndraw_trace_he=5` by default |
| **hdfe-stream** | Hutchinson estimate of `tr(Q S⁻ A'ΩA S⁻)` in coefficient space, 250 draws by default |

Both routes are unbiased for the same trace. The trace route never forms the
row-sized `Bᵢᵢ` and has lower Monte Carlo error at equal cost. Validated against
exact dense traces.

---

## 5. Standard errors (the strongly identified case)

Only LeaveOutTwoWay `leave_out_COMPLETE` and xhdfe report standard errors for the
components.

### 5.1 The variance formula  ·  *agrees*

`V̂ = 4 Σ σ̃²ᵢ (Cy)ᵢ² − 2 tr(CΩ̃CΩ̃)`, with `C` the zero-diagonal kernel of
the leave-out estimator — term for term the reference's
`V=(1/NT^2)*(4*SUM-left_part)` (`leave_out_estimation_two_way.m`).

### 5.2 Estimating the trace term  ·  *Estimator*

The references take `var(v'Cv)` over 1,000 Gaussian draws of `v = Ω̃^½z`, which
equals `2tr(CΩ̃CΩ̃)` for Gaussian `z`. This package averages `Σᵢ σ̃²ᵢ(Cv)ᵢ²`,
whose expectation is `tr(CΩ̃CΩ̃)` for any `z` with identity covariance, with
Rademacher draws; one application of `C` per draw either way. Default draws:
the point estimate's (250), not 1,000.

### 5.3 The smoother for `σ̃²`  ·  *Estimator*

| | |
|---|---|
| LeaveOutTwoWay `leave_out_COMPLETE` | `llr_fit`, default mode 2: movers' `σ̂²` by weighted local-linear lowess on `(Bᵢᵢ, Pᵢᵢ)` over percentile cells, span `NT^(−1/3)`; stayers by the mean within cells of `Tᵢ` |
| xhdfe | mode 4 by default: cell means on a 1,000 × 1,000 percentile grid; lowess (mode 0) optional |
| **hdfe-stream** | cell means on an `n^(1/6)` × `n^(1/6)` grid in quantile rank of `(Pᵢᵢ, Bᵢᵢ)`, interpolated bilinearly; movers and stayers pooled |

**Motivation.** The reference's lowess is not streamable. The grid size matches
the reference's bandwidth: lowess with span `n^(−1/3)` fits each point from
`n^(2/3)` neighbors, and `n^(1/6)` bins per axis gives cells of that size. The
interpolation keeps the fit continuous in its regressors, as lowess is. A step
function would make quantities resting on few rows, such as `V̂[b̂₁]` (6.4, 6.6),
jump with random-projection noise.

**Effect.** Coverage of the 95% interval with this smoother (and own `σ̂²` for
stayers, `n − 1`), 800 replications: 94.9 / 95.2 / 93.4% (var(psi) /
var(alpha) / cov, n = 1,540) and 95.0 / 95.0 / 94.6% (n = 5,120), against
93.9 / 94.7 / 93.1% and 94.6 / 94.9 / 94.4% with a fixed 12 × 12 grid instead. With
the weak-ID q = 1 interval on the six-bridge bottleneck, 95.0–97.8% across all
cells.

**A limitation shared with the reference.** The smoother's regressors `Pᵢᵢ` and
`Bᵢᵢ` are random-projection estimates, so which rows share a cell depends on the
seed, and each cell mean carries sampling noise from the noisy `σ̂²ᵢ`. Quantities
resting on few rows inherit it: on a 1,600-row bottleneck panel `V̂[b̂₁]` ranged
over roughly ±25% across five seeds (truth 0.0225, estimates 0.021–0.032), and
the q = 1 interval's upper end moved by about 15% of the interval's width. With
exact `Pᵢᵢ` and `Bᵢᵢ` the same estimate did not move with the seed at all. Cells
hold `n^(2/3)` rows, so this falls as `n^(−1/3)`; fixing `seed` makes results
reproducible. The reference's lowess is
fed random-projection `Pᵢᵢ` and `Bᵢᵢ` too, so it is exposed to the same mechanism;
that was not measured here.

### 5.4 What the smoother averages  ·  *agrees — alternative tested and rejected*

Both smooth the leave-out `σ̂²ᵢ` itself. An alternative was tested because that
input is noisy — its coefficient of variation is near 3, since `ỹᵢ` carries the
outcome's mean — and the noise shows up in `V̂[b̂₁]` (6.6), which rests on the few
rows the weak direction loads on. Smoothing `wᵢε̂²ᵢ/(1 − Pᵢᵢ)` instead drops that
factor; its expectation is a local weighted average of `σ²` with weights
`M²ᵢₗ/Mᵢᵢ` summing to one, which is what a smoother targets anyway.

Measured on the six-bridge bottleneck (300 replications): the dispersion of
`V̂[b̂₁]` fell only from a coefficient of variation of 0.20 to 0.17–0.18 (0.08 to
0.09 for `var(alpha)`), and coverage of the q = 1 interval was unchanged within
Monte Carlo error (94.7–97.3% against 95.3–98.0%). The dispersion is dominated by
how few rows `V[b̂₁]` rests on, not by the input's noise. Not decisive, so not
adopted: this package keeps the reference's input.

### 5.5 Negative values  ·  *Estimator*

The references neither floor the fitted `σ̃²` (`llr_fit` prints the share of
negative fitted values) nor guard a negative `V̂` (they take `sqrt(V_theta)`).
This package floors `σ̃²` at zero inside the trace draws, reports `se = NaN`
when `V̂ ≤ 0`, and always reports the first term alone as `se_conservative`
(upward biased; KSS Remark 9).

### 5.6 The split-sample `σ̃²` of the paper  ·  *shared*

KSS §4.2's cross-fit `σ̃²` needs two edge-disjoint paths per observation. No
reference implements it for the component standard errors, and neither does this
package.

### 5.7 Standard errors at match level  ·  *Estimator; reference: `leave_out_COMPLETE`, matches (beta)*

**Reference implementation:** LeaveOutTwoWay's `leave_out_COMPLETE.m` with
`leave_out_level='matches'`. That option is not its default (it leaves out an
observation by default), and its documentation says it "is currently in beta
mode and needs further testing". No maintained package offers anything else:
`leave_out_KSS` reports no standard errors, and xhdfe's are a port of this
path. Everything below follows `leave_out_COMPLETE`'s match-level path, except
the differences listed as such. It reports standard errors for var(psi) and the
covariance only.

**What is the same.**

- *The kernel.* Every regressor is constant within a match, so the person-year
  hat matrix's block for match m is `p_m 11'`. The reference's block leave-out
  kernel is `C = B − ½(Λ_B(I − Λ_P)⁻¹M + M(I − Λ_P)⁻¹Λ_B)`. Writing `E` for the
  person-year-by-match indicator and `D = diag(√T)`, every piece of it is
  `E(·)E'`, and conjugating by `D` gives exactly the collapsed regression's own
  zero-diagonal kernel `C̃ = B̃ − ½(diag(b̃)M̃ + M̃ diag(b̃))`, `b̃ = B̃ᵢᵢ/(1 − P̃ᵢᵢ)`.
  So the machinery the observation level already uses applies to the collapsed
  rows unchanged. The prototype checks the identity to 10⁻¹³, and streaming to
  10⁻¹⁰.
- *Stayers.* A stayer's match has `P̃ = 1`. For var(psi) and cov its `B̃ᵢᵢ` is
  exactly zero, because a stayer's outcome does not move ψ̂. So `b̃ = 0` there,
  as in the reference, and the diagonal stays zero.
- *var(alpha) is not reported.* It leans on stayers, whose variance is not a
  leave-out estimate. The summary says so.

**Validated against xhdfe, the reference's port.** `prototypes/kss_exact.py::xhdfe_match_se`
is a literal port of xhdfe's code, which xhdfe describes as a port of
`leave_out_COMPLETE` validated against it at machine precision. The port includes the raw outcome, the 1,000 × 1,000
quantile grid, the complex-valued simulation where σ̃ < 0, and the truncation
at zero. On a pruned simulated panel it reproduces:
- xhdfe's `theta_c` to 10⁻¹³;
- its standard error to within the simulation error of 20,000 draws;
- its zeros.

The recorded values are a regression test.

**Difference 1: the outcome is centered (as at observation level).** The
reference uses the raw outcome in `σ̂`, in `Cy` and in `theta_c`. Since
`C1 ≠ 0`, the estimator and its variance then carry the outcome's level. With
log earnings (mean about 10) the variance estimate is dominated by it and comes
out negative: xhdfe truncates the standard error to **zero**. In the Monte Carlo
below that happened in 84–99.6% of samples for var(psi) and 55–98% for cov, and
the same happens on xhdfe's observation-level path. Demeaning the outcome
first, which a user of xhdfe can do, gives positive standard errors. This
package centers the outcome, as the point estimate does. With
`centering="weighted"` the variance is that of `ỹ'C̃ỹ`, which *is* the point
estimate. With the reference's centering the point estimate is exactly
`y_r'Ky_r`, with `y_r = √T ȳ` and `K = C̃ + (1u' + u1')/(2M)`, `u = M̃b̃`. That is
a rank-two update whose diagonal, `u_m/M`, is O(1/M) rather than zero, and the
variance reported is that of `K`. So in every case the standard error describes
the estimate it is printed next to.

**Difference 2: σ̃ is the package's smoother, floored, with the Hutchinson
trace (5.2, 5.3, 5.5).** The reference's "mode 4" is a fixed 1,000 × 1,000 grid.
Below about 10⁶ person-years it leaves roughly one person-year per cell, so σ̃
is essentially unsmoothed and often negative. The simulated trace then goes
complex and inflates. Even with the outcome demeaned, the port's coverage was
89–92% with independent errors and 66–77% with spell shocks (table below).

**Difference 3: user weights.** The reference has none. The default
`se_variance="person_year"` therefore refuses a weighted fit before fitting,
and points to `"match"`, which supports weights.

**Difference 4: no q = 1 interval for the covariance** at match level; see 6.9.

**An option beyond the reference: what estimates each match's variance.** `V`
needs, per collapsed row, the variance of `√T` times the match mean,
`1'Ω_m1/T`. `se_variance` chooses the estimate:

- `"person_year"` (default): the reference's, and the only one
  `leave_out_COMPLETE` implements. It is `Σ_{i∈m} y_i η_h,i / T`, with the
  outcome centered (difference 1). In collapsed terms that is the within-match
  variance plus the collapsed leave-match-out σ̂² over `T`, the two pooled. It
  estimates `tr(Ω_m)/T`, only the *diagonal* of the match's error covariance, so
  it assumes errors are independent within a spell. A stayer gets its
  within-match variance, as in the reference (`η_h = η` on its singular block).
- `"match"`: not in any reference. It uses the collapsed leave-match-out σ̂²
  alone, which estimates the whole block, so it stays right when errors are
  correlated within a spell. It is noisier, having one degree of freedom per
  match. A stayer gets `sigma_for_stayers`, as in the point estimate.

**Effect**, measured by Monte Carlo (dense prototype, design and effects fixed,
errors redrawn, 500 replications per cell, so ±1 point). Coverage of the 95%
interval, var(psi) / cov:

| panel | errors | `"match"` | `"person_year"` | xhdfe port, demeaned | xhdfe port, raw |
|---|---|---|---|---|---|
| 2,128 py, 406 matches | independent | 92.8 / 87.0 | 94.8 / 95.0 | 89.8 / 89.8 | 14.8 / 45.0 |
| 2,128 py, 406 matches | half the variance a spell shock | 95.4 / 89.0 | 86.0 / 83.6 | 71.2 / 66.2 | 3.2 / 38.4 |
| 4,233 py, 837 matches | independent | 96.0 / 91.8 | 95.8 / 93.6 | 92.0 / 89.2 | 3.8 / 5.2 |
| 4,233 py, 837 matches | half the variance a spell shock | 95.0 / 92.2 | 86.2 / 82.0 | 77.4 / 68.6 | 0.4 / 1.4 |

(Package columns use the weighted centering. With the reference's centering
and `"match"`, coverage was 91.6–98.2%, and the estimates' standard deviation
was 2–5 times the weighted centering's: see 3.4.)

With independent errors, the reference's source is the better of the two at
small samples. The `"match"` source's shortfall for cov (87–92%) shrinks with
the number of matches. Under spell shocks the reference's source understates
the standard error by 25–30%, and more data does not fix it, because it
estimates only the diagonal of each spell's error covariance. That is worth
knowing, because within-spell correlation is what leaving out a match is meant
to be robust to. `"match"` is the choice when such correlation is plausible.

**Status.** `se_variance="person_year"` is the default, following the
reference implementation, with the differences 1–4 above. `"match"` is an
option, documented here as going beyond the reference.

### 5.8 Centering the outcome in the standard errors  ·  *Estimator*

At observation level too, `leave_out_COMPLETE` uses the raw outcome in `σ̂` and
`Cy`, while this package uses `√w(y − ȳ_w)` throughout, as its point estimate
(and `leave_out_KSS`'s) does. The kernel does not annihilate a constant,
`C1 = −½Mb ≠ 0`, so the two are different estimators of the same quantity.
Both are unbiased, and the raw one carries the outcome's level. On a simulated
log-earnings panel (1,503 person-years), xhdfe's observation-level `theta_c`
for var(psi) was 0.078 with the raw outcome and 0.050 with it demeaned, where
`leave_out_KSS`'s estimate is 0.050. Its standard errors were zero for all three
components with the raw outcome, and positive once it was demeaned.

---

## 6. Weak identification

Only LeaveOutTwoWay `leave_out_COMPLETE` (`eigen_diagno=1`) and xhdfe
(`eigen_diagnostics`) implement it.

### 6.1 Leading eigenvalues  ·  *Numerical*

The references run `eigs` on an explicit augmented matrix pencil (`eigAux.m`),
which needs `S` in memory. This package runs Lanczos in the `S⁻` inner product,
one solve per step, with the Krylov dimension capped at its bound and a
Cullum–Willoughby filter for spurious copies. Eigenvalues agree with dense
`eig(S⁻Q)` to 1e-13 at convergence; the default stopping tolerance (1e-4 on the
residual bound) leaves them within 6e-5 of a converged run.

### 6.2 `Σλ² = tr(Ã²)`  ·  *Estimator*

Both use Hutchinson with row-space Rademacher draws and a squared norm
(`trace_Atilde_sqr.m`; xhdfe 100 draws). This package also **deflates** by the
Ritz vectors Lanczos found — `tr(A²) = Σ_ℓ ‖Ax_ℓ‖² + E⟨Aζ_def, Aζ_def⟩`, exact for
any orthonormal `x_ℓ` — and stops drawing once no ratio is within two standard
errors of 1/10. Without deflation the relative error near a dominant eigenvalue
is about `√(2/p)` (14% at 100 draws), and the ratio `λ²₁/Σλ²` can exceed 1;
deflated, the same case matched dense to four decimals.

### 6.3 Choosing q, and when the interval is computed  ·  *Presentation*

The references report the ratios and, with `eigen_diagno=1`, the q = 1 interval
for every component. This package applies KSS's threshold rule (q counts leading
ratios ≥ 1/10), warns when q ≥ 1, and computes the q = 1 interval only for those
components. It also flags ratios within Monte Carlo error of the threshold.

### 6.4 The first-stage F  ·  *Estimator*

`b̂₁²/V̂[b̂₁]` in both. The references use the `(Pᵢᵢ, Bᵢᵢ)` smoother of 5.3 for
`V̂[b̂₁]`; the diagnostic here smooths on the leverage alone (it has not paid for
`Bᵢᵢ`). The interval itself (6.6) uses the full smoother.

### 6.5 The center `θ̂₁`  ·  *Estimator; follows the paper*

KSS define `θ̂₁ = θ̂ − λ₁(b̂₁² − V̂[b̂₁])` with `V̂[b̂₁] = Σ w²ᵢ₁σ̂²ᵢ`, the raw
leave-out values. The references substitute the smoothed `σ̃²`
(`theta_1=theta-(lambda_1/NT)*(b_1^2-COV_R1(1,1))`). This package follows the
paper, which makes `θ̂₁ = ỹ'C₂ỹ` exactly — the identity its implementation is
tested against.

### 6.6 The covariance `Σ₁` of `(b̂₁, θ̂₁)`  ·  *Estimator*

The same formula, with the trace term estimated as in 5.2. This package floors
the smoothed `σ̃²` at zero, which makes `Cov² ≤ V[b̂₁]·(first term of V[θ̂₁])`
hold by Cauchy–Schwarz, and when subtracting the trace term leaves `Σ₁`
indefinite falls back to the first term alone (upward biased; KSS Remark 9).
The references have no guard.

### 6.7 The critical value  ·  *Numerical*

The references take the nearest grid point in a table of simulated critical
values at a fixed 95% level (`tabulation_10K.mat`; xhdfe embeds the same table).
For q = 1 the critical value has an exact one-dimensional integral form, which
this package evaluates. It agrees with simulation to Monte Carlo error at every
curvature tested and hits both analytic limits (χ²(1) and χ²(2) quantiles) to
five decimals. It works at any confidence level.

### 6.8 Computing the interval  ·  *Reference issue*

The references solve a closed-form quartic (`AM_CI.m`; xhdfe's port is literal).
It takes `γ = √γ²`, so `γ ≥ 0` whatever the sign of `λ₁`, while `λ₁` enters the
endpoints with its sign. For `λ₁ > 0` it agrees with this package to 3e-15. For
**`λ₁ < 0` — the usual case for the covariance** — its upper endpoint can fall
inside the set it is meant to bound: against the definition (extremes of
`λ₁b² + t` over the filled ellipse, on a grid), on eight random configurations
its upper endpoint was low by up to 5% of the interval's width, e.g. 0.9594
where the definition gives 1.4230. Its lower endpoints were correct. This package
computes the extremes on the ellipse's boundary directly and matches the
definition for either sign.

### 6.9 Weak identification when leaving out a match  ·  *Partly implemented*

The diagnostic, meaning the eigenvalues, the ratios, q and F, runs on the
collapsed fit unchanged, since `S` and `Q` are the person-year ones.

- *The q = 1 interval for var(psi)* carries over exactly. A stayer's outcome
  moves neither ψ̂ nor the weak direction (`x̄₁ = 0` on stayers' matches), so the
  deflated kernel keeps a zero diagonal with `b₂ = 0` there.
- *The q = 1 interval for the covariance* does not. The weak direction loads on
  stayers' matches, the deflated kernel's diagonal there is `−λ₁x̄₁²`, and
  canceling it takes person-year terms (the within-match residuals of
  `sigma_for_stayers`) that the collapsed kernel cannot represent. The reference
  computes this interval with a person-year formulation. This package reports it
  as unavailable at match level, and the summary says so.

---

## 7. In the references, not here

| | LeaveOutTwoWay | VC-HDFE.jl | xhdfe | pytwoway |
|---|---|---|---|---|
| q = 1 interval for the covariance when leaving out a match (6.9) | ✓ (`COMPLETE`) | | ✓ | |
| exact leverages for small samples | ✓ | ✓ | ✓ | ✓ |
| `lincom`: regress effects on observables with KSS standard errors | ✓ | ✓ | ✓ | |
| homoskedastic (Andrews et al.) correction | ✓ (`COMPLETE`) | | | ✓ default |
| Lindeberg-condition diagnostic, `max x̄²ᵢ₁` | ✓ | | ✓ | |
| first-differenced two-way models | ✓ (`leave_out_FD`) | | | |
| q ≥ 2 interval | | | | |

The Lindeberg diagnostic would be cheap here: the row values `x̄ᵢ₁` are already
computed for the F statistic.

---

## 8. Summary

| # | item | class | this package |
|---|---|---|---|
| 1.1 | leave-out unit | Estimand | a match by default, as in `leave_out_KSS`, VarianceComponentsHDFE.jl and xhdfe; an observation optional |
| 1.2, 1.3 | pruning, single-observation workers | agrees | iterated pruning, singles dropped; matches xhdfe's sample row for row |
| 3.2 | stayers' `σ̂²` | agrees with LeaveOutTwoWay | its rules at both levels; pytwoway's firm-mean rule optional |
| 4.1 | divisor of the moments | agrees with `leave_out_KSS` | `n − 1` throughout |
| 2.2 | analytic weights | Estimand | beyond the MATLAB, Julia and xhdfe references; shared with pytwoway |
| 3.4 | match-level centering | Estimator | the reference's by default; `centering="weighted"` a documented deviation, 1.7–3× less variable |
| 5.3 | streamed smoother for `σ̃²` | Estimator | cell means on an `n^(1/6)` × `n^(1/6)` quantile grid, interpolated, in place of lowess |
| 5.4 | what the smoother averages | agrees | alternative tested, not adopted |
| 5.7 | standard errors at match level | Estimator | reference: `leave_out_COMPLETE`'s match-level path (beta there), validated through xhdfe's port; default `se_variance="person_year"` as in the reference; differences: outcome centered and the package's smoother (the reference's raw outcome gives zero standard errors on log earnings), no user weights, no cov q = 1 interval; `"match"` offered beyond the reference for spell-correlated errors |
| 5.8 | outcome centered in the standard errors | Estimator | `√w(y − ȳ_w)` throughout, as in the point estimate; the reference's raw outcome carries its level |
| 6.2, 6.5–6.8 | weak-ID computation | Estimator / Reference issue | deflated trace, the paper's `θ̂₁`, guarded `Σ₁`, exact critical value, interval correct for `λ₁ < 0` |
| 6.9 | q = 1 interval for cov at match level | not implemented | reported as unavailable |

Everything else is a numerical route to the same quantity, validated against
dense computation in `prototypes/`.

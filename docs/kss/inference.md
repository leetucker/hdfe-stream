# Standard errors and weak identification

Pass `se=True` for standard errors and 95% intervals. With them comes KSS's
diagnostic for whether the normal interval is justified, and, for every
component where it is not, the interval that stays valid. This page explains
all three, and how well they cover in simulations.

## Standard errors

```python
lo = leave_out_kss(fml, "data/*.parquet", workdir="scratch", se=True)
print(lo.summary())          # adds an se column and a 95% interval
lo.se.se["var(psi)"]         # and the pieces it came from
```

The estimator is a quadratic form `ỹ'Cỹ` whose kernel has a **zero diagonal**:

```
C = B − ½(diag(b)·M + M·diag(b)),   b_i = B_ii/(1 − P_ii)
```

`C_ii = B_ii − M_ii·(B_ii/M_ii) = 0` identically. That single fact gives
unbiasedness in one line, makes the variance

```
V[θ̂] = 4 Σ_i σ²_i (Cμ)_i² + 2 tr(CΩCΩ)
```

with **no appeal to normality** (the third- and fourth-moment terms all carry a
factor of `C_ii`), and gives an exact test of the implementation: `ỹ'Cỹ` must
reproduce `plug-in − bias` to floating point, which it does to 1e-14.

The estimator substitutes `σ̃²` for `σ²` and `ỹ` for `μ`. Its first term
overshoots by exactly `4 tr(CΩCΩ)`, so subtracting `2 tr` leaves it unbiased —
and means the first term **alone is conservative**. That one is reported too, as
`se_conservative`, and `se_trace=False` computes only it, saving about three
solves per draw. Leaving out an observation, the trace term was 1–9% of the
first, so it moved the standard error by under 5%. Leaving out a match there are
fewer rows and it can be much larger: about half the first term for the
covariance on a simulated panel of 376 matches with the weighted centering. So
`se_trace=False` is a real loss of precision there, though still conservative.

**Measured coverage leaving out an observation**, design and true effects held
fixed and errors redrawn (800 replications, so about ±0.8 points), with fourfold
heteroskedasticity. Coverage when leaving out a match is in the
[next section](#standard-errors-when-leaving-out-a-match).

| n | var(psi) | var(alpha) | cov |
|---|---|---|---|
| 1,540 | 94.9% | 95.2% | 93.4% |
| 5,120 | 95.0% | 95.0% | 94.6% |

and `√V̂` matches the true sampling standard deviation within 3%. The one gap
from 95%, the covariance on the smaller panel, has a specific cause: KSS's
Theorem 2 needs `λ²₁/Σλ² = o(1)` for the eigenvalues of `Ã = S^-½QS^-½`, and on
panels this small that ratio is 0.09–0.30 rather than small; ranking the three
components by it reproduces the coverage ranking. Panels from roughly 800
workers up satisfy it, and the output reports it — see
[below](#is-the-normal-interval-justified).

**One deviation from the paper, which the reference implementation also makes.**
KSS's `σ̃²` is cross-fit from two predictions built on disjoint information — in
a two-way model, two *edge-disjoint paths* through the worker–firm network, found
by Dijkstra — and their `V̂` then classifies every `(i, ℓ)` pair by the sparsity
pattern of those paths. A per-observation graph search and an O(n²) pair loop are
both out of reach for streaming, and neither is in Saggio's MATLAB reference
either: it smooths the ordinary leave-out `σ̂²` against `(P_ii, B_ii)` and
estimates the trace by simulation, which is what happens here. Their Lemma 5
covers this case — the bias of `V̂` is non-negative, so inference stays valid and
becomes conservative rather than anticonservative.

That smoothing is **required, not cosmetic**: with raw `σ̂²` the estimated
`tr(CΩ̂CΩ̂)` comes out *negative* (it is `Σ C²_iℓ σ²_i σ²_ℓ`, which cannot be),
because raw `σ̂²_i` is often negative — and coverage falls to 92%. See
[prototypes/SE_NOTES.md](https://github.com/leetucker/hdfe-stream/blob/main/prototypes/SE_NOTES.md) for the derivation, the
reference comparison and the full Monte Carlo.

Weights need nothing extra when leaving out an observation: inference runs in
the same square-root-weight metric the point estimate uses. Leaving out a match,
the default variance source has no weights in the reference, so a weighted fit
needs `se_variance="match"` (next section).

## Standard errors when leaving out a match

**Reference implementation:** LeaveOutTwoWay's older entry point,
`leave_out_COMPLETE`, with `leave_out_level='matches'`. That option is not its
default, and its own documentation calls it "currently in beta mode and needs
further testing". It is the only implementation there is: Saggio's current
`leave_out_KSS` reports no standard errors, and xhdfe's are a port of this path.
This package follows it except where listed below:
- the same kernel;
- the same conventions for stayers, whose single match has leverage one;
- no standard error for var(alpha);
- the same per-person-year error variance.

The reference's person-year block kernel turns out to be the collapsed
regression's own zero-diagonal kernel, so the same machinery serves both levels.
The prototype checks this to 10⁻¹³ and streaming to 10⁻¹⁰. A literal port of
xhdfe's code in the prototype reproduces xhdfe's numbers, and is kept as a
regression test.

Where this package differs from the reference:

- **The outcome is centered**, as the point estimate is. The reference uses the
  raw outcome, so on log earnings the variance estimate is dominated by the
  outcome's level and comes out negative. xhdfe then reports a standard error of
  **zero**, which it did in 84–99.6% of samples in the Monte Carlo below.
- **σ̃ uses the package's smoother.** The reference's fixed 1,000 × 1,000 grid
  leaves about one person-year per cell on panels below a million rows.
- **No user weights** with the default variance source, which has none in the
  reference.
- **No q = 1 interval for the covariance** (var(psi) has one).

`se_variance=` chooses how each match's error variance is estimated:

- `"person_year"` (default) is the reference's, and the only one it has. It
  pools the variation within the spell, which is efficient when errors are
  independent within a spell. It estimates only the diagonal of the spell's error
  covariance, so it assumes that independence.
- `"match"` goes beyond the reference. It uses the collapsed leave-match-out
  estimate, which stays right when errors are correlated within a spell, and it
  supports weights. It is noisier.

Coverage of the 95% interval, var(psi) / cov (500 replications per cell):

| panel | errors | `"person_year"` (default) | `"match"` | xhdfe, outcome demeaned first |
|---|---|---|---|---|
| 837 matches | independent | 95.8 / 93.6 | 96.0 / 91.8 | 92.0 / 89.2 |
| 837 matches | half the variance a spell shock | **86.2 / 82.0** | 95.0 / 92.2 | 77.4 / 68.6 |
| 406 matches | independent | 94.8 / 95.0 | 92.8 / 87.0 | 89.8 / 89.8 |
| 406 matches | half the variance a spell shock | **86.0 / 83.6** | 95.4 / 89.0 | 71.2 / 66.2 |

With independent errors the default is as good or better. If errors may be
correlated within a spell, use `"match"`: the default then understates the
standard error by 25–30%, and more data does not fix it. Details in
[methodological differences](differences.md) §5.7.

## Is the normal interval justified?

The interval above rests on KSS's Theorem 2 condition (ii): no single
eigen-direction may dominate the estimator's variance, `λ²₁/Σλ² = o(1)` for the
eigenvalues of `Ã = S^-½QS^-½`. A bottleneck in the mobility network — two
groups of firms joined by a handful of movers — breaks it: one linear
combination of the effects is then estimated badly, the estimator is no longer
approximately normal, and the interval undercovers.

With `se=True` the output says which case you are in, by KSS's own rule (§6.2):
`q` counts the leading eigenvalues with `λ²/Σλ² ≥ 1/10`, and only `q = 0`
justifies the normal interval. The diagnostic is the same at either leave-out
level. The outputs below are from `leave_out="observation"`, where all three
components get standard errors:

```
weak-identification diagnostic (KSS threshold 0.1 on lambda^2 / sum lambda^2)
component          lam1^2/sum lam2^2/sum lam3^2/sum    q        F  normal interval
----------------------------------------------------------------------------------
var(psi)               0.9826     0.0017     0.0017    1     6.26  NOT justified
var(alpha)             0.9022     0.0016     0.0015    1     0.50  NOT justified
cov(psi, alpha)        0.9919     0.0009     0.0009    1     2.59  NOT justified
```

That is two blocks of eight firms joined by six movers
(`simulate_bottleneck(n_bridge=6)`); a well-connected
3,000-worker panel reads 0.06 / 0.05 / 0.07, all `q = 0`. When `q ≥ 1` the
affected intervals are starred and a warning is raised; the point estimate is
unbiased either way. `F` is the first-stage F of the leading direction — how well
its nuisance parameter is identified. Pass `diagnose=False` to skip it, or
`diagnose=True` to run it without standard errors.

It costs little next to the standard errors — 9 s on a 452 s run over 1.3M rows
at the default budgets, about 2%, leaving out an observation — and it streams: the leading
eigenvalues come from Lanczos in the `S⁻` inner product (one solve per step, no
extra passes over the rows), and `Σλ²` from Hutchinson deflated by the Ritz
vectors Lanczos already found, so the Monte Carlo only has to estimate the small
remainder. That deflation is what makes the estimate accurate exactly where it
matters — on a bottleneck of two movers the plain estimator was off by up to 160%. Draws
stop as soon as no ratio is within two standard errors of 1/10, and a ratio still
that close when the budget runs out is marked `~`.

Every piece is checked against dense `eig(S⁻Q)` and `tr((S⁻Q)²)`: eigenvalues to
1e-13 at convergence, including a panel whose two leading eigenvalues are 2%
apart. See [prototypes/WEAKID_NOTES.md](https://github.com/leetucker/hdfe-stream/blob/main/prototypes/WEAKID_NOTES.md).

## When it is not: the q = 1 interval

For every component the diagnostic flags, `se=True` also reports KSS's
weak-identification interval (Theorem 3 with Andrews–Mikusheva critical
values), which stays valid there. Leaving out a match it is reported for
var(psi) only (see [below](#what-it-does-not-do-yet)); this output leaves out an
observation:

```
intervals valid under weak identification (KSS Theorem 3, q = 1, 95%)
component              leave-out                    interval   kappa   width vs normal
--------------------------------------------------------------------------------------
var(psi)                0.038995        [0.031393, 0.057348]    0.99             1.24x
var(alpha)              0.146609        [0.140923, 0.157213]    0.65             1.36x
cov(psi, alpha)        -0.000346       [-0.014715, 0.002551]    2.40             1.47x
```

The idea: split the estimate into the badly identified direction and the rest,
`θ̂ = λ₁(b̂₁² − V̂[b̂₁]) + θ̂₁`. The pair `(b̂₁, θ̂₁)` is jointly normal even when
`θ̂` is not, so take its confidence ellipse and map it through `λ₁b² + θ₁`. The
image is the interval — which is why it is **not symmetric** about the estimate:
the weak direction enters squared. Mapping a 2-D ellipse would ordinarily need
the χ²(2) critical value; Andrews and Mikusheva show the map's curvature `κ`
lets it come down toward χ²(1), so the interval is no wider than it must be.

How it is computed, and why that is exact rather than approximate:

- Deflating `B` by the leading direction gives `θ̂₁ = ỹ'C₂ỹ` **exactly**, again
  a quadratic form with a zero diagonal, so the covariance of `(b̂₁, θ̂₁)` follows
  from the same algebra as the standard errors, with no normality assumption.
  It streams the same way, reusing the eigenvector the diagnostic already found.
- The critical value has an exact one-dimensional integral form for q = 1, so
  there is no simulated lookup table (the reference ships one; its repository
  carries no license).
- The interval matches Saggio's closed-form quartic to 3e-15 and its own
  definition — extremes over the filled ellipse — to grid resolution.

**Measured coverage leaving out an observation**, with the design and true
effects fixed and errors redrawn (two blocks of eight firms joined by six
movers, λ₁²/Σλ² = 0.90–0.99):

| δ (block gap) | var(psi): normal → **q = 1** | var(alpha) | cov |
|---|---|---|---|
| 0.00 | 89.5% → **95.8%** | 94.2% → **97.0%** | 73.5% → **97.8%** |
| 0.25 | 91.7% → **95.0%** | 92.8% → **97.2%** | 85.3% → **95.8%** |
| 0.50 | 93.5% → **96.5%** | 94.3% → **97.3%** | 91.3% → **97.7%** |
| 1.00 | 93.0% → **95.7%** | 93.3% → **97.2%** | 92.5% → **96.0%** |

(600 replications per cell, so about ±0.9 points. δ shifts the second block's
firm effects, which moves the weak direction's true value away from zero.)
With sixteen bridging movers the design is less weak (0.57–0.94) and the normal
interval recovers to 88–95%, while the q = 1 interval stays at 93.8–97.3%; its
lowest cells are where the weak direction is well identified and even an oracle
using the true covariance only reaches about 95%.

The normal interval undercovers — down to 73.5% — and the q = 1 interval reaches
nominal coverage everywhere, at the cost of a wider interval (24–47% wider on the
panel shown above). With only two bridging movers the weak
direction rests on about eight rows, outside the theorem's conditions; there the
normal interval falls to 50% and the q = 1 interval is conservative, at 97–99.5%.
On such a design the normal standard error can also be undefined, because the
variance estimate comes out non-positive: with `simulate_bottleneck(n_bridge=2)`
and `seed=0` that happens for var(psi) and the covariance. The q = 1 interval is
still reported for them. `simulate_bottleneck` builds these designs if you want
to see it yourself.

`weak_interval=False` skips it; `confidence=` sets the level of both intervals.
It costs roughly one more set of standard-error draws per flagged component and
nothing when none is flagged.

## What it does not do yet

- **var(alpha)'s standard error when leaving out a match,** following the
  reference, and **the covariance's q = 1 interval** there. Stayers load on its
  weak direction, which the collapsed kernel cannot represent. Both are
  available with `leave_out="observation"`.
- **q ≥ 2.** When two or more directions are weak, the q = 1 interval accounts
  for only the leading one and may undercover; the output says so. KSS's q > 1
  interval needs a (q+1)-dimensional quadratic program and the full curvature
  maximization.
- **Several outcomes at once.** Fit them separately.


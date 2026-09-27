# Standard errors for the leave-out components

What the paper prescribes, what its own reference implementation does instead,
and what is implemented here. Written while building `component_variances()`.

## The estimator is a quadratic form with a zero diagonal

Work in the square-root-weight metric: `X̃ = W^½A`, `ỹ = W^½(y − ȳ_w)`, errors of
variance `σ²ᵢ = wᵢVar(εᵢ)` — which is exactly what `sigma2_leave_one_out`
returns, so the weighted case needs no separate algebra. Then with
`B = X̃S⁻QS⁻X̃'`, `P = X̃S⁻X̃'`, `M = I − P`, `bᵢ = Bᵢᵢ/(1 − Pᵢᵢ)`:

```
C = B − ½(diag(b)M + M diag(b))
```

Two facts do all the work.

**`Cᵢᵢ = Bᵢᵢ − Mᵢᵢ·(Bᵢᵢ/Mᵢᵢ) = 0`, identically.** So

```
ỹ'Cỹ = ỹ'Bỹ − Σᵢ bᵢ ỹᵢ(Mỹ)ᵢ = plug-in − Σᵢ Bᵢᵢ σ̂²ᵢ
```

which is the KSS estimator itself. That is an *exact identity*, and it is how the
streaming implementation is tested — no Monte Carlo needed to check the
operator. It also gives unbiasedness in one line: `E[ỹ'Cỹ] = μ'Cμ = θ`, with no
trace term to correct, because the diagonal vanishes.

**The variance needs no normality.** For independent errors,

```
V[θ̂] = 4 Σᵢ σ²ᵢ (Cμ)ᵢ² + 2 tr(CΩCΩ),    Ω = diag(σ²)
```

The third- and fourth-moment terms all carry a factor of `Cᵢᵢ` and drop out.
This is the variance in KSS Theorem 2, whose second term they write as
`2 Σᵢ Σ_{ℓ≠i} C²ᵢℓ σ²ᵢσ²ℓ` — equal to the trace precisely because the diagonal
is zero.

## The estimator of V, and why its correction has the sign it does

```
V̂ = 4 Σᵢ σ̃²ᵢ (Cỹ)ᵢ² − 2 tr(CΩ̃CΩ̃)
```

`E[(Cỹ)ᵢ²] = (Cμ)ᵢ² + Σ_ℓ C²ᵢℓσ²ℓ`, so the first term overshoots by exactly
`4 tr(CΩCΩ)`. Since `V` carries `+2 tr(CΩCΩ)`, subtracting `2 tr` leaves the
whole thing unbiased. The same arithmetic says the first term **alone** is
conservative, not anticonservative — which is why it is worth reporting on its
own (`se_conservative`).

## What KSS prescribe for σ̃², and why it is not implementable here

Their `σ̃²ᵢ` is cross-fit: `(yᵢ − x̂ᵢ'β̂₋ᵢ,₁)(yᵢ − x̂ᵢ'β̂₋ᵢ,₂)` from two
predictions built on disjoint information, `Pᵢℓ,₁Pᵢℓ,₂ = 0`. In a two-way model
those are **two edge-disjoint paths through the worker–firm network**, existing
by Menger's theorem when the design keeps full rank after dropping any two
observations, and found in their application by **Dijkstra's algorithm**
(Appendix B.4). Their `V̂` then classifies every `(i, ℓ)` pair by the sparsity
pattern of those predictions into four cases, three unbiased and one
conservative.

Neither part survives streaming: the first is a per-observation graph search, the
second is O(n²). Lemma 5 part 2 is the escape hatch — the bias of `V̂` is
non-negative, so inference stays valid, just conservative, when the pair
conditions fail.

**Saggio's own reference implementation does not do it either.** In
`leave_out_estimation_two_way.m`:

```matlab
inner = (W_to_use).*(W_to_use).*sigma_predict;
V = (1/NT^2)*(4*SUM - left_part);
```

where `W_to_use = B*y − 0.5(Lambda_B*eta_h + xi_hat)` — which is exactly the `C`
above, confirmed term by term against an independent derivation — and
`left_part = var(aux_SIM)` over 1000 Gaussian draws of `v'Cv` with `v = Ω̃^½z`.
For Gaussian `z`, `Var(v'Cv) = 2 tr(CΩCΩ)`: their second term, by simulation.

And `sigma_predict` is not the cross-fit at all. `llr_fit.m` runs a **lowess
smooth of `yᵢη̂ᵢ` on `(Pᵢᵢ, Bᵢᵢ)`** and uses the fitted values.

## Why smoothing σ̂² is required, not a refinement

Monte Carlo, design and true effects fixed, errors redrawn, heteroskedastic
(`sd` varying about fourfold), so `θ = β'Qβ` is a fixed target:

| n | component | sd(θ̂) | √V_true | √V̂ raw | cover raw | √V̂ smooth | cover smooth |
|---|---|---|---|---|---|---|---|
| 1540 | var(psi) | 0.00411 | 0.00404 | 0.00384 | 92.3% | 0.00400 | 93.9% |
| 1540 | var(alpha) | 0.00428 | 0.00422 | 0.00383 | 91.9% | 0.00431 | 94.7% |
| 1540 | cov | 0.00159 | 0.00156 | 0.00153 | 93.8% | 0.00153 | 93.1% |
| 5120 | var(psi) | 0.00341 | 0.00342 | 0.00325 | 93.4% | 0.00334 | 94.6% |
| 5120 | var(alpha) | 0.00263 | 0.00260 | 0.00259 | 94.2% | 0.00261 | 94.9% |
| 5120 | cov | 0.00133 | 0.00135 | 0.00150 | 95.1% | 0.00134 | 94.4% |

Three things to read off it.

1. **The variance formula is right.** `√V_true` tracks `sd(θ̂)` to within 2%
   everywhere. That is the formula validated independently of any estimator of it.
2. **Coverage improves with n** — 93.9/94.7/93.1 at n=1540 to 94.6/94.9/94.4 at
   n=5120. The residual gap has a specific cause, not a vague one: KSS Theorem 2
   condition (ii), `λ²₁/Σλ² = o(1)`, is *violated* on these panels, and ranking
   the components by that ratio reproduces the coverage ranking exactly on both
   panels. See [WEAKID_NOTES.md](WEAKID_NOTES.md). The one component that
   satisfies the condition, `var(alpha)` at n=5120, covers at 94.9%.
3. **Raw σ̂² is incoherent, not merely noisy.** At n=5120 the estimated
   `tr(CΩ̂CΩ̂)` came out **negative** — −1.6%, −7.8%, −34.2% of the first term.
   That quantity is `Σ C²ᵢℓσ²ᵢσ²ℓ` and cannot be negative; it goes negative only
   because raw `σ̂²ᵢ` is frequently negative. Subtracting a negative correction
   then *inflates* `V̂`, and the conservative variant covers **worse** (90.2% for
   the covariance) than the full one. Smoothing makes all three positive.

This is why the implementation smooths, and why it is not offered as an option to
turn off.

### How much to smooth: the reference's bandwidth, not a fixed grid

The table above used a fixed 12×12 grid on `(Pᵢᵢ, Bᵢᵢ)`. Building the weak-ID
interval showed that is wrong at both ends. On a 1,600-row panel it is ~11 rows a
cell, and `V[b̂₁]` — which rests on the few rows a weak direction loads on —
moved by up to 2× between two implementations of the same smoother (dense bins on
exact `Pᵢᵢ, Bᵢᵢ`, streaming on their random-projection estimates).

Saggio's `llr_fit` uses lowess with span `h = n^(−1/3)`, so each local fit sees
`n^(2/3)` points. A grid whose cells hold that many has **`n^(1/6)` bins per
axis** — 3 at 1,600 rows, 10 at 1.3M, 22 at 100M — and `n^(1/3)` bins for a
one-dimensional smoother. That is now the default (`smooth_bins=None`). It
brought the two implementations within 1–9% of each other, and it holds or
improves the coverage above:

| n | var(psi) | var(alpha) | cov |
|---|---|---|---|
| 1,540, fixed 12 bins | 93.9% | 94.7% | 93.1% |
| 1,540, `n^(1/6)` | **95.0%** | **95.1%** | 92.6% |
| 5,120, fixed 12 bins | 94.6% | 94.9% | 94.4% |
| 5,120, `n^(1/6)` | 94.5% | 94.9% | 94.6% |
| 1,540, final | **94.9%** | **95.2%** | 93.4% |
| 5,120, final | **95.0%** | **95.0%** | **94.6%** |

"Final" is the shipped configuration: `n^(1/6)` bins interpolated bilinearly,
every row's own `σ̂²` (stayers included), moments divided by `n − 1`.
(800 replications each; the remaining gap for `cov` at n=1,540 is the weak
identification [WEAKID_NOTES.md](WEAKID_NOTES.md) explains — its eigenvalue ratio
there is 0.30.)

## How much the trace term is worth

`2 tr(CΩCΩ)` is 1.3–2.5% of the first term for the variances and 6.7–9.4% for
the covariance, so it moves the standard error by under 5% and coverage by well
under a point. It costs roughly three solves per draw. `se_trace=False` skips it
and returns the conservative error, which is a reasonable trade on a very large
panel.

## Stayers

The identity `ỹ'Cỹ = plug-in − Σ Bᵢᵢσ̂²ᵢ` holds for the algebraic `σ̂²` on every
row. The point estimate used to impute stayers' `σ̂²` from the movers at the same
level of psi (pytwoway's rule), which made it a slightly different estimator
from the one `V` describes; the measured gap was under 0.1 standard errors. It
now follows LeaveOutTwoWay at observation level — every row its own `σ̂²` — so
the two describe the same estimator. They are still reported side by side
(`se.theta` against `leave_out`), which doubles as an end-to-end check, since
they share almost no code.

## Leaving out a match

The reference implementation is `leave_out_COMPLETE` with
`leave_out_level='matches'`, an option its documentation marks as beta. Its
person-year block kernel is the collapsed regression's own zero-diagonal
kernel. Conjugating by `diag(√T)` maps one to the other, so
the code above serves both levels. Stayers get `b = 0`, which is exact for
var(psi) and cov, and var(alpha) is not reported. For the reference's centering
the estimator is a rank-two update of `C`.

`xhdfe_match_se` is a literal port of xhdfe's code and reproduces it: `theta_c`
to 10⁻¹³, the standard error to within its simulation error, and its zeros. With
log earnings the reference's variance estimate is negative and truncated to
zero, because it uses the raw outcome with an essentially unsmoothed σ̃.
`kss_match_se` is the package's version. It follows the reference by default
(`se_variance="person_year"`), except for the differences listed in
`docs/kss_methodological_differences.md` §5.7, which also has the coverage study
and the `"match"` option that goes beyond the reference.

## Not implemented

Weak-identification inference — KSS Theorem 3 and the Andrews–Mikusheva
curvature critical values their `eigen_diagno` path computes. It needs the
largest eigenvalue and eigenvector of `Ã`, i.e. a Lanczos iteration driven by
`apply_inverse`, which is feasible here but a separate piece of work.

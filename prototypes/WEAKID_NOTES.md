# Weak-identification inference

KSS Section 5 (Theorem 3) and Section 6 (Andrews–Mikusheva critical values), and
what it takes to add them here.

**Status:** both stages are **built**. Stage 1, the diagnostic, is
`weak_id_diagnostics()` in `hdfe_stream/leaveout_weakid.py`; stage 2, the q = 1
interval, is `_weak_moments()` in `leaveout_se.py` plus the pure numerics in
`am_interval.py`. Both run by default with `se=True`, the interval only for
components the diagnostic flags. What building each taught is at the end
([stage 1](#what-building-stage-1-found), [stage 2](#what-building-stage-2-found));
the plan below is as written beforehand, kept for the record.

## Why it matters, measured rather than assumed

The normal interval we ship rests on Theorem 2 condition (ii),
`λ²₁ / Σλ² = o(1)`, where the `λ` are eigenvalues of `Ã = S^-½ Q S^-½`. KSS give
a threshold rule (§6.2): take `q` with `λ²_q/Σλ² ≥ 1/10` and
`λ²_{q+1}/Σλ² < 1/10`, and **`q = 0` when `λ²₁/Σλ² < 1/10`** — only then is the
normal interval justified.

Computed exactly on the panels this library is tested on:

| workers | n | var(psi) | var(alpha) | cov |
|---|---|---|---|---|
| 180 | 1,540 | 0.260 | 0.191 | 0.299 |
| 400 | 3,391 | 0.149 | 0.122 | 0.167 |
| 600 | 5,120 | 0.105 | 0.086 | 0.118 |
| 799 | 6,809 | 0.069 | 0.056 | 0.078 |
| 1,590 | 13,495 | 0.045 | 0.034 | 0.052 |
| 2,998 | 25,386 | 0.061 | 0.050 | 0.072 |

The ratio falls with `n` but **not monotonically** — it depends on the shape of
the mobility network, not its size. Panels from roughly 800 workers up are `q=0`;
the small test fixtures are `q=1` or `q=2`.

**This explains the coverage gap measured in [SE_NOTES.md](SE_NOTES.md).** Rank
the three components by eigenvalue ratio and by coverage and the orders match
exactly, on both panels:

| panel | component | λ²₁/Σλ² | q | coverage |
|---|---|---|---|---|
| 5,120 | var(alpha) | 0.086 | 0 | 94.9% |
| 5,120 | var(psi) | 0.105 | 1 | 94.6% |
| 5,120 | cov | 0.118 | 1 | 94.4% |
| 1,540 | var(alpha) | 0.191 | 2 | 94.7% |
| 1,540 | var(psi) | 0.260 | 2 | 93.9% |
| 1,540 | cov | 0.299 | 2 | 93.1% |

So the residual undercoverage is not vague "asymptotic approximation" — it is
condition (ii) failing, which is precisely what this machinery repairs. The
component with the smallest ratio (`var(alpha)`, `q=0`) covers at 94.9%.

A deliberate bottleneck — two blocks of four firms joined by a single mover —
reaches ratios of **0.75 to 0.91**, confirming a genuinely weakly identified
design is constructible for validation.

## What q=1 needs, piece by piece

Everything below is checked against both the paper and Saggio's implementation.

### 1. `λ₁` and the generalized eigenvector `u₁`  (new, cheap)

The eigenvalues of `Ã` are those of `S⁻Q`, so the top pair solves
`Q u = λ S u`. `S⁻Q` is self-adjoint in the `S` inner product, so Lanczos works
with:

- operator `v ↦ S⁻(Q v)` — **one `apply_inverse` per iteration**;
- Rayleigh quotient `λ = v'Qv / v'Sv`, where `v'Qv` is coefficient-space
  arithmetic (for the variances `Q` is diagonal-minus-rank-one at the level,
  `[diag(W_ℓ) − W W'/T]/T`) and `v'Sv = Σᵢ wᵢ(xᵢ'v)²` is one readout pass.

Roughly 30–50 iterations, so **~50 solves one-off** — cheaper than the 250-draw
leverage pass. Lanczos also hands back `λ₂`, `λ₃` for free, which the `q` rule
needs.

Worth noting: the reference (`eigAux.m`) cannot do this. It builds an augmented
sparse pair `(Adot, Sxxdot)` to absorb the centering and calls MATLAB `eigs`,
which needs `S` explicitly. Applying the operator directly is *simpler* here, not
harder — one of the few places where streaming is the easier design.

### 2. `Σλ² = tr(Ã²) = tr(S⁻QS⁻Q)`  (new, cheap)

Hutchinson: draw `z` in coefficient space, form `S⁻Q S⁻Q z`, accumulate
`z'(·)`. **2 solves per draw**, the same cost as the existing bias term. The
reference does this too (`trace_Atilde_sqr.m`). Verified numerically: summing
`λ²` over the exact eigenvalues reproduces `tr(Ã²)` to floating point, so the
Hutchinson target is the right object.

### 3. `x1bar`, `b̂₁`, `V̂[b̂₁]`  (one pass)

`x1barᵢ = wᵢ xᵢ'u₁` — the row-level readout of the eigenvector, normalized so
`Σᵢ x1barᵢ² = 1` (i.e. `u₁'Su₁ = 1`). Then `b̂₁ = Σᵢ x1barᵢ yᵢ` and
`V̂[b̂₁] = Σᵢ x1barᵢ² σ̃²ᵢ`, both in the same pass. `θ̂₁ = θ̂ − λ₁(b̂₁² − V̂[b̂₁])`.

### 4. `Σ₁`, the 2×2 covariance of `(b̂₁, θ̂₁)`  (a rank-one change to existing code)

This is the part already paid for. `Σ₁` needs the same `C` operator with `B`
**deflated by one rank**: `B₂ = B − λ₁ x1bar x1bar'`, so
`B₂v = Bv − λ₁ x1bar (x1bar'v)` — two extra scalars per pass inside
`_apply_c`. Then

```
Σ₁[1,1] = Σ x1bar² σ̃²
Σ₁[1,2] = 2 Σ x1bar σ̃² (C₂y)
Σ₁[2,2] = 4 Σ (C₂y)² σ̃² − 2 tr(C₂ Ω̃ C₂ Ω̃)
```

all three reusing `_apply_c` and the trace loop unchanged. Confirmed against
`leave_out_estimation_two_way.m`, where `Lambda_B2 = Lambda_B − diag(λ₁·x1bar²)`
plus an explicit `−λ₁(v'x1bar)²` in the simulation — the same rank-one deflation.

### 5. Curvature  (closed form)

`κ̂₁ = 2|λ₁| V̂[b̂₁] / (V̂[θ̂₁]^½ (1−ρ̂²)^½)`, `ρ̂` the correlation in `Σ₁`
(Appendix C.6). No maximization needed at `q=1`.

### 6. The critical value `z_{α,κ}`  (no external dependency)

Appendix C.6.1 defines it completely: the `(1−α)` quantile of

```
ρ(χ_q, χ₁, κ) = √(χ²_q + (χ₁ + 1/κ)²) − 1/κ
```

for independent `χ²_q`, `χ²₁`. Simulated at 2M draws it reproduces both known
limits — `z² = 3.836` at `κ=0` against `χ²₁(0.95) = 3.8415`, and `5.973` at
`κ=100` against `χ²₂(0.95) = 5.9915`:

| κ | 0 | 0.5 | 1 | 2 | 4 | 8 | 20 | 100 |
|---|---|---|---|---|---|---|---|---|
| z² | 3.836 | 4.548 | 4.941 | 5.325 | 5.608 | 5.784 | 5.904 | 5.973 |

**This matters practically.** The reference ships a precomputed
`tabulation_10K.mat`, and that repository carries **no license** — vendoring it
would be legally murky. Generating our own table is a dozen lines and removes the
question.

### 7. The interval  (quartic root-find)

`AM_CI.m` is 37 lines: build a quartic in `b`, take its real roots, evaluate a
closed form at each, and take the min and max. `np.roots` replaces MATLAB
`roots` directly. Appendix C.6.2 has the derivation.

## Cost

| stage | added cost |
|---|---|
| diagnostic (`λ₁`, `λ₂`, `Σλ²`, F-statistic) | ~50 solves one-off + 2/draw |
| the `q=1` interval | ~1.3–1.5× the current `se=True` |
| `q ≥ 2` | quadratic programming, see below |

## Recommended scope

**Stage 1 — the diagnostic. Worth doing on its own.** `λ²₁/Σλ²`, `λ²₂/Σλ²`, the
implied `q`, and the first-stage F `b̂₁²/V̂[b̂₁]`, with a warning when `q ≥ 1`.
This closes a real gap: we currently report a normal interval with no way for a
user to know whether it is justified, and the panels in our own test suite turn
out not to satisfy the condition. It is also the cheapest piece and needs no new
inference theory — only the eigensolver and one Hutchinson trace.

**Stage 2 — the `q=1` Andrews–Mikusheva interval.** Mostly a rank-one deflation
of machinery that exists and is validated, plus self-contained numerics.

**Stage 3 — `q ≥ 2`. Defer, possibly indefinitely.** Needs quadratic programming
over `q+1` unknowns and the full curvature maximization over `u`. Our smallest
fixtures would need it; data at the scale this library targets will not. KSS's own
suggestion — report the union over two consecutive `q` — is a cheaper partial
answer.

## Validation plan

The discipline used for the point estimate and the standard errors applies here
too, and it is the main risk to schedule rather than to feasibility.

1. `λ₁`, `u₁`, `Σλ²` streaming against dense `eigvals(S⁻Q)` on a small panel.
2. `Σ₁` streaming against a dense build of the deflated kernel.
3. `z_{α,κ}` against its two analytic limits (done above) and against the
   reference's tabulation at a few `κ` if a copy can be consulted.
4. **Monte Carlo coverage in the bottleneck design**, where the ratio is 0.75–0.91:
   show the normal interval undercovers and the AM interval does not. Without
   this the feature is unvalidated, and an interval that is wider but not
   demonstrably better is worse than none.

## Open risks

- **Lanczos convergence when `λ₁ ≈ λ₂`** — which is exactly the `q ≥ 2` case. The
  implementation must detect a small gap and say so rather than return a
  confident wrong `λ₁`. At 799 workers above, `λ₁/λ₂ = 1.02`.
- **`Σ₁` inherits the non-cross-fit `σ̃²`**, so the same conservative caveat as
  the `q=0` standard errors carries over (KSS Remark 9 covers this: a positive
  definite bias in `Σ̂_q` still gives valid, conservative inference).
- **`q` is assumed known.** The threshold rule is a heuristic KSS offer, not a
  consistent selector.


## What building stage 1 found

Three things the plan above did not anticipate, each caught by comparing against
dense `eig(S⁻Q)` rather than trusting a plausible-looking number.

### The Krylov space is bounded by the non-streamed dimension

A panel with 18 firms broke `var(psi)` outright: β fell from ~1e-3 to 2.1e-7 at
step 17 — exactly `n_psi − 1`, the rank of `Q_psi` — and the recurrence then
divided by noise and blew up geometrically. The fix generalizes: every
component's Krylov space is bounded by the *firm-side* dimension, however many
workers there are.

| component | bound |
|---|---|
| var(psi) | `rank Q_psi = n_psi − 1` |
| cov | `rank Q_cov ≤ 2(n_psi − 1)` |
| var(alpha) | `n_levels + n_cov + 1` — the streamed block's spectrum is one bulk eigenvalue `1/T` plus a firm-driven part of that rank, and one start vector sees each distinct value once |

That last one is visible in the numbers: var(psi) and var(alpha) share their
leading eigenvalues exactly (0.00703, 0.00445, …), and their `Σλ²` differ by the
bulk, `(n_workers − rank)/T²`.

So steps are capped at the bound, and when the basis a component would need fits
`scratch_mb` — which is precisely the small-space case where exhaustion is
reached — it is stored and every vector fully reorthogonalized. Large spaces
never approach exhaustion within `max_iter`; there the three-term recurrence
with Cullum–Willoughby ghost filtering is used, and it was checked on a panel
whose leading eigenvalues are 2% apart, run long enough to form 2–4 ghost copies
of λ₁: all merged correctly, eigenvalues to 1e-13.

### Full reorthogonalization needs a range projection

The first reorthogonalized runs exploded at step 40 while the three-term runs
were fine — the opposite of the usual story. Cause: rounding leaves a trace of
the constant in each vector. `Q_psi 1 = 0`, so that direction is outside
range(Q), and it is **invisible to the `S⁻` inner product** because the solve
drops null-space components before it starts; orthogonalization therefore can
never remove it. Worse, subtracting stored basis vectors *re-injects* every one
of their traces. Measured: the constant's component grew about 2.5× a step,
5e-14 → 5e-4 → 880 by step 44, with `X'Y − I` degrading in lockstep. Removing the
mean on each component's blocks every step holds it at 1e-16 and orthogonality
at 4e-13 through 60 steps.

### Plain Hutchinson fails exactly where the diagnostic matters

On the bottleneck design, Lanczos matched dense to the digit, yet the reported
ratios were 0.72 / 0.92 / 0.38 against a true 0.9997 / 0.9958 / 0.9999 — the
`Σλ²` estimate was off by up to 160%. Two compounding reasons:

- **Wrong metric.** Draws white in the coefficient basis estimate
  `tr((S⁻Q)²)` with variance governed by how far `S⁻Q`'s eigenvectors are from
  orthogonal *in that basis*, which is unbounded. Drawing `ζ = X̃'ξ` from
  row-space noise gives `E[ζζ'] = S`: white in the `S⁻` inner product, where
  `QS⁻` is self-adjoint and the estimate is a squared norm with variance at most
  `2Σλ⁴`.
- **No deflation.** Even in the right metric, one dominant eigenvalue means
  relative error `√(2/p)`. But `tr(A²) = Σ_ℓ ‖Ax_ℓ‖² + E⟨Aζ_def, Aζ_def⟩` holds
  exactly for *any* `S⁻`-orthonormal `x_ℓ`, and with the Ritz vectors Lanczos
  already found the first term is almost the whole answer, computed exactly.

Together: 0.9997 / 0.9958 / 0.9999, matching dense to four decimals — and faster,
since one reduction pass and one solve per block now serve all three components.

### The F statistic needs smoothed σ̂², like everything else

Section 3 above said raw `σ̂²` is fine for `V[b̂₁]` because it is linear. True
for bias, false in practice: on a well-connected 25,000-row panel the leading
direction loaded on few enough rows that `Σ x1bar²σ̂²` came out *negative* for
all three components. The reference uses its smoothed `sigma_predict` here too.
The diagnostic smooths against the leverage alone, which avoids paying for the
`B_ii` pass it otherwise has no need of.

### Cost

On a 1.3M-row panel: Lanczos converges in 30 steps at the default tolerance
(eigenvalues within 6e-5 of a fully converged run — two orders below the Monte
Carlo error in `Σλ²`), each step one three-column solve, run twice to rebuild the
Ritz vectors. The trace stops as soon as no ratio is within two standard errors
of 1/10 — 16 draws on that panel. At the default budgets the whole `se=True`
run went from 452 s to 461 s, about 2%; an early measurement at 32 SE draws made
it look like 75%, because the diagnostic's cost is fixed while the standard
errors' scales with their draws. The trace's width is held to `rhs_block` like
every other solve in the library, so no memory beyond what the standard errors
already use.


## What building stage 2 found

### θ̂₁ is exactly a zero-diagonal quadratic form

With `x̃₁ = X̃u₁` and `B = Σ_ℓ λ_ℓ x̃_ℓx̃_ℓ'`, deflate one rank,
`B₂ = B − λ₁x̃₁x̃₁'`, and build `C₂` from `B₂` as `C` is built from `B`. Then

```
ỹ'C₂ỹ = θ̂ − λ₁(b̂₁² − Σ x̃²ᵢ₁σ̂²ᵢ)
```

which is KSS's `θ̂₁`. Checked to 1e-15 dense and to 1e-9 streaming. The identity
is algebraic — it holds for *any* λ and direction — so the streaming test uses a
random `u` and arbitrary λs, testing the deflation with Lanczos nowhere
involved. Everything from the standard errors then carries over:
`V[θ̂₁] = 4Σσ²(C₂μ)² + 2tr(C₂ΩC₂Ω)`, `Cov(b̂₁, θ̂₁) = 2Σx̃ᵢσ²ᵢ(C₂μ)ᵢ`, no
normality assumption. Streaming, the deflation is one scalar per component
accumulated in the pass that already reads `v`, plus a `b₂` store.

### The critical value has an exact form for q = 1

`ρ ≤ z ⟺ a² + (b+c)² ≤ (z+c)²` for half-normals `a, b` and `c = 1/κ`, so
`P(ρ ≤ z) = ∫₀ᶻ 2φ(b)[2Φ(√((z−b)(z+b+2c))) − 1] db`, factored so nothing
cancels as κ → 0. It agrees with 4M-draw simulation to Monte Carlo error and hits
both limits to five decimals, so the reference's simulated lookup table — in an
unlicensed repository — is not needed.

### The interval matches the reference to machine precision

Computed as extremes of `λ₁b² + t` on the ellipse boundary (the map is linear in
`t`, so extremes are never interior): dense grid plus Brent. That matches
Saggio's closed-form quartic (`AM_CI.m`) to 3e-15 across random cases including
`b̂₁ = 0`, and the filled-ellipse definition to grid resolution — including
negative λ₁, where the quartic's `γ ≥ 0` is not meant to apply.

### The smoother's bandwidth was wrong, and only V[b̂₁] exposed it

`V[b̂₁] = Σx̃²σ̃²` rests on the few rows the weak direction loads on. With the
fixed 12×12 grid inherited from the standard errors (~11 rows a cell at 1,600
rows), dense and streaming disagreed on it by up to 2× and on `Cov` by up to 20×,
since the two place those few rows in different cells. The reference's lowess
bandwidth, span `n^(−1/3)`, corresponds to `n^(1/6)` bins per axis; adopting it
brought the two within 1–9% and did not hurt stage-0 coverage (see
[SE_NOTES.md](SE_NOTES.md)).

### Variance weights must be floored at zero

A bin mean of the raw leave-out `σ̂²` can be negative. With weights `s ≥ 0`,
Cauchy–Schwarz gives `Cov² ≤ V[b̂₁]·4Σs(C₂y)²`, so when subtracting the trace
term leaves the 2×2 indefinite, falling back to the first term alone (upward
biased, KSS Remark 9) is guaranteed to repair it. Without the floor that
guarantee fails, and it did, in the first Monte Carlo.

### Coverage — and where the theory stops

Oracle first: with the *true* `Σ₁`, the q = 1 interval covered at 97.7–99.3% on
every design tried, so the construction is right (and conservative, as an AM
interval at q = 1 is). The estimated `Σ₁` is unbiased on average — `V[b̂₁]`
estimated at 2.35–2.61e-2 against a true 2.59e-2 — so remaining shortfalls come
from its sampling noise.

With **two** bridging movers the weak direction loads on ~8 rows
(`max x̃² ≈ 0.12`), outside Theorem 3's `max w²ᵢ₁ = o(1)`. With the fixed 12×12
smoother the q = 1 interval reached only ~90% there, which I first put down to
the design being outside the theorem. It was mostly the smoother: at the
reference bandwidth the same design gives **97.2–99.5%** (600 replications,
final configuration), conservative, while the normal interval falls to 50–94%. The few rows `V[b̂₁]`
rests on had been sitting in cells of ~11 rows.

With **six** bridges (`max x̃² ≈ 0.04`, still λ₁²/Σλ² = 0.90–0.99) — inside the
theorem, and the design reported in docs/kss.md:

| δ | component | normal | **q = 1** | oracle Σ |
|---|---|---|---|---|
| 0.00 | var(psi) | 89.5% | **95.8%** | 97.3% |
| 0.00 | var(alpha) | 94.2% | **97.0%** | 97.2% |
| 0.00 | cov | 73.5% | **97.8%** | 98.5% |
| 0.25 | var(psi) | 91.7% | **95.0%** | 96.3% |
| 0.25 | var(alpha) | 92.8% | **97.2%** | 97.7% |
| 0.25 | cov | 85.3% | **95.8%** | 97.3% |
| 0.50 | var(psi) | 93.5% | **96.5%** | 97.5% |
| 0.50 | var(alpha) | 94.3% | **97.3%** | 97.2% |
| 0.50 | cov | 91.3% | **97.7%** | 98.3% |
| 1.00 | var(psi) | 93.0% | **95.7%** | 95.8% |
| 1.00 | var(alpha) | 93.3% | **97.2%** | 97.2% |
| 1.00 | cov | 92.5% | **96.0%** | 96.8% |

600 replications per cell (±0.9 points), rerun under the final configuration
(interpolated smoother, own `σ̂²` for stayers, `n − 1`); the theory-exact center `ỹ'C₂ỹ` gives
the same coverage as the reported one to within a point. Every q = 1 cell is at
or above 95%.

With **sixteen** bridges (`max x̃² ≈ 0.016`, λ₁²/Σλ² = 0.57–0.94) the normal
interval recovers to 88.2–95.0% and the q = 1 interval gives 93.8–97.3%. Its
lowest cells (93.8%, 94.5%) are at δ = 1, where the weak direction is well
identified and the oracle itself reaches only 95.0% — ordinary finite-sample
error rather than a failure of the construction.

One systematic feature: the estimated `V[b̂₁]` runs about 9% below the truth in
every cell (2.9e-2 against 3.2e-2). It rests on the σ² of the few rows the weak
direction loads on, and a smoother can recover their bin average but not their
individual values; in a fixed design those few rows happen to be noisier than
their bins. Coverage absorbs it here, but it is the reason the theorem's
`max w²ᵢ₁ = o(1)` condition matters in practice.

### Cost

The deflated trace is one more set of standard-error draws per flagged
component. Building it exposed that the trace called `_apply_c` once per
component, each call computing all three and discarding two; with per-component
inputs one call does the work, bit-identical and 2.46× faster, which more than
pays for the interval.

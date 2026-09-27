# The JLA algorithm, from the sources

Notes taken before implementing the Johnson–Lindenstrauss approximation, which
is the part that would let the KSS correction run out of core. Three sources,
and they do not all say the same thing:

- **KSS** — Kline, Saggio & Sølvsten, *Leave-out estimation of variance
  components*, Econometrica 88(5), 2020. Working paper:
  [arXiv:1806.01494](https://arxiv.org/pdf/1806.01494). §1.2 defines the JLA,
  §1.5 gives the asymptotic equivalence, Appendix B.1 the pruning algorithm,
  B.3 the implementation and the cost/accuracy evidence.
- **the note** — Kline, Saggio & Sølvsten, *Improved stochastic approximation of
  regression leverages for bias correction of variance components*, 21 January
  2021.
  [`doc/improved_JLA.pdf`](https://github.com/rsaggio87/LeaveOutTwoWay/blob/master/doc/improved_JLA.pdf)
  in Saggio's `LeaveOutTwoWay`. Supersedes the paper's non-linearity correction.
- **pytwoway** 0.3.21, `fe.py`. Implements the note's version, with two
  differences from it (below).

The formulas below were first read out of extracted PDF text and then checked
against the rendered pages, so the constants and signs are as printed. Where
pytwoway departs from them, the departure has been raced against the exact
answer rather than adjudicated by reading.

## The core, which is simple

With `S = X'X`, draw Rademacher vectors `q₁…q_p ∈ ℝⁿ` and form

```
P̂ᵢᵢ = (1/p) Σₛ (xᵢ' S⁻¹ X' qₛ)²
M̂ᵢᵢ = (1/p) Σₛ (q_{s,i} − xᵢ' S⁻¹ X' qₛ)²
```

Both are unbiased, and both come out of **one linear solve per Rademacher
vector** — the note is explicit that `M̂` is free given `P̂`. This is what my
earlier sketch got right: `q = P·r`, then `Pᵢᵢ ← qᵢ²` and `Mᵢᵢ ← (rᵢ − qᵢ)²`.

## Refinement 1: normalize, don't just average

`Pᵢᵢ + Mᵢᵢ = 1` identically, so imposing it on the estimates is free
information:

```
M̄ᵢᵢ = M̂ᵢᵢ / (M̂ᵢᵢ + P̂ᵢᵢ)        P̄ᵢᵢ = P̂ᵢᵢ / (M̂ᵢᵢ + P̂ᵢᵢ)
```

The note shows this is the feasible version of the variance-minimizing linear
combination of the two estimators. Three consequences worth having:

- estimates are guaranteed inside `[0, 1]`, so no nonsensical leverages;
- variance is strictly reduced except at the boundaries, so `p` can be smaller;
- it costs no extra solves.

It does introduce a shrinkage bias toward the middle of the support, of order
`1/p`, which the next refinement removes.

## Refinement 2: the non-linearity correction, and it changed

`σ̂²ᵢ` divides by `Mᵢᵢ`, so substituting an estimate is a non-linear operation
and biased even when the estimate is not. **There are two generations of the
correction**, and this is the thing I would have got wrong:

**KSS 2020 (§1.2)** — using the *raw* `P̂ᵢᵢ`:

```
σ̂²ᵢ,JLA = [ yᵢ(yᵢ − xᵢ'β̂) / (1 − P̂ᵢᵢ) ] · ( 1 − (1/p)·(3P̂³ᵢᵢ + P̂²ᵢᵢ)/(1 − P̂ᵢᵢ) )
```

**The note (2021)** — using the normalized `M̄ᵢᵢ`, and removing the whole
`O(1/p)` bias rather than part of it:

```
σ̂²ᵢ,JLA = [ yᵢ(yᵢ − xᵢ'β̂) / M̄ᵢᵢ ] · ( 1 − V̂ᵢ/M̄²ᵢᵢ + B̂ᵢ/M̄ᵢᵢ )
```

built from three second moments over the *same* draws:

```
m(P²ᵢᵢ)    = (1/p) Σₛ (xᵢ'S⁻¹X'qₛ)⁴
m(M²ᵢᵢ)    = (1/p) Σₛ (q_{s,i} − xᵢ'S⁻¹X'qₛ)⁴
m(Pᵢᵢ,Mᵢᵢ) = (1/p) Σₛ (xᵢ'S⁻¹X'qₛ)² (q_{s,i} − xᵢ'S⁻¹X'qₛ)²

V̂ᵢ = (1/p)( M̄²ᵢᵢ m(P²ᵢᵢ) + P̄²ᵢᵢ m(M²ᵢᵢ) − 2 P̄ᵢᵢ M̄ᵢᵢ m(Pᵢᵢ,Mᵢᵢ) )
B̂ᵢ = (1/p)( M̄ᵢᵢ m(P²ᵢᵢ) − P̄ᵢᵢ m(M²ᵢᵢ) + (M̄ᵢᵢ − P̄ᵢᵢ) m(Pᵢᵢ,Mᵢᵢ) )
```

The note claims this removes the entire `O(1/p)` bias, leaving `O(p⁻²)`, and
that there is therefore no bias of importance while `n/p⁴ = o(1)` — i.e. `p`
need only grow like `n^{1/4}`. For `n = 10⁷` that is about 56.

So the per-observation accumulators needed during the projection pass are five,
not one: `P̂`, `M̂`, and the three second moments. All are row-sized running sums,
which is the shape a streaming pass wants anyway.

### Two differences between pytwoway and the note — settled

Comparing `_estimate_approximate_leverages` against the formulas above, checked
against a rendered copy of page 3 of the note (not extracted text):

| | the note | pytwoway 0.3.21 |
|---|---|---|
| `V̂ᵢ` cross term | `− 2 P̄M̄ · m(P,M)` | `− 2·P̄·M̄·m(P,M)` — agrees |
| `B̂ᵢ` cross term | `+ (M̄ − P̄) · m(P,M)` | `+ 2(M̄ − P̄) · m(P,M)` |
| how `B̂ᵢ` enters | `1 − V̂/M̄² **+** B̂/M̄` | `1 − V̂/M̄² **−** B̂/M̄` |

There is a third difference elsewhere: the note's `σ̂²ᵢ` uses the raw
`yᵢ(yᵢ − xᵢ'β̂)`, while pytwoway (following Saggio's MATLAB) demeans the outcome.

**The note is right.** The correction exists to remove an `O(1/p)` bias in
`1/M̄ᵢᵢ`, so the versions can simply be raced against the exact `1/Mᵢᵢ` on a
panel small enough to compute it. Mean relative bias, 3000 replications:

| p | uncorrected | note (+B, 1×) | pytwoway (−B, 2×) | +B with 2× | −B with 1× |
|---:|---:|---:|---:|---:|---:|
| 4 | 0.0407 | **0.0099** | 0.0144 | 0.0264 | 0.0310 |
| 8 | 0.0155 | **0.0017** | 0.0051 | 0.0104 | 0.0137 |
| 16 | 0.0066 | **0.00023** | 0.0019 | 0.0046 | 0.0063 |
| 32 | 0.0031 | **0.00002** | 0.00084 | 0.0022 | 0.0030 |

The uncorrected bias falls as `1/p` exactly, as it should. The note's version
falls by a factor of ~500 across that range, consistent with removing the whole
`O(1/p)` term and leaving `O(p⁻²)` or better. pytwoway's falls by only ~17, so it
removes part of the `O(1/p)` bias and leaves a residual of the same order. Both
mixed variants are worse than the note's, so the sign and the coefficient each
matter independently — it is not a typo that happens to cancel.

**Implement the note's version.** In practice pytwoway's residual is tiny:
extrapolating its rate to `p = 250` gives roughly `5e-5` in relative terms, far
below the sampling noise of the estimate it sits inside, which is presumably why
this has gone unnoticed. It is no reason to distrust published pytwoway results.
It is simply free to get right when writing it fresh.

## The B̂ᵢᵢ term: two valid approaches, and they differ in cost

`θ̂ = β̂'Aβ̂ − Σᵢ B̂ᵢᵢ σ̂²ᵢ` needs `Bᵢᵢ = xᵢ'S⁻¹ A S⁻¹ xᵢ`. Decompose
`A = ½(A₁'A₂ + A₂'A₁)`; then with a second Rademacher matrix `R_B`,

```
B̂ᵢᵢ = (1/p) (R_B A₁ S⁻¹ xᵢ)' (R_B A₂ S⁻¹ xᵢ)
```

**KSS's Algorithm 3** does this per observation. For the two-way model the three
components use `A_ψ = A_f'A_f`, `A_α = A_d'A_d`, `A_αψ = ½(A_d'A_f + A_f'A_d)`
where `A_f'` and `A_d'` are the `1/√n`-scaled, centered firm and worker blocks —
which is exactly the `Q` construction in `kss_exact.quadratic_forms`. Because
`A_f` and `A_d` are shared, **the same 2p systems serve all three components**:

```
solve S z_{κ,ℓ} = r_{κ,ℓ}   for ℓ = 0 (leverage), 1 (firm), 2 (worker)
P̂ᵢᵢ = (1/p)‖Z₀'xᵢ‖²   B̂ᵢᵢ,ψ = (1/p)‖Z₁'xᵢ‖²
B̂ᵢᵢ,α = (1/p)‖Z₂'xᵢ‖²  B̂ᵢᵢ,αψ = (1/p)(Z₁'xᵢ)'(Z₂'xᵢ)
```

Total **3p solves** for the whole decomposition.

**pytwoway instead uses Hutchinson's trace estimator in coefficient space**:
draw `Z ∈ ℝᴷ` Rademacher, compute `S⁻¹A'ΩAS⁻¹Z` (two solves), and take
`Z'QZ`-style products. That estimates the scalar `tr(Q·S⁻¹A'ΩAS⁻¹)` directly and
never forms `B̂ᵢᵢ` at all. Also unbiased; `2p` solves.

**For streaming, pytwoway's choice is the better one**, and this is the most
useful thing in these notes. The trace approach needs no per-observation `B̂ᵢᵢ`
output — only a handful of scalars — so the only row-sized quantities the whole
algorithm needs are the five leverage accumulators. That halves what has to be
written per row and removes three row-sized columns.

## Pruning is easier than I said

I previously called the leave-one-out connected set the hardest remaining piece.
It is the easiest. Appendix B.1, Algorithm 1, is **not iterative**:

```
1: function PruningNetwork(G)          # G = connected bipartite worker-firm graph
2:     G₁ ← G minus all workers that are articulation points in G
3:     G  ← largest connected component of G₁
4:     return G
```

One articulation-point pass (Tarjan) plus one connected-component pass, on a
graph whose vertices are workers and firms and whose edges are matches. KSS
report it completing in under a minute at their scale, and note that most firms
it removes are attached to a single mover. (The *leave-two-out* set, Algorithm 2,
*is* iterative — that is where the complexity I was expecting actually lives.)

Memory is `O(V + E)` integer arrays: for a million workers and 66k firms, a few
tens of MB. That is nothing next to the peak of a fit, so **pruning is not a
streaming obstacle** even though it touches the streamed dimension.

It is, though, a genuinely different sample from what `kss_exact.prune_to_leave_out`
produces. Algorithm 1 deletes *every* articulation-point worker — in KSS's own
application that drops roughly half the firms — whereas the leverage-based
stand-in drops only observations whose leverage reaches one. Implementing
Algorithm 1 is what would let results be compared against KSS, pytwoway or xhdfe
on the same sample without the pre-pruning handshake the current test harness
uses.

## Cost and accuracy, from their measurements

| | |
|---|---|
| operating range for `p` | 250–500 |
| error at `p = 500`, >1M effects | `0.028765` vs exact `0.028883`, so ~1e-4 |
| error at `p = 2500` | `0.0289022` — little gained over 500 |
| speedup vs exact at that size | ~100× (exact ≈ 8 hours) |
| largest reported | ~15M person and year effects in **35 minutes** at `p = 250` |
| how `p` scales with `n` | it does not: "the distance between our approximation and the true variance component decreases with the sample size for a fixed `p`" |

That last row is the important one. `p` is a fixed budget, not a function of the
data, so the JLA cost is linear in the number of passes and the accuracy
*improves* as the panel grows.

## What this implies for a streaming implementation

Revised from the earlier estimate, now that the algorithm is pinned down:

**Solves.** `p` for the leverages plus `2p` for the trace, but every one of them
is a right-hand side against the *same* operator. `hdfe_stream`'s block PCG
already takes many right-hand sides in lockstep and `_build_explicit` factorizes
`S` once and reuses it, so 3p = 1500 right-hand sides at `p = 500` is
`1500/rhs_block ≈ 188` block solves against one factorization. This remains the
single best fit between the paper's algorithm and this library's existing design.

**Row passes.** Two per block of right-hand sides: one to form `X'q` (a reduction
into level-sized accumulators, the shape of `_nb_rhs`), one to evaluate
`xᵢ'S⁻¹X'q` and accumulate the five per-row moments (the shape of `_nb_pass2`).

**The coefficient-space vectors are the real constraint.** `Z ∈ ℝ^{K×p}` has a
worker block, and at 1M workers × 500 draws that is 4 GB — exactly what this
library exists to avoid. The fix is the trick the library already uses: store
only the *firm* block (`n_firms × p`, e.g. 66k × 500 × 8B ≈ 264 MB, or less with
blocking) and reconstruct each worker's own entries inside the group pass, which
is what `_nb_pass2` already does to recover the streamed effects. Blocking over
`p` bounds it wherever needed.

**Random vectors need no storage.** `q` is required in both the `X'q` pass and
the `(rᵢ − qᵢ)` accumulation. Row files are fixed after pass 0, so a
counter-based PRNG keyed on `(row ordinal, draw)` regenerates them identically
in both passes.

**The remaining unknown** is the `apply_inverse` refactor — applying `S⁻¹` to an
arbitrary vector rather than only to the design's own cell sums, including the
streamed dimension and the covariates via Frisch–Waugh. That has not changed, and
it is still the piece to build first.

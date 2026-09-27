# Prototypes

Exploratory work that is **not part of the `hdfe_stream` package**. Nothing here
is imported by the library, nothing here is covered by the package's test suite
(`pytest` collects `tests/` only), and none of it streams.

## `kss_exact.py` — Kline–Saggio–Sølvsten leave-out variance components

An exact, dense, small-sample implementation of the KSS bias correction for AKM
variance components: `var(psi)`, `var(alpha)` and `cov(psi, alpha)`.

Its purpose is **not** to be useful at scale — it forms `(X'X)⁻¹` densely, so it
is limited to a few thousand coefficients. Its purpose is to pin down *what the
estimator is*, on panels small enough that every quantity can be computed
exactly and checked against a known answer, so that a later streaming
implementation has a trustworthy reference to be tested against.

```python
import sys; sys.path.insert(0, "prototypes")
from hdfe_stream.simulate import simulate_akm
from kss_exact import Panel, kss, prune_to_leave_out

df = simulate_akm(n_workers=300, n_firms=30, seed=1)
panel, rounds = prune_to_leave_out(Panel.from_frame(df))
print(kss(panel).summary())
```

```
KSS leave-out variance components  (n=2,539, workers=300, firms=30, movers=1,196)
max leverage 0.328774   mean sigma2 0.094064

component                plug-in        bias         KSS
--------------------------------------------------------
var(psi)                0.039171    0.006859    0.032311
var(alpha)              0.162458    0.017110    0.145348
cov(psi, alpha)         0.008812   -0.005892    0.014705
```

### Validation

```bash
pytest prototypes/            # ~2.5 minutes
```

Three layers, in increasing order of how much they establish:

1. **Algebraic identities.** Leverages sum to the rank of the design and lie
   strictly inside the unit interval; the components are invariant to which firm
   is dropped for the normalization; pruning reaches a fixed point; a panel that
   is not leave-one-out connected is rejected with an explanation.

2. **Unbiasedness against known truth.** This is the load-bearing test.
   `simulate_akm(keep_effects=True)` returns the worker and firm effects that
   generated the data, so the estimand is known exactly. Over 50 noise draws on
   a fixed worker–firm structure:

   | component | truth | mean plug-in | bias | mean KSS | bias | t |
   |---|---:|---:|---:|---:|---:|---:|
   | var(psi) | 0.064830 | 0.069212 | +0.004382 | 0.063671 | −0.001159 | −0.96 |
   | var(alpha) | 0.136796 | 0.152206 | +0.015410 | 0.137166 | +0.000370 | 0.40 |
   | cov(psi, alpha) | 0.024798 | 0.020351 | −0.004447 | 0.024887 | +0.000090 | 0.20 |

   The plug-in estimator is biased, upward for the variances and toward zero for
   the covariance, exactly as theory says. The corrected estimator is
   indistinguishable from the truth: every t-statistic is inside ±1.

   Note the asymmetry the tests are careful about: bias is a property of the
   *expectation*. On a single draw the corrected estimate is sometimes further
   from the truth than the plug-in one, because KSS buys unbiasedness, not lower
   variance. Only the replication average can test the claim.

3. **Cross-check against pytwoway** on an identical sample. Skipped unless the
   reference environment below exists.

### The reference environment

pytwoway 0.3.21 needs `numpy < 1.24` (it uses `np.bool8`), while `hdfe_stream`
runs on numpy 2.x, so the comparison runs pytwoway under its own interpreter and
exchanges data through files:

```bash
uv venv --python 3.11 prototypes/.ref-venv
uv pip install --python prototypes/.ref-venv/bin/python \
    pytwoway "numpy<1.24" "scipy<1.11"
```

(`scipy` has to be pinned too: recent versions want `np.long`, which numpy
removed in 1.24 and only restored in 2.0.)

The comparison is run on the sample **after** `bipartitepandas` has pruned it to
its own leave-one-out connected set, and that pruned sample is handed back for
this implementation to use. The two select the sample differently — see
`prune_to_leave_out` — and comparing estimators on different samples would tell
you nothing.

Results of the cross-check:

| quantity | agreement |
|---|---|
| leverages (max) | exact, 1e-8 |
| leave-one-out σ̂² (mean) | exact, 1e-8 |
| plug-in components, all three | exact, 1e-8 |
| `var(psi)` bias and corrected value | exact, 1e-7 |
| `cov` and `var(alpha)`, vs pytwoway's **approximate** path | within 5% (its sampling noise) |
| `var(alpha)`, vs pytwoway's **exact-trace** path | **disagrees — see below** |

### A disagreement worth knowing about

With `exact_trace_he=True`, pytwoway 0.3.21 returns a bias term for `var(alpha)`
approximately equal to the entire plug-in value, so its corrected `var(alpha)`
comes out **negative** — impossible for a variance:

| | `var(alpha)` bias | corrected |
|---|---:|---:|
| pytwoway, `exact_trace_he=True` | 0.165441 | −0.003373 |
| pytwoway, Johnson–Lindenstrauss (300 draws) | 0.015863 | 0.146205 |
| this prototype, exact | 0.016017 | 0.146051 |

pytwoway's own approximate path agrees with the exact computation here to about
1%, and the Monte Carlo shows this implementation is unbiased for that component,
so the problem is on their side in that configuration. `var(psi)` and `cov` are
unaffected and agree to machine precision.

`test_pytwoway_exact_trace_disagrees_on_var_alpha` asserts the discrepancy so
that a future pytwoway release which fixes it fails the test loudly and prompts a
re-check, rather than the observation quietly rotting in a comment. I have not
investigated their code far enough to say whether this is a bug worth reporting
upstream or a documented limitation of that option.

### What this does not do

- **Scale.** `(X'X)⁻¹` is dense. This is the whole reason it is a prototype.
- **Johnson–Lindenstrauss approximation** of the leverages and bias terms, which
  is what a streaming version would need, including the non-linearity bias
  correction that pytwoway applies to σ̂² when leverages are approximate
  (`jla_factor`, from Saggio's `improved_JLA.pdf`).
- **The real leave-one-out connected set.** KSS define the sample by a graph
  condition: the largest set of firms that stays connected after removing any one
  worker's observations. `prune_to_leave_out` instead detects the *consequence* —
  a mover whose leverage is one — and iterates until none is left. The two often
  coincide but are not the same thing, which is exactly why the pytwoway
  comparison is run on a pre-pruned sample.
- **Inference** on the components.
- **Covariates or IV.** Two fixed effects, an outcome and an optional weight,
  nothing else.

### Where this leaves the streaming question

[`JLA_NOTES.md`](JLA_NOTES.md) works through the Johnson–Lindenstrauss algorithm
from the paper, the authors' 2021 follow-up note, and pytwoway's implementation.
Reading it revised two of the judgments made before it:

- the **leave-one-out connected set is the easiest** remaining piece, not the
  hardest — KSS's Algorithm 1 is an articulation-point pass, not the harder
  graph problem expected (that is the *leave-two-out* set). It does have to be
  repeated until no articulation point remains, as LeaveOutTwoWay and xhdfe do:
  removing one can create another. An early version here ran it once, a
  defect since fixed;
- the bias term should follow **pytwoway's** Hutchinson trace estimator rather
  than KSS's per-observation `B̂ᵢᵢ`, because it produces scalars and so needs no
  row-sized output. This still holds; see the note on the null space below,
  which does *not* change it.

**The null space, and why it does not change the choice.** Coefficient-space
Hutchinson is usually written assuming `S⁻¹` exists. pytwoway drops a firm
dummy, so for it that is true. `hdfe_stream` keeps every level of every
dimension, so `S` is singular — nullity 1 for two fixed effects, 2 for three —
and a *random* coefficient-space vector lands outside `range(S)`: measured at
4.75% of its norm, with the solve then diverging.

`apply_inverse` projects onto `range(S)` first, which is what the pseudo-inverse
does. The null space is known structurally (one vector per connected component,
plus one per dimension beyond the second), so the projection is a QR of a tiny
basis and two products — free in any practical sense. The result matches dense
`A·pinv(S)·v` to 1e-14.

This is worth knowing but it is **not** a reason to prefer KSS's row-space
`B̂ᵢᵢ`. That formulation solves for `S⁻(A_ℓ'r)`, which is equally a
coefficient-space vector — merely one assembled from rows — and whether it lands
in `range(S)` depends on the centering and the component structure rather than
being automatic. The projection question does not separate the two approaches.
(What *is* automatically consistent is a right-hand side reduced from rows by
`project_rows`, which is why none of this arose for the leverages.)

| piece | status |
|---|---|
| what the estimand is, and the exact form of the correction | **settled** (this file) |
| σ̂² conventions (demeaned outcome, stayers) | **settled** — LeaveOutTwoWay's: own `σ̂²` at observation level, `sigma_for_stayers` at match level |
| standard errors when leaving out a match | **built** — reference: `leave_out_COMPLETE` with `'matches'` (beta there). `xhdfe_match_se` ports xhdfe's port of it literally and reproduces it; `kss_match_se` is the package's version (reference by default, documented differences), streaming agrees with it within 10%; coverage in `docs/kss_methodological_differences.md` §5.7 |
| leaving out a match, as the references do by default | **built** — `kss_match()` here agrees with xhdfe to 5e-14–5e-12; `hdfe_stream/leaveout_match.py` agrees with it within Monte Carlo error, stayers' `σ̂²` to 1e-12 |
| pruning as the references do it (iterated, single-observation workers dropped) | **fixed** — matches xhdfe row for row on three panels |
| a trustworthy reference to test against | **settled** (Monte Carlo + pytwoway) |
| the JLA algorithm, its two generations, and the cost/accuracy budget | **settled** ([JLA_NOTES.md](JLA_NOTES.md)) |
| which `B̂ᵢᵢ` estimator suits streaming | **settled** ([JLA_NOTES.md](JLA_NOTES.md)) |
| the sign/coefficient discrepancy between pytwoway and the 2021 note | **settled** — the note is right, shown by a bias race ([JLA_NOTES.md](JLA_NOTES.md)) |
| applying `S⁻` to an arbitrary vector in the streaming pipeline | **built** — `hdfe_stream/inverse.py`, validated against dense to 1e-13 |
| the five per-row JLA accumulators | **built** — `hdfe_stream/leaveout.py`, with the note's normalization and non-linearity correction |
| Algorithm 1 pruning (articulation points) | **built** — `leave_one_out_connected()` in `hdfe_stream/leaveout.py`; takes a 3.4M-row panel from max leverage 1.0 to 0.71 in 0.8s |
| covariates in the projection | **fixed** — bordered solve with a Schur complement on the k×k covariate block; matches the dense `[X, D]` design to 1e-8, and `jla_leverages` now defaults to the fitted model's own covariates |
| coefficient-space `S⁻` (`apply_inverse`) | **built** — matches the dense pseudo-inverse to 1e-8 across covariates, 3 FE, weights, both solvers |
| the null-space problem it exposed | **found and fixed** — `range(S)` projection; does not change the choice of estimator |
| the Hutchinson trace itself | **built** — `hutchinson_trace()`; converges to the dense trace, block-invariant |
| assembling the estimator (plug-in + mover/stayer σ̂² + the pieces above) | **built** — `leave_out_components()` |
| validating the streaming estimator against `kss_exact` and pytwoway | **done** — plug-in exact; leave-out within 3e-5 of the prototype at 1024 draws, and within 0.2% of pytwoway on the same sample |
| weights in the leave-out layer | **built** — √W leverages, weighted plug-in and σ̂²; the weighted prototype is unbiased by Monte Carlo (t within ±1), and streaming matches it to 0.05% at 2048 draws with the unweighted path unchanged |
| standard errors for the components | **built** — `component_variances()`; the exact identity `ỹ'Cỹ == plug-in − bias` holds to 1e-14 through the streaming operator, streaming matches dense within 2%, and Monte Carlo coverage is 94.4–94.9% at n=5120 |
| KSS's split-sample σ̃² (edge-disjoint paths) | **not implementable here, and not in the reference either** — see [SE_NOTES.md](SE_NOTES.md) |
| weak-identification diagnostic (λ²/Σλ², q, first-stage F) | **built** — `weak_id_diagnostics()`, on by default with `se=True`. Eigenvalues match dense to 1e-13 at convergence; deflated Hutchinson matches `tr((S⁻Q)²)` to four decimals on a bottleneck where the plain estimator was 160% off. See [WEAKID_NOTES.md](WEAKID_NOTES.md) |
| weak-identification interval (Theorem 3 / Andrews–Mikusheva curvature, q = 1) | **built** — deflated kernel `C₂` exact to 1e-15 dense / 1e-9 streaming, interval matches Saggio's quartic to 3e-15, and Monte Carlo shows nominal coverage where the normal interval undercovers. See [WEAKID_NOTES.md](WEAKID_NOTES.md) |
| q ≥ 2 interval | not started — a (q+1)-dimensional quadratic program with the full curvature maximization |
| how it surfaces in the public API, and an example | not started |

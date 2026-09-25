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
   is dropped for the normalisation; pruning reaches a fixed point; a panel that
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
- **Covariates, weights, or IV.** Two fixed effects and an outcome, nothing else.

### Where this leaves the streaming question

The pieces a streaming implementation would need, and which of them this
prototype has settled:

| piece | status |
|---|---|
| what the estimand is, and the exact form of the correction | **settled here** |
| σ̂² conventions (demeaned outcome, stayer imputation) | **settled here** |
| a trustworthy reference to test against | **this file, plus the Monte Carlo** |
| applying `(X'X)⁻¹` to an arbitrary vector inside the streaming pipeline | not started |
| JLA projections and their bias correction | not started |
| the graph-theoretic leave-one-out set | not started |

# Draws and reproducibility

The leverages, the bias term and the standard errors are all approximated by
random draws, so leave-out estimates carry Monte Carlo noise, and they are
reproducible rather than deterministic. This page says how many draws are
enough, what the results depend on, and what to report in a paper.

## Choosing `n_draws`

The leverages are approximated, so the result carries Monte Carlo noise.
Accuracy does **not** have to grow with the data — KSS report the error falling
with sample size at fixed draws — so this is a budget, not a function of `n`.
They operate at 250–500 on panels with millions of effects, and so should you.

The way to know it is enough is to vary the seed and look at the spread.
Measured across five seeds on the example panel (8,000 workers, 67,913
person-years, 13,404 matches), as the standard deviation of each estimate:

| setting | 250 draws, % of the estimate | 1,000 draws | at 250 draws, relative to the standard error |
|---|---|---|---|
| leaving out a match (default) | 0.07–0.53% | 0.06–0.27% | 9–12% |
| leaving out a match, `centering="weighted"` | 0.06–0.21% | 0.02–0.10% | about 10% |
| leaving out an observation | 0.01–0.06% | 0.00–0.08% | 2.5–3.5% |

Leaving out a match is the noisier of the two: fewer rows carry the leverages,
and with the default centering each row's σ̂² carries the outcome's level (see
[methodological differences](differences.md) §3.4). Even
there, the approximation's noise at 250 draws is about a tenth of the standard
error, so it adds well under 1% to the total uncertainty. Check it on your own
panel rather than relying on these numbers.
See [Reproducibility](#reproducibility) for what to report in a paper.

## Reproducibility

A fit itself is deterministic (see [reproducibility](../examples/out_of_core.md#reproducibility)).
The leave-out machinery is the
exception: `jla_leverages` approximates statistical leverages by random
projection, and the bias term and standard errors use random draws too. Those
estimates depend on a draw sequence, so they are reproducible rather than
deterministic, and it is worth knowing exactly what they depend on.

| change this | do the leverages change? |
|---|---|
| re-running the same script | no |
| the input's row order (a shuffle, a re-export, a different upstream query) | no |
| thread count, or running on another machine | no |
| `seed=` | yes, by sampling noise |
| `n_draws=` | yes, by sampling noise, decreasing in `n_draws` |
| `rows_per_bucket`, `batch_rows`, `n_buckets` | yes, by sampling noise |

Row-order invariance is not free and is worth a word, because it is the one most
easily lost. The random vectors are keyed on a row's position in the sorted
working copy, so a fit that keeps its intermediates orders rows by the
fixed-effect codes *and then by the variables*, making the stored order a
function of the data rather than of the order it arrived in. Both halves of that
key matter: without the codes, a re-exported panel changes the answer; without
the variables, two observations of the same worker at the same firm stay tied and
can swap between runs, taking each other's draws.

An ordinary fit does not pay for this. Every fit sorts by the streamed dimension
regardless -- the algorithm needs whole groups together -- but the extra keys are
added only when the intermediates are kept, which is exactly when the leave-out
machinery can reach them. On a 3.4M-row panel with four variables the wider key
costs about 4% of fit time, and nothing at all if you are not doing leave-out
estimation.

The memory options are the remaining gap: they decide how rows are partitioned,
so they move rows and therefore move the draws. The effect is sampling noise,
not bias — but it means a study should pin them alongside the seed.

**For a paper, this is what to do:**

1. **Fix `seed` and `n_draws` explicitly** rather than relying on defaults, and
   report both. Defaults can change between versions; your paper should not.
2. **Pin the memory options too** (`rows_per_bucket`, `batch_rows`) if you set
   them at all, for the reason above.
3. **Report the estimate across several seeds, not just one.** This is the part
   that actually matters. The leave-out correction is a Monte Carlo
   approximation, so the honest claim is that your result is stable to the
   approximation — not that one particular draw is reproducible. Running five
   seeds and reporting the spread costs little and says something a single
   reproducible number does not.
4. **Raise `n_draws` until the spread is below the precision you report.** On
   the example panel the between-seed standard deviation of the estimates at 250
   draws was 0.07–0.53% of their value leaving out a match, and 0.01–0.06%
   leaving out an observation. That is about a tenth of the standard error and a
   thirtieth of it, respectively (see [Choosing `n_draws`](#choosing-n_draws)).
   Extreme statistics are far noisier than aggregates: the *maximum* leverage
   moved by 0.05 across seeds at 250 draws, which matters if you are using it to
   judge leave-one-out connectedness.

For comparison, pytwoway is also invariant to input row order, but its default
`rng=None` means two runs of the same script give different answers unless you
pass a generator explicitly. Here the seed has a fixed default, so the failure
mode is the opposite one: results are reproducible by default, and you have to
opt into varying the seed to learn how stable they are. Point 3 is that opt-in.

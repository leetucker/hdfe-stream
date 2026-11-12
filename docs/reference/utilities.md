# Utilities

## Reporting

```{eval-rst}
.. currentmodule:: hdfe_stream

.. autofunction:: etable
```

## Multiple testing

```{eval-rst}
.. autofunction:: bonferroni

.. autofunction:: holm

.. autofunction:: rwolf

.. autofunction:: wyoung
```

## Run directories

```{eval-rst}
.. autofunction:: cleanup
```

## Warnings

`ExperimentalWarning` marks a method that goes beyond what its authors developed.
Packages built on `hdfe_stream.core` raise subclasses of it for their own experimental
features. To silence all of them:

```python
import warnings
import hdfe_stream

warnings.filterwarnings("ignore", category=hdfe_stream.ExperimentalWarning)
```

```{eval-rst}
.. autoclass:: ExperimentalWarning
```

## Simulated data

```{eval-rst}
.. automodule:: hdfe_stream.simulate
   :no-members:

.. autofunction:: hdfe_stream.simulate.simulate_akm

.. autofunction:: hdfe_stream.simulate.simulate_rich

.. autofunction:: hdfe_stream.simulate.simulate_trends

.. autofunction:: hdfe_stream.simulate.simulate_bottleneck
```

## IV diagnostics from sums

The algebra behind `HDFEResult.iv_diag`, for estimators built on
`hdfe_stream.core` (it works on cross-products and score covariances, never
on data).

```{eval-rst}
.. automodule:: hdfe_stream.core.ivtests
   :no-members:

.. autofunction:: hdfe_stream.core.ivtests.identification

.. autofunction:: hdfe_stream.core.ivtests.kp_rk

.. autofunction:: hdfe_stream.core.ivtests.weak_f

.. autofunction:: hdfe_stream.core.ivtests.iid_omegas

.. autofunction:: hdfe_stream.core.ivtests.hansen_j

.. autofunction:: hdfe_stream.core.ivtests.sargan

.. autofunction:: hdfe_stream.core.ivtests.partial_r2

.. autofunction:: hdfe_stream.core.ivtests.effective_f

.. autofunction:: hdfe_stream.core.ivtests.anderson_rubin

.. autofunction:: hdfe_stream.core.ivtests.c_statistic
```

## Refit bootstrap

The machinery of a bootstrap that resamples whole units, or whole clusters of
units, with replacement and refits an estimator on each draw, for estimators
built on `hdfe_stream.core` (diff-stream's `refit_bootstrap` uses it). The
resamples are written to disk in bounded memory; refits run in the parent
process or in fresh worker processes, each with its share of the job's threads
and memory. The consumer supplies the fit and decides what a draw must satisfy.

```{eval-rst}
.. autofunction:: hdfe_stream.core.refit.refit_draws

.. autofunction:: hdfe_stream.core.refit.refit_workers

.. autoclass:: hdfe_stream.core.refit.RefitDraws

.. autoclass:: hdfe_stream.core.refit.RefitBootstrapResult
   :members: standard_errors, confint, tidy, draws, failures, refit_warnings, summary_json

.. autoclass:: hdfe_stream.core.refit.BootstrapFailureWarning

.. autoclass:: hdfe_stream.core.resample.DiskGroupResampler
   :members: materialize

.. autoclass:: hdfe_stream.core.partition.GroupPartitions

.. autofunction:: hdfe_stream.core.partition.whole_group_batches

.. autofunction:: hdfe_stream.core.partition.preprocess_budget_mb

.. autoclass:: hdfe_stream.core.process.ProcessBatch
```

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

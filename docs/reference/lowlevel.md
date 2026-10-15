# Low-level interface

`StreamingHDFE` and `StreamingGLM` take column names, or `(name, Polars
expression)` pairs, instead of a formula, and need no pyfixest. The formula
functions build one of them for each set of fixed effects.

```{eval-rst}
.. currentmodule:: hdfe_stream

.. autoclass:: StreamingHDFE
   :members: fit

.. autoclass:: StreamingGLM
```

`StreamingGLM` has the same `fit` method as `StreamingHDFE`.

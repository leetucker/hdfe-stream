# Reporting

Streaming results convert to pyfixest model objects, so pyfixest's reporting
(`etable`, `summary`, `coefplot`, `iplot`) works on them, and streaming and
ordinary pyfixest models can appear in the same table. This is output-side
only: nothing here is needed to fit a model.

## Regression tables

[`hdfe_stream.etable`](../reference/utilities.md) takes streaming results,
multi-model results and pyfixest models, mixed freely, and passes everything
else through to `pf.etable`:

```python
from hdfe_stream import etable

etable([weighted, iv], type="df", model_stats=["N", "se_type", "r2", "r2_within"])
```

```text
                                log_earn                 y_iv
                                     (1)                  (2)
coef  age_squared    0.078*** \n (0.014)  0.002*** \n (0.001)
      age_cubed     -0.012*** \n (0.001)
      x2                                  0.197*** \n (0.002)
fe    firm_id                          x                    x
      worker_id                        x                    x
      year                             x                    -
stats Observations               169,833              169,833
      S.E. type            by: worker_id        by: worker_id
      R²                           0.798                    -
      Within R²                    0.019                    -
```

A multi-model result expands into one column per model, and `type="tex"`
writes LaTeX.

A small model is quicker with pyfixest; a big one is quicker with hdfe_stream,
and one too big for memory needs it. They can go in the same table, and on the
same data they agree:

```python
in_memory = pf.feols(fml, data=pl.read_parquet(path).to_pandas(),
                     vcov={"CRV1": "worker_id"}, fixef_rm="none")
streaming = feols_stream(fml, path, workdir="scratch", vcov={"CRV1": "worker_id"})
etable([in_memory, streaming], type="df")
```

`etable` also normalizes the spacing of the fixed-effect side, so `a + b` and
`a+b` share a row. (`pf.etable` itself matches rows on the literal text between
the `+` signs, spaces included.) A dimension spelled differently, or in a
different position within an interaction, is still a separate row.

## pyfixest's own reporting

`to_pyfixest()` hands back a pyfixest `Feols` (or `Feiv`) view of the result,
for `pf.summary`, `pf.coefplot` and `pf.iplot`. The interaction terms of an
event study are flagged on the converted model, so `iplot` finds them:

```python
pf.summary(iv.to_pyfixest())
figure = pf.iplot(event.to_pyfixest(), plot_backend="matplotlib")
```

Only the stored results are available. Methods that need the data (`predict`,
recomputing the vcov, the wild bootstrap) are not, because the converted model
has no data behind it. Ask for the vcovs you want at fit time with `cluster=`,
switch with `with_vcov()`, and convert again.

:::{dropdown} The full script: `examples/reporting.py`
```{literalinclude} ../../examples/reporting.py
:language: python
```
:::

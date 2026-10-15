# One fixed effect, or none

The library is built around several fixed effects, one of them streamed, but
nothing requires more than one, or any.

One dimension
: a within regression, `y ~ x | worker_id`. The dimension is streamed and
  demeaned group by group, and with no other dimension there is nothing for the
  solver to do.

None
: ordinary least squares, `y ~ x`. The rows are read in plain batches and never
  grouped or sorted, and the model keeps its intercept as a coefficient, named
  `Intercept` as in pyfixest.

Either way the rows are streamed, so the memory ceiling is the same as for a
full AKM model: set by the batch size and the number of covariates, not by the
number of rows.

```python
X = "age_squared + age_cubed + i(occ)"
ols = feols_stream(f"log_earn ~ {X}", data, workdir="scratch")
within = feols_stream(f"log_earn ~ {X} | worker_id", data, workdir="scratch")
within.fixef("worker_id").collect()        # 20,000 worker effects
```

`csw0()` adds fixed effects one at a time, starting from none. Each set is its
own group of passes over the data:

```python
steps = feols_stream(f"log_earn ~ {X} | csw0(worker_id, firm_id)", data,
                     workdir="scratch", vcov="hetero")
for fit in steps:
    print(" + ".join(fit.fe_names) or "(none)", fit.coef()["age_squared"], fit.r2)
```

```text
  (none)                 +0.0751   R2 0.018
  worker_id              +0.0936   R2 0.761
  worker_id + firm_id    +0.0930   R2 0.800
```

:::{dropdown} The full script: `examples/fewer_fixed_effects.py`
```{literalinclude} ../../examples/fewer_fixed_effects.py
:language: python
```
:::

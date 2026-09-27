"""One fixed effect, or none.

The library is built around several fixed effects, one of them streamed, but
nothing requires more than one -- or any.

One dimension
    a within regression. The dimension is streamed and demeaned group by
    group, and with no other dimension there is nothing for the solver to do.

None
    ordinary least squares. The rows are read in plain batches and never
    grouped or sorted, and the model keeps its intercept as a coefficient,
    named "Intercept" as in pyfixest. The default vcov is then iid, again as in
    pyfixest (with fixed effects it is CRV1 by the first one).

Either way the rows are streamed, so the memory ceiling is the same as for a
full AKM model: set by the batch size and the number of covariates, not by the
number of rows.

    python examples/fewer_fixed_effects.py
"""

from hdfe_stream import feols_stream
from simulated_data import panel_path, workdir

data = panel_path()
X = "age_squared + age_cubed + i(occ)"

# ------------------------------------------------------------------- none: OLS
ols = feols_stream(f"log_earn ~ {X}", data, workdir=workdir("ols"), verbose=False)
print("no fixed effects (vcov defaults to iid):")
ols.summary()
print(f"solver: {ols.solver_info['solver']!r}; fixed effects: {ols.fe_names}")

# ------------------------------------------------------- one: a within model
within = feols_stream(f"log_earn ~ {X} | worker_id", data, workdir=workdir("within"),
                      verbose=False)
print("\none fixed effect (vcov defaults to CRV1 by worker):")
within.summary()
effects = within.fixef("worker_id").collect()
print(f"{effects.height:,} worker effects, e.g.\n{effects.head(3)}")

# ------------------------------------------- zero, one and two in one formula
# csw0() adds the fixed effects one at a time, starting from none. Each set is
# its own group of passes over the data; the table lines them up.
steps = feols_stream(f"log_earn ~ {X} | csw0(worker_id, firm_id)", data,
                     workdir=workdir("csw0"), vcov="hetero", verbose=False)
print("\nthe age_squared coefficient as fixed effects are added:")
for fit in steps:
    fe = " + ".join(fit.fe_names) or "(none)"
    print(f"  {fe:<22} {fit.coef()['age_squared']:+.4f}   R2 {fit.r2:.3f}")

for fit in (ols, within, steps):
    fit.cleanup()

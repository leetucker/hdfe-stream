"""Worker-specific time trends (varying slopes, FEIS), and choosing which
dimension to stream.

A worker fixed effect allows each worker their own level. Varying slopes allow
each worker their own *trend* as well -- `worker_id[t]` is a worker intercept
plus a worker-specific slope on t. That is a lot of parameters (one per worker
per slope), which is exactly the case that does not fit in memory as a design
matrix, and exactly the case this library handles by keeping the work inside
each worker's own rows.

Because the projection stays local to a worker's rows, the slopes have to be on
the streamed dimension, and that dimension is chosen automatically when any
dimension carries slopes.

    python examples/varying_slopes.py
"""

from hdfe_stream import StreamingHDFE, feols_stream
from simulated_data import trends_path, workdir

data = trends_path()

# In this panel each worker has their own linear and quadratic trend, so
# `y ~ x | worker_id[t] + firm_id` is the correctly specified model.
#
# x enters y directly with coefficient 0.3, and again through xe (which is
# 0.3 x + ...) with coefficient 0.2, so with xe left out of the regression the
# coefficient on x identifies 0.3 + 0.2 * 0.3 = 0.36.
intercepts = feols_stream("y ~ x | worker_id + firm_id", data,
                          workdir=workdir("feis_none"), verbose=False)
linear = feols_stream("y ~ x | worker_id[t] + firm_id", data,
                      workdir=workdir("feis_linear"), verbose=False)
quadratic = feols_stream("y ~ x | worker_id[t, t2] + firm_id", data,
                         workdir=workdir("feis_quad"), verbose=False)

print("true coefficient on x: 0.360000\n")
print(f"{'specification':<34}{'x':>11}{'se':>10}{'RSS':>12}{'FE params':>11}")
print("-" * 78)
for label, fit in [("worker_id (intercepts only)", intercepts),
                   ("worker_id[t] (+ linear trend)", linear),
                   ("worker_id[t, t2] (+ quadratic)", quadratic)]:
    print(f"{label:<34}{fit.beta[0]:>11.6f}{fit.se[0]:>10.6f}"
          f"{fit.rss:>12.1f}{fit.k_fe:>11,}")

print("\nAdding the trends the data actually has cuts the residual sum of "
      "squares,\nat the cost of many more fixed-effect parameters.")

# The fixed-effect output carries the intercept and one column per slope
print("\nper-worker intercepts and slopes:")
print(linear.fixef("worker_id").head(5).collect())

# A covariate inside the span of the worker intercepts and their trends cannot
# be identified and is dropped. age is birth year plus t, so age_squared lies in
# the span of {1, t, t2} within a worker.
absorbed = feols_stream("y ~ x + age_squared | worker_id[t, t2] + firm_id", data,
                        workdir=workdir("feis_collinear"), verbose=False)
print("\ndropped as collinear with the worker trends:", absorbed.collin_vars)

# ------------------------------------------------------- the streamed dimension
# One dimension is never held in memory as a vector: it is streamed. Pick the
# one with the most levels whose groups each touch only a few levels of the
# others -- workers, not firms. The default does this for you.
print(f"\n{'=' * 78}\nwhich dimension gets streamed\n{'=' * 78}")
for fe, stream in [(["worker_id", "firm_id"], None),
                   (["firm_id", "worker_id"], None),
                   (["worker_id[t]", "firm_id"], None),
                   (["worker_id", "firm_id"], "firm_id")]:
    fit = StreamingHDFE("y", ["x"], fe, workdir=workdir("stream"),
                        stream=stream, verbose=False).fit(data)
    choice = fit.diagnostics["stream"]
    asked = f"stream={stream!r}" if stream else "default"
    print(f"  fe={str(fe):<32}{asked:<18}-> {choice['dim']}")
    print(f"      because: {choice['reason']}")

# ------------------------------------------------------- low-level interface
# `StreamingHDFE` takes column names or (name, Polars expression) pairs instead
# of a formula, which is handy when a covariate is easier to write in Polars --
# and it needs no pyfixest at all.
import polars as pl  # noqa: E402

est = StreamingHDFE(
    y="y",
    x=["x", ("late_period", (pl.col("year") >= 2010).cast(pl.Float64))],
    fe=["worker_id[t]", "firm_id"],
    workdir=workdir("lowlevel"),
    verbose=False,
)
print(f"\n{'=' * 78}\nlow-level interface with a Polars expression\n{'=' * 78}")
print(est.fit(data, vcov={"CRV1": "firm_id"}).tidy())

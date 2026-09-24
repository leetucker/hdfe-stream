"""Regression tables and plots, through pyfixest.

Streaming results convert to pyfixest model objects, so pyfixest's reporting --
`etable`, `summary`, `coefplot`, `iplot` -- works on them, and streaming and
ordinary pyfixest models can appear in the same table.

This is output-side only: nothing here is needed to fit a model.

    python examples/reporting.py
"""

import polars as pl
import pyfixest as pf

from hdfe_stream import etable, feols_stream
from simulated_data import OUTPUT, panel_path, workdir

data = panel_path()

weighted = feols_stream("log_earn ~ age_squared + age_cubed | worker_id + firm_id + year",
                        data, workdir=workdir("rep_wls"), weights="wa",
                        vcov={"CRV1": "worker_id"}, verbose=False)
iv = feols_stream("y_iv ~ age_squared | worker_id + firm_id | x2 ~ z1 + z2",
                  data, workdir=workdir("rep_iv"), vcov={"CRV1": "worker_id"},
                  verbose=False)
event = feols_stream("log_earn ~ i(year, treat, ref=2009) + age_squared"
                     " | worker_id + firm_id",
                     data, workdir=workdir("rep_event"), vcov={"CRV1": "firm_id"},
                     verbose=False)
multi = feols_stream("log_earn + y2 ~ age_squared + age_cubed | worker_id + firm_id",
                     data, workdir=workdir("rep_multi"), vcov="hetero", verbose=False)

# ------------------------------------------------------------------- one table
# `etable` takes streaming results, multi-model results, and pyfixest models,
# mixed freely. Everything else is passed through to `pf.etable`.
print("side by side, weighted OLS and 2SLS:\n")
print(etable([weighted, iv], type="df",
             model_stats=["N", "se_type", "r2", "r2_within"]).to_string())

# A multi-model result expands into one column per model
print("\n\ntwo outcomes from one call:\n")
print(etable(multi, type="df").to_string())

# ------------------------------------------------- mixing with pyfixest models
# A model that fits in memory is quicker with pyfixest; a big one needs
# hdfe_stream. They can go in the same table, and on the same data they agree.
#
# One thing to know: pf.etable lines up fixed-effect rows by the literal text
# between the '+' signs, so the same dimension in a different position in the
# formula becomes a separate row. That is pyfixest's own behaviour -- it happens
# between two pyfixest models too -- so keep the fixed-effect side spelled the
# same way across models you want to compare.
FML = "log_earn ~ age_squared + age_cubed | worker_id + firm_id + year"
in_memory = pf.feols(FML, data=pl.read_parquet(data).to_pandas(),
                     vcov={"CRV1": "worker_id"}, fixef_rm="none")
streaming = feols_stream(FML, data, workdir=workdir("rep_same"),
                         vcov={"CRV1": "worker_id"}, fe_dof="pyfixest",
                         verbose=False)
print("\n\nsame model, pyfixest in memory (1) and hdfe_stream streaming (2):\n")
print(etable([in_memory, streaming], type="df").to_string())

# ------------------------------------------------------------------ LaTeX
tex_path = OUTPUT / "table.tex"
etable([weighted, iv], type="tex", file_name=str(tex_path))
print(f"\n\nLaTeX table written to {tex_path.name}")

# --------------------------------------------------- pyfixest's own reporting
# `to_pyfixest()` hands back a Feols (or Feiv) view of the result.
print("\n\npf.summary on a streaming result:\n")
pf.summary(iv.to_pyfixest())

# Event-study plot. The interaction terms are flagged on the converted model, so
# iplot finds them.
import matplotlib                                    # noqa: E402
matplotlib.use("Agg")                                # write a file, don't open a window

figure = pf.iplot(event.to_pyfixest(), plot_backend="matplotlib")
plot_path = OUTPUT / "event_study.png"
figure.savefig(plot_path, dpi=120, bbox_inches="tight")
print(f"\nevent-study plot written to {plot_path.name}")

# One thing the converted model cannot do: recompute its own vcov, because it
# has no data behind it. Ask for the vcov you want at fit time via `cluster=`
# and switch with `with_vcov` instead.
try:
    iv.to_pyfixest().vcov("hetero")
except NotImplementedError as err:
    print(f"\nas expected: {err}")

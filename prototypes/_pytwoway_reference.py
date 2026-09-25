"""Run pytwoway's KSS estimator and report its numbers as JSON.

    python _pytwoway_reference.py <panel.csv> <cleaned_out.csv> [--approximate-trace]

This runs under a *different interpreter* from the rest of the project:
pytwoway 0.3.21 needs numpy < 1.24 (it uses `np.bool8`), while hdfe_stream runs
on numpy 2.x. So the comparison talks to it over files rather than importing it.
See prototypes/README.md for how to build the reference environment.

It also writes back the panel *after* bipartitepandas has pruned it to the
leave-one-out connected set, so the prototype can be run on exactly the same
sample. That matters: the two implementations select the sample differently
(see `kss_exact.prune_to_leave_out`), and comparing estimators on different
samples would tell us nothing.

Input CSV columns: i (worker), j (firm), t (period), y (outcome).
"""

import json
import sys

import bipartitepandas as bpd
import pandas as pd
import pytwoway as tw
from pytwoway import Q


def main(in_path, cleaned_path, exact_trace=True):
    raw = pd.read_csv(in_path)

    cleaned = bpd.BipartiteDataFrame(raw).clean(
        bpd.clean_params({"connectedness": "leave_out_observation",
                          "drop_single_stayers": False,
                          "verbose": False}))
    # hand the same sample back for the prototype to use
    pd.DataFrame({"i": cleaned["i"].to_numpy(), "j": cleaned["j"].to_numpy(),
                  "t": cleaned["t"].to_numpy(), "y": cleaned["y"].to_numpy()}
                 ).to_csv(cleaned_path, index=False)

    estimator = tw.FEEstimator(cleaned, tw.fe_params({
        "he": True,                 # heteroskedasticity-robust == KSS
        "exact_lev_he": True,       # exact leverages, no random projection
        "exact_trace_he": exact_trace,
        **({} if exact_trace else {"ndraw_trace_he": 300}),
        "Q_var": [Q.VarPsi(), Q.VarAlpha()],
        "Q_cov": [Q.CovPsiAlpha()],
        "progress_bars": False,
        "verbose": False,
    }))
    estimator.fit()
    res = estimator.res

    out = {
        "n_obs": int(res["n"]) if "n" in res else int(len(cleaned)),
        "n_workers": int(res.get("n_workers", cleaned["i"].nunique())),
        "n_firms": int(res.get("n_firms", cleaned["j"].nunique())),
        "plug_in": {name: float(res[f"{name}_fe"])
                    for name in ("var(psi)", "var(alpha)", "cov(psi, alpha)")},
        "kss": {name: float(res[f"{name}_he"])
                for name in ("var(psi)", "var(alpha)", "cov(psi, alpha)")},
        "bias": {name: float(res[f"tr_{name}_he"])
                 for name in ("var(psi)", "var(alpha)", "cov(psi, alpha)")},
        "var_eps_he": float(res["var(eps)_he"]),
        "max_lev": float(res["max_lev"]),
        "min_lev": float(res["min_lev"]),
    }
    print("RESULT " + json.dumps(out))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2],
         exact_trace="--approximate-trace" not in sys.argv)

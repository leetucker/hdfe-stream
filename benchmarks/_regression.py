"""What the OLS, IV and binary-outcome benchmarks share: the data, one runner per
library, and the parent loop (ols_benchmark.py, iv_benchmark.py and
binary_benchmark.py say what each measures and keep their own CONFIGS).

A configuration is a (library, fixed effects) pair: 0 or 1 fixed effects, the
column g. Every library runs on the data of `_harness.general_data_path`, in a
subprocess of its own, and is timed from the Parquet file to the estimates:
`t_read` is reading the file and turning it into whatever the library takes
(a pandas frame, a Polars frame, NumPy arrays), `t_fit` the estimation.

What a library returns differs, and the CSV says which: scikit-learn, and the
plugin fits of polars-ols and polars-ds for logistic regression, return
coefficients only, so their rows have no standard errors. Where a library
offers standard errors, the fit asks for the classical ones (iid), the others'
default, and hdfe_stream is run with `vcov="iid"` too. A fit that demeans first
(the "demeaned" rows) has no standard errors reported: the degrees of freedom
of the group effects are not in them.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from _harness import (DATA_DIR, RESULTS_DIR, external_command,
                      external_missing, general_data_path, is_done, measure,
                      run_child, upsert_csv, write_machine_info)

OUTCOME = {"ols": "y", "iv": "yiv", "logit": "yb", "probit": "yb"}
# the coefficients that every library's estimates are compared on
KEYS = {"ols": ("x0", "x1"), "iv": ("x0", "x1", "d"),
        "logit": ("x0", "x1"), "probit": ("x0", "x1")}
# agreement needed before the timings are called comparable: the linear
# estimators agree to rounding, the binary ones to where their iterations stop
TOLERANCE = {"ols": 1e-6, "iv": 1e-6, "logit": 1e-5, "probit": 1e-5}
TIMEOUT_S = 30 * 60
N_COVARIATES = 10


def covariates(path):
    import pyarrow.parquet as pq
    names = pq.read_schema(path).names
    return sorted((c for c in names if c[0] == "x" and c[1:].isdigit()),
                  key=lambda c: int(c[1:]))


def variables(task, fe, xs):
    """The columns a fit reads."""
    return [OUTCOME[task], *xs, *(["g"] if fe else []),
            *(["d", "z1", "z2"] if task == "iv" else [])]


def packed(t_read, t_fit, n_obs, names, coefs, ses=None):
    """The report of a child: the coefficients (and standard errors, if any)
    under the names the libraries share."""
    def pick(values):
        return {k: float(v) for k, v in zip(names, values)}
    return {"t_read": t_read, "t_fit": t_fit, "n_obs": int(n_obs),
            "coefs": pick(coefs), "ses": pick(ses) if ses is not None else {}}


# --------------------------------------------------------------------------
# the child process: one configuration, one size
# --------------------------------------------------------------------------

def run_hdfe(task, fe, path, workdir, device="cpu"):
    from hdfe_stream import feglm_stream, feols_stream

    xs = covariates(path)
    rhs = " + ".join(xs)
    options = dict(workdir=str(workdir), vcov="iid", verbose=False, save_resid=False)
    if device != "cpu":
        options["device"] = device
    t0 = time.perf_counter()
    if task == "ols":
        fit = feols_stream(f"y ~ {rhs}" + (" | g" if fe else ""), str(path), **options)
    elif task == "iv":
        fit = feols_stream(f"yiv ~ {rhs} | " + ("g | " if fe else "") + "d ~ z1 + z2",
                           str(path), **options)
    else:
        fit = feglm_stream(f"yb ~ {rhs}", str(path), task, **options)
    t_fit = time.perf_counter() - t0
    coefs = fit.coef()
    out = packed(0.0, t_fit, fit.n_obs, list(coefs), list(coefs.values()), list(fit.se))
    fit.cleanup()
    return out


def run_pyfixest(task, fe, path):
    import pandas as pd
    import pyfixest as pf

    xs = covariates(path)
    t0 = time.perf_counter()
    frame = pd.read_parquet(path, columns=variables(task, fe, xs))
    t_read = time.perf_counter() - t0

    rhs = " + ".join(xs)
    t0 = time.perf_counter()
    if task == "ols":
        fit = pf.feols(f"y ~ {rhs}" + (" | g" if fe else ""), data=frame, vcov="iid")
    elif task == "iv":
        fit = pf.feols(f"yiv ~ {rhs} | " + ("g | " if fe else "") + "d ~ z1 + z2",
                       data=frame, vcov="iid")
    else:
        fit = pf.feglm(f"yb ~ {rhs}", data=frame, family=task, vcov="iid")
    t_fit = time.perf_counter() - t0
    coefs, ses = fit.coef(), fit.se()
    return packed(t_read, t_fit, fit._N, list(coefs.index), coefs.values, ses.values)


def run_statsmodels(task, fe, path):
    import pandas as pd
    import statsmodels.api as sm

    xs = covariates(path)
    t0 = time.perf_counter()
    frame = pd.read_parquet(path, columns=variables(task, fe, xs))
    X = sm.add_constant(frame[xs].to_numpy())
    y = frame[OUTCOME[task]].to_numpy()
    t_read = time.perf_counter() - t0

    t0 = time.perf_counter()
    model = {"ols": sm.OLS, "logit": sm.Logit, "probit": sm.Probit}[task]
    fit = model(y, X).fit() if task == "ols" else model(y, X).fit(disp=0)
    t_fit = time.perf_counter() - t0
    return packed(t_read, t_fit, len(y), ["Intercept", *xs], fit.params, fit.bse)


def run_sklearn(task, fe, path):
    import pandas as pd

    xs = covariates(path)
    t0 = time.perf_counter()
    frame = pd.read_parquet(path, columns=variables(task, fe, xs))
    X, y = frame[xs].to_numpy(), frame[OUTCOME[task]].to_numpy()
    t_read = time.perf_counter() - t0

    t0 = time.perf_counter()
    if task == "ols":
        from sklearn.linear_model import LinearRegression
        fit = LinearRegression().fit(X, y)
    else:
        from sklearn.linear_model import LogisticRegression
        fit = LogisticRegression(penalty=None, tol=1e-10, max_iter=1000).fit(X, y)
    t_fit = time.perf_counter() - t0
    return packed(t_read, t_fit, len(y), [*xs, "Intercept"],
                  [*fit.coef_.ravel(), float(fit.intercept_ if task == "ols"
                                             else fit.intercept_[0])])


def run_glum(task, fe, path):
    import pandas as pd
    from glum import GeneralizedLinearRegressor

    xs = covariates(path)
    t0 = time.perf_counter()
    frame = pd.read_parquet(path, columns=variables(task, fe, xs))
    X, y = frame[xs].to_numpy(), frame[OUTCOME[task]].to_numpy()
    t_read = time.perf_counter() - t0

    t0 = time.perf_counter()
    fit = GeneralizedLinearRegressor(family="binomial", alpha=0.0,
                                     gradient_tol=1e-10).fit(X, y)
    t_fit = time.perf_counter() - t0
    return packed(t_read, t_fit, len(y), [*xs, "Intercept"],
                  [*fit.coef_, float(fit.intercept_)])


def demeaned(frame, columns):
    """`columns` less their means within g: what a Polars user would write."""
    import polars as pl
    return frame.with_columns([pl.col(c) - pl.col(c).mean().over("g") for c in columns])


def run_polars_ols(task, fe, path):
    import polars as pl
    import polars_ols  # noqa: F401  (registers the least_squares namespace)

    xs = covariates(path)
    t0 = time.perf_counter()
    frame = pl.read_parquet(path, columns=variables(task, fe, xs))
    t_read = time.perf_counter() - t0

    def ols(target, features):
        out = frame.select(pl.col(target).least_squares.ols(
            *features, add_intercept=not fe, mode="statistics").alias("s")).unnest("s")
        names = [("Intercept" if n == "const" else n) for n in out["feature_names"][0]]
        return (names, out["coefficients"][0].to_list(),
                None if fe else out["standard_errors"][0].to_list())

    t0 = time.perf_counter()
    if fe:
        frame = demeaned(frame, [c for c in frame.columns if c != "g"])
    if task == "ols":
        names, coefs, ses = ols("y", xs)
    else:
        # two stages by hand: the first stage's fitted values stand in for d.
        # The coefficients are 2SLS's; the second stage's standard errors are not
        stage1 = frame.select(pl.col("d").least_squares.ols(
            *xs, "z1", "z2", add_intercept=not fe, mode="predictions").alias("d_hat"))
        frame = frame.with_columns(stage1)
        names, coefs, _ = ols("yiv", ["d_hat", *xs])
        names, ses = ["d" if n == "d_hat" else n for n in names], None
    t_fit = time.perf_counter() - t0
    return packed(t_read, t_fit, frame.height, names, coefs, ses)


def run_polars_ds(task, fe, path):
    import polars as pl
    import polars_ds as pds

    xs = covariates(path)
    t0 = time.perf_counter()
    frame = pl.read_parquet(path, columns=variables(task, fe, xs))
    t_read = time.perf_counter() - t0

    t0 = time.perf_counter()
    if task == "ols":
        out = frame.select(pds.lin_reg_report(*xs, target="y", add_bias=True)
                           .alias("r")).unnest("r")
        names = ["Intercept" if f == "__bias__" else f for f in out["features"]]
        result = (names, out["beta"], out["std_err"])
    else:
        coefs = frame.select(pds.logistic_reg(*xs, target="yb", add_bias=True, tol=1e-10,
                                              max_iter=1000).alias("r"))["r"][0]
        result = ([*xs, "Intercept"], coefs, None)
    t_fit = time.perf_counter() - t0
    return packed(t_read, t_fit, frame.height, *result)


def run_numpy(task, fe, path):
    """The normal equations in NumPy: about the least work an OLS or 2SLS fit
    can do, as a floor for the others (with the classical standard errors)."""
    import numpy as np
    import polars as pl

    xs = covariates(path)
    t0 = time.perf_counter()
    frame = pl.read_parquet(path, columns=variables(task, fe, xs))
    ones = np.ones(frame.height)
    X = np.column_stack([ones, frame.select(xs).to_numpy()])
    y = frame[OUTCOME[task]].to_numpy()
    if task == "iv":
        endog = frame["d"].to_numpy()
        Z = np.column_stack([X, frame["z1"].to_numpy(), frame["z2"].to_numpy()])
    t_read = time.perf_counter() - t0

    t0 = time.perf_counter()
    if task == "ols":
        bread = np.linalg.inv(X.T @ X)
        beta = bread @ (X.T @ y)
        resid = y - X @ beta
        names = ["Intercept", *xs]
    else:
        W = np.column_stack([X, endog])
        fitted = Z @ np.linalg.solve(Z.T @ Z, Z.T @ W)
        bread = np.linalg.inv(fitted.T @ fitted)
        beta = bread @ (fitted.T @ y)
        resid = y - W @ beta
        names = ["Intercept", *xs, "d"]
    ses = np.sqrt(np.diag(bread) * (resid @ resid) / (len(y) - len(beta)))
    t_fit = time.perf_counter() - t0
    return packed(t_read, t_fit, len(y), names, beta, ses)


def run_linearmodels(task, fe, path):
    import pandas as pd

    xs = covariates(path)
    t0 = time.perf_counter()
    frame = pd.read_parquet(path, columns=variables(task, fe, xs))
    y = frame[OUTCOME[task]]
    exog = frame[xs]
    if task == "iv" and fe:
        # linearmodels absorbs fixed effects in least squares only: demean first
        cols = [c for c in frame.columns if c != "g"]
        frame[cols] = frame[cols] - frame.groupby("g")[cols].transform("mean")
        y, exog = frame[OUTCOME[task]], frame[xs]
    elif not fe:
        exog = exog.assign(Intercept=1.0)
    t_read = time.perf_counter() - t0

    t0 = time.perf_counter()
    if task == "iv":
        from linearmodels.iv import IV2SLS
        fit = IV2SLS(y, exog, frame[["d"]], frame[["z1", "z2"]]).fit(cov_type="unadjusted")
    else:
        from linearmodels.iv.absorbing import AbsorbingLS
        fit = AbsorbingLS(y, exog, absorb=frame[["g"]].astype("category")).fit(
            cov_type="unadjusted")
    t_fit = time.perf_counter() - t0
    ses = None if (task == "iv" and fe) else fit.std_errors.values
    return packed(t_read, t_fit, len(y), list(fit.params.index), fit.params.values, ses)


RUNNERS = {"pyfixest": run_pyfixest, "statsmodels": run_statsmodels,
           "sklearn": run_sklearn, "glum": run_glum, "polars_ols": run_polars_ols,
           "polars_ds": run_polars_ds, "numpy": run_numpy,
           "linearmodels": run_linearmodels}


def runner(kind, variant, path, workdir):
    task, library, fe = kind.split(":")
    fe = int(fe)
    if library == "hdfe":
        return run_hdfe(task, fe, path, workdir, device=variant)
    return RUNNERS[library](task, fe, path)


# --------------------------------------------------------------------------
# the parent process
# --------------------------------------------------------------------------

def external(task, library, fe, path):
    """The command for R (fixest, or base lm and glm), or None."""
    xs = covariates(path)
    return external_command(
        "regr", path=path, task=task, library=library, fe=fe,
        outcome=OUTCOME[task], covariates=xs)


def agreement(task, reports, stored):
    """The largest relative difference between configurations' coefficients,
    among those just measured and `stored` rows of the same task."""
    ok = [r["coefs"] for r in reports.values() if r["status"] == "ok"] + list(stored)
    if len(ok) < 2:
        return 0.0
    base, keys = ok[0], KEYS[task]
    return max(abs(c[k] - base[k]) / max(abs(base[k]), 1e-12)
               for c in ok[1:] for k in keys)


def stored_coefs(csv_path, task, n_rows, skip):
    import csv
    if not Path(csv_path).exists():
        return []
    with open(csv_path, newline="") as handle:
        return [{k: float(r[f"coef_{k}"]) for k in KEYS[task]}
                for r in csv.DictReader(handle)
                if int(r["n_rows"]) == n_rows and r["status"] == "ok"
                and r["configuration"] not in skip and r.get(f"coef_{KEYS[task][0]}")]


def task_of(task, label):
    """The model of a configuration: the script's task, or for the binary
    outcome benchmark, logit or probit as the label says."""
    if task != "binary":
        return task
    return "probit" if "probit" in label else "logit"


def main(task, script, configs, default_sizes, doc, csv_name):
    parser = argparse.ArgumentParser(description=doc)
    parser.add_argument("sizes", nargs="*", type=int, default=default_sizes,
                        help="number of rows in each data set")
    parser.add_argument("--only", nargs="+", metavar="CONFIG",
                        help="run only these configurations (by label)")
    parser.add_argument("--covariates", type=int, default=N_COVARIATES,
                        help="number of covariates (default %(default)s)")
    parser.add_argument("--child", nargs=4,
                        metavar=("KIND", "VARIANT", "PATH", "WORKDIR"),
                        help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.child:
        run_child(runner, *args.child)
        return

    csv_path = RESULTS_DIR / csv_name
    key = ("n_rows", "configuration")
    chosen = {k: v for k, v in configs.items() if not args.only or k in args.only}
    write_machine_info(RESULTS_DIR / "machine.json")
    workdir = DATA_DIR / f"work_{task}"
    for n_rows in args.sizes:
        path = general_data_path(n_rows, args.covariates)
        print(f"\n=== {n_rows:,} rows, {args.covariates} covariates "
              f"({path.stat().st_size / 1e6:.0f} MB on disk) ===", flush=True)

        todo = [label for label in chosen
                if not is_done(csv_path, {"n_rows": n_rows, "configuration": label},
                               key=key)]
        warm = {(task_of(task, label), chosen[label][1])
                for label in todo if chosen[label][0] == "hdfe"}
        for warm_task, fe in sorted(warm):
            # compile the numba kernels before timing anything
            measure(script, f"{warm_task}:hdfe:{fe}", "cpu", general_data_path(100_000),
                    workdir)

        reports = {}
        for order, (label, (library, fe)) in enumerate(configs.items()):
            if label not in chosen:
                continue
            if label not in todo:
                print(f"  {label:<40}measured before, kept", flush=True)
                continue
            model = task_of(task, label)
            command = None
            if library in ("fixest", "base"):
                command = external(model, library, fe, path)
                if command is None:
                    print(f"  {label:<40}{external_missing('regr')}", flush=True)
                    continue
            report = measure(script, f"{model}:{library}:{fe}", "cpu", path, workdir,
                             timeout=TIMEOUT_S, command=command)
            reports[label] = report
            row = {"n_rows": n_rows, "n_covariates": args.covariates, "task": model,
                   "fixed_effects": fe, "configuration": label, "order": order,
                   "status": report["status"]}
            if report["status"] == "ok":
                row.update(
                    wall_s=round(report["t_read"] + report["t_fit"], 3),
                    read_s=round(report["t_read"], 3),
                    fit_s=round(report["t_fit"], 3),
                    peak_memory_mb=round(report["peak_rss_mb"], 1),
                    import_memory_mb=round(report["baseline_rss_mb"], 1),
                    n_obs=report["n_obs"],
                    **{f"coef_{k}": repr(report["coefs"][k]) for k in KEYS[model]},
                    **{f"se_{k}": repr(report["ses"][k])
                       for k in KEYS[model] if k in report.get("ses", {})})
                print(f"  {label:<40}{row['wall_s']:8.2f} s   peak "
                      f"{row['peak_memory_mb']:8,.0f} MB", flush=True)
            else:
                print(f"  {label:<40}{report['status']}", flush=True)
            upsert_csv(csv_path, [row], key=key)

        # the estimates must agree within each model and number of fixed effects
        for model, fe in sorted({(task_of(task, label), c[1])
                                 for label, c in chosen.items()}):
            same = {label for label, c in configs.items()
                    if task_of(task, label) == model and c[1] == fe}
            group = {label: r for label, r in reports.items() if label in same}
            others = set(configs) - same
            worst = agreement(model, group, stored_coefs(
                csv_path, model, n_rows, set(group) | others))
            verdict = "agree" if worst < TOLERANCE[model] else "DISAGREE"
            print(f"  {model}, {fe} fixed effects: estimates {verdict}: largest "
                  f"relative difference {worst:.1e}", flush=True)

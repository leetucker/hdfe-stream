"""Leaving out a whole worker-firm match, following LeaveOutTwoWay.

LeaveOutTwoWay's `leave_out_KSS` (Saggio), VarianceComponentsHDFE.jl and xhdfe
all leave out a match by default rather than a single observation. Leaving out
one year of a spell treats the worker's other years at the same firm as
independent of it; with errors correlated within a spell, as earnings errors
usually are, the leave-one-observation-out variance estimate is then biased.
Leaving out the whole match is not.

The reference implements it in three steps, and this module follows them:

  1. Partial out everything except the worker and firm effects, using the full
     model: covariates, and any further fixed effects. Here that is
     y~ = fe_worker + fe_firm + resid from the row-level fit.
  2. Collapse to one row per match: the mean of y~, weighted by the spell's
     length (its total weight, with user weights).
  3. Run KSS on the weighted collapsed regression. In the square-root-weight
     metric the collapsed row's leverage is the match's, so this *is* leaving
     out a match, and the rest of the machinery -- leverages, the bias term,
     standard errors, the weak-identification layer -- applies unchanged. The
     quadratic forms stay person-year moments, divided by person-years - 1.

Two details follow the reference exactly because they change numbers:

  * sigma2 is centered in the transformed metric, (sqrt(w) ybar - its mean over
    matches) times the transformed leave-out residual (`leave_out_KSS.m`). This
    is the default, `centering="reference"`; `centering="weighted"` centers ybar
    at its weighted mean first. Both are unbiased, but with the outcome far
    from zero the reference's is several times more variable (see
    docs/kss_methodological_differences.md, 3.4);
  * a stayer's single match cannot be left out -- its leverage is 1 -- so it
    gets `sigma_for_stayers.m`: its own person-year residuals from the collapsed
    fit, with leverage 1/T_i, as (y_it - ybar) e_it / (1 - 1/T_i), averaged over
    the match. Under errors correlated within the spell this targets a slightly
    different variance than the movers' match-level estimate; that is the
    reference's choice, kept here and recorded in
    docs/kss_methodological_differences.md.

Standard errors: the reference implementation is LeaveOutTwoWay's
`leave_out_COMPLETE` with leave_out_level='matches', an option its
documentation marks as beta. xhdfe ports it, and the prototype was validated
against that port. What is kept:
  * the kernel: the collapsed rows' own zero-diagonal kernel is exactly the
    reference's person-year block kernel;
  * b = 0 on stayers' matches;
  * no standard error for var(alpha);
  * by default (se_variance="person_year"), its person-year error variance.
What differs, and why, is in docs/kss_methodological_differences.md 5.7: the
centered outcome, the smoother, no user weights with the default variance
source, no q = 1 interval for the covariance, and the "match" variance source,
which goes beyond the reference.
"""

from __future__ import annotations

from pathlib import Path

#: column names in the match table that cannot clash with a user's
MATCH_WEIGHT = "_match_weight"
MATCH_ROWS = "_match_rows"


def build_match_table(row_fit, alpha, psi, path):
    """Collapse a row-level fit's partialled-out outcome to one row per match.

    Writes Parquet to `path` in one streaming pass and returns the number of
    person-year rows behind it. The outcome keeps the fit's name, so the
    collapsed fit reads naturally.
    """
    import polars as pl

    y = row_fit.depvar
    rows = row_fit.resid()
    names = rows.collect_schema().names()
    weight = pl.col("weights") if "weights" in names else pl.lit(1.0)
    partialled = pl.col(f"fe_{alpha}") + pl.col(f"fe_{psi}") + pl.col("resid")
    table = (rows.select(pl.col(alpha), pl.col(psi), partialled.alias("_y"),
                         weight.alias("_w"))
             .group_by(alpha, psi)
             .agg((pl.col("_w") * pl.col("_y")).sum().alias("_wy"),
                  pl.col("_w").sum().alias(MATCH_WEIGHT),
                  pl.len().alias(MATCH_ROWS))
             .select(alpha, psi,
                     (pl.col("_wy") / pl.col(MATCH_WEIGHT)).alias(y),
                     MATCH_WEIGHT, MATCH_ROWS))
    table.sink_parquet(path)
    return int(pl.scan_parquet(path).select(pl.col(MATCH_ROWS).sum())
               .collect().item())


def stayer_sigma2(row_fit, match_fit, alpha, psi):
    """`sigma_for_stayers.m`, for every match of a worker with only one.

    Returns a Polars frame of (alpha, psi, sigma2) for the stayers' matches, on
    the scale the collapsed rows' sigma2 is on: the match row's error variance
    in the square-root-weight metric, sum w^2 Var(e_it) / sum w -- a plain mean
    of the person-year terms when unweighted.
    """
    import polars as pl

    rows = row_fit.resid()
    names = rows.collect_schema().names()
    weight = pl.col("weights") if "weights" in names else pl.lit(1.0)
    y_tilde = (pl.col(f"fe_{alpha}") + pl.col(f"fe_{psi}")
               + pl.col("resid")).alias("_y")
    person = rows.select(pl.col(alpha), pl.col(psi), y_tilde, weight.alias("_w"))

    # the collapsed fit's effects, read back onto the person-year rows
    effects = (match_fit.resid()
               .select(alpha, psi, f"fe_{alpha}", f"fe_{psi}")
               .with_columns((pl.col(f"fe_{alpha}") + pl.col(f"fe_{psi}"))
                             .alias("_fitted"))
               .select(alpha, psi, "_fitted"))
    firms_per_worker = (person.group_by(alpha)
                        .agg(pl.col(psi).n_unique().alias("_n_firms"),
                             pl.col("_w").sum().alias("_worker_w")))
    totals = person.select((pl.col("_w") * pl.col("_y")).sum().alias("wy"),
                           pl.col("_w").sum().alias("w")).collect(
                               engine="streaming").row(0, named=True)
    y_mean = totals["wy"] / totals["w"]

    return (person
            .join(firms_per_worker, on=alpha)
            .filter(pl.col("_n_firms") == 1)
            .join(effects, on=[alpha, psi])
            .with_columns(
                ((pl.col("_y") - y_mean) * (pl.col("_y") - pl.col("_fitted"))
                 / (1.0 - pl.col("_w") / pl.col("_worker_w"))).alias("_r"))
            .group_by(alpha, psi)
            .agg(((pl.col("_w") ** 2 * pl.col("_r")).sum()
                  / pl.col("_w").sum()).alias("sigma2"))
            .collect(engine="streaming"))


def within_match_variance(row_fit, alpha, psi):
    """(1/T) sum over the match of (y~_it - ybar_m)^2, per match.

    The within-match part of LeaveOutTwoWay's person-year variance estimate for
    the standard errors (leave_out_COMPLETE's sigma_i = y_i eta_h,i, averaged
    over the match): in collapsed terms that estimate is this plus the
    collapsed leave-match-out sigma2 over T, and exactly this for a stayer.
    Unweighted fits only, as in the reference.
    """
    import polars as pl

    y_tilde = (pl.col(f"fe_{alpha}") + pl.col(f"fe_{psi}")
               + pl.col("resid")).alias("_y")
    rows = row_fit.resid().select(pl.col(alpha), pl.col(psi), y_tilde)
    means = rows.group_by(alpha, psi).agg(pl.col("_y").mean().alias("_m"))
    return (rows.join(means, on=[alpha, psi])
            .group_by(alpha, psi)
            .agg(((pl.col("_y") - pl.col("_m")) ** 2).mean().alias("within"))
            .collect(engine="streaming"))


def leave_out_match(row_fit, alpha, psi, workdir, n_draws, seed, block,
                    se, se_draws, se_trace, diagnose, diagnose_draws,
                    weak_interval, confidence, verbose, logger,
                    centering="reference", se_variance="person_year"):
    """KSS leaving out a match, from a row-level fit of the user's model.

    Returns the `LeaveOutComponents` of the collapsed fit, with that fit
    attached as `.match_fit` and the row-level one as `.fit`.
    """
    from .api import feols_stream
    from .report import _log

    if se and se_variance == "person_year" and row_fit.weights is not None:
        raise ValueError(
            "se_variance='person_year' is LeaveOutTwoWay's, which has no user "
            "weights; use se_variance='match' with a weighted fit")
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    table = workdir / "matches.parquet"
    person_years = build_match_table(row_fit, alpha, psi, table)
    _log(verbose, f"leave-out: collapsed {person_years:,} person-years to "
                  "worker-firm matches, to leave out a match at a time", logger)

    # every match stays: a stayer's one match looks like a singleton here, and
    # is handled by `stayer_sigma2`
    match_fit = feols_stream(f"{row_fit.depvar} ~ 1 | {alpha} + {psi}",
                             str(table), workdir=workdir / "match_fit",
                             weights=MATCH_WEIGHT, keep_intermediates=True,
                             stream=alpha, fixef_rm="none", verbose=verbose,
                             logger=logger)
    estimator = match_fit._estimator
    estimator.reload_intermediates()
    estimator._match_person_years = person_years
    estimator._leave_out_level = "match"

    stayers = stayer_sigma2(row_fit, match_fit, alpha, psi)
    within = (within_match_variance(row_fit, alpha, psi)
              if se and se_variance == "person_year" else None)
    components = estimator.leave_out_components(
        match_fit, n_draws=n_draws, block=block, seed=seed, psi=psi,
        stayers="within_match", stayer_sigma2=stayers,
        centering=centering, se_variance=se_variance, match_within=within,
        se=se, se_draws=se_draws, se_trace=se_trace, diagnose=diagnose,
        diagnose_draws=diagnose_draws, weak_interval=weak_interval,
        confidence=confidence)
    # the components are person-year moments, so that is the sample size;
    # n_movers stays the count of movers' matches, the rows left out
    components.match_fit = match_fit
    components.n_obs = person_years
    components.diagnostics["n_matches"] = match_fit.n_obs
    components.diagnostics["leave_out"] = "match"
    components.diagnostics["centering"] = centering
    if se:
        components.diagnostics["se_variance"] = se_variance
    components.diagnostics["person_years"] = person_years
    return components

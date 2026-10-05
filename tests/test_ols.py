"""Core OLS: hdfe_stream against pyfixest, across solvers and FE sets.

Uses the low-level `StreamingHDFE` interface so the formula front end is not
in the picture; `test_formula.py` covers that. Clustering here is one-way only
-- multi-way (Cameron-Gelbach-Miller) vcovs live in `test_vcov.py`.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from conftest import (FE_DOF_PF, TOL_BETA, TOL_RESID, TOL_SE, TOL_STAT,
                      cluster_spec, ensure_interactions, pf_feols, rel, scaled)

pytest.importorskip("pyfixest")

from hdfe_stream import StreamingHDFE  # noqa: E402

X = ["age_squared", "age_cubed"]
SOLVERS = ["explicit", "stream_cg", "within"]
FE_SETS = [
    ["worker_id", "firm_id", "year"],
    ["worker_id", "firm_id"],
    ["worker_id", "firm_id^year"],
]


def fml_for(fe):
    return f"log_earn ~ {' + '.join(X)} | {' + '.join(fe)}"


def fit_stream(panel, fe, workdir, solver="auto", fe_dof=FE_DOF_PF, **options):
    """A streaming fit with all the vcov flavors computed in one pass.

    Small `batch_rows` and several buckets on purpose: it forces the
    multi-batch and multi-bucket code paths that a large real dataset hits.
    """
    options.setdefault("batch_rows", 5_000)
    options.setdefault("n_buckets", 3)
    est = StreamingHDFE("log_earn", X, fe, workdir=workdir, solver=solver,
                        keep=["year"], tol=1e-11, verbose=False, **options)
    return est.fit(panel.src, vcov="iid", cluster=tuple(fe), fe_dof=fe_dof)


@pytest.fixture(scope="module")
def references(akm):
    """One pyfixest reference per FE set, fitted once for the whole module.

    pyfixest recomputes the vcov in place via `.vcov(...)`, so each test sets
    the flavor it needs before reading standard errors.
    """
    ensure_interactions(akm, [d for fe in FE_SETS for d in fe])
    return {tuple(fe): pf_feols(fml_for(fe), akm) for fe in FE_SETS}


@pytest.fixture(scope="module")
def fits(akm, tmp_path_factory):
    """One streaming fit per (FE set, solver), fitted once for the module.

    `outputs="keep"` so the residual and fixed-effect files outlive the fit
    and can be read by several tests.
    """
    out = {}
    base = tmp_path_factory.mktemp("ols")
    for fe in FE_SETS:
        for solver in SOLVERS:
            tag = "_".join(fe).replace("^", "x")
            out[(tuple(fe), solver)] = fit_stream(
                akm, fe, base / f"{tag}_{solver}", solver=solver, outputs="keep")
    yield out
    for r in out.values():
        r.cleanup()


@pytest.mark.parametrize("fe", FE_SETS, ids=lambda fe: "+".join(fe))
@pytest.mark.parametrize("solver", SOLVERS)
def test_coefficients_match_pyfixest(fits, references, fe, solver):
    res = fits[(tuple(fe), solver)]
    ref = references[tuple(fe)]
    assert res.coefnames == X
    assert rel(res.beta, ref.coef().to_numpy()) < TOL_BETA


@pytest.mark.parametrize("fe", FE_SETS, ids=lambda fe: "+".join(fe))
@pytest.mark.parametrize("solver", SOLVERS)
def test_iid_and_hetero_se_match_pyfixest(fits, references, fe, solver):
    res = fits[(tuple(fe), solver)]
    ref = references[tuple(fe)]
    for key, spec in [("iid", "iid"), ("hetero", "hetero")]:
        ref.vcov(spec)
        assert rel(res.with_vcov(key).se, ref.se().to_numpy()) < TOL_SE, key


@pytest.mark.parametrize("fe", FE_SETS, ids=lambda fe: "+".join(fe))
def test_one_way_clustered_se_match_pyfixest(fits, references, fe):
    """One-way CRV1 on each FE dimension in turn.

    For the `firm_id^year` set this clusters on the interacted dimension, i.e.
    one cluster per firm-year cell -- still a single clustering dimension.
    """
    res = fits[(tuple(fe), "explicit")]
    ref = references[tuple(fe)]
    for dim in fe:
        ref.vcov({"CRV1": cluster_spec(dim)})
        got = res.with_vcov({"CRV1": dim})
        assert rel(got.se, ref.se().to_numpy()) < TOL_SE, dim
        assert got.df_t == pytest.approx(ref._df_t, abs=0.5), dim


@pytest.mark.parametrize("fe", FE_SETS, ids=lambda fe: "+".join(fe))
def test_solvers_agree_with_each_other(fits, fe):
    """The three solvers solve the same linear system, so they should agree
    far more tightly than any of them agrees with pyfixest."""
    betas = [fits[(tuple(fe), s)].beta for s in SOLVERS]
    for other in betas[1:]:
        assert rel(other, betas[0]) < 1e-9


@pytest.mark.parametrize("fe", FE_SETS, ids=lambda fe: "+".join(fe))
def test_residuals_and_fixed_effects_match_pyfixest(fits, references, akm, fe):
    """Row-level check: residuals and the sum of fitted FE, joined back to the
    input rows. The streaming residual file is in bucket order, so the join is
    on the worker-year key rather than on position."""
    res = fits[(tuple(fe), "explicit")]
    ref = references[tuple(fe)]
    fe_cols = [f"fe_{d}" for d in fe]
    keys = ["worker_id", "year"]

    mine = res.resid().select(*keys, "resid", *fe_cols).collect()
    beta = ref.coef().to_numpy()
    # pyfixest's rows are the ones it kept (singletons are dropped)
    used = akm.pandas.loc[ref._data.index]
    theirs = pl.DataFrame({
        **{k: used[k].to_numpy() for k in keys},
        "resid_ref": ref.resid(),
        "fesum_ref": ref.predict() - used[X].to_numpy() @ beta,
    })
    joined = mine.join(theirs, on=keys)
    assert joined.height == len(used) == res.n_obs  # the key is unique per row

    assert scaled(joined["resid"], joined["resid_ref"]) < TOL_RESID
    fesum = sum(joined[c] for c in fe_cols)
    assert scaled(fesum, joined["fesum_ref"]) < TOL_RESID


@pytest.mark.parametrize("fe", FE_SETS, ids=lambda fe: "+".join(fe))
def test_fit_statistics_match_pyfixest(fits, references, fe):
    res = fits[(tuple(fe), "explicit")]
    ref = references[tuple(fe)]
    ref.vcov("iid")
    assert res.n_obs == ref._N
    for attr, ref_attr in [("r2", "_r2"), ("adj_r2", "_adj_r2"),
                           ("r2_within", "_r2_within"), ("rmse", "_rmse")]:
        assert rel(getattr(res, attr), getattr(ref, ref_attr)) < TOL_STAT, attr


def test_exact_fe_dof_differs_from_pyfixest_only_through_dof(akm, workdir, references):
    """hdfe_stream's default `fe_dof="exact"` counts the fixed-effect
    parameters that are actually identified, which pyfixest does not. That must
    leave the point estimates untouched and rescale every standard error by the
    same factor -- a pure degrees-of-freedom difference, not a different fit.
    """
    fe = FE_SETS[0]
    ref = references[tuple(fe)]
    ref.vcov("iid")
    exact = fit_stream(akm, fe, workdir / "exact", fe_dof="exact")
    loose = fit_stream(akm, fe, workdir / "pf", fe_dof=FE_DOF_PF)

    assert rel(exact.beta, loose.beta) < 1e-12
    assert rel(loose.se, ref.se().to_numpy()) < TOL_SE

    ratio = exact.se / ref.se().to_numpy()
    assert np.ptp(ratio) < 1e-9, f"not a constant rescaling: {ratio}"
    assert exact.k_fe >= loose.k_fe


def _three_way_panel(seed=3, n=3000):
    """Three overlapping interactions, as in an event study with
    unit-by-time, unit-by-group and group-by-season effects: irregular
    enough that the redundant levels depend on which pair is counted."""
    rng = np.random.default_rng(seed)
    t = rng.integers(0, 12, n)
    df = pl.DataFrame({"u": rng.integers(0, 40, n), "g": rng.integers(0, 10, n),
                       "t": t, "s": t % 4, "x": rng.normal(size=n)})
    return df.with_columns(y=pl.col("x") + pl.lit(rng.normal(size=n)))


def _fe_rank(df, fe):
    """Identified FE parameters: the rank of the stacked indicator matrices."""
    blocks = [np.eye(n)[codes] for n, codes in
              ((int(c.max()) + 1, c) for c in
               (df.select(pl.struct(f.split("^")).rank("dense") - 1).to_series().to_numpy()
                for f in fe))]
    return np.linalg.matrix_rank(np.hstack(blocks))


def test_exact_fe_dof_is_conservative_and_order_free():
    """With three fixed effects `fe_dof="exact"` gives the same count for
    every order of the formula and every streamed dimension, and never fewer
    parameters than are identified; with two it is the identified count."""
    from itertools import permutations

    from hdfe_stream import feols_stream

    df = _three_way_panel()
    fe = ["t^u", "g^u", "s^g"]
    counts = set()
    for order in permutations(fe):
        for stream in (None, order[0]):
            with feols_stream(f"y ~ x | {' + '.join(order)}", df, stream=stream,
                              verbose=False) as res:
                counts.add(res.k_fe)
                sample = res.sample().filter("in_sample").collect()
    k_fe, = counts
    rank = _fe_rank(sample, fe)
    with feols_stream(f"y ~ x | {' + '.join(fe)}", df, fe_dof=FE_DOF_PF,
                      verbose=False) as loose:
        assert rank <= k_fe < loose.k_fe

    with feols_stream("y ~ x | t^u + g^u", df, verbose=False) as two:
        sample = two.sample().filter("in_sample").collect()
        assert two.k_fe == _fe_rank(sample, ["t^u", "g^u"])


def test_exact_fe_dof_skips_pairs_nested_in_the_clusters():
    """Clustered by `g` with every fixed effect nested in it, all their
    levels leave K and only the intercept stays, as in pyfixest: the
    components of a pair of nested fixed effects are not subtracted again."""
    from hdfe_stream import feols_stream

    df = _three_way_panel()
    with feols_stream("y ~ x | g^u + s^g", df, vcov={"CRV1": "g"}, verbose=False) as exact, \
         feols_stream("y ~ x | g^u + s^g", df, vcov={"CRV1": "g"}, fe_dof=FE_DOF_PF,
                      verbose=False) as loose:
        assert exact.k_fe < loose.k_fe          # the pair's components still count for iid
        np.testing.assert_allclose(exact.se, loose.se, rtol=1e-12)


@pytest.mark.parametrize("assembly", ["cells", "rows"])
def test_assembly_paths_agree(akm, workdir, references, assembly):
    """The cross-products can be accumulated from the cell table or from a
    second row pass; both must give the same normal equations."""
    fe = FE_SETS[0]
    res = fit_stream(akm, fe, workdir / assembly, assembly=assembly)
    ref = references[tuple(fe)]
    ref.vcov("iid")
    assert res.diagnostics["assembly"] == assembly
    assert rel(res.beta, ref.coef().to_numpy()) < TOL_BETA
    assert rel(res.se, ref.se().to_numpy()) < TOL_SE


def test_internal_consistency_diagnostics(fits):
    """The fit reports two independent computations of the residual sum of
    squares and of the bread matrix; they must agree."""
    for (fe, solver), res in fits.items():
        d = res.diagnostics
        assert rel(d["rss_assembled"], d["rss_rows"]) < 1e-9, (fe, solver)
        assert d["bread_rel_diff"] < 1e-9, (fe, solver)
        assert res.n_components == 1, (fe, solver)    # the panel is connected


def test_solver_converged(fits):
    for (fe, solver), res in fits.items():
        info = res.solver_info
        assert info["converged"], (fe, solver, info)
        if solver != "within":
            assert info["solver"] == solver

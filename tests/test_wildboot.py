"""The wild cluster bootstrap (`bootcluster=`, `wildboottest`, `rwolf`,
`wyoung`).

Two references. The first needs nothing extra: the model refitted for each
draw with the fixed effects as dummy variables, on the bootstrap sample the
draw defines, which is the bootstrap by definition. Our bootstrap t statistics
must equal those, draw by draw, for every kind of cluster term (the streamed
dimension, another fixed effect, any other column), weighted or not,
restricted or not. The second is pyfixest's `wildboottest`, `rwolf` and
`wyoung` (which need the optional `wildboottest` package): same seed, same
draws, so the same p-values.
"""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl
import pytest

from conftest import FE_DOF_PF

from hdfe_stream import StreamingGLM, feols_stream, rwolf, wyoung  # noqa: E402
from hdfe_stream.wildboot import _draws, _Test  # noqa: E402

G_STATES = 12


@pytest.fixture(scope="module")
def frame():
    """States, firms nested in states, a few years, and outcomes with a
    state-level shock (so the clustering matters)."""
    rng = np.random.default_rng(0)
    n = 3000
    state = rng.integers(0, G_STATES, n)
    firm = state * 15 + rng.integers(0, 15, n)
    year = rng.integers(0, 5, n)
    x = rng.normal(size=n) + 0.3 * state
    shock = rng.normal(size=G_STATES)[state]
    return pl.DataFrame({
        "y": 0.05 * x + 0.1 * state + rng.normal(size=n) + shock,
        "y2": rng.normal(size=n) + shock,
        "y3": 0.08 * x + rng.normal(size=n) + 0.5 * shock,
        "x": x, "x2": rng.normal(size=n), "w": rng.uniform(0.5, 2, n),
        "state": state, "firm": firm, "year": year,
        "region": (state // 4).astype(str)})


def fit(fml, frame, **kwargs):
    kwargs = {"fe_dof": FE_DOF_PF, "verbose": False, "tol": 1e-12, **kwargs}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return feols_stream(fml, frame, **kwargs)


def brute_tstar(frame, covs, fes, cluster, j, V, adj, weights=None, impose_null=True):
    """Bootstrap t statistics by refitting, with dummies for the fixed
    effects: y* = Z g_r + v_g u_r (restricted) or Z g + v_g u (not), then the
    CRV1 t statistic of coefficient j, scaled by `adj`. V's rows are the
    clusters in sorted order."""
    cols = [frame[c].to_numpy().astype(float) for c in covs]
    for f in fes:
        cols += [(frame[f] == lv).cast(float).to_numpy() for lv in frame[f].unique().sort()]
    Z = np.column_stack(cols)
    y = frame["y"].to_numpy()
    sw = np.sqrt(frame[weights].to_numpy()) if weights else np.ones(len(y))
    Z, y = Z * sw[:, None], y * sw
    _, g = np.unique(frame[cluster].to_numpy(), return_inverse=True)
    ZtZi = np.linalg.pinv(Z.T @ Z)

    def t_of(yy, center=0.0):
        b = ZtZi @ Z.T @ yy
        u = yy - Z @ b
        sc = np.array([(ZtZi @ Z[g == c].T @ u[g == c])[j] for c in range(g.max() + 1)])
        return (b[j] - center) / np.sqrt(adj * np.sum(sc ** 2))

    base = np.delete(Z, j, axis=1) if impose_null else Z
    coef = np.linalg.lstsq(base, y, rcond=None)[0]
    u0 = y - base @ coef
    # the unrestricted bootstrap's t statistics are centered at the estimate
    center = 0.0 if impose_null else coef[j]
    return t_of(y), np.array([t_of(base @ coef + V[g, b] * u0, center)
                              for b in range(V.shape[1])])


@pytest.mark.parametrize("fml, fes, cluster, options", [
    # the cluster is another column ("extra"), the streamed dimension
    # ("seg"), another fixed effect ("fe"), or a string column
    ("y ~ x + x2 + C(year) | firm", ["firm"], "state", {}),
    ("y ~ x + x2 | state", ["state"], "state", {}),
    ("y ~ x + x2 | firm + state", ["firm", "state"], "state", {"stream": "firm"}),
    ("y ~ x + x2 | state", ["state"], "region", {}),
    ("y ~ x + x2 + C(year) | firm", ["firm"], "state", {"weights": "w"}),
])
@pytest.mark.parametrize("weights_type", ["rademacher", "mammen", "webb"])
@pytest.mark.parametrize("impose_null", [True, False])
def test_matches_refitting(frame, fml, fes, cluster, options, weights_type, impose_null):
    res = fit(fml, frame, vcov={"CRV1": cluster}, bootcluster=cluster, **options)
    with res:
        G = res._boot["levels"].height
        V = next(_draws(weights_type, G, 60, 4))
        ours = _Test(res, 0, impose_null).tstar(V)
        covs = ["x", "x2"]
        if "C(year)" in fml:
            frame_ = frame.with_columns([(pl.col("year") == t).cast(float).alias(f"yr{t}")
                                         for t in range(1, 5)])
            covs += [f"yr{t}" for t in range(1, 5)]
        else:
            frame_ = frame
        t, theirs = brute_tstar(frame_, covs, fes, cluster, 0, V, res._boot["adj"],
                                weights=options.get("weights"), impose_null=impose_null)
        assert np.max(np.abs(ours - theirs)) < 1e-8
        out = res.wildboottest("x", reps=60, seed=4, weights_type=weights_type,
                               impose_null=impose_null)
        assert out["t value"] == pytest.approx(t, rel=1e-9)
        # the t statistic is the one tidy reports under CRV1 by the clusters
        assert out["t value"] == pytest.approx(res.tidy()["t value"][0], rel=1e-9)
        assert out["Pr(>|t|)"] == pytest.approx(np.mean(np.abs(theirs) >= abs(t) * (1 - 1e-9)))


def test_full_enumeration(frame):
    small = frame.filter(pl.col("state") < 6)
    with fit("y ~ x + x2 | firm", small, bootcluster="state") as res:
        with pytest.warns(UserWarning, match="every Rademacher sign pattern"):
            out = res.wildboottest("x", reps=999, seed=0)
        assert out["reps"] == 2 ** 6
        # patterns come in +/- pairs with the same |t*|, so the p-value is a
        # multiple of 2 / 2^G
        assert (out["Pr(>|t|)"] * 2 ** 6) % 2 == pytest.approx(0)


def test_frequency_weights_are_repeated_rows(frame):
    """A row with frequency weight 2 is two rows: same clusters, same sums,
    same draws, same p-value."""
    fw = frame.with_columns(fw=(pl.int_range(pl.len()) % 2 + 1).cast(pl.Int64))
    repeated = fw.select(pl.all().repeat_by("fw").explode()).drop("fw")
    a = fit("y ~ x + x2 | firm", fw, weights="fw", weights_type="fweights",
            bootcluster="state")
    b = fit("y ~ x + x2 | firm", repeated, bootcluster="state")
    with a, b:
        pa, pb = (r.wildboottest("x", reps=499, seed=2) for r in (a, b))
        assert pa["Pr(>|t|)"] == pb["Pr(>|t|)"]
        assert pa["t value"] == pytest.approx(pb["t value"], rel=1e-8)


def test_shared_draws_across_models(frame):
    """Each cluster gets the same weight in every model, keyed by its value:
    a model on a subset of the states sees the full set's weights for its
    states."""
    from hdfe_stream.wildboot import _bootstrap

    full = fit("y ~ x + x2 | firm", frame, bootcluster="state")
    part = fit("y ~ x + x2 | firm", frame.filter(pl.col("state") != 0), bootcluster="state")
    with full, part:
        _, tstar, _ = _bootstrap([(full, 0), (part, 0)], 50, "rademacher", True, 9)
        V = next(_draws("rademacher", G_STATES, 50, 9))
        assert np.allclose(tstar[:, 0], _Test(full, 0, True).tstar(V))
        assert np.allclose(tstar[:, 1], _Test(part, 0, True).tstar(V[1:]))


def test_refusals(frame):
    with fit("y ~ x | firm", frame) as res:
        with pytest.raises(ValueError, match="bootcluster="):
            res.wildboottest("x")
    with pytest.raises(ValueError, match="'year' is not nested.*i\\(year\\)"):
        fit("y ~ x | firm + year", frame, bootcluster="state")
    with pytest.raises(ValueError, match="not nested"):
        fit("y ~ x | firm", frame, bootcluster="year")
    with pytest.raises(ValueError, match="one-way"):
        fit("y ~ x | firm", frame, bootcluster="state+year")
    with pytest.raises(ValueError, match="IV"):
        fit("y ~ x2 | firm | x ~ y2", frame, bootcluster="state")
    with pytest.raises(ValueError, match="GLMs"):
        StreamingGLM("cnt", ["x"], ["firm"], family="poisson", verbose=False).fit(
            frame.with_columns(cnt=pl.lit(1.0)), bootcluster="state")
    a = fit("y ~ x | firm", frame, bootcluster="state")
    b = fit("y ~ x | firm", frame, bootcluster="region")
    with a, b:
        with pytest.raises(ValueError, match="same clustering"):
            rwolf([a, b], "x", reps=99, seed=0)
        with pytest.raises(ValueError, match="not a coefficient"):
            a.wildboottest("x2")


def test_option_off_costs_nothing(frame):
    """Without bootcluster= nothing is kept."""
    with fit("y ~ x | firm", frame, vcov={"CRV1": "state"}) as res:
        assert res._boot is None


# ---------------------------------------------------------------- vs pyfixest

@pytest.fixture(scope="module")
def pf_ok():
    pytest.importorskip("pyfixest")
    pytest.importorskip("wildboottest")


@pytest.mark.parametrize("reps, weights_type, impose_null", [
    (999, "rademacher", True), (999, "webb", True), (999, "mammen", True),
    (999, "rademacher", False), (5000, "rademacher", True)])
def test_wildboottest_matches_pyfixest(frame, pf_ok, reps, weights_type, impose_null):
    import pyfixest as pf
    fml = "y ~ x + x2 + C(year) | firm"
    with fit(fml, frame, vcov={"CRV1": "state"}, bootcluster="state") as res:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ours = res.wildboottest("x", reps=reps, seed=3, weights_type=weights_type,
                                    impose_null=impose_null)
            ref = pf.feols(fml, frame.to_pandas(), vcov={"CRV1": "state"}, fixef_tol=1e-12)
            theirs = ref.wildboottest(param="x", reps=reps, seed=3,
                                      weights_type=weights_type, impose_null=impose_null)
    assert ours["Pr(>|t|)"] == pytest.approx(float(theirs["Pr(>|t|)"]), abs=1e-12)


def test_rwolf_wyoung_match_pyfixest(frame, pf_ok):
    import pyfixest as pf
    fml = "y + y2 + y3 ~ x + x2 | firm"
    multi = fit(fml, frame, vcov={"CRV1": "state"}, bootcluster="state")
    with multi:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ref = pf.feols(fml, frame.to_pandas(), vcov={"CRV1": "state"}, fixef_tol=1e-12)
            rw, rw_ref = rwolf(multi, "x", reps=999, seed=11), pf.rwolf(ref.to_list(), "x",
                                                                       reps=999, seed=11)
            wy, wy_ref = wyoung(multi, "x", reps=999, seed=11), pf.wyoung(ref.to_list(), "x",
                                                                          reps=999, seed=11)
    assert np.allclose(rw["RW Pr(>|t|)"], rw_ref.loc["RW Pr(>|t|)"].to_numpy(dtype=float))
    assert np.allclose(wy["WY Pr(>|t|)"], wy_ref.loc["WY Pr(>|t|)"].to_numpy(dtype=float))

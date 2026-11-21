"""core.sensitivity (Rambachan and Roth's sensitivity analysis for parallel
trends) against the R package HonestDiD 0.2.8, and its own building blocks.

The reference values are in tests/data/honestdid_reference.json.gz, written by
benchmarks/external/honestdid_reference.R. R's two simulated quantities are
made comparable there: its folded-normal quantile is replaced by the exact one
(which is what this package computes), and its least-favorable critical values
are recorded and substituted here, so that every grid point's decision can be
compared. The "stock" section checks R as distributed, within its simulation
error.
"""

import gzip
import json
import sys
import threading
import warnings
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from numpy.testing import assert_allclose
from scipy import optimize

from hdfe_stream.core import _sensitivity_kernels as K
from hdfe_stream.core import sensitivity as S
from hdfe_stream.core._sensitivity_flci import folded_normal_quantile, optimal_flci
from hdfe_stream.core.results_base import RegressionResult

REF = Path(__file__).parent / "data" / "honestdid_reference.json.gz"
_ORIGINAL_SETUP = S._setup_piece


@pytest.fixture(scope="module")
def ref():
    with gzip.open(REF, "rt") as f:
        return json.load(f)


def _case(ref, name):
    c = ref["cases"][name]
    return np.array(c["beta"], float), np.array(c["sigma"], float), int(c["n_pre"])


def _options(job):
    out = dict(restriction=job["restriction"], ci_method=job["method"])
    if job["variant"] == "bias":
        out["bias"] = job["direction"]
    elif job["variant"] == "monotonicity":
        out["monotonicity"] = job["direction"]
    return out


class _RecordedLF:
    """Stand-ins for the least-favorable critical values that return R's. R records
    one per polyhedron of the restriction, in the order this package builds them (the
    largest pre-period change at s = -(n_pre - 1), ..., 0, positive before negative),
    so the piece being prepared is identified by its position; the recorded
    covariance's fingerprint checks the match. (Two polyhedra can have the same
    moments in another order, and so the same fingerprint, but different simulated
    critical values.)"""

    def __init__(self, items, pieces):
        self.items = items
        self.pieces = pieces
        self.local = threading.local()

    def setup(self, original):
        def wrapped(piece, *args, **kwargs):
            idx = [j for j, p in enumerate(self.pieces)
                   if p.A.shape == piece.A.shape and np.array_equal(p.A, piece.A)]
            assert len(idx) == 1
            self.local.index = idx[0]
            return original(piece, *args, **kwargs)
        return wrapped

    def _get(self, SA, XA):
        it = self.items[self.local.index]
        mine = np.array([SA.sum(), np.trace(SA), len(SA)])
        theirs = np.array([it["sum"], it["trace"], it["m"]], float)
        assert np.abs(mine - theirs).max() <= 1e-8 * (np.abs(mine).max() + 1e-300)
        if XA is not None:
            assert abs(XA.sum() - it["xsum"]) <= 1e-8 * (abs(XA.sum()) + 1)
        return float(it["cv"])

    def plain(self, SA, kappa, simulations, seed):
        return self._get(SA, None)

    def arp(self, SA, XA, dual, kappa, simulations, seed):
        return self._get(SA, XA)


def _exact_jobs(ref):
    return [(i, j) for i, job in enumerate(ref["exact"]) for j in range(len(job["results"]))]


def _job_id(ref_jobs):
    return [f"{j['case']}-{j['target']}-{j['restriction']}-{j['variant']}-{j['direction']}-{j['method']}"
            for j in ref_jobs]


def _load():
    with gzip.open(REF, "rt") as f:
        return json.load(f)


_REF = _load() if REF.exists() else {"exact": [], "stock": [], "defaults": []}
_EXACT = _REF["exact"]


@pytest.mark.parametrize("job", _EXACT, ids=_job_id(_EXACT))
def test_matches_r_on_every_grid_point(ref, job, monkeypatch):
    beta, sigma, n_pre = _case(ref, job["case"])
    l = np.atleast_1d(np.array(job["l"], float))
    opts = _options(job)
    for res in job["results"]:
        M = res["M"]
        if "error" in res:
            # R stops ("system is exactly singular") at a grid point where the binding
            # moments' combination has no variance; this package applies R's own rule
            # for that case (reject when eta > 0) and returns an interval
            r = S.trend_sensitivity(beta, sigma, n_pre, M=[M], target=l, **opts)
            lo, hi = r.table.row(0)[1:]
            assert np.isfinite(lo) and np.isfinite(hi) and lo <= hi
            continue
        if job["method"] == "fixed_length":
            vec, half = optimal_flci(sigma, M, n_pre, len(l), l, 0.05)
            assert_allclose(half, res["half"], rtol=2e-6)
            if M == 0:
                # at M = 0 R takes the direction from a conic solve at exactly the
                # smallest variance, where its solver is not accurate: R's direction
                # has a larger variance than the half-length it reports (computed
                # from the variance problem), so only the half-length is comparable
                # (R's direction can even have a smaller variance, by violating the
                # constraint on the weights within its solver's tolerance); this
                # package's is the constrained minimum, which a direct solve confirms
                assert_allclose(vec @ sigma @ vec, _min_variance(sigma, n_pre, l), rtol=1e-9)
                continue
            # R finds the direction with a conic solver: the center is accurate only to
            # about the square root of the solver's tolerance on the variance
            tol = 2e-4 * half
            assert_allclose(vec @ beta - half, res["lb"], atol=tol)
            assert_allclose(vec @ beta + half, res["ub"], atol=tol)
            r = S.trend_sensitivity(beta, sigma, n_pre, M=[M], target=l, **opts)
            assert_allclose(r.table.row(0)[1:], [res["lb"], res["ub"]], atol=tol)
            continue
        pieces = S._pieces(job["restriction"], n_pre, len(l), M, opts.get("bias"),
                           opts.get("monotonicity"))
        rec = _RecordedLF(res["lf"], pieces)
        if job["method"] == "hybrid_least_favorable":
            assert len(res["lf"]) == len(pieces)
        monkeypatch.setattr(S, "_lf_cv_plain", rec.plain)
        monkeypatch.setattr(S, "_lf_cv_arp", rec.arp)
        monkeypatch.setattr(S, "_setup_piece", rec.setup(_ORIGINAL_SETUP))
        r = S.trend_sensitivity(beta, sigma, n_pre, M=[M], target=l,
                                grid_bounds=(res["grid_lb"], res["grid_ub"]),
                                grid_points=res["points"], **opts)
        mine = r._grid_accept[0].astype(int)
        theirs = np.array([int(c) for c in res["accept"]])
        assert len(mine) == len(theirs)
        if M == 0 and job["restriction"] != "smoothness" and job["variant"] != "none":
            # the sign or monotonicity rows repeat moments of the restriction at
            # M = 0; R makes a duplicate a truncation bound by the sign of rounding
            # error (rho ~ 1e-18) and accepts values it should reject.
            # test_duplicate_moments_do_not_change_the_test checks this case.
            continue
        thetas = np.linspace(res["grid_lb"], res["grid_ub"], res["points"])
        ok = np.ones(len(thetas), bool)
        if job["method"] == "hybrid_fixed_length":
            # the first stage is the FLCI at level alpha/10, and R's grid is that
            # interval: its ends are the first stage's boundary, decided by rounding,
            # and at M = 0 R's FLCI center is inexact (see above), so the points
            # between its first-stage interval and this package's are not comparable
            ok[[0, -1]] = False
            if M == 0:
                vec, half = optimal_flci(sigma, M, n_pre, len(l), l, 0.005)
                c = vec @ beta
                if abs(c - (res["grid_lb"] + res["grid_ub"]) / 2) > 1e-4 * half:
                    # R's inexact direction also enters the second stage (through
                    # the first stage's weights), so nothing is comparable
                    continue
                inside_mine = np.abs(thetas - c) <= half
                inside_r = (thetas >= res["grid_lb"]) & (thetas <= res["grid_ub"])
                rc, rh = (res["grid_lb"] + res["grid_ub"]) / 2, (res["grid_ub"] - res["grid_lb"]) / 2
                inside_r = np.abs(thetas - rc) <= rh * (1 - 1e-12)
                ok &= inside_mine == inside_r
        diff = np.flatnonzero((mine != theirs) & ok)
        # a decision exactly at the boundary of the accepted set can flip on
        # rounding; allow one grid point, and only next to a change of decision
        assert len(diff) <= 1, f"M={M}: {len(diff)} grid points differ from R (at {diff[:10]})"
        for i in diff:
            near = theirs[max(i - 1, 0):i + 2]
            assert near.min() != near.max(), f"M={M}: grid point {i} differs inside a run"


def test_hybrid_tests_its_first_stage_boundary():
    """The FLCI hybrid's grid is its first-stage interval, so the grid's ends are
    on the first stage's boundary, where the truncation bound is the statistic
    itself. Rounding put the statistic just outside it at M = 0 here, R's rule
    then accepted the end of the grid, and the interval reached it ([0.085, 0.350]
    for R's [0.085, 0.238]). The end is tested on the bound instead."""
    rng = np.random.default_rng(1004)
    n = 8
    F = rng.normal(size=(n, 2))
    C = F @ F.T + np.eye(n)
    d = np.sqrt(np.diag(C))
    se = 0.05 * np.exp(rng.normal(scale=0.3, size=n))
    sigma = (C / np.outer(d, d)) * np.outer(se, se)
    times = np.concatenate([-np.arange(4, 0, -1), np.arange(1, 5)])
    beta = 0.005 * times + np.concatenate([np.zeros(4), 0.15 + 0.02 * np.arange(1, 5)])
    beta = beta + np.linalg.cholesky(sigma) @ rng.normal(size=n)
    r = S.trend_sensitivity(beta, sigma, 4, restriction="smoothness",
                            ci_method="hybrid_fixed_length", M=[0.0, 1e-6])
    (_, lo0, hi0), (_, lo1, hi1) = r.table.rows()
    grid_lb, grid_ub = r.diagnostics["grid_bounds"][0]
    assert hi0 < grid_ub - 0.1
    # HonestDiD 0.2.8: [0.08487657, 0.2380951] at M = 0
    assert_allclose([lo0, hi0], [0.08487657, 0.2380951], atol=5e-4)
    assert_allclose([lo0, hi0], [lo1, hi1], atol=(grid_ub - grid_lb) / 999 * 1.01)


def test_duplicate_moments_do_not_change_the_test(monkeypatch):
    """Relative magnitudes at M = 0 with a sign restriction repeat moments; the
    decisions are those of the restriction without the repeats."""
    rng = np.random.default_rng(13)
    beta, sigma = _small_event_study(rng, 3, 3)
    monkeypatch.setattr(S, "_lf_cv_arp", lambda *a: 2.5)
    l = np.full(3, 1 / 3)
    thetas = np.linspace(-1, 1, 400)
    for piece in S._pieces("relative_magnitudes", 3, 3, 0.0, "positive", None):
        A = piece.A
        keep = [i for i in range(len(A)) if not any(np.array_equal(A[i], A[j]) for j in range(i))]
        assert len(keep) < len(A)
        dedup = S._Piece(A[keep], piece.d[keep])
        a = S._test_grid(S._setup_piece(piece, beta, sigma, 3, 3, l, "relative_magnitudes",
                                        "hybrid_least_favorable", 0.05, 0.005, 1000, 0, None, {}),
                         thetas, beta, sigma, 3)
        b = S._test_grid(S._setup_piece(dedup, beta, sigma, 3, 3, l, "relative_magnitudes",
                                        "hybrid_least_favorable", 0.05, 0.005, 1000, 0, None, {}),
                         thetas, beta, sigma, 3)
        assert np.array_equal(a, b)


def _min_variance(sigma, n_pre, l):
    """The smallest variance of v'beta, v = (D w, l), sum(w) = (1, ..., n_post)'l, by a
    general-purpose solver."""
    K = n_pre
    D = np.eye(K) - np.eye(K, k=-1)
    cw = np.dot(np.arange(1, len(l) + 1), l)

    def var(w):
        v = np.concatenate([D @ w, l])
        return v @ sigma @ v

    res = optimize.minimize(var, np.full(K, cw / K), method="SLSQP",
                            constraints=[{"type": "eq", "fun": lambda w: w.sum() - cw}],
                            options={"ftol": 1e-16, "maxiter": 1000})
    return res.fun


def test_r_defaults(ref):
    for d in ref["defaults"]:
        beta, sigma, n_pre = _case(ref, d["case"])
        r = S.trend_sensitivity(beta, sigma, n_pre, restriction=d["restriction"])
        assert_allclose(r.M, d["M"], rtol=1e-10)
        assert r.ci_method == "fixed_length" and set(d["method"]) == {"FLCI"}
        half = (np.array(d["ub"]) - np.array(d["lb"])) / 2
        assert_allclose(r.table["lower"], d["lb"], atol=2e-4 * half.max())
        assert_allclose(r.table["upper"], d["ub"], atol=2e-4 * half.max())
        assert_allclose((r.table["upper"] - r.table["lower"]) / 2, half, rtol=2e-6)


@pytest.mark.parametrize("job", _REF["stock"], ids=_job_id(_REF["stock"]))
def test_stock_r_within_simulation_error(ref, job):
    """R's FLCI uses a simulated folded-normal quantile (10^6 draws): the
    exact one differs by well under 1% of the half-length."""
    beta, sigma, n_pre = _case(ref, job["case"])
    l = np.atleast_1d(np.array(job["l"], float))
    for res in job["results"]:
        if job["method"] == "fixed_length":
            vec, half = optimal_flci(sigma, res["M"], n_pre, len(l), l, 0.05)
            assert_allclose(half, res["half"], rtol=5e-3)
        else:
            if res["M"] == 0:
                vec, half = optimal_flci(sigma, 0.0, n_pre, len(l), l, 0.005)
                if abs(vec @ beta - (res["grid_lb"] + res["grid_ub"]) / 2) > 1e-4 * half:
                    continue  # R's inexact direction at M = 0 (see above)
            r = S.trend_sensitivity(beta, sigma, n_pre, M=[res["M"]], target=l, **_options(job))
            width = res["ub"] - res["lb"]
            lo, hi = r.table.row(0)[1:]
            # the hybrid's first stage is the FLCI at level 0.005, whose simulated tail
            # quantile is the least precise
            assert abs(lo - res["lb"]) < 0.04 * width and abs(hi - res["ub"]) < 0.04 * width


# --------------------------------------------------------------------------
# building blocks
# --------------------------------------------------------------------------

def test_folded_normal_quantile():
    assert_allclose(folded_normal_quantile(0.95, 0.0), 1.959963984540054, rtol=1e-12)
    for t in (0.3, 2.0, 15.0):
        c = folded_normal_quantile(0.9, t)
        from scipy.special import ndtr
        assert_allclose(ndtr(c - t) - ndtr(-c - t), 0.9, atol=1e-13)


def _random_dual(rng, m, k):
    X = rng.normal(size=(m, k))
    sd = rng.uniform(0.5, 2, size=m)
    WT = np.ascontiguousarray(np.column_stack([sd / sd.max(), X]))
    b = np.zeros(k + 1)
    b[0] = 1.0
    return X, sd, WT, b


def test_simplex_matches_highs_on_random_programs():
    rng = np.random.default_rng(3)
    for _ in range(40):
        m, k = rng.integers(4, 30), rng.integers(0, 4)
        X, sd, WT, b = _random_dual(rng, m, k)
        status, basis = K.phase_one(WT, b)
        ref = optimize.linprog(np.zeros(m), A_eq=WT.T, b_eq=b, bounds=[(0, None)] * m,
                               method="highs")
        assert (status == K.OPTIMAL) == (ref.status == 0)
        if status != K.OPTIMAL:
            continue
        for _ in range(5):
            c = rng.normal(size=m)
            st, xB = K.simplex(WT, b, c, basis, 5000)
            hi = optimize.linprog(-c, A_eq=WT.T, b_eq=b, bounds=[(0, None)] * m, method="highs")
            assert st == K.OPTIMAL
            assert_allclose(c[basis] @ xB, -hi.fun, rtol=1e-9, atol=1e-10)


def test_flci_is_the_shortest_affine_interval():
    """At M = 0 the FLCI's direction is the smallest-variance one; at M > 0 no
    direction satisfying the constraint on the weights gives a shorter interval
    (the worst-case bias computed by a linear program, the variance directly)."""
    rng = np.random.default_rng(1)
    K, T = 4, 2
    n = K + T
    F = rng.normal(size=(n, n))
    sigma = F @ F.T / n + np.eye(n) * 0.1
    l = np.array([0.5, 0.5])
    vec0, _ = optimal_flci(sigma, 0.0, K, T, l, 0.05)
    assert_allclose(vec0 @ sigma @ vec0, _min_variance(sigma, K, l), rtol=1e-9)
    A = S._A_SD(K, T)
    D = np.eye(K) - np.eye(K, k=-1)
    cw = np.dot(np.arange(1, T + 1), l)

    def half_length(v, M):
        bias = -optimize.linprog(-v, A_ub=A, b_ub=np.ones(len(A)), bounds=[(None, None)] * n,
                                 method="highs").fun
        h = np.sqrt(v @ sigma @ v)
        return h * folded_normal_quantile(0.95, M * bias / h)

    for M in (0.05, 0.3, 2.0):
        vec, half = optimal_flci(sigma, M, K, T, l, 0.05)
        assert_allclose(half_length(vec, M), half, rtol=1e-8)
        w_opt = np.linalg.solve(D, vec[:K])
        for scale in (1.0, 1e-2, 1e-4):
            for _ in range(100):
                w = (w_opt if scale < 1 else 0) + scale * rng.normal(size=K)
                w += (cw - w.sum()) / K
                v = np.concatenate([D @ w, l])
                assert half_length(v, M) >= half * (1 - 1e-9)


def _small_event_study(rng, n_pre=3, n_post=3):
    n = n_pre + n_post
    F = rng.normal(size=(n, 2))
    C = F @ F.T + np.eye(n)
    se = 0.05 * np.ones(n)
    sigma = np.diag(se) @ (C / np.sqrt(np.outer(np.diag(C), np.diag(C)))) @ np.diag(se)
    beta = np.concatenate([0.01 * np.arange(-n_pre, 0), 0.2 + 0.01 * np.arange(1, n_post + 1)])
    return beta + rng.multivariate_normal(np.zeros(n), sigma), sigma


def test_threads_do_not_change_results():
    rng = np.random.default_rng(5)
    beta, sigma = _small_event_study(rng)
    a = S.trend_sensitivity(beta, sigma, 3, M=[0.5, 1.0], n_threads=1)
    b = S.trend_sensitivity(beta, sigma, 3, M=[0.5, 1.0], n_threads=4)
    assert a.table.equals(b.table)


def test_seed_controls_the_least_favorable_draws():
    rng = np.random.default_rng(6)
    beta, sigma = _small_event_study(rng)
    a = S.trend_sensitivity(beta, sigma, 3, M=[1.0], seed=1)
    b = S.trend_sensitivity(beta, sigma, 3, M=[1.0], seed=1)
    assert a.table.equals(b.table)


def test_open_grid_is_extended():
    """R's grid for relative magnitudes is +-20 standard errors around zero;
    an effect beyond it is found by extending the grid."""
    rng = np.random.default_rng(7)
    beta, sigma = _small_event_study(rng)
    beta = beta.copy()
    beta[3:] += 30 * 0.05
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r = S.trend_sensitivity(beta, sigma, 3, M=[0.5])
    lo, hi = r.table.row(0)[1:]
    assert r.diagnostics["grid_extended"] == [True]
    assert lo > 20 * 0.05 and np.isfinite(hi)


def test_breakdown_and_summary():
    rng = np.random.default_rng(8)
    beta, sigma = _small_event_study(rng)
    r = S.trend_sensitivity(beta, sigma, 3, M=[0, 0.5, 1, 2, 4])
    excl = [(lo > 0 or hi < 0) for lo, hi in r.table.select("lower", "upper").iter_rows()]
    expected = None
    for M, e in zip(r.M, excl):
        if not e:
            break
        expected = M
    assert r.breakdown == expected
    text = r.summary_text()
    assert "relative magnitudes" in text and "C-LF" in text
    json.dumps(r.summary_dict())


def test_validation():
    rng = np.random.default_rng(9)
    beta, sigma = _small_event_study(rng)
    with pytest.raises(ValueError, match="restriction"):
        S.trend_sensitivity(beta, sigma, 3, restriction="parallel")
    with pytest.raises(ValueError, match="allows ci_method"):
        S.trend_sensitivity(beta, sigma, 3, ci_method="fixed_length")
    with pytest.raises(ValueError, match="at most one"):
        S.trend_sensitivity(beta, sigma, 3, bias="positive", monotonicity="increasing")
    with pytest.raises(ValueError, match="two\\s+pre-treatment"):
        S.trend_sensitivity(beta[2:], sigma[2:, 2:], 1, restriction="smoothness_relative_magnitudes")
    with pytest.raises(ValueError, match="target"):
        S.trend_sensitivity(beta, sigma, 3, target=[1, 0])
    with pytest.warns(UserWarning, match="does not use the sign"):
        S.trend_sensitivity(beta, sigma, 3, restriction="smoothness", ci_method="fixed_length",
                            bias="positive", M=[0.1])


def test_event_study_inputs():
    times = [2, -3, 0, -2, 1]
    beta = np.array([2.0, -3.0, 0.0, -2.0, 1.0])
    V = np.diag(np.arange(1.0, 6.0))
    b, Vs, n_pre, w, pre, post = S.event_study_inputs(times, beta, V)
    assert pre == [-3, -2] and post == [0, 1, 2] and n_pre == 2
    assert_allclose(b, [-3, -2, 0, 1, 2])
    assert_allclose(np.diag(Vs), [2, 4, 3, 5, 1])
    assert_allclose(w, [1, 0, 0])
    assert_allclose(S.event_study_inputs(times, beta, V, target="average")[3], [1 / 3] * 3)
    assert_allclose(S.event_study_inputs(times, beta, V, target={1: 0.5, 2: 0.5})[3], [0, 0.5, 0.5])
    with pytest.raises(ValueError, match="consecutive"):
        S.event_study_inputs([-4, -2, 0], beta[:3], V[:3, :3])
    with pytest.raises(ValueError, match="omit"):
        S.event_study_inputs([-2, -1, 0], beta[:3], V[:3, :3])
    with pytest.raises(ValueError, match="no pre-treatment"):
        S.event_study_inputs([0, 1], beta[:2], V[:2, :2])


def _result(names, beta, vcov):
    return RegressionResult(fml="y ~ i(rel, ref=-1)", depvar="y", coefnames=names,
                            beta=np.asarray(beta), vcov=np.asarray(vcov), vcov_type="iid",
                            df_t=100, n_obs=1000)


def test_regression_result_method():
    rng = np.random.default_rng(10)
    beta, sigma = _small_event_study(rng)
    names = ["rel::-4", "rel::-3", "rel::-2", "rel::0", "rel::1", "rel::2"]
    res = _result(names + ["x"], np.append(beta, 1.0),
                  np.block([[sigma, np.zeros((6, 1))], [np.zeros((1, 6)), np.ones((1, 1))]]))
    a = res.trend_sensitivity(M=[0.5])
    b = S.trend_sensitivity(beta, sigma, 3, M=[0.5])
    assert a.table.equals(b.table) and a.pre == [-4, -3, -2] and a.post == [0, 1, 2]
    with pytest.raises(ValueError, match="several variables"):
        _result(names + ["z::1"], np.append(beta, 1.0), np.eye(7)).trend_sensitivity()
    c = res.trend_sensitivity(keep="rel::", target="average", restriction="smoothness", M=[0.01])
    d = S.trend_sensitivity(beta, sigma, 3, target=[1 / 3] * 3, restriction="smoothness", M=[0.01])
    assert c.table.equals(d.table)


def test_plot(tmp_path):
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    rng = np.random.default_rng(11)
    beta, sigma = _small_event_study(rng)
    r = S.trend_sensitivity(beta, sigma, 3, M=[0, 0.5, 1])
    fig = r.plot()
    fig.savefig(tmp_path / "sensitivity.png")
    assert (tmp_path / "sensitivity.png").stat().st_size > 0


def test_plot_without_matplotlib(monkeypatch):
    rng = np.random.default_rng(12)
    beta, sigma = _small_event_study(rng)
    r = S.trend_sensitivity(beta, sigma, 3, M=[0.5])
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", None)
    with pytest.raises(ImportError, match="hdfe-stream\\[plot\\]"):
        r.plot()
    assert isinstance(r.table, pl.DataFrame)


def test_plot_event_study(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    rng = np.random.default_rng(14)
    beta, sigma = _small_event_study(rng)
    names = ["rel::-4", "rel::-3", "rel::-2", "rel::0", "rel::1", "rel::2"]
    res = _result(names, beta, sigma)
    fig = res.plot_event_study()
    ax = fig.axes[0]
    xs = np.concatenate([line.get_xdata() for line in ax.lines if len(line.get_xdata())])
    assert -1 in set(np.round(xs).tolist())          # the reference, drawn at zero
    fig.savefig(tmp_path / "event_study.png")
    fig = res.plot_event_study(joint=True, seed=1, reference=None)
    assert fig.axes[0].get_ylabel().endswith("95% simultaneous interval")
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", None)
    with pytest.raises(ImportError, match="hdfe-stream\\[plot\\]"):
        res.plot_event_study()

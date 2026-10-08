"""Johnson-Lindenstrauss leverages, against exact leverages and known rates.

The reference is again dense linear algebra on a panel small enough to invert,
but the quantity being checked is now a *random* approximation, so the tests are
about distributions and rates rather than equalities: is it unbiased, does the
error fall as 1/sqrt(p), does the non-linearity correction actually remove the
bias it claims to.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import scipy.sparse as sp

from hdfe_stream._columns import GCODE, vcol
from hdfe_stream import StreamingHDFE as _StreamingHDFE
from hdfe_stream.simulate import simulate_akm, simulate_bottleneck

# These tests build panels with single-observation workers and matches on
# purpose, and check the leave-out machinery against references computed on the
# panel as built, so the fit must keep every row.
StreamingHDFE = partial(_StreamingHDFE, fixef_rm="none")

FE = ["worker_id", "firm_id"]


@pytest.fixture(scope="module")
def estimator(tmp_path_factory):
    """A fit whose intermediates survive, so the operator stays usable."""
    base = tmp_path_factory.mktemp("leaveout")
    path = base / "panel.parquet"
    simulate_akm(n_workers=150, n_firms=15, seed=7).write_parquet(path)
    est = StreamingHDFE("log_earn", [], FE, workdir=base / "work", verbose=False,
                        keep_intermediates=True, solver="explicit", tol=1e-13)
    est.fit(str(path))
    return est.reload_intermediates()


@pytest.fixture(scope="module")
def exact_leverage(estimator):
    """Exact P_ii for the same rows, in the order the operator iterates them."""
    ids = {c: [] for c in FE}

    def sink(ordinal, chunk, out):
        for c in FE:
            ids[c].append(np.asarray(chunk[c]))

    estimator.project_rows(estimator.rademacher(), 1, sink, extra_columns=tuple(FE))
    keys = [np.concatenate(ids[c]) for c in FE]
    n = len(keys[0])
    rows = np.arange(n)
    blocks = []
    for key in keys:
        _, code = np.unique(key, return_inverse=True)
        blocks.append(sp.csr_matrix((np.ones(n), (rows, code)),
                                    shape=(n, int(code.max()) + 1)))
    A = sp.hstack(blocks, format="csr").toarray()
    return np.einsum("ij,ij->i", A @ np.linalg.pinv(A.T @ A, hermitian=True), A)


def test_leverage_estimates_track_the_exact_ones(estimator, exact_leverage):
    result = estimator.jla_leverages(n_draws=256, block=16, seed=5)
    assert result.n_rows == len(exact_leverage)
    assert result.leverage.mean() == pytest.approx(exact_leverage.mean(), abs=0.005)
    assert np.corrcoef(result.leverage, exact_leverage)[0, 1] > 0.9


def test_leverages_are_inside_the_unit_interval(estimator):
    """Guaranteed by the normalization, not merely likely: both parts are sums
    of squares, so their ratio cannot leave [0, 1]. With few draws the raw
    estimates would routinely fall outside it."""
    result = estimator.jla_leverages(n_draws=8, block=8, seed=1)
    assert result.leverage.min() >= 0.0
    assert result.leverage.max() <= 1.0
    assert np.allclose(result.leverage + result.complement, 1.0)


def test_blocking_does_not_change_the_answer(estimator):
    """Block size is purely a memory knob. Each block takes its own stretch of
    one draw sequence rather than reseeding, so the same total number of draws
    gives the same answer however it is split up.

    Not bit-identical, though: summing 64 columns in one go rounds differently
    from summing 8 columns eight times. The agreement is to floating-point
    accumulation order, which is what "the same answer" can mean here.
    """
    whole = estimator.jla_leverages(n_draws=64, block=64, seed=3)
    split = estimator.jla_leverages(n_draws=64, block=8, seed=3)
    uneven = estimator.jla_leverages(n_draws=64, block=10, seed=3)

    assert np.allclose(whole.leverage, split.leverage, rtol=1e-12, atol=0)
    assert np.allclose(whole.leverage, uneven.leverage, rtol=1e-12, atol=0)
    assert np.allclose(whole.factor, uneven.factor, rtol=1e-10, atol=0)
    assert whole.diagnostics["blocks"] == 1
    assert split.diagnostics["blocks"] == 8
    assert uneven.diagnostics["blocks"] == 7        # 6 full blocks plus a short one


def test_memory_mapped_and_in_memory_agree(estimator):
    on_disk = estimator.jla_leverages(n_draws=32, block=8, seed=9)
    in_ram = estimator.jla_leverages(n_draws=32, block=8, seed=9, in_memory=True)
    # same accumulation order either way, so this one really is bit-identical
    assert np.array_equal(on_disk.leverage, in_ram.leverage)
    assert np.array_equal(on_disk.factor, in_ram.factor)


def test_error_falls_with_the_square_root_of_draws(estimator, exact_leverage):
    """Sixteen times the draws should cut the error by roughly four."""
    errors = {}
    for draws in (32, 512):
        result = estimator.jla_leverages(n_draws=draws, block=16, seed=11)
        errors[draws] = np.sqrt(((result.leverage - exact_leverage) ** 2).mean())
    assert 2.5 < errors[32] / errors[512] < 6.5, errors


def test_correction_factor_approaches_one_as_draws_grow(estimator):
    """The non-linearity correction is O(1/p), so it should shrink in proportion
    to the draws and never be large."""
    sizes = {}
    for draws in (16, 256):
        result = estimator.jla_leverages(n_draws=draws, block=16, seed=13)
        sizes[draws] = np.abs(result.factor - 1.0).max()
    assert sizes[16] < 0.2                      # a correction, not a rescaling
    assert sizes[256] < sizes[16] / 4           # falling at roughly 1/p


def test_correction_removes_the_bias_it_claims_to(estimator, exact_leverage):
    """The point of the correction: sigma^2 divides by M_ii, and substituting an
    estimate is a non-linear operation that biases 1/M even though the estimate
    of M is unbiased. Averaged over draw sequences, the corrected estimator of
    1/M_ii should be markedly closer to the truth than the uncorrected one.

    Deliberately run at few draws -- the bias is O(1/p), so at a realistic p it
    would be invisible. See prototypes/JLA_NOTES.md for the fuller comparison,
    which is also what established that the published note's formula is right
    and pytwoway's variant is not.
    """
    target = 1.0 / (1.0 - exact_leverage)
    reps, draws = 40, 8
    uncorrected = np.zeros(len(target))
    corrected = np.zeros(len(target))
    for rep in range(reps):
        result = estimator.jla_leverages(n_draws=draws, block=draws, seed=500 + rep)
        inverse = 1.0 / result.complement
        uncorrected += inverse
        corrected += inverse * result.factor

    bias_raw = np.mean(uncorrected / reps / target - 1)
    bias_fixed = np.mean(corrected / reps / target - 1)
    assert bias_raw > 0.005, bias_raw            # there was a bias to remove
    assert abs(bias_fixed) < bias_raw / 3, (bias_raw, bias_fixed)


def test_results_align_with_the_residual_file(tmp_path):
    """The leverages come back in the residual file's row order, which is what
    lets them be combined with residuals without a join. Checked with several
    buckets and small batches, where the order is least obvious."""
    path = tmp_path / "panel.parquet"
    simulate_akm(n_workers=150, n_firms=15, seed=7).write_parquet(path)
    est = StreamingHDFE("log_earn", [], FE, workdir=tmp_path / "work", verbose=False,
                        keep_intermediates=True, outputs="keep", solver="explicit",
                        tol=1e-13, batch_rows=400, n_buckets=3)
    result = est.fit(str(path))
    est.reload_intermediates()

    resid = result.resid().select(*FE).collect()
    ids = {c: [] for c in FE}
    est.project_rows(est.rademacher(), 1,
                     lambda o, ch, out: [ids[c].append(np.asarray(ch[c])) for c in FE],
                     extra_columns=tuple(FE))

    assert est.jla_leverages(n_draws=8, block=8).n_rows == resid.height
    for c in FE:
        assert np.array_equal(np.concatenate(ids[c]), resid[c].to_numpy()), c
    result.cleanup()


def test_diagnostics_report_the_sample(estimator):
    result = estimator.jla_leverages(n_draws=64, block=16, seed=17)
    d = result.diagnostics
    assert d["min_leverage"] <= d["mean_leverage"] <= d["max_leverage"]
    assert d["leave_one_out_connected"] is True
    assert d["converged"] is True
    # the raw estimates satisfy P + M = 1 only in expectation; the deviation is
    # what the normalization absorbs, and it should be small but not zero
    assert 0 < d["raw_sum_deviation"] < 0.5
    assert "JLA leverages" in result.summary()


@pytest.mark.parametrize("n_draws,block", [(0, 8), (8, 0), (-1, 4)])
def test_invalid_draw_counts_are_rejected(estimator, n_draws, block):
    with pytest.raises(ValueError, match="at least 1"):
        estimator.jla_leverages(n_draws=n_draws, block=block)


def test_different_seeds_give_different_but_comparable_estimates(estimator,
                                                                 exact_leverage):
    a = estimator.jla_leverages(n_draws=64, block=16, seed=1)
    b = estimator.jla_leverages(n_draws=64, block=16, seed=2)
    assert not np.array_equal(a.leverage, b.leverage)
    errors = [np.sqrt(((x.leverage - exact_leverage) ** 2).mean()) for x in (a, b)]
    assert 0.5 < errors[0] / errors[1] < 2.0


# --------------------------------------------------------------------------
# sample selection: the leave-one-out connected set
# --------------------------------------------------------------------------

from hdfe_stream.kernels_graph import (_nb_articulation_points,  # noqa: E402
                                       _nb_components, build_bipartite)
from hdfe_stream.leaveout import leave_one_out_connected  # noqa: E402


def graph_of(pairs, n_workers, n_firms):
    pairs = np.asarray(pairs)
    return build_bipartite(pairs[:, 0], pairs[:, 1], n_workers, n_firms)


def cut_vertices(indptr, indices, n_vertices):
    is_cut = np.zeros(n_vertices, bool)
    _nb_articulation_points(indptr, indices, is_cut,
                            np.ones(len(indptr) - 1, bool))
    return is_cut


def cut_vertices_by_brute_force(indptr, indices, n_vertices):
    """Remove each vertex in turn and see whether the component count rises.

    Quadratic, so only usable on tiny graphs -- which is exactly what makes it a
    good independent check on Tarjan's linear-time version.
    """
    def components(alive):
        label = np.empty(n_vertices, np.int64)
        return _nb_components(indptr, indices, alive, label)

    base = components(np.ones(n_vertices, bool))
    out = np.zeros(n_vertices, bool)
    for v in range(n_vertices):
        alive = np.ones(n_vertices, bool)
        alive[v] = False
        out[v] = components(alive) > base
    return out


@pytest.mark.parametrize("trial", range(12))
def test_articulation_points_match_brute_force(trial):
    """Tarjan's algorithm against the definition, on random bipartite graphs."""
    rng = np.random.default_rng(trial)
    n_workers = int(rng.integers(3, 14))
    n_firms = int(rng.integers(2, 8))
    n_edges = int(rng.integers(n_workers, n_workers * 3))
    pairs = np.unique(np.stack([rng.integers(0, n_workers, n_edges),
                                rng.integers(0, n_firms, n_edges)], axis=1), axis=0)
    indptr, indices = graph_of(pairs, n_workers, n_firms)
    n = n_workers + n_firms
    assert np.array_equal(cut_vertices(indptr, indices, n),
                          cut_vertices_by_brute_force(indptr, indices, n))


def test_articulation_points_on_a_known_graph():
    """w0-f0-w1-f1-w2 is a path; its cut vertices are the interior ones, so
    worker 1 is essential and the two workers at the ends are not."""
    indptr, indices = graph_of([(0, 0), (1, 0), (1, 1), (2, 1)], 3, 2)
    is_cut = cut_vertices(indptr, indices, 5)
    assert np.flatnonzero(is_cut[:3]).tolist() == [1]


def test_no_cut_vertices_when_two_workers_span_both_firms():
    """With two workers each seen at both firms, neither is essential: removing
    one leaves the other holding the network together."""
    indptr, indices = graph_of([(0, 0), (0, 1), (1, 0), (1, 1)], 2, 2)
    assert not cut_vertices(indptr, indices, 4).any()


def test_components_are_found_on_a_disconnected_graph():
    """Two disjoint worker-firm pairs are two components."""
    indptr, indices = graph_of([(0, 0), (1, 1)], 2, 2)
    label = np.empty(4, np.int64)
    assert _nb_components(indptr, indices, np.ones(4, bool), label) == 2


@pytest.fixture
def panel_frame():
    return simulate_akm(n_workers=400, n_firms=40, seed=13)


def test_pruning_leaves_the_panel_leave_one_out_connected(tmp_path):
    """The point of the whole exercise. A panel that has an observation of
    leverage one -- so its leave-one-out residual is undefined -- must not after
    pruning.
    """
    frame = simulate_akm(n_workers=2000, n_firms=80, seed=21)
    raw = tmp_path / "raw.parquet"
    frame.write_parquet(raw)

    result = leave_one_out_connected(str(raw))
    pruned = tmp_path / "pruned.parquet"
    result.data.sink_parquet(pruned)

    def max_leverage(source, tag):
        est = StreamingHDFE("log_earn", [], FE, workdir=tmp_path / tag,
                            verbose=False, keep_intermediates=True,
                            solver="explicit", tol=1e-11)
        est.fit(str(source))
        est.reload_intermediates()
        return est.jla_leverages(n_draws=64, block=16, seed=3).diagnostics

    after = max_leverage(pruned, "after")
    assert after["leave_one_out_connected"] is True
    assert after["max_leverage"] < 1.0


def test_pruning_is_a_fixed_point(tmp_path, panel_frame):
    """Deleting all the cut vertices at once cannot create new ones, which is
    why the algorithm does not iterate. Pruning twice must change nothing."""
    path = tmp_path / "panel.parquet"
    panel_frame.write_parquet(path)

    once = leave_one_out_connected(str(path))
    again = leave_one_out_connected(once.data)
    assert again.diagnostics["cut_workers"] == 0
    assert again.diagnostics["components_before"] == 1
    assert np.array_equal(np.sort(again.workers), np.sort(once.workers))
    assert np.array_equal(np.sort(again.firms), np.sort(once.firms))


def test_pruning_accepts_a_lazyframe(tmp_path, panel_frame):
    path = tmp_path / "panel.parquet"
    panel_frame.write_parquet(path)
    from_path = leave_one_out_connected(str(path))
    from_frame = leave_one_out_connected(panel_frame.lazy())
    assert np.array_equal(np.sort(from_path.workers), np.sort(from_frame.workers))


def test_pruned_data_contains_only_surviving_levels(tmp_path, panel_frame):
    path = tmp_path / "panel.parquet"
    panel_frame.write_parquet(path)
    result = leave_one_out_connected(str(path))
    kept = result.data.collect()

    assert set(kept["worker_id"].to_list()) <= set(result.workers.tolist())
    assert set(kept["firm_id"].to_list()) <= set(result.firms.tolist())
    assert kept.height <= panel_frame.height


def test_pruning_keeps_only_the_largest_component(tmp_path):
    """Two groups of workers and firms that never touch: only the bigger one
    survives, because effects in different components are not comparable."""
    import polars as pl

    big = simulate_akm(n_workers=300, n_firms=30, seed=5)
    small = simulate_akm(n_workers=40, n_firms=6, seed=6).with_columns(
        worker_id=pl.col("worker_id") + 10_000_000,
        firm_id=pl.lit("X") + pl.col("firm_id"))
    path = tmp_path / "split.parquet"
    pl.concat([big, small]).write_parquet(path)

    result = leave_one_out_connected(str(path))
    assert result.diagnostics["components_before"] >= 2
    # nothing from the small island survives
    assert not any(str(f).startswith("X") for f in result.firms.tolist())
    assert result.diagnostics["workers_after"] < result.diagnostics["workers_before"]


def test_diagnostics_are_internally_consistent(tmp_path, panel_frame):
    path = tmp_path / "panel.parquet"
    panel_frame.write_parquet(path)
    result = leave_one_out_connected(str(path))
    d = result.diagnostics

    assert d["workers_after"] == len(result.workers) <= d["workers_before"]
    assert d["firms_after"] == len(result.firms) <= d["firms_before"]
    assert d["matches_after"] <= d["matches_before"]
    assert d["components_before"] >= 1
    assert "leave-one-out connected set" in result.summary()


# --------------------------------------------------------------------------
# reproducibility
# --------------------------------------------------------------------------

def leverages_of(frame, workdir, tag, covariates=(), **options):
    """Fit and estimate leverages, returning them in stored row order.

    Deliberately *not* sorted. Sorting would compare the multiset of estimates
    and hide a permutation -- and a permutation is exactly the failure mode
    here, since two rows that are tied in the sort key can swap between runs and
    take each other's random draws. Comparing in order is the stronger claim and
    the one that matters.
    """
    path = workdir / f"{tag}.parquet"
    frame.write_parquet(path)
    est = StreamingHDFE("log_earn", list(covariates), FE,
                        workdir=workdir / f"w_{tag}", verbose=False,
                        keep_intermediates=True, solver="explicit",
                        tol=1e-11, **options)
    est.fit(str(path))
    est.reload_intermediates()
    return est.jla_leverages(n_draws=64, block=16, seed=3).leverage


@pytest.fixture(scope="module")
def order_panel():
    return simulate_akm(n_workers=400, n_firms=40, seed=7)


def test_leverages_do_not_depend_on_the_input_row_order(tmp_path, order_panel):
    """The same data written in a different order must give the same answer.

    The random vectors are keyed on a row's position in the stored row files, so
    this only holds because pass 0 sorts the rows into an order determined by
    the data. Without that, a re-serialized input would silently give a
    different (equally valid, but unreproducible) answer.
    """
    original = leverages_of(order_panel, tmp_path, "orig")
    shuffled = leverages_of(order_panel.sample(fraction=1.0, shuffle=True, seed=99),
                            tmp_path, "shuf")
    resorted = leverages_of(order_panel.sort("firm_id", "year"), tmp_path, "sort")

    assert np.array_equal(original, shuffled)
    assert np.array_equal(original, resorted)


def test_row_order_is_canonical_even_with_covariates(tmp_path, order_panel):
    """The sort key has to include the variables, not only the fixed-effect
    codes.

    Two observations of the same worker at the same firm are tied on the codes
    but differ in their covariates. If the order of such a pair were left to the
    input, they would swap between runs and take each other's draws -- giving a
    different answer for the same data, which is precisely what the ordering is
    supposed to rule out.
    """
    covariates = ("age_squared",)
    original = leverages_of(order_panel, tmp_path, "cov_orig", covariates)
    shuffled = leverages_of(order_panel.sample(fraction=1.0, shuffle=True, seed=5),
                            tmp_path, "cov_shuf", covariates)
    assert np.array_equal(original, shuffled)


def test_leverages_are_reproducible_from_the_seed_alone(tmp_path, order_panel):
    """No hidden entropy: a fixed seed is enough, with no global random state to
    set and no dependence on thread or process count."""
    first = leverages_of(order_panel, tmp_path, "one")
    second = leverages_of(order_panel, tmp_path, "two")
    assert np.array_equal(first, second)


def test_bucketing_changes_the_draws_but_not_by_much(tmp_path, order_panel):
    """A documented limit of the above. `n_buckets` and `rows_per_bucket` decide
    how rows are partitioned, which changes their stored positions and so the
    draws each row receives. The answer moves by sampling noise, not more -- but
    it does move, so a study should hold those options fixed alongside the seed.
    """
    default = leverages_of(order_panel, tmp_path, "def")
    rebucketed = leverages_of(order_panel, tmp_path, "buck", n_buckets=3)

    assert not np.array_equal(default, rebucketed)
    # same distribution, different draws
    assert default.mean() == pytest.approx(rebucketed.mean(), abs=0.01)


# --------------------------------------------------------------------------
# which design the leverages belong to
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def covariate_fit(tmp_path_factory):
    """A fit with covariates, plus the exact leverages of both candidate
    designs: the full one and the fixed effects alone."""
    from hdfe_stream.simulate import simulate_rich

    base = tmp_path_factory.mktemp("cov_lev")
    path = base / "rich.parquet"
    simulate_rich(n_workers=200, n_firms=20, seed=7).write_parquet(path)
    xs = ["age_squared", "x2"]
    est = StreamingHDFE("log_earn", xs, FE, workdir=base / "work", verbose=False,
                        keep_intermediates=True, solver="explicit", tol=1e-13)
    est.fit(str(path))
    est.reload_intermediates()

    x_columns = tuple(vcol(est.vidx[n]) for n in xs)
    seen = {c: [] for c in tuple(FE) + x_columns}
    est.project_rows(est.rademacher(), 1,
                     lambda o, ch, out: [seen[c].append(np.asarray(ch[c]))
                                         for c in seen],
                     extra_columns=tuple(FE) + x_columns, covariates=())
    seen = {c: np.concatenate(v) for c, v in seen.items()}

    n = len(seen[FE[0]])
    rows = np.arange(n)
    blocks = []
    for c in FE:
        _, code = np.unique(seen[c], return_inverse=True)
        blocks.append(sp.csr_matrix((np.ones(n), (rows, code)),
                                    shape=(n, int(code.max()) + 1)))
    A_fe = sp.hstack(blocks, format="csr").toarray()
    A_full = np.column_stack([A_fe] + [seen[c] for c in x_columns])

    def leverage_of(A):
        return np.einsum("ij,ij->i", A @ np.linalg.pinv(A.T @ A, hermitian=True), A)

    return est, xs, leverage_of(A_fe), leverage_of(A_full), n


def test_leverages_default_to_the_fitted_models_design(covariate_fit):
    """The default must be the design that was actually fitted. Fixed-effect
    leverages for a model with controls are leverages of a different design --
    wrong in a way that would not announce itself."""
    est, xs, _, _, _ = covariate_fit
    result = est.jla_leverages(n_draws=64, block=16, seed=5)
    assert result.diagnostics["covariates"] == xs


def test_empty_covariates_can_be_asked_for_explicitly(covariate_fit):
    est, _, _, _, _ = covariate_fit
    result = est.jla_leverages(n_draws=64, block=16, seed=5, covariates=())
    assert result.diagnostics["covariates"] == []


def test_covariates_raise_the_leverage_total_by_their_own_count(covariate_fit):
    """Leverages sum to the rank of the design, so adding k covariates to the
    design adds exactly k to the total -- a sharp, scale-free check that the
    right design is being used, far more discriminating than comparing means.
    """
    est, xs, exact_fe, exact_full, n_rows = covariate_fit
    assert exact_full.sum() - exact_fe.sum() == pytest.approx(len(xs), abs=1e-6)

    with_x = est.jla_leverages(n_draws=512, block=32, seed=5)
    without_x = est.jla_leverages(n_draws=512, block=32, seed=5, covariates=())

    # the estimates carry Monte Carlo noise, so allow a margin around k
    difference = with_x.leverage.sum() - without_x.leverage.sum()
    assert difference == pytest.approx(len(xs), abs=1.0), difference
    assert with_x.leverage.sum() == pytest.approx(exact_full.sum(), rel=0.02)
    assert without_x.leverage.sum() == pytest.approx(exact_fe.sum(), rel=0.02)


def test_several_models_with_different_covariates_are_rejected(tmp_path):
    """A fit covering models with different right-hand sides has no single
    design, so there is no single set of leverages to report."""
    from hdfe_stream.simulate import simulate_rich

    path = tmp_path / "rich.parquet"
    simulate_rich(n_workers=150, n_firms=15, seed=7).write_parquet(path)
    est = StreamingHDFE(
        "log_earn", ["age_squared", "x2"], FE, workdir=tmp_path / "work",
        verbose=False, keep_intermediates=True, solver="explicit",
        models=[{"fml": "a", "y": "log_earn", "x": ["age_squared"]},
                {"fml": "b", "y": "log_earn", "x": ["age_squared", "x2"]}])
    est.fit(str(path))
    est.reload_intermediates()

    with pytest.raises(ValueError, match="several models with different covariates"):
        est.jla_leverages(n_draws=8, block=8)
    # but an explicit choice is fine
    assert est.jla_leverages(n_draws=8, block=8,
                             covariates=("age_squared",)).n_rows > 0


# --------------------------------------------------------------------------
# the bias term
# --------------------------------------------------------------------------

def dense_trace_target(est, seen, gcode, sigma2, psi_dim):
    """tr(Q S^- A' Omega A S^-) for the three forms, densely.

    Built in hdfe_stream's own parametrization -- every level of every dimension
    kept, so S is singular and the pseudo-inverse is the right target, exactly
    as the streaming estimator computes it.
    """
    n = len(gcode)
    rows = np.arange(n)
    offsets, total_levels = est._offsets()
    n_groups = int(gcode.max()) + 1

    streamed = sp.csr_matrix((np.ones(n), (rows, gcode)),
                             shape=(n, n_groups)).toarray()
    levels = np.zeros((n, total_levels))
    for dim in est.o_fe:
        _, code = np.unique(seen[dim], return_inverse=True)
        levels[rows, offsets[dim] + code] = 1.0
    A = np.column_stack([streamed, levels])
    S_inv = np.linalg.pinv(A.T @ A, hermitian=True)
    middle = S_inv @ (A.T @ (sigma2[:, None] * A)) @ S_inv

    # the row-level selectors for the two effects, centered
    psi_rows = levels[:, offsets[psi_dim]:offsets[psi_dim] + est.n_levels[psi_dim]]
    alpha_rows = streamed
    psi_block = np.zeros((n, A.shape[1]))
    psi_block[:, n_groups + offsets[psi_dim]:
              n_groups + offsets[psi_dim] + psi_rows.shape[1]] = psi_rows
    alpha_block = np.zeros((n, A.shape[1]))
    alpha_block[:, :n_groups] = alpha_rows

    def centered(M):
        return M - M.mean(axis=0, keepdims=True)

    Pc, Ac = centered(psi_block), centered(alpha_block)
    forms = {"var(psi)": Pc.T @ Pc / n,
             "var(alpha)": Ac.T @ Ac / n,
             "cov(psi, alpha)": (Pc.T @ Ac + Ac.T @ Pc) / (2 * n)}
    return {name: float(np.trace(Q @ middle)) for name, Q in forms.items()}


@pytest.fixture(scope="module")
def trace_case(tmp_path_factory):
    """A fit plus the dense target its trace estimator should converge to."""
    from hdfe_stream.leaveout import leave_one_out_connected

    base = tmp_path_factory.mktemp("trace")
    raw = base / "raw.parquet"
    simulate_akm(n_workers=250, n_firms=25, seed=13).write_parquet(raw)
    path = base / "panel.parquet"
    leave_one_out_connected(str(raw)).data.sink_parquet(path)

    est = StreamingHDFE("log_earn", [], FE, workdir=base / "work", verbose=False,
                        keep_intermediates=True, solver="explicit", tol=1e-12)
    est.fit(str(path))
    est.reload_intermediates()

    seen = {c: [] for c in FE}
    gcodes = []

    def sink(ordinal, chunk, out):
        gcodes.append(np.asarray(chunk[GCODE]))
        for c in FE:
            seen[c].append(np.asarray(chunk[c]))

    est.project_rows(est.rademacher(), 1, sink, extra_columns=tuple(FE))
    seen = {c: np.concatenate(v) for c, v in seen.items()}
    gcode = np.concatenate(gcodes)

    # any positive per-row variance will do: the trace is what is being tested,
    # not where sigma2 came from
    rng = np.random.default_rng(2)
    sigma2 = rng.uniform(0.5, 1.5, len(gcode))
    return est, sigma2, dense_trace_target(est, seen, gcode, sigma2, FE[1])


@pytest.mark.parametrize("component", ["var(psi)", "var(alpha)", "cov(psi, alpha)"])
def test_trace_converges_to_the_exact_value(trace_case, component):
    """Hutchinson's estimator is unbiased, so with enough draws it must land on
    the trace computed densely. The tolerance reflects its 1/sqrt(p) rate, not
    a belief that it should be exact."""
    est, sigma2, target = trace_case
    estimate = est.hutchinson_trace(sigma2, n_draws=1024, block=32, seed=4)
    got = estimate.as_dict()[component]
    scale = max(abs(target[component]), 1e-12)
    assert abs(got - target[component]) / scale < 0.12, (got, target[component])


def test_trace_error_falls_with_more_draws(trace_case):
    """More draws, less error -- averaged over the three components, since any
    one of them can move the wrong way on a single pair of seeds."""
    est, sigma2, target = trace_case
    values = np.array([target[k] for k in ("var(psi)", "var(alpha)",
                                           "cov(psi, alpha)")])

    def error(draws):
        got = est.hutchinson_trace(sigma2, n_draws=draws, block=32, seed=4)
        estimate = np.array([got.var_psi, got.var_alpha, got.cov])
        return np.abs(estimate - values).sum() / np.abs(values).sum()

    assert error(1024) < error(32)


def test_trace_does_not_depend_on_the_block_size(trace_case):
    """As for the leverages, blocking is a memory knob: each block takes its own
    stretch of one draw sequence rather than reseeding."""
    est, sigma2, _ = trace_case
    whole = est.hutchinson_trace(sigma2, n_draws=64, block=64, seed=7)
    split = est.hutchinson_trace(sigma2, n_draws=64, block=8, seed=7)
    uneven = est.hutchinson_trace(sigma2, n_draws=64, block=10, seed=7)

    for other in (split, uneven):
        assert other.var_psi == pytest.approx(whole.var_psi, rel=1e-10)
        assert other.var_alpha == pytest.approx(whole.var_alpha, rel=1e-10)
        assert other.cov == pytest.approx(whole.cov, rel=1e-10)
    assert whole.diagnostics["blocks"] == 1
    assert split.diagnostics["blocks"] == 8


def test_trace_reports_which_dimensions_it_used(trace_case):
    est, sigma2, _ = trace_case
    estimate = est.hutchinson_trace(sigma2, n_draws=16, block=16)
    assert estimate.diagnostics["psi"] == FE[1]        # firm_id
    assert estimate.diagnostics["alpha"] == FE[0]      # the streamed dimension
    assert estimate.diagnostics["n_obs"] == pytest.approx(est.row_count())
    assert "bias terms" in estimate.summary()


def test_psi_dimension_must_be_a_non_streamed_one(trace_case):
    est, sigma2, _ = trace_case
    with pytest.raises(ValueError, match="non-streamed dimension"):
        est.hutchinson_trace(sigma2, n_draws=8, psi=FE[0])


@pytest.mark.parametrize("n_draws,block", [(0, 8), (8, 0)])
def test_trace_rejects_invalid_draw_counts(trace_case, n_draws, block):
    est, sigma2, _ = trace_case
    with pytest.raises(ValueError, match="at least 1"):
        est.hutchinson_trace(sigma2, n_draws=n_draws, block=block)


# --------------------------------------------------------------------------
# the assembled estimator
# --------------------------------------------------------------------------

PROTOTYPES = Path(__file__).resolve().parents[1] / "prototypes"


@pytest.fixture(scope="module")
def assembled(tmp_path_factory):
    """A fit on a leave-one-out connected panel, ready for the estimator."""
    from hdfe_stream.leaveout import leave_one_out_connected

    base = tmp_path_factory.mktemp("assembled")
    raw = base / "raw.parquet"
    simulate_akm(n_workers=400, n_firms=40, seed=13).write_parquet(raw)
    path = base / "panel.parquet"
    leave_one_out_connected(str(raw)).data.sink_parquet(path)

    est = StreamingHDFE("log_earn", [], FE, workdir=base / "work", verbose=False,
                        keep_intermediates=True, outputs="keep",
                        solver="explicit", tol=1e-12)
    result = est.fit(str(path))
    est.reload_intermediates()
    return est, result


@pytest.fixture(scope="module")
def exact_components(assembled):
    """The same components from the dense prototype, on the same rows.

    This is what `prototypes/kss_exact.py` exists for: it is validated against
    known truth by Monte Carlo and against pytwoway, so it can serve as the
    reference for the streaming version.
    """
    import sys

    sys.path.insert(0, str(PROTOTYPES))
    kss_exact = pytest.importorskip("kss_exact")

    _, result = assembled
    rows = result.resid().select(*FE, "log_earn").collect()
    panel = kss_exact.Panel(worker=np.asarray(rows[FE[0]]),
                            firm=np.asarray(rows[FE[1]]),
                            y=np.asarray(rows["log_earn"], float)).recode()
    return kss_exact.kss(panel)


def test_plug_in_components_match_the_exact_ones(assembled, exact_components):
    """The plug-in half involves no approximation at all -- it is a moment of
    the fitted effects -- so it should agree to floating point."""
    est, result = assembled
    got = est.leave_out_components(result, n_draws=32, block=32, seed=4)
    for key, value in exact_components.plug_in.items():
        assert got.plug_in[key] == pytest.approx(value, abs=1e-9), key


@pytest.mark.parametrize("component", ["var(psi)", "var(alpha)", "cov(psi, alpha)"])
def test_leave_out_components_converge_to_the_exact_ones(assembled,
                                                         exact_components,
                                                         component):
    """End to end against the dense estimator: leverages, sigma2, the bias term
    and the arithmetic, all at once."""
    est, result = assembled
    got = est.leave_out_components(result, n_draws=1024, block=32, seed=4)
    target = exact_components.kss[component]
    assert got.leave_out[component] == pytest.approx(target, rel=0.02), (
        got.leave_out[component], target)


def test_more_draws_get_closer(assembled, exact_components):
    est, result = assembled
    keys = list(exact_components.kss)
    target = np.array([exact_components.kss[k] for k in keys])

    def error(draws):
        got = est.leave_out_components(result, n_draws=draws, block=32, seed=4)
        return np.abs(np.array([got.leave_out[k] for k in keys])
                      - target).sum() / np.abs(target).sum()

    assert error(1024) < error(32)


def test_components_are_the_plug_in_minus_the_bias(assembled):
    """The arithmetic that ties the three reported numbers together."""
    est, result = assembled
    got = est.leave_out_components(result, n_draws=32, block=32, seed=4)
    for key in got.plug_in:
        assert got.leave_out[key] == pytest.approx(got.plug_in[key] - got.bias[key])
    assert got.bias == got.trace.as_dict()


def test_the_bias_goes_the_way_theory_says(assembled):
    """Estimation error inflates the variances and attenuates the covariance."""
    est, result = assembled
    got = est.leave_out_components(result, n_draws=512, block=32, seed=4)
    assert got.bias["var(psi)"] > 0
    assert got.bias["var(alpha)"] > 0
    assert got.bias["cov(psi, alpha)"] < 0


def test_sigma2_recovers_the_error_variance(assembled):
    """The simulated residual has standard deviation 0.3."""
    est, result = assembled
    got = est.leave_out_components(result, n_draws=256, block=32, seed=4)
    assert got.diagnostics["sigma2_mean"] == pytest.approx(0.09, rel=0.15)


def test_movers_are_identified(assembled):
    est, result = assembled
    got = est.leave_out_components(result, n_draws=32, block=32, seed=4)
    assert 0 < got.n_movers < got.n_obs
    assert got.n_obs == result.n_obs
    assert "leave-out variance components" in got.summary()


def test_an_unpruned_panel_is_refused(tmp_path):
    """Without leave-one-out connectedness some observation's leave-one-out
    residual does not exist, so the estimator must decline rather than divide by
    something indistinguishable from zero.

    Built so the condition is certain rather than left to chance: firm B is seen
    in exactly one observation, so that row alone determines firm B's effect and
    carries leverage one.
    """
    import polars as pl

    rng = np.random.default_rng(0)
    worker = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    firm = ["A", "A", "A", "A", "A", "B", "A", "A", "A"]
    frame = pl.DataFrame({
        "worker_id": worker, "firm_id": firm,
        "log_earn": rng.normal(0, 1, len(worker))})
    path = tmp_path / "bridge.parquet"
    frame.write_parquet(path)

    est = StreamingHDFE("log_earn", [], FE, workdir=tmp_path / "work",
                        verbose=False, keep_intermediates=True, outputs="keep",
                        solver="explicit", tol=1e-11)
    result = est.fit(str(path))
    est.reload_intermediates()

    leverages = est.jla_leverages(n_draws=32, block=16, seed=3)
    assert leverages.diagnostics["max_leverage"] == pytest.approx(1.0, abs=1e-9)
    assert leverages.diagnostics["leave_one_out_connected"] is False
    with pytest.raises(ValueError, match="not leave-one-out connected"):
        est.leave_out_components(result, n_draws=32, block=16, seed=3)


@pytest.fixture(scope="module")
def weighted(tmp_path_factory):
    """The same setup as `assembled`, with analytic weights."""
    from hdfe_stream.leaveout import leave_one_out_connected

    base = tmp_path_factory.mktemp("weighted")
    raw = base / "raw.parquet"
    panel = simulate_akm(n_workers=400, n_firms=40, seed=13)
    draws = np.random.default_rng(5).uniform(0.3, 3.0, len(panel))
    panel.with_columns(wt=pl.Series(draws)).write_parquet(raw)
    path = base / "panel.parquet"
    leave_one_out_connected(str(raw)).data.sink_parquet(path)

    est = StreamingHDFE("log_earn", [], FE, workdir=base / "work", verbose=False,
                        keep_intermediates=True, outputs="keep",
                        solver="explicit", tol=1e-12, weights="wt")
    result = est.fit(str(path))
    est.reload_intermediates()
    return est, result


@pytest.fixture(scope="module")
def exact_weighted(weighted):
    """The dense prototype on the same rows, weighted the same way."""
    import sys

    sys.path.insert(0, str(PROTOTYPES))
    kss_exact = pytest.importorskip("kss_exact")

    _, result = weighted
    rows = result.resid().select(*FE, "log_earn", "weights").collect()
    panel = kss_exact.Panel(worker=np.asarray(rows[FE[0]]),
                            firm=np.asarray(rows[FE[1]]),
                            y=np.asarray(rows["log_earn"], float),
                            weight=np.asarray(rows["weights"], float)).recode()
    return kss_exact.kss(panel)


def test_weighted_plug_in_matches_the_exact_one(weighted, exact_weighted):
    """The weighted plug-in moments involve no approximation, so they should
    agree to floating point -- and they are the weighted variances, not the
    unweighted ones, because that is what the bias term is a bias for."""
    est, result = weighted
    got = est.leave_out_components(result, n_draws=32, block=32, seed=4)
    for key, value in exact_weighted.plug_in.items():
        assert got.plug_in[key] == pytest.approx(value, abs=1e-9), key


@pytest.mark.parametrize("component", ["var(psi)", "var(alpha)", "cov(psi, alpha)"])
def test_weighted_components_converge_to_the_exact_ones(weighted, exact_weighted,
                                                       component):
    """End to end with weights: the square-root-weight leverages, the weighted
    sigma2 and stayer imputation, and the trace, against dense algebra."""
    est, result = weighted
    got = est.leave_out_components(result, n_draws=1024, block=32, seed=4)
    target = exact_weighted.kss[component]
    assert got.leave_out[component] == pytest.approx(target, rel=0.02), (
        got.leave_out[component], target)


def test_weighted_leverages_are_the_square_root_metric_ones(weighted,
                                                            exact_weighted):
    """With weights there are two candidate hat matrices and only one is a
    projection. Getting the other would show up as a biased leverage mean, since
    the leverages must sum to the rank of the design either way."""
    est, _ = weighted
    leverages = est.jla_leverages(n_draws=512, block=64, seed=4)
    assert leverages.leverage.max() < 1.0
    assert leverages.diagnostics["max_leverage"] == pytest.approx(
        exact_weighted.max_leverage, rel=0.1)


def test_weights_change_the_answer(weighted, assembled):
    """A guard against the weights being silently ignored: these two fixtures
    are the same panel, one weighted and one not."""
    (west, wresult), (est, result) = weighted, assembled
    a = west.leave_out_components(wresult, n_draws=128, block=32, seed=4)
    b = est.leave_out_components(result, n_draws=128, block=32, seed=4)
    assert not np.allclose([a.leave_out[k] for k in a.leave_out],
                           [b.leave_out[k] for k in b.leave_out], rtol=1e-3)


def test_invalid_options_are_rejected(assembled):
    est, result = assembled
    with pytest.raises(ValueError, match="stayers must be"):
        est.leave_out_components(result, n_draws=8, stayers="invent")
    with pytest.raises(ValueError, match="non-streamed dimension"):
        est.leave_out_components(result, n_draws=8, psi=FE[0])


# --------------------------------------------------------------------------
# the public API
# --------------------------------------------------------------------------

def test_headline_function_prunes_fits_and_decomposes(tmp_path):
    """`leave_out_kss` in one call: the whole pipeline, in the order that makes
    the sample correct."""
    from hdfe_stream import leave_out_kss

    path = tmp_path / "panel.parquet"
    simulate_akm(n_workers=600, n_firms=50, seed=13).write_parquet(path)

    lo = leave_out_kss("log_earn ~ age_squared | worker_id + firm_id", str(path),
                       workdir=tmp_path / "work", n_draws=64, block=32, seed=4,
                       verbose=False)

    assert set(lo.leave_out) == {"var(psi)", "var(alpha)", "cov(psi, alpha)"}
    assert lo.pruning is not None                  # it pruned, and says so
    assert lo.fit is not None                      # and kept the regression
    assert lo.diagnostics["psi"] == "firm_id"
    assert lo.diagnostics["alpha"] == "worker_id"
    assert "pruning" in lo.summary() or "after pruning" in lo.summary()

    tidy = lo.tidy()
    assert tidy.height == 3
    assert set(tidy.columns) == {"component", "plug_in", "bias", "leave_out"}


def test_the_two_paths_agree_exactly(tmp_path):
    """The headline function is a wrapper over the explicit steps, so the two
    must give the same numbers -- not merely similar ones."""
    from hdfe_stream import feols_stream, leave_one_out_connected, leave_out_kss

    path = tmp_path / "panel.parquet"
    simulate_akm(n_workers=600, n_firms=50, seed=13).write_parquet(path)
    fml = "log_earn ~ age_squared | worker_id + firm_id"

    one = leave_out_kss(fml, str(path), workdir=tmp_path / "one", n_draws=64,
                        block=32, seed=4, verbose=False)

    panel = leave_one_out_connected(str(path))
    fit = feols_stream(fml, panel.data, workdir=tmp_path / "two",
                       keep_intermediates=True, stream="worker_id",
                       verbose=False)
    two = fit.leave_out_kss(n_draws=64, block=32, seed=4)

    # Same computation, so agreement is to rounding. Not bitwise: aggregation
    # runs in parallel, and once under heavy CPU load from another process two
    # identical fits differed in the last digit (README, Reproducibility).
    for key in one.leave_out:
        assert one.leave_out[key] == pytest.approx(two.leave_out[key], rel=1e-12,
                                                   abs=1e-15), key
        assert one.plug_in[key] == pytest.approx(two.plug_in[key], rel=1e-12,
                                                 abs=1e-15), key


def test_a_fit_that_discarded_its_intermediates_refuses_politely(tmp_path):
    """Leaving out an observation needs the row fit's intermediates. They are
    gone, so this cannot work -- and the message has to name the flag that would
    have kept them, and the easier way round. (Match level needs only the
    residuals, so it works from the same fit.)"""
    from hdfe_stream import feols_stream

    path = tmp_path / "panel.parquet"
    simulate_akm(n_workers=300, n_firms=30, seed=13).write_parquet(path)
    fit = feols_stream("log_earn ~ age_squared | worker_id + firm_id", str(path),
                       workdir=tmp_path / "work", verbose=False)
    with pytest.raises(RuntimeError, match="keep_intermediates=True"):
        fit.leave_out_kss(leave_out="observation")
    assert fit.leave_out_kss(n_draws=16).diagnostics["leave_out"] == "match"


def test_prune_false_demands_an_already_connected_panel(tmp_path):
    """For callers who pruned upstream: check, do not silently re-prune."""
    from hdfe_stream import leave_one_out_connected, leave_out_kss

    raw = tmp_path / "raw.parquet"
    simulate_akm(n_workers=600, n_firms=50, seed=21).write_parquet(raw)
    fml = "log_earn ~ age_squared | worker_id + firm_id"

    # already pruned: prune=False is fine and does not prune again
    clean = tmp_path / "clean.parquet"
    leave_one_out_connected(str(raw)).data.sink_parquet(clean)
    ok = leave_out_kss(fml, str(clean), workdir=tmp_path / "ok", n_draws=32,
                       block=32, seed=4, prune=False, verbose=False)
    assert ok.pruning is None

    # not pruned: it must object rather than produce something meaningless.
    # Built so the condition is certain: firm B has a single observation, so
    # that row alone determines its effect and carries leverage one.
    bridge = tmp_path / "bridge.parquet"
    rng = np.random.default_rng(0)
    worker = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    pl.DataFrame({
        "worker_id": worker,
        "firm_id": ["A", "A", "A", "A", "A", "B", "A", "A", "A"],
        "age_squared": rng.normal(0, 1, len(worker)),
        "log_earn": rng.normal(0, 1, len(worker)),
    }).write_parquet(bridge)
    with pytest.raises(ValueError, match="not leave-one-out connected"):
        leave_out_kss(fml, str(bridge), workdir=tmp_path / "bad", n_draws=32,
                      block=32, seed=4, prune=False, verbose=False)


@pytest.mark.parametrize("fml,message", [
    ("log_earn ~ age_squared", "no fixed effects"),
    ("log_earn ~ age_squared | worker_id", "needs at least"),
])
def test_formulas_without_two_fixed_effects_are_rejected(tmp_path, fml, message):
    from hdfe_stream import leave_out_kss

    path = tmp_path / "panel.parquet"
    simulate_akm(n_workers=200, n_firms=20, seed=13).write_parquet(path)
    with pytest.raises(ValueError, match=message):
        leave_out_kss(fml, str(path), workdir=tmp_path / "work", verbose=False)


def test_psi_must_name_one_of_the_fixed_effects(tmp_path):
    from hdfe_stream import leave_out_kss

    path = tmp_path / "panel.parquet"
    simulate_akm(n_workers=200, n_firms=20, seed=13).write_parquet(path)
    with pytest.raises(ValueError, match="psi must be one of the fixed effects"):
        leave_out_kss("log_earn ~ age_squared | worker_id + firm_id", str(path),
                      workdir=tmp_path / "work", psi="worker_id", verbose=False)


def test_the_wider_sort_is_only_paid_for_when_it_is_usable(tmp_path):
    """The canonical row order costs a wider sort key, so an ordinary fit does
    not pay for it.

    Keeping the intermediates is the condition, because that is exactly what
    makes the leave-out operator reachable. The invariant worth having is: if
    you can get at the row files, their order is determined by the data. A fit
    that throws them away cannot use the guarantee and is not charged for it.

    Either way the estimates are identical -- the regression does not depend on
    row order at all.
    """
    path = tmp_path / "panel.parquet"
    simulate_akm(n_workers=400, n_firms=40, seed=13).write_parquet(path)
    covariates = ["age_squared", "age_cubed"]

    def fit(tag, **options):
        est = StreamingHDFE("log_earn", covariates, FE,
                            workdir=tmp_path / tag, verbose=False,
                            solver="explicit", tol=1e-12, **options)
        return est.fit(str(path)), est

    plain, plain_est = fit("plain")
    kept, kept_est = fit("kept", keep_intermediates=True)

    assert np.allclose(plain.beta, kept.beta, rtol=0, atol=1e-12)
    assert plain_est.keep_intermediates is False
    assert kept_est.keep_intermediates is True

    # and the kept intermediates really are in canonical order: re-fitting the
    # same data shuffled gives the same leverages, row for row
    shuffled = tmp_path / "shuffled.parquet"
    simulate_akm(n_workers=400, n_firms=40, seed=13).sample(
        fraction=1.0, shuffle=True, seed=7).write_parquet(shuffled)
    other = StreamingHDFE("log_earn", covariates, FE, workdir=tmp_path / "shuf",
                          verbose=False, solver="explicit", tol=1e-12,
                          keep_intermediates=True)
    other.fit(str(shuffled))

    a = kept_est.reload_intermediates().jla_leverages(
        n_draws=32, block=16, seed=3).leverage
    b = other.reload_intermediates().jla_leverages(
        n_draws=32, block=16, seed=3).leverage
    assert np.array_equal(a, b)


# --------------------------------------------------------------------------
# standard errors
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def exact_se(assembled):
    """The dense prototype's standard errors on the same rows."""
    import sys

    sys.path.insert(0, str(PROTOTYPES))
    kss_exact = pytest.importorskip("kss_exact")

    _, result = assembled
    rows = result.resid().select(*FE, "log_earn").collect()
    panel = kss_exact.Panel(worker=np.asarray(rows[FE[0]]),
                            firm=np.asarray(rows[FE[1]]),
                            y=np.asarray(rows["log_earn"], float)).recode()
    return kss_exact.kss_se(panel, sigma2_rule="smooth"), panel, kss_exact


def test_c_has_a_zero_diagonal(exact_se):
    """C_ii = B_ii - M_ii (B_ii / M_ii) = 0 by construction. Everything else
    depends on it: it is why the estimator is unbiased with no trace term and
    why the second variance term is a clean trace."""
    dense, _panel, _mod = exact_se
    for key in dense:
        assert dense[key]["max_abs_C_diagonal"] < 1e-12, key


def test_streaming_c_reproduces_the_point_estimate_exactly(assembled, exact_se):
    """y~'C y~ == plug-in - bias, to floating point.

    This is the strong test of the C operator: two sequential solves, three
    quadratic forms applied in coefficient space, the covariance cross sums and
    the row-level readouts, all checked against an identity rather than against a
    noisy approximation. It uses the *exact* B_ii so that only C is under test,
    and the algebraic sigma2 on every row, since the stayer imputation is what
    breaks the identity.
    """
    est, result = assembled
    _dense, panel, mod = exact_se

    A, J, W = mod.design(panel)
    S_inv = mod._dense_inverse(A)
    P_ii = mod.leverages(A, S_inv)
    forms = mod.quadratic_forms(panel, J, W)
    from hdfe_stream.leaveout_se import COMPONENTS

    b_ii = {name: mod.bii_rows(A, forms[name], S_inv) for name in COMPONENTS}
    b = [b_ii[name] / (1.0 - P_ii) for name in COMPONENTS]

    y = np.asarray(panel.y, float)
    beta = S_inv @ (A.T @ y)
    resid = y - A @ beta
    sigma2 = (y - y.mean()) * resid / (1.0 - P_ii)
    y_tilde = y - y.mean()

    got = np.zeros(3)

    def sink(ordinal, _chunk, cv):
        for k in range(3):
            got[k] += float(y_tilde[ordinal:ordinal + cv.shape[0]] @ cv[:, k, 0])

    est._apply_c(lambda o, c, n, m: y_tilde[o:o + n, None], 1, b, sink, (), FE[1])

    for k, name in enumerate(COMPONENTS):
        target = float(beta @ forms[name] @ beta) - float(b_ii[name] @ sigma2)
        assert got[k] == pytest.approx(target, abs=1e-9, rel=1e-9), name


def test_bii_rows_converges_to_the_exact_diagonal(assembled, exact_se):
    """The per-row B_ii the point estimate never needs and inference does."""
    est, _result = assembled
    _dense, panel, mod = exact_se
    from hdfe_stream.leaveout_se import COMPONENTS

    A, J, W = mod.design(panel)
    S_inv = mod._dense_inverse(A)
    forms = mod.quadratic_forms(panel, J, W)
    got = est.bii_rows(n_draws=1024, block=64, seed=2)
    for name in COMPONENTS:
        exact = mod.bii_rows(A, forms[name], S_inv)
        assert np.corrcoef(exact, got[name])[0, 1] > 0.99, name
        assert got[name].mean() == pytest.approx(exact.mean(), rel=0.05), name


@pytest.mark.parametrize("component", ["var(psi)", "var(alpha)", "cov(psi, alpha)"])
def test_standard_errors_match_the_dense_ones(assembled, exact_se, component):
    est, result = assembled
    dense, _panel, _mod = exact_se
    got = est.leave_out_components(result, n_draws=512, block=64, seed=4, se=True)
    assert got.se.se[component] == pytest.approx(
        dense[component]["se"], rel=0.10), (got.se.se[component],
                                            dense[component]["se"])
    assert got.se.se_conservative[component] == pytest.approx(
        dense[component]["se_conservative"], rel=0.10)


def test_the_conservative_error_is_larger(assembled):
    """The first term alone overshoots V by 2 tr(C Om C Om), so dropping the
    trace correction can only widen the interval."""
    est, result = assembled
    got = est.leave_out_components(result, n_draws=256, block=64, seed=4, se=True)
    for key in got.plug_in:
        assert got.se.se_conservative[key] >= got.se.se[key], key
        assert got.se.trace_term[key] > 0, key


def test_skipping_the_trace_gives_the_conservative_error(assembled):
    est, result = assembled
    full = est.leave_out_components(result, n_draws=256, block=64, seed=4, se=True)
    cheap = est.leave_out_components(result, n_draws=256, block=64, seed=4,
                                     se=True, se_trace=False)
    for key in full.plug_in:
        assert cheap.se.se[key] == pytest.approx(full.se.se_conservative[key],
                                                 rel=0.05), key
        assert cheap.se.trace_term[key] == 0.0
    assert cheap.se.diagnostics["trace_included"] is False


def test_theta_cross_checks_the_point_estimate(assembled):
    """`se.theta` is y~'Cy~, which reaches the same estimand by a route sharing
    almost no code with the point estimate: per-row B_ii by random projection
    rather than the coefficient-space trace, and no stayer imputation. Agreement
    to well within one standard error is a real end-to-end check."""
    est, result = assembled
    got = est.leave_out_components(result, n_draws=1024, block=64, seed=4,
                                   se=True, se_draws=256)
    for key in got.plug_in:
        gap = abs(got.se.theta[key] - got.leave_out[key]) / got.se.se[key]
        assert gap < 0.75, (key, gap)


def test_standard_errors_do_not_disturb_the_point_estimate(assembled):
    est, result = assembled
    plain = est.leave_out_components(result, n_draws=256, block=64, seed=4)
    with_se = est.leave_out_components(result, n_draws=256, block=64, seed=4,
                                       se=True, se_draws=64)
    assert plain.leave_out == with_se.leave_out
    assert plain.se is None and with_se.se is not None


def test_summary_reports_intervals(assembled):
    est, result = assembled
    got = est.leave_out_components(result, n_draws=256, block=64, seed=4,
                                   se=True, se_draws=64)
    text = got.summary()
    assert "95% interval" in text and "se" in text
    assert "standard errors from" in text
    assert "95% interval" not in est.leave_out_components(
        result, n_draws=64, block=64, seed=4).summary()


def test_weighted_standard_errors_match_the_dense_ones(weighted):
    """Weights run through inference the same way they run through the point
    estimate: the square-root-weight metric, where sigma2 already lives."""
    import sys

    sys.path.insert(0, str(PROTOTYPES))
    kss_exact = pytest.importorskip("kss_exact")
    from hdfe_stream.leaveout_se import COMPONENTS

    est, result = weighted
    rows = result.resid().select(*FE, "log_earn", "weights").collect()
    panel = kss_exact.Panel(worker=np.asarray(rows[FE[0]]),
                            firm=np.asarray(rows[FE[1]]),
                            y=np.asarray(rows["log_earn"], float),
                            weight=np.asarray(rows["weights"], float)).recode()
    dense = kss_exact.kss_se(panel, sigma2_rule="smooth")
    got = est.leave_out_components(result, n_draws=512, block=64, seed=4, se=True)
    for name in COMPONENTS:
        assert got.se.se[name] == pytest.approx(dense[name]["se"], rel=0.10), (
            name, got.se.se[name], dense[name]["se"])


# --------------------------------------------------------------------------
# memory discipline
# --------------------------------------------------------------------------

def test_scratch_budget_does_not_change_the_answer(assembled):
    """`scratch_mb` shrinks the row chunk to bound the working set. Every random
    vector in these passes is keyed on the absolute row ordinal rather than on a
    per-chunk counter, so the chunk size must not be visible in the results."""
    est, result = assembled
    previous = est.scratch_mb
    try:
        est.scratch_mb = 1
        small = est.leave_out_components(result, n_draws=64, block=8, seed=3,
                                         se=True, se_draws=16)
        est.scratch_mb = 4096
        large = est.leave_out_components(result, n_draws=64, block=8, seed=3,
                                        se=True, se_draws=16)
    finally:
        est.scratch_mb = previous
    for key in small.leave_out:
        assert small.leave_out[key] == pytest.approx(large.leave_out[key],
                                                     rel=1e-12), key
        assert small.se.se[key] == pytest.approx(large.se.se[key], rel=1e-12), key


def test_row_sized_results_are_memory_mapped(assembled):
    """The row-sized accumulators are handed back as mapped views, not copied
    into RAM. At a hundred million rows the copies were the difference between a
    few hundred MB and a dozen GB, so this is load-bearing rather than tidy."""
    est, _result = assembled
    lev = est.jla_leverages(n_draws=8, block=8, seed=1)
    assert isinstance(lev.leverage, np.memmap), type(lev.leverage)
    assert isinstance(lev.complement, np.memmap)
    assert isinstance(lev.factor, np.memmap)
    b_ii = est.bii_rows(n_draws=8, block=8, seed=1)
    for name, array in b_ii.items():
        assert isinstance(array, np.memmap), (name, type(array))


def test_successive_results_do_not_alias(assembled):
    """Each store gets its own file. Sharing one would mean a second call
    silently overwrote the arrays a caller still held -- which is exactly what
    returning mapped views instead of copies would otherwise invite."""
    est, _result = assembled
    first = est.jla_leverages(n_draws=8, block=8, seed=1)
    kept = np.array(first.leverage, copy=True)
    second = est.jla_leverages(n_draws=32, block=8, seed=9)
    assert np.array_equal(first.leverage, kept), "first result was overwritten"
    assert not np.shares_memory(first.leverage, second.leverage)


# --------------------------------------------------------------------------
# weak identification: the eigenvalue diagnostic
# --------------------------------------------------------------------------

def _dense_weak_id(result, weights=False, top=3):
    import sys

    sys.path.insert(0, str(PROTOTYPES))
    kss_exact = pytest.importorskip("kss_exact")
    cols = [*FE, "log_earn"] + (["weights"] if weights else [])
    rows = result.resid().select(*cols).collect()
    panel = kss_exact.Panel(
        worker=np.asarray(rows[FE[0]]), firm=np.asarray(rows[FE[1]]),
        y=np.asarray(rows["log_earn"], float),
        weight=np.asarray(rows["weights"], float) if weights else None).recode()
    return kss_exact.weak_id(panel, top=top)


@pytest.fixture(scope="module")
def diagnosed(assembled):
    est, result = assembled
    got = est.leave_out_components(result, n_draws=512, block=32, seed=4,
                                   diagnose=True, diagnose_draws=128)
    return got.weak_id, _dense_weak_id(result)


@pytest.mark.parametrize("component", ["var(psi)", "var(alpha)", "cov(psi, alpha)"])
def test_weak_id_eigenvalues_match_dense(diagnosed, component):
    """Lanczos in the S^- inner product, against eig(S^- Q)."""
    got, dense = diagnosed
    lam = np.array(got.eigenvalues[component])
    ref = np.array(dense[component]["eigenvalues"])
    assert len(lam) == 3
    # the default stops at a residual bound of 1e-4, which is ample for a ratio
    # judged against 1/10; the long-run test below checks full convergence
    assert np.allclose(lam, ref, rtol=0, atol=5e-4 * abs(ref[0])), (lam, ref)
    assert got.converged[component]


@pytest.mark.parametrize("component", ["var(psi)", "var(alpha)", "cov(psi, alpha)"])
def test_weak_id_sum_of_squares_matches_dense(diagnosed, component):
    """sum lambda^2 by deflated Hutchinson, against tr((S^- Q)^2)."""
    got, dense = diagnosed
    target = dense[component]["sum_sq"]
    assert got.sum_sq[component] == pytest.approx(target, rel=0.03)
    assert abs(got.sum_sq[component] - target) < 4 * got.sum_sq_se[component] + 1e-12


def test_weak_id_ratios_and_q_match_dense(diagnosed):
    got, dense = diagnosed
    for name in dense:
        assert np.allclose(got.ratios[name], dense[name]["ratios"], rtol=0.03)
        if not got.q_borderline[name]:
            assert got.q[name] == min(dense[name]["q"], 3), name


def test_weak_id_first_stage_f_is_close_to_dense(diagnosed):
    """F depends on the leave-out sigma2, whose leverages are random here and
    exact in the reference, so only rough agreement is expected."""
    got, dense = diagnosed
    for name in dense:
        assert got.f_stat[name] == pytest.approx(dense[name]["f_stat"], rel=0.3,
                                                 abs=0.05), name


def test_weak_id_flags_a_bottleneck(tmp_path):
    """The design KSS warn about: a bottleneck in the mobility network makes one
    eigenvalue dominate, and the diagnostic has to say so -- loudly."""
    path = tmp_path / "bottleneck.parquet"
    simulate_bottleneck(n_bridge=2).write_parquet(path)
    est = StreamingHDFE("log_earn", [], FE, workdir=tmp_path / "work",
                        verbose=False, keep_intermediates=True, outputs="keep",
                        solver="explicit", tol=1e-12)
    result = est.fit(str(path))
    est.reload_intermediates()
    with pytest.warns(UserWarning, match="normal interval is not justified"):
        got = est.leave_out_components(result, n_draws=128, block=16, seed=1,
                                       diagnose=True, diagnose_draws=64)
    dense = _dense_weak_id(result)
    for name in dense:
        assert got.weak_id.q[name] >= 1, name
        assert got.weak_id.ratios[name][0] > 0.9, name
        # deflation is what makes this accurate: the plain estimator's error is
        # worst exactly when one eigenvalue dominates
        assert got.weak_id.ratios[name][0] == pytest.approx(
            dense[name]["ratios"][0], abs=5e-3), name


def test_weak_id_is_quiet_on_a_well_connected_panel(tmp_path):
    path = tmp_path / "panel.parquet"
    from hdfe_stream.leaveout import leave_one_out_connected

    raw = tmp_path / "raw.parquet"
    simulate_akm(n_workers=1500, n_firms=150, seed=11).write_parquet(raw)
    leave_one_out_connected(str(raw)).data.sink_parquet(path)
    est = StreamingHDFE("log_earn", [], FE, workdir=tmp_path / "work",
                        verbose=False, keep_intermediates=True, outputs="keep",
                        solver="explicit", tol=1e-12)
    result = est.fit(str(path))
    est.reload_intermediates()
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        got = est.leave_out_components(result, n_draws=128, block=16, seed=1,
                                       diagnose=True, diagnose_draws=64)
    assert all(got.weak_id.q[name] == 0 for name in got.plug_in)
    assert not got.weak_id.weakly_identified


@pytest.mark.parametrize("reorth", [False, True])
def test_lanczos_survives_long_runs(assembled, reorth):
    """Both recurrences, run well past convergence.

    Three-term: loss of orthogonality produces ghost copies of converged
    eigenvalues, which the Cullum-Willoughby filter must merge.
    Reorthogonalized: rounding leaves a trace of the constant that is invisible
    to the S^- inner product and that reorthogonalization re-injects, so
    without the range projection it grows geometrically and the run explodes.
    """
    est, result = assembled
    dense = _dense_weak_id(result)
    L = est._se_layout(est._model_covariates(None), None)
    caps, _ = est._krylov_plan(L, 10_000)
    with est._quiet_solver():
        a, b, lengths, _ = est._lanczos(L, (), 120, 0,
                                        caps=[min(120, c) for c in caps],
                                        reorth=[reorth] * 3)
    for c, name in enumerate(dense):
        theta, _bounds, _S = est._ritz_top(a[c], b[c], 3)
        ref = np.array(dense[name]["eigenvalues"])
        assert np.allclose(theta, ref, rtol=1e-8), (name, theta, ref)
        assert np.all(np.isfinite(a[c])) and np.all(np.abs(a[c]) < 1.0), name


def test_diagnose_defaults_to_se(assembled):
    """The diagnostic judges the interval, so it runs when there is one."""
    est, result = assembled
    plain = est.leave_out_components(result, n_draws=32, block=16, seed=4)
    assert plain.weak_id is None
    with pytest.warns(UserWarning):
        with_se = est.leave_out_components(result, n_draws=32, block=16, seed=4,
                                           se=True, se_draws=16,
                                           diagnose_draws=16)
    assert with_se.weak_id is not None
    # the same estimate; parallel sums need not agree in the last bits
    assert with_se.leave_out == pytest.approx(plain.leave_out, rel=1e-10)


def test_summary_marks_weakly_identified_intervals(assembled):
    est, result = assembled
    with pytest.warns(UserWarning):
        got = est.leave_out_components(result, n_draws=64, block=16, seed=4,
                                       se=True, se_draws=16, diagnose_draws=32)
    text = got.summary()
    assert "weak-identification diagnostic" in text
    if got.weak_id.weakly_identified:
        assert "NOT justified" in text and "*" in text


def test_weighted_weak_id_matches_dense(weighted):
    est, result = weighted
    got = est.leave_out_components(result, n_draws=256, block=32, seed=4,
                                   diagnose=True, diagnose_draws=128)
    dense = _dense_weak_id(result, weights=True)
    for name in dense:
        ref = np.array(dense[name]["eigenvalues"])
        assert np.allclose(got.weak_id.eigenvalues[name], ref, rtol=0,
                           atol=5e-4 * abs(ref[0])), name
        assert got.weak_id.sum_sq[name] == pytest.approx(dense[name]["sum_sq"],
                                                          rel=0.03), name


# --------------------------------------------------------------------------
# the q = 1 weak-identification interval
# --------------------------------------------------------------------------

def test_deflated_c_reproduces_theta1_exactly(assembled, exact_se):
    """y~'C2 y~ == y~'C y~ - lambda (b1^2 - sum x1^2 sigma2), to floating point.

    The identity is algebraic: it holds for *any* lambda and any direction
    x1 = X~ u, not only the eigen-direction. So a random u and arbitrary
    lambdas test the deflation machinery on its own -- the rank-one term the
    first pass accumulates, and b2 -- with Lanczos nowhere involved.
    """
    est, _result = assembled
    _dense, panel, mod = exact_se
    from hdfe_stream.leaveout_se import COMPONENTS

    A, J, W = mod.design(panel)
    S_inv = mod._dense_inverse(A)
    P_ii = mod.leverages(A, S_inv)
    forms = mod.quadratic_forms(panel, J, W)
    b_ii = [mod.bii_rows(A, forms[name], S_inv) for name in COMPONENTS]
    y = np.asarray(panel.y, float)
    beta = S_inv @ (A.T @ y)
    resid = y - A @ beta
    sigma2 = (y - y.mean()) * resid / (1.0 - P_ii)
    y_tilde = y - y.mean()

    L = est._se_layout((), FE[1])
    rng = np.random.default_rng(5)
    U = (rng.standard_normal((L["total_levels"], 3)), np.zeros((0, 3)),
         rng.standard_normal((L["n_groups"], 3)))
    lams = np.array([0.7, 1.3, -0.4])
    x1 = np.empty((len(y), 3))
    for ordinal, chunk, starts, codes, w in est._chunks():
        x1[ordinal:ordinal + len(w)] = est._eigen_rows(U, chunk, starts, codes,
                                                       w, [])
    b = [b_ii[k] / (1.0 - P_ii) for k in range(3)]
    b2 = [(b_ii[k] - lams[k] * x1[:, k] ** 2) / (1.0 - P_ii) for k in range(3)]

    def run(bs, deflate):
        out = np.zeros(3)

        def sink(ordinal, _chunk, cv):
            for k in range(3):
                out[k] += float(y_tilde[ordinal:ordinal + cv.shape[0]] @ cv[:, k, 0])

        est._apply_c(lambda o, c, n, m: y_tilde[o:o + n, None], 1, bs, sink, (),
                     FE[1], deflate=deflate)
        return out

    plain, deflated = run(b, None), run(b2, (lams, U))
    for k in range(3):
        b1 = float(x1[:, k] @ y_tilde)
        target = plain[k] - lams[k] * (b1 ** 2 - float(x1[:, k] ** 2 @ sigma2))
        assert deflated[k] == pytest.approx(target, abs=1e-9, rel=1e-9), k


@pytest.fixture(scope="module")
def bottleneck_fit(tmp_path_factory):
    """A weakly identified panel that still meets Theorem 3's conditions: six
    movers bridge two blocks of firms, so the leading direction spreads over
    enough rows (max x1^2 about 0.04) for the q = 1 theory to apply."""
    base = tmp_path_factory.mktemp("bottleneck")
    path = base / "panel.parquet"
    simulate_bottleneck(n_bridge=6).write_parquet(path)
    est = StreamingHDFE("log_earn", [], FE, workdir=base / "work", verbose=False,
                        keep_intermediates=True, outputs="keep",
                        solver="explicit", tol=1e-12)
    result = est.fit(str(path))
    est.reload_intermediates()
    with pytest.warns(UserWarning, match="not justified"):
        got = est.leave_out_components(result, n_draws=512, block=32, seed=3,
                                       se=True, se_draws=256, diagnose_draws=128)
    import sys

    sys.path.insert(0, str(PROTOTYPES))
    kss_exact = pytest.importorskip("kss_exact")
    rows = result.resid().select(*FE, "log_earn").collect()
    panel = kss_exact.Panel(worker=np.asarray(rows[FE[0]]),
                            firm=np.asarray(rows[FE[1]]),
                            y=np.asarray(rows["log_earn"], float)).recode()
    return est, result, got, kss_exact.kss_weak(panel)


@pytest.mark.parametrize("component", ["var(psi)", "var(alpha)", "cov(psi, alpha)"])
def test_weak_interval_matches_dense(bottleneck_fit, component):
    _est, _result, got, dense = bottleneck_fit
    piece, ref = got.se.weak[component], dense[component]
    assert piece["lambda_1"] == pytest.approx(ref["lam1"], rel=1e-6)
    assert abs(piece["b1"]) == pytest.approx(abs(ref["b1"]), rel=1e-6)
    assert piece["theta1"] == pytest.approx(ref["theta1"], abs=0.02 * abs(ref["theta1"])
                                            + 1e-4)
    # The deterministic pieces above agree tightly. What follows rests on the
    # smoothed sigma2, whose regressors are random-projection estimates here and
    # exact in the reference: on this design V[b1] moves by about +-25% across
    # seeds of the same streaming fit, while with exact regressors it does not
    # move at all (docs/kss_methodological_differences.md, 5.3). The tolerances are
    # set from that measured dispersion, not from the solver's precision.
    assert piece["sigma"][0, 0] == pytest.approx(ref["Sigma"][0, 0], rel=0.5)
    assert piece["sigma"][1, 1] == pytest.approx(ref["Sigma"][1, 1], rel=0.3)
    lo, hi = piece["interval"]
    dlo, dhi = ref["weak"]
    width = dhi - dlo
    assert abs(lo - dlo) < 0.25 * width and abs(hi - dhi) < 0.25 * width


def test_weak_interval_is_reported_where_it_is_needed(bottleneck_fit):
    _est, _result, got, _dense = bottleneck_fit
    assert set(got.se.weak) == set(got.weak_id.weakly_identified)
    for piece in got.se.weak.values():
        lo, hi = piece["interval"]
        assert lo < hi and piece["kappa"] >= 0
    text = got.summary()
    assert "intervals valid under weak identification" in text
    assert "q = 1" in text


def test_no_weak_interval_on_a_well_identified_panel(assembled):
    """Nothing is computed -- or paid for -- when no component is flagged."""
    est, result = assembled
    with pytest.warns(UserWarning):
        got = est.leave_out_components(result, n_draws=64, block=16, seed=4,
                                       se=True, se_draws=16, diagnose_draws=32)
    flagged = got.weak_id.weakly_identified
    assert set(got.se.weak or {}) == set(flagged)


def test_weak_interval_can_be_switched_off(bottleneck_fit):
    est, result, _got, _dense = bottleneck_fit
    with pytest.warns(UserWarning):
        got = est.leave_out_components(result, n_draws=64, block=16, seed=3,
                                       se=True, se_draws=16, diagnose_draws=32,
                                       weak_interval=False)
    assert got.se.weak is None
    assert "intervals valid under weak identification" not in got.summary()


def test_confidence_sets_both_intervals(bottleneck_fit):
    est, result, _got, _dense = bottleneck_fit
    with pytest.warns(UserWarning):
        wide = est.leave_out_components(result, n_draws=64, block=16, seed=3,
                                        se=True, se_draws=16, diagnose_draws=32)
    with pytest.warns(UserWarning):
        narrow = est.leave_out_components(result, n_draws=64, block=16, seed=3,
                                          se=True, se_draws=16, diagnose_draws=32,
                                          confidence=0.80)
    assert "80% interval" in narrow.summary()
    for name, piece in narrow.se.weak.items():
        w_lo, w_hi = wide.se.weak[name]["interval"]
        n_lo, n_hi = piece["interval"]
        assert n_hi - n_lo < w_hi - w_lo


# --------------------------------------------------------------------------
# pruning: iteration and single observations
# --------------------------------------------------------------------------

def _tiny_panel(spec):
    rows = [(worker, firm) for worker, firms in spec for firm in firms]
    return (pl.DataFrame(rows, schema=[FE[0], FE[1]], orient="row")
            .with_columns(log_earn=pl.lit(1.0)).lazy())


def test_pruning_repeats_until_no_cut_vertex_is_left():
    """Deleting a cut vertex can create a new one. W1 links F1, F2 and a
    pendant F3; once it is gone, W2 is the only link between F1 and F2, so a
    single round of pruning leaves a set that is not leave-one-out connected."""
    from hdfe_stream.leaveout import leave_one_out_connected

    spec = [(1, ["F1", "F2", "F3"]), (2, ["F1", "F2"])]
    spec += [(10 * (j + 1) + k, [f, f]) for k in range(3)
             for j, f in enumerate(("F1", "F2", "F3"))]
    kept = leave_one_out_connected(_tiny_panel(spec))
    assert kept.diagnostics["pruning_rounds"] == 2
    assert kept.diagnostics["cut_workers"] == 2
    movers = (kept.data.collect().group_by(FE[0]).agg(pl.col(FE[1]).n_unique())
              .filter(pl.col(FE[1]) > 1))
    assert movers.height == 0


def test_pruning_drops_workers_observed_once():
    """Their leverage is exactly 1, and as pendant vertices they are never
    articulation points, so pruning on the network alone keeps them."""
    from hdfe_stream.leaveout import leave_one_out_connected

    spec = [(1, ["F1", "F2"]), (2, ["F1", "F2"]), (3, ["F2", "F1"]), (99, ["F1"])]
    kept = leave_one_out_connected(_tiny_panel(spec))
    assert 99 not in kept.data.collect()[FE[0]].to_list()
    assert kept.diagnostics["single_observation_workers"] == 1


def test_pruned_panel_is_leave_one_out_connected(tmp_path):
    """The property pruning exists to deliver, checked on the outcome: after
    pruning a panel with single observations and a fragile network, no
    observation has leverage 1."""
    from hdfe_stream.leaveout import leave_one_out_connected

    raw = simulate_akm(n_workers=600, n_firms=120, p_move=0.03, p_obs=0.35,
                       seed=4)
    path = tmp_path / "raw.parquet"
    raw.write_parquet(path)
    kept = leave_one_out_connected(str(path))
    assert kept.diagnostics["single_observation_workers"] > 0
    pruned = tmp_path / "pruned.parquet"
    kept.data.sink_parquet(pruned)
    est = StreamingHDFE("log_earn", [], FE, workdir=tmp_path / "work",
                        verbose=False, keep_intermediates=True,
                        solver="explicit", tol=1e-12)
    est.fit(str(pruned))
    est.reload_intermediates()
    lev = est.jla_leverages(n_draws=64, block=16, seed=0)
    assert lev.diagnostics["max_leverage"] < 1.0 - 1e-6


# --------------------------------------------------------------------------
# leaving out a match (the default)
# --------------------------------------------------------------------------

MATCH_FML = "log_earn ~ 1 | worker_id + firm_id"


def _kss_exact():
    import sys

    sys.path.insert(0, str(PROTOTYPES))
    return pytest.importorskip("kss_exact")


@pytest.fixture(scope="module")
def match_case(tmp_path_factory):
    """A raw panel with spells of varying length, its pruned rows, and the
    streaming match-level estimate under each centering."""
    import warnings

    from hdfe_stream import leave_out_kss

    base = tmp_path_factory.mktemp("match")
    raw = base / "raw.parquet"
    simulate_akm(n_workers=400, n_firms=40, seed=13).write_parquet(raw)
    clean = leave_one_out_connected(str(raw)).data.collect()
    got = {}
    for centering in ("weighted", "reference"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            got[centering] = leave_out_kss(MATCH_FML, str(raw),
                                          workdir=base / centering,
                                          centering=centering, n_draws=1024,
                                          block=32, seed=4, verbose=False)
    return raw, clean, got


def test_leaving_out_a_match_is_the_default(match_case):
    _, _, got = match_case
    for centering, components in got.items():
        d = components.diagnostics
        assert d["leave_out"] == "match" and d["centering"] == centering
        assert d["person_years"] == components.fit.n_obs
        assert components.n_obs == components.fit.n_obs
        assert d["n_matches"] == components.match_fit.n_obs < components.n_obs
        assert d["max_leverage"] < 1.0
        assert "matches" in components.summary()


def test_the_match_table_is_the_collapsed_panel(match_case):
    """One row per match: the spell's mean partialled-out outcome, weighted by
    its length. With no covariates the partialled-out outcome is the outcome."""
    kss_exact = _kss_exact()
    _, clean, got = match_case
    collapsed, _ = kss_exact.collapse_matches(kss_exact.Panel.from_frame(clean))
    table = (got["weighted"].match_fit.resid()
             .select(*FE, "log_earn", "weights").collect()
             .sort(FE))
    workers = np.unique(clean[FE[0]])
    firms = np.unique(clean[FE[1]])
    exact = (pl.DataFrame({FE[0]: workers[collapsed.worker],
                           FE[1]: firms[collapsed.firm],
                           "y": collapsed.y, "w": collapsed.w})
             .sort(FE))
    assert table[FE[0]].to_list() == exact[FE[0]].to_list()
    assert table[FE[1]].to_list() == exact[FE[1]].to_list()
    np.testing.assert_allclose(table["log_earn"], exact["y"], atol=1e-9)
    np.testing.assert_array_equal(table["weights"], exact["w"])


@pytest.mark.parametrize("centering", ["weighted", "reference"])
def test_match_level_plug_in_is_exact(match_case, centering):
    kss_exact = _kss_exact()
    _, clean, got = match_case
    exact = kss_exact.kss_match(kss_exact.Panel.from_frame(clean),
                                centering=centering)
    for key, value in exact.plug_in.items():
        assert got[centering].plug_in[key] == pytest.approx(value, abs=1e-9), key


@pytest.mark.parametrize("centering,rel", [
    ("weighted", 0.02),
    # the reference's centering carries the outcome's level (about 10 here) into
    # sigma2, so the trace is noisier at the same number of draws; 4% at most
    # over three seeds when this was written, with seed 4 fixed
    ("reference", 0.06),
])
def test_match_level_components_converge_to_the_exact_ones(match_case, centering,
                                                           rel):
    """End to end against the dense LeaveOutTwoWay-style estimator, which agrees
    with xhdfe to machine precision (prototypes/README.md)."""
    kss_exact = _kss_exact()
    _, clean, got = match_case
    exact = kss_exact.kss_match(kss_exact.Panel.from_frame(clean),
                                centering=centering)
    for key, value in exact.kss.items():
        assert got[centering].leave_out[key] == pytest.approx(value, rel=rel), key


def test_stayers_get_the_within_match_variance_exactly(match_case):
    """sigma_for_stayers.m has no randomness in it: the stayer's person-year
    residuals from the collapsed fit, with leverage 1/T."""
    from hdfe_stream.leaveout_match import stayer_sigma2

    kss_exact = _kss_exact()
    _, clean, got = match_case
    d = kss_exact.kss_match(kss_exact.Panel.from_frame(clean)).diagnostics
    workers = np.unique(clean[FE[0]])
    firms = np.unique(clean[FE[1]])
    stayer = d["stayer"]
    exact = pl.DataFrame({FE[0]: workers[d["worker"][stayer]],
                          FE[1]: firms[d["firm"][stayer]],
                          "exact": d["sigma2"][stayer]})
    components = got["weighted"]
    mine = stayer_sigma2(components.fit, components.match_fit, *FE)
    joined = exact.join(mine, on=FE)
    assert len(joined) == len(exact) == len(mine) > 0
    np.testing.assert_allclose(joined["sigma2"], joined["exact"], atol=1e-10)


def test_centerings_agree_when_every_spell_is_equally_long(tmp_path):
    """sqrt(w) is then a constant, so the two centerings are the same number:
    the only difference between them is the one documented."""
    import warnings

    from hdfe_stream import leave_out_kss

    one = (simulate_akm(n_workers=400, n_firms=40, seed=13)
           .unique(FE, keep="first", maintain_order=True))
    path = tmp_path / "pairs.parquet"
    pl.concat([one, one]).write_parquet(path)
    got = {}
    for centering in ("weighted", "reference"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            got[centering] = leave_out_kss(MATCH_FML, str(path),
                                          workdir=tmp_path / centering,
                                          centering=centering, n_draws=32,
                                          block=32, seed=4, verbose=False)
    for key in got["weighted"].leave_out:
        assert got["reference"].leave_out[key] == pytest.approx(
            got["weighted"].leave_out[key], rel=1e-9, abs=1e-12), key


def test_weighted_centering_does_not_depend_on_the_outcomes_location(
        tmp_path):
    """Adding a constant to the outcome leaves the weighted centering's
    estimate unchanged; the reference's moves (docs/kss_methodological_differences.md
    3.4), which is the case for having the option."""
    import warnings

    from hdfe_stream import leave_out_kss

    panel = simulate_akm(n_workers=400, n_firms=40, seed=13)
    got = {}
    for shift in (0.0, -10.0):
        path = tmp_path / f"shift{shift}.parquet"
        panel.with_columns(pl.col("log_earn") + shift).write_parquet(path)
        for centering in ("weighted", "reference"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                got[centering, shift] = leave_out_kss(
                    MATCH_FML, str(path), workdir=tmp_path / f"{centering}{shift}",
                    centering=centering, n_draws=32, block=32, seed=4,
                    verbose=False).leave_out
    for key in got["weighted", 0.0]:
        assert got["weighted", -10.0][key] == pytest.approx(
            got["weighted", 0.0][key], rel=1e-6), key
    assert any(abs(got["reference", -10.0][k] / got["reference", 0.0][k] - 1)
               > 0.05 for k in got["reference", 0.0])


def test_the_two_step_path_gives_the_same_match_level_answer(match_case,
                                                             tmp_path):
    """Pruning, fitting, then asking the fit is what leave_out_kss does."""
    import warnings

    from hdfe_stream import feols_stream

    _, clean, got = match_case
    path = tmp_path / "clean.parquet"
    clean.write_parquet(path)
    fit = feols_stream(MATCH_FML, str(path), workdir=tmp_path / "work",
                       stream=FE[0], verbose=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mine = fit.leave_out_kss(centering="weighted", n_draws=1024, block=32,
                                 seed=4)
    for key, value in got["weighted"].leave_out.items():
        assert mine.leave_out[key] == pytest.approx(value, rel=1e-9), key


@pytest.mark.parametrize("options,error,message", [
    ({"stayers": "own"}, ValueError, "within-match rule"),
    ({"centering": "median"}, ValueError, "centering must be"),
    ({"leave_out": "spell"}, ValueError, "leave_out must be"),
])
def test_match_level_refuses_before_fitting(tmp_path, match_case, options,
                                            error, message):
    """Nothing is fitted -- and no work directory made -- for a request that
    cannot be served."""
    from hdfe_stream import leave_out_kss

    raw, _, _ = match_case
    workdir = tmp_path / "never"
    with pytest.raises(error, match=message):
        leave_out_kss(MATCH_FML, str(raw), workdir=workdir, prune=False,
                      verbose=False, **options)
    assert not workdir.exists()


def test_weighted_match_level_matches_the_exact_one(tmp_path):
    """User weights and spell lengths compose: a match's weight is its total
    user weight, and the stayer rule uses each person-year's share of it."""
    import warnings

    from hdfe_stream import leave_out_kss

    kss_exact = _kss_exact()
    panel = simulate_akm(n_workers=400, n_firms=40, seed=13)
    draws = np.random.default_rng(5).uniform(0.3, 3.0, len(panel))
    raw = tmp_path / "raw.parquet"
    panel.with_columns(wt=pl.Series(draws)).write_parquet(raw)
    clean = leave_one_out_connected(str(raw)).data.collect()
    exact = kss_exact.kss_match(kss_exact.Panel.from_frame(clean, weight="wt"),
                                centering="weighted")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        got = leave_out_kss(MATCH_FML, str(raw), workdir=tmp_path / "work",
                            weights="wt", centering="weighted", n_draws=1024,
                            block=32, seed=4, verbose=False)
    for key, value in exact.plug_in.items():
        assert got.plug_in[key] == pytest.approx(value, abs=1e-9), key
    for key, value in exact.kss.items():
        assert got.leave_out[key] == pytest.approx(value, rel=0.03), key


# --------------------------------------------------------------------------
# standard errors when leaving out a match
# --------------------------------------------------------------------------

def _collapsed_dense(components, clean, kss_exact):
    """The dense collapsed design on the match fit's own rows, in its order:
    what the streaming kernel should reproduce."""
    rows = (components.match_fit.resid()
            .select(*FE, "log_earn", "weights").collect())
    workers = np.unique(clean[FE[0]])
    firms = np.unique(clean[FE[1]])
    mp = kss_exact.Panel(
        worker=np.searchsorted(workers, np.asarray(rows[FE[0]])),
        firm=np.searchsorted(firms, np.asarray(rows[FE[1]])),
        y=np.asarray(rows["log_earn"], float),
        weight=np.asarray(rows["weights"], float))
    A, J, W = kss_exact.design(mp)
    w = mp.w
    S_inv = kss_exact._dense_inverse(A, w)
    P = kss_exact.leverages(A, S_inv, w)
    stayer = np.bincount(mp.worker)[mp.worker] == 1
    forms = kss_exact.quadratic_forms(mp, J, W, person_years=len(clean))
    return mp, A, w, S_inv, P, stayer, forms


@pytest.mark.parametrize("centering", ["weighted", "reference"])
def test_match_level_kernel_reproduces_the_point_estimate_exactly(
        match_case, centering):
    """The collapsed kernel, with b = 0 on stayers' matches and, for the
    reference's centering, the rank-two term, is the point estimate itself:
    y' K y == plug-in - bias for var(psi) and cov, to floating point. Exact
    b, so only the operator is under test."""
    from hdfe_stream.leaveout_se import COMPONENTS

    kss_exact = _kss_exact()
    _, clean, got = match_case
    components = got["weighted"]
    est = components.match_fit._estimator.reload_intermediates()
    mp, A, w, S_inv, P, stayer, forms = _collapsed_dense(components, clean,
                                                          kss_exact)
    b = []
    for name in COMPONENTS:
        b_ii = kss_exact.bii_rows(A, forms[name], S_inv, w)
        b.append(np.where(stayer, 0.0, b_ii / np.where(stayer, 1.0, 1.0 - P)))
    root = np.sqrt(w)
    y = root * (mp.y if centering == "reference"
                else mp.y - (w @ mp.y) / w.sum())
    rank_two = None
    if centering == "reference":
        u = est._m_times(b, (), FE[1], "test_mb", in_memory=True)
        X = root[:, None] * A.toarray()
        M = np.eye(len(w)) - X @ S_inv @ X.T
        for k in range(3):
            np.testing.assert_allclose(u[k], M @ b[k], atol=1e-9)
        rank_two = (u, 0.5 / len(w))

    total = np.zeros(3)

    def sink(ordinal, _chunk, cv):
        for k in range(3):
            total[k] += float(y[ordinal:ordinal + cv.shape[0]] @ cv[:, k, 0])

    est._apply_c(lambda o, c, n, m: y[o:o + n, None], 1, b, sink, (), FE[1],
                 rank_two=rank_two)
    exact = kss_exact.kss_match(kss_exact.Panel.from_frame(clean),
                                centering=centering)
    for k, name in enumerate(COMPONENTS):
        if name == "var(alpha)":
            continue            # stayers' b is not zero there; not reported
        assert total[k] == pytest.approx(exact.kss[name], abs=1e-10), name


MATCH_SE_CASES = [("reference", "person_year"), ("weighted", "person_year"),
                  ("weighted", "match")]


@pytest.fixture(scope="module")
def match_se(match_case, tmp_path_factory):
    """Match-level standard errors, streaming and dense: the defaults, the
    weighted centering, and the match-level variance source."""
    import warnings

    from hdfe_stream import leave_out_kss

    kss_exact = _kss_exact()
    raw, clean, _ = match_case
    base = tmp_path_factory.mktemp("match_se")
    panel = kss_exact.Panel.from_frame(clean)
    out = {}
    for centering, source in MATCH_SE_CASES:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            got = leave_out_kss(MATCH_FML, str(raw),
                                workdir=base / f"{centering}_{source}",
                                centering=centering, se=True,
                                se_variance=source, n_draws=1024, block=64,
                                seed=1, verbose=False)
        out[centering, source] = (got, kss_exact.kss_match_se(
            panel, centering=centering, se_variance=source))
    return out


@pytest.mark.parametrize("case", MATCH_SE_CASES)
@pytest.mark.parametrize("component", ["var(psi)", "cov(psi, alpha)"])
def test_match_level_standard_errors_match_the_dense_ones(match_se, case,
                                                          component):
    """Streaming against the dense kernel, same conventions: they differ only
    in random-projection leverages and the simulated trace."""
    got, dense = match_se[case]
    assert got.se.se[component] == pytest.approx(dense[component]["se"],
                                                 rel=0.10)
    assert got.se.theta[component] == pytest.approx(got.leave_out[component],
                                                    rel=0.05)


def test_match_level_reports_no_var_alpha_error(match_se):
    """leave_out_COMPLETE's rule, and the summary says so."""
    got, _ = match_se["reference", "person_year"]
    assert got.diagnostics["se_variance"] == "person_year"
    assert np.isnan(got.se.se["var(alpha)"])
    assert got.se.diagnostics["reported"] == ["var(psi)", "cov(psi, alpha)"]
    assert got.se.diagnostics["stayer_rows"] > 0
    assert "no standard error when leaving out a match" in got.summary()
    assert "se_variance='person_year'" in got.summary()


def test_match_level_weak_interval_is_for_var_psi_only(match_se):
    """Stayers load on the covariance's weak direction, so its deflated kernel
    has no zero diagonal at match level; no q = 1 interval is formed for it."""
    for got, _ in match_se.values():
        assert set(got.se.weak or {}) <= {"var(psi)"}


def test_person_year_variance_refuses_user_weights(tmp_path, match_case):
    """The reference's variance source has no user weights; say so before
    fitting, and name the one that does."""
    from hdfe_stream import leave_out_kss

    raw, _, _ = match_case
    with pytest.raises(ValueError, match="se_variance='match'"):
        leave_out_kss(MATCH_FML, str(raw), workdir=tmp_path / "never",
                      se=True, weights="age", prune=False, verbose=False)
    assert not (tmp_path / "never").exists()


# Recorded from xhdfe v2.28.0 (e0c2362), akm_kss(y, w, f, leverages="exact",
# compute_se=True, prune=False, se_nsim=20000, seed=1) on the panel
# `_anchor_panel` builds, with the outcome as simulated ("raw") and demeaned.
# xhdfe ports LeaveOutTwoWay's leave_out_COMPLETE; these anchor
# `xhdfe_match_se`, the prototype's literal port, and through it the
# conventions the package's own version keeps.
XHDFE_MATCH = {
    "raw": {"kss_var_psi": 0.03620125055018599,
            "theta_var_psi": 0.01393328188082155,
            "theta_cov": 0.015201777781926502,
            "se_var_psi": 0.0, "se_cov": 0.0},
    "demeaned": {"kss_var_psi": 0.01155864383204772,
                 "theta_var_psi": 0.011567732722419963,
                 "theta_cov": 0.01224403719848443,
                 "se_var_psi": 0.003003610634675229,
                 "se_cov": 0.003836870427594062},
}


def _anchor_panel(kss_exact, tmp_path, demeaned):
    raw = tmp_path / "anchor.parquet"
    simulate_akm(n_workers=200, n_firms=20, seed=11).write_parquet(raw)
    frame = leave_one_out_connected(str(raw)).data.collect()
    worker = np.unique(frame[FE[0]].to_numpy(), return_inverse=True)[1]
    firm = np.unique(frame[FE[1]].to_numpy(), return_inverse=True)[1]
    y = frame["log_earn"].to_numpy().astype(float)
    return kss_exact.Panel(worker=worker, firm=firm,
                           y=y - y.mean() if demeaned else y).recode()


@pytest.mark.parametrize("outcome", ["raw", "demeaned"])
def test_prototype_reproduces_xhdfe_match_level_inference(outcome, tmp_path):
    """The point estimate and theta_c exactly; the standard error to within
    xhdfe's own simulation error; and zero where xhdfe's is zero -- which, with
    the outcome as log earnings, it is: the variance estimate comes out
    negative and is truncated (docs/kss_methodological_differences.md 5.7)."""
    kss_exact = _kss_exact()
    panel = _anchor_panel(kss_exact, tmp_path, outcome == "demeaned")
    ref = XHDFE_MATCH[outcome]

    point = kss_exact.kss_match(panel)
    assert point.kss["var(psi)"] == pytest.approx(ref["kss_var_psi"], abs=1e-10)

    got = kss_exact.xhdfe_match_se(panel)
    assert got["var(psi)"]["theta_c"] == pytest.approx(ref["theta_var_psi"],
                                                       abs=1e-10)
    assert got["cov(psi, alpha)"]["theta_c"] == pytest.approx(ref["theta_cov"],
                                                              abs=1e-10)
    for name, key in (("var(psi)", "se_var_psi"), ("cov(psi, alpha)", "se_cov")):
        if ref[key] == 0.0:
            assert got[name]["se_numerator"] < 0, name
        else:
            assert got[name]["se"] == pytest.approx(ref[key], rel=0.01), name

"""Applying S^- and the projection A S^- A' to arbitrary vectors.

The reference is dense linear algebra on the same rows: build the design
explicitly, form the pseudo-inverse, and compare. That is only possible on a
small panel, which is the point -- the streaming operator has to agree exactly
with the answer nobody doubts before it is trusted at a scale where nothing can
be checked directly.

These are the primitives leave-out estimation needs (statistical leverages, and
from them the Kline-Saggio-Solvsten bias correction); see `prototypes/` for what
they are for.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from hdfe_stream import StreamingHDFE
from hdfe_stream.simulate import simulate_akm

# Small enough for a dense pseudo-inverse of the full design.
N_WORKERS = 150
N_FIRMS = 15
FE_SETS = [["worker_id", "firm_id"], ["worker_id", "firm_id", "year"]]
SOLVERS = ["explicit", "stream_cg"]


@pytest.fixture(scope="module")
def panel_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("inverse") / "panel.parquet"
    simulate_akm(n_workers=N_WORKERS, n_firms=N_FIRMS, seed=7).write_parquet(path)
    return str(path)


def fitted(panel_path, workdir, fe, solver="explicit", weights=None, **options):
    """A fit whose intermediates survive, with the cell arrays mapped back in.

    `keep_intermediates=True` keeps the row files; `reload_intermediates()` is
    needed because the fit drops its references to the cell arrays either way.
    """
    est = StreamingHDFE("log_earn", [], fe, workdir=workdir, solver=solver,
                        weights=weights, verbose=False, keep_intermediates=True,
                        tol=1e-13, **options)
    est.fit(panel_path)
    return est.reload_intermediates()


def collect(est, n_vectors, seed, fe):
    """Run the projection and return everything needed to check it.

    The row files are in bucket order, not input order, so the identity columns
    come back alongside the results and the dense reference is built from those.
    """
    columns = tuple(c for d in fe for c in d.split("^"))
    if est.weights is not None:
        columns += ("w",)
    vectors = est.rademacher(seed=seed)
    projected, drawn, ids = [], [], {c: [] for c in columns}

    def sink(ordinal, chunk, out):
        projected.append(out.copy())
        drawn.append(vectors(ordinal, len(out), n_vectors))
        for c in columns:
            ids[c].append(np.asarray(chunk[c]))

    info = est.project_rows(vectors, n_vectors, sink, extra_columns=columns)
    return (np.vstack(projected), np.vstack(drawn),
            {c: np.concatenate(v) for c, v in ids.items()}, info)


def dense_projection(ids, fe, weights=None):
    """A, and the projection A (A'WA)^- A' W, densely."""
    keys = [ids[c] for d in fe for c in d.split("^")] if any("^" in d for d in fe) \
        else [ids[d] for d in fe]
    n = len(keys[0])
    rows = np.arange(n)
    blocks = []
    for key in keys:
        _, code = np.unique(key, return_inverse=True)
        blocks.append(sp.csr_matrix((np.ones(n), (rows, code)),
                                    shape=(n, int(code.max()) + 1)))
    A = sp.hstack(blocks, format="csr").toarray()
    w = np.ones(n) if weights is None else weights
    S_inv = np.linalg.pinv(A.T @ (w[:, None] * A), hermitian=True)
    return A, A @ S_inv @ A.T @ np.diag(w)


@pytest.mark.parametrize("fe", FE_SETS, ids=lambda fe: "+".join(fe))
@pytest.mark.parametrize("solver", SOLVERS)
def test_projection_matches_dense(panel_path, tmp_path, fe, solver):
    est = fitted(panel_path, tmp_path / f"{solver}", fe, solver=solver)
    projected, drawn, ids, _ = collect(est, 4, 11, fe)
    _, P = dense_projection(ids, fe)
    reference = P @ drawn
    assert np.abs(projected - reference).max() / np.abs(reference).max() < 1e-10


@pytest.mark.parametrize("fe", FE_SETS, ids=lambda fe: "+".join(fe))
def test_projection_matches_dense_when_weighted(panel_path, tmp_path, fe):
    """The weighted projection is A (A'WA)^- A' W, not the unweighted one."""
    frame = simulate_akm(n_workers=N_WORKERS, n_firms=N_FIRMS, seed=7)
    rng = np.random.default_rng(3)
    path = tmp_path / "weighted.parquet"
    frame.with_columns(wa=rng.uniform(0.5, 2.0, frame.height)).write_parquet(path)

    est = fitted(str(path), tmp_path / "w", fe, weights="wa")
    projected, drawn, ids, _ = collect(est, 4, 11, fe)
    _, P = dense_projection(ids, fe, weights=ids["w"])
    reference = P @ drawn
    assert np.abs(projected - reference).max() / np.abs(reference).max() < 1e-10


def test_projection_survives_bucketing_and_small_batches(panel_path, tmp_path):
    """Several buckets and batches smaller than a chunk: the row ordinal has to
    stay aligned across both passes for the regenerated vectors to match."""
    fe = FE_SETS[0]
    est = fitted(panel_path, tmp_path / "buckets", fe, n_buckets=4, batch_rows=200)
    projected, drawn, ids, _ = collect(est, 3, 5, fe)
    _, P = dense_projection(ids, fe)
    reference = P @ drawn
    assert np.abs(projected - reference).max() / np.abs(reference).max() < 1e-10


def test_projection_is_idempotent(panel_path, tmp_path):
    """P is a projection, so projecting an already-projected vector changes
    nothing. This checks the operator against itself, with no dense reference."""
    fe = FE_SETS[0]
    est = fitted(panel_path, tmp_path / "idem", fe)
    once, _, _, _ = collect(est, 2, 21, fe)

    # feed the first result straight back in, addressed by row ordinal
    def replay(ordinal, n_rows, n_vectors):
        return once[ordinal:ordinal + n_rows, :n_vectors]

    twice = []
    est.project_rows(replay, once.shape[1],
                     lambda ordinal, chunk, out: twice.append(out.copy()))
    twice = np.vstack(twice)
    assert np.abs(twice - once).max() / np.abs(once).max() < 1e-9


def test_design_columns_project_to_themselves(panel_path, tmp_path):
    """Anything already in the column space of the design is unchanged by the
    projection. A column of ones is in that space (the fixed effects span it)."""
    fe = FE_SETS[0]
    est = fitted(panel_path, tmp_path / "span", fe)
    ones = []
    est.project_rows(lambda ordinal, n_rows, n: np.ones((n_rows, n)), 1,
                     lambda ordinal, chunk, out: ones.append(out.copy()))
    ones = np.vstack(ones)
    assert np.abs(ones - 1.0).max() < 1e-9


# --------------------------------------------------------------------------
# what the operator is for: leverages
# --------------------------------------------------------------------------

def jla_leverages(est, n_draws, seed, fe):
    """The Johnson-Lindenstrauss leverage estimates, both of them.

    P_ii is estimated by (P r)_i^2 and M_ii by (r_i - (P r)_i)^2, from the same
    draws -- the second is free given the first, which is what makes the
    algorithm affordable.
    """
    columns = tuple(fe)
    vectors = est.rademacher(seed=seed)
    P_hat, M_hat, ids = [], [], {c: [] for c in columns}

    def sink(ordinal, chunk, q):
        r = vectors(ordinal, len(q), n_draws)
        P_hat.append((q ** 2).mean(axis=1))
        M_hat.append(((r - q) ** 2).mean(axis=1))
        for c in columns:
            ids[c].append(np.asarray(chunk[c]))

    est.project_rows(vectors, n_draws, sink, extra_columns=columns)
    return (np.concatenate(P_hat), np.concatenate(M_hat),
            {c: np.concatenate(v) for c, v in ids.items()})


@pytest.fixture(scope="module")
def leverage_case(panel_path, tmp_path_factory):
    fe = FE_SETS[0]
    est = fitted(panel_path, tmp_path_factory.mktemp("lev"), fe)
    P_hat, M_hat, ids = jla_leverages(est, 600, 4321, fe)
    A, P = dense_projection(ids, fe)
    return P_hat, M_hat, np.diag(P).copy(), A


def test_leverage_estimates_are_unbiased(leverage_case):
    """Averaged over rows, the estimate lands on the exact leverage. Per-row it
    is noisy at any feasible number of draws -- that is inherent to the method,
    not a defect -- so the check is on the mean."""
    P_hat, _, exact, _ = leverage_case
    assert P_hat.mean() == pytest.approx(exact.mean(), abs=0.005)
    assert np.corrcoef(P_hat, exact)[0, 1] > 0.9


def test_leverage_and_its_complement_sum_to_one(leverage_case):
    """P_ii + M_ii = 1 identically, and the two estimates satisfy it on average.
    This is what the 2021 follow-up note exploits to constrain the estimates."""
    P_hat, M_hat, _, _ = leverage_case
    assert (P_hat + M_hat).mean() == pytest.approx(1.0, abs=1e-3)


def test_complement_estimate_is_unbiased(leverage_case):
    P_hat, M_hat, exact, _ = leverage_case
    assert M_hat.mean() == pytest.approx((1 - exact).mean(), abs=0.005)


def test_exact_leverages_sum_to_the_rank(leverage_case):
    """Sanity check on the dense reference itself: trace(P) = rank(A)."""
    _, _, exact, A = leverage_case
    assert exact.sum() == pytest.approx(np.linalg.matrix_rank(A), rel=1e-9)


def test_leverage_error_falls_with_the_square_root_of_draws(panel_path, tmp_path):
    """The approximation error should go as 1/sqrt(p). Sixteen times the draws
    should therefore cut the error by roughly four."""
    fe = FE_SETS[0]
    est = fitted(panel_path, tmp_path / "rate", fe)
    _, _, ids = jla_leverages(est, 1, 0, fe)
    _, P = dense_projection(ids, fe)
    exact = np.diag(P)

    errors = {}
    for draws in (50, 800):
        P_hat, _, _ = jla_leverages(est, draws, 4321, fe)
        errors[draws] = np.sqrt(((P_hat - exact) ** 2).mean())
    ratio = errors[50] / errors[800]
    assert 2.5 < ratio < 6.5, errors


# --------------------------------------------------------------------------
# contract
# --------------------------------------------------------------------------

def test_rademacher_draws_are_reproducible_and_chunk_independent(panel_path, tmp_path):
    """The vectors are regenerated rather than stored, so the same entry must
    come back regardless of how the rows are chunked."""
    est = fitted(panel_path, tmp_path / "rng", FE_SETS[0])
    generate = est.rademacher(seed=77)

    whole = generate(0, 100, 3)
    assert np.array_equal(generate(0, 100, 3), whole)          # repeatable
    assert np.array_equal(generate(40, 60, 3), whole[40:])     # offset-addressable
    assert set(np.unique(whole)) <= {-1.0, 1.0}
    assert abs(whole.mean()) < 0.2                             # roughly balanced
    assert not np.array_equal(est.rademacher(seed=78)(0, 100, 3), whole)


def test_within_solver_is_rejected(panel_path, tmp_path):
    """The `within` backend solves only for the design's own right-hand sides,
    so it cannot supply the operator; say so rather than failing obscurely."""
    est = fitted(panel_path, tmp_path / "within", FE_SETS[0], solver="within")
    with pytest.raises(NotImplementedError, match="explicit.*stream_cg"):
        est.project_rows(est.rademacher(), 1, lambda *a: None)


def test_reduced_operator_is_built_once(panel_path, tmp_path):
    """Hundreds of right-hand sides are the point, so the expensive part -- the
    reduced matrix -- must be cached across calls."""
    est = fitted(panel_path, tmp_path / "cache", FE_SETS[0])
    est.project_rows(est.rademacher(), 1, lambda *a: None)
    first = est._reduced()
    est.project_rows(est.rademacher(), 1, lambda *a: None)
    assert est._reduced() is first


def test_row_count_matches_the_fit(panel_path, tmp_path):
    est = fitted(panel_path, tmp_path / "count", FE_SETS[0])
    result = est.fit(panel_path)
    assert est.row_count() == result.n_obs


# --------------------------------------------------------------------------
# covariates in the design
# --------------------------------------------------------------------------

COVARIATE_CASES = [
    (["age_squared"], ["worker_id", "firm_id"], {}),
    (["age_squared", "age_cubed", "x2"], ["worker_id", "firm_id"], {}),
    (["age_squared", "x2"], ["worker_id", "firm_id", "year"], {}),
    (["age_squared", "x2"], ["worker_id", "firm_id"], {"solver": "stream_cg"}),
]
COVARIATE_IDS = ["1x-2fe", "3x-2fe", "2x-3fe", "2x-stream_cg"]


@pytest.fixture(scope="module")
def rich_path(tmp_path_factory):
    from hdfe_stream.simulate import simulate_rich
    path = tmp_path_factory.mktemp("cov") / "rich.parquet"
    simulate_rich(n_workers=N_WORKERS, n_firms=N_FIRMS, seed=7).write_parquet(path)
    return str(path)


def project_with(rich_path, workdir, xs, fe, covariates, **options):
    """Fit with covariates, project, and return everything for a dense check."""
    est = StreamingHDFE("log_earn", xs, fe, workdir=workdir, verbose=False,
                        keep_intermediates=True, tol=1e-13,
                        **{"solver": "explicit", **options})
    est.fit(rich_path)
    est.reload_intermediates()

    fe_columns = tuple(c for d in fe for c in d.split("^"))
    x_columns = tuple(f"v{est.vidx[n]}" for n in xs)
    got, drawn = [], []
    seen = {c: [] for c in fe_columns + x_columns}
    vectors = est.rademacher(seed=4)

    def sink(ordinal, chunk, out):
        got.append(out.copy())
        drawn.append(vectors(ordinal, len(out), 3))
        for c in seen:
            seen[c].append(np.asarray(chunk[c]))

    est.project_rows(vectors, 3, sink, extra_columns=fe_columns + x_columns,
                     covariates=covariates)
    return (np.vstack(got), np.vstack(drawn),
            {c: np.concatenate(v) for c, v in seen.items()},
            fe_columns, x_columns)


def dense_with_covariates(seen, fe_columns, x_columns, include_covariates):
    n = len(seen[fe_columns[0]])
    rows = np.arange(n)
    blocks = []
    for c in fe_columns:
        _, code = np.unique(seen[c], return_inverse=True)
        blocks.append(sp.csr_matrix((np.ones(n), (rows, code)),
                                    shape=(n, int(code.max()) + 1)))
    A = sp.hstack(blocks, format="csr").toarray()
    if include_covariates:
        A = np.column_stack([A] + [seen[c] for c in x_columns])
    return A, A @ np.linalg.pinv(A.T @ A, hermitian=True) @ A.T


@pytest.mark.parametrize("xs,fe,options", COVARIATE_CASES, ids=COVARIATE_IDS)
def test_projection_with_covariates_matches_dense(rich_path, tmp_path, xs, fe, options):
    """With covariates named, the operator must be the projection onto the full
    design, not onto the fixed effects alone. The two are genuinely different
    operators -- the test checks it matches one and not the other."""
    got, drawn, seen, fe_cols, x_cols = project_with(
        rich_path, tmp_path / "with", xs, fe, tuple(xs), **options)

    _, P_full = dense_with_covariates(seen, fe_cols, x_cols, True)
    _, P_fe = dense_with_covariates(seen, fe_cols, x_cols, False)
    reference = P_full @ drawn
    assert np.abs(got - reference).max() / np.abs(reference).max() < 1e-8
    # and is not accidentally the fixed-effect-only operator
    fe_only = P_fe @ drawn
    assert np.abs(got - fe_only).max() / np.abs(fe_only).max() > 1e-3


def test_empty_covariates_still_gives_the_fixed_effect_operator(rich_path, tmp_path):
    """Asking for no covariates on a model that has them is a legitimate
    request, and must give the fixed-effect design exactly."""
    got, drawn, seen, fe_cols, x_cols = project_with(
        rich_path, tmp_path / "without", ["age_squared"], ["worker_id", "firm_id"], ())
    _, P_fe = dense_with_covariates(seen, fe_cols, x_cols, False)
    reference = P_fe @ drawn
    assert np.abs(got - reference).max() / np.abs(reference).max() < 1e-10


def test_projection_with_covariates_and_weights(rich_path, tmp_path):
    est = StreamingHDFE("log_earn", ["age_squared", "x2"], ["worker_id", "firm_id"],
                        workdir=tmp_path / "w", verbose=False, keep_intermediates=True,
                        solver="explicit", tol=1e-13, weights="wa")
    est.fit(rich_path)
    est.reload_intermediates()

    fe_cols = ("worker_id", "firm_id")
    x_cols = tuple(f"v{est.vidx[n]}" for n in ("age_squared", "x2"))
    got, drawn = [], []
    seen = {c: [] for c in fe_cols + x_cols + ("w",)}
    vectors = est.rademacher(seed=4)

    def sink(ordinal, chunk, out):
        got.append(out.copy())
        drawn.append(vectors(ordinal, len(out), 3))
        for c in seen:
            seen[c].append(np.asarray(chunk[c]))

    est.project_rows(vectors, 3, sink, extra_columns=fe_cols + x_cols + ("w",),
                     covariates=("age_squared", "x2"))
    got, drawn = np.vstack(got), np.vstack(drawn)
    seen = {c: np.concatenate(v) for c, v in seen.items()}

    A, _ = dense_with_covariates(seen, fe_cols, x_cols, True)
    w = seen["w"]
    P = A @ np.linalg.pinv(A.T @ (w[:, None] * A), hermitian=True) @ A.T @ np.diag(w)
    reference = P @ drawn
    assert np.abs(got - reference).max() / np.abs(reference).max() < 1e-8


def test_unknown_covariate_is_rejected(rich_path, tmp_path):
    est = StreamingHDFE("log_earn", ["age_squared"], ["worker_id", "firm_id"],
                        workdir=tmp_path / "bad", verbose=False,
                        keep_intermediates=True, solver="explicit")
    est.fit(rich_path)
    est.reload_intermediates()
    with pytest.raises(ValueError, match="not variables of this fit"):
        est.project_rows(est.rademacher(), 1, lambda *a: None,
                         covariates=("nosuch",))


def test_covariate_border_is_computed_once(rich_path, tmp_path):
    """The border costs a pass over the rows and k solves, so hundreds of
    projections must share one."""
    est = StreamingHDFE("log_earn", ["age_squared"], ["worker_id", "firm_id"],
                        workdir=tmp_path / "cache", verbose=False,
                        keep_intermediates=True, solver="explicit", tol=1e-12)
    est.fit(rich_path)
    est.reload_intermediates()
    est.project_rows(est.rademacher(), 1, lambda *a: None, covariates=("age_squared",))
    first = est._border(("age_squared",))
    est.project_rows(est.rademacher(), 1, lambda *a: None, covariates=("age_squared",))
    assert est._border(("age_squared",)) is first
    # a different design gets its own border
    assert est._border(()) is not first


# --------------------------------------------------------------------------
# the coefficient-space entry point
# --------------------------------------------------------------------------

def dense_full_design(est, seen, gcode, fe, x_columns, weights=None):
    """The design in hdfe_stream's own parametrization: every level of every
    dimension kept, so S is singular exactly as the library leaves it."""
    n = len(gcode)
    rows = np.arange(n)
    offsets, total_levels = est._offsets()
    streamed = sp.csr_matrix((np.ones(n), (rows, gcode)),
                             shape=(n, int(gcode.max()) + 1)).toarray()
    levels = np.zeros((n, total_levels))
    for dim in fe[1:]:
        parts = dim.split("^")
        key = seen[parts[0]] if len(parts) == 1 else np.char.add(
            seen[parts[0]].astype(str), seen[parts[1]].astype(str))
        _, code = np.unique(key, return_inverse=True)
        levels[rows, offsets[dim] + code] = 1.0
    A = np.column_stack([streamed, levels] + [seen[c] for c in x_columns])
    w = np.ones(n) if weights is None else weights
    return A, A.T @ (w[:, None] * A)


def coefficient_case(rich_path, workdir, xs, fe, **options):
    est = StreamingHDFE("log_earn", xs, fe, workdir=workdir, verbose=False,
                        keep_intermediates=True, tol=1e-13,
                        **{"solver": "explicit", **options})
    est.fit(rich_path)
    est.reload_intermediates()

    fe_columns = tuple(c for d in fe for c in d.split("^"))
    x_columns = tuple(f"v{est.vidx[n]}" for n in xs)
    wanted = fe_columns + x_columns + (("w",) if options.get("weights") else ())
    seen = {c: [] for c in wanted}
    gcodes = []

    def sink(ordinal, chunk, out):
        gcodes.append(np.asarray(chunk["gcode"]))
        for c in wanted:
            seen[c].append(np.asarray(chunk[c]))

    est.project_rows(est.rademacher(), 1, sink, extra_columns=wanted,
                     covariates=tuple(xs))
    return (est, {c: np.concatenate(v) for c, v in seen.items()},
            np.concatenate(gcodes), x_columns)


@pytest.mark.parametrize("xs,fe,options", [
    ([], ["worker_id", "firm_id"], {}),
    (["age_squared", "x2"], ["worker_id", "firm_id"], {}),
    (["age_squared", "x2"], ["worker_id", "firm_id", "year"], {}),
    (["age_squared"], ["worker_id", "firm_id"], {"weights": "wa"}),
    (["age_squared"], ["worker_id", "firm_id"], {"solver": "stream_cg"}),
], ids=["fe-only", "2x", "3fe-2x", "weighted", "stream_cg"])
def test_apply_inverse_matches_the_dense_pseudo_inverse(rich_path, tmp_path,
                                                        xs, fe, options):
    """A right-hand side given in coefficient space, against dense.

    The comparison is against the *pseudo*-inverse, which is the right target:
    S is singular here, so an arbitrary coefficient-space vector has a component
    that no solution can account for.
    """
    est, seen, gcode, x_columns = coefficient_case(
        rich_path, tmp_path / "coef", xs, fe, **options)
    A, S = dense_full_design(est, seen, gcode, fe, x_columns,
                             seen.get("w") if options.get("weights") else None)

    rng = np.random.default_rng(3)
    _, total_levels = est._offsets()
    n_groups = int(gcode.max()) + 1
    v_levels = rng.normal(size=(total_levels, 3))
    v_streamed = rng.normal(size=(n_groups, 3))
    v_covariates = rng.normal(size=(len(xs), 3))

    got = []
    est.apply_inverse(v_levels, v_covariates, v_streamed, covariates=tuple(xs),
                      sink=lambda o, ch, out: got.append(out.copy()))
    got = np.vstack(got)

    v = np.vstack([v_streamed, v_levels, v_covariates])
    reference = A @ (np.linalg.pinv(S, hermitian=True) @ v)
    assert np.abs(got - reference).max() / np.abs(reference).max() < 1e-8


def test_the_null_space_is_what_the_design_says_it_is(rich_path, tmp_path):
    """Keeping every level of every dimension makes S singular by construction,
    and the null space is known structurally rather than discovered: one vector
    per connected component, plus one per dimension beyond the second.
    """
    est, seen, gcode, x_columns = coefficient_case(
        rich_path, tmp_path / "null", [], ["worker_id", "firm_id", "year"])
    A, S = dense_full_design(est, seen, gcode, ["worker_id", "firm_id", "year"],
                             x_columns)
    nullity = A.shape[1] - np.linalg.matrix_rank(S)

    q_streamed, q_levels = est._null_space()
    assert q_levels.shape[1] == nullity
    # and it really is the null space: A annihilates every basis vector
    basis = np.vstack([q_streamed, q_levels])
    assert np.abs(A[:, :basis.shape[0]] @ basis).max() < 1e-9


def test_a_right_hand_side_in_the_range_is_left_alone(rich_path, tmp_path):
    """The projection must not touch a vector that is already solvable -- only
    the part no solution could have accounted for."""
    est, seen, gcode, _ = coefficient_case(
        rich_path, tmp_path / "range", [], ["worker_id", "firm_id"])
    _, total_levels = est._offsets()
    n_groups = int(gcode.max()) + 1

    rng = np.random.default_rng(5)
    levels = rng.normal(size=(total_levels, 2))
    streamed = rng.normal(size=(n_groups, 2))
    projected_levels, projected_streamed, share = est._drop_null_component(
        levels, np.zeros((0, 2)), streamed)
    assert share > 1e-3                          # a random vector really does have one

    again_levels, again_streamed, share_again = est._drop_null_component(
        projected_levels, np.zeros((0, 2)), projected_streamed)
    assert share_again < 1e-12                   # and projecting twice changes nothing
    assert np.allclose(again_levels, projected_levels, atol=1e-12)
    assert np.allclose(again_streamed, projected_streamed, atol=1e-12)

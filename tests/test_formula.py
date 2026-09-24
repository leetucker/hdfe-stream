"""The formula front end, against pyfixest, over a battery of formulas.

Each case must reproduce pyfixest exactly: the same coefficient *names* in the
same order (the categorical coding has to agree, not just the numbers), the
same coefficients and standard errors, the same number of observations after
dropping rows with missing values, and the same set of terms dropped for
collinearity.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import (FE_DOF_PF, TOL_BETA, TOL_RESID, TOL_SE, ensure_interactions,
                      pf_feols, rel, resid_sorted, scaled)

pytest.importorskip("pyfixest")

from hdfe_stream import HDFEMulti  # noqa: E402
from hdfe_stream import feols_stream  # noqa: E402

# (formula, vcov, extra options). vcov=None exercises the default, which -- as
# in pyfixest -- clusters on the first fixed-effect dimension.
CASES = [
    # plain covariates
    ("log_earn ~ age_squared + age_cubed | worker_id + firm_id + year", None, {}),
    # transformations and an interaction between two numeric terms
    ("log_earn ~ I(age**2) + log(age) + age_squared:age_cubed | worker_id + firm_id + year",
     "hetero", {}),
    # event-study interaction with a reference year
    ("log_earn ~ i(year, treat, ref=2009) + age_squared | worker_id + firm_id",
     {"CRV1": "firm_id"}, {}),
    # categorical covariate, and a string categorical interacted with a numeric
    ("log_earn ~ C(occ) + cat:age_squared + age_squared | worker_id + firm_id + year",
     "iid", {}),
    # interacted fixed effects
    ("log_earn ~ age_squared | worker_id + firm_id^year", {"CRV1": "firm_id"}, {}),
    # clustering on a variable that is not a fixed effect
    ("log_earn ~ age_squared + age_cubed | worker_id + firm_id^year", {"CRV1": "state"}, {}),
    # doubly collinear: i(year) duplicates the year fixed effect, and `treat` is
    # constant within worker so C(treat) duplicates the worker fixed effect.
    # Both must be dropped, and the estimate on age_squared must survive intact.
    # See test_exactly_collinear_covariates_are_dropped for why this case warns
    # about solver convergence.
    ("log_earn ~ i(year) + C(treat) + age_squared | worker_id + firm_id + year", None, {}),
    # no covariates at all
    ("log_earn ~ 1 | worker_id + firm_id", "hetero", {}),
    # two interaction sets at once, solved in blocks narrower than the RHS
    ("log_earn ~ i(year, age_squared) + i(year, treat, ref=2005) | worker_id + firm_id",
     {"CRV1": "worker_id"}, {"rhs_block": 4}),
    # a covariate with missing values: rows must be dropped as pyfixest drops them
    ("log_earn ~ age_squared + x_na | worker_id + firm_id + year", {"CRV1": "state"}, {}),
    # interacted FE built from a string column
    ("log_earn ~ age_squared + i(year, treat, ref=2009) | worker_id^cat + firm_id + year",
     {"CRV1": "worker_id"}, {}),
    # multiple estimation: 2 outcomes x 2 covariate sets x 2 FE sets = 8 models
    ("log_earn + y2 ~ csw(age_squared, age_cubed) | "
     "sw(worker_id + firm_id, worker_id + firm_id + year)", "iid", {}),
]
IDS = [f"{i}" for i in range(len(CASES))]


def default_vcov(fml):
    """What `feols_stream` uses when vcov is not given: CRV1 on the first FE."""
    first_fe = fml.split("|")[1].split("+")[0].strip()
    return {"CRV1": first_fe}


class Fitted:
    """Lazily fits each case once and caches it, so the tests below can each
    look at a different aspect without paying for a refit."""

    def __init__(self, panel):
        self.panel = panel
        self._cache = {}

    def __call__(self, index):
        if index not in self._cache:
            fml, vcov, options = CASES[index]
            stream = feols_stream(fml, self.panel.src, workdir=self.workdir / str(index),
                                  vcov=vcov, verbose=False, n_buckets=3, batch_rows=5_000,
                                  fe_dof=FE_DOF_PF, tol=1e-11, outputs="keep", **options)
            ref = pf_feols(fml, self.panel,
                           vcov=vcov if vcov is not None else default_vcov(fml))
            models = list(stream) if isinstance(stream, HDFEMulti) else [stream]
            refs = (ref.all_fitted_models if hasattr(ref, "all_fitted_models")
                    else {models[0].fml: ref})
            self._cache[index] = (models, refs)
        return self._cache[index]

    def close(self):
        for models, _ in self._cache.values():
            for m in models:
                m.cleanup()


@pytest.fixture(scope="module")
def fitted(rich, tmp_path_factory):
    ensure_interactions(rich, ["firm_id^year", "worker_id^cat"])
    f = Fitted(rich)
    f.workdir = tmp_path_factory.mktemp("formula")
    yield f
    f.close()


@pytest.mark.parametrize("index", range(len(CASES)), ids=IDS)
def test_coefficient_names_match_pyfixest(fitted, index):
    """Names, in order. This is the strict test of the formula front end: it
    means the categorical coding and the term ordering agree with pyfixest,
    not merely that some set of numbers came out right."""
    models, refs = fitted(index)
    for model in models:
        assert model.coefnames == list(refs[model.fml].coef().index), model.fml


@pytest.mark.parametrize("index", range(len(CASES)), ids=IDS)
def test_coefficients_match_pyfixest(fitted, index):
    models, refs = fitted(index)
    for model in models:
        ref = refs[model.fml]
        assert rel(model.beta, ref.coef().to_numpy()) < TOL_BETA, model.fml


@pytest.mark.parametrize("index", range(len(CASES)), ids=IDS)
def test_standard_errors_match_pyfixest(fitted, index):
    models, refs = fitted(index)
    for model in models:
        ref = refs[model.fml]
        assert rel(model.se, ref.se().to_numpy()) < TOL_SE, model.fml


@pytest.mark.parametrize("index", range(len(CASES)), ids=IDS)
def test_sample_and_collinearity_match_pyfixest(fitted, index):
    """Same rows kept, and the same terms dropped as collinear."""
    models, refs = fitted(index)
    for model in models:
        ref = refs[model.fml]
        assert model.n_obs == ref._N, model.fml
        dropped_ref = getattr(ref, "_collin_vars", None) or []
        assert len(model.collin_vars) == len(dropped_ref), (
            model.fml, model.collin_vars, dropped_ref)


@pytest.mark.parametrize("index", range(len(CASES)), ids=IDS)
def test_residuals_match_pyfixest(fitted, index):
    models, refs = fitted(index)
    for model in models:
        ref = refs[model.fml]
        mine = resid_sorted(model)
        theirs = np.sort(np.asarray(ref.resid()))
        assert scaled(mine, theirs) < TOL_RESID, model.fml


def test_missing_values_drop_rows(fitted, rich):
    """The `x_na` case must lose exactly the rows where x_na is null."""
    models, _ = fitted(9)
    expected = rich.frame.height - rich.frame["x_na"].null_count()
    assert models[0].n_obs == expected
    assert rich.frame["x_na"].null_count() > 0      # the fixture really has gaps


def test_exactly_collinear_covariates_are_dropped(fitted):
    """Covariates spanned by the fixed effects are dropped, exactly as
    pyfixest drops them, and the surviving coefficient is unaffected.

    Two kinds appear here: `i(year)` duplicates the year fixed effect, and
    `C(treat)` duplicates the worker fixed effect because `treat` is constant
    within worker.

    The two are detected at different points. `C(treat)` is annihilated by the
    streamed fixed effect, so its right-hand side is numerically zero and the
    solver short-circuits it (see `test_stream_absorbed_covariate_short_circuits`).
    `i(year)` has a perfectly ordinary right-hand side -- its degeneracy is in
    the null space of the reduced system, which the solver handles normally --
    and it is caught by the rank check in step 3.
    """
    models, refs = fitted(6)
    model = models[0]
    ref = refs[model.fml]

    dropped = set(model.collin_vars)
    assert any(d.startswith("year::") for d in dropped), dropped
    assert any("treat" in d for d in dropped), dropped
    assert dropped == set(getattr(ref, "_collin_vars", None) or [])

    assert model.coefnames == ["age_squared"]


def test_stream_absorbed_covariate_short_circuits(fitted):
    """A covariate the streamed fixed effect absorbs entirely must not cost the
    solver anything.

    Its right-hand side is zero up to round-off, so the relative convergence
    test can never be satisfied and the solve would run to `maxiter` -- while
    already holding the right answer, since b = 0 means Gamma = 0. The solver
    reports such variables in `fe_spanned` and still converges.
    """
    models, _ = fitted(6)
    info = models[0].solver_info

    assert info["fe_spanned"] == ["C(treat)[T.1]"]
    assert info["converged"]
    iterations = info["iterations"]
    iterations = iterations if isinstance(iterations, list) else [iterations]
    assert max(iterations) < 500, iterations


def test_no_short_circuit_without_an_absorbed_covariate(fitted):
    """The check must not fire on ordinary covariates."""
    for index in (0, 1, 2):
        models, _ = fitted(index)
        for model in models:
            assert "fe_spanned" not in model.solver_info, (index, model.fml)
            assert model.solver_info["converged"], (index, model.fml)


def test_multiple_estimation_expands_to_all_models(fitted):
    """`y1 + y2 ~ csw(...) | sw(...)` is 2 x 2 x 2 models, and models sharing a
    fixed-effect set share one solve."""
    models, refs = fitted(11)
    assert len(models) == 8
    assert len(refs) == 8
    assert {m.depvar for m in models} == {"log_earn", "y2"}
    assert len({m.fml for m in models}) == 8

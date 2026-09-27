"""Validation for the exact KSS prototype.

Run with:  pytest prototypes/

Three kinds of check, in increasing order of how much they tell you:

1. algebraic identities that must hold for any correct implementation
   (leverages sum to the rank, lie in [0, 1], and so on);
2. **agreement with known truth**, by Monte Carlo. The simulator draws the
   worker and firm effects explicitly, so the estimand is known exactly. This
   is the test that actually establishes the estimator is right: the plug-in
   estimator must be biased, in the documented direction, and the corrected one
   must not be;
3. agreement with pytwoway, an independent implementation of the same
   estimator, on an identical sample. Skipped unless the reference environment
   has been built -- see prototypes/README.md.

(2) is the load-bearing one. (3) is a cross-check, and it found a disagreement
worth knowing about -- see `test_matches_pytwoway_approximate_trace`.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from hdfe_stream.simulate import simulate_akm          # noqa: E402
from kss_exact import (Panel, design, kss, largest_connected_set,  # noqa: E402
                       leverages, prune_to_leave_out)

KEYS = ["var(psi)", "var(alpha)", "cov(psi, alpha)"]
SD_NOISE = 0.3
REF_VENV = Path(__file__).resolve().parent / ".ref-venv" / "bin" / "python"


def make_structure(n_workers=250, n_firms=25, structure_seed=11):
    """A pruned worker-firm structure with the true effects attached.

    The worker-firm structure comes from the simulator (realistic mobility and
    assortative matching), but the outcome is rebuilt by `with_noise` as

        y = alpha + psi + noise

    with no age profile. That matters: the simulator's own `log_earn` contains an
    age term, and a bare two-way model has nowhere to put it, so the estimated
    worker effect would absorb each worker's mean age effect and would not be
    estimating `alpha`. Rebuilding y makes the model correctly specified, which
    is what lets ground truth be compared against at all.

    Pruning happens here, once: leverage depends only on the design, not on the
    outcome, so the leave-one-out sample is the same for every noise draw. That
    also keeps the estimand fixed across replications.
    """
    base = simulate_akm(n_workers=n_workers, n_firms=n_firms,
                        seed=structure_seed, keep_effects=True)
    coded = Panel.from_frame(base, y="log_earn")
    panel = Panel(worker=coded.worker, firm=coded.firm,
                  y=np.zeros(coded.n_obs),          # replaced by with_noise
                  extra={"alpha": base["true_worker_effect"].to_numpy(),
                         "psi": base["true_firm_effect"].to_numpy()})
    panel, _ = prune_to_leave_out(panel)
    return panel


def with_noise(structure, noise_seed=0, sd_noise=SD_NOISE):
    """One draw of the outcome on a fixed structure."""
    rng = np.random.default_rng(noise_seed)
    y = (structure.extra["alpha"] + structure.extra["psi"]
         + rng.normal(0, sd_noise, structure.n_obs))
    return Panel(worker=structure.worker, firm=structure.firm, y=y,
                 extra=structure.extra)


def make_panel(n_workers=250, n_firms=25, structure_seed=11, noise_seed=0,
               sd_noise=SD_NOISE):
    return with_noise(make_structure(n_workers, n_firms, structure_seed),
                      noise_seed, sd_noise)


def truth_of(panel):
    """The estimand: observation-level moments of the true effects, on the
    sample that survived pruning."""
    alpha, psi = panel.extra["alpha"], panel.extra["psi"]
    return {"var(psi)": float(np.var(psi)),
            "var(alpha)": float(np.var(alpha)),
            "cov(psi, alpha)": float(np.cov(psi, alpha, ddof=0)[0, 1])}


# --------------------------------------------------------------------------
# 1. algebraic identities
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def panel():
    return make_panel()


def test_leverages_sum_to_the_rank(panel):
    """sum_i P_ii = trace(P) = rank(A). A hat matrix is a projection, so its
    trace is the dimension it projects onto."""
    A, _, _ = design(panel)
    P_ii = leverages(A)
    rank = np.linalg.matrix_rank(A.toarray())
    assert P_ii.sum() == pytest.approx(rank, rel=1e-9)


def test_leverages_are_in_the_unit_interval(panel):
    A, _, _ = design(panel)
    P_ii = leverages(A)
    assert P_ii.min() > 0
    assert P_ii.max() < 1.0            # guaranteed by prune_to_leave_out


def test_sigma2_recovers_the_error_variance(panel):
    """The leave-one-out estimates average to the true error variance. This is
    the single most informative one-line check of the correction: it is the
    quantity the whole bias term is built from."""
    res = kss(panel)
    assert res.sigma2_mean == pytest.approx(SD_NOISE ** 2, rel=0.1)


def test_estimated_bias_term_has_the_expected_sign(panel):
    """Estimation error inflates the variances and attenuates the covariance, so
    the correction subtracts a positive amount from the variances and a negative
    amount from the covariance.

    This is asserted on the estimated bias *term*, which is a deterministic
    function of the data. It is deliberately not asserted by comparing a single
    draw's plug-in estimate against the truth: the bias is a property of the
    expectation, and on one draw sampling noise can and does swamp it. That
    comparison is made over replications in `test_plug_in_is_detectably_biased`.
    """
    res = kss(panel)
    assert res.bias["var(psi)"] > 0
    assert res.bias["var(alpha)"] > 0
    assert res.bias["cov(psi, alpha)"] < 0
    assert res.kss["var(psi)"] < res.plug_in["var(psi)"]
    assert res.kss["var(alpha)"] < res.plug_in["var(alpha)"]
    assert res.kss["cov(psi, alpha)"] > res.plug_in["cov(psi, alpha)"]


def test_components_are_invariant_to_which_firm_is_dropped(panel):
    """The normalization sets one firm effect to zero; the variance components
    are location-invariant and must not depend on which firm that is."""
    reversed_firms = Panel(worker=panel.worker,
                           firm=panel.n_firms - 1 - panel.firm,
                           y=panel.y, extra=panel.extra)
    a = kss(panel)
    b = kss(reversed_firms)
    for key in KEYS:
        assert a.kss[key] == pytest.approx(b.kss[key], rel=1e-7), key
        assert a.plug_in[key] == pytest.approx(b.plug_in[key], rel=1e-9), key


def test_pruning_reaches_a_fixed_point(panel):
    """Re-pruning an already-pruned panel must change nothing."""
    again, rounds = prune_to_leave_out(panel)
    assert rounds == 0
    assert again.n_obs == panel.n_obs


def test_panel_that_is_not_leave_one_out_connected_is_rejected():
    """A panel with a bridge observation must fail with an explanation rather
    than silently producing nonsense.

    Constructed so the condition is certain, not left to chance: worker 1 is the
    only worker ever seen at firm 1, so that single observation is the only thing
    identifying both its own worker effect and firm 1's effect. Its leverage is
    exactly 1 and its leave-one-out residual is undefined.
    """
    raw = Panel(worker=np.array([0, 0, 1, 1, 2, 2]),
                firm=np.array([0, 0, 0, 1, 0, 0]),
                y=np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
    raw = largest_connected_set(raw)
    A, _, _ = design(raw)
    assert leverages(A).max() > 1 - 1e-9, "the bridge observation should have P_ii = 1"
    with pytest.raises(ValueError, match="leave-one-out connected"):
        kss(raw)

    pruned, rounds = prune_to_leave_out(raw)
    assert rounds >= 1
    assert pruned.n_obs < raw.n_obs


# --------------------------------------------------------------------------
# 2. unbiasedness against known truth
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def monte_carlo():
    """Repeated noise draws on one fixed worker-firm structure.

    Holding the structure fixed holds the estimand fixed, so the average of the
    estimates can be compared against a single known number.
    """
    reps = 50
    structure = make_structure()
    truth = truth_of(structure)
    plug_in, corrected = [], []
    for rep in range(reps):
        res = kss(with_noise(structure, noise_seed=1000 + rep))
        plug_in.append([res.plug_in[k] for k in KEYS])
        corrected.append([res.kss[k] for k in KEYS])
    return (np.array(plug_in), np.array(corrected), truth, reps, structure.n_obs)


@pytest.mark.parametrize("index,key", list(enumerate(KEYS)))
def test_correction_is_unbiased(monte_carlo, index, key):
    """The headline test: averaged over noise draws, the corrected estimator
    lands on the truth. Tolerance is three Monte Carlo standard errors.
    """
    plug_in, corrected, truth, reps, _ = monte_carlo
    column = corrected[:, index]
    bias = column.mean() - truth[key]
    se = column.std(ddof=1) / np.sqrt(reps)
    assert abs(bias) < 3 * se, (
        f"{key}: corrected mean {column.mean():.6f} vs truth {truth[key]:.6f}, "
        f"bias {bias:+.6f} = {bias / se:.1f} Monte Carlo standard errors")


@pytest.mark.parametrize("index,key", list(enumerate(KEYS)))
def test_plug_in_is_detectably_biased(monte_carlo, index, key):
    """The other half of the claim: there was a bias to remove. If this failed,
    the test above would be vacuous.
    """
    plug_in, corrected, truth, reps, _ = monte_carlo
    column = plug_in[:, index]
    bias = column.mean() - truth[key]
    se = column.std(ddof=1) / np.sqrt(reps)
    assert abs(bias) > 3 * se, f"{key}: plug-in bias {bias:+.6f} is not detectable"
    # and in the documented direction
    assert (bias > 0) == key.startswith("var"), key


def test_correction_beats_the_plug_in_on_average(monte_carlo):
    """Root mean squared error, over the replications, for each component."""
    plug_in, corrected, truth, _, _ = monte_carlo
    target = np.array([truth[k] for k in KEYS])
    rmse_pi = np.sqrt(((plug_in - target) ** 2).mean(axis=0))
    rmse_kss = np.sqrt(((corrected - target) ** 2).mean(axis=0))
    assert np.all(rmse_kss < rmse_pi), dict(zip(KEYS, zip(rmse_pi, rmse_kss)))


# --------------------------------------------------------------------------
# 3. cross-check against pytwoway
# --------------------------------------------------------------------------

def run_pytwoway(panel, tmp_path, exact_trace=True):
    """Run pytwoway on `panel` under its own interpreter; return (result, sample).

    pytwoway also hands back the sample after its own leave-one-out pruning, and
    the comparison is run on *that*, so any difference in sample selection
    cannot be mistaken for a difference in the estimator.
    """
    if not REF_VENV.exists():
        pytest.skip(f"reference environment not built at {REF_VENV}; "
                    "see prototypes/README.md")
    import pandas as pd

    raw = tmp_path / "raw.csv"
    cleaned = tmp_path / "cleaned.csv"
    pd.DataFrame({"i": panel.worker, "j": panel.firm,
                  "t": np.arange(panel.n_obs) % 10, "y": panel.y}
                 ).to_csv(raw, index=False)

    script = Path(__file__).resolve().parent / "_pytwoway_reference.py"
    command = [str(REF_VENV), str(script), str(raw), str(cleaned)]
    if not exact_trace:
        command.append("--approximate-trace")
    proc = subprocess.run(command, capture_output=True, text=True,
                          env=dict(os.environ, PYTHONWARNINGS="ignore"))
    if proc.returncode != 0:
        pytest.fail(f"pytwoway reference failed:\n{proc.stderr[-2000:]}")
    line = next(ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT "))
    reference = json.loads(line[len("RESULT "):])

    frame = pd.read_csv(cleaned)
    same_sample = Panel(worker=frame["i"].to_numpy(), firm=frame["j"].to_numpy(),
                        y=frame["y"].to_numpy(dtype=float)).recode()
    return reference, same_sample


@pytest.fixture(scope="module")
def pytwoway_exact(tmp_path_factory):
    panel = make_panel(n_workers=400, n_firms=40, structure_seed=5, noise_seed=99)
    return run_pytwoway(panel, tmp_path_factory.mktemp("pt_exact"), exact_trace=True)


def test_matches_pytwoway_on_leverages_and_sigma2(pytwoway_exact):
    """The core machinery: leverages and the leave-one-out variance estimates.

    These are what everything else is built from, and they are computed by two
    completely independent code paths, so agreement here is the strongest
    cross-check available.
    """
    reference, panel = pytwoway_exact
    mine = kss(panel)
    assert mine.max_leverage == pytest.approx(reference["max_lev"], rel=1e-8)
    assert mine.sigma2_mean == pytest.approx(reference["var_eps_he"], rel=1e-8)


def test_matches_pytwoway_on_plug_in_estimates(pytwoway_exact):
    reference, panel = pytwoway_exact
    mine = kss(panel)
    for key in KEYS:
        assert mine.plug_in[key] == pytest.approx(reference["plug_in"][key],
                                                 rel=1e-8), key


def test_matches_pytwoway_exact_trace_for_var_psi(pytwoway_exact):
    """var(psi) is the component pytwoway's exact-trace path gets right, and
    there the two implementations agree to machine precision."""
    reference, panel = pytwoway_exact
    mine = kss(panel)
    assert mine.bias["var(psi)"] == pytest.approx(reference["bias"]["var(psi)"],
                                                 rel=1e-7)
    assert mine.kss["var(psi)"] == pytest.approx(reference["kss"]["var(psi)"],
                                                rel=1e-7)


def test_pytwoway_exact_trace_disagrees_on_var_alpha(pytwoway_exact):
    """A disagreement, recorded deliberately rather than worked around.

    With `exact_trace_he=True`, pytwoway 0.3.21 returns a bias term for
    var(alpha) roughly equal to the whole plug-in value, so its corrected
    var(alpha) comes out *negative* -- impossible for a variance. Its own
    approximate (Johnson-Lindenstrauss) path does not do this and agrees with
    the exact computation here to about 1%, and the Monte Carlo above shows this
    implementation is unbiased. So the disagreement is on their side, in that
    configuration.

    This test pins the observation so that a future version of pytwoway that
    fixes it will make the test fail loudly and prompt a re-check, rather than
    the discrepancy being quietly carried in a comment forever.
    """
    reference, panel = pytwoway_exact
    mine = kss(panel)
    assert reference["kss"]["var(alpha)"] < 0, (
        "pytwoway's exact-trace var(alpha) is no longer negative -- it may have "
        "been fixed; re-validate and tighten this test")
    assert mine.kss["var(alpha)"] > 0
    assert reference["bias"]["var(alpha)"] > 5 * mine.bias["var(alpha)"]


@pytest.fixture(scope="module")
def pytwoway_approximate(tmp_path_factory):
    panel = make_panel(n_workers=400, n_firms=40, structure_seed=5, noise_seed=99)
    return run_pytwoway(panel, tmp_path_factory.mktemp("pt_jla"), exact_trace=False)


@pytest.mark.parametrize("key", KEYS)
def test_matches_pytwoway_approximate_trace(pytwoway_approximate, key):
    """Against pytwoway's Johnson-Lindenstrauss path, which is the one it uses
    at scale and the one the eventual streaming implementation will mirror.

    The tolerance is loose because the reference is itself a random
    approximation with a few hundred draws; the point is that the exact
    computation sits inside its sampling noise for every component, including
    the var(alpha) one where the exact path disagrees.
    """
    reference, panel = pytwoway_approximate
    mine = kss(panel)
    assert mine.kss[key] == pytest.approx(reference["kss"][key], rel=0.05), key

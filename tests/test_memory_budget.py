"""The per-thread accumulators of `stream_cg` and the leave-out passes are
capped by `max_s_gb`: fewer copies, and so fewer threads on those passes,
when the budget is tight. The copies are a way of splitting the work, so the
answer must not depend on how many there are."""

from __future__ import annotations

import numba as nb
import pytest

from conftest import TOL_BETA, TOL_SE, rel

from hdfe_stream import StreamingHDFE, feols_stream, leave_out_kss

FML = "log_earn ~ age_squared + age_cubed | worker_id + firm_id + year"


def test_copies_follow_the_budget(workdir):
    est = StreamingHDFE("y", ["x"], ["worker_id", "firm_id"], workdir=workdir,
                        max_s_gb=1.0)
    nt = nb.get_num_threads()
    # 1,000 levels x 8 columns is 64 kB a copy: one per thread fits easily
    assert est._acc_copies(1_000, 8) == nt
    # half the 1 GB budget holds 6 copies of 10M levels x 1 column (80 MB each)
    assert est._acc_copies(10_000_000, 1) == min(nt, 6)
    # one copy always, even when a single one is over the budget
    assert est._acc_copies(10**9, 8) == 1


def test_stream_cg_with_one_accumulator_matches_explicit(akm, workdir):
    """A budget too small for S sends `solver="auto"` to stream_cg, and too
    small for a second accumulator runs its passes on one thread: the fit is
    the explicit solver's all the same."""
    explicit = feols_stream(FML, akm.src, workdir=workdir / "e", verbose=False,
                            vcov={"CRV1": "worker_id"})
    tight = feols_stream(FML, akm.src, workdir=workdir / "t", verbose=False,
                         vcov={"CRV1": "worker_id"}, max_s_gb=1e-9)
    assert explicit.solver_info["solver"] == "explicit"
    assert tight.solver_info["solver"] == "stream_cg"
    assert rel(tight.beta, explicit.beta) < TOL_BETA
    assert rel(tight.se, explicit.se) < TOL_SE


@pytest.mark.parametrize("leave_out", ["match", "observation"])
def test_leave_out_with_one_accumulator_matches(akm, workdir, leave_out):
    """The leave-out passes split their groups among the accumulator copies;
    one copy gives the same components, standard errors and diagnostic. The
    solver is fixed so that only the copies change."""
    fml = "log_earn ~ age_squared | worker_id + firm_id"
    kw = dict(n_draws=16, seed=3, se=True, leave_out=leave_out, verbose=False,
              solver="stream_cg")
    wide = leave_out_kss(fml, akm.src, workdir=workdir / "w", **kw)
    narrow = leave_out_kss(fml, akm.src, workdir=workdir / "n", max_s_gb=1e-9, **kw)
    for key in wide.leave_out:
        assert rel(narrow.leave_out[key], wide.leave_out[key]) < 1e-8, key
        assert rel(narrow.plug_in[key], wide.plug_in[key]) < 1e-8, key
        if wide.se.se[key] == wide.se.se[key]:          # not nan
            assert rel(narrow.se.se[key], wide.se.se[key]) < 1e-6, key

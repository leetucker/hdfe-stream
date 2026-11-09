"""Collapsing rows into cells (`collapse_cells`), and the right-hand side of
the solve that pass 1 builds.

Collapsing the rows into one cell per combination of fixed-effect levels is a
compression, not part of the model, so a fit that skips it -- leaving the rows
unsorted by cell, so that each is a cell of its own unless an identical row
happens to be next to it -- must give the same estimates, standard errors,
residuals and fixed effects. These tests check that on both kinds of panel:
one where every row is its own cell anyway (worker, firm and year effects on a
worker-year panel), where "auto" does not sort the rows into cells, and one
where cells repeat (worker and firm effects), where it does.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from conftest import TOL_BETA, TOL_SE, rel, scaled

from hdfe_stream import StreamingHDFE, fepois_stream, feols_stream

UNIQUE = "log_earn ~ age_squared + age_cubed | worker_id + firm_id + year"
REPEATED = "log_earn ~ age_squared + age_cubed | worker_id + firm_id"
WEIGHTS = {"none": {}, "aweights": {"weights": "wa"},
           "fweights": {"weights": "wf", "weights_type": "fweights"}}


def _fit(fml, panel, workdir, tag, **options):
    return feols_stream(fml, panel.src, workdir=workdir / tag, verbose=False,
                        vcov={"CRV1": "worker_id"}, cluster=["firm_id"], **options)


def _resid(result):
    return (result.resid().select("row_id", "resid").collect()
                  .sort("row_id")["resid"].to_numpy())


def _assert_same_fit(a, b):
    assert a.coefnames == b.coefnames
    assert rel(a.beta, b.beta) < TOL_BETA
    for key in ("iid", "hetero", "CRV1:worker_id", "CRV1:firm_id"):
        assert rel(a.with_vcov(key).se, b.with_vcov(key).se) < TOL_SE, key
    assert a.n_obs == b.n_obs and a.k_fe == b.k_fe and a.df_t == b.df_t
    assert a.n_components == b.n_components
    assert scaled(_resid(a), _resid(b)) < 1e-8
    for dim in ("firm_id", "worker_id"):
        fa = a.fixef(dim).sort(dim).collect()
        fb = b.fixef(dim).sort(dim).collect()
        assert scaled(fa[f"fe_{dim}"], fb[f"fe_{dim}"]) < 1e-8, dim


@pytest.mark.parametrize("weights", list(WEIGHTS))
@pytest.mark.parametrize("solver", ["explicit", "stream_cg"])
def test_rows_as_cells_give_the_same_fit(rich, workdir, weights, solver):
    """Every row its own cell, or the rows collapsed: the same fit, with
    analytic, frequency or no weights, under either solver."""
    options = dict(WEIGHTS[weights], solver=solver)
    rows = _fit(UNIQUE, rich, workdir, "rows", collapse_cells=False, **options)
    cells = _fit(UNIQUE, rich, workdir, "cells", collapse_cells=True, **options)
    assert rows.diagnostics["cells_collapsed"] is False
    assert cells.diagnostics["cells_collapsed"] is True
    _assert_same_fit(rows, cells)


@pytest.mark.parametrize("assembly", ["cells", "rows"])
def test_rows_as_cells_with_either_assembly(rich, workdir, assembly):
    """With `assembly="cells"` the within-cell cross-products are zero when
    every row is a cell; with "rows" they come from a row pass either way."""
    rows = _fit(UNIQUE, rich, workdir, "rows", collapse_cells=False, assembly=assembly)
    cells = _fit(UNIQUE, rich, workdir, "cells", collapse_cells=True, assembly=assembly)
    _assert_same_fit(rows, cells)


def test_rows_as_cells_where_cells_repeat(rich, workdir):
    """Not collapsing is still right when combinations repeat (a worker's
    years at one firm): the cells are only more numerous than they could be."""
    rows = _fit(REPEATED, rich, workdir, "rows", collapse_cells=False)
    cells = _fit(REPEATED, rich, workdir, "cells", collapse_cells=True)
    assert cells.diagnostics["cells_per_obs"] < 0.5
    assert rows.diagnostics["cells_per_obs"] > cells.diagnostics["cells_per_obs"]
    _assert_same_fit(rows, cells)


def test_rows_as_cells_with_varying_slopes(trends, workdir):
    """The cells carry the slope moments too; collapsed or not, the same fit."""
    kw = dict(workdir=workdir, verbose=False, vcov={"CRV1": "worker_id"})
    rows = feols_stream("y ~ x | worker_id[t] + firm_id", trends.src,
                        collapse_cells=False, **{**kw, "workdir": workdir / "r"})
    cells = feols_stream("y ~ x | worker_id[t] + firm_id", trends.src,
                         collapse_cells=True, **{**kw, "workdir": workdir / "c"})
    assert rel(rows.beta, cells.beta) < TOL_BETA
    assert rel(rows.se, cells.se) < TOL_SE
    assert rows.k_fe == cells.k_fe


def test_auto_skips_the_collapse_only_when_rows_are_distinct(rich, workdir):
    """"auto" uses the rows as cells on a worker-year panel with year
    effects, where they are all distinct, and collapses them with worker and
    firm effects alone, where they repeat."""
    unique = _fit(UNIQUE, rich, workdir, "unique")
    repeated = _fit(REPEATED, rich, workdir, "repeated")
    assert unique.diagnostics["cells_collapsed"] is False
    assert repeated.diagnostics["cells_collapsed"] is True


@pytest.fixture
def no_room_for_wide_rhs(monkeypatch):
    """The memory budget refuses pass 1's right-hand side for more variables
    than one block, so the solve builds it block by block from the cell sums."""
    from stream_limits import MemoryBudget
    monkeypatch.setattr(MemoryBudget, "fits", lambda *args, **kw: False)


def test_rhs_built_in_pass_1b_matches_the_blockwise_one(rich, workdir, no_room_for_wide_rhs):
    """Pass 1 builds the right-hand side when every variable fits in one
    solve block; with `rhs_block=1` and no room for it, the solve builds it
    block by block, as before. Both give the same fit, for collapsed cells
    and for rows."""
    for collapse in (True, False):
        one = _fit(UNIQUE, rich, workdir, f"one{collapse}", collapse_cells=collapse)
        blocks = _fit(UNIQUE, rich, workdir, f"blocks{collapse}", collapse_cells=collapse,
                      rhs_block=1)
        _assert_same_fit(one, blocks)


def test_rhs_for_more_variables_than_a_block_is_built_in_pass_1(rich, workdir, monkeypatch):
    """With room for it, pass 1 builds the right-hand side of all the
    variables though they take several solve blocks (the solve builds none
    from the cell sums), and the fit is the same."""
    from hdfe_stream.core import solve
    calls = []
    original = solve._nb_rhs
    monkeypatch.setattr(solve, "_nb_rhs", lambda *a: (calls.append(1), original(*a))[1])
    for collapse in (True, False):
        one = _fit(UNIQUE, rich, workdir, f"one{collapse}", collapse_cells=collapse)
        blocks = _fit(UNIQUE, rich, workdir, f"wide{collapse}", collapse_cells=collapse,
                      rhs_block=1)
        assert not calls
        _assert_same_fit(one, blocks)


def _sums_files(workdir, tag):
    return list((workdir / tag).rglob("ident_sums.f64"))


def test_cell_sums_are_not_written_when_nothing_reads_them(rich, workdir):
    """Rows assembly, the default solver and a right-hand side built in pass 1
    leave the file of cell sums unread, so it is not written; the fit is the
    one that writes it."""
    kept = dict(keep_intermediates=True, rhs_block=1)
    skipped = _fit(UNIQUE, rich, workdir, "skipped", assembly="rows", **kept)
    assert not _sums_files(workdir, "skipped")
    written = _fit(UNIQUE, rich, workdir, "written", assembly="cells", **kept)
    assert _sums_files(workdir, "written")
    _assert_same_fit(skipped, written)


@pytest.mark.parametrize("options", [{"solver": "within"}, {"assembly": "cells"}])
def test_cell_sums_are_written_when_the_solver_or_assembly_reads_them(
        rich, workdir, options):
    if options.get("solver") == "within":
        pytest.importorskip("within")
    fit = _fit(UNIQUE, rich, workdir, "read", keep_intermediates=True, rhs_block=1, **options)
    assert _sums_files(workdir, "read") and fit is not None


def test_rhs_goes_to_a_file_when_it_does_not_fit_in_memory(
        rich, workdir, no_room_for_wide_rhs, monkeypatch):
    """With no room for the right-hand side in memory, pass 1 accumulates it in
    a memory-mapped file instead: the cell sums are not written for step 2 to
    read back, the file is deleted after the solve, and the fit is the same."""
    opened = []
    original = np.lib.format.open_memmap
    monkeypatch.setattr(np.lib.format, "open_memmap",
                        lambda path, *a, **k: (opened.append(str(path)), original(path, *a, **k))[1])
    one = _fit(UNIQUE, rich, workdir, "one")     # fewer variables than a block: in memory
    assert not [p for p in opened if p.endswith("rhs.npy")]
    from hdfe_stream.core.passes import _PassesMixin
    handed_on = []
    pass1 = _PassesMixin._pass1_cells
    monkeypatch.setattr(_PassesMixin, "_pass1_cells",
                        lambda self: (pass1(self), handed_on.append(self._rhs_pass1))[1])
    wide = _fit(UNIQUE, rich, workdir, "wide", keep_intermediates=True, assembly="rows",
                rhs_block=1)
    assert [p for p in opened if p.endswith("rhs.npy")]
    # pass 1 unmaps the file (its pages would count as resident memory in the
    # budget's reading) and hands step 2 its path
    assert handed_on and all(isinstance(h, str) and h.endswith("rhs.npy") for h in handed_on)
    assert not _sums_files(workdir, "wide")
    assert not list((workdir / "wide").rglob("rhs.npy"))
    _assert_same_fit(one, wide)


def test_intermediates_reload_without_the_cell_sums(rich, workdir):
    """The leave-out operator reloads a kept fit's cell arrays; without the
    file of cell sums it still works, from the reduced system."""
    est = StreamingHDFE("log_earn", ["age_squared", "age_cubed"], ["worker_id", "firm_id", "year"], workdir=workdir / "reload",
                        verbose=False, keep_intermediates=True, assembly="rows",
                        rhs_block=1, solver="explicit")
    est.fit(rich.src)
    assert not _sums_files(workdir, "reload")
    est.reload_intermediates()
    assert est.sums is None and est.codes is not None
    est.unload_intermediates()


def test_rhs_built_in_pass_1b_with_varying_slopes(trends, workdir, no_room_for_wide_rhs):
    """The same, with the varying-slopes projection in the right-hand side."""
    one = feols_stream("y ~ x | worker_id[t] + firm_id", trends.src,
                       workdir=workdir / "one", verbose=False)
    blocks = feols_stream("y ~ x | worker_id[t] + firm_id", trends.src,
                          workdir=workdir / "blocks", verbose=False, rhs_block=1)
    assert rel(one.beta, blocks.beta) < TOL_BETA
    assert rel(one.se, blocks.se) < TOL_SE


@pytest.mark.parametrize("options", [
    {"keep_intermediates": True},
    {"fe": ["worker_id"]},
])
def test_collapse_cells_false_is_refused_where_not_implemented(workdir, options):
    """Leaving the rows unsorted by cell is a choice for linear models with two
    or more fixed effects whose intermediates are not kept; asking for it
    elsewhere is an error rather than a silent collapse."""
    kw = {"fe": ["worker_id", "firm_id"], **options}
    with pytest.raises(ValueError, match="collapse_cells=False"):
        StreamingHDFE("y", ["x"], workdir=workdir, collapse_cells=False, **kw)


def test_collapse_cells_false_is_refused_for_glms(akm, workdir):
    frame = akm.frame.with_columns(visits=(pl.col("log_earn") > 10).cast(pl.Float64))
    with pytest.raises(ValueError, match="collapse_cells=False"):
        fepois_stream("visits ~ age_squared | worker_id + firm_id", frame,
                      workdir=workdir, collapse_cells=False, verbose=False)


def test_collapse_cells_must_be_auto_true_or_false(workdir):
    with pytest.raises(ValueError, match="collapse_cells must be"):
        StreamingHDFE("log_earn", ["age_squared", "age_cubed"], ["worker_id", "firm_id", "year"], workdir=workdir,
                      collapse_cells="never")

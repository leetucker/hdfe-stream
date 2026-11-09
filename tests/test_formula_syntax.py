"""Formula syntax that depends on the pyfixest generation, and the spellings
that must mean the same thing on both.

pyfixest 0.70 writes fixed-effect interactions as ``a:b``, instrument blocks
as ``y ~ x + [endog ~ z] | fe``, and several outcomes as ``sw(y, y2)``; 0.60
has ``a^b``, ``y ~ x | fe | endog ~ z`` and ``y + y2``. hdfe_stream accepts
``a^b`` and ``a:b`` with either, and the instrument block and the expression
slopes (``fe[x ** 2]``) with 0.70 only.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pyfixest")

from hdfe_stream import feols_stream  # noqa: E402
from hdfe_stream.core.feterms import interactions_to_caret, rewrite_fe_interactions  # noqa: E402
from hdfe_stream.core.formula import _new_parser, parse_formula, pyfixest_formula_api  # noqa: E402

NEW_PARSER = _new_parser(pyfixest_formula_api()[0])
needs_070 = pytest.mark.skipif(not NEW_PARSER, reason="needs pyfixest 0.70 or later")


# ------------------------------------------------------------ text rewriting

def test_rewrite_changes_only_the_fixed_effect_part():
    assert (rewrite_fe_interactions("y ~ a:b | f^g[t] + h", ":")
            == "y ~ a:b | f:g[t] + h")
    assert (rewrite_fe_interactions("y ~ a:b | f:g + h:k", "^")
            == "y ~ a:b | f^g + h^k")
    # a second `|` part (the legacy instrument block) is left alone
    assert (rewrite_fe_interactions("y ~ x | f^g | e ~ z:w", ":")
            == "y ~ x | f:g | e ~ z:w")


def test_rewrite_leaves_formulas_without_fixed_effects_alone():
    assert rewrite_fe_interactions("y ~ a:b + c", "^") == "y ~ a:b + c"


def test_rewrite_ignores_separators_inside_brackets_and_calls():
    assert (rewrite_fe_interactions("y ~ x | f[g:h] + i(j, k)", "^")
            == "y ~ x | f[g:h] + i(j, k)")


def test_interactions_to_caret():
    assert interactions_to_caret("w[t] + f:g") == "w[t] + f^g"
    assert interactions_to_caret("a + b") == "a + b"


# -------------------------------------------------------------------- parsing

@pytest.mark.parametrize("spelling", ["y ~ x | w + f^g", "y ~ x | w + f:g"])
def test_interactions_parse_the_same_either_way(spelling):
    (spec,) = parse_formula(spelling)
    assert spec.fe.replace(" ", "") == "w+f^g"
    assert (spec.dep, spec.iv) == ("y", None)


@pytest.mark.parametrize("spelling", ["y + y2 ~ csw(a, b) | f", "sw(y, y2) ~ csw(a, b) | f"])
def test_several_outcomes_expand_to_one_model_each(spelling):
    specs = parse_formula(spelling)
    assert len(specs) == 4
    assert {s.dep for s in specs} == {"y", "y2"}
    assert len({s.key for s in specs}) == 4


def test_legacy_instrument_syntax_gives_the_endogenous_variable():
    (spec,) = parse_formula("y ~ x | f | e ~ z")
    endog, first_stage_rhs = spec.iv
    assert endog == "e"
    assert "z" in first_stage_rhs and "x" in first_stage_rhs


@needs_070
def test_instrument_block_parses_like_the_legacy_syntax():
    (new,) = parse_formula("y ~ x + [e ~ z] | f")
    (old,) = parse_formula("y ~ x | f | e ~ z")
    assert new == old


@pytest.mark.skipif(NEW_PARSER, reason="the older pyfixest cannot parse it")
def test_instrument_block_is_refused_with_the_older_pyfixest():
    with pytest.raises(ValueError, match="pyfixest 0.70"):
        parse_formula("y ~ x + [e ~ z] | f")


# ---------------------------------------------------------------------- fits

def fit(src, fml, tmp_path, **options):
    return feols_stream(fml, src, workdir=tmp_path, verbose=False, **options)


def test_caret_and_colon_interactions_give_the_same_fit(trends, tmp_path):
    a = fit(trends.src, "y ~ x | worker_id + firm_id^year", tmp_path / "a")
    b = fit(trends.src, "y ~ x | worker_id + firm_id:year", tmp_path / "b")
    np.testing.assert_allclose(a.beta, b.beta, rtol=1e-10)
    np.testing.assert_allclose(a.se, b.se, rtol=1e-10)
    a.cleanup(), b.cleanup()


def test_slopes_on_an_interacted_dimension(trends, tmp_path):
    a = fit(trends.src, "y ~ x | worker_id^firm_id[t]", tmp_path / "a")
    b = fit(trends.src, "y ~ x | worker_id:firm_id[t]", tmp_path / "b")
    np.testing.assert_allclose(a.beta, b.beta, rtol=1e-10)
    a.cleanup(), b.cleanup()


def test_slope_expression_matches_the_precomputed_column(trends, tmp_path):
    """`worker_id[t, (t - m) ** 2]` is `worker_id[t, t2]`, where `t2` is the
    centered square the simulated panel carries."""
    mid = float(np.mean(trends.frame["year"].unique().to_numpy()))
    a = fit(trends.src, "y ~ x | worker_id[t, t2] + firm_id", tmp_path / "a")
    b = fit(trends.src, f"y ~ x | worker_id[t, (t - {mid!r}) ** 2] + firm_id", tmp_path / "b")
    np.testing.assert_allclose(a.beta, b.beta, rtol=1e-6)
    assert b.coefnames == a.coefnames
    a.cleanup(), b.cleanup()


def test_slope_expression_is_not_a_column_of_the_result(trends, tmp_path):
    b = fit(trends.src, "y ~ x | worker_id[t ** 2] + firm_id", tmp_path)
    assert b.coefnames == ["x"]
    b.cleanup()


def test_slopes_without_the_intercept_are_not_available_yet(trends, tmp_path):
    with pytest.raises(NotImplementedError, match=r"\[\[\.\.\.\]\]"):
        fit(trends.src, "y ~ x | worker_id[[t]] + firm_id", tmp_path)


def test_several_outcomes_spellings_agree(trends, tmp_path):
    a = fit(trends.src, "y + xe ~ x | worker_id", tmp_path / "a")
    b = fit(trends.src, "sw(y, xe) ~ x | worker_id", tmp_path / "b")
    for ma, mb in zip(a, b):
        np.testing.assert_allclose(ma.beta, mb.beta, rtol=1e-10)
        ma.cleanup(), mb.cleanup()


@needs_070
def test_instrument_block_fits_like_the_legacy_syntax(trends, tmp_path):
    a = fit(trends.src, "y ~ x | worker_id[t] + firm_id | xe ~ z", tmp_path / "a")
    b = fit(trends.src, "y ~ x + [xe ~ z] | worker_id[t] + firm_id", tmp_path / "b")
    assert a.coefnames == b.coefnames == ["x", "xe"]
    np.testing.assert_allclose(a.beta, b.beta, rtol=1e-10)
    np.testing.assert_allclose(a.se, b.se, rtol=1e-10)
    a.cleanup(), b.cleanup()


def test_expressions_in_slopes_need_columns_that_exist(trends, tmp_path):
    with pytest.raises(Exception, match="nope"):
        fit(trends.src, "y ~ x | worker_id[nope ** 2] + firm_id", tmp_path)

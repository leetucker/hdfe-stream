"""Formula front end: pyfixest syntax -> Polars expressions.

Depends on pyfixest's *internal* formula modules (see `_pyfixest_formula_api`),
which is why the package pins a pyfixest range. Required for `feols_stream`;
`StreamingHDFE` works without it.
"""

from __future__ import annotations

import ast
import re

import numpy as np
import polars as pl

from .feterms import _parse_fe


# --------------------------------------------------------------------------
# formula front end: pyfixest syntax -> Polars expressions
#
# pyfixest's parser splits formulas and expands multiple-estimation syntax.
# To get exactly pyfixest's coefficient names and categorical coding, the
# covariate side is run through pyfixest/formulaic on a tiny *synthetic*
# frame that contains every level of every categorical variable. Each
# resulting column name is then compiled back to a Polars expression:
# components are split on ':' and are either an indicator for one level
# ("C(g)[T.2]", "cat[a]", "year::2005") or a numeric expression ("x",
# "I(age ** 2)", "log(age)") translated through a small AST compiler.
# --------------------------------------------------------------------------

def _pyfixest_formula_api():
    try:
        from pyfixest.estimation.formula.model_matrix import create_model_matrix
        from pyfixest.estimation.formula.parse import Formula
    except ImportError as err:  # internal API; pin the pyfixest version
        raise ImportError(
            "formula support uses pyfixest's formula parser "
            "(pyfixest.estimation.formula), which this pyfixest version does not "
            "provide; pin a compatible pyfixest or use StreamingHDFE directly") from err
    return Formula, create_model_matrix


_FUNCS = {
    "log": lambda e: e.log(), "log1p": lambda e: e.log1p(), "log10": lambda e: e.log10(),
    "exp": lambda e: e.exp(), "sqrt": lambda e: e.sqrt(), "abs": lambda e: e.abs(),
}
_BINOPS = {
    ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b, ast.Div: lambda a, b: a / b,
    ast.Pow: lambda a, b: a.pow(b), ast.FloorDiv: lambda a, b: a // b,
    ast.Mod: lambda a, b: a % b,
}
_CMPOPS = {
    ast.Eq: lambda a, b: a == b, ast.NotEq: lambda a, b: a != b,
    ast.Lt: lambda a, b: a < b, ast.LtE: lambda a, b: a <= b,
    ast.Gt: lambda a, b: a > b, ast.GtE: lambda a, b: a >= b,
}


def _ast_to_expr(node, numeric=True):
    """Compile a small, safe subset of Python expressions to Polars."""
    if isinstance(node, ast.Expression):
        return _ast_to_expr(node.body, numeric)
    if isinstance(node, ast.Name):
        return pl.col(node.id).cast(pl.Float64) if numeric else pl.col(node.id)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (bool, int, float)) and numeric:
            return pl.lit(float(node.value))
        return pl.lit(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        return _BINOPS[type(node.op)](_ast_to_expr(node.left), _ast_to_expr(node.right))
    if isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.USub):
            return -_ast_to_expr(node.operand)
        if isinstance(node.op, ast.UAdd):
            return _ast_to_expr(node.operand)
    if isinstance(node, ast.Compare) and len(node.ops) == 1 and type(node.ops[0]) in _CMPOPS:
        out = _CMPOPS[type(node.ops[0])](_ast_to_expr(node.left, False),
                                         _ast_to_expr(node.comparators[0], False))
        return out.cast(pl.Float64)
    if isinstance(node, ast.Call) and len(node.args) == 1 and not node.keywords:
        f = node.func
        fname = f.id if isinstance(f, ast.Name) else (f.attr if isinstance(f, ast.Attribute) else None)
        if fname == "I":
            return _ast_to_expr(node.args[0])
        if fname in _FUNCS:
            return _FUNCS[fname](_ast_to_expr(node.args[0]))
    raise ValueError(f"unsupported expression in formula: {ast.unparse(node)!r}. Supported: "
                     "columns, + - * / ** // %, comparisons, I(), "
                     f"{', '.join(sorted(_FUNCS))}, C(), i(). Construct anything else "
                     "as a column in the input LazyFrame.")


def _numeric_expr(src):
    return _ast_to_expr(ast.parse(src, mode="eval")).cast(pl.Float64)


def _split_components(name):
    """Split a model-matrix column name on top-level ':' (keeping '::')."""
    s = name.replace("::", "\x00")
    parts, depth, cur = [], 0, ""
    for ch in s:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        if ch == ":" and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    parts.append(cur)
    return [p.replace("\x00", "::") for p in parts]


def _categorical_vars(rhs, schema):
    """Variables that need level discovery: first argument of C()/i(), a
    non-numeric second argument of i(), and non-numeric columns."""
    import formulaic
    cats = set()
    for term in formulaic.Formula(rhs):
        for fac in term.factors:
            src = fac.expr
            if src in schema:
                if not (schema[src].is_numeric() or schema[src] == pl.Boolean):
                    cats.add(src)
                continue
            try:
                tree = ast.parse(src, mode="eval")
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                        and node.func.id in ("C", "i"):
                    kw = {k.arg for k in node.keywords}
                    if node.func.id == "C" and (len(node.args) != 1 or kw):
                        raise ValueError(f"only plain C(var) is supported, got {src!r}")
                    if node.func.id == "i" and kw & {"bin", "bin2"}:
                        raise ValueError("i(..., bin=) is not supported; bin the variable "
                                         "in the input LazyFrame instead")
                    for j, arg in enumerate(node.args[:2]):
                        if not isinstance(arg, ast.Name):
                            raise ValueError(f"arguments of {node.func.id}() must be column "
                                             f"names, got {src!r}")
                        if j == 0 or not (schema[arg.id].is_numeric()
                                          or schema[arg.id] == pl.Boolean):
                            cats.add(arg.id)
    return cats


def _compile_design(depvar, rhs, schema, levels):
    """Return (depvar expr, [(coefname, expr), ...]) for one model."""
    import formulaic
    Formula, create_model_matrix = _pyfixest_formula_api()
    needed = {v for v in formulaic.Formula(rhs).required_variables if v in schema}
    missing = {v for v in formulaic.Formula(rhs).required_variables
               if v not in schema and v not in ("i", "C", "I", "np", *_FUNCS)}
    if missing:
        raise ValueError(f"unknown variables in formula: {sorted(missing)}")

    # synthetic frame: every categorical level appears at least once
    import pandas as pd
    R = max([3] + [len(levels[v]) for v in needed if v in levels])
    rng = np.random.default_rng(0)
    synth = {"__y__": rng.uniform(1, 2, R)}
    for v in needed:
        if v in levels:
            synth[v] = np.resize(np.array(levels[v], dtype=object if isinstance(
                levels[v][0], str) else None), R)
        else:
            synth[v] = rng.uniform(1, 2, R)
    mm = create_model_matrix(Formula.parse(f"__y__ ~ {rhs}")[0], pd.DataFrame(synth),
                             drop_intercept=True)
    names = [c for c in mm.independent.columns if c != "Intercept"]

    cat_labels = {}
    for v in levels:
        cat_labels[f"C({v})"] = v
        if v in schema and not schema[v].is_numeric():
            cat_labels[v] = v
    lvl_lookup = {v: {str(x): x for x in lv} for v, lv in levels.items()}

    def component(c):
        m = re.fullmatch(r"(.+)\[(T\.)?(.*)\]", c)
        if m and m.group(1) in cat_labels:
            v = cat_labels[m.group(1)]
            return (pl.col(v) == pl.lit(lvl_lookup[v][m.group(3)])).cast(pl.Float64)
        if "::" in c:
            v, lv = c.split("::", 1)
            if v in lvl_lookup and lv in lvl_lookup[v]:
                return (pl.col(v) == pl.lit(lvl_lookup[v][lv])).cast(pl.Float64)
        return _numeric_expr(c)

    cols = []
    for nm in names:
        expr = None
        for c in _split_components(nm):
            e = component(c)
            expr = e if expr is None else expr * e
        cols.append((nm, expr))
    return (depvar, _numeric_expr(depvar)), cols


def _plan_formula(fml, lf):
    """Expand a pyfixest formula into groups of models that share a set of
    fixed effects (and therefore one pass 0/1 and one solve)."""
    Formula, _ = _pyfixest_formula_api()
    schema = lf.collect_schema()
    specs = Formula.parse(fml)
    for s in specs:
        if len(_parse_fe(s.fixed_effects)) < 2:
            raise ValueError(f"'{s.formula}': at least two fixed-effect dimensions are "
                             "required (the first is streamed)")

    # level discovery for all categorical variables (one streaming pass each)
    cats = set()
    for s in specs:
        cats |= _categorical_vars(s.second_stage.split("~", 1)[1], schema)
        if s.first_stage is not None:
            cats |= _categorical_vars(s.first_stage.split("~", 1)[1], schema)
    levels = {}
    for v in sorted(cats):
        u = lf.select(pl.col(v).drop_nulls().unique().sort()).collect(engine="streaming")
        levels[v] = u[v].to_list()

    groups = {}
    for s in specs:
        dep, rhs = (t.strip() for t in s.second_stage.split("~", 1))
        (dname, dexpr), cols = _compile_design(dep, rhs, schema, levels)
        fe = tuple(_parse_fe(s.fixed_effects))
        g = groups.setdefault(fe, {"y": {}, "x": {}, "models": []})
        g["y"][dname] = dexpr
        model = {"fml": s.formula, "y": dname, "x": [nm for nm, _ in cols]}
        extra = []
        if s.first_stage is not None:
            # first stage "endog ~ instruments + exogenous": its columns are the
            # instrument set Z; the endogenous columns come from compiling the
            # left-hand side as a right-hand side
            endog, fs_rhs = (t.strip() for t in s.first_stage.split("~", 1))
            _, fs_cols = _compile_design(dep, fs_rhs, schema, levels)
            _, en_cols = _compile_design(dep, endog, schema, levels)
            model["iv"] = {"endog": [nm for nm, _ in en_cols], "z": [nm for nm, _ in fs_cols]}
            missing = [e for e in model["iv"]["endog"] if e not in model["x"]]
            if missing:
                raise ValueError(f"endogenous columns {missing} not found among regressors")
            extra = fs_cols + en_cols
        for nm, ex in cols + extra:
            g["x"].setdefault(nm, ex)
        g["models"].append(model)
    return groups

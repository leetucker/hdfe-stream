"""`hdfe_stream.core` is the machinery other estimators build on, so none of
its modules may import the rest of the package (the leave-out estimators,
the formula API) when it is imported. Imports inside functions, and under
`if TYPE_CHECKING:`, may reach up: they run only when called."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import hdfe_stream.core

CORE = Path(hdfe_stream.core.__file__).parent
MODULES = sorted(CORE.glob("*.py"))


def _import_time_imports(tree):
    """The import statements a module runs when it is imported: those in its
    body, its classes and its if/try/with blocks, but not in functions or
    under `if TYPE_CHECKING:`."""
    todo = list(tree.body)
    while todo:
        node = todo.pop()
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            yield node
        elif isinstance(node, ast.If):
            if not (isinstance(node.test, ast.Name) and node.test.id == "TYPE_CHECKING"):
                todo += node.body
            todo += node.orelse
        elif isinstance(node, ast.Try):
            todo += node.body + node.orelse + node.finalbody
            todo += [n for h in node.handlers for n in h.body]
        elif isinstance(node, (ast.ClassDef, ast.With)):
            todo += node.body


def test_the_layer_has_its_modules():
    names = {p.stem for p in MODULES}
    assert {"estimator", "passes", "solve", "inference", "irls", "results_base",
            "results", "kernels_base", "kernels_groups"} <= names


@pytest.mark.parametrize("path", MODULES, ids=lambda p: p.name)
def test_core_imports_only_core_at_import_time(path):
    tree = ast.parse(path.read_text(), filename=str(path))
    outside = []
    for node in _import_time_imports(tree):
        if isinstance(node, ast.ImportFrom) and node.level:
            # from .x import ... must name a core module; from .. reaches up
            ok = node.level == 1 and (node.module is None
                                      or (CORE / f"{node.module.split('.')[0]}.py").exists())
        else:
            names = [node.module] if isinstance(node, ast.ImportFrom) else [
                a.name for a in node.names]
            ok = all(n != "hdfe_stream" and not n.startswith("hdfe_stream.")
                     or n == "hdfe_stream.core" or n.startswith("hdfe_stream.core.")
                     for n in names)
        if not ok:
            outside.append(f"line {node.lineno}: {ast.unparse(node)}")
    assert not outside, f"{path.name} imports from outside core: {outside}"

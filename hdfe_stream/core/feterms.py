"""Fixed-effect term syntax: 'worker_id[t]', 'firm_id^year'.

Deliberately free of any pyfixest dependency: `StreamingHDFE` parses FE terms
in its constructor and must work without the formula front end installed.
"""

from __future__ import annotations

import re


_FE_TERM = re.compile(r"^\s*([^\[\]]+?)\s*(\[\[?)(.*?)(\]\]?)\s*$")


def _parse_fe_term(term):
    """'worker_id[t, t2]' -> ('worker_id', ['t', 't2']);
    'firm_id' -> ('firm_id', [])."""
    m = _FE_TERM.match(term)
    if not m:
        return "^".join(c.strip() for c in term.split("^")), []
    name, opening, inner, closing = m.groups()
    if opening == "[[" or closing == "]]":
        raise NotImplementedError(f"{term!r}: slopes without the FE intercept ([[...]]) are not "
                                  "supported; use name[x] (intercept + slopes)")
    slopes = [v.strip() for v in inner.split(",") if v.strip()]
    if not slopes:
        raise ValueError(f"{term!r}: no slope variables inside [...]")
    return "^".join(c.strip() for c in name.split("^")), slopes


def parse_fe(fe_str):
    """The terms of a formula's fixed-effect part, ``"a + b^c + d[t]"`` ->
    ``["a", "b^c", "d[t]"]`` (none for an empty part)."""
    return [t.strip() for t in fe_str.split("+") if t.strip()] if fe_str else []


def _split_top(text, seps):
    """Split `text` on the characters in `seps` that sit outside any (), []
    pair; return the pieces and the separators found between them."""
    pieces, found, depth, cur = [], [], 0, []
    for ch in text:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        if depth == 0 and ch in seps:
            pieces.append("".join(cur))
            found.append(ch)
            cur = []
        else:
            cur.append(ch)
    pieces.append("".join(cur))
    return pieces, found


def rewrite_fe_interactions(fml, to):
    """`fml` with the interaction operator of its fixed-effect part (the text
    after the first top-level ``|``, up to a second one) written as `to`:
    ``"^"`` (fixest) or ``":"`` (formulaic). Slope brackets and the covariates
    are left as they are, so ``"y ~ a:b | f^g[t]"`` is not touched in ``a:b``.
    """
    parts, _ = _split_top(fml, "|")
    if len(parts) < 2:
        return fml
    pieces, seps = _split_top(parts[1], "^:")
    out = pieces[0]
    for sep, piece in zip(seps, pieces[1:]):
        out += to + piece
    parts[1] = out
    return "|".join(parts)


def interactions_to_caret(fe_str):
    """A fixed-effect part with ``:`` interactions written as ``^``, the form
    used for fixed-effect names throughout the package."""
    pieces, _ = _split_top(fe_str, ":")
    return "^".join(p.strip() for p in pieces)

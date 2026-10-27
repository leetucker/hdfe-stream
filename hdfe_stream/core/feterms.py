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


def _parse_fe(fe_str):
    return [t.strip() for t in fe_str.split("+") if t.strip()] if fe_str else []

"""Names of the columns the estimator adds to the rows it works on.

Pass 0 writes these next to the user's own columns (fixed effects, cluster
variables, `keep=`), and every later pass reads them. They all start with
`PREFIX`, and a fit refuses a source that already has a column starting with
it (`check_source`), so no name chosen here can collide with a user's.
"""

from __future__ import annotations

from typing import Iterable

PREFIX = "__hdfe_"

WCOL = f"{PREFIX}w"                  # weights (only with weights=)
GCODE = f"{PREFIX}gcode"            # level code of the streamed dimension
BUCKET = f"{PREFIX}bucket"          # partition key; the bucket directories' name


def vcol(j: int) -> str:
    """Design column j: the outcome and covariates, in `var_names` order."""
    return f"{PREFIX}v{j}"


def tcol(j: int) -> str:
    """Slope variable j (counting from 1)."""
    return f"{PREFIX}t{j}"


def ccol(d: int) -> str:
    """Level code of the dth dimension other than the streamed one (from 1)."""
    return f"{PREFIX}c{d}"


def kcol(i: int) -> str:
    """Code of the ith extra cluster variable."""
    return f"{PREFIX}k{i}"


def check_source(columns: Iterable[str]) -> None:
    """Raise if the source has a column the estimator might overwrite."""
    clash = [c for c in columns if c.startswith(PREFIX)]
    if clash:
        raise ValueError(
            f"the data has columns whose names start with {PREFIX!r}, which the "
            f"estimator uses for its own columns: {clash}. Rename them.")

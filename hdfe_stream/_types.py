"""Type aliases shared by the public signatures."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence, Union

import polars as pl

PathLike = Union[str, Path]
# a Parquet path or glob, or a Polars LazyFrame (or DataFrame, already in memory)
Source = Union[PathLike, pl.LazyFrame, pl.DataFrame]
# 'iid', 'hetero', 'HC1', 'CRV1:var', or {'CRV1': 'var'}
Vcov = Union[str, Mapping[str, str]]
# column name, (name, expression), or a mapping / sequence of them
Variables = Union[
    str, tuple, Sequence[Union[str, tuple]], Mapping[str, pl.Expr]]
Options = Any

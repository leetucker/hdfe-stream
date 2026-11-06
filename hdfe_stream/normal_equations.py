"""Solving a fit's normal equations for right-hand sides built from rows.

`HDFEResult.solve_normal_equations` is the public entry point; the solves
themselves are the inverse machinery's (`inverse.py`).
"""

from __future__ import annotations

import shutil
from itertools import count
from pathlib import Path

import numpy as np
import polars as pl

from .core._columns import PREFIX, check_source
from .core.utils import as_lazy

_SEQUENCE = count()


class NormalEquationsSolution:
    """The solutions u_j = S^- Z'v_j of `HDFEResult.solve_normal_equations`.

    Each solution is a set of coefficients for the model's design: an effect
    for every level of every fixed effect, and one per covariate. Like the
    fitted fixed effects, the level effects are identified only up to the
    model's normalization; `predict` gives what is identified, z'u_j for a row
    with design z.

    The tables are Parquet files in the fit's result directory, deleted with
    the fit's other result files or by `cleanup`.

    Attributes
    ----------
    names : list of str
        The names of the right-hand sides, in the order they were given.
    coef : polars.DataFrame
        The covariate block: a ``coefficient`` column (the fit's coefficient
        names) and one column per right-hand side.
    """

    def __init__(self, result, names, directory, blocks, coef):
        self._result = result
        self.names = list(names)
        self._directory = Path(directory)
        self._blocks = blocks
        self.coef = coef

    def fixef(self, name: str) -> pl.LazyFrame:
        """One fixed effect's block of the solutions: its level columns and
        one column per right-hand side, in the order of `HDFEResult.fixef`."""
        if name not in self._result.fe_names:
            raise KeyError(f"{name!r} is not a fixed effect of this model; it has "
                           + ", ".join(map(repr, self._result.fe_names)))
        files = [block[name] for block in self._blocks]
        if not all(Path(f).exists() for f in files):
            raise FileNotFoundError("the solution files are gone: cleanup() was called, "
                                    "or the fit's result files were removed")
        keys = self._result._estimator.fe_cols[name]
        parts = [pl.scan_parquet(files[0])]
        parts += [pl.scan_parquet(f).drop(keys) for f in files[1:]]
        return pl.concat(parts, how="horizontal")

    def predict(self, newdata) -> pl.LazyFrame:
        """`newdata`, lazily, with one column per right-hand side added: z'u_j
        for each row's design z (its fixed effects and covariates).

        A row gets nulls if one of its levels is not in the estimation sample.
        For an estimation row i of a fit with weights w, w_i z_i'u_j is the
        derivative of the weighted sum of predictions that right-hand side j
        stands for with respect to that row's outcome.
        """
        est = self._result._estimator
        frame = as_lazy(newdata)
        schema = frame.collect_schema()
        check_source(schema.names())
        clash = [n for n in self.names if n in schema]
        if clash:
            raise ValueError(f"newdata already has columns named {clash}")
        tmp = []
        terms = {n: [] for n in self.names}
        for i, d in enumerate(self._result.fe_names):
            keys = est.fe_cols[d]
            table = self.fixef(d)
            tschema = table.collect_schema()
            tkeys = [f"{PREFIX}nkey{i}_{j}" for j in range(len(keys))]
            values = {n: f"{PREFIX}nfe{i}_{k}" for k, n in enumerate(self.names)}
            table = table.select(*[pl.col(k).alias(t) for k, t in zip(keys, tkeys)],
                                 *[pl.col(n).alias(v) for n, v in values.items()])
            frame = frame.with_columns(pl.col(k).cast(tschema[k], strict=False).alias(t)
                                       for k, t in zip(keys, tkeys))
            frame = frame.join(table, on=tkeys, how="left", maintain_order="left")
            for n, v in values.items():
                terms[n].append(pl.col(v))
            tmp += [*tkeys, *values.values()]
        coefficients = self.coef["coefficient"].to_list()
        for n in self.names:
            beta = self.coef[n].to_numpy()
            terms[n] += [float(b) * est.var_exprs[c] for c, b in zip(coefficients, beta)]
        out = []
        for n, ts in terms.items():
            value = pl.sum_horizontal(ts) if ts else pl.lit(0.0)
            if ts:
                value = pl.when(pl.any_horizontal(t.is_null() for t in ts)).then(None) \
                          .otherwise(value)
            out.append(value.cast(pl.Float64).alias(n))
        return frame.with_columns(out).drop(tmp)

    def cleanup(self) -> None:
        """Delete the solution files."""
        shutil.rmtree(self._directory, ignore_errors=True)


def solve_normal_equations(result, data, columns, block=16):
    est = result._estimator
    if est is None:
        raise ValueError("this result has no estimator attached (it was converted)")
    if result.family:
        raise ValueError("the normal equations are solved for linear models; this is a "
                         f"{result.family} model")
    if getattr(result, "first_stage", None):
        raise ValueError("solving the normal equations is not available for IV models")
    if est.slopes:
        raise ValueError("solving the normal equations is not available with varying slopes")
    if not result.fe_names:
        raise ValueError("the model has no fixed effects; solve its Gram matrix directly")
    if not getattr(est, "keep_intermediates", False):
        raise RuntimeError("solving the normal equations needs the fit's intermediates; "
                           "refit with keep_intermediates=True")
    if est.solver == "within":
        raise ValueError("solving the normal equations needs solver='auto', 'explicit' "
                         "or 'stream_cg'")
    if isinstance(columns, str):
        columns = [columns]
    columns = list(columns)
    if not columns or len(set(columns)) != len(columns):
        raise ValueError("columns must name one or more distinct columns")
    if not isinstance(block, int) or isinstance(block, bool) or block < 1:
        raise ValueError("block must be a positive integer")

    frame = as_lazy(data)
    schema = frame.collect_schema()
    check_source(schema.names())
    missing = [c for c in columns if c not in schema]
    if missing:
        raise ValueError(f"data lacks the columns {missing}")

    # Map each row to its level codes: a table's row number is its code.
    codes, code_columns = frame, []
    for i, d in enumerate(result.fe_names):
        keys = est.fe_cols[d]
        table = result.fixef(d).select(keys)
        tschema = table.collect_schema()
        tkeys = [f"{PREFIX}skey{i}_{j}" for j in range(len(keys))]
        code = f"{PREFIX}scode{i}"
        table = row_numbers(table.select(*[pl.col(k).alias(t) for k, t in zip(keys, tkeys)]),
                            code)
        codes = codes.with_columns(pl.col(k).cast(tschema[k], strict=False).alias(t)
                                   for k, t in zip(keys, tkeys))
        codes = codes.join(table, on=tkeys, how="left").drop(tkeys)
        code_columns.append(code)
    coefficients = list(result.coefnames)
    unknown = codes.select(pl.any_horizontal(pl.col(c).is_null() for c in code_columns)
                           .sum().alias("n")).collect(engine="streaming").item()
    if unknown:
        raise ValueError(f"{unknown:,} rows of data have a fixed-effect level that is not "
                         "in the estimation sample")
    def right_hand_sides(names):
        """Z'v for these columns: level, covariate and streamed blocks.

        One pass over `data` per block of columns, so that memory holds the
        streamed block (a row per group) for the block only.
        """
        vectors = [pl.col(c).cast(pl.Float64) for c in names]
        queries = [codes.group_by(code).agg(v.sum().alias(n) for v, n in zip(vectors, names))
                   for code in code_columns]
        queries.append(codes.select([(v * est.var_exprs[c]).sum().alias(f"{PREFIX}x{i}_{k}")
                                     for i, c in enumerate(coefficients)
                                     for k, v in enumerate(vectors)])
                       if coefficients else codes.select(pl.len()))
        tables = pl.collect_all(queries, engine="streaming")
        levels = np.zeros((total_levels, len(names)))
        streamed = np.zeros((est.n_levels[est.g_fe], len(names)))
        for d, code, table in zip(result.fe_names, code_columns, tables):
            index = table[code].to_numpy().astype(np.int64)
            values = table.select(names).to_numpy()
            if d == est.g_fe:
                streamed[index] = values
            else:
                levels[off[d] + index] = values
        covariate_values = (np.array(tables[-1].row(0), dtype=np.float64)
                            .reshape(len(coefficients), len(names))
                            if coefficients else np.zeros((0, len(names))))
        return levels, covariate_values, streamed

    m = len(columns)
    off, total_levels = est._offsets()
    directory = Path(result.paths["dir"]) / f"normal_equations_{next(_SEQUENCE)}"
    directory.mkdir(parents=True, exist_ok=True)
    blocks, coef = [], np.zeros((len(coefficients), m))
    fe_tables = {d: result.fixef(d).select(est.fe_cols[d]).collect()
                 for d in result.fe_names}
    est.reload_intermediates()
    try:
        for start in range(0, m, block):
            names = columns[start:start + block]
            levels, covariate_values, streamed = right_hand_sides(names)
            U, C, U_streamed = est.apply_inverse(levels, covariate_values, streamed,
                                                 covariates=coefficients)
            del levels, streamed
            coef[:, start:start + len(names)] = C
            files = {}
            for d in result.fe_names:
                values = U_streamed if d == est.g_fe else U[off[d]:off[d] + est.n_levels[d]]
                path = directory / f"{d}_{start}.parquet"
                fe_tables[d].with_columns(
                    pl.Series(n, values[:, k]) for k, n in enumerate(names)
                ).write_parquet(path)
                files[d] = path
            blocks.append(files)
    except BaseException:
        shutil.rmtree(directory, ignore_errors=True)
        raise
    finally:
        est.unload_intermediates()
    coef_frame = pl.DataFrame({"coefficient": coefficients,
                               **{n: coef[:, k] for k, n in enumerate(columns)}},
                              schema={"coefficient": pl.String,
                                      **{n: pl.Float64 for n in columns}})
    return NormalEquationsSolution(result, columns, directory, blocks, coef_frame)


def row_numbers(frame: pl.LazyFrame, name: str) -> pl.LazyFrame:
    return frame.with_row_index(name).with_columns(pl.col(name).cast(pl.Int64))

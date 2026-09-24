"""
hdfe_stream: out-of-core OLS with several high-dimensional fixed effects.

    from hdfe_stream import feols_stream
    fit = feols_stream("log_earn ~ age_squared + age_cubed | worker_id + firm_id + year",
                       "data/*.parquet", workdir="hdfe_work", vcov={"CRV1": "worker_id"})
    fit.summary()

Formulas follow pyfixest syntax and are parsed with pyfixest's own parser:
covariates, interactions (`:` and `*`), `I()`, `log()`, `C()`, `i()`,
several dependent variables (`y1 + y2 ~ ...`), `sw()` / `csw()` stepwise
syntax, fixed-effect interactions (`firm_id^year`), and IV
(`y ~ x | fe | endog ~ z`). Every covariate term is compiled to a Polars
expression, so constructing the design never leaves the streaming engine.
Weighted least squares via `weights=` (aweights or fweights). The
lower-level `StreamingHDFE` class takes column names or Polars expressions
directly.

Reporting: `result.to_pyfixest()` returns a pyfixest Feols/Feiv view for
pf.etable, pf.summary, pf.coefplot and pf.iplot; `hdfe_stream.etable(...)`
does the conversion for you and accepts ordinary pyfixest models too.

Model:  y = X b + sum_d  gamma_d[level_d(i)] + e     for d = 0 ... D-1

One fixed-effect dimension (`fe[0]` internally) is *streamed*: it is never
represented as a vector in memory. It should be the one with the most levels
whose groups each touch only a few levels of the other dimensions (workers,
not firms). It is chosen as: the `stream=` option if given; else the
dimension with varying slopes; else the highest approximate cardinality
(HyperLogLog counts gathered during pass 0's first scan), ties going to the
dimension listed first. Every other dimension is represented by level-sized
vectors in RAM.

Varying slopes (FEIS) on the streamed dimension use fixest syntax:
`| worker_id[t] + firm_id` gives worker effects plus worker-specific slopes on
t (several: `worker_id[t, t2]`). Projecting out the streamed dimension
then means removing each worker's own weighted regression on [1, slopes],
which is still local to the worker.

Memory: rows are only ever streamed. The arrays held in RAM are sized by the
non-streamed dimensions' level counts (times a block of at most `rhs_block`
variables at a time), plus, for the explicit solver, the sparse reduced
matrix S, whose size depends on how levels co-occur and not on rows. Polars
steps stream; the sort and cell group_by run one fe[0] hash bucket at a time.

Pipeline
--------
Pass 0 (Polars, streaming)   evaluate the design as Polars expressions; drop
                             rows with missing / non-finite values; factorize
                             the non-streamed FE and cluster ids;
                             hash-partition rows into fe[0] buckets; sort each
                             bucket and assign dense fe[0] codes.
Pass 1 (Polars, streaming)   per bucket: group_by(fe[0], other codes) -> cell
                             table with n and sums (plus within-cell
                             cross-products when assembly="cells").
Pass 1b (streamed + numba)   keep cells of fe[0] groups with more than one
                             cell as memory-mapped arrays; Jacobi diagonal.
Step 2 (solver, per block)   solve  S Gamma = D_o' M_0 V,  S = D_o' M_0 D_o,
                             for all variables V in blocks of `rhs_block`
                             columns: "explicit" (sparse S built once, block
                             PCG in memory), "stream_cg" (S applied by
                             streaming the cells), or "within".
Step 3 (numba)               assemble V' M_D V (from cells, or from a row
                             pass when there are many variables); drop
                             collinear covariates per model; beta.
Step 4 (streamed + numba)    per model: stream rows for fe[0] effects,
                             residuals, RSS and the meat for HC1 / CRV1.

Layout
------
api.py             `feols_stream`
estimator.py       `StreamingHDFE`: options, layout, the driver
passes.py          pass 0/1/1b   (mixin)
solve.py           step 2        (mixin)
inference.py       steps 3/4     (mixin)
kernels_base.py    numba kernels
kernels_slopes.py  numba kernels, varying slopes
formula.py         pyfixest formula -> Polars expressions (input side)
reporting.py       pyfixest etable/coefplot views (output side)
feterms.py         'worker_id[t]' / 'firm_id^year' parsing
results.py         `HDFEResult`, `HDFEMulti`
workspace.py       run directories, `cleanup()`
report.py          logging / printing
utils.py           small shared helpers
"""

from .api import feols_stream
from .estimator import StreamingHDFE
from .reporting import etable
from .results import HDFEMulti, HDFEResult
from .utils import iter_group_chunks
from .workspace import cleanup

__all__ = [
    "feols_stream",
    "StreamingHDFE",
    "HDFEResult",
    "HDFEMulti",
    "etable",
    "cleanup",
    "iter_group_chunks",
]

def _detect_version():
    try:
        from importlib.metadata import version
        return version("hdfe-stream")
    except Exception:                 # source tree without an installed dist
        return "0.0.1"


__version__ = _detect_version()

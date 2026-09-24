"""Shared fixtures and comparison helpers for the hdfe_stream test suite.

Every test runs on simulated data (`hdfe_stream.simulate`). The three panels
are generated once per session and written to Parquet in a temporary
directory, because the point of the library is reading rows off disk.

Most tests assert agreement with pyfixest, which is the reference
implementation for everything except varying slopes (pyfixest has no FEIS, so
those tests compare against a brute-force regression with explicit
worker-by-slope dummies instead).

Panels are kept small enough that the pyfixest reference fits are quick. That
is a real constraint: the reference demeaner is run to a very tight tolerance
so that the comparison measures hdfe_stream's error and not pyfixest's.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from hdfe_stream.simulate import simulate_akm, simulate_rich, simulate_trends

# Panel sizes. 2,500 workers is the smallest that reliably leaves the
# worker-firm graph connected (one component), which keeps the fixed effects
# identified without a large panel.
N_WORKERS = 2_500
N_TREND_WORKERS = 300

# Agreement tolerances, as relative differences. These are ~100x the
# differences actually observed, so they catch a broken estimator without
# failing on the last bits of floating point.
TOL_BETA = 1e-8
TOL_SE = 1e-7
TOL_VCOV = 1e-6
TOL_RESID = 1e-6
TOL_STAT = 1e-9          # fit statistics (R2, RMSE, ...)

# The FE degrees-of-freedom convention has to match for standard errors to
# agree with pyfixest; "exact" is hdfe_stream's own, stricter default.
FE_DOF_PF = "pyfixest"


@dataclass
class Panel:
    """A simulated panel on disk, with cached in-memory views."""

    path: Path
    frame: pl.DataFrame

    @property
    def pandas(self):
        """pandas view, for handing to pyfixest."""
        if not hasattr(self, "_pandas"):
            self._pandas = self.frame.to_pandas()
        return self._pandas

    @property
    def src(self) -> str:
        """What to pass to hdfe_stream as the data source."""
        return str(self.path)


def _panel(tmp_dir, name, frame) -> Panel:
    path = Path(tmp_dir) / f"{name}.parquet"
    frame.write_parquet(path)
    return Panel(path=path, frame=frame)


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("panels")


@pytest.fixture(scope="session")
def akm(data_dir):
    """Base AKM panel: worker_id, firm_id, year, age polynomials, log_earn."""
    return _panel(data_dir, "akm", simulate_akm(n_workers=N_WORKERS))


@pytest.fixture(scope="session")
def rich(data_dir):
    """Base panel plus categoricals, cluster variables, weights, instruments,
    a second outcome and a column with missing values."""
    return _panel(data_dir, "rich", simulate_rich(n_workers=N_WORKERS))


@pytest.fixture(scope="session")
def trends(data_dir):
    """Small panel with worker-specific time trends, for varying slopes."""
    return _panel(data_dir, "trends", simulate_trends(n_workers=N_TREND_WORKERS))


@pytest.fixture
def workdir(tmp_path):
    """A scratch directory for one fit's run directory."""
    return tmp_path / "work"


# --------------------------------------------------------------------------
# comparison helpers
# --------------------------------------------------------------------------

def rel(a, b, floor=1e-8):
    """Largest relative difference between two arrays, guarding tiny `b`."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape:
        raise AssertionError(f"shape mismatch: {a.shape} vs {b.shape}")
    if a.size == 0:
        return 0.0
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), floor)))


def scaled(a, b):
    """Largest absolute difference, divided by the overall magnitude of `b`.

    The right measure for quantities that legitimately pass through zero --
    residuals and fitted fixed effects -- where an elementwise relative
    difference blows up on the values nearest zero even when every value
    agrees to machine precision.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape:
        raise AssertionError(f"shape mismatch: {a.shape} vs {b.shape}")
    if a.size == 0:
        return 0.0
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-12))


def interaction_name(dim):
    """Column name holding an 'a^b' dimension as a single variable."""
    return dim.replace("^", "_x_")


def ensure_interactions(panel, dims):
    """Materialize each 'a^b' dimension as one column in the pandas view.

    hdfe_stream accepts "firm_id^year" directly as a fixed effect or a cluster
    variable, but pyfixest's `^` works only in the fixed-effect part of a
    formula -- clustering needs a real column. Building it here lets the two be
    compared on the interacted dimension instead of skipping it.

    Call this before the reference fit: pyfixest clusters using the frame it
    was given, so the column has to exist by then.
    """
    pdf = panel.pandas
    for dim in dims:
        if "^" not in dim:
            continue
        name = interaction_name(dim)
        if name not in pdf.columns:
            parts = dim.split("^")
            joined = pdf[parts[0]].astype(str)
            for part in parts[1:]:
                joined = joined + "^" + pdf[part].astype(str)
            pdf[name] = joined
    return pdf


def cluster_spec(dim):
    """hdfe_stream's spelling of a one-way cluster dimension, as pyfixest
    needs it (see `ensure_interactions`)."""
    return interaction_name(dim)


def pf_feols(fml, panel, tol=1e-12, **kwargs):
    """pyfixest reference fit, demeaned to `tol` so the comparison measures
    hdfe_stream's error rather than pyfixest's convergence.

    `fixef_rm="none"` keeps pyfixest from dropping singleton observations,
    which hdfe_stream does not do either.
    """
    import pyfixest as pf

    kwargs.setdefault("fixef_rm", "none")
    try:
        return pf.feols(fml, data=panel.pandas,
                        demeaner=pf.LsmrDemeaner(fixef_atol=tol, fixef_btol=tol),
                        **kwargs)
    except ValueError:
        # the LSMR preconditioner fails on some IV first stages; fall back to
        # the alternating-projections demeaner at a comparable tolerance
        return pf.feols(fml, data=panel.pandas, fixef_tol=tol,
                        fixef_maxiter=500_000, **kwargs)


def resid_sorted(result):
    """Residuals of a streaming result, sorted, as a numpy array.

    Sorted because the streaming residual file is in fe[0]-bucket order, not
    the input row order; comparing sorted residuals checks the multiset.
    """
    return np.sort(result.resid().select("resid").collect()["resid"].to_numpy())

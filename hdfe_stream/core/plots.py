"""Plots of results, drawn with matplotlib, which is not a dependency of this
package: each function imports it when called and says how to install it when it
is missing (``pip install hdfe-stream[plot]``)."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def require_matplotlib(what: str):
    """matplotlib.pyplot, or an ImportError that names the `plot` extra."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            f"{what} needs matplotlib, which is not installed; install it with "
            "`pip install matplotlib` (or the `plot` extra, `pip install hdfe-stream[plot]`)"
        ) from exc
    return plt


def event_study_plot(times: Sequence[int], estimates: Any, lower: Any, upper: Any, *,
                     reference: int | None = None, ax: Any = None, title: str | None = None,
                     ylabel: str = "Estimate and confidence interval",
                     xlabel: str = "Event time"):
    """An event study: each coefficient and its interval against its event time, the
    omitted reference period at zero (hollow), a line at zero and a dashed line between
    the reference period and the first period after it. Draws on `ax` if given;
    returns the figure.

    Parameters
    ----------
    times : sequence of int
        The coefficients' event times.
    estimates, lower, upper : array-like
        The coefficients and their intervals' ends.
    reference : int, optional
        The omitted reference event time, drawn at zero when it is not among
        `times`.
    ax : matplotlib.axes.Axes, optional
    title, ylabel, xlabel : str, optional
    """
    plt = require_matplotlib("event_study_plot")
    times = np.asarray(times, dtype=float)
    est = np.asarray(estimates, dtype=float)
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    order = np.argsort(times)
    times, est, lo, hi = times[order], est[order], lo[order], hi[order]
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    else:
        fig = ax.figure
    color = "#1b6ca8"
    ax.axhline(0, color="0.6", linewidth=0.8)
    if reference is not None:
        ax.axvline(reference + 0.5, color="0.6", linewidth=0.8, linestyle="--")
    ax.errorbar(times, est, yerr=[est - lo, hi - est], fmt="o", color=color, capsize=3,
                markersize=5)
    if reference is not None and reference not in set(times.tolist()):
        ax.plot([reference], [0.0], marker="o", markersize=5, markerfacecolor="white",
                markeredgecolor=color, linestyle="none")
    shown = np.concatenate([times, [reference]]) if reference is not None else times
    ax.set_xticks(np.arange(shown.min(), shown.max() + 1))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    return fig

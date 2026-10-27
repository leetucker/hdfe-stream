"""Progress messages, warnings and report text.

Everything user-facing goes through here so that a caller who passes
`logger=` gets the whole run in their log, and a caller who does not gets
timestamped prints.
"""

from __future__ import annotations

import logging
import time
import warnings


def _log(verbose, msg, logger=None, level=None):
    """Progress message: to `logger` if given (its own formatting and
    timestamps apply), else printed to stdout with a timestamp."""
    if not verbose:
        return
    if logger is not None:
        logger.log(level if level is not None else logging.INFO, msg)
    else:
        print(f"[hdfe {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _warn(msg, logger=None):
    """Warnings go to logger.warning when a logger is given (so they show up
    in the log in real time), else through the warnings module."""
    if logger is not None:
        logger.warning(msg)
    else:
        warnings.warn(msg, stacklevel=3)


def _emit(text, logger=None, level=None):
    """Multi-line report text (summaries): one log record, or print."""
    if logger is not None:
        logger.log(level if level is not None else logging.INFO, text)
    else:
        print(text)

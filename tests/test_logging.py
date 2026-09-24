"""Where progress messages, warnings and summaries go.

A long out-of-core fit is not interactive, so the routing matters: with a
`logger=` everything must go to the logger *as it happens* (not buffered until
the end), and nothing must leak to stdout. Without one, everything prints.
"""

from __future__ import annotations

import contextlib
import io
import logging
import time

import pytest

from hdfe_stream import feols_stream

pytest.importorskip("pyfixest")

FML = "log_earn ~ age_squared + i(year) | worker_id + firm_id + year"


class Recorder(logging.Handler):
    """Captures records with the wall-clock time each one arrived."""

    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append((time.time(), record.levelname, record.getMessage()))

    @property
    def messages(self):
        return [m for _, _, m in self.records]

    def levels(self, name):
        return [m for _, level, m in self.records if level == name]


@pytest.fixture
def logger():
    log = logging.getLogger("hdfe_test")
    log.setLevel(logging.DEBUG)
    log.propagate = False
    recorder = Recorder()
    log.addHandler(recorder)
    yield log, recorder
    log.removeHandler(recorder)


def test_logger_receives_progress_and_nothing_reaches_stdout(rich, workdir, logger):
    log, recorder = logger
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        fit = feols_stream(FML, rich.src, workdir=workdir, logger=log)
        fit.summary(logger=log)

    assert stdout.getvalue() == ""
    assert len(recorder.levels("INFO")) > 5
    assert any("pass 0" in m for m in recorder.messages)
    fit.cleanup()


def test_progress_is_logged_as_it_happens(rich, workdir, logger):
    """Records must be spread across the fit, not emitted in one burst at the
    end -- otherwise a long job looks hung."""
    log, recorder = logger
    with contextlib.redirect_stdout(io.StringIO()):
        start = time.time()
        fit = feols_stream(FML, rich.src, workdir=workdir, logger=log)
        elapsed = time.time() - start

    stamps = [t for t, level, _ in recorder.records if level == "INFO"]
    assert stamps[0] - start < elapsed          # the first record predates the end
    assert stamps[-1] - stamps[0] > 0
    fit.cleanup()


def test_warnings_go_to_the_logger(rich, workdir, logger):
    """i(year) is collinear with the year fixed effect, so this fit warns. With
    a logger the warning must be a WARNING record, not a Python warning."""
    log, recorder = logger
    with contextlib.redirect_stdout(io.StringIO()):
        fit = feols_stream(FML, rich.src, workdir=workdir, logger=log, verbose=False)

    warnings_logged = recorder.levels("WARNING")
    assert any("multicollinearity" in m for m in warnings_logged), warnings_logged
    fit.cleanup()


def test_verbose_false_logs_warnings_but_not_progress(rich, workdir, logger):
    log, recorder = logger
    with contextlib.redirect_stdout(io.StringIO()):
        fit = feols_stream("log_earn ~ age_squared | worker_id + firm_id", rich.src,
                           workdir=workdir, logger=log, verbose=False)
    assert recorder.levels("INFO") == []
    fit.cleanup()


def test_log_level_is_configurable(rich, workdir, logger):
    log, recorder = logger
    with contextlib.redirect_stdout(io.StringIO()):
        fit = feols_stream("log_earn ~ age_squared | worker_id + firm_id", rich.src,
                           workdir=workdir, logger=log, log_level=logging.DEBUG)
    assert recorder.levels("DEBUG")
    assert recorder.levels("INFO") == []
    fit.cleanup()


def test_summary_is_one_record(rich, workdir, logger):
    """The summary is a multi-line report; it goes out as a single record so a
    log aggregator does not interleave it with other lines."""
    log, recorder = logger
    with contextlib.redirect_stdout(io.StringIO()):
        fit = feols_stream("log_earn ~ age_squared | worker_id + firm_id", rich.src,
                           workdir=workdir, verbose=False)
    fit.summary(logger=log)
    assert len(recorder.records) == 1
    assert recorder.messages[0].startswith("###")
    assert "\n" in recorder.messages[0]
    fit.cleanup()


def test_multi_summary_can_be_one_record_or_one_per_model(rich, workdir, logger):
    log, recorder = logger
    with contextlib.redirect_stdout(io.StringIO()):
        multi = feols_stream("log_earn + y2 ~ age_squared | worker_id + firm_id",
                             rich.src, workdir=workdir, verbose=False)
    multi.summary(logger=log)
    assert len(recorder.records) == 1

    recorder.records.clear()
    multi.summary(logger=log, per_model=True)
    assert len(recorder.records) == len(multi)
    multi.cleanup()


def test_without_a_logger_everything_prints(rich, workdir):
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        feols_stream("log_earn ~ age_squared | worker_id + firm_id", rich.src,
                     workdir=workdir).summary()
    printed = stdout.getvalue()
    assert printed.count("[hdfe ") > 5      # timestamped progress lines
    assert "###" in printed                 # and the summary

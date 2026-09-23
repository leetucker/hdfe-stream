"""Check logger routing: progress in real time, warnings, summaries, default print."""
import io, logging, time, contextlib, warnings
from hdfe_stream import feols_stream

class Recorder(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []
    def emit(self, record):
        self.records.append((time.time(), record.levelname, record.getMessage()))

log = logging.getLogger("hdfe_test")
log.setLevel(logging.DEBUG)
rec = Recorder()
log.addHandler(rec)
log.propagate = False

buf = io.StringIO()
with contextlib.redirect_stdout(buf), warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    t0 = time.time()
    fit = feols_stream("log_earn ~ age_squared + i(year) | pik + sein + year", "simf.parquet",
                       workdir="lt/a", logger=log)
    t1 = time.time()
    fit.summary(logger=log)
print(f"stdout during fit+summary with logger: {len(buf.getvalue())} chars; python warnings: {len(w)}")
prog = [r for r in rec.records if r[1] == "INFO"]
print(f"{len(rec.records)} records: {sum(r[1] == 'INFO' for r in rec.records)} INFO, "
      f"{sum(r[1] == 'WARNING' for r in rec.records)} WARNING")
print(f"first progress record {prog[0][0] - t0:.2f}s after start; fit took {t1 - t0:.2f}s "
      f"(records spread over {prog[-2][0] - prog[0][0]:.2f}s -> real time)")
for _, lv, msg in rec.records[:4] + [r for r in rec.records if r[1] == "WARNING"]:
    print(f"   {lv:7s} {msg[:100]}")
print("summary record starts:", repr(rec.records[-1][2][:60]))

# verbose=False + logger: no progress, warnings still logged; custom level
rec.records.clear()
multi = feols_stream("log_earn + y2 ~ age_squared | pik + sein", "simf.parquet", workdir="lt/b",
                     logger=log, verbose=False, log_level=logging.DEBUG)
print(f"verbose=False: {len(rec.records)} records")
multi.summary(logger=log, per_model=True)
print(f"multi summary per_model: {len(rec.records)} records")

# default: prints
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    feols_stream("log_earn ~ age_squared | pik + sein", "simf.parquet", workdir="lt/c").summary()
print(f"default print: {buf.getvalue().count('[hdfe ')} progress lines + summary "
      f"({'### ' in buf.getvalue()})")
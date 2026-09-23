"""Disk cleanup behaviour of hdfe_stream."""
import gc, os, shutil, socket, time
from pathlib import Path
import hdfe_stream as hs
from hdfe_stream import feols_stream

WD = Path("ct")
shutil.rmtree(WD, ignore_errors=True)
FML = "log_earn ~ age_squared + age_cubed | pik + sein + year"


def runs():
    return sorted(p.name for p in WD.glob("hdfe_run_*")) if WD.exists() else []


def listing(run):
    return sorted(str(p.relative_to(run)) for p in run.rglob("*") if p.is_file())


# 1. normal fit: only result files remain; they go when the result is dropped
fit = feols_stream(FML, "simf.parquet", workdir=WD, verbose=False)
run = Path(fit.files_dir).parents[1]
print("1. after fit:", listing(run))
print(f"   peak disk {fit.diagnostics['disk_peak_gb']} GB, results {fit.diagnostics['disk_results_gb']} GB")
n = fit.resid().select("resid").collect().height
del fit
gc.collect()
print(f"   resid rows read before drop: {n:,}; after del: run dirs = {runs()}")

# 2. lazy frames outlive the result -> clear error
fit = feols_stream(FML, "simf.parquet", workdir=WD, verbose=False)
lf_ok = fit.fixef("sein").collect()          # collected: safe
fit2 = fit.with_vcov("hetero")               # copies share the run
del fit
gc.collect()
print("2. copy keeps files alive:", fit2.resid().select("resid").collect().height > 0)
fit2.cleanup()
try:
    fit2.resid()
except FileNotFoundError as e:
    print("   after cleanup():", str(e)[:70], "...")
print("   run dirs:", runs())

# 3. outputs="keep": survives the object; removed by hs.cleanup(workdir)
fit = feols_stream(FML, "simf.parquet", workdir=WD, verbose=False, outputs="keep")
del fit
gc.collect()
print("3. keep, after del:", len(runs()), "run dir(s)")
freed = hs.cleanup(WD)
print(f"   hs.cleanup(WD) removed {len(freed)} run(s), {sum(b for _, b in freed) / 1e6:.1f} MB; left: {runs()}")

# 4. exception after files were written (all covariates collinear -> ValueError in step 3)
try:
    feols_stream("log_earn ~ i(year) | pik + sein + year", "simf.parquet", workdir=WD, verbose=False)
except ValueError as e:
    print("4. error:", str(e)[:60], "| run dirs:", runs())

# 5. interrupt in the middle of the solve
orig = hs.StreamingHDFE._solve
def boom(self):
    raise KeyboardInterrupt
hs.StreamingHDFE._solve = boom
try:
    feols_stream(FML, "simf.parquet", workdir=WD, verbose=False)
except KeyboardInterrupt:
    print("5. KeyboardInterrupt mid-solve | run dirs:", runs())
hs.StreamingHDFE._solve = orig

# 6. sw() over FE sets where the second set fails: the first set's files go too
try:
    feols_stream("log_earn ~ age_squared | sw(pik + sein, pik + nosuch)", "simf.parquet",
                 workdir=WD, verbose=False)
except ValueError as e:
    print("6. second FE set failed:", str(e)[:45], "| run dirs:", runs())

# 7. save_resid=False, and the context manager
with feols_stream(FML, "simf.parquet", workdir=WD, verbose=False, save_resid=False) as fit:
    print("7. files:", listing(Path(fit.files_dir).parents[1]))
    try:
        fit.resid()
    except FileNotFoundError as e:
        print("   resid():", e)
print("   after with-block:", runs())

# 8. leftovers from a killed process (dead pid) are found and removed
dead = WD / "hdfe_run_19700101_000000_deadbeef"
(dead / "rows").mkdir(parents=True)
(dead / "rows" / "x.bin").write_bytes(b"0" * 1_000_000)
(dead / hs._MARKER).write_text(f"{socket.gethostname()} 999999 {time.time()}\n")
print("8. dry run:", [(Path(p).name, b) for p, b in hs.cleanup(WD, dry_run=True)])
print("   removed:", len(hs.cleanup(WD)), "| left:", runs())

# 9. leftover partial/intermediate files never linger after success
fit = feols_stream("log_earn ~ age_squared | pik[year] + sein", "simf.parquet", workdir=WD,
                   verbose=False, cluster=["pik+sein"])
print("9. FEIS + clustering, after fit:", listing(Path(fit.files_dir).parents[1]))
del fit
gc.collect()
print("   final:", runs(), "| workdir contents:", os.listdir(WD))
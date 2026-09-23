"""Compare hdfe_stream against pyfixest for arbitrary FE sets."""
import sys, time
import numpy as np, polars as pl, pyfixest as pf
from hdfe_stream import StreamingHDFE

SRC = sys.argv[1] if len(sys.argv) > 1 else "sim.parquet"
FE = (sys.argv[2] if len(sys.argv) > 2 else "pik+sein+year").split("+")
FE_DOF = sys.argv[3] if len(sys.argv) > 3 else "exact"
FML = f"log_earn ~ age_squared + age_cubed | {' + '.join(FE)}"
print(FML)

df = pl.read_parquet(SRC).to_pandas()
vtypes = [("iid", "iid"), ("hetero", "hetero")] + [(f"CRV1:{f}", {"CRV1": f}) for f in FE]
t0 = time.time()
ref = {v: pf.feols(FML, data=df, vcov=vc, fixef_rm="none",
                  demeaner=pf.LsmrDemeaner(fixef_atol=1e-12, fixef_btol=1e-12))
       for v, vc in vtypes}
print(f"pyfixest: {time.time()-t0:.1f}s")
r0 = ref["iid"]; b_ref = r0.coef().to_numpy()
print(r0.tidy())

for solver in ["explicit", "stream_cg", "within"]:
    print(f"\n====== solver={solver} fe_dof={FE_DOF} ======")
    est = StreamingHDFE("log_earn", ["age_squared", "age_cubed"], FE, workdir=f"wh_{solver}",
                        solver=solver, keep=["year"], batch_rows=100_000, n_buckets=3,
                        tol=1e-11, verbose=False)
    res = est.fit(SRC, vcov="iid", cluster=tuple(FE), fe_dof=FE_DOF)
    res.summary(); print("diagnostics:", res.diagnostics)
    print(f"max |beta diff|/|beta|: {np.max(np.abs(res.beta-b_ref)/np.abs(b_ref)):.2e}")
    for v, fit in ref.items():
        se_ref = fit.se().to_numpy(); se = np.sqrt(np.diag(res.all_vcovs[v][0]))
        print(f"  SE {v:14s} rel diff: {np.max(np.abs(se-se_ref)/se_ref):.2e}")
    cols = list(dict.fromkeys(["pik", "year", *FE]))
    mine = res.resid().select(*cols, "resid", *[f"fe_{f}" for f in FE]).collect()
    theirs = pl.DataFrame({**{f: df[f] for f in cols}, "resid_ref": r0.resid(),
                           "fesum_ref": r0.predict() - df[["age_squared", "age_cubed"]].to_numpy() @ b_ref})
    m = mine.join(theirs, on=["pik", "year"])
    fesum = sum(m[f"fe_{f}"] for f in FE)
    print(f"  rows matched {m.height:,}/{len(df):,}   "
          f"max|resid diff| {np.abs(m['resid']-m['resid_ref']).max():.2e}   "
          f"max|FE sum diff| {np.abs(fesum - m['fesum_ref']).max():.2e}")
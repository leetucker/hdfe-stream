"""Validate IV, weights, fit statistics and the pf.etable adapter against pyfixest."""
import os, sys, warnings
import numpy as np, polars as pl, pyfixest as pf
from hdfe_stream import feols_stream, etable

warnings.filterwarnings("ignore")
SRC = "simiv.parquet"


def make_data(path=SRC):
    from validate_formula import SRC as FSRC, make_data as make_f
    if not os.path.exists(FSRC):
        make_f()
    df = pl.read_parquet(FSRC)
    rng = np.random.default_rng(21)
    n = df.height
    z1, z2, v = (pl.Series(rng.normal(0, 1, n)) for _ in range(3))
    df = df.with_columns(z1=z1, z2=z2,
                         x2=0.6 * z1 + 0.3 * z2 + v + 0.1 * pl.col("age_squared"),
                         wa=pl.Series(rng.uniform(0.5, 2.0, n)),
                         wf=pl.Series(rng.integers(1, 5, n)).cast(pl.Float64))
    df = df.with_columns(y_iv=pl.col("log_earn") + 0.2 * pl.col("x2") + 0.5 * v)
    df.write_parquet(path)


if not os.path.exists(SRC):
    make_data()
df = pl.read_parquet(SRC).to_pandas()

CASES = [
    ("log_earn ~ age_squared + age_cubed | pik + sein + year", dict(weights="wa")),
    ("log_earn ~ age_squared + age_cubed | pik + sein + year", dict(weights="wf", weights_type="fweights")),
    ("y_iv ~ age_squared | pik + sein | x2 ~ z1 + z2", {}),
    ("y_iv ~ age_squared | pik + sein | x2 ~ z1 + z2", dict(weights="wa")),
    ("y_iv ~ age_squared + i(year, treat, ref=2009) | pik + sein + year | x2 ~ z1", {}),
    ("y_iv ~ 1 | pik + sein^year | x2 ~ z1 + z2", dict(weights="wf", weights_type="fweights")),
]
VC = [("iid", "iid"), ("hetero", "hetero"), ("CRV1:pik", {"CRV1": "pik"}),
      ("CRV1:state", {"CRV1": "state"})]
only = [int(a) for a in sys.argv[1:]]
worst = 0.0
mine_all, ref_all = [], []
for ci, (fml, wopts) in enumerate(CASES):
    if only and ci not in only:
        continue
    res = feols_stream(fml, SRC, workdir=f"vw/{ci}", vcov={"CRV1": "pik"}, cluster=["state"],
                       fe_dof="pyfixest", verbose=False, n_buckets=3, batch_rows=100_000,
                       tol=1e-11, **wopts)
    print(f"[{ci}] {fml}  {wopts}")
    print(f"   N {res.n_obs} | k {len(res.coefnames)} | solver {res.solver_info['solver']}"
          + (f" | first-stage F {res.f_stat_1st_stage}" if res.is_iv else ""))
    kw = dict(fixef_rm="none", weights=wopts.get("weights"),
              weights_type=wopts.get("weights_type", "aweights"))
    try:
        ref = pf.feols(fml, data=df, vcov="iid", **kw,
                       demeaner=pf.LsmrDemeaner(fixef_atol=1e-12, fixef_btol=1e-12))
    except ValueError:      # pyfixest LSMR preconditioner issue in some IV first stages
        ref = pf.feols(fml, data=df, vcov="iid", **kw, fixef_tol=1e-12, fixef_maxiter=500_000)
    for key, vc in VC:
        ref.vcov(vc)
        r = res.with_vcov(key)
        b_ref, se_ref = ref.coef().to_numpy(), ref.se().to_numpy()
        db = np.max(np.abs(r.beta - b_ref) / np.maximum(np.abs(b_ref), 1e-8)) if len(b_ref) else 0
        ds = np.max(np.abs(r.se - se_ref) / se_ref) if len(se_ref) else 0
        line = f"   {key:11s} beta {db:.1e}  se {ds:.1e}"
        if key == "iid":
            stats_ = {"N": (r.n_obs, ref._N), "r2": (r.r2, ref._r2),
                      "adj_r2": (r.adj_r2, ref._adj_r2), "r2_within": (r.r2_within, ref._r2_within),
                      "adj_r2_within": (r.adj_r2_within, ref._adj_r2_within),
                      "rmse": (r.rmse, ref._rmse)}
            line += "  | " + "  ".join(f"{k} {abs(a - b) / max(abs(b), 1e-12):.0e}"
                                       for k, (a, b) in stats_.items())
            e_me = np.sort(res.resid().select("resid").collect()["resid"].to_numpy())
            e_ref = np.sort(np.asarray(ref.resid()))
            line += f"  | resid {np.abs(e_me - e_ref).max():.1e}"
            fin = [(a, b) for a, b in stats_.values() if np.isfinite(b)]
            nan_ok = all(np.isnan(a) == np.isnan(b) for a, b in stats_.values())
            line += "" if nan_ok else "  NaN-MISMATCH"
            worst = max([worst] + [abs(a - b) / max(abs(b), 1e-12) for a, b in fin])
        if res.is_iv and key in ("iid", "CRV1:pik"):
            # reference: pyfixest's first-stage OLS with the same weights and
            # vcov, Wald F on the excluded instruments (this equals pyfixest's
            # IV_Diag F except with fweights, where IV_Diag differs from its
            # own first-stage OLS)
            from pyfixest.estimation.formula.parse import Formula
            spec = Formula.parse(fml)[0]
            fs = pf.feols(f"{spec.first_stage} | {spec.fixed_effects}", data=df, vcov=vc, **kw)
            excl = [c for c in fs.coef().index if c not in ref.coef().index]
            idx = [list(fs.coef().index).index(c) for c in excl]
            bz = fs.coef().to_numpy()[idx]
            refF = bz @ np.linalg.solve(fs._vcov[np.ix_(idx, idx)], bz) / len(idx)
            myF = (res.f_stat_1st_stage[0] if key == res.vcov_type else
                   feols_stream(fml, SRC, workdir=f"vw/{ci}F", vcov=vc, fe_dof="pyfixest",
                                verbose=False, tol=1e-11, **wopts).f_stat_1st_stage[0])
            line += f"  | 1st-stage F {myF:.8g} vs {refF:.8g}"
            worst = max(worst, abs(myF - refF) / refF)
            del fs
        if key == "CRV1:pik" and ci in (0, 2):
            mine_all.append(res)
            ref_all.append(pf.feols(fml, data=df, vcov={"CRV1": "pik"}, fixef_rm="none",
                                    weights=wopts.get("weights")))
        worst = max(worst, db, ds)
        print(line)
    del ref
print(f"\nworst relative difference: {worst:.1e}")

print("\n=== pf.etable: streaming results (left) vs pyfixest (right) ===")
tab = etable([mine_all[0], ref_all[0], mine_all[1], ref_all[1]], type="df")
print(tab.to_string())
print("\n=== pf.summary on an adapter ===")
pf.summary(mine_all[1].to_pyfixest())
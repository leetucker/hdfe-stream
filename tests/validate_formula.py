"""Validate feols_stream against pyfixest.feols over a battery of formulas."""
import sys, time, warnings
import numpy as np, polars as pl, pyfixest as pf
from hdfe_stream import feols_stream, HDFEMulti

warnings.filterwarnings("ignore")
SRC = "simf.parquet"


def make_data(path=SRC):
    """Simulated panel (simulate.py) plus categorical, string, cluster and
    missing-value columns for exercising the formula features."""
    from simulate import simulate
    df = simulate(n_workers=50_000, n_firms=3_333)
    rng = np.random.default_rng(9)
    firms = df["sein"].unique().sort()
    state = pl.DataFrame({"sein": firms, "state": rng.integers(0, 12, len(firms))})
    df = (df.join(state, on="sein")
            .with_columns(treat=(pl.col("pik") % 3 == 0).cast(pl.Int64),
                          occ=pl.Series(rng.integers(0, 6, df.height)),
                          cat=pl.lit("k") + pl.Series(rng.integers(0, 3, df.height)).cast(pl.Utf8),
                          y2=pl.col("log_earn") * 0.5 + pl.Series(rng.normal(0, 0.2, df.height)),
                          x_na=pl.when(pl.Series(rng.random(df.height)) < 0.05).then(None)
                                .otherwise(pl.Series(rng.normal(0, 1, df.height))))
            .with_columns(log_earn=pl.col("log_earn") + 0.03 * pl.col("occ")
                          + pl.when(pl.col("year") >= 2010).then(0.1 * pl.col("treat")).otherwise(0)))
    df.write_parquet(path)


import os
CASES = [
    ("log_earn ~ age_squared + age_cubed | pik + sein + year", None, {}),
    ("log_earn ~ I(age**2) + log(age) + age_squared:age_cubed | pik + sein + year", "hetero", {}),
    ("log_earn ~ i(year, treat, ref=2009) + age_squared | pik + sein", {"CRV1": "sein"}, {}),
    ("log_earn ~ C(occ) + cat:age_squared + age_squared | pik + sein + year", "iid", {}),
    ("log_earn ~ age_squared | pik + sein^year", {"CRV1": "sein"}, {}),
    ("log_earn ~ age_squared + age_cubed | pik + sein^year", {"CRV1": "state"}, {}),
    ("log_earn ~ i(year) + C(treat) + age_squared | pik + sein + year", None, {}),
    ("log_earn ~ 1 | pik + sein", "hetero", {}),
    ("log_earn ~ i(year, age_squared) + i(year, treat, ref=2005) | pik + sein", {"CRV1": "pik"},
     {"rhs_block": 4}),
    ("log_earn ~ age_squared + x_na | pik + sein + year", {"CRV1": "state"}, {}),
    ("log_earn ~ age_squared + i(year, treat, ref=2009) | pik^cat + sein + year", {"CRV1": "pik"}, {}),
    ("log_earn + y2 ~ csw(age_squared, age_cubed) | sw(pik + sein, pik + sein + year)", "iid", {}),
]
if __name__ == "__main__":
    if not os.path.exists(SRC):
        make_data()
    df = pl.read_parquet(SRC).to_pandas()
    only = [int(a) for a in sys.argv[1:]]
    worst = {}
    for ci, (fml, vcov, opts) in enumerate(CASES):
        if only and ci not in only:
            continue
        t = time.time()
        ref = pf.feols(fml, data=df, vcov=vcov if vcov is not None else {"CRV1": fml.split("|")[1].split("+")[0].strip().split("(")[-1]},
                       fixef_rm="none", demeaner=pf.LsmrDemeaner(fixef_atol=1e-12, fixef_btol=1e-12))
        t_ref = time.time() - t
        t = time.time()
        res = feols_stream(fml, SRC, workdir=f"vf/{ci}", vcov=vcov, verbose=False, n_buckets=3, fe_dof="pyfixest",
                           batch_rows=100_000, tol=1e-11, **opts)
        t_me = time.time() - t
        mine = list(res) if isinstance(res, HDFEMulti) else [res]
        refs = ref.all_fitted_models if hasattr(ref, "all_fitted_models") else {mine[0].fml: ref}
        print(f"[{ci}] {fml}   vcov={vcov}  ({len(mine)} model(s); pyfixest {t_ref:.1f}s, stream {t_me:.1f}s)")
        for r in mine:
            f = refs[r.fml]
            rc = f.coef()
            names_ok = list(rc.index) == r.coefnames
            if r.coefnames:
                b = np.array(r.beta); br = rc.to_numpy()
                db = np.max(np.abs(b - br) / np.maximum(np.abs(br), 1e-8))
                ds = np.max(np.abs(r.se - f.se().to_numpy()) / f.se().to_numpy())
            else:
                db = ds = 0.0
            e_ref = f.resid()
            e_me = r.resid().select("resid").collect()["resid"].to_numpy()
            n_ok = (len(e_ref) == len(e_me))
            de = abs(np.sort(e_me) - np.sort(e_ref)).max() if n_ok else np.inf
            print(f"   {r.fml:70s} names={'ok' if names_ok else 'MISMATCH'} k={len(r.coefnames):2d} "
                  f"collin={len(r.collin_vars)}/{len(getattr(f, "_collin_vars", []) or [])} n={r.n_obs}/{f._N} "
                  f"beta {db:.1e} se {ds:.1e} resid {de:.1e}  [{r.solver_info['solver']}, {r.diagnostics['assembly']}]")
            if not names_ok:
                print("      mine:", r.coefnames[:6], "\n      ref: ", list(rc.index)[:6])
            worst[r.fml] = max(db, ds)
    print("\nworst beta/SE relative difference over all models:", f"{max(worst.values()):.1e}")
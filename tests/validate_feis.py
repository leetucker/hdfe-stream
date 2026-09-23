"""Validate varying slopes on the streamed dimension (FEIS) against a
brute-force pyfixest regression with explicit worker-by-slope dummies, and
check the choice of streamed dimension."""
import os, warnings
import numpy as np, polars as pl, pyfixest as pf
from hdfe_stream import feols_stream, StreamingHDFE

warnings.filterwarnings("ignore")
SRC = "simfeis.parquet"


def make_data(path=SRC, n_workers=600, n_firms=60, seed=3):
    from simulate import simulate
    df = simulate(n_workers=n_workers, n_firms=n_firms, seed=seed)
    rng = np.random.default_rng(seed)
    wk = df["pik"].unique().sort()
    trend = pl.DataFrame({"pik": wk, "g": rng.normal(0.02, 0.02, len(wk)),
                          "g2": rng.normal(0, 0.002, len(wk))})
    n = df.height
    df = (df.join(trend, on="pik")
            .with_columns(t=pl.col("year").cast(pl.Float64),
                          x=pl.Series(rng.normal(0, 1, n)), z=pl.Series(rng.normal(0, 1, n)),
                          wa=pl.Series(rng.uniform(0.5, 2, n)), v=pl.Series(rng.normal(0, 1, n)))
            .with_columns(t2=(pl.col("t") - 2009.5) ** 2,
                          xe=pl.col("z") + 0.5 * pl.col("v") + 0.3 * pl.col("x"))
            .with_columns(y=pl.col("log_earn") + pl.col("g") * (pl.col("t") - 2005)
                          + pl.col("g2") * pl.col("t2") + 0.3 * pl.col("x") + 0.2 * pl.col("xe")
                          + 0.4 * pl.col("v"))
            .drop("g", "g2"))
    df.write_parquet(path)


if not os.path.exists(SRC):
    make_data()
df = pl.read_parquet(SRC).with_columns(tc=pl.col("t") - 2009.5).to_pandas()   # centered for the dummies
print(f"{len(df):,} rows, {df.pik.nunique():,} workers, {df.sein.nunique():,} firms")

CASES = [
    ("y ~ x | pik[t] + sein", "y ~ x + C(pik):tc | pik + sein", {}),
    ("y ~ x | sein + pik[t] + year", "y ~ x + C(pik):tc | sein + pik + year", {}),
    ("y ~ x | pik[t, t2] + sein", "y ~ x + C(pik):tc + C(pik):t2 | pik + sein", {"weights": "wa"}),
    ("y ~ x | pik[t] + sein | xe ~ z", "y ~ x + C(pik):tc | pik + sein | xe ~ z", {}),
]
worst = 0.0
for ci, (fml, brute, opts) in enumerate(CASES):
    for solver in ("explicit", "stream_cg"):
        r = feols_stream(fml, SRC, workdir=f"vfe/{ci}{solver}", vcov="iid",
                         cluster=["sein", "pik"], fe_dof="pyfixest", verbose=False, tol=1e-12,
                         solver=solver, **opts)
        ref = pf.feols(brute, data=df, vcov="iid", fixef_rm="none", weights=opts.get("weights"), collin_tol=1e-7,
                       demeaner=pf.LsmrDemeaner(fixef_atol=1e-13, fixef_btol=1e-13))
        names = r.coefnames
        rc = ref.coef()[names].to_numpy()
        db = np.max(np.abs(r.beta - rc) / np.abs(rc))
        line = f"[{ci}] {fml:42s} {solver:9s} stream={r.diagnostics['stream']['dim']:4s} beta {db:.1e}"
        idx = [list(ref.coef().index).index(c) for c in names]
        for key, vc in [("iid", "iid"), ("hetero", "hetero"), ("CRV1:sein", {"CRV1": "sein"})]:
            ref.vcov(vc)
            se_ref = ref.se().to_numpy()[idx]
            ds = np.max(np.abs(r.with_vcov(key).se - se_ref) / se_ref)
            line += f"  {key} {ds:.1e}"
            worst = max(worst, ds)
        # clustering on the streamed dimension: fixest counts the worker slopes
        # as FE nested in the cluster (dropped from K); the dummy regression
        # counts them as covariates. Check the SE ratio equals that dof ratio.
        ref.vcov({"CRV1": "pik"})
        N, n_slopes = r.n_obs, r.diagnostics["fe_params"]["pik"] - r.n_levels["pik"]
        Kc_me = len(names) + r.k_fe - r.diagnostics["fe_params"]["pik"] + 1
        ratio = np.sqrt((N - Kc_me) / (N - (Kc_me + n_slopes)))
        dp = np.max(np.abs(r.with_vcov("CRV1:pik").se * ratio - ref.se().to_numpy()[idx])
                    / ref.se().to_numpy()[idx])
        line += f"  CRV1:pik(dof-adj) {dp:.1e}"
        e_me = np.sort(r.resid().select("resid").collect()["resid"].to_numpy())
        line += f"  resid {np.abs(e_me - np.sort(np.asarray(ref.resid()))).max():.1e}"
        worst = max(worst, db, dp)
        print(line)
print(f"\nworst relative difference: {worst:.1e}")

# age^2 = (birth + t)^2 is spanned by worker intercepts and slopes on t, t2
r2 = feols_stream("y ~ x + age_squared | pik[t, t2] + sein", SRC, workdir="vfe/col", verbose=False)
print("collinear with worker slopes:", r2.collin_vars)

# worker FE file: intercept and slope per worker
print(r.fixef("pik").head(3).collect())

# choice of the streamed dimension
for fe, stream, expect in [(["sein", "pik"], None, "pik"), (["sein", "pik[t]"], None, "pik"),
                           (["pik", "sein"], "sein", "sein"), (["year", "sein"], None, "sein")]:
    e = StreamingHDFE("y", ["x"], fe, workdir="vfe/s", stream=stream, verbose=False)
    r = e.fit(SRC)
    print(f"fe={fe} stream={stream}: streamed {r.diagnostics['stream']['dim']} "
          f"[{r.diagnostics['stream']['reason']}] {'ok' if r.diagnostics['stream']['dim'] == expect else 'WRONG'}")
for fe, stream in [(["pik[t]", "sein"], "sein"), (["pik[t]", "sein[t]"], None)]:
    try:
        StreamingHDFE("y", ["x"], fe, workdir="vfe/s", stream=stream)
    except ValueError as err:
        print(f"fe={fe} stream={stream}: ValueError: {err}")
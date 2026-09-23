"""Validate multi-way clustered SEs against pyfixest (two-way) and against a
direct inclusion-exclusion computation from pyfixest's scores (three-way)."""
import os, sys, warnings
import numpy as np, polars as pl, pyfixest as pf
from hdfe_stream import feols_stream

warnings.filterwarnings("ignore")
SRC = "simiv.parquet"
if not os.path.exists(SRC):
    from validate_iv_weights import make_data
    make_data()
df = pl.read_parquet(SRC).with_columns(
    region=(pl.col("state") % 4), cohort=(pl.col("year") - pl.col("age")) // 10).to_pandas()
pl.from_pandas(df).write_parquet("simmw.parquet")

CASES = [  # (formula, two-way cluster spec, options)
    ("log_earn ~ age_squared + age_cubed | pik + sein + year", "pik+sein", {}),
    ("log_earn ~ age_squared + age_cubed | pik + sein + year", "sein+year", {}),
    ("log_earn ~ age_squared + age_cubed | pik + sein", "pik+state", {}),
    ("log_earn ~ age_squared + age_cubed | pik + sein", "sein+state", {}),
    ("log_earn ~ age_squared + i(year, treat, ref=2009) | pik + sein", "state+year", {}),
    ("log_earn ~ age_squared + age_cubed | pik + sein^year", "sein+cohort", {"weights": "wa"}),
    ("y_iv ~ age_squared | pik + sein | x2 ~ z1 + z2", "pik+sein", {}),
    ("log_earn ~ age_squared + age_cubed | pik + sein", "pik+sein",
     {"weights": "wf", "weights_type": "fweights"}),
]
only = [int(a) for a in sys.argv[1:]]
worst = 0.0
for ci, (fml, cl, opts) in enumerate(CASES):
    if only and ci not in only:
        continue
    res = feols_stream(fml, "simmw.parquet", workdir=f"vm/{ci}", vcov={"CRV1": cl},
                       fe_dof="pyfixest", verbose=False, n_buckets=3, batch_rows=100_000,
                       tol=1e-11, **opts)
    ref = pf.feols(fml, data=df, vcov={"CRV1": cl}, fixef_rm="none",
                   weights=opts.get("weights"), weights_type=opts.get("weights_type", "aweights"),
                   demeaner=pf.LsmrDemeaner(fixef_atol=1e-12, fixef_btol=1e-12))
    # compare the vcov itself: few-cluster multi-way vcovs can have negative
    # variances (NaN SEs), identically in both
    ds = np.max(np.abs(res.vcov - ref._vcov)) / np.max(np.abs(ref._vcov))
    db = np.max(np.abs(res.beta - ref.coef().to_numpy()) / np.abs(ref.coef().to_numpy()))
    print(f"[{ci}] {fml:70s} CRV1 {cl:12s} G {res.n_clusters[cl]} vs {ref._G[:2]}  "
          f"df_t {res.df_t} vs {ref._df_t:.0f}  beta {db:.1e}  vcov {ds:.1e}  {opts or ''}")
    worst = max(worst, db, ds)
    del ref
print(f"\nworst two-way relative difference: {worst:.1e}")

# three-way: inclusion-exclusion from pyfixest's own scores, pyfixest-style ssc
fml, ways = "log_earn ~ age_squared + age_cubed | pik + sein", ["pik", "sein", "year"]
ref = pf.feols(fml, data=df, vcov="hetero", fixef_rm="none",
               demeaner=pf.LsmrDemeaner(fixef_atol=1e-12, fixef_btol=1e-12))
scores, bread = ref._scores, np.linalg.inv(ref._tZX)
N, K = ref._N, len(ref.coef()) + sum(ref._k_fe) - 1
from itertools import combinations
Gs = [df[w].nunique() for w in ways]
Gm = min(Gs)
nested = [d for d in ["pik", "sein"] if d in ways]          # FEs that are cluster vars
Kc = K - sum(df[d].nunique() for d in nested) + len(nested)
V = np.zeros_like(bread)
for r in range(1, 4):
    for combo in combinations(ways, r):
        key = df[list(combo)].astype(str).agg("-".join, axis=1) if r > 1 else df[combo[0]]
        codes = key.factorize()[0]
        S = np.column_stack([np.bincount(codes, weights=scores[:, j]) for j in range(scores.shape[1])])
        V += (-1) ** (r + 1) * Gm / (Gm - 1) * (N - 1) / (N - Kc) * bread @ (S.T @ S) @ bread
res = feols_stream(fml, "simmw.parquet", workdir="vm/3way", vcov={"CRV1": "pik+sein+year"},
                   fe_dof="pyfixest", verbose=False, tol=1e-11)
se_ref = np.sqrt(np.diag(V))
print(f"three-way pik+sein+year: G {res.n_clusters['pik+sein+year']} vs {Gs}; "
      f"se rel diff {np.max(np.abs(res.se - se_ref) / se_ref):.1e}; df_t {res.df_t} vs {Gm - 1}")
print(res.tidy())
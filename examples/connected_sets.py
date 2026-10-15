"""Connected sets: which workers and firms a sample keeps, and why.

Worker and firm effects are only comparable within a connected set: firms
linked, directly or through other firms, by workers who moved between them.
`connected_components` finds the components of the worker-firm network by
union-find over the rows as they stream, and returns small tables -- one per
dimension, giving each level its component -- not rows. Components are
numbered by size, so component 0 is the largest.

Three ways of restricting the sample are compared here:

  the largest connected set        connected_components(...).largest()
  pruning by degree                min_degree=: drop levels with too few
                                   distinct partners, repeatedly, first
  the leave-one-out connected set  leave_one_out=True: what KSS estimation needs

first on a network small enough to read row by row, then on a simulated panel
with little mobility, and last inside a fit (connected=, min_degree=), where
`sample()` says which rows each one dropped.

    python examples/connected_sets.py
"""

import time
import warnings

import polars as pl

from hdfe_stream import connected_components, feols_stream
from simulated_data import thin_panel_path, workdir

FE = "worker_id + firm_id"
# the fits below report their singletons in the table instead
warnings.filterwarnings("ignore", message=".*singleton observations dropped")

# ---------------------------------------------------------- a network to read
# Workers 1-4 move around firms A, B and C, which they tie together several
# times over. Around them:
#   5   stays at A for three years             a stayer
#   6   moves between B and D, and is D's only worker
#   7   is seen once, at C
#   8   moves between C and E, and is the only link to E ...
#   9   ... where 9 stays
#   10, 11 move between G and H, which no one else touches: a second component
# Worker 1 also has a year with no firm recorded. A missing id is in no
# component: it is not one big firm that every such worker shares.
rows = [(1, "A"), (1, "B"), (1, None), (2, "A"), (2, "B"), (2, "B"), (3, "B"), (3, "C"),
        (4, "C"), (4, "A"), (5, "A"), (5, "A"), (5, "A"), (6, "B"), (6, "D"), (7, "C"),
        (8, "C"), (8, "E"), (9, "E"), (9, "E"), (10, "G"), (10, "H"), (11, "G"),
        (11, "H")]
small = pl.DataFrame(rows, schema=["worker_id", "firm_id"], orient="row")

cc = connected_components(small, FE, verbose=False)
print(cc.summary())
print("\ncomponent sizes:")
print(cc.sizes)
print("\neach firm's component (the level tables are what is returned, not rows):")
print(cc.levels["firm_id"])

# "Largest" counts rows by default; rank_by= counts the levels of either
# dimension instead. Here the order is the same either way.
by_firms = connected_components(small, FE, rank_by="firm_id", verbose=False)
print(f"\nranked by firms instead of rows: {by_firms.sizes['firm_id'].to_list()} firms "
      f"in components 0 and 1")

# One label() per setting, side by side: a null component means the row is in
# no component -- its id is missing, or one of its levels was removed.
settings = {
    "components": {},
    "min_degree=(2, 1)": {"min_degree": (2, 1)},
    "min_degree=2": {"min_degree": 2},
    "leave_one_out": {"leave_one_out": True},
}
found = {name: connected_components(small, FE, verbose=False, **kw)
         for name, kw in settings.items()}
side_by_side = small.with_columns(
    [found[name].label().collect()["component"].alias(name) for name in settings])
with pl.Config(tbl_rows=30, tbl_cols=10, tbl_width_chars=120):
    print("\nevery row's component under each setting (null: dropped):")
    print(side_by_side)

print("""
Reading the columns:

  min_degree=(2, 1)  keeps movers only: workers seen at two or more firms (5, 7
                     and 9 go), and the firms they were seen at. Nothing else
                     needs repeating, so the island G-H stays, as component 1.
  min_degree=2       also needs two workers at every firm, and repeats: D goes
                     (one worker), which leaves 6 with one firm, so 6 goes;
                     9 goes (one firm), which leaves E with one worker, so E
                     goes, and then 8. What is left is the 2-core: workers 1-4
                     on A, B and C, and the island, still a component of its own.
  leave_one_out      takes the largest component first, so the island goes. It
                     drops 7, seen only once (its leverage would be 1), and the
                     workers whose departure would split the network: 6, the
                     only link to D, and 8, the only link to E and 9. But it
                     keeps 5: a stayer with three observations is fine for KSS,
                     while min_degree=2 removes every stayer.""")
for name in ("min_degree=(2, 1)", "min_degree=2", "leave_one_out"):
    print(f"\n{found[name].summary()}")

# ----------------------------------------------------------- a thin panel
# 20,000 workers over ten years, half the person-years observed, 2% moving in a
# year: hundreds of components, and many firms held by a single mover. Each
# setting is one call; the data is read from disk in batches every time.
data = thin_panel_path()


def movers(frame):
    """Workers seen at two or more firms."""
    return (frame.group_by("worker_id").agg(pl.col("firm_id").n_unique())
                 .filter(pl.col("firm_id") > 1).height)


print(f"\n{'=' * 78}\nthe same settings on a panel with little mobility\n{'=' * 78}")
print(f"{'setting':<28}{'comps':>7}{'rows':>9}{'workers':>9}{'movers':>8}{'firms':>7}"
      f"{'seconds':>9}")
settings_panel = {
    "everything": None,
    "largest connected set": {},
    "min_degree=(2, 1), largest": {"min_degree": (2, 1)},
    "min_degree=(1, 5), largest": {"min_degree": (1, 5)},
    "min_degree=2, largest": {"min_degree": 2},
    "leave_one_out": {"leave_one_out": True},
}
for name, kw in settings_panel.items():
    t0 = time.time()
    if kw is None:
        kept = pl.scan_parquet(data).collect()
        comps = "-"
    else:
        sets = connected_components(data, FE, verbose=False, **kw)
        kept = sets.largest().collect()
        comps = f"{sets.n_components:,}"
    seconds = time.time() - t0
    print(f"{name:<28}{comps:>7}{kept.height:>9,}{kept['worker_id'].n_unique():>9,}"
          f"{movers(kept):>8,}{kept['firm_id'].n_unique():>7,}{seconds:>9.2f}")

print("""
The largest connected set loses a tenth of the rows, in hundreds of small
components. Movers only keeps a few thousand workers but every firm of the
largest set, since each is reached by a mover. min_degree=(1, 5) keeps firms
with at least five distinct workers: the cut is on firms, and the workers who go
are those whose every firm was cut, so one round settles it. min_degree=2
removes every stayer, and firms reached by one mover. leave_one_out keeps the
stayers, which is why its rows are so many more, but loses the firms held to the
rest by a single mover; every mover and firm it keeps, min_degree=2 keeps too.""")

# Partners are distinct levels of the other dimension. With firm-year effects
# the network is worker x firm^year, and a stayer seen in two years at one firm
# has two partners, so movers only (2, 1) is no longer movers only.
year = connected_components(data, "worker_id + firm_id^year", min_degree=(2, 1),
                            verbose=False)
print(f"\nmin_degree=(2, 1) on worker_id x firm_id^year keeps "
      f"{year.sizes['worker_id'][0]:,} workers in its largest set:\n"
      "anyone seen in two firm-years, stayers included.")

# ----------------------------------------------------------------- in a fit
# The same restrictions as fit options, applied to the rows the model can use,
# before singletons are dropped -- so a fit equals one on the data restricted
# beforehand. sample() says why each row was dropped.
print(f"\n{'=' * 78}\nin a fit\n{'=' * 78}")
FML = "log_earn ~ age_squared | worker_id + firm_id + year"
fits = {
    "no restriction": {},
    'connected="largest"': {"connected": "largest"},
    'min_degree=(2, 1), "largest"': {"min_degree": (2, 1), "connected": "largest"},
    'connected="leave_one_out"': {"connected": "leave_one_out"},
}
for name, options in fits.items():
    fit = feols_stream(FML, data, workdir=workdir("connected_sets"), verbose=False,
                       **options)
    reasons = dict(fit.sample().collect().group_by("dropped_because").len()
                      .sort("dropped_because").iter_rows())
    dropped = ", ".join(f"{k} {v:,}" for k, v in reasons.items() if k is not None)
    print(f"{name:<32} n = {fit.n_obs:>7,}   age_squared {fit.coef()['age_squared']:+.5f}"
          f"   components: {fit.n_components:,}")
    print(f"{'':<32} dropped: {dropped}")
    fit.cleanup()

# The pair defaults to the first two fixed effects listed; with year listed
# second, connected_fe names the pair (worker x year would be the wrong graph).
fit = feols_stream("log_earn ~ age_squared | worker_id + year + firm_id", data,
                   workdir=workdir("connected_sets"), verbose=False,
                   connected="largest", connected_fe="worker_id + firm_id")
print(f"\nwith year listed second and connected_fe='worker_id + firm_id': "
      f"n = {fit.n_obs:,}, the same sample")
fit.cleanup()

# The fit's component labels (on the firm effects) are numbered the same way:
# component 0 is the largest, by rows in the estimation sample. Keeping the
# singletons makes that sample the data itself, so the firms of component 0
# are exactly those of the largest connected set.
fit = feols_stream(FML, data, workdir=workdir("connected_sets"), verbose=False,
                   fixef_rm="none")
firms = fit.fixef("firm_id").filter(pl.col("component") == 0).collect()["firm_id"]
largest = (connected_components(data, FE, verbose=False).levels["firm_id"]
           .filter(pl.col("component") == 0)["firm_id"])
print(f"\nwithout a restriction the firm effects span {fit.n_components:,} components; the "
      f"{firms.len():,} firms\nof component 0 are the largest connected set's: "
      f"{set(firms.to_list()) == set(largest.to_list())}")
fit.cleanup()

# Connected sets

Worker and firm effects are only comparable within a connected set: firms
linked, directly or through other firms, by workers who moved between them.
[`connected_components`](../reference/connected.md) finds the connected
components of the graph of two fixed effects (either may be an interaction,
`firm_id^year`) by union-find over the rows as they stream, and returns small
tables, one per dimension, giving each level its component: not rows.
Components are numbered by size, so `component == 0` is the largest.
`.largest()` is the largest connected set and `.label()` every row with its
component, both lazily.

```python
from hdfe_stream import connected_components

cc = connected_components("data/*.parquet", "worker_id + firm_id")
cc.sizes                       # rows and levels per component
akm = cc.largest()             # the largest connected set, lazily
rows = cc.label()              # every row, with its component
```

"Largest" counts rows by default; `rank_by=` counts the levels of either
dimension instead. Rows with a missing id are in no component: a missing firm
is not one big firm that every such worker shares.

## Flagging the largest connected set

To keep every row and mark the ones in the largest set, use `.label()`: it adds
`component` (null for a row in no component) and `in_largest`. It does what you
would do by hand, a left join of the smaller dimension's level table onto the
rows, and it is lazy, so the rows stream through the join and are never held.
After `min_degree=` or `leave_one_out=True` it also checks the other dimension's
levels, which a join on one dimension alone would miss.

```python
cc = connected_components("data/*.parquet", "worker_id + firm_id")
flagged = cc.label()                  # lazy: the data plus component, in_largest
flagged.sink_parquet("flagged.parquet")
```

To drop the other rows instead, `.largest()` filters with a semi-join (or an
anti-join, whichever level list is shorter). Doing the join yourself, or testing
`is_in` against the component-0 levels, gives the same answer in the same time:
on 8.5 million rows and 3,391 components each took under 0.1 s once the
components were found, and the cost of a fit-sized pass over the data is the
reading, not the join. The time is in `connected_components` itself, so call it
once and reuse `cc` for `largest()`, `label()` and `sizes`.

## Three ways of restricting the sample

The largest connected set
: `connected_components(...).largest()`, or `connected="largest"` in a fit.

Pruning by degree
: `min_degree` first removes levels with too few distinct partners in the other
  dimension (distinct firms for a worker, distinct workers for a firm),
  repeating until none is left, then finds the components of what remains.
  `min_degree=(2, 1)` keeps movers only: workers seen at two or more firms, and
  the firms they were seen at. `min_degree=(1, 5)` keeps firms with at least
  five distinct workers. `min_degree=2` also removes firms with a single worker,
  and repeats, since removing a firm can leave a worker with one firm and
  removing a worker can leave a firm with one worker, until every worker left
  was seen at two or more firms and every firm employed two or more of them (the
  network's "2-core").

The leave-one-out connected set
: `leave_one_out=True` instead prunes to the set that
  [KSS estimation](../kss/index.md) needs (and `leave_out_kss` applies), so it
  can be found ahead of time: the largest connected set, without workers
  observed only once, and without any worker whose departure would split the
  network into pieces, removed repeatedly until none is left.

The leave-one-out connected set differs from `min_degree=2`. It keeps stayers
with at least two observations, which `min_degree=2` removes, and it removes
movers that `min_degree=2` keeps. Every mover and firm it keeps, `min_degree=2`
keeps too, and on the movers and firms the two coincide only when no single
mover holds the network together.

## On a network small enough to read

The example starts with eleven workers and seven firms. Workers 1–4 move around
firms A, B and C, tying them together several times over. Around them, worker 5
stays at A for three years; 6 moves between B and D, and is D's only worker; 7 is
seen once, at C; 8 moves between C and E and is the only link to E, where 9
stays; and 10 and 11 move between G and H, which no one else touches: a second
component. Worker 1 also has a year with no firm recorded. Each row's
component under each setting, with null for a row in none:

```text
┌───────────┬─────────┬────────────┬───────────────────┬──────────────┬───────────────┐
│ worker_id ┆ firm_id ┆ components ┆ min_degree=(2, 1) ┆ min_degree=2 ┆ leave_one_out │
╞═══════════╪═════════╪════════════╪═══════════════════╪══════════════╪═══════════════╡
│ 1         ┆ A       ┆ 0          ┆ 0                 ┆ 0            ┆ 0             │
│ 1         ┆ B       ┆ 0          ┆ 0                 ┆ 0            ┆ 0             │
│ 1         ┆ null    ┆ null       ┆ null              ┆ null         ┆ null          │
│ 2–4       ┆ A, B, C ┆ 0          ┆ 0                 ┆ 0            ┆ 0             │
│ 5         ┆ A (×3)  ┆ 0          ┆ null              ┆ null         ┆ 0             │
│ 6         ┆ B, D    ┆ 0          ┆ 0                 ┆ null         ┆ null          │
│ 7         ┆ C       ┆ 0          ┆ null              ┆ null         ┆ null          │
│ 8         ┆ C, E    ┆ 0          ┆ 0                 ┆ null         ┆ null          │
│ 9         ┆ E (×2)  ┆ 0          ┆ null              ┆ null         ┆ null          │
│ 10, 11    ┆ G, H    ┆ 1          ┆ 1                 ┆ 1            ┆ null          │
└───────────┴─────────┴────────────┴───────────────────┴──────────────┴───────────────┘
```

`min_degree=(2, 1)`
: keeps movers only: workers seen at two or more firms (5, 7 and 9 go), and the
  firms they were seen at. Nothing else needs repeating, so the island G–H
  stays, as component 1.

`min_degree=2`
: also needs two workers at every firm, and repeats: D goes (one worker), which
  leaves 6 with one firm, so 6 goes; 9 goes (one firm), which leaves E with one
  worker, so E goes, and then 8. What is left is the 2-core: workers 1–4 on A, B
  and C, and the island, still a component of its own.

`leave_one_out`
: takes the largest component first, so the island goes. It drops 7, seen only
  once (its leverage would be 1), and the workers whose departure would split
  the network: 6, the only link to D, and 8, the only link to E and 9. But it
  keeps 5: a stayer with three observations is fine for KSS, while
  `min_degree=2` removes every stayer.

## On a panel with little mobility

The same settings on 20,000 workers over ten years, half the person-years
observed and 2% moving in a year: hundreds of components, and many firms held by
a single mover. Each setting is one call, reading the data from disk in batches:

```text
setting                       comps     rows  workers  movers  firms  seconds
everything                        -  100,030   19,979   2,634  1,992     0.01
largest connected set           381   89,688   17,898   2,600  1,578     0.03
min_degree=(2, 1), largest       28   13,700    2,600   2,600  1,578     0.03
min_degree=(1, 5), largest      200   86,291   17,339   2,349  1,358     0.02
min_degree=2, largest             1   10,763    2,054   2,054    980     0.03
leave_one_out                     1   68,767   13,614   1,985    956     0.03
```

The largest connected set loses a tenth of the rows, in hundreds of small
components. Movers only keeps a few thousand workers but every firm of the
largest set, since each is reached by a mover. `min_degree=(1, 5)` keeps firms
with at least five distinct workers: the cut is on firms, and the workers who go
are those whose every firm was cut, so one round settles it. `min_degree=2`
removes every stayer, and firms reached by one mover. `leave_one_out` keeps the
stayers, which is why its rows are so many more, but loses the firms held to the
rest by a single mover.

Partners are distinct levels of the other dimension. With firm-year effects the
network is worker × `firm_id^year`, and a stayer seen in two years at one firm
has two partners, so `min_degree=(2, 1)` is no longer movers only.

## In a fit

In a fit, `min_degree=` prunes first, then `connected="largest"` keeps the
largest connected set or `connected="leave_one_out"` the leave-one-out
connected set, all on the first two fixed effects listed unless `connected_fe=`
names another pair. They apply to the rows the model can use, before singletons
are dropped, so the fit equals one on data restricted beforehand. `sample()`
labels the rows they drop `"min_degree"`, `"disconnected"` and
`"leave_one_out"`:

```python
fit = feols_stream("log_earn ~ age_squared | worker_id + firm_id + year", data,
                   workdir="scratch", connected="largest")
fit.sample().group_by("dropped_because").len().collect()
```

```text
no restriction                   n =  99,845   components: 381
                                 dropped: singleton 185
connected="largest"              n =  89,524   components: 1
                                 dropped: disconnected 10,342, singleton 164
min_degree=(2, 1), "largest"     n =  13,560   components: 1
                                 dropped: min_degree 86,146, disconnected 184, singleton 140
connected="leave_one_out"        n =  68,767   components: 1
                                 dropped: leave_one_out 31,263
```

The pair defaults to the first two fixed effects listed. With
`worker_id + year + firm_id`, that would be workers and years, the wrong graph;
`connected_fe="worker_id + firm_id"` names the right one.

The `component` column of `fixef()` for the first non-streamed dimension is
numbered the same way, by rows in the estimation sample, so without a
restriction the firms of component 0 are exactly those of the largest connected
set.

With three or more fixed effects, connectedness of one pair is necessary for
identification but not sufficient: a further dimension can make levels of the
pair redundant that no graph of the pair shows (year effects, say, when every
move between firms coincides with a change of year).

:::{dropdown} The full script: `examples/connected_sets.py`
```{literalinclude} ../../examples/connected_sets.py
:language: python
```
:::

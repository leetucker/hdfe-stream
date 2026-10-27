"""numba kernels for the bipartite worker-firm network.

The graph has one vertex per level of each of the two dimensions and one edge
per distinct match between them. That makes it level-sized, not row-sized: for a
million workers and tens of thousands of firms it is a few tens of MB of integer
arrays, so it fits comfortably even though one of the dimensions is the streamed
one.

Both traversals here are **iterative**. The recursive formulations are shorter,
but a depth-first search on a graph with a million vertices would exhaust the
interpreter's stack, and numba's is no deeper.
"""

from __future__ import annotations

import numba as nb
import numpy as np

from .kernels_base import _uf_find


@nb.njit(cache=True)
def _nb_articulation_points(indptr, indices, is_cut, alive):
    """Mark every vertex whose removal would disconnect the graph.

    Only vertices with `alive` set take part: the rest are treated as already
    removed, which is what lets pruning iterate on the surviving network.

    Tarjan's algorithm, iteratively. For each vertex `u` in the depth-first
    tree, `low[u]` is the earliest-discovered vertex reachable from `u`'s subtree
    using at most one back edge. A non-root `u` is a cut vertex when some child
    `v` cannot reach above it, `low[v] >= disc[u]`; the root is one when it has
    more than one child in the tree.

    Works on each connected component in turn, so the graph need not be
    connected.
    """
    n = len(indptr) - 1
    disc = np.full(n, -1, np.int64)
    low = np.zeros(n, np.int64)
    parent = np.full(n, -1, np.int64)
    stack_v = np.empty(n, np.int64)
    stack_ptr = np.empty(n, np.int64)
    timer = 0

    for start in range(n):
        if disc[start] != -1 or not alive[start]:
            continue
        # begin a new depth-first tree at `start`
        disc[start] = low[start] = timer
        timer += 1
        stack_v[0] = start
        stack_ptr[0] = indptr[start]
        top = 0
        root_children = 0

        while top >= 0:
            u = stack_v[top]
            if stack_ptr[top] < indptr[u + 1]:
                v = indices[stack_ptr[top]]
                stack_ptr[top] += 1
                if v == parent[u] or not alive[v]:
                    continue                # the edge we arrived by, or gone
                if disc[v] == -1:
                    parent[v] = u
                    disc[v] = low[v] = timer
                    timer += 1
                    top += 1
                    stack_v[top] = v
                    stack_ptr[top] = indptr[v]
                    if u == start:
                        root_children += 1
                elif disc[v] < low[u]:
                    low[u] = disc[v]        # back edge
            else:
                # `u` is finished: fold its low-point into its parent's
                top -= 1
                if top >= 0:
                    p = stack_v[top]
                    if low[u] < low[p]:
                        low[p] = low[u]
                    if p != start and low[u] >= disc[p]:
                        is_cut[p] = True

        if root_children > 1:
            is_cut[start] = True


@nb.njit(cache=True)
def _nb_bfs_components(indptr, indices, alive, label):
    """Label the connected components of the subgraph induced by `alive`.

    Breadth-first, with an explicit queue. Dead vertices get -1.
    """
    n = len(indptr) - 1
    queue = np.empty(n, np.int64)
    for v in range(n):
        label[v] = -1
    component = 0
    for start in range(n):
        if not alive[start] or label[start] != -1:
            continue
        label[start] = component
        queue[0] = start
        head, tail = 0, 1
        while head < tail:
            u = queue[head]
            head += 1
            for k in range(indptr[u], indptr[u + 1]):
                v = indices[k]
                if alive[v] and label[v] == -1:
                    label[v] = component
                    queue[tail] = v
                    tail += 1
        component += 1
    return component


# --------------------------------------------------------------------------
# connected components of a pair of dimensions (connected.py)
#
# Vertices are the levels of the first dimension, 0..n0-1, then those of the
# second, n0..n0+n1-1, so a union-find that always links toward the smaller
# index (as `_nb_union_pairs` does) leaves each component's root at its
# smallest level of the first dimension. Distinct (a, b) pairs are held as
# one sorted uint64 key, a * n1 + b, with a count of the rows behind each.
# --------------------------------------------------------------------------

@nb.njit(cache=True)
def _nb_count_into(labels, out):
    """out[labels[i]] += 1, skipping negative labels."""
    for i in range(len(labels)):
        if labels[i] >= 0:
            out[labels[i]] += 1


@nb.njit(cache=True)
def _nb_add_into(labels, weights, out):
    """out[labels[i]] += weights[i], skipping negative labels."""
    for i in range(len(labels)):
        if labels[i] >= 0:
            out[labels[i]] += weights[i]


@nb.njit(cache=True)
def _nb_merge_pairs(ka, ca, kb, cb):
    """Merge two sorted runs of distinct keys with their row counts into one,
    adding the counts of a key found in both. The output is counted first so
    it is allocated at its final size."""
    i = j = n = 0
    while i < len(ka) and j < len(kb):
        if ka[i] < kb[j]:
            i += 1
        elif kb[j] < ka[i]:
            j += 1
        else:
            i += 1
            j += 1
        n += 1
    n += (len(ka) - i) + (len(kb) - j)
    k = np.empty(n, ka.dtype)
    c = np.empty(n, ca.dtype)
    i = j = o = 0
    while i < len(ka) and j < len(kb):
        if ka[i] < kb[j]:
            k[o], c[o] = ka[i], ca[i]
            i += 1
        elif kb[j] < ka[i]:
            k[o], c[o] = kb[j], cb[j]
            j += 1
        else:
            k[o], c[o] = ka[i], ca[i] + cb[j]
            i += 1
            j += 1
        o += 1
    while i < len(ka):
        k[o], c[o] = ka[i], ca[i]
        i += 1
        o += 1
    while j < len(kb):
        k[o], c[o] = kb[j], cb[j]
        j += 1
        o += 1
    return k, c


@nb.njit(cache=True)
def _nb_peel(keys, n1, k0, k1, alive0, alive1):
    """Remove every level with fewer than k0 (first dimension) or k1 (second)
    distinct partners among the levels still present, repeating until none
    is left: the (k0, k1)-core of the graph of distinct pairs `keys`.
    `n1` is a uint64. Returns the number of rounds that removed something.

    All levels below the threshold go at once in each round, which removes no
    level of the core, since a level of the core keeps its partners in the
    core. Each round is one pass over the pairs; graphs of matches peel in a
    few rounds."""
    deg0 = np.zeros(len(alive0), np.int64)
    deg1 = np.zeros(len(alive1), np.int64)
    rounds = 0
    while True:
        deg0[:] = 0
        deg1[:] = 0
        for e in range(len(keys)):
            a = np.int64(keys[e] // n1)
            b = np.int64(keys[e] % n1)
            if alive0[a] and alive1[b]:
                deg0[a] += 1
                deg1[b] += 1
        changed = False
        for v in range(len(alive0)):
            if alive0[v] and deg0[v] < k0:
                alive0[v] = False
                changed = True
        for v in range(len(alive1)):
            if alive1[v] and deg1[v] < k1:
                alive1[v] = False
                changed = True
        if not changed:
            return rounds
        rounds += 1


@nb.njit(cache=True)
def _nb_union_keys(parent, keys, n1, alive0, alive1):
    """`_nb_union_pairs` over the pairs `keys` whose two levels are both
    alive; the second dimension's levels are stored at len(alive0) + b."""
    off = len(alive0)
    for e in range(len(keys)):
        a = np.int64(keys[e] // n1)
        b = np.int64(keys[e] % n1)
        if not (alive0[a] and alive1[b]):
            continue
        ra = _uf_find(parent, a)
        rb = _uf_find(parent, off + b)
        if ra != rb:
            if ra < rb:
                parent[rb] = ra
            else:
                parent[ra] = rb


@nb.njit(cache=True)
def _nb_dense_labels(parent, alive):
    """Turn a union-find `parent` into component labels 0, 1, ... in place,
    numbered in the order of their roots (their smallest vertex), with -1
    for vertices that are not `alive`. Returns the number of components.

    In one ascending pass: a root is numbered when it is reached, and every
    other vertex comes after its root, whose number is by then in parent[root]."""
    for x in range(len(parent)):
        parent[x] = _uf_find(parent, x)
    nc = 0
    for x in range(len(parent)):
        if not alive[x]:
            parent[x] = -1
        elif parent[x] == x:
            parent[x] = nc
            nc += 1
        else:
            parent[x] = parent[parent[x]]
    return nc


@nb.njit(cache=True)
def _nb_relabel(labels, new):
    """labels[i] = new[labels[i]], leaving negative labels alone."""
    for i in range(len(labels)):
        if labels[i] >= 0:
            labels[i] = new[labels[i]]


@nb.njit(cache=True)
def _nb_pair_rows(keys, counts, n1, labels, out):
    """Rows per component: each pair's count, added to its component when
    both of its levels are labeled (a pair with a removed level is removed
    with it). `labels` covers both dimensions, the second from n0 =
    len(labels) - n1."""
    off = len(labels) - np.int64(n1)
    for e in range(len(keys)):
        a = np.int64(keys[e] // n1)
        b = np.int64(keys[e] % n1)
        if labels[a] >= 0 and labels[off + b] >= 0:
            out[labels[a]] += counts[e]


@nb.njit(parallel=True, cache=True)
def _nb_hash_lookup(sorted_h, codes, starts, shift, h, out):
    """Code of each hash h[i]: its position in `sorted_h` (the levels'
    hashes, sorted), through `codes` (the level codes in that order).
    `starts` indexes `sorted_h` by the top bits of the hash (h >> shift), so
    a lookup searches a few entries instead of the whole array. Returns the
    number of hashes not found, which must be zero."""
    missing = 0
    for i in nb.prange(len(h)):
        x = h[i]
        k = x >> shift
        lo, hi = starts[k], starts[k + 1]
        while lo < hi:
            mid = (lo + hi) >> 1
            if sorted_h[mid] < x:
                lo = mid + 1
            else:
                hi = mid
        if lo < starts[k + 1] and sorted_h[lo] == x:
            out[i] = codes[lo]
        else:
            out[i] = 0
            missing += 1
    return missing


@nb.njit(cache=True)
def _nb_pairs_degree(keys, n1, n0, alive, indptr):
    """First half of a CSR adjacency, in both directions, of the distinct
    pairs `keys` whose two levels are `alive` (first dimension 0..n0-1, second
    from n0): each vertex's degree, accumulated into the n0 + n1 + 1 entries
    of `indptr`, so that indptr[-1] is the length `indices` needs."""
    indptr[:] = 0
    for e in range(len(keys)):
        a = np.int64(keys[e] // n1)
        b = n0 + np.int64(keys[e] % n1)
        if alive[a] and alive[b]:
            indptr[a + 1] += 1
            indptr[b + 1] += 1
    for v in range(len(indptr) - 1):
        indptr[v + 1] += indptr[v]


@nb.njit(cache=True)
def _nb_pairs_fill(keys, n1, n0, alive, indptr, indices):
    """Second half: the neighbours, in the order of `keys`."""
    pos = indptr[:-1].copy()
    for e in range(len(keys)):
        a = np.int64(keys[e] // n1)
        b = n0 + np.int64(keys[e] % n1)
        if alive[a] and alive[b]:
            indices[pos[a]] = b
            pos[a] += 1
            indices[pos[b]] = a
            pos[b] += 1


@nb.njit(cache=True)
def _nb_level_rows(keys, counts, n1, alive, out):
    """Rows per first-dimension level, over the pairs whose two levels are
    both `alive`."""
    n0 = len(out)
    for e in range(len(keys)):
        a = np.int64(keys[e] // n1)
        if alive[a] and alive[n0 + np.int64(keys[e] % n1)]:
            out[a] += counts[e]

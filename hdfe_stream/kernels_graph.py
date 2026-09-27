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
def _nb_components(indptr, indices, alive, label):
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


def build_bipartite(left_code, right_code, n_left, n_right):
    """CSR adjacency for the bipartite graph of distinct matches.

    Vertices are the left levels 0..n_left-1 followed by the right levels
    n_left..n_left+n_right-1. Each match contributes one undirected edge, stored
    in both directions.
    """
    n_vertices = n_left + n_right
    u = left_code.astype(np.int64)
    v = right_code.astype(np.int64) + n_left

    # list every edge twice, once from each end, then group by source
    source = np.concatenate([u, v])
    target = np.concatenate([v, u])
    order = np.argsort(source, kind="stable")

    indptr = np.zeros(n_vertices + 1, np.int64)
    np.cumsum(np.bincount(source, minlength=n_vertices), out=indptr[1:])
    return indptr, target[order]

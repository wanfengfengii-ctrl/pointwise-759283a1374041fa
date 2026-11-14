"""Isotonic regression on a rooted tree with arbitrary node weights.

The problem
-----------
Given a rooted tree (``parents[i] < i`` for every non-root ``i``),
observations ``values[i]`` and positive weights ``weights[i]``, find

    x* = argmin  sum_i weights[i] * (x[i] - values[i])**2
         s.t.   x[parents[i]] <= x[i]   for every i != 0.

This is the weighted ``L2`` isotonic regression on the tree order.  The
constraints couple a node with its parent, so the solution is obtained by
merging adjacent blocks whose optimal constant would violate the order.

Algorithm
---------
Processing nodes in reverse index order is a valid post-order because
``parents[i] < i``.  For a node ``v`` we already know, for every child ``c``,
the block containing ``c`` (its block mean) and the forest of "residual"
blocks that ``c`` accumulated but did not merge into its own block.  Those
residual blocks are exactly the roots of a heap keyed by block mean.

At ``v``:

* meld all child block-heaps plus the child blocks themselves into one
  min-heap keyed by block mean (leftist heaps, iterative meld);
* start the current block with ``W = weights[v]``, ``S = weights[v]*values[v]``,
  mean ``m = values[v]``;
* while the smallest residual block has mean ``< m``, pop it, merge it into
  the current block (``W += W_b``, ``S += S_b``, ``m = S / W``) and meld the
  popped block's own residual heap back into the working heap;
* the resulting block belongs to ``v``; its mean is the calibrated value of
  every node in the block (union-find records the membership).

Each block is pushed/popped at most once, and every meld touches a leftist
heap path of length ``O(log n)``.  Total worst-case time is ``O(n log n)``
(within the required ``O(n log^2(n+1))`` bound); space is ``O(n)``.  Nothing
is recursive, so a 50 000-node chain is safe.
"""

from __future__ import annotations

__all__ = ["solve"]


def solve(parents, values, weights):
    """Return the unique weighted isotonic tree regression.

    Parameters
    ----------
    parents : sequence of int
        ``parents[0] == -1`` and ``0 <= parents[i] < i`` for ``i >= 1``.
    values : sequence of int/float
        Observations, ``|values[i]| <= 10_000``.
    weights : sequence of int/float
        Positive weights, ``0.1 <= weights[i] <= 10``.

    Returns
    -------
    list of float
        Calibrated values in node order.  The input sequences are not
        modified.
    """
    n = len(parents)
    if n == 0:
        return []
    if n == 1:
        return [float(values[0])]

    # Work on private float copies so the caller's sequences are never touched
    # and every returned entry is a float even for integer inputs.
    val = [float(v) for v in values]
    wt = [float(w) for w in weights]

    # Children stored as forward-star (CSR-like) arrays: first_child[i] is the
    # head of i's child list, next_sibling links the rest.  Order is irrelevant.
    first_child = [-1] * n
    next_sibling = [-1] * n
    for i in range(1, n):
        p = parents[i]
        next_sibling[i] = first_child[p]
        first_child[p] = i

    # ---- disjoint-set forest (root of a block records the calibrated mean) --
    dsu = list(range(n))

    def find(a):
        root = a
        while dsu[root] != root:
            root = dsu[root]
        while dsu[a] != root:          # path compression
            nxt = dsu[a]
            dsu[a] = root
            a = nxt
        return root

    # ---- data of the block represented by each node -------------------------
    mean = [0.0] * n    # block mean (heap key once the node became a block)
    tw = [0.0] * n      # block weight sum
    ts = [0.0] * n      # block weighted-value sum
    res_root = [-1] * n # heap of residual blocks left behind by this block

    # ---- iterative leftist heap over block indices, keyed by mean -----------
    hl = [-1] * n
    hr = [-1] * n
    rank = [0] * n

    def heap_rank(x):
        return rank[x] if x >= 0 else 0

    def meld(a, b):
        """Iterative leftist-heap meld; returns the new root or -1."""
        if a < 0:
            return b
        if b < 0:
            return a
        path = []
        while a >= 0 and b >= 0:
            if mean[a] <= mean[b]:
                path.append(a)
                a = hr[a]
            else:
                path.append(b)
                b = hr[b]
        h = a if a >= 0 else b
        while path:
            u = path.pop()
            hr[u] = h
            if heap_rank(hl[u]) < heap_rank(hr[u]):
                hl[u], hr[u] = hr[u], hl[u]
            rank[u] = heap_rank(hr[u]) + 1
            h = u
        return h

    # Reverse index order is a valid bottom-up order (parents[i] < i).
    for v in range(n - 1, -1, -1):
        w = wt[v]
        s = wt[v] * val[v]
        m = val[v]
        heap = -1

        c = first_child[v]
        while c >= 0:
            # ``c`` becomes a singleton heap node.
            hl[c] = -1
            hr[c] = -1
            rank[c] = 1
            heap = meld(heap, c)
            c = next_sibling[c]

        # Merge every residual block that would violate the order.
        while heap >= 0 and mean[heap] < m:
            b = heap
            heap = meld(hl[b], hr[b])
            dsu[b] = v
            w += tw[b]
            s += ts[b]
            m = s / w
            heap = meld(heap, res_root[b])

        mean[v] = m
        tw[v] = w
        ts[v] = s
        res_root[v] = heap
        # v itself is a fresh singleton for the moment it is merged upwards.
        hl[v] = -1
        hr[v] = -1
        rank[v] = 1

    return [mean[find(i)] for i in range(n)]

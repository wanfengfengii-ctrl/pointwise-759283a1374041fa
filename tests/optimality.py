"""Independent optimality checks for tree isotonic regression.

Nothing in this module uses ``tree_calibration.solver`` internals; every helper
only consumes ``(parents, values, weights, x)``.

Two independent certificates are provided:

1. ``dual_certificate`` -- exact weak duality.  For any multipliers
   ``lam[e] >= 0`` on the tree edges the Lagrangian

       L(x, lam) = sum_i w_i (x_i - v_i)^2 + sum_e lam_e (x_parent - x_child)

   is a lower bound on the optimal cost when ``lam >= 0`` (feasible ``x`` makes
   the added term <= 0).  We build multipliers from the candidate values by
   leaf elimination over the *tight* edges and evaluate the dual lower bound.
   ``gap = primal - dual >= 0``; ``gap / max(1, primal)`` near machine
   precision proves optimality of the candidate.

2. ``chain_exact_pava`` -- exact rational PAVA for a chain-shaped tree,
   implemented with ``fractions.Fraction``.  Used to bound floating-point
   error of the solver where a brute-force enumeration is too expensive.

Also ``enumerate_exact`` -- brute-force enumeration of tight-edge subsets on
tiny trees (exact rational objective), used for exact optimality checks.
"""

from fractions import Fraction

__all__ = [
    "objective",
    "max_violation",
    "dual_certificate",
    "enumerate_exact",
    "chain_exact_pava",
]


def objective(parents, values, weights, x):
    return sum(weights[i] * (x[i] - values[i]) ** 2 for i in range(len(values)))


def max_violation(parents, x):
    """Largest ``x[parent] - x[child]`` (<= 0 means feasible)."""
    worst = 0.0
    for i in range(1, len(x)):
        d = x[parents[i]] - x[i]
        if d > worst:
            worst = d
    return worst


def dual_certificate(parents, values, weights, x):
    """Validate primal feasibility and build a dual lower bound from ``x``.

    Returns a dict with:
      * ``violation``  -- max violation of x[parent] <= x[child]
      * ``tight``      -- number of edges with exactly equal endpoint values
      * ``neg_lam``    -- how negative the raw multipliers went (rounding only)
      * ``primal``     -- objective at x
      * ``dual``       -- weak-duality lower bound with clamped multipliers
      * ``gap``        -- primal - dual
      * ``rel_gap``    -- gap / max(1, primal)
    """
    n = len(parents)
    if n == 0:
        return dict(violation=0.0, tight=0, neg_lam=0.0, primal=0.0, dual=0.0,
                    gap=0.0, rel_gap=0.0)
    w = [float(t) for t in weights]
    v = [float(t) for t in values]
    violation = max_violation(parents, x)

    # Tight edges: block members receive literally the same float, so equality
    # is bitwise.  Separate blocks whose means merely coincide get lam ~ 0.
    tight = [False] * (n - 1)
    for j in range(n - 1):
        c = j + 1
        if x[parents[c]] == x[c]:
            tight[j] = True

    # Adjacency of the tight-edge forest.
    adj = [[] for _ in range(n)]
    for j in range(n - 1):
        if tight[j]:
            c = j + 1
            p = parents[c]
            adj[p].append((c, j))
            adj[c].append((p, j))

    lam = [0.0] * (n - 1)
    # Residual of the stationarity equation, restricted to tight edges.
    resid = [2.0 * w[i] * (x[i] - v[i]) for i in range(n)]
    seen = [False] * n
    deg = [len(a) for a in adj]
    stack = [i for i in range(n) if deg[i] <= 1]
    neg_lam = 0.0
    while stack:
        u = stack.pop()
        if seen[u]:
            continue
        seen[u] = True
        for nb, j in adj[u]:
            if seen[nb]:
                continue
            if parents[u] == nb:          # u is the child on edge j
                lv = resid[u]
                resid[nb] += lv
            else:                          # nb is the child on edge j
                lv = -resid[u]
                resid[nb] -= lv
            lam[j] = lv
            if lv < 0.0:
                neg_lam = max(neg_lam, -lv)
            deg[nb] -= 1
            if deg[nb] <= 1:
                stack.append(nb)
            break

    lam_pos = [t if t > 0.0 else 0.0 for t in lam]
    # a_i = sum_{children edges} lam - lam on the parent edge.
    a = [0.0] * n
    for j in range(n - 1):
        l = lam_pos[j]
        c = j + 1
        p = parents[c]
        a[p] += l
        a[c] -= l

    dual = 0.0
    for i in range(n):
        dual += a[i] * v[i] - a[i] * a[i] / (4.0 * w[i])
    primal = objective(parents, values, weights, x)
    gap = primal - dual
    return dict(
        violation=violation,
        tight=sum(tight),
        neg_lam=neg_lam,
        primal=primal,
        dual=dual,
        gap=gap,
        rel_gap=gap / max(1.0, abs(primal)),
    )


def enumerate_exact(parents, values, weights):
    """Exact optimum for tiny trees by enumerating tight-edge subsets.

    Each feasible candidate is a contraction of a subset of edges with block
    weighted means; feasibility plus non-negative KKT multipliers and exact
    stationarity certify each candidate.  Returns ``(x, cost)`` as Fractions.
    """
    n = len(parents)
    fv = [Fraction(v) for v in values]
    fw = [Fraction(w) for w in weights]
    edges = [(parents[i], i) for i in range(1, n)]
    best_cost = None
    best_x = None
    for mask in range(1 << (n - 1)):
        par = list(range(n))

        def find(a):
            while par[a] != a:
                par[a] = par[par[a]]
                a = par[a]
            return a

        for j, (a, b) in enumerate(edges):
            if mask >> j & 1:
                ra, rb = find(a), find(b)
                if ra != rb:
                    par[ra] = rb
        comps = {}
        for i in range(n):
            comps.setdefault(find(i), []).append(i)
        x = [Fraction(0)] * n
        for nodes in comps.values():
            W = sum(fw[i] for i in nodes)
            m = sum(fw[i] * fv[i] for i in nodes) / W
            for i in nodes:
                x[i] = m
        if any(x[parents[i]] > x[i] for i in range(1, n)):
            continue

        adj = [[] for _ in range(n)]
        for j, (a, b) in enumerate(edges):
            if mask >> j & 1:
                adj[a].append((b, j))
                adj[b].append((a, j))
        resid = [2 * fw[i] * (x[i] - fv[i]) for i in range(n)]
        lam = [Fraction(0)] * (n - 1)
        seen = [False] * n
        deg = [len(a) for a in adj]
        stack = [i for i in range(n) if deg[i] <= 1]
        ok = True
        while stack:
            u = stack.pop()
            if seen[u]:
                continue
            seen[u] = True
            for nb, j in adj[u]:
                if not seen[nb]:
                    if parents[u] == nb:
                        lv = resid[u]
                        resid[nb] += lv
                    else:
                        lv = -resid[u]
                        resid[nb] -= lv
                    lam[j] = lv
                    if lv < 0:
                        ok = False
                    deg[nb] -= 1
                    if deg[nb] <= 1:
                        stack.append(nb)
                    break
        if not ok:
            continue
        stat = [2 * fw[i] * (x[i] - fv[i]) for i in range(n)]
        for j, (a, b) in enumerate(edges):
            stat[a] += lam[j]
            stat[b] -= lam[j]
        if any(t != 0 for t in stat):
            continue
        cost = sum(fw[i] * (x[i] - fv[i]) ** 2 for i in range(n))
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_x = x
    return best_x, best_cost


def chain_exact_pava(values, weights):
    """Exact rational isotonic regression for the chain 0 -> 1 -> ... -> n-1.

    Returns a list of Fractions.  Only valid when ``parents`` is a chain
    (every node's parent is its predecessor); supplied separately as an
    independent algorithm for precision tests.
    """
    fv = [Fraction(v) for v in values]
    fw = [Fraction(w) for w in weights]
    blocks = []  # [W, S, count_prefix] -- merged in order
    for i in range(len(fv)):
        W = fw[i]
        S = fw[i] * fv[i]
        while blocks and (blocks[-1][1] / blocks[-1][0]) > (S / W):
            pW, pS, _ = blocks.pop()
            W += pW
            S += pS
        blocks.append([W, S, i])
    out = []
    start = 0
    for W, S, end in blocks:
        m = S / W
        out.extend([m] * (end - start + 1))
        start = end + 1
    return out

#!/usr/bin/env python3
"""Independent verification of tree_calibration.solve.

Checks (all independent of the solver's internal data structures):

1. The published example.
2. Random tiny trees against exact rational enumeration of tight-edge
   subsets (``tests.optimality.enumerate_exact``).
3. Chains against exact rational PAVA (``tests.optimality.chain_exact_pava``).
4. Large adversarial shapes: primal feasibility plus an exact weak-duality
   certificate (``tests.optimality.dual_certificate``) -- a feasible point
   whose objective equals a valid dual lower bound is provably optimal.
5. Input immutability.

Exit status is 0 only when every check passes.
"""

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tree_calibration import solve
from tests.optimality import (
    chain_exact_pava,
    dual_certificate,
    enumerate_exact,
    max_violation,
    objective,
)


def report(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    return ok


def shape(name, n, rng):
    if name == "deep_chain":
        return [-1] + list(range(n - 1)), list(range(n, 0, -1)), [1.0] * n
    if name == "deep_chain_alt":
        return (
            [-1] + list(range(n - 1)),
            [(-10000.0 if i % 2 else 10000.0) for i in range(n)],
            [1.0] * n,
        )
    if name == "star":
        return (
            [-1] + [0] * (n - 1),
            [rng.uniform(-10000, 10000) for _ in range(n)],
            [rng.uniform(0.1, 10) for _ in range(n)],
        )
    if name == "multi_level":
        return (
            [-1] + [(i - 1) // 3 for i in range(1, n)],
            [rng.uniform(-10000, 10000) for _ in range(n)],
            [rng.uniform(0.1, 10) for _ in range(n)],
        )
    if name == "broom":
        k = n // 2
        return (
            [-1] + list(range(k - 1)) + [k - 1] * (n - k),
            [rng.uniform(-10000, 10000) for _ in range(n)],
            [rng.uniform(0.1, 10) for _ in range(n)],
        )
    if name == "random":
        return (
            [-1] + [rng.randrange(i) for i in range(1, n)],
            [rng.uniform(-10000, 10000) for _ in range(n)],
            [rng.uniform(0.1, 10) for _ in range(n)],
        )
    raise ValueError(name)


def main():
    ok = True

    # 1. Published example.
    x = solve([-1, 0, 0], [6, 0, 9], [1, 1, 1])
    ok &= report(
        "example [-1,0,0]/[6,0,9]/[1,1,1] -> [3,3,9]",
        all(abs(a - b) <= 1e-12 for a, b in zip(x, [3.0, 3.0, 9.0])),
        repr(x),
    )

    # 2. Tiny trees versus exact rational enumeration.
    rng = random.Random(1234567)
    worst_err = 0.0
    worst_rel = 0.0
    trials = 400
    for _ in range(trials):
        n = rng.randint(1, 8)
        p = [-1] + [rng.randrange(i) for i in range(1, n)]
        v = [
            rng.choice(
                [
                    rng.randint(-10000, 10000),
                    rng.randint(-5, 5) / 7.0,
                    rng.randint(-1, 1),
                ]
            )
            for _ in range(n)
        ]
        w = [
            rng.choice([rng.randint(1, 100) / 10.0, 0.1, 10.0, 1.0])
            for _ in range(n)
        ]
        got = solve(p, v, w)
        ex, ex_cost = enumerate_exact(p, v, w)
        scale = max(1.0, max(abs(t) for t in v))
        err = max(abs(a - float(b)) for a, b in zip(got, ex)) / scale
        worst_err = max(worst_err, err)
        gap = objective(p, v, w, got) - float(ex_cost)
        worst_rel = max(worst_rel, max(0.0, gap) / max(1.0, float(ex_cost)))
    ok &= report(
        f"exact enumeration on {trials} random trees (n<=8)",
        worst_err <= 1e-7 and worst_rel <= 1e-12,
        f"max pointwise err / scale = {worst_err:.3e}, max objective excess = {worst_rel:.3e}",
    )

    # 3. Chains versus exact rational PAVA.
    rng = random.Random(99)
    worst_chain = 0.0
    for _ in range(30):
        n = rng.randint(1, 300)
        p = [-1] + list(range(n - 1))
        v = [rng.choice([rng.randint(-10000, 10000), rng.randint(-2, 2)]) for _ in range(n)]
        w = [rng.choice([0.1, 1.0, 10.0, rng.randint(1, 100) / 10.0]) for _ in range(n)]
        got = solve(p, v, w)
        ex = chain_exact_pava(v, w)
        scale = max(1.0, max(abs(t) for t in v))
        worst_chain = max(worst_chain, max(abs(a - float(b)) for a, b in zip(got, ex)) / scale)
    ok &= report(
        "exact rational PAVA on 30 chains (n<=300)",
        worst_chain <= 1e-9,
        f"max pointwise err / scale = {worst_chain:.3e}",
    )

    # 4. Large adversarial shapes with dual certificates.
    rng = random.Random(2026)
    for name in [
        "deep_chain",
        "deep_chain_alt",
        "star",
        "multi_level",
        "broom",
        "random",
    ]:
        p, v, w = shape(name, 50000, rng)
        p_copy, v_copy, w_copy = list(p), list(v), list(w)
        x = solve(p, v, w)
        cert = dual_certificate(p, v, w, x)
        immutable = p == p_copy and v == v_copy and w == w_copy
        good = (
            cert["violation"] == 0.0
            and cert["rel_gap"] <= 1e-9
            and max_violation(p, x) == 0.0
            and immutable
            and len(x) == len(p)
        )
        ok &= report(
            f"{name} n=50000 dual certificate",
            good,
            "violation={:.1e}, rel_gap={:.2e}, tight={}, feasible={}".format(
                cert["violation"], cert["rel_gap"], cert["tight"], immutable
            ),
        )

    # 5. Sanity: a feasible constant can never beat the returned solution.
    rng = random.Random(11)
    beats = 0
    for _ in range(50):
        n = rng.randint(2, 200)
        p = [-1] + [rng.randrange(i) for i in range(1, n)]
        v = [rng.uniform(-10000, 10000) for _ in range(n)]
        w = [rng.uniform(0.1, 10) for _ in range(n)]
        x = solve(p, v, w)
        base = objective(p, v, w, x)
        for c in (max(v), min(v), 0.0):
            if base > objective(p, v, w, [c] * n) + 1e-6 * max(1.0, base):
                beats += 1
    ok &= report("returned cost never exceeds feasible constants", beats == 0)

    print()
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

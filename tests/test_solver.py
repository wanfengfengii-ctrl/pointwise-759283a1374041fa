"""Tests for tree_calibration.solve.

Optimality is checked independently of the implementation:

* brute-force enumeration of tight-edge subsets with exact rationals on tiny
  trees (``tests.optimality.enumerate_exact``),
* exact rational PAVA on chains (``tests.optimality.chain_exact_pava``),
* exact weak-duality certificates on large random trees
  (``tests.optimality.dual_certificate``).
"""

import random
import unittest

from tree_calibration import solve
from tests.optimality import (
    chain_exact_pava,
    dual_certificate,
    enumerate_exact,
    max_violation,
    objective,
)


def feasible(parents, x):
    return max_violation(parents, x) <= 0.0


class TestBasic(unittest.TestCase):
    def test_example(self):
        got = solve([-1, 0, 0], [6, 0, 9], [1, 1, 1])
        self.assertEqual(len(got), 3)
        for a, b in zip(got, [3.0, 3.0, 9.0]):
            self.assertAlmostEqual(a, b, places=12)
        self.assertTrue(feasible([-1, 0, 0], got))

    def test_single(self):
        self.assertEqual(solve([-1], [7], [2]), [7.0])
        self.assertEqual(solve([-1], [-3.5], [0.1]), [-3.5])

    def test_chain_already_sorted(self):
        p = [-1, 0, 1, 2]
        x = solve(p, [1, 2, 3, 4], [1, 1, 1, 1])
        self.assertEqual(x, [1.0, 2.0, 3.0, 4.0])

    def test_chain_decreasing_pooled(self):
        n = 2000
        p = [-1] + list(range(n - 1))
        x = solve(p, list(range(n, 0, -1)), [1.0] * n)
        for i in range(n):
            self.assertAlmostEqual(x[i], 1000.5, places=9)

    def test_inputs_unmodified(self):
        p = [-1, 0, 0]
        v = [6, 0, 9]
        w = [1, 1, 1]
        solve(p, v, w)
        self.assertEqual(p, [-1, 0, 0])
        self.assertEqual(v, [6, 0, 9])
        self.assertEqual(w, [1, 1, 1])

    def test_accepts_tuples_and_ints(self):
        got = solve((-1, 0), (1, 0), (1, 3))
        self.assertAlmostEqual(got[0], 0.25)
        self.assertAlmostEqual(got[0], got[1], places=15)

    def test_star(self):
        p = [-1] + [0] * 4
        v = [10, 1, 2, 3, 4]
        w = [1, 1, 1, 1, 1]
        x = solve(p, v, w)
        self.assertTrue(feasible(p, x))
        ex, ex_cost = enumerate_exact(p, v, w)
        for a, b in zip(x, ex):
            self.assertAlmostEqual(a, float(b), places=12)
        self.assertLessEqual(objective(p, v, w, x), float(ex_cost) + 1e-9)

    def test_all_weights_extreme(self):
        p = [-1, 0, 1]
        x = solve(p, [0, 100, -100], [0.1, 0.1, 10.0])
        self.assertTrue(feasible(p, x))
        pooled = (0.1 * 0 + 0.1 * 100 + 10.0 * -100) / 10.2
        for xi in x:
            self.assertAlmostEqual(xi, pooled, places=9)

    def test_negative_values(self):
        p = [-1, 0, 0]
        x = solve(p, [-6, -10, -2], [1, 1, 1])
        self.assertTrue(feasible(p, x))
        self.assertAlmostEqual(x[0], -8.0, places=12)
        self.assertAlmostEqual(x[1], -8.0, places=12)
        self.assertAlmostEqual(x[2], -2.0, places=12)


class TestAgainstExact(unittest.TestCase):
    def test_random_small_rational(self):
        rng = random.Random(20241114)
        for trial in range(400):
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
            err = max(abs(a - float(b)) for a, b in zip(got, ex))
            self.assertLessEqual(
                err, 1e-7 * scale, msg=f"trial {trial}: {p} {v} {w} -> {got} vs {ex}"
            )
            self.assertLessEqual(
                objective(p, v, w, got),
                float(ex_cost) + 1e-9 * max(1.0, float(ex_cost)),
            )

    def test_chain_against_exact_pava(self):
        rng = random.Random(5)
        for trial in range(30):
            n = rng.randint(1, 200)
            p = [-1] + list(range(n - 1))
            v = [rng.choice([rng.randint(-10000, 10000), rng.randint(-3, 3)]) for _ in range(n)]
            w = [rng.choice([0.1, 1.0, 10.0, rng.randint(1, 100) / 10.0]) for _ in range(n)]
            got = solve(p, v, w)
            ex = chain_exact_pava(v, w)
            err = max(abs(a - float(b)) for a, b in zip(got, ex))
            self.assertLessEqual(err, 1e-9 * max(1.0, max(abs(t) for t in v)))


class TestCertificates(unittest.TestCase):
    def test_dual_certificate_random_trees(self):
        rng = random.Random(17)
        for trial in range(40):
            n = rng.randint(2, 500)
            p = [-1] + [rng.randrange(i) for i in range(1, n)]
            v = [rng.uniform(-10000, 10000) for _ in range(n)]
            w = [rng.uniform(0.1, 10.0) for _ in range(n)]
            x = solve(p, v, w)
            cert = dual_certificate(p, v, w, x)
            self.assertEqual(cert["violation"], 0.0, msg=f"trial {trial}")
            self.assertLessEqual(cert["rel_gap"], 1e-12, msg=f"trial {trial}")

    def test_dual_certificate_adversarial(self):
        n = 20000
        cases = {
            "chain_dec": ([-1] + list(range(n - 1)), list(range(n, 0, -1)), [1.0] * n),
            "chain_flat": ([-1] + list(range(n - 1)), [5.0] * n, [1.0] * n),
            "star": ([-1] + [0] * (n - 1), [(i * 37) % 401 - 200 for i in range(n)], [1.0] * n),
            "binary": ([-1] + [(i - 1) // 2 for i in range(1, n)],
                       [(i * 131) % 2001 - 1000 for i in range(n)],
                       [0.5 + (i % 19) / 2 for i in range(n)]),
        }
        for name, (p, v, w) in cases.items():
            x = solve(p, v, w)
            cert = dual_certificate(p, v, w, x)
            self.assertEqual(cert["violation"], 0.0, msg=name)
            self.assertLessEqual(cert["rel_gap"], 1e-11, msg=name)

    def test_cost_never_above_feasible_alternative(self):
        rng = random.Random(23)
        for trial in range(20):
            n = rng.randint(2, 300)
            p = [-1] + [rng.randrange(i) for i in range(1, n)]
            v = [rng.uniform(-10000, 10000) for _ in range(n)]
            w = [rng.uniform(0.1, 10.0) for _ in range(n)]
            x = solve(p, v, w)
            best = objective(p, v, w, x)
            # A few feasible constructions must not beat the optimum.
            # Constants always satisfy the tree order.
            candidates = [
                [max(v) for _ in range(n)],
                [min(v) for _ in range(n)],
                [0.0 for _ in range(n)],
            ]
            for cand in candidates:
                self.assertLessEqual(best, objective(p, v, w, cand) + 1e-6 * max(1.0, best))


class TestLargeShapes(unittest.TestCase):
    def test_deep_chain_50000(self):
        n = 50000
        p = [-1] + list(range(n - 1))
        v = [n - i for i in range(n)]
        w = [0.1 + (i % 10) / 10.0 for i in range(n)]
        x = solve(p, v, w)
        self.assertEqual(len(x), n)
        self.assertEqual(max_violation(p, x), 0.0)
        self.assertAlmostEqual(x[0], x[-1], places=6)

    def test_deep_chain_alternating(self):
        n = 50000
        p = [-1] + list(range(n - 1))
        v = [10000 if i % 2 else -10000 for i in range(n)]
        w = [1.0] * n
        x = solve(p, v, w)
        self.assertEqual(max_violation(p, x), 0.0)
        ex = chain_exact_pava(v, w)
        for a, b in zip(x, ex):
            self.assertAlmostEqual(a, float(b), places=9)

    def test_wide_star_50000(self):
        n = 50000
        p = [-1] + [0] * (n - 1)
        v = [((i * 97) % 20001) - 10000 for i in range(n)]
        x = solve(p, v, [1.0] * n)
        self.assertEqual(max_violation(p, x), 0.0)
        # Root must be at most every child; all nodes are individual blocks
        # except root pooled with the minimum child.
        self.assertLessEqual(x[0], min(x[1:]) + 1e-12)

    def test_multilevel_branching_50000(self):
        n = 50000
        rng = random.Random(7)
        p = [-1] + [(i - 1) // 5 for i in range(1, n)]
        v = [rng.uniform(-10000, 10000) for _ in range(n)]
        w = [rng.uniform(0.1, 10.0) for _ in range(n)]
        x = solve(p, v, w)
        self.assertEqual(len(x), n)
        self.assertEqual(max_violation(p, x), 0.0)
        self.assertTrue(all(-10000 - 1e-9 <= xi <= 10000 + 1e-9 for xi in x))


if __name__ == "__main__":
    unittest.main()

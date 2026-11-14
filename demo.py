#!/usr/bin/env python3
"""Demo: show tree_calibration.solve on the published example and a small tree.

Prints the example from the task statement plus one multi-level illustration.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tree_calibration import solve


def show(parents, values, weights, label):
    print(label)
    print(f"  parents = {parents}")
    print(f"  values  = {values}")
    print(f"  weights = {weights}")
    x = solve(parents, values, weights)
    print(f"  solve   = {x}")
    print(f"  order holds: {all(x[parents[i]] <= x[i] for i in range(1, len(x)))}")
    print()
    return x


def main():
    show([-1, 0, 0], [6, 0, 9], [1, 1, 1], "Published example")

    # A three-level tree where the root is pulled down by its children.
    #        0 (v=10)
    #       / \
    #     1(9) 2(2)
    #     / \
    #  3(1) 4(8)
    show(
        [-1, 0, 0, 1, 1],
        [10, 9, 2, 1, 8],
        [1, 1, 1, 1, 1],
        "Three-level branching tree",
    )

    # Weighted case: the heavy child dominates the root.
    show(
        [-1, 0, 0],
        [5, 0, 6],
        [0.1, 10.0, 1.0],
        "Weighted star (heavy child pulls the root up)",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

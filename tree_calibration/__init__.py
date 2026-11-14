"""Weighted L2 isotonic regression on a rooted tree.

Public API: ``tree_calibration.solve(parents, values, weights)``.
"""

from .solver import solve

__all__ = ["solve"]

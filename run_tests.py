#!/usr/bin/env python3
"""Run the unit-test suite from any working directory."""

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def main():
    suite = unittest.defaultTestLoader.discover(
        os.path.join(HERE, "tests"), top_level_dir=HERE
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Benchmark: doubling-size timing and peak memory for tree_calibration.solve.

Every measurement runs in a fresh ``python -I`` subprocess, so timing and
peak-RSS figures are isolated per data point.  Two subprocess modes are used
per data point:

* ``timing``  -- median and minimum wall time over repeated calls,
* ``memory``  -- one call under tracemalloc, with peak RSS before/after.

Shapes: deep chain (a 50 000-deep linked list), star (one root with 49 999
leaves), and a multi-level branching tree (3-way fork at every level).  Sizes
double from 1 563 to 50 000 nodes.

Output: a table on stdout plus ``bench_results.json`` next to this script.
Run with ``python3 -I bench.py [--json path]``.
"""

import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WORKER = os.path.join(HERE, "bench_worker.py")

SIZES = [1563, 3125, 6250, 12500, 25000, 50000]
SHAPES = ["deep_chain", "star", "multi_level"]
SEED = 42


def run_worker(shape, n, mode, repeats):
    out = subprocess.run(
        [sys.executable, "-I", WORKER, shape, str(n), str(SEED), mode, str(repeats)],
        capture_output=True,
        text=True,
        check=True,
    )
    lines = [ln for ln in out.stdout.splitlines() if ln.startswith("{")]
    if not lines:
        raise RuntimeError(f"worker failed:\n{out.stdout}\n{out.stderr}")
    return json.loads(lines[-1])


def main():
    rows = []
    for shape in SHAPES:
        print(f"\n== {shape} ==")
        header = (
            f"{'n':>7} {'time_med_s':>11} {'time_min_s':>11} {'ratio':>7} "
            f"{'py_peak_MB':>10} {'RSS_base_MB':>11} {'RSS_peak_MB':>11} "
            f"{'RSS_growth_MB':>13} {'OK':>4}"
        )
        print(header)
        print("-" * len(header))
        prev = None
        for n in SIZES:
            repeats = 15 if n <= 12500 else 9
            timing = run_worker(shape, n, "timing", repeats)
            mem = run_worker(shape, n, "memory", 1)
            rss = run_worker(shape, n, "rss", 1)
            ratio = "-" if prev is None else f"{timing['time_s'] / prev:.2f}x"
            print(
                f"{n:>7} {timing['time_s']:>11.6f} {timing['time_min_s']:>11.6f} "
                f"{ratio:>7} {mem['py_peak_mb']:>10.3f} {rss['rss_base_mb']:>11.3f} "
                f"{rss['rss_peak_mb']:>11.3f} {rss['rss_delta_mb']:>13.3f} "
                f"{str(rss['feasible'] and mem['feasible']):>4}"
            )
            rows.append(
                {
                    "shape": shape,
                    "n": n,
                    "seed": SEED,
                    "repeats": repeats,
                    "time_median_s": timing["time_s"],
                    "time_min_s": timing["time_min_s"],
                    "time_max_s": timing["time_max_s"],
                    "py_peak_mb": mem["py_peak_mb"],
                    "rss_base_mb": rss["rss_base_mb"],
                    "rss_peak_mb": rss["rss_peak_mb"],
                    "rss_growth_mb": rss["rss_delta_mb"],
                    "feasible": bool(rss["feasible"] and mem["feasible"]),
                }
            )
            prev = timing["time_s"]
        print("ratio = median(n) / median(n/2); O(n log n) predicts ~2.0-2.3x.")

    path = os.path.join(HERE, "bench_results.json")
    if "--json" in sys.argv:
        path = sys.argv[sys.argv.index("--json") + 1]
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

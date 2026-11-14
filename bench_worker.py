#!/usr/bin/env python3
"""Single shape/size benchmark worker; prints one JSON line.

Usage: bench_worker.py SHAPE N SEED MODE REPEATS

MODE is ``timing`` (median wall time over ``repeats`` calls), ``memory``
(one call under tracemalloc plus peak-RSS accounting) or ``rss`` (one call,
peak RSS only, no tracemalloc overhead).

``bench.py`` launches this with ``python -I`` (isolated mode), so the script
directory is not implicitly importable; the workspace root is injected below.
The worker also confirms the returned solution satisfies the tree order.
"""

import gc
import json
import os
import random
import resource
import statistics
import sys
import time
import tracemalloc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tree_calibration import solve


def make_inputs(shape, n, seed):
    rng = random.Random(seed)
    if shape == "deep_chain":
        parents = [-1] + list(range(n - 1))
        values = [n - i for i in range(n)]
        weights = [0.1 + (i % 10) / 10.0 for i in range(n)]
    elif shape == "star":
        parents = [-1] + [0] * (n - 1)
        values = [rng.uniform(-10000, 10000) for _ in range(n)]
        weights = [rng.uniform(0.1, 10) for _ in range(n)]
    elif shape == "multi_level":
        parents = [-1] + [(i - 1) // 3 for i in range(1, n)]
        values = [rng.uniform(-10000, 10000) for _ in range(n)]
        weights = [rng.uniform(0.1, 10) for _ in range(n)]
    else:
        raise ValueError(shape)
    return parents, values, weights


def peak_rss_kib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def main():
    shape = sys.argv[1]
    n = int(sys.argv[2])
    seed = int(sys.argv[3])
    mode = sys.argv[4]
    repeats = max(1, int(sys.argv[5]))
    parents, values, weights = make_inputs(shape, n, seed)

    result = {"shape": shape, "n": n, "seed": seed, "mode": mode}

    if mode == "timing":
        times = []
        gc.collect()
        gc.disable()
        try:
            for _ in range(repeats):
                t0 = time.perf_counter()
                solve(parents, values, weights)
                times.append(time.perf_counter() - t0)
        finally:
            gc.enable()
        result.update(
            repeats=repeats,
            time_s=statistics.median(times),
            time_min_s=min(times),
            time_max_s=max(times),
        )
    elif mode == "memory":
        gc.collect()
        rss_base = peak_rss_kib()
        tracemalloc.start()
        t0 = time.perf_counter()
        x = solve(parents, values, weights)
        result["traced_time_s"] = time.perf_counter() - t0
        _, py_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        rss_peak = peak_rss_kib()
        result.update(
            py_peak_mb=py_peak / 1e6,
            rss_base_mb=rss_base / 1024.0,
            rss_peak_mb=rss_peak / 1024.0,
            rss_delta_mb=(rss_peak - rss_base) / 1024.0,
            feasible=bool(all(x[parents[i]] <= x[i] for i in range(1, n))),
        )
    elif mode == "rss":
        # Clean peak-RSS measurement: one solve, no tracemalloc bookkeeping.
        gc.collect()
        rss_base = peak_rss_kib()
        x = solve(parents, values, weights)
        rss_peak = peak_rss_kib()
        result.update(
            rss_base_mb=rss_base / 1024.0,
            rss_peak_mb=rss_peak / 1024.0,
            rss_delta_mb=(rss_peak - rss_base) / 1024.0,
            feasible=bool(all(x[parents[i]] <= x[i] for i in range(1, n))),
        )
    else:
        raise ValueError(mode)

    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())

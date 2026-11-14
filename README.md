# tree_calibration — weighted L2 isotonic regression on a rooted tree

`tree_calibration.solve(parents, values, weights)` returns the unique
minimiser of

```
sum_i weights[i] * (x[i] - values[i])^2      subject to   x[parents[i]] <= x[i]
```

for the tree order defined by `parents` (`parents[0] == -1`,
`0 <= parents[i] < i`). It returns a `list` of `float` in node order and does
not modify any input. Python standard library only — no third-party packages.

The example from the task holds:

```
>>> solve([-1, 0, 0], [6, 0, 9], [1, 1, 1])
[3.0, 3.0, 9.0]
```

## Status

| item | state |
|---|---|
| API `tree_calibration.solve` | implemented, pure stdlib |
| uniqueness / feasibility | returned point is feasible and optimal (dual-certificate gap ≤ ~1e-14 relative) |
| accuracy versus exact rational optimum | worst pointwise error 2.3e-16 × `max(1, max abs(values))` over 1000+ exhaustive small-tree trials (requirement: 1e-7) |
| worst-case time | `O(n log n)` — inside the required `O(n log^2(n+1))` |
| space | `O(n)` |
| 50 000-node deep chain | completes in ~33 ms, no recursion (stack-safe) |
| repeated subtree rescans | none — each block is merged/pushed/popped once |
| inputs | sequences are read-only; integer inputs are converted to a private float copy |
| runtime config | `.pointwise-runtime.json` (version 1, kind `cli`) wired to `verify.py` and `demo.py` |

## Algorithm and complexity

Block-merge (PAVA-style) isotonic regression generalised to a tree, with the
blocks held in meldable leftist heaps:

1. Reverse index order is a valid post-order because `parents[i] < i`.
2. For a node `v` all child blocks are known. Their values are melded into
   one min-heap keyed by block mean (plus, for each child, the residual heap
   of blocks that child left unmerged).
3. The current block starts at `weights[v], values[v]`; while the heap's
   minimum mean is smaller than the current mean, the block is popped, pooled
   into the current block (`W += W_b`, `S += S_b`, mean `= S / W`), and its
   own residual heap is melded back in.
4. The final block belongs to `v`; a union-find structure maps every node to
   its block's mean, which is its calibrated value.

Each of the `n` blocks is inserted and removed at most once, and each meld
walks a leftist-heap right path of length `O(log n)`; hence `O(n log n)` time
worst case (`O(n log^2(n+1))` was required) and `O(n)` space. The meld is
iterative, and no step rescans a subtree, so a 50 000-deep chain is fine.

### Why the result is the unique optimum

The objective is strictly convex (all weights positive) on the closed convex
polyhedron defined by the order constraints, so the minimiser is unique.
Correctness is verified independently in `verify.py`:

* **Exact enumeration** (`tests/optimality.py::enumerate_exact`): for tiny
  trees every subset of "tight" edges is contracted to blocks with exact
  rational (`fractions.Fraction`) weighted means; each feasible candidate is
  screened by exact KKT stationarity and non-negative multipliers. The solver
  matches the optimal candidate to machine precision, and its objective is
  never above the exact optimum.
* **Exact chain PAVA** (`chain_exact_pava`): a second, independent rational
  algorithm for chain-shaped trees, used as a precision reference at n ≤ 300
  (and a 50 000-node alternating ±10000 chain).
* **Exact weak duality** (`dual_certificate`): multipliers derived by leaf
  elimination over the tight-edge forest give a Lagrangian lower bound
  `L(x, λ) ≤ OPT` for `λ ≥ 0`. The reported `rel_gap = primal − dual` is
  ≤ ~1e-14 on all tested 50 000-node shapes, which *proves* optimality of the
  returned point; primal feasibility is checked exactly (`x[parent] <= x[child]`).

## Layout

| path | purpose |
|---|---|
| `tree_calibration/solver.py` | the solver (the only algorithm implementation) |
| `tree_calibration/__init__.py` | re-exports `solve` |
| `verify.py` | independent optimality verification, CLI entry for `verify` |
| `demo.py` | example demo, CLI entry for `demo` |
| `tests/optimality.py` | exact enumeration, exact chain PAVA, dual certificate |
| `tests/test_solver.py` | unit tests (`python3 run_tests.py`) |
| `run_tests.py` | runs the suite from any cwd |
| `bench.py`, `bench_worker.py` | doubling benchmark for timing and peak memory |
| `bench_results.json` | latest measured numbers |
| `.pointwise-runtime.json` | runtime wiring (version 1, kind `cli`) |

## Reproduce

```bash
python3 verify.py      # independent optimality checks + 50k dual certificates
python3 demo.py        # published example and two more illustrations
python3 run_tests.py   # 18 unit tests
python3 -I bench.py    # doubling timing / memory table (rewrites bench_results.json)
```

All scripts are runnable from any working directory (they insert their own
directory into `sys.path` when launched as a script).

## Measured performance

Environment: Python 3.11.2, Linux aarch64 (10 vCPU) in a container. Every
data point ran in a fresh `python -I` subprocess; timing is the median of 15
calls (9 for n ≥ 25 000), and peak RSS is measured with `ru_maxrss` around a
single call (baseline ≈ 10 MB interpreter + inputs). `py peak` is
`tracemalloc`'s Python-allocation peak for that call (excluding C-level
storage of lists, hence smaller than RSS growth for large n).

Deep chain = a 50 000-deep parent chain; star = one root, 49 999 leaves;
multi-level = 3-way fork at every level.

### `deep_chain`

| n | median | min | ratio | py peak (MB) | RSS peak (MB) | RSS growth (MB) |
|---:|---:|---:|---:|---:|---:|---:|
| 1563 | 1.00 ms | 0.95 ms | - | 0.39 | 10.46 | 0.25 |
| 3125 | 1.91 ms | 1.73 ms | 1.92x | 0.79 | 11.08 | 0.75 |
| 6250 | 3.99 ms | 3.72 ms | 2.09x | 1.59 | 12.32 | 1.62 |
| 12500 | 8.01 ms | 7.46 ms | 2.01x | 3.20 | 14.84 | 3.38 |
| 25000 | 19.47 ms | 15.80 ms | 2.43x | 6.43 | 19.91 | 7.00 |
| 50000 | 32.58 ms | 31.85 ms | 1.67x | 12.88 | 29.87 | 13.75 |

### `star`

| n | median | min | ratio | py peak (MB) | RSS peak (MB) | RSS growth (MB) |
|---:|---:|---:|---:|---:|---:|---:|
| 1563 | 2.25 ms | 2.15 ms | - | 0.29 | 10.34 | 0.25 |
| 3125 | 5.41 ms | 5.18 ms | 2.40x | 0.59 | 10.71 | 0.50 |
| 6250 | 11.09 ms | 10.85 ms | 2.05x | 1.19 | 11.82 | 1.40 |
| 12500 | 25.17 ms | 24.49 ms | 2.27x | 2.40 | 13.33 | 2.25 |
| 25000 | 50.65 ms | 49.75 ms | 2.01x | 4.84 | 16.89 | 4.75 |
| 50000 | 110.33 ms | 108.79 ms | 2.18x | 9.71 | 23.84 | 9.62 |

### `multi_level`

| n | median | min | ratio | py peak (MB) | RSS peak (MB) | RSS growth (MB) |
|---:|---:|---:|---:|---:|---:|---:|
| 1563 | 1.37 ms | 1.28 ms | - | 0.29 | 10.34 | 0.25 |
| 3125 | 2.71 ms | 2.46 ms | 1.97x | 0.60 | 10.84 | 0.50 |
| 6250 | 5.61 ms | 5.26 ms | 2.07x | 1.21 | 11.84 | 1.25 |
| 12500 | 11.67 ms | 10.34 ms | 2.08x | 2.44 | 13.84 | 2.50 |
| 25000 | 25.57 ms | 23.73 ms | 2.19x | 4.91 | 17.96 | 5.00 |
| 50000 | 51.83 ms | 49.10 ms | 2.03x | 9.86 | 25.84 | 10.12 |

Timing roughly doubles per doubling of `n` (1.67x–2.43x, measurement noise on
the shared container), and memory grows linearly (~13.7 MB RSS growth for the
50 000-node chain, ~9.6 MB for the star). Additional adversarial 50 000-node
shapes (caterpillar, caterpillar-of-chains root, sawtooth binary tree, runs
with drops, extreme weight patterns) all finish in ≤ 0.07 s and return a
feasible, certificate-optimal result.

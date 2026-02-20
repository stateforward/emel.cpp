# Benchmark snapshot policy

This project requires a benchmark snapshot for new state machines once they are marked complete.
The benchmark gate exists to prevent performance regressions and to force explicit, repeatable
performance characterization for each machine.

## Summary

- Every **new** `src/emel/**/sm.hpp` must include a benchmark readiness marker.
- Markers live in the `sm.hpp` file header comment:
  - `// benchmark: scaffold` means no benchmark required yet.
  - `// benchmark: ready` means a benchmark **is required** and will be enforced by gates.
- Benchmarks are only required for machines marked `ready`.
- Benchmarks are enforced via a snapshot diff with a **5% regression tolerance**.
- Reference comparison is pinned to a specific upstream commit in
  `tools/bench/reference_ref.txt` for reproducibility.

## Benchmark harness

We use a single benchmark harness that lives in `tools/bench/bench_main.cpp` and
links EMEL with the reference allocator implementation for apples-to-apples
comparisons.
- Output format is deterministic and diff-friendly:
  - `machine_path ns_per_op=123.4 iter=100000 runs=5`
  - `machine_path` matches the `src/emel/<path>/sm.hpp` path without the prefix and suffix.
- Optional warmup controls (shared by EMEL + reference benches):
  - `EMEL_BENCH_WARMUP_ITERS` (default: min(`EMEL_BENCH_ITERS`, 1000))
  - `EMEL_BENCH_WARMUP_RUNS` (default: 1)
- Toolchain defaults to Zig (`zig cc` / `zig c++`) for determinism. Use
  `scripts/bench.sh --system` if you explicitly want system compilers.
- Reference commit can be overridden via `BENCH_REF_OVERRIDE` if needed.

## External reference compare (optional)

We also maintain an optional comparison mode that runs EMEL and reference benchmarks
in the same tool build.

To add a new comparison case:
1. Add the EMEL case in `tools/bench/bench_main.cpp`.
2. Add the matching reference case in `tools/bench/bench_main.cpp`.
3. Keep the case names identical for proper matching.

The compare mode expects exact case-name matches and prints `ratio` values for quick inspection.
Run it via `scripts/bench.sh --compare`.
When updating the comparison snapshot used by documentation, run:
`scripts/bench.sh --compare-update`.

## Gate behavior

The benchmark gate script enforces the following:
- If a new `sm.hpp` file is added:
  - It must declare `// benchmark: scaffold` or `// benchmark: ready`.
  - If `ready`, a benchmark case **must** exist and be registered.
- Snapshot regression failure if `ns_per_op` exceeds baseline by > 5%.
- Snapshot baseline can be updated intentionally with `--update`.

## Developer workflow

1. Add a new machine `sm.hpp` with a benchmark marker.
2. If the machine is complete, set `// benchmark: ready` and add a benchmark case.
3. Run `scripts/bench.sh --snapshot` locally to verify, or
   `scripts/bench.sh --snapshot --update` to update the baseline.

## Rationale

Scaffolded machines often produce trivial workloads that distort baselines. The readiness marker
allows incremental scaffolding without blocking CI, while still forcing benchmarks once behavior
is complete.

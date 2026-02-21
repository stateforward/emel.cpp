# benchmark snapshot policy

this project requires a benchmark snapshot for new state machines once they are marked complete.
the benchmark gate exists to prevent performance regressions and to force explicit, repeatable
performance characterization for each machine.

## summary

- every **new** `src/emel/**/sm.hpp` must include a benchmark readiness marker.
- markers live in the `sm.hpp` file header comment:
  - `// benchmark: scaffold` means no benchmark required yet.
  - `// benchmark: ready` means a benchmark **is required** and will be enforced by gates.
- benchmarks are only required for machines marked `ready`.
- benchmarks are enforced via a snapshot diff with a **10% regression tolerance**.
- reference comparison is pinned to a specific upstream commit in
  `tools/bench/reference_ref.txt` for reproducibility.

## benchmark harness

we use a single benchmark harness that lives in `tools/bench/bench_main.cpp` and
links EMEL with the reference allocator implementation for apples-to-apples
comparisons.
- output format is deterministic and diff-friendly:
  - `machine_path ns_per_op=123.4 iter=100000 runs=5`
  - `machine_path` matches the `src/emel/<path>/sm.hpp` path without the prefix and suffix.
- optional warmup controls (shared by EMEL + reference benches):
  - `EMEL_BENCH_WARMUP_ITERS` (default: min(`EMEL_BENCH_ITERS`, 1000))
  - `EMEL_BENCH_WARMUP_RUNS` (default: 1)
- toolchain defaults to zig (`zig cc` / `zig c++`) for determinism. use
  `scripts/bench.sh --system` if you explicitly want system compilers.
- reference commit can be overridden via `BENCH_REF_OVERRIDE` if needed.

## external reference compare (optional)

we also maintain an optional comparison mode that runs EMEL and reference benchmarks
in the same tool build.

to add a new comparison case:
1. add the EMEL case in `tools/bench/bench_main.cpp`.
2. add the matching reference case in `tools/bench/bench_main.cpp`.
3. keep the case names identical for proper matching.

the compare mode expects exact case-name matches and prints `ratio` values for quick inspection.
run it via `scripts/bench.sh --compare`.
when updating the comparison snapshot used by documentation, run:
`scripts/bench.sh --compare-update`.

## gate behavior

the benchmark gate script enforces the following:
- if a new `sm.hpp` file is added:
  - it must declare `// benchmark: scaffold` or `// benchmark: ready`.
  - if `ready`, a benchmark case **must** exist and be registered.
- snapshot regression failure if `ns_per_op` exceeds baseline by > 5%.
- snapshot baseline can be updated intentionally with `--update`.

## developer workflow

1. add a new machine `sm.hpp` with a benchmark marker.
2. if the machine is complete, set `// benchmark: ready` and add a benchmark case.
3. run `scripts/bench.sh --snapshot` locally to verify, or
   `scripts/bench.sh --snapshot --update` to update the baseline.

## rationale

scaffolded machines often produce trivial workloads that distort baselines. the readiness marker
allows incremental scaffolding without blocking CI, while still forcing benchmarks once behavior
is complete.

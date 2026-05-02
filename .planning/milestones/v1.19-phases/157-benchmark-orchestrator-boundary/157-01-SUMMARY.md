---
phase: 157
plan: 01
status: complete
requirements-completed:
  - ORCH-01
  - LANE-01
key_files:
  added:
    - tools/bench/bench_runner.hpp
    - tools/bench/bench_runner.cpp
  modified:
    - tools/bench/bench_main.cpp
    - tools/bench/CMakeLists.txt
    - tools/bench/bench_runner_tests.cpp
completed: 2026-05-01
---

# Summary

Phase 157 moved shared benchmark CLI/config/report ownership behind a runner boundary.

## Changes

- Added `emel::bench::run_bench_cli(...)` in `bench_runner.hpp` / `bench_runner.cpp`.
- Kept `bench_main.cpp` as a minimal process shim.
- Preserved the existing benchmark execution body, mode parsing, environment config, output
  schemas, and lane construction logic.
- Updated CMake to compile the new runner source and header.
- Added focused source tests proving `bench_main.cpp` delegates and no longer owns runner logic.

## Verification

Commands passed:

```sh
git diff --check -- tools/bench/bench_main.cpp tools/bench/bench_runner.cpp tools/bench/bench_runner.hpp tools/bench/CMakeLists.txt tools/bench/bench_runner_tests.cpp .planning/phases/157-benchmark-orchestrator-boundary
cmake -S tools/bench -B build/bench_tools_ninja -G Ninja -DCMAKE_BUILD_TYPE=Release -DEMEL_BENCH_SUITE_FILTER=
cmake --build build/bench_tools_ninja --target bench_runner_tests -j2
ctest --test-dir build/bench_tools_ninja --output-on-failure -R bench_runner_tests
EMEL_QUALITY_GATES_CHANGED_FILES="tools/bench/bench_main.cpp tools/bench/bench_runner.cpp tools/bench/bench_runner.hpp tools/bench/CMakeLists.txt tools/bench/bench_runner_tests.cpp .planning/phases/157-benchmark-orchestrator-boundary/157-CONTEXT.md .planning/phases/157-benchmark-orchestrator-boundary/157-01-PLAN.md" EMEL_QUALITY_GATES_BENCH_SUITE=generation scripts/quality_gates.sh
```

Code review status: clean.

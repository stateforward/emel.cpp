---
phase: 159
plan: 01
status: complete
requirements-completed:
  - DISC-01
key_files:
  added:
    - tools/bench/bench_runner_registry.hpp
    - tools/bench/bench_runner_registry.cpp
  modified:
    - tools/bench/bench_runner.cpp
    - tools/bench/CMakeLists.txt
    - tools/bench/bench_runner_tests.cpp
completed: 2026-05-01
---

# Summary

Phase 159 moved broad benchmark runner registration out of the orchestrator and into a localized
registry surface.

## Changes

- Added `bench_runner_registry.hpp` / `.cpp` as the deterministic suite metadata surface.
- Moved default runner and kernel runner case lists out of `bench_runner.cpp`.
- Registered tokenizer as a normal suite entry while preserving the existing
  `include_tokenizer` filter.
- Updated `run_bench_cli(...)` to consume `default_runner_cases()` and `kernel_runner_cases()`.
- Added source and registry tests proving registration ownership moved out of the orchestrator.

## Verification

Commands passed:

```sh
git diff --check -- tools/bench/bench_runner_registry.hpp tools/bench/bench_runner_registry.cpp tools/bench/bench_runner.cpp tools/bench/CMakeLists.txt tools/bench/bench_runner_tests.cpp .planning/phases/159-benchmark-runner-discovery-and-registration/159-CONTEXT.md .planning/phases/159-benchmark-runner-discovery-and-registration/159-01-PLAN.md
cmake -S tools/bench -B build/bench_tools_ninja -G Ninja -DCMAKE_BUILD_TYPE=Release -DEMEL_BENCH_SUITE_FILTER=
cmake --build build/bench_tools_ninja --target bench_runner_tests -j2
ctest --test-dir build/bench_tools_ninja --output-on-failure -R bench_runner_tests
EMEL_QUALITY_GATES_CHANGED_FILES="tools/bench/bench_runner_registry.hpp tools/bench/bench_runner_registry.cpp tools/bench/bench_runner.cpp tools/bench/CMakeLists.txt tools/bench/bench_runner_tests.cpp .planning/phases/159-benchmark-runner-discovery-and-registration/159-CONTEXT.md .planning/phases/159-benchmark-runner-discovery-and-registration/159-01-PLAN.md" EMEL_QUALITY_GATES_BENCH_SUITE=generation scripts/quality_gates.sh
```

Code review status: clean.

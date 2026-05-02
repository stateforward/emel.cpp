---
phase: 158
plan: 01
status: complete
requirements-completed:
  - RUNNER-01
  - RUNNER-02
key_files:
  added:
    - tools/bench/bench_runner_contract.hpp
  modified:
    - tools/bench/bench_runner.cpp
    - tools/bench/CMakeLists.txt
    - tools/bench/bench_runner_tests.cpp
completed: 2026-05-01
---

# Summary

Phase 158 added the benchmark runner request/result contract and a deterministic serialized
process seam.

## Changes

- Added `runner_mode`, `runner_request`, and `runner_result` contract types.
- Added newline-delimited `bench_runner_request/v1` and `bench_runner_result/v1` serialization and
  parse helpers.
- Wired `run_bench_cli(...)` to create a normalized `runner_request` from existing CLI/env inputs.
- Preserved existing EMEL/reference/compare and kernel execution branches.
- Added focused tests for request/result round-trips and malformed payload rejection.

## Verification

Commands passed:

```sh
git diff --check -- tools/bench/bench_runner_contract.hpp tools/bench/bench_runner.cpp tools/bench/CMakeLists.txt tools/bench/bench_runner_tests.cpp .planning/phases/158-benchmark-runner-contract-and-process-seam
cmake -S tools/bench -B build/bench_tools_ninja -G Ninja -DCMAKE_BUILD_TYPE=Release -DEMEL_BENCH_SUITE_FILTER=
cmake --build build/bench_tools_ninja --target bench_runner_tests -j2
ctest --test-dir build/bench_tools_ninja --output-on-failure -R bench_runner_tests
EMEL_QUALITY_GATES_CHANGED_FILES="tools/bench/bench_runner_contract.hpp tools/bench/bench_runner.cpp tools/bench/CMakeLists.txt tools/bench/bench_runner_tests.cpp .planning/phases/158-benchmark-runner-contract-and-process-seam/158-CONTEXT.md .planning/phases/158-benchmark-runner-contract-and-process-seam/158-01-PLAN.md" EMEL_QUALITY_GATES_BENCH_SUITE=generation scripts/quality_gates.sh
```

Code review status: clean.

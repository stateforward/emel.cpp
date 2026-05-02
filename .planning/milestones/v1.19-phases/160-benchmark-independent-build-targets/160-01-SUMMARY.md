---
phase: 160
plan: 01
status: complete
requirements-completed:
  - BUILD-01
  - BUILD-02
key_files:
  modified:
    - tools/bench/CMakeLists.txt
    - tools/bench/bench_runner_tests.cpp
completed: 2026-05-01
---

# Summary

Phase 160 split benchmark suite source compilation into independent CMake object targets.

## Changes

- Added shared CMake helpers for benchmark runner include/link settings and artifact definitions.
- Updated `add_bench_runner_suite(...)` to create `bench_runner_suite_<suite>` object targets.
- Linked selected suite object files into the existing `bench_runner` executable.
- Preserved `bench_disabled_cases.cpp` compile definitions for filtered builds.
- Added focused source checks for object-target suite wiring.

## Verification

Commands passed:

```sh
git diff --check -- tools/bench/CMakeLists.txt tools/bench/bench_runner_tests.cpp .planning/phases/160-benchmark-independent-build-targets .planning/ROADMAP.md .planning/REQUIREMENTS.md .planning/STATE.md
cmake -S tools/bench -B build/bench_tools_ninja -G Ninja -DCMAKE_BUILD_TYPE=Release -DEMEL_BENCH_SUITE_FILTER=
cmake --build build/bench_tools_ninja --target bench_runner_tests -j2
ctest --test-dir build/bench_tools_ninja --output-on-failure -R bench_runner_tests
EMEL_QUALITY_GATES_CHANGED_FILES="tools/bench/CMakeLists.txt tools/bench/bench_runner_tests.cpp .planning/phases/160-benchmark-independent-build-targets/160-CONTEXT.md .planning/phases/160-benchmark-independent-build-targets/160-01-PLAN.md .planning/phases/160-benchmark-independent-build-targets/160-01-SUMMARY.md .planning/phases/160-benchmark-independent-build-targets/160-VERIFICATION.md .planning/phases/160-benchmark-independent-build-targets/160-REVIEW.md .planning/ROADMAP.md .planning/REQUIREMENTS.md .planning/STATE.md" EMEL_QUALITY_GATES_BENCH_SUITE=generation scripts/quality_gates.sh
```

Code review status: clean.

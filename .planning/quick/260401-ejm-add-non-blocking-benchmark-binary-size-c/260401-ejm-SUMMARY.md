---
quick_task: 260401-ejm
title: Add non-blocking benchmark binary size comparison between emel and llama for reference in quality gates/benchmarks
completed: 2026-04-01
commit: 3b9888e
plan: .planning/quick/260401-ejm-add-non-blocking-benchmark-binary-size-c/260401-ejm-PLAN.md
verification:
  - "cmake -S tools/bench -B build/bench_tools_ninja -G Ninja -DCMAKE_BUILD_TYPE=Release -DEMEL_ENABLE_TESTS=OFF && cmake --build build/bench_tools_ninja --parallel --target bench_runner_tests"
  - "ctest --test-dir build/bench_tools_ninja --output-on-failure -R bench_runner_tests"
  - "scripts/generate_docs.sh --check"
  - "scripts/quality_gates.sh"
---

# Quick Task 260401-ejm Summary

Added a reference-only binary size metadata line to the maintained benchmark compare output.

## Outcome

- `tools/bench/CMakeLists.txt` now injects the built `emel` and `llama` artifact paths into
  `bench_runner`.
- `tools/bench/bench_main.cpp` now prints one `# binary_size_compare:` metadata line during
  compare-mode runs. The line is non-blocking and falls back to `status=unavailable` instead of
  failing if an artifact size cannot be read.
- `tools/bench/bench_runner_tests.cpp` now verifies the metadata line exists and includes non-zero
  byte counts without depending on exact size values.

## Compatibility

- `scripts/bench.sh` and `scripts/quality_gates.sh` required no code changes because they already
  pass through `#`-prefixed compare metadata.
- `tools/docsgen/docsgen.cpp` required no change because it already ignores unknown `#` metadata
  lines while preserving existing compare-row parsing.

## Verification

- Bench runner build and test passed.
- `scripts/generate_docs.sh --check` passed.
- `scripts/quality_gates.sh` passed and preserved the existing warning-only benchmark behavior.
  The run still reported an ignored benchmark regression in `text/encoders/bpe_long`, which
  remained non-blocking.

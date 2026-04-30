---
phase: 131
status: passed
requirements:
  - TEXTGEN-03
  - TEXTGEN-04
  - TEXTGEN-05
---

# Phase 131 Verification

## Result

Passed.

## Evidence

- `cmake -S . -B build/zig-generator -G Ninja -DCMAKE_BUILD_TYPE=Release
  -DEMEL_ENABLE_TESTS=ON -DEMEL_TEST_SHARDS=generator_and_runtime
  -DCMAKE_C_COMPILER=/opt/homebrew/bin/zig -DCMAKE_C_COMPILER_ARG1=cc
  -DCMAKE_CXX_COMPILER=/opt/homebrew/bin/zig -DCMAKE_CXX_COMPILER_ARG1=c++`
  passed.
- `cmake --build build/zig-generator --target emel_tests_bin -j2` passed.
- `ctest --test-dir build/zig-generator -R emel_tests_generator_and_runtime
  --output-on-failure` passed.

---
phase: 149
plan: 01
status: complete
requirements-completed:
  - PARITY-02
  - ENGINE-01
key_files:
  created:
    - tools/paritychecker/parity_engine.hpp
    - tools/paritychecker/parity_engine.cpp
    - tools/paritychecker/parity_engines.hpp
    - tools/paritychecker/parity_engines.cpp
  modified:
    - tools/paritychecker/CMakeLists.txt
    - tools/paritychecker/parity_runner.cpp
    - tools/paritychecker/paritychecker_tests.cpp
completed: 2026-05-01
---

# Summary

Phase 149 split parity mode execution behind explicit runner-facing engine adapters.

## Changes

- Added `engine_adapter` metadata and `find_engine(...)` lookup in `parity_engine.hpp` / `.cpp`.
- Added named adapter entrypoints in `parity_engines.hpp` and moved the existing bulk implementation
  from `parity_runner.cpp` to `parity_engines.cpp`.
- Recreated `parity_runner.cpp` as orchestration only: it resolves the adapter for `opts.mode` and
  invokes the adapter run function.
- Updated paritychecker CMake targets so both the executable and tests link the adapter and moved
  engine implementation files.
- Added doctest coverage for every maintained adapter, invalid-mode lookup, and source evidence
  that bulk mode functions do not live in `parity_runner.cpp`.

## Verification

Commands passed:

```sh
cmake --build build/paritychecker_zig --target paritychecker_tests -j2
ctest --test-dir build/paritychecker_zig --output-on-failure
cmake --build build/paritychecker_zig --target paritychecker -j2
```

```sh
EMEL_QUALITY_GATES_CHANGED_FILES="tools/paritychecker/parity_engine.hpp tools/paritychecker/parity_engine.cpp tools/paritychecker/parity_engines.hpp tools/paritychecker/parity_engines.cpp tools/paritychecker/parity_runner.cpp tools/paritychecker/paritychecker_tests.cpp tools/paritychecker/CMakeLists.txt" scripts/quality_gates.sh
```

Code review status: clean.

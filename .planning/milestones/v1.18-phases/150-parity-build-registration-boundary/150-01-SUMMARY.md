---
phase: 150
plan: 01
status: complete
requirements-completed:
  - ENGINE-02
  - BUILD-01
  - BUILD-02
key_files:
  modified:
    - tools/paritychecker/CMakeLists.txt
    - tools/paritychecker/paritychecker_tests.cpp
completed: 2026-05-01
---

# Summary

Phase 150 made the paritychecker build and registration boundary auditable and shared.

## Changes

- Factored `tools/paritychecker/CMakeLists.txt` into explicit source groups for CLI, runner,
  engine registration, engine implementation, tokenizer engines, and reference support.
- Added `PARITYCHECKER_COMMON_SOURCES` and used it for both `paritychecker` and
  `paritychecker_tests`.
- Added source regressions proving the modular CMake groups exist, both targets consume the shared
  common source list, and invalid mode lookup remains fail-closed instead of falling back to
  tokenizer.

## Verification

Commands passed:

```sh
cmake --build build/paritychecker_zig --target paritychecker_tests -j2
ctest --test-dir build/paritychecker_zig --output-on-failure
cmake --build build/paritychecker_zig --target paritychecker -j2
```

```sh
EMEL_QUALITY_GATES_CHANGED_FILES="tools/paritychecker/CMakeLists.txt tools/paritychecker/parity_engine.cpp tools/paritychecker/paritychecker_tests.cpp" scripts/quality_gates.sh
```

Code review status: clean.

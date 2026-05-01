---
phase: 151
plan: 01
status: complete
requirements-completed:
  - MANIFEST-01
  - MANIFEST-02
  - MANIFEST-03
key_files:
  added:
    - tools/paritychecker/parity_dependency_manifest.hpp
    - tools/paritychecker/parity_dependency_manifest.cpp
    - tools/paritychecker/dependency_manifest.md
  modified:
    - tools/paritychecker/CMakeLists.txt
    - tools/paritychecker/paritychecker_tests.cpp
completed: 2026-05-01
---

# Summary

Phase 151 added deterministic parity dependency manifest emission for the paritychecker.

## Changes

- Added `emel::paritychecker::dependency_manifest` with typed records, per-mode lookup,
  deterministic line-oriented rendering, file writing, and conservative freshness semantics.
- Added source-backed records for tokenizer, GBNF, kernel, Jinja, and generation parity modes.
- Documented `parity_dependency_manifest/v1` in `tools/paritychecker/dependency_manifest.md`.
- Added manifest source to the shared paritychecker CMake source groups.
- Added doctests for registered-engine coverage, source/config/fixture/model/script/snapshot
  coverage, full-gate fallback on missing/stale/uncertain data, deterministic rendering, and file
  writing.

## Verification

Commands passed:

```sh
git diff --check
cmake --build build/paritychecker_zig --target paritychecker_tests -j2
ctest --test-dir build/paritychecker_zig --output-on-failure
cmake --build build/paritychecker_zig --target paritychecker -j2
```

```sh
EMEL_QUALITY_GATES_CHANGED_FILES="tools/paritychecker/CMakeLists.txt tools/paritychecker/parity_dependency_manifest.hpp tools/paritychecker/parity_dependency_manifest.cpp tools/paritychecker/dependency_manifest.md tools/paritychecker/paritychecker_tests.cpp" scripts/quality_gates.sh
```

The quality gate rewrote `snapshots/quality_gates/timing.txt` with scoped-run timings; that
unapproved snapshot churn was restored before commit.

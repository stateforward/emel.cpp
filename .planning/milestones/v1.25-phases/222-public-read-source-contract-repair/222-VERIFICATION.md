---
phase: 222-public-read-source-contract-repair
status: passed
verified: 2026-05-06T04:46:52Z
requirements:
  - PLAT-01
  - TIO-03
  - VAL-02
  - VAL-04
---

# Phase 222 Verification

## Requirement Status

| Requirement | Status | Source-Backed Evidence |
|-------------|--------|------------------------|
| PLAT-01 | Passed | Setup-time source-byte loading is exposed through `src/emel/io/source/any.hpp`, outside the `io/read` actor; `io/read` still consumes event-provided source spans. |
| TIO-03 | Passed | Maintained generation, Sortformer, embedded probe, and paritychecker lanes no longer include `emel/io/read/detail.hpp` and continue to report `read_copy` from public model-loader evidence. |
| VAL-02 | Passed | Guardrails in `tests/model/loader/lifecycle_tests.cpp` and `tools/paritychecker/paritychecker_tests.cpp` reject actor-internal read detail reach-through. |
| VAL-04 | Passed | Focused paritychecker and generation compare evidence passes after the public source contract repair. |

## Verification Commands

- `cmake --build build/zig --target emel_tests_bin` passed.
- `cmake --build build/paritychecker_zig --target paritychecker_tests` passed.
- `ctest --test-dir build/zig --output-on-failure -R emel_tests_model_and_batch`
  passed.
- `ctest --test-dir build/paritychecker_zig --output-on-failure -R paritychecker_tests`
  passed.
- `PATH=/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin ctest --test-dir build/bench_tools_ninja_generation --output-on-failure -R generation_compare_tests`
  passed after reconfiguring `build/bench_tools_ninja` with
  `-DGIT_EXECUTABLE=/usr/bin/git`.
- `scripts/check_domain_boundaries.sh` passed.
- `EMEL_QUALITY_GATES_CHANGED_FILES=... scripts/quality_gates.sh` exited 0.

## Notes

The first generation compare attempt reproduced the audit failure because the
reference build cache pointed CMake at the atmux Git shim. Reconfiguring the
maintained reference build cache with `/usr/bin/git` allowed the same test to
pass.

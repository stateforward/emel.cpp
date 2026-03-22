---
phase: 15-runtime-adoption-and-proof
plan: 01
subsystem: runtime-flash-observability
tags: [generator, kernel, aarch64, observability, runtime]
requires:
  - phase: 14-aarch64-flash-kernel
    provides: backend-local optimized/shared AArch64 flash accounting
provides:
  - shipped generator accessors for optimized vs shared flash dispatch counts
  - backend-wrapper forwarding through `kernel::any` without transition-table changes
  - failing-then-passing generator runtime proof for canonical ARM flash selection
affects: [15-02 negative proof, 15-03 parity publication]
tech-stack:
  added: []
  patterns: [wrapper-level backend observability, runtime-seam proof]
key-files:
  created: []
  modified:
    [src/emel/kernel/aarch64/sm.hpp, src/emel/kernel/any.hpp, src/emel/generator/sm.hpp, tests/generator/lifecycle_tests.cpp]
key-decisions:
  - "Expose optimized/shared counts through wrappers instead of adding new events or transition rows."
  - "Keep non-AArch64 backends explicit by reporting zero optimized/shared flash counts."
patterns-established:
  - "Backend-local proof can surface through `sm_any::visit(...)` without changing orchestration."
  - "Generator runtime tests should assert backend-selection truth at the shipped seam, not only in kernel-local tests."
requirements-completed: []
duration: 0min
completed: 2026-03-22
---

# Phase 15 Plan 1 Summary

**The shipped generator seam now reports optimized-vs-shared flash path selection**

## Accomplishments

- Added a failing-first generator runtime proof in
  [lifecycle_tests.cpp](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/tests/generator/lifecycle_tests.cpp)
  that expected new optimized/shared flash counters to exist at the shipped generator seam.
- Extended
  [sm.hpp](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/kernel/aarch64/sm.hpp)
  with wrapper-level accessors for the backend-local optimized/shared flash counters already owned
  by AArch64 context.
- Forwarded those counters through
  [any.hpp](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/kernel/any.hpp)
  and exposed generator-facing accessors in
  [sm.hpp](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/generator/sm.hpp)
  so runtime proof consumers can distinguish optimized AArch64 execution from shared fallback.

## Task Commits

None - execution is staying local and `commit_docs` is `false`.

## Verification

- `cmake --build build/zig --target emel_tests_bin --parallel 8`
- `./build/zig/emel_tests_bin --test-case='*generator*flash*' --no-breaks`

## Deviations from Plan

- The runtime seam did not need generator detail changes; `dispatch_flash_attention(...)` already
  hit the correct backend path, so the work stayed at wrapper and proof level.

## Next Readiness

- Plan 2 can tighten the negative runtime case so unsupported requests cannot silently claim either
  optimized or shared flash execution.
- Plan 3 can publish the same backend-selection truth on paritychecker's maintained CLI surface.

---
*Phase: 15-runtime-adoption-and-proof*
*Completed: 2026-03-22*

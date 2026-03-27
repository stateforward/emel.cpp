---
phase: 17-vectorized-q2-k-kernel
plan: 02
subsystem: q2-kernel-seam-proof
tags: [kernel, aarch64, q2_k, proof, lifecycle]
requires:
  - phase: 17-01
    provides: vectorized q2 row helper wired into the AArch64 quantized seam
provides:
  - backend-local q2 path attribution for supported optimized vs shared fallback execution
  - maintained kernel and lifecycle proof for supported q2 vectorized execution
  - explicit record that the widened longer-decode parity gate remains deferred
affects: [18-vectorized-q3-kernel]
tech-stack:
  added: []
  patterns: [backend-local attribution counters, kernel-seam proof]
key-files:
  created: []
  modified:
    [src/emel/kernel/aarch64/context.hpp, src/emel/kernel/aarch64/actions.hpp, src/emel/kernel/aarch64/sm.hpp, tests/kernel/aarch64_tests.cpp, tests/kernel/lifecycle_tests.cpp]
key-decisions:
  - "Keep q2 path attribution local to the AArch64 backend seam and do not widen `kernel::any` or generator wrappers in Phase 17."
  - "Record the repo-wide longer-decode parity failure as deferred verification debt rather than stretching Phase 17 into a parity-repair phase."
patterns-established:
  - "Backend-local path-selection proof can live in context plus wrapper accessors without touching transition tables."
  - "Autonomous execution can advance after a user-approved defer decision when verification debt is outside the current phase boundary."
requirements-completed: [PORT-04]
duration: 0min
completed: 2026-03-22
---

# Phase 17 Plan 2 Summary

**The maintained kernel seam can now distinguish optimized q2 execution from shared fallback**

## Accomplishments

- Added backend-local q2 path attribution counters in
  [context.hpp](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/kernel/aarch64/context.hpp)
  and surfaced them through
  [sm.hpp](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/kernel/aarch64/sm.hpp)
  without changing any transition rows.
- Updated
  [actions.hpp](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/kernel/aarch64/actions.hpp)
  so optimized AArch64 q2 requests increment `optimized_q2_dispatch_count`, while the shared
  scalar fallback path increments `shared_q2_dispatch_count`.
- Added kernel and lifecycle assertions in
  [aarch64_tests.cpp](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/tests/kernel/aarch64_tests.cpp)
  and
  [lifecycle_tests.cpp](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/tests/kernel/lifecycle_tests.cpp)
  proving supported q2 requests hit the vectorized seam on AArch64 and do not make false shared
  fallback claims.

## Task Commits

None - execution is staying local and `commit_docs` is `false`.

## Verification

- `cmake --build build/zig --target emel_tests_bin -j4`
- `./build/zig/emel_tests_bin --test-case='*kernel_aarch64_sm_reports_q2_vectorized_dispatch_at_kernel_seam*' --no-breaks`
- `./build/zig/emel_tests_bin --test-case='*kernel_aarch64_backend_reports_q2_vectorized_or_shared_dispatch*' --no-breaks`
- `./build/zig/emel_tests_bin --test-case='*kernel_aarch64_quantized_mul_mat_simd_matches_scalar*' --no-breaks`
- `scripts/quality_gates.sh` (fails only in widened paritychecker generation coverage at `max_tokens=100` and `1000`; deferred by user)

## Deviations from Plan

- The required repo-wide quality gate did not pass because
  [paritychecker_tests.cpp](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/tools/paritychecker/paritychecker_tests.cpp)
  now enforces `1/10/100/1000` generation parity and the existing longer-decode drift still fails
  at `100` and `1000`. The user explicitly chose not to prioritize that parity repair right now,
  so the phase records it as deferred debt instead of expanding Phase 17 scope.

## Next Readiness

- Phase 18 can follow the same seam pattern for q3 while leaving the longer-decode parity issue
  isolated as a milestone-level debt item.
- Phase 20 remains the right place to revisit runtime/parity publication once the full q2/q3/q6
  kernel set is in place.

---
*Phase: 17-vectorized-q2-k-kernel*
*Completed: 2026-03-22*

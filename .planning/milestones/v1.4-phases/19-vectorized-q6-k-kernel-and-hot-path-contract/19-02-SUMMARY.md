---
phase: 19-vectorized-q6-k-kernel-and-hot-path-contract
plan: 02
subsystem: q6-kernel-seam-and-hot-path-proof
tags: [kernel, aarch64, q6_k, proof, allocation]
requires:
  - phase: 19-01
    provides: vectorized q6 row helper wired into the AArch64 quantized seam
provides:
  - backend-local q6 path attribution for supported optimized vs shared fallback execution
  - alloc-free proof for supported q2/q3/q6 quantized dispatch on the maintained backend seam
  - explicit record that the widened longer-decode parity gate remains deferred
affects: [20-runtime-integration-and-proof]
tech-stack:
  added: [shared test allocation tracker]
  patterns: [backend-local attribution counters, alloc-free proof, kernel-seam proof]
key-files:
  created:
    [tests/allocation_tracker.hpp]
  modified:
    [src/emel/kernel/aarch64/context.hpp, src/emel/kernel/aarch64/actions.hpp, src/emel/kernel/aarch64/sm.hpp, tests/graph/graph_tests.cpp, tests/kernel/aarch64_tests.cpp, tests/kernel/lifecycle_tests.cpp]
key-decisions:
  - "Keep q6 path attribution local to the AArch64 backend seam and do not widen `kernel::any`, generator, parity, or benchmark wrappers in Phase 19."
  - "Reuse the existing test-binary allocation hook through a shared test helper instead of adding a new production observability surface."
  - "Record the repo-wide longer-decode parity failure as deferred verification debt rather than stretching Phase 19 into parity repair."
patterns-established:
  - "The maintained q2/q3/q6 quantized path can be proven alloc-free at the backend seam with test-only allocation tracking."
  - "Backend-local optimized/shared counters are sufficient to show supported requests no longer depend on shared scalar row helpers."
requirements-completed: [PORT-06, PORT-07]
duration: 0min
completed: 2026-03-22
---

# Phase 19 Plan 2 Summary

**The maintained kernel seam now proves q6 vectorized execution and alloc-free q2/q3/q6 dispatch**

## Accomplishments

- Added backend-local q6 path attribution counters in
  [context.hpp](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/kernel/aarch64/context.hpp)
  and surfaced them through
  [sm.hpp](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/kernel/aarch64/sm.hpp)
  without changing any transition rows.
- Updated
  [actions.hpp](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/kernel/aarch64/actions.hpp)
  so optimized AArch64 q6 requests increment `optimized_q6_dispatch_count`, while the shared
  scalar fallback path increments `shared_q6_dispatch_count`.
- Added
  [allocation_tracker.hpp](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/tests/allocation_tracker.hpp)
  and reused it from
  [graph_tests.cpp](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/tests/graph/graph_tests.cpp)
  plus
  [aarch64_tests.cpp](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/tests/kernel/aarch64_tests.cpp)
  so maintained kernel tests can prove q2/q3/q6 quantized dispatch stays allocation-free.
- Added kernel and lifecycle assertions in
  [aarch64_tests.cpp](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/tests/kernel/aarch64_tests.cpp)
  and
  [lifecycle_tests.cpp](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/tests/kernel/lifecycle_tests.cpp)
  proving supported q6 requests hit the vectorized seam on AArch64 and that supported q2/q3/q6
  dispatch stays out of shared fallback while remaining alloc-free.

## Task Commits

None - execution is staying local and `commit_docs` is `false`.

## Verification

- `cmake --build build/zig --target emel_tests_bin -j4`
- `./build/zig/emel_tests_bin --test-case='*kernel_aarch64_q6_row_neon_matches_scalar*' --no-breaks`
- `./build/zig/emel_tests_bin --test-case='*kernel_aarch64_sm_reports_q6_vectorized_dispatch_at_kernel_seam*' --no-breaks`
- `./build/zig/emel_tests_bin --test-case='*kernel_aarch64_backend_reports_q6_vectorized_or_shared_dispatch*' --no-breaks`
- `./build/zig/emel_tests_bin --test-case='*kernel_aarch64_supported_quantized_dispatch_is_alloc_free*' --no-breaks`
- `./build/zig/emel_tests_bin --test-case='*kernel_aarch64_quantized_mul_mat_simd_matches_scalar*' --no-breaks`
- `./build/zig/emel_tests_bin --test-case='*graph_machine_compute_lifecycle_dispatch_is_alloc_free*' --no-breaks`
- `scripts/quality_gates.sh` (fails only in widened paritychecker generation coverage at `max_tokens=100` and `1000`; deferred by user)

## Deviations from Plan

- The required repo-wide quality gate did not pass because
  [paritychecker_tests.cpp](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/tools/paritychecker/paritychecker_tests.cpp)
  continues to enforce `1/10/100/1000` generation parity and the known longer-decode drift still
  fails at `100` and `1000`. The user explicitly chose not to prioritize that parity repair right
  now, so the phase records it as deferred debt instead of expanding Phase 19 scope.

## Next Readiness

- Phase 20 can now focus on adopting the full q2/q3/q6 vectorized kernel set in the shipped
  runtime chain without still carrying a supported q6 scalar-row dependency.
- Phase 21 remains the place to publish maintained benchmark attribution once runtime-chain proof
  and parity publication are in scope.

---
*Phase: 19-vectorized-q6-k-kernel-and-hot-path-contract*
*Completed: 2026-03-22*

---
phase: 19-vectorized-q6-k-kernel-and-hot-path-contract
plan: 01
subsystem: aarch64-q6-kernel-cutover
tags: [kernel, aarch64, q6_k, quantized, vectorized]
requires:
  - phase: 18-vectorized-q3-k-kernel
    provides: maintained q2/q3 seam pattern for dtype-local vectorized cutovers
provides:
  - backend-local `q6_K x q8_K` row helper built from EMEL-owned AArch64 block arithmetic
  - AArch64 quantized mul-mat routing that uses the vectorized q6 seam instead of the shared scalar q6 row helper
  - focused q6 kernel equivalence coverage at the maintained backend seam
affects: [19-02 hot-path contract proof]
tech-stack:
  added: []
  patterns: [backend-local row helper, scalar-oracle equivalence tests]
key-files:
  created: []
  modified:
    [src/emel/kernel/aarch64/actions.hpp, tests/kernel/aarch64_tests.cpp]
key-decisions:
  - "Keep the q6 cutover inside `execute_neon_mul_mat(...)` and leave q2/q3 routing unchanged."
  - "Build the q6 row path from the existing `dot_q6_k_q8_k_block_neon(...)` arithmetic instead of widening runtime seams."
patterns-established:
  - "The final maintained quantized dtype can land with the same backend-local row-helper swap used by q2 and q3."
  - "Row-level q6 proof remains the narrowest safe guardrail before cross-dtype contract proof."
requirements-completed: [PORT-06]
duration: 0min
completed: 2026-03-22
---

# Phase 19 Plan 1 Summary

**The AArch64 q6 hot path now uses an EMEL-owned vectorized row helper**

## Accomplishments

- Added
  [actions.hpp](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/kernel/aarch64/actions.hpp)
  `dot_q6_k_q8_k_row_neon(...)`, which accumulates staged `q8_K` blocks through the existing
  backend-local `dot_q6_k_q8_k_block_neon(...)` arithmetic.
- Rewired the `dtype_q6_k` branch in
  [actions.hpp](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/kernel/aarch64/actions.hpp)
  `execute_neon_mul_mat(...)` to call the new vectorized q6 row helper instead of
  `kernel/detail.hpp::dot_q6_k_q8_k_row_scalar(...)`.
- Added focused q6 seam coverage in
  [aarch64_tests.cpp](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/tests/kernel/aarch64_tests.cpp)
  proving the new q6 row helper matches the shared scalar row oracle on the maintained operand
  layout.

## Task Commits

None - execution is staying local and `commit_docs` is `false`.

## Verification

- `cmake --build build/zig --target emel_tests_bin -j4`
- `./build/zig/emel_tests_bin --test-case='*kernel_aarch64_q6_row_neon_matches_scalar*' --no-breaks`
- `./build/zig/emel_tests_bin --test-case='*kernel_aarch64_quantized_mul_mat_simd_matches_scalar*' --no-breaks`

## Deviations from Plan

- None in scope. The q6 seam followed the same narrow backend-local cutover pattern as q2 and q3,
  so the implementation stayed inside the maintained AArch64 quantized branch without any SML,
  event, or public API changes.

## Next Readiness

- Plan 2 can add truthful backend-local q6 path attribution plus the alloc-free cross-dtype proof
  without widening runtime, parity, or benchmark surfaces.
- Phase 20 can consume the now-complete q2/q3/q6 kernel set for runtime-chain proof once Phase 19
  contract verification is recorded.

---
*Phase: 19-vectorized-q6-k-kernel-and-hot-path-contract*
*Completed: 2026-03-22*

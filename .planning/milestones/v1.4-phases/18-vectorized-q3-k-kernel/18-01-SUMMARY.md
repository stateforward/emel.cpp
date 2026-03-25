---
phase: 18-vectorized-q3-k-kernel
plan: 01
subsystem: aarch64-q3-kernel-cutover
tags: [kernel, aarch64, q3_k, quantized, vectorized]
requires:
  - phase: 17-vectorized-q2-k-kernel
    provides: maintained q2 seam pattern for dtype-local vectorized cutovers
provides:
  - backend-local `q3_K x q8_K` row helper built from EMEL-owned AArch64 block arithmetic
  - AArch64 quantized mul-mat routing that uses the vectorized q3 seam instead of the shared scalar q3 row helper
  - focused q3 kernel equivalence coverage at the maintained backend seam
affects: [18-02 kernel-seam proof]
tech-stack:
  added: []
  patterns: [backend-local row helper, scalar-oracle equivalence tests]
key-files:
  created: []
  modified:
    [src/emel/kernel/aarch64/actions.hpp, tests/kernel/aarch64_tests.cpp]
key-decisions:
  - "Keep the q3 cutover inside `execute_neon_mul_mat(...)` and leave q2/q6 routing unchanged."
  - "Build the q3 row path from the existing `dot_q3_k_q8_k_block_neon(...)` arithmetic instead of widening runtime seams."
patterns-established:
  - "Each dtype cutover can land as a backend-local row-helper swap with scalar-oracle proof before attribution work."
  - "Failing-first kernel-seam proof can compile-fail on a missing helper before the arithmetic implementation lands."
requirements-completed: [PORT-05]
duration: 0min
completed: 2026-03-22
---

# Phase 18 Plan 1 Summary

**The AArch64 q3 hot path now uses an EMEL-owned vectorized row helper**

## Accomplishments

- Added
  [actions.hpp](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/kernel/aarch64/actions.hpp)
  `dot_q3_k_q8_k_row_neon(...)`, which accumulates staged `q8_K` blocks through the existing
  backend-local `dot_q3_k_q8_k_block_neon(...)` arithmetic.
- Rewired the `dtype_q3_k` branch in
  [actions.hpp](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/kernel/aarch64/actions.hpp)
  `execute_neon_mul_mat(...)` to call the new vectorized q3 row helper instead of
  `kernel/detail.hpp::dot_q3_k_q8_k_row_scalar(...)`.
- Added focused q3 seam coverage in
  [aarch64_tests.cpp](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/tests/kernel/aarch64_tests.cpp)
  proving the new q3 row helper matches the shared scalar row oracle on the maintained operand
  layout.

## Task Commits

None - execution is staying local and `commit_docs` is `false`.

## Verification

- `cmake --build build/zig --target emel_tests_bin -j4`
- `./build/zig/emel_tests_bin --test-case='*kernel_aarch64_q3_row_neon_matches_scalar*' --no-breaks`
- `./build/zig/emel_tests_bin --test-case='*kernel_aarch64_quantized_mul_mat_simd_matches_scalar*' --no-breaks`

## Deviations from Plan

- None in scope. The q3 seam followed the same narrow backend-local cutover pattern as q2, so the
  implementation stayed inside the maintained AArch64 quantized branch without any SML, event, or
  public API changes.

## Next Readiness

- Plan 2 can add truthful backend-local q3 path attribution without widening runtime, parity, or
  benchmark surfaces.
- Phase 19 can reuse the same cutover-plus-proof structure for q6 once Phase 18 proof is recorded.

---
*Phase: 18-vectorized-q3-k-kernel*
*Completed: 2026-03-22*

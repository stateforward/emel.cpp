---
phase: 17-vectorized-q2-k-kernel
plan: 01
subsystem: aarch64-q2-kernel-cutover
tags: [kernel, aarch64, q2_k, quantized, vectorized]
requires:
  - phase: 16-benchmark-attribution-and-evidence
    provides: maintained ARM flash runtime and attribution baseline
provides:
  - backend-local `q2_K x q8_K` row helper built from EMEL-owned AArch64 block arithmetic
  - AArch64 quantized mul-mat routing that uses the vectorized q2 seam instead of the shared scalar q2 row helper
  - focused q2 kernel equivalence coverage at the maintained backend seam
affects: [17-02 kernel-seam proof]
tech-stack:
  added: []
  patterns: [backend-local row helper, scalar-oracle equivalence tests]
key-files:
  created: []
  modified:
    [src/emel/kernel/aarch64/actions.hpp, tests/kernel/aarch64_tests.cpp]
key-decisions:
  - "Keep the cutover inside `execute_neon_mul_mat(...)` and leave q3/q6 behavior unchanged."
  - "Use the existing q2 NEON block helper as the only arithmetic building block for the new row path."
patterns-established:
  - "Quantized phase work can swap one dtype-specific hot-path helper without changing SML structure."
  - "Kernel-seam proof should compare the new row helper directly against the shared scalar row oracle before broader runtime proof begins."
requirements-completed: [PORT-04]
duration: 0min
completed: 2026-03-22
---

# Phase 17 Plan 1 Summary

**The AArch64 q2 hot path now uses an EMEL-owned vectorized row helper**

## Accomplishments

- Added
  [actions.hpp](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/kernel/aarch64/actions.hpp)
  `dot_q2_k_q8_k_row_neon(...)`, which accumulates staged `q8_K` blocks through the existing
  backend-local `dot_q2_k_q8_k_block_neon(...)` arithmetic.
- Rewired the `dtype_q2_k` branch in
  [actions.hpp](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/kernel/aarch64/actions.hpp)
  `execute_neon_mul_mat(...)` to call the new vectorized q2 row helper instead of
  `kernel/detail.hpp::dot_q2_k_q8_k_row_scalar(...)`.
- Added focused q2 seam coverage in
  [aarch64_tests.cpp](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/tests/kernel/aarch64_tests.cpp)
  proving the new q2 row helper still matches the shared scalar row oracle.

## Task Commits

None - execution is staying local and `commit_docs` is `false`.

## Verification

- `cmake --build build/zig --target emel_tests_bin -j4`
- `./build/zig/emel_tests_bin --test-case='*kernel_aarch64_q2_row_neon_matches_scalar*' --no-breaks`
- `./build/zig/emel_tests_bin --test-case='*kernel_aarch64_quantized_mul_mat_simd_matches_scalar*' --no-breaks`

## Deviations from Plan

- None in scope. The seam behaved exactly as expected: the only code change required for the q2
  cutover was backend-local row-helper wiring inside the existing quantized AArch64 branch.

## Next Readiness

- Plan 2 can add truthful backend-local q2 path attribution without widening runtime or benchmark
  surfaces.
- Phase 18 can reuse the same seam pattern for the q3 cutover once Phase 17 proof is recorded.

---
*Phase: 17-vectorized-q2-k-kernel*
*Completed: 2026-03-22*
